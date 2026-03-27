#!/usr/bin/env python3
"""
Liquidity Levels Indicator

Detects key liquidity pools and market structure:
1. Session High/Low - Asian, London, NY session ranges (liquidity targets)
2. Equal Highs/Lows (EQH/EQL) - Double/triple tops/bottoms (stop hunts)
3. Order Blocks - Last opposing candle before impulse move (institutional entry)

These levels represent areas where stop losses cluster and are frequently
targeted by smart money before reversals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple


class LiquidityLevels:
    """
    Calculates liquidity levels for confluence trading:
    - Session highs/lows (Asian -> London target, London -> NY target)
    - Equal highs/lows (EQH/EQL) - liquidity pools
    - Order blocks - institutional entry zones
    """

    # Session times in GMT/UTC
    SESSIONS = {
        'asian': {'start': 0, 'end': 8},      # 00:00 - 08:00 GMT
        'london': {'start': 8, 'end': 16},    # 08:00 - 16:00 GMT
        'new_york': {'start': 13, 'end': 21}, # 13:00 - 21:00 GMT (overlaps London)
    }

    def __init__(self, tolerance_pct: float = 0.0003):
        """
        Initialize liquidity levels calculator

        Args:
            tolerance_pct: Price tolerance for level detection (0.03% = 3 pips default)
        """
        self.tolerance_pct = tolerance_pct

    def calculate_session_levels(
        self,
        h1_data: pd.DataFrame,
        broker_gmt_offset: int = 0
    ) -> Dict:
        """
        Calculate session high/low levels

        Args:
            h1_data: H1 OHLC data with 'time' column
            broker_gmt_offset: Broker's GMT offset for time conversion

        Returns:
            Dict with session levels:
            {
                'asian_high': float,
                'asian_low': float,
                'london_high': float,
                'london_low': float,
                'prev_asian_high': float,
                'prev_asian_low': float,
                'prev_london_high': float,
                'prev_london_low': float,
            }
        """
        if len(h1_data) < 48:  # Need at least 2 days of H1 data
            return {}

        # Convert broker time to GMT
        df = h1_data.copy()
        if 'time' not in df.columns:
            return {}

        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        # Convert to GMT
        df['gmt_time'] = df['time'] - pd.Timedelta(hours=broker_gmt_offset)
        df['gmt_hour'] = df['gmt_time'].dt.hour
        df['gmt_date'] = df['gmt_time'].dt.date

        # Get today and yesterday
        today = df['gmt_date'].iloc[-1]
        yesterday = today - timedelta(days=1)

        levels = {}

        # Calculate session levels for each session
        for session_name, session_times in self.SESSIONS.items():
            start_hour = session_times['start']
            end_hour = session_times['end']

            # Today's session
            if start_hour < end_hour:
                today_session = df[
                    (df['gmt_date'] == today) &
                    (df['gmt_hour'] >= start_hour) &
                    (df['gmt_hour'] < end_hour)
                ]
            else:
                # Session crosses midnight (not currently used but future-proof)
                today_session = df[
                    (df['gmt_date'] == today) &
                    ((df['gmt_hour'] >= start_hour) | (df['gmt_hour'] < end_hour))
                ]

            if len(today_session) > 0:
                levels[f'{session_name}_high'] = float(today_session['high'].max())
                levels[f'{session_name}_low'] = float(today_session['low'].min())
                levels[f'{session_name}_mid'] = (levels[f'{session_name}_high'] + levels[f'{session_name}_low']) / 2

            # Yesterday's session (for "previous session" levels)
            if start_hour < end_hour:
                prev_session = df[
                    (df['gmt_date'] == yesterday) &
                    (df['gmt_hour'] >= start_hour) &
                    (df['gmt_hour'] < end_hour)
                ]
            else:
                prev_session = df[
                    (df['gmt_date'] == yesterday) &
                    ((df['gmt_hour'] >= start_hour) | (df['gmt_hour'] < end_hour))
                ]

            if len(prev_session) > 0:
                levels[f'prev_{session_name}_high'] = float(prev_session['high'].max())
                levels[f'prev_{session_name}_low'] = float(prev_session['low'].min())

        return levels

    def detect_equal_levels(
        self,
        h1_data: pd.DataFrame,
        lookback: int = 50,
        tolerance_pips: float = 3.0
    ) -> Dict:
        """
        Detect Equal Highs (EQH) and Equal Lows (EQL) - liquidity pools

        EQH/EQL are double/triple tops or bottoms that represent clustered
        stop losses. Smart money often sweeps these before reversing.

        Args:
            h1_data: H1 OHLC data
            lookback: Number of bars to look back
            tolerance_pips: Pip tolerance for "equal" (default 3 pips)

        Returns:
            Dict with equal levels:
            {
                'equal_highs': [(price, count, last_touch_idx), ...],
                'equal_lows': [(price, count, last_touch_idx), ...],
                'nearest_eqh': float or None,
                'nearest_eql': float or None,
            }
        """
        if len(h1_data) < lookback:
            return {'equal_highs': [], 'equal_lows': [], 'nearest_eqh': None, 'nearest_eql': None}

        df = h1_data.tail(lookback).copy()
        df = df.reset_index(drop=True)

        # Convert tolerance to price (assume 5-digit broker)
        tolerance = tolerance_pips * 0.0001

        # Find swing highs and lows
        swing_highs = self._find_swing_points(df, 'high', window=5)
        swing_lows = self._find_swing_points(df, 'low', window=5)

        # Group equal highs
        equal_highs = self._group_equal_levels(swing_highs, tolerance)

        # Group equal lows
        equal_lows = self._group_equal_levels(swing_lows, tolerance)

        # Find nearest EQH/EQL to current price
        current_price = float(df['close'].iloc[-1])
        nearest_eqh = None
        nearest_eql = None

        # EQH should be above current price (resistance)
        eqh_above = [eq for eq in equal_highs if eq[0] > current_price and eq[1] >= 2]
        if eqh_above:
            nearest_eqh = min(eqh_above, key=lambda x: x[0] - current_price)[0]

        # EQL should be below current price (support)
        eql_below = [eq for eq in equal_lows if eq[0] < current_price and eq[1] >= 2]
        if eql_below:
            nearest_eql = max(eql_below, key=lambda x: x[0])[0]

        return {
            'equal_highs': equal_highs,
            'equal_lows': equal_lows,
            'nearest_eqh': nearest_eqh,
            'nearest_eql': nearest_eql,
            'eqh_count': len([eq for eq in equal_highs if eq[1] >= 2]),
            'eql_count': len([eq for eq in equal_lows if eq[1] >= 2]),
        }

    def _find_swing_points(
        self,
        df: pd.DataFrame,
        column: str,
        window: int = 5
    ) -> List[Tuple[float, int]]:
        """Find swing highs or lows"""
        points = []

        for i in range(window, len(df) - window):
            if column == 'high':
                # Swing high: higher than surrounding bars
                is_swing = all(
                    df[column].iloc[i] >= df[column].iloc[i - j] and
                    df[column].iloc[i] >= df[column].iloc[i + j]
                    for j in range(1, window + 1)
                )
            else:
                # Swing low: lower than surrounding bars
                is_swing = all(
                    df[column].iloc[i] <= df[column].iloc[i - j] and
                    df[column].iloc[i] <= df[column].iloc[i + j]
                    for j in range(1, window + 1)
                )

            if is_swing:
                points.append((float(df[column].iloc[i]), i))

        return points

    def _group_equal_levels(
        self,
        points: List[Tuple[float, int]],
        tolerance: float
    ) -> List[Tuple[float, int, int]]:
        """Group swing points that are within tolerance into equal levels"""
        if not points:
            return []

        # Sort by price
        sorted_points = sorted(points, key=lambda x: x[0])

        groups = []
        current_group = [sorted_points[0]]

        for i in range(1, len(sorted_points)):
            price, idx = sorted_points[i]
            group_avg = sum(p[0] for p in current_group) / len(current_group)

            if abs(price - group_avg) <= tolerance:
                current_group.append(sorted_points[i])
            else:
                # Save current group if it has 2+ touches
                if len(current_group) >= 2:
                    avg_price = sum(p[0] for p in current_group) / len(current_group)
                    last_idx = max(p[1] for p in current_group)
                    groups.append((avg_price, len(current_group), last_idx))
                current_group = [sorted_points[i]]

        # Don't forget the last group
        if len(current_group) >= 2:
            avg_price = sum(p[0] for p in current_group) / len(current_group)
            last_idx = max(p[1] for p in current_group)
            groups.append((avg_price, len(current_group), last_idx))

        return groups

    def detect_order_blocks(
        self,
        h1_data: pd.DataFrame,
        lookback: int = 50,
        min_impulse_pips: float = 20.0
    ) -> Dict:
        """
        Detect Order Blocks - institutional entry zones

        Bullish OB: Last bearish candle before a strong bullish move
        Bearish OB: Last bullish candle before a strong bearish move

        Args:
            h1_data: H1 OHLC data
            lookback: Number of bars to look back
            min_impulse_pips: Minimum impulse move size in pips

        Returns:
            Dict with order blocks:
            {
                'bullish_obs': [(high, low, idx), ...],
                'bearish_obs': [(high, low, idx), ...],
                'nearest_bullish_ob': {'high': float, 'low': float} or None,
                'nearest_bearish_ob': {'high': float, 'low': float} or None,
            }
        """
        if len(h1_data) < lookback:
            return {
                'bullish_obs': [],
                'bearish_obs': [],
                'nearest_bullish_ob': None,
                'nearest_bearish_ob': None
            }

        df = h1_data.tail(lookback).copy()
        df = df.reset_index(drop=True)

        min_impulse = min_impulse_pips * 0.0001
        bullish_obs = []
        bearish_obs = []

        for i in range(2, len(df) - 3):
            # Check for bullish order block
            # Pattern: Bearish candle followed by strong bullish impulse
            candle_body = df['close'].iloc[i] - df['open'].iloc[i]

            if candle_body < 0:  # Bearish candle (potential bullish OB)
                # Check for bullish impulse after (3 candles)
                impulse_high = df['high'].iloc[i+1:i+4].max()
                impulse_start = df['close'].iloc[i]

                if (impulse_high - impulse_start) >= min_impulse:
                    bullish_obs.append({
                        'high': float(df['high'].iloc[i]),
                        'low': float(df['low'].iloc[i]),
                        'idx': i,
                        'strength': float(impulse_high - impulse_start)
                    })

            elif candle_body > 0:  # Bullish candle (potential bearish OB)
                # Check for bearish impulse after (3 candles)
                impulse_low = df['low'].iloc[i+1:i+4].min()
                impulse_start = df['close'].iloc[i]

                if (impulse_start - impulse_low) >= min_impulse:
                    bearish_obs.append({
                        'high': float(df['high'].iloc[i]),
                        'low': float(df['low'].iloc[i]),
                        'idx': i,
                        'strength': float(impulse_start - impulse_low)
                    })

        # Find nearest unmitigated order blocks to current price
        current_price = float(df['close'].iloc[-1])

        # Bullish OB should be below current price (support)
        valid_bullish = [ob for ob in bullish_obs if ob['high'] < current_price]
        nearest_bullish_ob = None
        if valid_bullish:
            nearest_bullish_ob = max(valid_bullish, key=lambda x: x['high'])

        # Bearish OB should be above current price (resistance)
        valid_bearish = [ob for ob in bearish_obs if ob['low'] > current_price]
        nearest_bearish_ob = None
        if valid_bearish:
            nearest_bearish_ob = min(valid_bearish, key=lambda x: x['low'])

        return {
            'bullish_obs': bullish_obs,
            'bearish_obs': bearish_obs,
            'nearest_bullish_ob': nearest_bullish_ob,
            'nearest_bearish_ob': nearest_bearish_ob,
            'bullish_ob_count': len(bullish_obs),
            'bearish_ob_count': len(bearish_obs),
        }

    def check_confluence(
        self,
        current_price: float,
        direction: str,
        session_levels: Dict,
        equal_levels: Dict,
        order_blocks: Dict,
        tolerance_pct: float = None
    ) -> Dict:
        """
        Check if price is at a liquidity confluence level

        Args:
            current_price: Current price
            direction: Trade direction ('buy' or 'sell')
            session_levels: From calculate_session_levels()
            equal_levels: From detect_equal_levels()
            order_blocks: From detect_order_blocks()
            tolerance_pct: Price tolerance for level detection

        Returns:
            Dict with confluence signals:
            {
                'at_session_low': bool,
                'at_session_high': bool,
                'at_prev_session_low': bool,
                'at_prev_session_high': bool,
                'near_eql': bool,  # Near equal lows (liquidity sweep potential)
                'near_eqh': bool,  # Near equal highs (liquidity sweep potential)
                'at_bullish_ob': bool,
                'at_bearish_ob': bool,
                'liquidity_score': int,
                'factors': list,
            }
        """
        if tolerance_pct is None:
            tolerance_pct = self.tolerance_pct

        result = {
            'at_session_low': False,
            'at_session_high': False,
            'at_prev_session_low': False,
            'at_prev_session_high': False,
            'at_asian_low': False,
            'at_asian_high': False,
            'at_london_low': False,
            'at_london_high': False,
            'near_eql': False,
            'near_eqh': False,
            'at_bullish_ob': False,
            'at_bearish_ob': False,
            'liquidity_score': 0,
            'factors': [],
        }

        tolerance = current_price * tolerance_pct

        # Check session levels
        session_checks = [
            ('asian_low', 'at_asian_low'),
            ('asian_high', 'at_asian_high'),
            ('london_low', 'at_london_low'),
            ('london_high', 'at_london_high'),
            ('prev_asian_low', 'at_prev_session_low'),
            ('prev_asian_high', 'at_prev_session_high'),
            ('prev_london_low', 'at_prev_session_low'),
            ('prev_london_high', 'at_prev_session_high'),
        ]

        for level_key, result_key in session_checks:
            if level_key in session_levels:
                level = session_levels[level_key]
                if abs(current_price - level) <= tolerance:
                    result[result_key] = True
                    if 'low' in level_key:
                        result['at_session_low'] = True
                        result['factors'].append(f'Session Low ({level_key})')
                    else:
                        result['at_session_high'] = True
                        result['factors'].append(f'Session High ({level_key})')

        # Check equal levels
        if equal_levels.get('nearest_eql'):
            eql = equal_levels['nearest_eql']
            if abs(current_price - eql) <= tolerance:
                result['near_eql'] = True
                result['factors'].append('Equal Lows (EQL)')

        if equal_levels.get('nearest_eqh'):
            eqh = equal_levels['nearest_eqh']
            if abs(current_price - eqh) <= tolerance:
                result['near_eqh'] = True
                result['factors'].append('Equal Highs (EQH)')

        # Check order blocks
        if order_blocks.get('nearest_bullish_ob'):
            ob = order_blocks['nearest_bullish_ob']
            if ob['low'] <= current_price <= ob['high']:
                result['at_bullish_ob'] = True
                result['factors'].append('Bullish Order Block')

        if order_blocks.get('nearest_bearish_ob'):
            ob = order_blocks['nearest_bearish_ob']
            if ob['low'] <= current_price <= ob['high']:
                result['at_bearish_ob'] = True
                result['factors'].append('Bearish Order Block')

        # Calculate liquidity score based on direction
        # For BUY: Want to be at session lows, EQL, or bullish OB (support)
        # For SELL: Want to be at session highs, EQH, or bearish OB (resistance)
        if direction.lower() == 'buy':
            if result['at_session_low']:
                result['liquidity_score'] += 2
            if result['near_eql']:
                result['liquidity_score'] += 2
            if result['at_bullish_ob']:
                result['liquidity_score'] += 3
            if result['at_prev_session_low']:
                result['liquidity_score'] += 1
        else:  # sell
            if result['at_session_high']:
                result['liquidity_score'] += 2
            if result['near_eqh']:
                result['liquidity_score'] += 2
            if result['at_bearish_ob']:
                result['liquidity_score'] += 3
            if result['at_prev_session_high']:
                result['liquidity_score'] += 1

        return result

    def get_all_levels(
        self,
        h1_data: pd.DataFrame,
        broker_gmt_offset: int = 0
    ) -> Dict:
        """
        Calculate all liquidity levels at once

        Args:
            h1_data: H1 OHLC data
            broker_gmt_offset: Broker's GMT offset

        Returns:
            Dict with all liquidity data
        """
        session_levels = self.calculate_session_levels(h1_data, broker_gmt_offset)
        equal_levels = self.detect_equal_levels(h1_data)
        order_blocks = self.detect_order_blocks(h1_data)

        return {
            'session_levels': session_levels,
            'equal_levels': equal_levels,
            'order_blocks': order_blocks,
        }


def main():
    """Test the liquidity levels indicator"""
    print("Liquidity Levels Indicator Test")
    print("=" * 50)

    # Create sample data
    import numpy as np

    np.random.seed(42)
    n_bars = 100

    # Generate sample OHLC data
    base_price = 1.1000
    prices = [base_price]
    for _ in range(n_bars - 1):
        change = np.random.normal(0, 0.0005)
        prices.append(prices[-1] + change)

    df = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=n_bars, freq='H'),
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.0003)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.0003)) for p in prices],
        'close': [p + np.random.normal(0, 0.0002) for p in prices],
        'volume': [np.random.randint(100, 1000) for _ in range(n_bars)],
    })

    # Test the indicator
    ll = LiquidityLevels()

    print("\n1. Session Levels:")
    session_levels = ll.calculate_session_levels(df, broker_gmt_offset=0)
    for key, value in session_levels.items():
        print(f"   {key}: {value:.5f}")

    print("\n2. Equal Levels:")
    equal_levels = ll.detect_equal_levels(df)
    print(f"   EQH count: {equal_levels['eqh_count']}")
    print(f"   EQL count: {equal_levels['eql_count']}")
    if equal_levels['nearest_eqh']:
        print(f"   Nearest EQH: {equal_levels['nearest_eqh']:.5f}")
    if equal_levels['nearest_eql']:
        print(f"   Nearest EQL: {equal_levels['nearest_eql']:.5f}")

    print("\n3. Order Blocks:")
    order_blocks = ll.detect_order_blocks(df)
    print(f"   Bullish OB count: {order_blocks['bullish_ob_count']}")
    print(f"   Bearish OB count: {order_blocks['bearish_ob_count']}")
    if order_blocks['nearest_bullish_ob']:
        ob = order_blocks['nearest_bullish_ob']
        print(f"   Nearest Bullish OB: {ob['low']:.5f} - {ob['high']:.5f}")
    if order_blocks['nearest_bearish_ob']:
        ob = order_blocks['nearest_bearish_ob']
        print(f"   Nearest Bearish OB: {ob['low']:.5f} - {ob['high']:.5f}")

    print("\n4. Confluence Check (BUY at current price):")
    current_price = df['close'].iloc[-1]
    confluence = ll.check_confluence(
        current_price=current_price,
        direction='buy',
        session_levels=session_levels,
        equal_levels=equal_levels,
        order_blocks=order_blocks
    )
    print(f"   Current price: {current_price:.5f}")
    print(f"   Liquidity score: {confluence['liquidity_score']}")
    print(f"   Factors: {confluence['factors']}")


if __name__ == '__main__':
    main()
