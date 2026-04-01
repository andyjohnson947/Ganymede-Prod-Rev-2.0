#!/usr/bin/env python3
"""
Continuous ML Trade Logger
Runs in background, logs every trade's confluence factors automatically
Integrates with trading bot to capture real-time data
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add paths
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'trading_bot'))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from indicators.vwap import VWAP
from indicators.volume_profile import VolumeProfile
from indicators.htf_levels import HTFLevels
from indicators.adx import calculate_adx, analyze_candle_momentum
from utils.data_utils import convert_numpy_types
from config.strategy_config import (
    CONFLUENCE_WEIGHTS,
    LEVEL_TOLERANCE_PCT,
    ADX_PERIOD,
    TREND_FILTER_ENABLED,
    CANDLE_MOMENTUM_ENABLED,
    CANDLE_MOMENTUM_LOOKBACK,
    CANDLE_MOMENTUM_MIN_SHRINK_RATIO,
)

# Auto-tuner integration - import lazily to avoid circular imports
_auto_tuner_module = None

def _get_auto_tuner():
    """Lazy import of auto_tuner module"""
    global _auto_tuner_module
    if _auto_tuner_module is None:
        try:
            from ml_system import auto_tuner as at
            _auto_tuner_module = at
        except ImportError as e:
            print(f"[WARN] Could not import auto_tuner: {e}")
            _auto_tuner_module = False  # Mark as unavailable
    return _auto_tuner_module if _auto_tuner_module else None


class ContinuousMLLogger:
    """
    Background service that logs every trade's confluence factors
    Runs continuously, monitors for new trades, captures data automatically
    """

    def __init__(self, output_dir: str = None, use_existing_connection: bool = False, api_lock=None):
        # Use absolute path based on project root to avoid duplication
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "ml_system" / "outputs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Continuous log file (append mode)
        self.continuous_log = self.output_dir / "continuous_trade_log.jsonl"

        # SQLite mirror (write-only, JSONL remains source of truth)
        try:
            from ml_system.trade_db import TradeDatabase
            self.trade_db = TradeDatabase(self.output_dir / "trades.db")
        except Exception as e:
            print(f"[WARN] SQLite mirror not available: {e}")
            self.trade_db = None

        # Track which trades we've already logged
        self.logged_tickets = self._load_logged_tickets()

        # Initialize bot's indicator modules
        self.vwap = VWAP()
        self.volume_profile = VolumeProfile()
        self.htf_levels = HTFLevels()

        # Threading fix: Allow reusing existing MT5 connection from main thread
        self.use_existing_connection = use_existing_connection
        self.mt5 = None

        # Thread-safe MT5 API access lock
        self.api_lock = api_lock

    def _load_logged_tickets(self) -> set:
        """Load set of already-logged ticket IDs"""
        # Try SQLite first
        if hasattr(self, 'trade_db') and self.trade_db:
            try:
                return self.trade_db.get_all_tickets()
            except Exception as e:
                print(f"[WARN] SQLite ticket load failed, falling back to JSONL: {e}")

        # Fallback: parse JSONL
        logged = set()
        if self.continuous_log.exists():
            with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        logged.add(record['ticket'])
                    except:
                        pass
        return logged

    def connect_mt5(self, login: int, password: str, server: str) -> bool:
        """
        Connect to MT5

        Threading fix: If use_existing_connection=True, reuses the MT5 connection
        already initialized in the main thread instead of creating a new one.
        This prevents MT5 API thread-safety violations.
        """
        if not MT5_AVAILABLE:
            return False

        if self.use_existing_connection:
            # Reuse existing MT5 connection from main thread
            # IMPORTANT: Don't make ANY MT5 API calls here - just reuse the module
            # The connection was already established by MT5Manager in the main thread
            self.mt5 = mt5
            return True
        else:
            # Standalone mode: Create new connection (for backwards compatibility)
            if mt5.initialize() and mt5.login(login, password, server):
                self.mt5 = mt5
                return True
            return False

    def _with_lock(self, func):
        """
        Execute MT5 API call with lock for thread-safe access.
        Uses the same lock as MT5Manager to prevent conflicts.
        """
        if self.api_lock:
            with self.api_lock:
                return func()
        return func()

    def fetch_bars_at_time(
        self,
        symbol: str,
        timeframe: int,
        target_time: datetime,
        bars: int = 100
    ) -> Optional[pd.DataFrame]:
        """Fetch historical bars at specific time (thread-safe)"""
        if not self.mt5:
            return None

        # Thread-safe MT5 API call
        rates = self._with_lock(lambda: self.mt5.copy_rates_from(symbol, timeframe, target_time, bars))
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def detect_fair_value_gaps(
        self,
        data: pd.DataFrame,
        entry_price: float,
        tolerance_pct: float = 0.003
    ) -> Dict:
        """
        Detect Fair Value Gaps (FVGs) in OHLC data.

        FVG = Price imbalance where candle 1's high < candle 3's low (bullish)
                                 or candle 1's low > candle 3's high (bearish)

        Args:
            data: DataFrame with OHLC data
            entry_price: Current entry price
            tolerance_pct: Price tolerance (0.3% = 30 pips)

        Returns:
            Dictionary with FVG detection results
        """
        if len(data) < 10:
            return {
                'near_bullish_fvg': False,
                'near_bearish_fvg': False,
                'bullish_fvg_count': 0,
                'bearish_fvg_count': 0,
                'closest_bullish_fvg': None,
                'closest_bearish_fvg': None
            }

        bullish_fvgs = []
        bearish_fvgs = []

        # Look back through recent candles (last 20 for daily, last 10 for weekly)
        lookback = min(20, len(data) - 3)

        for i in range(len(data) - 3, len(data) - 3 - lookback, -1):
            if i < 0:
                break

            candle1 = data.iloc[i]
            candle2 = data.iloc[i + 1]
            candle3 = data.iloc[i + 2]

            # Bullish FVG: Candle 1 high < Candle 3 low (gap up, unfilled zone)
            if candle1['high'] < candle3['low']:
                gap_low = candle1['high']
                gap_high = candle3['low']
                bullish_fvgs.append({
                    'low': float(gap_low),
                    'high': float(gap_high),
                    'midpoint': float((gap_low + gap_high) / 2),
                    'size_pct': float((gap_high - gap_low) / gap_low * 100),
                    'age': len(data) - i - 2  # How many candles ago
                })

            # Bearish FVG: Candle 1 low > Candle 3 high (gap down, unfilled zone)
            elif candle1['low'] > candle3['high']:
                gap_high = candle1['low']
                gap_low = candle3['high']
                bearish_fvgs.append({
                    'low': float(gap_low),
                    'high': float(gap_high),
                    'midpoint': float((gap_low + gap_high) / 2),
                    'size_pct': float((gap_high - gap_low) / gap_high * 100),
                    'age': len(data) - i - 2
                })

        # Find closest FVGs to entry price
        closest_bullish = None
        closest_bearish = None
        min_bull_dist = float('inf')
        min_bear_dist = float('inf')

        for fvg in bullish_fvgs:
            dist = abs(fvg['midpoint'] - entry_price) / entry_price
            if dist < min_bull_dist:
                min_bull_dist = dist
                closest_bullish = fvg

        for fvg in bearish_fvgs:
            dist = abs(fvg['midpoint'] - entry_price) / entry_price
            if dist < min_bear_dist:
                min_bear_dist = dist
                closest_bearish = fvg

        # Check if price is near an FVG (within tolerance)
        near_bullish_fvg = False
        near_bearish_fvg = False

        if closest_bullish:
            # Price is near bullish FVG if within tolerance of the gap zone
            if (closest_bullish['low'] * (1 - tolerance_pct) <= entry_price <=
                closest_bullish['high'] * (1 + tolerance_pct)):
                near_bullish_fvg = True

        if closest_bearish:
            # Price is near bearish FVG if within tolerance of the gap zone
            if (closest_bearish['low'] * (1 - tolerance_pct) <= entry_price <=
                closest_bearish['high'] * (1 + tolerance_pct)):
                near_bearish_fvg = True

        return {
            'near_bullish_fvg': near_bullish_fvg,
            'near_bearish_fvg': near_bearish_fvg,
            'bullish_fvg_count': len(bullish_fvgs),
            'bearish_fvg_count': len(bearish_fvgs),
            'closest_bullish_fvg': closest_bullish,
            'closest_bearish_fvg': closest_bearish
        }

    def calculate_confluence_factors(
        self,
        h1_data: pd.DataFrame,
        d1_data: pd.DataFrame,
        w1_data: pd.DataFrame,
        entry_price: float
    ) -> Dict:
        """Calculate confluence factors using bot's modules"""
        if len(h1_data) < 200 or len(d1_data) < 2 or len(w1_data) < 2:
            return {}

        factors = {}

        # VWAP
        h1_with_vwap = self.vwap.calculate(h1_data.copy())
        vwap_signals = self.vwap.get_signals(h1_with_vwap)
        factors['vwap'] = {
            'value': float(vwap_signals.get('vwap', 0)),
            'distance_pct': float(vwap_signals.get('distance_pct', 0)),
            'direction': vwap_signals.get('direction', ''),
            'in_band_1': vwap_signals.get('in_band_1', False),
            'in_band_2': vwap_signals.get('in_band_2', False),
            'band_1_score': CONFLUENCE_WEIGHTS.get('vwap_band_1', 1) if vwap_signals.get('in_band_1') else 0,
            'band_2_score': CONFLUENCE_WEIGHTS.get('vwap_band_2', 1) if vwap_signals.get('in_band_2') else 0,
        }

        # Volume Profile
        vp_signals = self.volume_profile.get_signals(h1_with_vwap, entry_price, lookback=200)
        factors['volume_profile'] = {
            'at_poc': vp_signals.get('at_poc', False),
            'above_vah': vp_signals.get('above_vah', False),
            'below_val': vp_signals.get('below_val', False),
            'at_lvn': vp_signals.get('at_lvn', False),
            'at_swing_high': vp_signals.get('at_swing_high', False),
            'at_swing_low': vp_signals.get('at_swing_low', False),
        }

        # HTF Levels
        htf_all_levels = self.htf_levels.get_all_levels(d1_data, w1_data)
        htf_confluence = self.htf_levels.check_confluence(entry_price, htf_all_levels, LEVEL_TOLERANCE_PCT)
        daily_levels = self.htf_levels.calculate_daily_levels(d1_data)
        weekly_levels = self.htf_levels.calculate_weekly_levels(w1_data)

        factors['htf_levels'] = {
            'total_score': htf_confluence.get('score', 0),
            'factors_matched': htf_confluence.get('factors', []),
            'prev_day_vah': float(daily_levels.get('prev_day_vah', 0)),
            'prev_day_poc': float(daily_levels.get('prev_day_poc', 0)),
            'weekly_hvn_count': len(weekly_levels.get('weekly_hvn', [])),
            'weekly_poc': float(weekly_levels.get('weekly_poc', 0)),
        }

        # Fair Value Gaps (Daily and Weekly)
        daily_fvgs = self.detect_fair_value_gaps(d1_data, entry_price, LEVEL_TOLERANCE_PCT)
        weekly_fvgs = self.detect_fair_value_gaps(w1_data, entry_price, LEVEL_TOLERANCE_PCT)

        factors['fair_value_gaps'] = {
            'daily_bullish_fvg': daily_fvgs['near_bullish_fvg'],
            'daily_bearish_fvg': daily_fvgs['near_bearish_fvg'],
            'weekly_bullish_fvg': weekly_fvgs['near_bullish_fvg'],
            'weekly_bearish_fvg': weekly_fvgs['near_bearish_fvg'],
            'daily_bullish_count': daily_fvgs['bullish_fvg_count'],
            'daily_bearish_count': daily_fvgs['bearish_fvg_count'],
            'weekly_bullish_count': weekly_fvgs['bullish_fvg_count'],
            'weekly_bearish_count': weekly_fvgs['bearish_fvg_count'],
            'closest_daily_bullish': daily_fvgs['closest_bullish_fvg'],
            'closest_daily_bearish': daily_fvgs['closest_bearish_fvg'],
            'closest_weekly_bullish': weekly_fvgs['closest_bullish_fvg'],
            'closest_weekly_bearish': weekly_fvgs['closest_bearish_fvg'],
        }

        # Trend Filter
        if TREND_FILTER_ENABLED:
            h1_with_adx = calculate_adx(h1_with_vwap.copy(), period=ADX_PERIOD)
            latest_adx = h1_with_adx.iloc[-1]
            factors['trend_filter'] = {
                'enabled': True,
                'adx': float(latest_adx.get('adx', 0)),
                'plus_di': float(latest_adx.get('plus_di', 0)),
                'minus_di': float(latest_adx.get('minus_di', 0)),
            }
        else:
            factors['trend_filter'] = {'enabled': False}

        # Candle Momentum (retrospective calculation for ML)
        if CANDLE_MOMENTUM_ENABLED and h1_with_vwap is not None and len(h1_with_vwap) >= CANDLE_MOMENTUM_LOOKBACK + 2:
            try:
                # Calculate for both directions so ML has full picture
                buy_momentum = analyze_candle_momentum(
                    data=h1_with_vwap,
                    direction='buy',
                    lookback=CANDLE_MOMENTUM_LOOKBACK,
                    shrink_ratio=CANDLE_MOMENTUM_MIN_SHRINK_RATIO
                )
                sell_momentum = analyze_candle_momentum(
                    data=h1_with_vwap,
                    direction='sell',
                    lookback=CANDLE_MOMENTUM_LOOKBACK,
                    shrink_ratio=CANDLE_MOMENTUM_MIN_SHRINK_RATIO
                )
                factors['candle_momentum'] = {
                    'buy': buy_momentum,
                    'sell': sell_momentum,
                }
            except Exception as e:
                factors['candle_momentum'] = {'error': str(e)}
        else:
            factors['candle_momentum'] = {'enabled': False}

        # NEW: Volatility Features (Phase 1)
        factors['volatility'] = self._calculate_volatility_features(h1_data, entry_price)

        # NEW: Entry Quality Features (Phase 1)
        factors['entry_quality'] = self._calculate_entry_quality(h1_data, entry_price, vp_signals, htf_all_levels)

        return factors

    def _calculate_volatility_features(self, h1_data: pd.DataFrame, entry_price: float) -> Dict:
        """
        Calculate volatility context features (Phase 1 enhancement)
        - ATR (Average True Range)
        - ATR percentile
        - Recent range
        - Momentum
        """
        try:
            if len(h1_data) < 50:
                return {}

            # Calculate ATR (14-period)
            high_low = h1_data['high'] - h1_data['low']
            high_close = np.abs(h1_data['high'] - h1_data['close'].shift())
            low_close = np.abs(h1_data['low'] - h1_data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(window=14).mean().iloc[-1]

            # ATR percentile (where is current ATR vs last 50 bars?)
            atr_series = true_range.rolling(window=14).mean()
            atr_percentile = (atr_series.iloc[-1] > atr_series.tail(50)).sum() / 50.0

            # Recent range (10-bar high-low range)
            recent_high = h1_data['high'].tail(10).max()
            recent_low = h1_data['low'].tail(10).min()
            recent_range_pips = (recent_high - recent_low) * 10000

            # Momentum (20-bar rate of change)
            close_current = h1_data['close'].iloc[-1]
            close_20_ago = h1_data['close'].iloc[-20] if len(h1_data) >= 20 else h1_data['close'].iloc[0]
            momentum_pct = ((close_current - close_20_ago) / close_20_ago) * 100

            # Distance from recent high/low (entry quality)
            distance_from_high = ((entry_price - recent_high) / entry_price) * 100
            distance_from_low = ((entry_price - recent_low) / entry_price) * 100

            return {
                'atr_14': float(atr_14),
                'atr_percentile': float(atr_percentile),
                'recent_range_pips': float(recent_range_pips),
                'momentum_20_pct': float(momentum_pct),
                'distance_from_high_pct': float(distance_from_high),
                'distance_from_low_pct': float(distance_from_low),
            }

        except Exception as e:
            print(f"[WARN] Volatility calculation failed: {e}")
            return {}

    def _calculate_entry_quality(self, h1_data: pd.DataFrame, entry_price: float,
                                  vp_signals: Dict, htf_all_levels: Dict) -> Dict:
        """
        Calculate entry quality metrics (Phase 1 enhancement)
        - Distance to nearest swing level
        - HTF timeframe alignment
        - Signal conviction
        """
        try:
            if len(h1_data) < 20:
                return {}

            # Distance to nearest swing level
            swing_highs = vp_signals.get('swing_highs', [])
            swing_lows = vp_signals.get('swing_lows', [])

            min_distance_to_swing = 999.0
            if swing_highs or swing_lows:
                all_swings = list(swing_highs) + list(swing_lows)
                distances = [abs((entry_price - swing) / entry_price) * 100 for swing in all_swings]
                min_distance_to_swing = min(distances) if distances else 999.0

            # HTF alignment (how many HTF levels near entry?)
            htf_factors_count = len(htf_all_levels.get('factors', [])) if htf_all_levels else 0
            htf_aligned = htf_factors_count >= 3  # 3+ HTF levels = good alignment

            # Signal conviction (based on factor strength)
            conviction_score = 0
            if vp_signals.get('at_poc'):
                conviction_score += 2
            if vp_signals.get('at_swing_high') or vp_signals.get('at_swing_low'):
                conviction_score += 2
            if vp_signals.get('at_lvn'):
                conviction_score += 1
            if htf_factors_count >= 3:
                conviction_score += 2
            if htf_factors_count >= 5:
                conviction_score += 1

            # Normalized conviction (0-10)
            conviction_normalized = min(conviction_score, 10)

            return {
                'distance_to_swing_pct': float(min_distance_to_swing),
                'htf_aligned': bool(htf_aligned),
                'htf_factors_count': int(htf_factors_count),
                'signal_conviction': int(conviction_normalized),
            }

        except Exception as e:
            print(f"[WARN] Entry quality calculation failed: {e}")
            return {}

    def _calculate_trade_sequencing(self, entry_time: datetime) -> Dict:
        """
        Calculate trade sequencing features (Phase 1 enhancement)
        - Trades in last hour
        - Trades today
        - Win streak
        - Minutes since last trade
        - Open position count (if MT5 available)
        """
        try:
            # Load recent trades from log
            if not self.continuous_log.exists():
                return {}

            recent_trades = []
            with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        trade = json.loads(line)
                        # Handle multiple timestamp formats
                        entry_time_str = trade.get('entry_time')
                        if not entry_time_str and 'execution_quality' in trade:
                            entry_time_str = trade['execution_quality'].get('timestamp')
                        if not entry_time_str:
                            continue
                        trade_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                        recent_trades.append({
                            'time': trade_time,
                            'profit': trade.get('outcome', {}).get('profit', 0),
                            'closed': trade.get('outcome', {}).get('status') == 'closed'
                        })
                    except:
                        continue

            # Trades in last hour
            one_hour_ago = entry_time - timedelta(hours=1)
            trades_last_hour = sum(1 for t in recent_trades if t['time'] >= one_hour_ago and t['time'] < entry_time)

            # Trades today
            today_start = entry_time.replace(hour=0, minute=0, second=0, microsecond=0)
            trades_today = sum(1 for t in recent_trades if t['time'] >= today_start and t['time'] < entry_time)

            # Win streak (recent closed trades)
            closed_trades = [t for t in recent_trades if t['closed']]
            closed_trades.sort(key=lambda x: x['time'], reverse=True)

            win_streak = 0
            for trade in closed_trades[:10]:  # Last 10 closed trades
                if trade['profit'] > 0:
                    win_streak += 1
                else:
                    break  # Streak broken

            # Minutes since last trade
            if recent_trades:
                recent_trades_sorted = sorted(recent_trades, key=lambda x: x['time'], reverse=True)
                last_trade_time = recent_trades_sorted[0]['time']
                minutes_since_last = (entry_time - last_trade_time).total_seconds() / 60
            else:
                minutes_since_last = 999.0

            # Open position count (if MT5 available)
            open_count = 0
            if self.mt5:
                try:
                    positions = self._with_lock(lambda: self.mt5.positions_get())
                    open_count = len(positions) if positions else 0
                except:
                    open_count = 0

            return {
                'trades_last_hour': int(trades_last_hour),
                'trades_today': int(trades_today),
                'win_streak': int(win_streak),
                'minutes_since_last_trade': float(minutes_since_last),
                'open_position_count': int(open_count),
            }

        except Exception as e:
            print(f"[WARN] Trade sequencing calculation failed: {e}")
            return {}

    def _get_trading_session(self, hour: int) -> str:
        """
        Determine trading session based on hour (UTC)
        Tokyo: 0-8, London: 8-16, NY: 13-21, Sydney: 21-24
        """
        if 0 <= hour < 8:
            return 'Tokyo'
        elif 8 <= hour < 13:
            return 'London'
        elif 13 <= hour < 21:
            return 'NY'
        else:
            return 'Sydney'

    def _get_regime_state(self) -> Dict:
        """Load current regime state from regime_state.json for ML logging."""
        try:
            regime_file = Path(__file__).parent / 'outputs' / 'regime_state.json'
            if not regime_file.exists():
                return {}
            import json
            with open(regime_file, 'r') as f:
                state = json.load(f)
            return {
                'regime': state.get('regime'),
                'confidence': state.get('confidence'),
                'allowed_strategies': state.get('allowed_strategies'),
                'size_multiplier': state.get('size_multiplier'),
                'h4_bar_time': state.get('h4_bar_time'),
            }
        except Exception:
            return {}

    def _calculate_execution_quality(self, symbol: str, expected_price: float, actual_price: float,
                                        spread_at_entry_pips: float = 0.0, fill_time_ms: int = 0,
                                        requotes: int = 0) -> Dict:
        """
        Calculate execution quality metrics at trade entry time.

        Args:
            symbol: Trading symbol
            expected_price: Expected/intended entry price
            actual_price: Actual filled price
            spread_at_entry_pips: Spread at entry (in pips)
            fill_time_ms: Fill time in milliseconds
            requotes: Number of requotes

        Returns:
            Dict with execution quality metrics (slippage, spread, score, etc.)
        """
        try:
            # Calculate slippage in pips
            if expected_price and actual_price:
                point = 0.01 if 'JPY' in symbol else 0.0001
                slippage_pips = abs(actual_price - expected_price) / point
            else:
                slippage_pips = 0.0

            # Calculate execution quality score (0-100)
            score = 100

            # Penalize slippage (up to 30 points)
            if slippage_pips > 2:
                score -= min(30, slippage_pips * 5)

            # Penalize wide spreads (up to 20 points)
            if spread_at_entry_pips > 2:
                score -= min(20, (spread_at_entry_pips - 2) * 10)

            # Penalize slow fills (20 points if > 1 second)
            if fill_time_ms > 1000:
                score -= 20

            # Penalize requotes (10 points each)
            score -= requotes * 10

            score = max(0, score)

            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'expected_price': float(expected_price) if expected_price else None,
                'actual_price': float(actual_price) if actual_price else None,
                'slippage_pips': round(float(slippage_pips), 2),
                'spread_at_entry_pips': float(spread_at_entry_pips),
                'fill_time_ms': int(fill_time_ms),
                'requotes': int(requotes),
                'execution_quality_score': int(score)
            }

        except Exception as e:
            print(f"[WARN] Execution quality calculation failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'expected_price': None,
                'actual_price': None,
                'slippage_pips': 0.0,
                'spread_at_entry_pips': 0.0,
                'fill_time_ms': 0,
                'requotes': 0,
                'execution_quality_score': 100
            }

    def _calculate_market_microstructure(self, symbol: str, entry_price: float, filled_price: float) -> Dict:
        """
        Calculate market microstructure features (Phase 3 enhancement)
        - Spread (bid-ask)
        - Spread percentile
        - Slippage
        """
        try:
            # Get current symbol info for spread (thread-safe)
            if self.mt5:
                symbol_info = self._with_lock(lambda: self.mt5.symbol_info_tick(symbol))
                if symbol_info:
                    spread_pips = (symbol_info.ask - symbol_info.bid) * 10000

                    # Get recent ticks to calculate spread percentile (thread-safe)
                    recent_ticks = self._with_lock(lambda: self.mt5.copy_ticks_from(symbol, datetime.now(), 50, self.mt5.COPY_TICKS_ALL))
                    if recent_ticks is not None and len(recent_ticks) > 0:
                        recent_spreads = [(tick['ask'] - tick['bid']) * 10000 for tick in recent_ticks]
                        avg_spread = np.mean(recent_spreads)
                        spread_percentile = (spread_pips > np.array(recent_spreads)).mean()
                    else:
                        avg_spread = spread_pips
                        spread_percentile = 0.5
                else:
                    spread_pips = 0.0
                    avg_spread = 0.0
                    spread_percentile = 0.5
            else:
                spread_pips = 0.0
                avg_spread = 0.0
                spread_percentile = 0.5

            # Calculate slippage (difference between intended and filled price)
            slippage_pips = abs(filled_price - entry_price) * 10000

            return {
                'spread_pips': float(spread_pips),
                'spread_percentile': float(spread_percentile),
                'spread_vs_avg_pct': float((spread_pips / avg_spread - 1) * 100) if avg_spread > 0 else 0.0,
                'slippage_pips': float(slippage_pips),
            }

        except Exception as e:
            print(f"[WARN] Market microstructure calculation failed: {e}")
            return {
                'spread_pips': 0.0,
                'spread_percentile': 0.5,
                'spread_vs_avg_pct': 0.0,
                'slippage_pips': 0.0,
            }

    def _calculate_position_sizing_context(self, symbol: str, volume: float, entry_time: datetime) -> Dict:
        """
        Calculate position sizing context features (Phase 3 enhancement)
        - Risk percent
        - Position vs average
        - Account drawdown
        - Daily volume used
        """
        try:
            # Get account info (thread-safe)
            if self.mt5:
                account_info = self._with_lock(lambda: self.mt5.account_info())
                if account_info:
                    balance = account_info.balance
                    equity = account_info.equity
                    margin = account_info.margin

                    # Calculate drawdown
                    account_drawdown = ((balance - equity) / balance * 100) if balance > 0 else 0.0

                    # Calculate risk percent (margin used by this position) (thread-safe)
                    symbol_info = self._with_lock(lambda: self.mt5.symbol_info(symbol))
                    if symbol_info:
                        # Rough estimation: 1 lot = ~$100k, leverage typically 1:500
                        # So 0.01 lot ≈ $20 margin
                        position_margin = volume * 2000  # Approximate
                        risk_percent = (position_margin / balance * 100) if balance > 0 else 0.0
                    else:
                        risk_percent = 0.0
                else:
                    balance = 10000.0  # Default
                    account_drawdown = 0.0
                    risk_percent = 0.0
            else:
                balance = 10000.0
                account_drawdown = 0.0
                risk_percent = 0.0

            # Calculate average position size from recent trades
            today_start = entry_time.replace(hour=0, minute=0, second=0, microsecond=0)
            recent_volumes = []
            daily_volume = 0.0

            try:
                with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        trade = json.loads(line)
                        # Handle multiple timestamp formats
                        entry_time_str = trade.get('entry_time')
                        if not entry_time_str and 'execution_quality' in trade:
                            entry_time_str = trade['execution_quality'].get('timestamp')
                        if not entry_time_str:
                            continue
                        trade_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))

                        # Collect recent volumes (last 30 days)
                        if (entry_time - trade_time).days <= 30:
                            recent_volumes.append(trade.get('volume', 0.0))

                        # Collect today's volume
                        if trade_time >= today_start:
                            daily_volume += trade.get('volume', 0.0)
            except:
                pass

            # Calculate position vs average
            avg_volume = np.mean(recent_volumes) if recent_volumes else 0.01
            position_vs_avg = (volume / avg_volume) if avg_volume > 0 else 1.0

            return {
                'risk_percent': float(min(risk_percent, 100.0)),  # Cap at 100%
                'position_vs_avg': float(position_vs_avg),
                'account_drawdown': float(abs(account_drawdown)),
                'daily_volume_used': float(daily_volume),
                'margin_level_pct': float((equity / margin * 100) if margin > 0 else 999.0),
            }

        except Exception as e:
            print(f"[WARN] Position sizing context calculation failed: {e}")
            return {
                'risk_percent': 0.0,
                'position_vs_avg': 1.0,
                'account_drawdown': 0.0,
                'daily_volume_used': 0.0,
                'margin_level_pct': 999.0,
            }

    def update_execution_quality(self, ticket: int, expected_price: float, actual_price: float,
                                    spread_at_entry_pips: float = 0.0, fill_time_ms: int = 0,
                                    requotes: int = 0) -> bool:
        """
        Update execution quality for an existing trade record.

        Call this from trading bot immediately after trade execution when you have
        the expected price and actual fill price.

        Args:
            ticket: Trade ticket number
            expected_price: Expected/intended entry price
            actual_price: Actual filled price
            spread_at_entry_pips: Spread at entry in pips
            fill_time_ms: Fill time in milliseconds
            requotes: Number of requotes

        Returns:
            True if update successful, False otherwise
        """
        if not self.continuous_log.exists():
            return False

        try:
            # Read all trades
            all_trades = []
            with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        all_trades.append(json.loads(line))
                    except:
                        continue

            # Find and update the trade
            updated = False
            for trade in all_trades:
                if trade['ticket'] == ticket:
                    symbol = trade.get('symbol', '')
                    exec_quality = self._calculate_execution_quality(
                        symbol=symbol,
                        expected_price=expected_price,
                        actual_price=actual_price,
                        spread_at_entry_pips=spread_at_entry_pips,
                        fill_time_ms=fill_time_ms,
                        requotes=requotes
                    )
                    trade['execution_quality'] = exec_quality
                    updated = True
                    break

            if updated:
                # Write back
                with open(self.continuous_log, 'w', encoding='utf-8', errors='ignore') as f:
                    for trade in all_trades:
                        trade = convert_numpy_types(trade)
                        f.write(json.dumps(trade) + '\n')
                return True

            return False

        except Exception as e:
            print(f"[WARN] Failed to update execution quality for {ticket}: {e}")
            return False

    def find_recovery_actions(self, ticket: int, from_date: datetime, to_date: datetime) -> Dict:
        """Find all recovery actions (DCA, hedge, grid) for a specific ticket"""
        recovery = {
            'dca_count': 0,
            'hedge_count': 0,
            'grid_count': 0,
            'dca_levels': [],
            'hedge_ratios': [],
            'grid_levels': [],
            'total_recovery_volume': 0.0,
            'recovery_cost': 0.0  # Total loss from recovery trades
        }

        # Get all deals for this period (thread-safe)
        deals = self._with_lock(lambda: self.mt5.history_deals_get(from_date, to_date))
        if not deals:
            return recovery

        # Create multiple ticket search patterns (MT5 comments sometimes truncate ticket numbers)
        # Example: ticket 54573716123 -> comment shows "DCA L1 - 5457371"
        # NOTE: Hedge/Grid use LAST 5 digits (e.g., "Hedge - 16123")
        ticket_str = str(ticket)
        ticket_patterns = [
            ticket_str,           # Full ticket: "54573716123"
            ticket_str[:7],       # Truncated: "5457371"
            ticket_str[:8],       # Truncated: "54573716"
            ticket_str[:9],       # Truncated: "545737161"
            ticket_str[-5:],      # Last 5 digits: "16123" (for Hedge/Grid)
        ]

        for deal in deals:
            comment = deal.comment or ""

            # Check if this deal is related to our ticket
            # Match by position_id OR by ticket pattern in comment
            is_related = (deal.position_id == ticket)

            if not is_related:
                for pattern in ticket_patterns:
                    if pattern in comment:
                        is_related = True
                        break

            if is_related:
                if 'DCA' in comment:
                    recovery['dca_count'] += 1
                    recovery['dca_levels'].append({
                        'price': float(deal.price),
                        'volume': float(deal.volume),
                        'time': datetime.fromtimestamp(deal.time).isoformat()
                    })
                    recovery['total_recovery_volume'] += float(deal.volume)
                    if deal.profit < 0:
                        recovery['recovery_cost'] += abs(float(deal.profit))

                elif 'Hedge' in comment:
                    recovery['hedge_count'] += 1
                    recovery['hedge_ratios'].append({
                        'price': float(deal.price),
                        'volume': float(deal.volume),
                        'time': datetime.fromtimestamp(deal.time).isoformat()
                    })
                    recovery['total_recovery_volume'] += float(deal.volume)
                    if deal.profit < 0:
                        recovery['recovery_cost'] += abs(float(deal.profit))

                elif 'Grid' in comment:
                    # Grid orders have comment like "Grid L1 - 12345"
                    recovery['grid_count'] += 1
                    recovery['grid_levels'].append({
                        'price': float(deal.price),
                        'volume': float(deal.volume),
                        'profit': float(deal.profit),
                        'time': datetime.fromtimestamp(deal.time).isoformat()
                    })

        return recovery

    def find_partial_closes(self, ticket: int, from_date: datetime, to_date: datetime) -> Dict:
        """Find all partial close actions for a specific ticket"""
        partial_closes = {
            'count': 0,
            'closes': [],
            'total_profit_from_partials': 0.0
        }

        deals = self._with_lock(lambda: self.mt5.history_deals_get(from_date, to_date))
        if not deals:
            return partial_closes

        for deal in deals:
            comment = deal.comment or ""

            if deal.position_id == ticket and ('Partial' in comment or 'partial' in comment or 'PC1' in comment or 'PC2' in comment):
                partial_closes['count'] += 1
                partial_closes['closes'].append({
                    'price': float(deal.price),
                    'volume': float(deal.volume),
                    'profit': float(deal.profit),
                    'time': datetime.fromtimestamp(deal.time).isoformat(),
                    'comment': comment
                })
                partial_closes['total_profit_from_partials'] += float(deal.profit)

        return partial_closes

    def _calculate_exit_strategy_data(self, entry_price: float, exit_price: float,
                                       direction: str, partial_closes: Dict, exit_comment: str) -> Dict:
        """
        Calculate comprehensive exit strategy metrics for ML analysis.

        Args:
            entry_price: Entry price
            exit_price: Final exit price
            direction: 'BUY' or 'SELL'
            partial_closes: Partial close data
            exit_comment: MT5 exit comment

        Returns:
            Dict with exit strategy metrics
        """
        try:
            # Calculate total pips moved
            if direction == 'BUY':
                final_pips = (exit_price - entry_price) * 10000
            else:  # SELL
                final_pips = (entry_price - exit_price) * 10000

            # Detect exit method from comment and partial data
            exit_method = 'unknown'
            pc1_triggered = False
            pc2_triggered = False
            trailing_triggered = False
            vwap_exit = False
            peak_pips = final_pips  # Default to final if no partials

            # Check partial closes
            if partial_closes['count'] > 0:
                # Analyze partial close prices to infer PC1/PC2
                partial_prices = [p['price'] for p in partial_closes['closes']]

                # Calculate pips for each partial
                partial_pips = []
                for price in partial_prices:
                    if direction == 'BUY':
                        pips = (price - entry_price) * 10000
                    else:
                        pips = (entry_price - price) * 10000
                    partial_pips.append(pips)

                # Infer PC1/PC2 from partial pips
                for pips in partial_pips:
                    if 8 <= pips <= 15:  # PC1 range (10-12 pips)
                        pc1_triggered = True
                    elif 18 <= pips <= 28:  # PC2 range (20-25 pips)
                        pc2_triggered = True

                # Peak pips = highest partial pip value
                peak_pips = max(partial_pips + [final_pips])

            # Detect exit method from comment keywords
            comment_lower = exit_comment.lower()
            if 'trail' in comment_lower or 'trailing' in comment_lower:
                exit_method = 'trailing_stop'
                trailing_triggered = True
            elif 'vwap' in comment_lower:
                exit_method = 'vwap'
                vwap_exit = True
            elif 'pc1' in comment_lower or ('partial' in comment_lower and pc1_triggered and not pc2_triggered):
                exit_method = 'pc1_only'
            elif 'pc2' in comment_lower or ('partial' in comment_lower and pc2_triggered):
                exit_method = 'pc2_full'
            elif partial_closes['count'] == 0:
                # No partials - likely VWAP or manual
                if final_pips < 10:
                    exit_method = 'vwap'
                    vwap_exit = True
                else:
                    exit_method = 'manual'
            else:
                # Has partials but unknown exit
                if pc1_triggered and pc2_triggered:
                    exit_method = 'trailing_stop'  # Assume trailing after PC2
                    trailing_triggered = True
                elif pc1_triggered:
                    exit_method = 'pc1_only'
                else:
                    exit_method = 'manual'

            # Calculate pips given back from peak (for trailing stops)
            pips_from_peak = peak_pips - final_pips

            return {
                'exit_method': exit_method,
                'pc1_triggered': bool(pc1_triggered),
                'pc2_triggered': bool(pc2_triggered),
                'trailing_triggered': bool(trailing_triggered),
                'vwap_exit': bool(vwap_exit),
                'final_pips': float(final_pips),
                'peak_pips': float(peak_pips),
                'pips_from_peak': float(pips_from_peak),  # How much given back
                'capture_ratio': float(final_pips / peak_pips) if peak_pips > 0 else 1.0,  # % of peak captured
                'partial_count': int(partial_closes['count']),
            }

        except Exception as e:
            print(f"[WARN] Exit strategy calculation failed: {e}")
            return {
                'exit_method': 'unknown',
                'pc1_triggered': False,
                'pc2_triggered': False,
                'trailing_triggered': False,
                'vwap_exit': False,
                'final_pips': 0.0,
                'peak_pips': 0.0,
                'pips_from_peak': 0.0,
                'capture_ratio': 1.0,
                'partial_count': 0,
            }

    def _aggregate_trailing_events(self, ticket: int) -> Dict:
        """
        Aggregate all trailing stop events for a specific trade ticket.

        Reads from trailing_stop_events.jsonl and consolidates:
        - Stop loss data (initial, final, was hit)
        - Break even data (when/if activated)
        - PC1/PC2 trigger data (timing, pips)

        Args:
            ticket: Position ticket to aggregate events for

        Returns:
            Dict with aggregated SL/BE/PC data
        """
        events_log = self.output_dir / "trailing_stop_events.jsonl"

        # Default structure
        aggregated = {
            'stop_loss': {
                'initial_price': None,
                'initial_pips': None,
                'final_price': None,
                'was_hit': False,
                'sl_type': None,  # 'hard_adx', 'normal', 'recovery_disabled'
                'adx_at_entry': None,
            },
            'break_even': {
                'activated': False,
                'trigger_price': None,
                'trigger_time': None,
            },
            'pc_data': {
                'pc1_triggered': False,
                'pc1_price': None,
                'pc1_pips': None,
                'pc1_target_pips': None,
                'pc1_time': None,
                'pc2_triggered': False,
                'pc2_price': None,
                'pc2_pips': None,
                'pc2_time': None,
                'trailing_distance_pips': None,
            },
            'trailing': {
                'active': False,
                'updates_count': 0,
                'final_stop_price': None,
                'peak_price': None,
                'capture_ratio': None,
            },
            'mfe_mae': {
                'mfe_price': None,
                'mae_price': None,
                'mfe_pips': None,
                'mae_pips': None,  # How far price went against before profit/exit
            }
        }

        if not events_log.exists():
            return aggregated

        try:
            with open(events_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get('ticket') != ticket:
                            continue

                        event_type = event.get('event_type', '')

                        # Hard SL set at entry
                        if event_type == 'hard_sl_set':
                            aggregated['stop_loss']['initial_price'] = event.get('sl_price')
                            aggregated['stop_loss']['initial_pips'] = event.get('sl_pips')
                            aggregated['stop_loss']['sl_type'] = event.get('reason')
                            aggregated['stop_loss']['adx_at_entry'] = event.get('adx')

                        # SL hit (exit via stop)
                        elif event_type == 'sl_hit':
                            aggregated['stop_loss']['was_hit'] = True
                            aggregated['stop_loss']['final_price'] = event.get('exit_price')
                            if not aggregated['stop_loss']['sl_type']:
                                aggregated['stop_loss']['sl_type'] = event.get('sl_type')

                        # SL moved to breakeven
                        elif event_type == 'sl_to_breakeven':
                            aggregated['break_even']['activated'] = True
                            aggregated['break_even']['trigger_price'] = event.get('breakeven_price')
                            aggregated['break_even']['trigger_time'] = event.get('timestamp')
                            aggregated['stop_loss']['final_price'] = event.get('breakeven_price')

                        # PC1 trigger
                        elif event_type == 'pc1_trigger':
                            aggregated['pc_data']['pc1_triggered'] = True
                            aggregated['pc_data']['pc1_price'] = event.get('current_price')
                            aggregated['pc_data']['pc1_pips'] = event.get('profit_pips')
                            aggregated['pc_data']['pc1_target_pips'] = event.get('pc1_pips_target')
                            aggregated['pc_data']['pc1_time'] = event.get('timestamp')

                        # PC2 trigger
                        elif event_type == 'pc2_trigger':
                            aggregated['pc_data']['pc2_triggered'] = True
                            aggregated['pc_data']['pc2_price'] = event.get('current_price')
                            aggregated['pc_data']['pc2_pips'] = event.get('profit_pips')
                            aggregated['pc_data']['pc2_time'] = event.get('timestamp')
                            aggregated['pc_data']['trailing_distance_pips'] = event.get('trailing_distance_pips')
                            aggregated['trailing']['active'] = True
                            aggregated['trailing']['final_stop_price'] = event.get('trailing_stop_price')

                        # Trailing stop update
                        elif event_type == 'trailing_update':
                            aggregated['trailing']['updates_count'] += 1
                            aggregated['trailing']['final_stop_price'] = event.get('new_stop')

                        # Trailing stop hit
                        elif event_type == 'trailing_hit':
                            aggregated['trailing']['final_stop_price'] = event.get('trailing_stop_price')
                            aggregated['trailing']['peak_price'] = event.get('peak_price')
                            aggregated['trailing']['capture_ratio'] = event.get('capture_ratio')
                            aggregated['stop_loss']['was_hit'] = True  # Trailing is a form of SL

                        # MFE/MAE data
                        elif event_type == 'mfe_mae':
                            aggregated['mfe_mae']['mfe_price'] = event.get('mfe_price')
                            aggregated['mfe_mae']['mae_price'] = event.get('mae_price')
                            aggregated['mfe_mae']['mfe_pips'] = event.get('mfe_pips')
                            aggregated['mfe_mae']['mae_pips'] = event.get('mae_pips')

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"[WARN] Failed to aggregate trailing events for {ticket}: {e}")

        return aggregated

    def update_closed_trades(self):
        """Check for closed positions and update their records with outcome data"""
        if not self.mt5:
            return

        # Get recent closed positions (last 24 hours)
        to_date = datetime.now()
        from_date = to_date - timedelta(hours=24)

        # Get closed positions (thread-safe)
        deals = self._with_lock(lambda: self.mt5.history_deals_get(from_date, to_date))
        if not deals:
            return

        updated_count = 0

        # Read existing log to find trades that need outcome updates
        if not self.continuous_log.exists():
            return

        trades_to_update = []
        with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    # Skip if already has outcome
                    if 'outcome' not in trade or trade['outcome'].get('status') == 'open':
                        trades_to_update.append(trade)
                except:
                    continue

        # Check each open trade for closure
        for trade in trades_to_update:
            ticket = trade['ticket']

            # Find FINAL exit deal for this ticket (last exit, not partials)
            # A position can have multiple exit deals: PC1, PC2, and final close
            # We want the LAST one (final close) to get correct exit price/time
            exit_deal = None
            for deal in deals:
                if deal.position_id == ticket and deal.entry == 1:  # Exit deal
                    exit_deal = deal  # Keep overwriting - last one wins

            if exit_deal:
                # Trade is closed - get recovery and partial close data
                # Handle multiple timestamp formats
                entry_time_str = trade.get('entry_time')
                if not entry_time_str and 'execution_quality' in trade:
                    entry_time_str = trade['execution_quality'].get('timestamp')
                if not entry_time_str:
                    continue  # Skip trades without timestamp
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                exit_time = datetime.fromtimestamp(exit_deal.time)

                # Use exit_time + 1 min buffer for deal searches
                # PC2 and final close can happen at the same second
                search_end = exit_time + timedelta(minutes=1)

                # Find recovery actions
                recovery = self.find_recovery_actions(ticket, entry_time, search_end)

                # Find partial closes
                partial_closes = self.find_partial_closes(ticket, entry_time, search_end)

                # Calculate hold time
                hold_seconds = (exit_time - entry_time).total_seconds()
                hold_hours = hold_seconds / 3600

                # Calculate exit strategy data
                exit_strategy = self._calculate_exit_strategy_data(
                    entry_price=trade['entry_price'],
                    exit_price=float(exit_deal.price),
                    direction=trade.get('direction', 'BUY'),
                    partial_closes=partial_closes,
                    exit_comment=exit_deal.comment or ""
                )

                # Aggregate trailing events (SL/BE/PC data) for this trade
                trailing_data = self._aggregate_trailing_events(ticket)

                # Detect if SL was hit by comparing exit price to known SL price
                # This handles cases where we didn't get real-time sl_hit event
                sl_data = trailing_data['stop_loss']
                exit_price = float(exit_deal.price)
                entry_price = trade['entry_price']
                direction = trade.get('direction', 'BUY')

                # Infer SL hit if:
                # 1. We have a recorded SL price AND
                # 2. Exit was at or near that price (within 2 pips tolerance)
                if sl_data.get('initial_price') and not sl_data.get('was_hit'):
                    sl_price = sl_data['initial_price']
                    # Check BE SL first (if activated)
                    if trailing_data['break_even']['activated'] and trailing_data['break_even'].get('trigger_price'):
                        sl_price = trailing_data['break_even']['trigger_price']
                    # Check trailing SL (if active)
                    elif trailing_data['trailing']['active'] and trailing_data['trailing'].get('final_stop_price'):
                        sl_price = trailing_data['trailing']['final_stop_price']

                    # Compare exit price to SL price (within 2 pip tolerance)
                    pip_tolerance = 0.0002  # 2 pips
                    if abs(exit_price - sl_price) < pip_tolerance:
                        sl_data['was_hit'] = True
                        sl_data['final_price'] = exit_price

                # Build outcome data
                outcome = {
                    'status': 'closed',
                    'exit_time': exit_time.isoformat(),
                    'exit_price': float(exit_deal.price),
                    'profit': float(exit_deal.profit),
                    'hold_hours': float(hold_hours),
                    'recovery': recovery,
                    'partial_closes': partial_closes,
                    'exit_strategy': exit_strategy,  # Exit strategy metrics
                    'stop_loss': sl_data,  # SL tracking (with inferred hit detection)
                    'break_even': trailing_data['break_even'],  # BE tracking
                    'pc_data': trailing_data['pc_data'],  # PC1/PC2 detailed tracking
                    'trailing': trailing_data['trailing'],  # Trailing stop tracking
                    'net_profit': float(exit_deal.profit) - recovery['recovery_cost'],
                    'had_recovery': recovery['dca_count'] + recovery['hedge_count'] > 0,
                    'had_grid': recovery['grid_count'] > 0,
                    'had_partial_close': partial_closes['count'] > 0,
                    'updated_at': datetime.now().isoformat()
                }

                trade['outcome'] = outcome
                updated_count += 1

                # Mirror outcome to SQLite
                if self.trade_db:
                    try:
                        self.trade_db.update_outcome(ticket, outcome)
                    except Exception as e:
                        print(f"[WARN] SQLite update failed for #{ticket}: {e}")

                print(f"[UPDATE] Trade #{ticket} closed: ${outcome['profit']:.2f} | "
                      f"DCA: {recovery['dca_count']} | Hedge: {recovery['hedge_count']} | "
                      f"Grid: {recovery['grid_count']} | Partials: {partial_closes['count']}")

        # Rewrite log with updates
        if updated_count > 0:
            all_trades = []
            with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        all_trades.append(json.loads(line))
                    except:
                        continue

            # Update trades
            for i, trade in enumerate(all_trades):
                for updated_trade in trades_to_update:
                    if trade['ticket'] == updated_trade['ticket'] and 'outcome' in updated_trade:
                        all_trades[i] = updated_trade
                        break

            # Write back
            with open(self.continuous_log, 'w', encoding='utf-8', errors='ignore') as f:
                for trade in all_trades:
                    trade = convert_numpy_types(trade)
                    f.write(json.dumps(trade) + '\n')

            print(f"[OK] Updated {updated_count} closed trades with outcome data")

            # Record trades to setup classifier for tier learning
            try:
                from strategies.setup_classifier import get_setup_classifier
                classifier = get_setup_classifier()
                for trade in trades_to_update:
                    if 'outcome' in trade and 'confluence_factors' in trade:
                        # Extract context for granular setup tracking
                        market_context = trade.get('market_context', {})
                        classifier.record_trade(
                            factors=trade.get('confluence_factors', []),
                            direction=trade.get('direction', 'buy'),
                            symbol=trade.get('symbol', 'UNKNOWN'),
                            is_win=trade['outcome'].get('profit', 0) > 0,
                            profit=trade['outcome'].get('profit', 0),
                            htf_score=trade.get('htf_score', 0),
                            entry_score=trade.get('entry_score', 0),
                            ticket=trade.get('ticket'),
                            strategy_type=trade.get('strategy_type'),       # NEW: Pass strategy
                            hour=market_context.get('hour'),                 # NEW: Pass hour
                            day_of_week=market_context.get('day_of_week')    # NEW: Pass day
                        )
                print(f"[SETUP CLASSIFIER] Recorded {len(trades_to_update)} trades for tier learning")
            except Exception as e:
                print(f"[WARN] Setup classifier recording failed: {e}")

            # Trigger auto-tuner for each closed trade
            auto_tuner = _get_auto_tuner()
            if auto_tuner:
                for _ in range(updated_count):
                    auto_tuner.on_trade_closed()

    def backfill_missed_trades(self):
        """Backfill any trades that were missed while logger was offline"""
        if not self.mt5:
            print("[BACKFILL] MT5 not connected, skipping backfill")
            return

        try:
            # Find last logged trade timestamp
            last_logged_time = None
            if self.continuous_log.exists():
                with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            trade = json.loads(line)
                            # Handle multiple timestamp formats
                            entry_time_str = trade.get('entry_time')
                            if not entry_time_str and 'execution_quality' in trade:
                                entry_time_str = trade['execution_quality'].get('timestamp')
                            if not entry_time_str:
                                continue
                            trade_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                            if last_logged_time is None or trade_time > last_logged_time:
                                last_logged_time = trade_time
                        except:
                            continue

            # If no trades logged, look back 30 days to capture all available history
            to_date = datetime.now()
            if last_logged_time:
                from_date = last_logged_time
                print(f"[BACKFILL] Checking for missed trades since {from_date.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                from_date = to_date - timedelta(days=30)
                print(f"[BACKFILL] No existing log, checking last 30 days for initial dataset")

            # Get all deals in this period (this can take time for large date ranges)
            print(f"[BACKFILL] Fetching deals from MT5...")
            deals = self._with_lock(lambda: self.mt5.history_deals_get(from_date, to_date))
            if not deals:
                print("[BACKFILL] No deals found to backfill")
                return

            print(f"[BACKFILL] Found {len(deals)} total deals, processing...")

            backfilled = 0
            for deal in deals:
                comment = deal.comment or ""

                # Only log confluence entry trades
                # Old format: "Confluence:", "VWAP:", "BREAKOUT:"
                # New format: "MR|GLD|C8", "BO|HI|C5"
                is_entry_trade = any(marker in comment for marker in ['Confluence:', 'VWAP:', 'BREAKOUT:', 'MR|', 'BO|'])
                if not is_entry_trade:
                    continue

                # Only log entry deals (type 0 or 1)
                if deal.type not in [0, 1]:
                    continue

                ticket = deal.position_id

                # Skip if already logged
                if ticket in self.logged_tickets:
                    continue

                # Log this trade (same logic as check_for_new_trades)
                try:
                    trade_time = datetime.fromtimestamp(deal.time)
                    symbol = deal.symbol
                    entry_price = deal.price

                    # Parse strategy type and confluence score from comment
                    # New format: "MR|GLD|C8", "BO|HI|C5"
                    # Old format: "VWAP:C12", "BREAKOUT:C8", "Confluence:12"
                    strategy_type = None
                    confluence_score = None

                    try:
                        # New format: "MR|GLD|C8" or "BO|HI|C5"
                        if comment.startswith('MR|'):
                            strategy_type = 'mean_reversion'
                            # Parse "MR|GLD|C8" -> confluence_score = 8
                            parts = comment.split('|')
                            if len(parts) >= 3 and parts[2].startswith('C'):
                                confluence_score = int(parts[2][1:].split()[0])
                        elif comment.startswith('BO|'):
                            strategy_type = 'breakout'  # Momentum through levels
                            # Parse "BO|HI|C5" -> confluence_score = 5
                            parts = comment.split('|')
                            if len(parts) >= 3 and parts[2].startswith('C'):
                                confluence_score = int(parts[2][1:].split()[0])
                        # Legacy formats
                        elif 'VWAP:' in comment:
                            strategy_type = 'mean_reversion'
                            confluence_score = int(comment.split('VWAP:C')[1].split()[0])
                        elif 'BREAKOUT:' in comment:
                            strategy_type = 'breakout'
                            confluence_score = int(comment.split('BREAKOUT:C')[1].split()[0])
                        elif 'Confluence:' in comment:
                            strategy_type = 'confluence'  # Legacy format
                            confluence_score = int(comment.split('Confluence:')[1].split()[0])
                    except:
                        pass

                    # Get bars at entry time
                    h1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_H1, trade_time, bars=500)
                    d1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_D1, trade_time, bars=100)
                    w1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_W1, trade_time, bars=50)

                    if h1_bars is None or d1_bars is None or w1_bars is None:
                        continue

                    # Calculate confluence factors
                    confluence_factors = self.calculate_confluence_factors(h1_bars, d1_bars, w1_bars, entry_price)

                    if not confluence_factors:
                        continue

                    # Build factor list for setup classifier
                    # For BO trades, use DI-specific factors from comment instead of reconstructing MR factors
                    factor_list = []
                    if strategy_type == 'breakout' and comment.startswith('BO|'):
                        # Parse BO-specific factors from comment: "BO|DI|HI|C5" or "BO|WK|HI|C5"
                        parts = comment.split('|')
                        subtype_labels = {'DI': 'DI Momentum', 'RB': 'Range Breakout', 'LV': 'LVN Breakout', 'WK': 'Weekly Breakout'}
                        bo_label = subtype_labels.get(parts[1], 'Breakout') if len(parts) >= 2 else 'Breakout'
                        conf_label = {'HI': 'High', 'ME': 'Medium', 'LO': 'Low'}.get(parts[2] if len(parts) >= 4 else parts[1] if len(parts) >= 3 else '', '')
                        factor_list.append(f'{bo_label} ({conf_label} confidence)')
                    else:
                        # MR trades: reconstruct from volume profile / VWAP / HTF data
                        vwap_data = confluence_factors.get('vwap', {})
                        vp_data = confluence_factors.get('volume_profile', {})
                        htf_data = confluence_factors.get('htf_levels', {})

                        if vwap_data.get('in_band_1'):
                            factor_list.append('VWAP Band 1')
                        if vwap_data.get('in_band_2'):
                            factor_list.append('VWAP Band 2')
                        if vp_data.get('at_poc'):
                            factor_list.append('POC')
                        if vp_data.get('above_vah'):
                            factor_list.append('Above VAH')
                        if vp_data.get('below_val'):
                            factor_list.append('Below VAL')
                        if vp_data.get('at_lvn'):
                            factor_list.append('Low Volume Node')
                        if vp_data.get('at_swing_high'):
                            factor_list.append('Swing High')
                        if vp_data.get('at_swing_low'):
                            factor_list.append('Swing Low')
                        # Add HTF factors
                        factor_list.extend(htf_data.get('factors_matched', []))

                    # Calculate market microstructure first (used for execution_quality)
                    market_micro = self._calculate_market_microstructure(symbol, entry_price, deal.price)

                    # Calculate execution quality using spread from market_microstructure
                    # Note: expected_price not available from MT5 deal history, use actual price
                    exec_quality = self._calculate_execution_quality(
                        symbol=symbol,
                        expected_price=entry_price,  # Best approximation from deal
                        actual_price=deal.price,
                        spread_at_entry_pips=market_micro.get('spread_pips', 0.0),
                        fill_time_ms=0,  # Not available from MT5 deal history
                        requotes=0  # Not available from MT5 deal history
                    )

                    # Build trade record
                    trade_record = {
                        'ticket': int(ticket),
                        'symbol': symbol,
                        'entry_time': trade_time.isoformat(),
                        'entry_price': float(entry_price),
                        'direction': 'BUY' if deal.type == 0 else 'SELL',
                        'volume': float(deal.volume),
                        'strategy_type': strategy_type,  # 'revision', 'breakout', or 'confluence' (legacy)
                        'confluence_score': confluence_score,
                        'confluence_factors': factor_list,  # For setup classifier
                        'vwap': confluence_factors.get('vwap', {}),
                        'volume_profile': confluence_factors.get('volume_profile', {}),
                        'htf_levels': confluence_factors.get('htf_levels', {}),
                        'fair_value_gaps': confluence_factors.get('fair_value_gaps', {}),
                        'trend_filter': confluence_factors.get('trend_filter', {}),
                        'candle_momentum': confluence_factors.get('candle_momentum', {}),  # Candle body trend
                        'regime_state': self._get_regime_state(),  # HMM regime at entry
                        'volatility': confluence_factors.get('volatility', {}),  # NEW: Phase 1
                        'entry_quality': confluence_factors.get('entry_quality', {}),  # NEW: Phase 1
                        'execution_quality': exec_quality,  # NEW: Execution quality tracking
                        'market_context': {
                            'hour': trade_time.hour,
                            'day_of_week': trade_time.strftime('%A'),
                            'session': self._get_trading_session(trade_time.hour),
                        },
                        'trade_sequencing': self._calculate_trade_sequencing(trade_time),  # NEW: Phase 1
                        'market_microstructure': market_micro,  # NEW: Phase 3
                        'position_sizing': self._calculate_position_sizing_context(symbol, deal.volume, trade_time),  # NEW: Phase 3
                        'logged_at': datetime.now().isoformat()
                    }

                    # Convert numpy types to native Python types
                    trade_record = convert_numpy_types(trade_record)

                    # Append to continuous log
                    with open(self.continuous_log, 'a', encoding='utf-8', errors='ignore') as f:
                        f.write(json.dumps(trade_record) + '\n')

                    # Mirror to SQLite
                    if self.trade_db:
                        try:
                            self.trade_db.insert_trade(trade_record)
                        except Exception as e:
                            print(f"[WARN] SQLite backfill insert failed for #{ticket}: {e}")

                    # Mark as logged
                    self.logged_tickets.add(ticket)
                    backfilled += 1

                    strategy_label = strategy_type.upper() if strategy_type else 'UNKNOWN'
                    print(f"[BACKFILL] {symbol} #{ticket} ({strategy_label}, C{confluence_score})")

                except Exception as e:
                    print(f"[WARN] [BACKFILL] Failed to log trade {ticket}: {e}")
                    import traceback
                    traceback.print_exc()

            if backfilled > 0:
                print(f"[OK] Backfilled {backfilled} missed trades")
            else:
                print(f"[OK] No missed trades to backfill")

        except Exception as e:
            print(f"[ERROR] Backfill failed: {e}")
            import traceback
            traceback.print_exc()

    def check_for_new_trades(self):
        """Check for new trades and log them"""
        if not self.mt5:
            return

        # Get recent deals (last 48 hours to catch missed trades) (thread-safe)
        to_date = datetime.now()
        from_date = to_date - timedelta(hours=48)
        deals = self._with_lock(lambda: self.mt5.history_deals_get(from_date, to_date))

        if not deals:
            return

        new_trades_logged = 0

        for deal in deals:
            comment = deal.comment or ""

            # Only log confluence entry trades
            # Old format: "Confluence:", "VWAP:", "BREAKOUT:"
            # New format: "MR|GLD|C8", "BO|HI|C5"
            is_entry_trade = any(marker in comment for marker in ['Confluence:', 'VWAP:', 'BREAKOUT:', 'MR|', 'BO|'])
            if not is_entry_trade:
                continue

            # Only log entry deals (type 0 or 1)
            if deal.type not in [0, 1]:
                continue

            ticket = deal.position_id

            # Skip if already logged
            if ticket in self.logged_tickets:
                continue

            # Log this trade
            try:
                trade_time = datetime.fromtimestamp(deal.time)
                symbol = deal.symbol
                entry_price = deal.price

                # Parse strategy type and confluence score from comment
                # New format: "MR|GLD|C8" or "BO|RB|HI|C5"
                # Legacy: "VWAP:C12", "BREAKOUT:C8", "Confluence:12", "BO|HI|C5"
                strategy_type = None
                confluence_score = None
                breakout_subtype = None
                breakout_confidence = None

                try:
                    # New format: "MR|GLD|C8" or "BO|RB|HI|C5"
                    if comment.startswith('MR|'):
                        strategy_type = 'mean_reversion'
                        parts = comment.split('|')
                        if len(parts) >= 3 and parts[2].startswith('C'):
                            confluence_score = int(parts[2][1:].split()[0])
                    elif comment.startswith('BO|'):
                        strategy_type = 'breakout'  # Momentum through levels
                        parts = comment.split('|')
                        # New format: "BO|RB|HI|C5" (4 parts with subtype)
                        if len(parts) >= 4 and parts[3].startswith('C'):
                            subtype_map = {'RB': 'range_breakout', 'LV': 'lvn_breakout', 'WK': 'weekly_breakout', 'DI': 'di_momentum'}
                            breakout_subtype = subtype_map.get(parts[1], 'unknown')
                            breakout_confidence = parts[2]
                            confluence_score = int(parts[3][1:].split()[0])
                        # Legacy format: "BO|HI|C5" (3 parts, no subtype)
                        elif len(parts) >= 3 and parts[2].startswith('C'):
                            breakout_subtype = 'unknown'
                            breakout_confidence = parts[1]
                            confluence_score = int(parts[2][1:].split()[0])
                    # Legacy formats
                    elif 'VWAP:' in comment:
                        strategy_type = 'mean_reversion'
                        confluence_score = int(comment.split('VWAP:C')[1].split()[0])
                    elif 'BREAKOUT:' in comment:
                        strategy_type = 'breakout'
                        confluence_score = int(comment.split('BREAKOUT:C')[1].split()[0])
                    elif 'Confluence:' in comment:
                        strategy_type = 'confluence'  # Legacy format
                        confluence_score = int(comment.split('Confluence:')[1].split()[0])
                except:
                    pass

                # Get bars at entry time
                h1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_H1, trade_time, bars=500)
                d1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_D1, trade_time, bars=100)
                w1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_W1, trade_time, bars=50)

                if h1_bars is None or d1_bars is None or w1_bars is None:
                    continue

                # Calculate confluence factors
                confluence_factors = self.calculate_confluence_factors(h1_bars, d1_bars, w1_bars, entry_price)

                if not confluence_factors:
                    continue

                # Build factor list for setup classifier
                # For BO trades, use DI-specific factors from comment instead of reconstructing MR factors
                factor_list = []
                if strategy_type == 'breakout' and comment.startswith('BO|'):
                    # Parse BO-specific factors from comment: "BO|DI|HI|C5" or "BO|WK|HI|C5"
                    parts = comment.split('|')
                    subtype_labels = {'DI': 'DI Momentum', 'RB': 'Range Breakout', 'LV': 'LVN Breakout', 'WK': 'Weekly Breakout'}
                    bo_label = subtype_labels.get(parts[1], 'Breakout') if len(parts) >= 2 else 'Breakout'
                    conf_label = {'HI': 'High', 'ME': 'Medium', 'LO': 'Low'}.get(parts[2] if len(parts) >= 4 else parts[1] if len(parts) >= 3 else '', '')
                    factor_list.append(f'{bo_label} ({conf_label} confidence)')
                else:
                    # MR trades: reconstruct from volume profile / VWAP / HTF data
                    vwap_data = confluence_factors.get('vwap', {})
                    vp_data = confluence_factors.get('volume_profile', {})
                    htf_data = confluence_factors.get('htf_levels', {})

                    if vwap_data.get('in_band_1'):
                        factor_list.append('VWAP Band 1')
                    if vwap_data.get('in_band_2'):
                        factor_list.append('VWAP Band 2')
                    if vp_data.get('at_poc'):
                        factor_list.append('POC')
                    if vp_data.get('above_vah'):
                        factor_list.append('Above VAH')
                    if vp_data.get('below_val'):
                        factor_list.append('Below VAL')
                    if vp_data.get('at_lvn'):
                        factor_list.append('Low Volume Node')
                    if vp_data.get('at_swing_high'):
                        factor_list.append('Swing High')
                    if vp_data.get('at_swing_low'):
                        factor_list.append('Swing Low')
                    # Add HTF factors
                    factor_list.extend(htf_data.get('factors_matched', []))

                # Calculate market microstructure first (used for execution_quality)
                market_micro = self._calculate_market_microstructure(symbol, entry_price, deal.price)

                # Calculate execution quality using spread from market_microstructure
                # Note: expected_price not available from MT5 deal history, use actual price
                exec_quality = self._calculate_execution_quality(
                    symbol=symbol,
                    expected_price=entry_price,  # Best approximation from deal
                    actual_price=deal.price,
                    spread_at_entry_pips=market_micro.get('spread_pips', 0.0),
                    fill_time_ms=0,  # Not available from MT5 deal history
                    requotes=0  # Not available from MT5 deal history
                )

                # Build trade record
                trade_record = {
                    'ticket': int(ticket),
                    'symbol': symbol,
                    'entry_time': trade_time.isoformat(),
                    'entry_price': float(entry_price),
                    'direction': 'BUY' if deal.type == 0 else 'SELL',
                    'volume': float(deal.volume),
                    'strategy_type': strategy_type,  # 'revision', 'breakout', or 'confluence' (legacy)
                    'breakout_subtype': breakout_subtype,  # range_breakout, lvn_breakout, weekly_breakout, di_momentum
                    'breakout_confidence': breakout_confidence,  # HI, ME, LO
                    'confluence_score': confluence_score,
                    'confluence_factors': factor_list,  # For setup classifier
                    'vwap': confluence_factors.get('vwap', {}),
                    'volume_profile': confluence_factors.get('volume_profile', {}),
                    'htf_levels': confluence_factors.get('htf_levels', {}),
                    'fair_value_gaps': confluence_factors.get('fair_value_gaps', {}),
                    'trend_filter': confluence_factors.get('trend_filter', {}),
                    'candle_momentum': confluence_factors.get('candle_momentum', {}),  # Candle body trend
                    'regime_state': self._get_regime_state(),  # HMM regime at entry
                    'volatility': confluence_factors.get('volatility', {}),  # NEW: Phase 1
                    'entry_quality': confluence_factors.get('entry_quality', {}),  # NEW: Phase 1
                    'execution_quality': exec_quality,  # NEW: Execution quality tracking
                    'market_context': {
                        'hour': trade_time.hour,
                        'day_of_week': trade_time.strftime('%A'),
                        'session': self._get_trading_session(trade_time.hour),
                    },
                    'trade_sequencing': self._calculate_trade_sequencing(trade_time),  # NEW: Phase 1
                    'market_microstructure': market_micro,  # NEW: Phase 3
                    'position_sizing': self._calculate_position_sizing_context(symbol, deal.volume, trade_time),  # NEW: Phase 3
                    'logged_at': datetime.now().isoformat()
                }

                # Convert numpy types to native Python types
                trade_record = convert_numpy_types(trade_record)

                # Append to continuous log
                with open(self.continuous_log, 'a', encoding='utf-8', errors='ignore') as f:
                    f.write(json.dumps(trade_record) + '\n')

                # Mirror to SQLite
                if self.trade_db:
                    try:
                        self.trade_db.insert_trade(trade_record)
                    except Exception as e:
                        print(f"[WARN] SQLite insert failed for #{ticket}: {e}")

                # Mark as logged
                self.logged_tickets.add(ticket)
                new_trades_logged += 1

                strategy_label = strategy_type.upper() if strategy_type else 'UNKNOWN'
                print(f"[LOG] New trade: {symbol} #{ticket} ({strategy_label}, C{confluence_score})")

            except Exception as e:
                print(f"[WARN] [LOGGER] Failed to log trade {ticket}: {e}")

        if new_trades_logged > 0:
            print(f"[OK] Logged {new_trades_logged} new trades")

    def log_trailing_event(self, event_type: str, ticket: int, symbol: str, **kwargs):
        """
        Log real-time trailing stop events for ML monitoring.

        Events tracked:
        - pc2_trigger: When PC2 closes and trailing activates
        - sl_to_breakeven: When hardware SL moves to breakeven
        - trailing_update: When trailing stop moves with price
        - trailing_hit: When trailing stop closes position

        Args:
            event_type: Type of event (pc2_trigger, sl_to_breakeven, trailing_update, trailing_hit)
            ticket: Position ticket
            symbol: Trading symbol
            **kwargs: Event-specific data
        """
        try:
            # Create events log if it doesn't exist
            events_log = self.output_dir / "trailing_stop_events.jsonl"

            event_record = {
                'event_type': event_type,
                'ticket': ticket,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                **kwargs  # Include all additional event data
            }

            # Convert numpy types
            event_record = convert_numpy_types(event_record)

            # Append to events log
            with open(events_log, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(json.dumps(event_record) + '\n')

        except Exception as e:
            print(f"[WARN] [LOGGER] Failed to log trailing event for {ticket}: {e}")

    def log_pc1_trigger(self, ticket: int, symbol: str, current_price: float,
                        entry_price: float, profit_pips: float, close_volume: float,
                        pc1_pips_target: float):
        """
        Log when PC1 triggers (first 25% partial close).

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Entry price
            profit_pips: Profit in pips at trigger
            close_volume: Volume closed
            pc1_pips_target: PC1 target in pips for this instrument
        """
        self.log_trailing_event(
            event_type='pc1_trigger',
            ticket=ticket,
            symbol=symbol,
            current_price=float(current_price),
            entry_price=float(entry_price),
            profit_pips=float(profit_pips),
            close_volume=float(close_volume),
            pc1_pips_target=float(pc1_pips_target),
            vwap_exits_disabled=True
        )

    def log_pc2_trigger(self, ticket: int, symbol: str, current_price: float,
                        entry_price: float, trailing_distance_pips: float,
                        trailing_stop_price: float):
        """
        Log when PC2 triggers (25% close, trailing activates, SL to BE).

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Entry price (will become new SL)
            trailing_distance_pips: ATR-based trailing distance
            trailing_stop_price: Initial trailing stop price
        """
        profit_pips = abs(current_price - entry_price) * 10000
        self.log_trailing_event(
            event_type='pc2_trigger',
            ticket=ticket,
            symbol=symbol,
            current_price=float(current_price),
            entry_price=float(entry_price),
            profit_pips=float(profit_pips),
            trailing_distance_pips=float(trailing_distance_pips),
            trailing_stop_price=float(trailing_stop_price),
            sl_moved_to_breakeven=True
        )

    def log_sl_to_breakeven(self, ticket: int, symbol: str, breakeven_price: float):
        """
        Log when hardware SL moves to breakeven at PC2.

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            breakeven_price: Breakeven price (entry price)
        """
        self.log_trailing_event(
            event_type='sl_to_breakeven',
            ticket=ticket,
            symbol=symbol,
            breakeven_price=float(breakeven_price)
        )

    def log_trailing_update(self, ticket: int, symbol: str, old_stop: float,
                            new_stop: float, current_price: float, pips_moved: float):
        """
        Log when trailing stop moves up with price.

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            old_stop: Previous stop price
            new_stop: New stop price
            current_price: Current market price
            pips_moved: How many pips the stop moved
        """
        self.log_trailing_event(
            event_type='trailing_update',
            ticket=ticket,
            symbol=symbol,
            old_stop=float(old_stop),
            new_stop=float(new_stop),
            current_price=float(current_price),
            pips_moved=float(pips_moved)
        )

    def log_trailing_hit(self, ticket: int, symbol: str, trailing_stop_price: float,
                         current_price: float, entry_price: float, peak_price: float):
        """
        Log when trailing stop is hit and position closes.

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            trailing_stop_price: Trailing stop price that was hit
            current_price: Price when stop hit
            entry_price: Original entry price
            peak_price: Highest profitable price reached
        """
        final_pips = abs(current_price - entry_price) * 10000
        peak_pips = abs(peak_price - entry_price) * 10000
        pips_from_peak = peak_pips - final_pips
        capture_ratio = final_pips / peak_pips if peak_pips > 0 else 0

        self.log_trailing_event(
            event_type='trailing_hit',
            ticket=ticket,
            symbol=symbol,
            trailing_stop_price=float(trailing_stop_price),
            exit_price=float(current_price),
            entry_price=float(entry_price),
            peak_price=float(peak_price),
            final_pips=float(final_pips),
            peak_pips=float(peak_pips),
            pips_from_peak=float(pips_from_peak),
            capture_ratio=float(capture_ratio)
        )

    def log_mfe_mae(self, ticket: int, symbol: str, entry_price: float,
                    mfe_price: float, mae_price: float, direction: str,
                    strategy_type: str = None):
        """
        Log Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE) at position close.

        MFE = Best price reached during trade (highest for BUY, lowest for SELL)
        MAE = Worst price reached during trade (lowest for BUY, highest for SELL)

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            entry_price: Original entry price
            mfe_price: Best price reached (highest_profit_price)
            mae_price: Worst price reached (lowest_profit_price)
            direction: 'buy' or 'sell'
            strategy_type: 'mean_reversion' or 'breakout'
        """
        # Calculate MFE/MAE in pips
        if direction.lower() == 'buy':
            mfe_pips = (mfe_price - entry_price) * 10000
            mae_pips = (entry_price - mae_price) * 10000  # Positive = how much went against
        else:  # sell
            mfe_pips = (entry_price - mfe_price) * 10000
            mae_pips = (mae_price - entry_price) * 10000  # Positive = how much went against

        self.log_trailing_event(
            event_type='mfe_mae',
            ticket=ticket,
            symbol=symbol,
            entry_price=float(entry_price),
            mfe_price=float(mfe_price),
            mae_price=float(mae_price),
            mfe_pips=float(mfe_pips),
            mae_pips=float(mae_pips),
            direction=direction.lower(),
            strategy_type=strategy_type
        )

    def log_sl_hit(self, ticket: int, symbol: str, sl_type: str, entry_price: float,
                   exit_price: float, sl_price: float, loss_pips: float,
                   adx_at_entry: float = None, strategy_type: str = None):
        """
        Log when a stop loss is hit.

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            sl_type: Type of SL ('hard_sl', 'trailing_sl', 'be_sl', 'stack_sl')
            entry_price: Original entry price
            exit_price: Actual exit price
            sl_price: Stop loss price that was set
            loss_pips: Loss in pips
            adx_at_entry: ADX value when trade was opened
            strategy_type: 'mean_reversion' or 'breakout'
        """
        self.log_trailing_event(
            event_type='sl_hit',
            ticket=ticket,
            symbol=symbol,
            sl_type=sl_type,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            sl_price=float(sl_price),
            loss_pips=float(loss_pips),
            adx_at_entry=float(adx_at_entry) if adx_at_entry else None,
            strategy_type=strategy_type
        )

    def log_hard_sl_set(self, ticket: int, symbol: str, entry_price: float,
                        sl_price: float, sl_pips: float, adx: float,
                        reason: str, strategy_type: str = None):
        """
        Log when a hard stop loss is set on entry.

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            entry_price: Entry price
            sl_price: Stop loss price
            sl_pips: Stop loss distance in pips
            adx: ADX value at entry
            reason: Why SL was set ('recovery_disabled', 'adx_trending')
            strategy_type: 'mean_reversion' or 'breakout'
        """
        self.log_trailing_event(
            event_type='hard_sl_set',
            ticket=ticket,
            symbol=symbol,
            entry_price=float(entry_price),
            sl_price=float(sl_price),
            sl_pips=float(sl_pips),
            adx=float(adx),
            reason=reason,
            strategy_type=strategy_type
        )

    def log_m15_entry_block(self, symbol: str, direction: str, price: float,
                             consecutive_bullish: int, consecutive_bearish: int,
                             confluence_score: int, signal_factors: list):
        """
        Log when M15 trend detection blocks initial trade entry.

        This allows ML to analyze:
        - How often M15 blocks entries
        - What would have happened if we entered anyway (counterfactual)
        - Whether the 3-candle threshold is optimal

        Args:
            symbol: Trading symbol
            direction: Intended trade direction ('buy' or 'sell')
            price: Entry price that was blocked
            consecutive_bullish: Number of consecutive bullish M15 candles
            consecutive_bearish: Number of consecutive bearish M15 candles
            confluence_score: Confluence score of the blocked signal
            signal_factors: List of confluence factors that triggered signal
        """
        m15_blocks_log = self.output_dir / "m15_entry_blocks.jsonl"

        block_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'price': float(price),
            'consecutive_bullish': int(consecutive_bullish),
            'consecutive_bearish': int(consecutive_bearish),
            'blocked_by': 'bearish_trend' if direction == 'buy' else 'bullish_trend',
            'confluence_score': int(confluence_score),
            'signal_factors': signal_factors,
            'event_type': 'm15_entry_block'
        }

        # Convert numpy types
        block_record = convert_numpy_types(block_record)

        try:
            with open(m15_blocks_log, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(json.dumps(block_record) + '\n')
        except Exception as e:
            print(f"[WARN] [LOGGER] Failed to log M15 entry block for {symbol}: {e}")

    def log_m15_recovery_block(self, ticket: int, symbol: str, recovery_type: str,
                                consecutive_bullish: int, consecutive_bearish: int,
                                pips_underwater: float, position_type: str,
                                blocked_by: str):
        """
        Log when M15 trend detection blocks DCA or hedge addition.

        This allows ML to analyze:
        - How often M15 blocks recovery
        - Whether blocks correlate with better outcomes (avoiding -$500 losses)
        - Whether the 3-candle threshold is optimal for recovery

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            recovery_type: 'dca' or 'hedge'
            consecutive_bullish: Number of consecutive bullish M15 candles
            consecutive_bearish: Number of consecutive bearish M15 candles
            pips_underwater: How many pips underwater position is
            position_type: 'buy' or 'sell'
            blocked_by: Reason for block ('m15_consecutive', 'm15_atr', 'h1_adx')
        """
        m15_recovery_blocks_log = self.output_dir / "m15_recovery_blocks.jsonl"

        block_record = {
            'timestamp': datetime.now().isoformat(),
            'ticket': int(ticket),
            'symbol': symbol,
            'recovery_type': recovery_type,
            'consecutive_bullish': int(consecutive_bullish),
            'consecutive_bearish': int(consecutive_bearish),
            'pips_underwater': float(pips_underwater),
            'position_type': position_type,
            'blocked_by': blocked_by,
            'event_type': 'm15_recovery_block'
        }

        # Convert numpy types
        block_record = convert_numpy_types(block_record)

        try:
            with open(m15_recovery_blocks_log, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(json.dumps(block_record) + '\n')
        except Exception as e:
            print(f"[WARN] [LOGGER] Failed to log M15 recovery block for {ticket}: {e}")

    def log_hedge_dca(self, original_ticket: int, hedge_ticket: int, symbol: str,
                      hedge_dca_ticket: int, level: int, volume: float,
                      hedge_pips_underwater: float, original_pips: float,
                      hedge_type: str, dca_type: str, m15_confirmed: bool):
        """
        Log when hedge DCA is added (positions in ORIGINAL direction to help recovery).

        This tracks the new hedge intelligence feature where DCAs are added to
        underwater hedges (in the original direction) to accelerate recovery.

        Args:
            original_ticket: Original position ticket
            hedge_ticket: Hedge position ticket
            symbol: Trading symbol
            hedge_dca_ticket: New hedge DCA position ticket
            level: DCA level (1, 2, etc.)
            volume: DCA volume in lots
            hedge_pips_underwater: How far hedge is underwater (means original recovering)
            original_pips: Original position pips (negative = underwater, positive = profitable)
            hedge_type: Hedge direction ('buy' or 'sell')
            dca_type: DCA direction (ORIGINAL direction, opposite of hedge)
            m15_confirmed: Whether M15 trend confirmed (3 consecutive candles)
        """
        hedge_intelligence_log = self.output_dir / "hedge_intelligence.jsonl"

        event_record = {
            'event_type': 'hedge_dca',
            'timestamp': datetime.now().isoformat(),
            'original_ticket': int(original_ticket),
            'hedge_ticket': int(hedge_ticket),
            'hedge_dca_ticket': int(hedge_dca_ticket),
            'symbol': symbol,
            'level': int(level),
            'volume': float(volume),
            'hedge_pips_underwater': float(hedge_pips_underwater),
            'original_pips': float(original_pips),
            'hedge_type': hedge_type,
            'dca_type': dca_type,  # ORIGINAL direction (opposite of hedge)
            'm15_confirmed': bool(m15_confirmed)
        }

        # Convert numpy types
        event_record = convert_numpy_types(event_record)

        try:
            with open(hedge_intelligence_log, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(json.dumps(event_record) + '\n')
        except Exception as e:
            print(f"[WARN] [LOGGER] Failed to log hedge DCA for {hedge_ticket}: {e}")

    def log_hedge_partial_close(self, original_ticket: int, hedge_ticket: int, symbol: str,
                                 close_percent: float, reason: str, original_pips: float,
                                 hedge_profit: float, positions_closed: int,
                                 hedge_dca_count: int):
        """
        Log when hedge + hedge DCAs are partially/fully closed due to original recovery.

        This tracks the intelligent hedge close feature where hedges are closed in stages
        (50%/75%/100%) as the original position recovers.

        Args:
            original_ticket: Original position ticket
            hedge_ticket: Hedge position ticket
            symbol: Trading symbol
            close_percent: Percentage closed (0.5 = 50%, 0.75 = 75%, 1.0 = 100%)
            reason: Why closing (e.g., "Original recovered 50%")
            original_pips: Original position pips at close time
            hedge_profit: Hedge profit/loss at close time
            positions_closed: Number of positions closed (hedge + hedge DCAs)
            hedge_dca_count: Number of hedge DCAs that were closed
        """
        hedge_intelligence_log = self.output_dir / "hedge_intelligence.jsonl"

        event_record = {
            'event_type': 'hedge_partial_close',
            'timestamp': datetime.now().isoformat(),
            'original_ticket': int(original_ticket),
            'hedge_ticket': int(hedge_ticket),
            'symbol': symbol,
            'close_percent': float(close_percent),
            'reason': reason,
            'original_pips': float(original_pips),
            'hedge_profit': float(hedge_profit),
            'positions_closed': int(positions_closed),
            'hedge_dca_count': int(hedge_dca_count)
        }

        # Convert numpy types
        event_record = convert_numpy_types(event_record)

        try:
            with open(hedge_intelligence_log, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(json.dumps(event_record) + '\n')
        except Exception as e:
            print(f"[WARN] [LOGGER] Failed to log hedge partial close for {hedge_ticket}: {e}")

    def log_recovery_trigger(self, recovery_type: str, original_ticket: int, symbol: str,
                            pips_underwater: float, time_since_entry_minutes: float,
                            current_adx: float = None, recovery_level: int = 1,
                            volume: float = 0.0, trigger_threshold: float = 0.0):
        """
        Log when recovery (DCA/hedge/grid) is triggered with all relevant conditions.
        Enables analysis of optimal trigger points and market conditions.

        Args:
            recovery_type: 'dca', 'hedge', 'grid', or 'hedge_dca'
            original_ticket: Original position ticket
            symbol: Trading symbol
            pips_underwater: How many pips underwater (negative = losing)
            time_since_entry_minutes: Minutes since position opened
            current_adx: Current ADX value (if available)
            recovery_level: Level number (DCA L1, L2, Grid L1, etc.)
            volume: Volume of recovery order
            trigger_threshold: Configured trigger threshold (e.g., 30 pips for DCA)
        """
        recovery_triggers_log = self.output_dir / "recovery_triggers.jsonl"

        event_record = {
            'event_type': 'recovery_trigger',
            'timestamp': datetime.now().isoformat(),
            'recovery_type': recovery_type,
            'original_ticket': int(original_ticket),
            'symbol': symbol,
            'pips_underwater': float(pips_underwater),
            'time_since_entry_minutes': float(time_since_entry_minutes),
            'current_adx': float(current_adx) if current_adx is not None else None,
            'recovery_level': int(recovery_level),
            'volume': float(volume),
            'trigger_threshold': float(trigger_threshold)
        }

        # Convert numpy types
        event_record = convert_numpy_types(event_record)

        try:
            with open(recovery_triggers_log, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(json.dumps(event_record) + '\n')
        except Exception as e:
            print(f"[WARN] [LOGGER] Failed to log recovery trigger for {original_ticket}: {e}")

    def run(self, mt5_login: int, mt5_password: str, mt5_server: str, check_interval: int = 60):
        """
        Run continuous logging service

        Args:
            mt5_login: MT5 account login
            mt5_password: MT5 password
            mt5_server: MT5 server
            check_interval: Seconds between checks (default 60)
        """
        print("="*80)
        print("CONTINUOUS ML TRADE LOGGER")
        print("="*80)
        print(f"Check interval: {check_interval} seconds")
        print(f"Output: {self.continuous_log}")
        print()

        # Connect to MT5
        if not self.connect_mt5(mt5_login, mt5_password, mt5_server):
            print("[ERROR] Failed to connect to MT5")
            return

        print("[OK] Connected to MT5")
        print(f"[OK] Already logged: {len(self.logged_tickets)} trades")
        print("[INFO] Tracking: Entry confluence + Recovery (DCA/Hedge) + Grid + Partial closes")
        print()

        # Backfill any missed trades while logger was offline
        print("[INFO] Checking for missed trades...")
        self.backfill_missed_trades()
        print()

        print("Starting continuous monitoring... (Press Ctrl+C to stop)")
        print()

        try:
            iteration = 0
            while True:
                iteration += 1

                # Check for new entry trades
                self.check_for_new_trades()

                # Check for closed trades and update with recovery/grid/partial data
                # Update every cycle to catch exits quickly
                self.update_closed_trades()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n[STOP] Continuous logger stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Logger crashed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Continuous ML Trade Logger')
    parser.add_argument('--login', type=int, required=True, help='MT5 login')
    parser.add_argument('--password', type=str, required=True, help='MT5 password')
    parser.add_argument('--server', type=str, required=True, help='MT5 server')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')

    args = parser.parse_args()

    logger = ContinuousMLLogger()
    logger.run(
        mt5_login=args.login,
        mt5_password=args.password,
        mt5_server=args.server,
        check_interval=args.interval
    )


if __name__ == '__main__':
    main()
