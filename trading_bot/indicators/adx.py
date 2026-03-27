"""
Average Directional Index (ADX) - Trend Strength Indicator
Used to determine if market is trending or ranging
"""

import pandas as pd
import numpy as np


def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index)

    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        period: Period for ADX calculation (default 14)

    Returns:
        DataFrame with ADX, +DI, -DI columns added
    """
    df = data.copy()

    # Calculate True Range
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    # Positive and Negative Directional Movement
    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )

    # Smooth the values using Wilder's smoothing (exponential moving average)
    alpha = 1.0 / period

    df['atr'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=alpha, adjust=False).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=alpha, adjust=False).mean() / df['atr'])

    # Calculate DX (Directional Index)
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])

    # Calculate ADX (smoothed DX)
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()

    # Clean up intermediate columns
    df.drop(['high_low', 'high_close', 'low_close', 'tr', 'up_move', 'down_move',
             'plus_dm', 'minus_dm', 'dx'], axis=1, inplace=True)

    return df


def interpret_adx(adx_value: float, plus_di: float, minus_di: float) -> dict:
    """
    Interpret ADX reading and trend direction

    Args:
        adx_value: ADX value
        plus_di: +DI value
        minus_di: -DI value

    Returns:
        Dict with interpretation
    """
    # Determine trend strength
    if adx_value < 20:
        strength = "weak"
        market_type = "ranging"
    elif adx_value < 25:
        strength = "developing"
        market_type = "weak_trend"
    elif adx_value < 40:
        strength = "moderate"
        market_type = "trending"
    elif adx_value < 50:
        strength = "strong"
        market_type = "strong_trend"
    else:
        strength = "very_strong"
        market_type = "very_strong_trend"

    # Determine trend direction
    if plus_di > minus_di:
        direction = "bullish"
    else:
        direction = "bearish"

    # Confidence (higher when DI lines are far apart)
    confidence = abs(plus_di - minus_di)

    return {
        'adx': adx_value,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'strength': strength,
        'market_type': market_type,
        'direction': direction,
        'confidence': confidence,
        'is_ranging': adx_value < 25,
        'is_trending': adx_value >= 25
    }


def analyze_candle_direction(data: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Analyze recent candle direction to confirm trend

    Args:
        data: DataFrame with OHLC data
        lookback: Number of candles to look back

    Returns:
        Dict with candle analysis
    """
    recent = data.tail(lookback)

    # Count bullish vs bearish candles
    bullish_candles = (recent['close'] > recent['open']).sum()
    bearish_candles = (recent['close'] < recent['open']).sum()

    # Calculate average body size
    body_sizes = np.abs(recent['close'] - recent['open'])
    avg_body = body_sizes.mean()

    # Calculate percentage of aligned candles
    total_candles = len(recent)
    bullish_pct = (bullish_candles / total_candles) * 100
    bearish_pct = (bearish_candles / total_candles) * 100

    # Determine if candles are aligned (mostly same direction)
    alignment_threshold = 70  # 70% of candles in same direction

    if bullish_pct >= alignment_threshold:
        alignment = "strong_bullish"
        aligned = True
        direction = "bullish"
    elif bearish_pct >= alignment_threshold:
        alignment = "strong_bearish"
        aligned = True
        direction = "bearish"
    elif bullish_pct >= 60:
        alignment = "weak_bullish"
        aligned = False
        direction = "bullish"
    elif bearish_pct >= 60:
        alignment = "weak_bearish"
        aligned = False
        direction = "bearish"
    else:
        alignment = "mixed"
        aligned = False
        direction = "neutral"

    return {
        'lookback': lookback,
        'bullish_candles': bullish_candles,
        'bearish_candles': bearish_candles,
        'bullish_pct': bullish_pct,
        'bearish_pct': bearish_pct,
        'alignment': alignment,
        'aligned': aligned,
        'direction': direction,
        'avg_body': avg_body
    }


def analyze_candle_momentum(
    data: pd.DataFrame,
    direction: str,
    lookback: int = 4,
    shrink_ratio: float = 0.85
) -> dict:
    """
    Measure whether recent H1 candle bodies are GROWING or SHRINKING.

    Used by both strategies with opposite meaning:
      - BO: Shrinking bodies in BO direction = momentum fading = BLOCK (late entry)
      - MR: Shrinking bodies against MR direction = trend exhausting = BOOST (reversion)

    Analyses absolute body sizes of the last N closed candles (skipping the
    current forming candle at index -1). A pair "shrinks" when the later body
    is smaller than the earlier body times shrink_ratio.

    Args:
        data: DataFrame with 'open' and 'close' columns (H1 candle data)
        direction: Trade direction to analyse ('buy' or 'sell')
        lookback: Number of closed candles to analyse (default 4)
        shrink_ratio: Threshold — body[i+1] < body[i] * ratio = shrinking (default 0.85)

    Returns:
        Dict with:
            body_sizes: list of absolute body sizes [lookback floats]
            shrinking_pairs: count of consecutive pairs where body decreased
            growing_pairs: count of consecutive pairs where body increased
            total_pairs: total pairs compared (lookback - 1)
            momentum: 'fading' | 'building' | 'weakening' | 'strengthening' | 'mixed'
            is_fading: True if shrinking_pairs >= total_pairs - 1
            direction_analyzed: the direction that was checked
    """
    # Need at least lookback + 1 rows (lookback closed candles + current forming)
    if len(data) < lookback + 1:
        return {
            'body_sizes': [],
            'shrinking_pairs': 0,
            'growing_pairs': 0,
            'total_pairs': 0,
            'momentum': 'insufficient_data',
            'is_fading': False,
            'direction_analyzed': direction
        }

    # Get closed candles only (exclude current forming candle at -1)
    # For lookback=4: iloc[-5], iloc[-4], iloc[-3], iloc[-2]
    closed = data.iloc[-(lookback + 1):-1]

    # Calculate absolute body sizes
    body_sizes = np.abs(closed['close'].values - closed['open'].values)

    # Count shrinking vs growing consecutive pairs
    shrinking_pairs = 0
    growing_pairs = 0
    total_pairs = len(body_sizes) - 1

    for i in range(total_pairs):
        prev_body = body_sizes[i]
        curr_body = body_sizes[i + 1]

        if prev_body > 0 and curr_body < prev_body * shrink_ratio:
            shrinking_pairs += 1
        elif curr_body > prev_body:
            growing_pairs += 1

    # Classify momentum state
    if shrinking_pairs >= total_pairs:
        momentum = 'fading'
    elif growing_pairs >= total_pairs:
        momentum = 'building'
    elif shrinking_pairs > growing_pairs:
        momentum = 'weakening'
    elif growing_pairs > shrinking_pairs:
        momentum = 'strengthening'
    else:
        momentum = 'mixed'

    is_fading = shrinking_pairs >= max(total_pairs - 1, 1)

    return {
        'body_sizes': body_sizes.tolist(),
        'shrinking_pairs': int(shrinking_pairs),
        'growing_pairs': int(growing_pairs),
        'total_pairs': int(total_pairs),
        'momentum': momentum,
        'is_fading': is_fading,
        'direction_analyzed': direction
    }


def should_trade_based_on_trend(
    adx_value: float,
    plus_di: float,
    minus_di: float,
    candle_data: pd.DataFrame,
    candle_lookback: int = 5,
    adx_threshold: float = 25,
    allow_weak_trends: bool = True,
    max_adx: float = 40.0
) -> tuple[bool, str]:
    """
    Determine if we should trade based on trend analysis

    Args:
        adx_value: Current ADX value
        plus_di: Current +DI value
        minus_di: Current -DI value
        candle_data: Recent candle data
        candle_lookback: Number of candles to analyze
        adx_threshold: ADX threshold for "trending" market
        allow_weak_trends: Allow trading in weak trends (ADX 20-25)
        max_adx: Maximum ADX for MR trading (per-symbol, ML-tunable)

    Returns:
        Tuple of (should_trade, reason)
    """
    # Get ADX interpretation
    adx_info = interpret_adx(adx_value, plus_di, minus_di)

    # Get candle alignment
    candle_info = analyze_candle_direction(candle_data, candle_lookback)

    # Rule 1: Strong trend (ADX > max_adx) = NO TRADE
    if adx_value > max_adx:
        return False, f"Strong trend detected (ADX: {adx_value:.1f}) - Mean reversion unsafe"

    # Rule 2: Moderate trend (ADX 25-40) + aligned candles = NO TRADE
    if adx_value >= adx_threshold and candle_info['aligned']:
        return False, f"Trending market (ADX: {adx_value:.1f}) + {candle_info['alignment']} candles - Mean reversion risky"

    # Rule 3: Weak trend (ADX 20-25) = TRADE if candles not strongly aligned
    if 20 <= adx_value < adx_threshold:
        if candle_info['aligned']:
            return False, f"Weak trend (ADX: {adx_value:.1f}) with aligned candles - Proceed with caution"
        else:
            if allow_weak_trends:
                return True, f"Weak trend (ADX: {adx_value:.1f}) + mixed candles - OK to trade"
            else:
                return False, f"Weak trend detected (ADX: {adx_value:.1f}) - Trading disabled"

    # Rule 4: Ranging market (ADX < 20) = TRADE
    if adx_value < 20:
        return True, f"Ranging market (ADX: {adx_value:.1f}) - Ideal for mean reversion"

    # Rule 5: Moderate trend but candles NOT aligned = TRADE (trend may be weakening)
    if adx_value >= adx_threshold and not candle_info['aligned']:
        return True, f"Trend (ADX: {adx_value:.1f}) but mixed candles - Possible reversal"

    # Default: Allow trade
    return True, "Trend analysis passed"
