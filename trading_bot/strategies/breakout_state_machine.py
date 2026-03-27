#!/usr/bin/env python3
"""
SMC Structural Breakout State Machine

Tracks the 6-stage breakout sequence per symbol:
  0: IDLE           - Scanning for compression
  1: COMPRESSION    - Tight range / consolidation identified
  2: LIQUIDITY_BUILT - EQH/EQL cluster found near range boundary
  3: SWEEP          - False break / liquidity sweep detected
  4: BOS_CONFIRMED  - Real structural break (BOS) confirmed
  5: RETEST_ENTRY   - Price retests BOS level or fills FVG -> ENTRY ZONE
  6: EXPANSION      - Trade active, managing expansion leg

Entry only at stage 5 after full structural confirmation.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# SMC library (optional — graceful fallback)
try:
    import io, sys
    # Suppress the library's unicode-heavy startup message on Windows
    _orig_stdout = sys.stdout
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
    from smartmoneyconcepts import smc as smc_lib
    sys.stdout = _orig_stdout
    SMC_AVAILABLE = True
except (ImportError, Exception):
    SMC_AVAILABLE = False
    try:
        sys.stdout = _orig_stdout
    except Exception:
        pass
    print("[SMC_BO] smartmoneyconcepts not available — using simple BOS fallback")

from config.strategy_config import (
    SMC_BO_COMPRESSION_ATR_PERCENTILE,
    SMC_BO_COMPRESSION_MAX_RANGE_PIPS,
    SMC_BO_COMPRESSION_MIN_BARS,
    SMC_BO_COMPRESSION_LOOKBACK,
    SMC_BO_COMPRESSION_ADX_MAX,
    SMC_BO_EQL_EQH_TOLERANCE_PIPS,
    SMC_BO_EQL_EQH_MIN_TOUCHES,
    SMC_BO_LIQUIDITY_TIMEOUT_BARS,
    SMC_BO_SWEEP_MIN_PIPS,
    SMC_BO_SWEEP_MAX_PIPS,
    SMC_BO_SWEEP_REVERSAL_BARS,
    SMC_BO_SWEEP_TIMEOUT_BARS,
    SMC_BO_BOS_SWING_LENGTH,
    SMC_BO_BOS_REQUIRE_CANDLE_CLOSE,
    SMC_BO_BOS_TIMEOUT_BARS,
    SMC_BO_RETEST_TOLERANCE_PCT,
    SMC_BO_RETEST_TIMEOUT_BARS,
    SMC_BO_FVG_ENTRY_ENABLED,
    SMC_BO_ENTRY_ADX_MIN,
    SMC_BO_ENTRY_ADX_MAX,
    SMC_BO_MAX_SEQUENCE_BARS,
    SMC_BO_INVALIDATION_ATR_MULTIPLE,
    SMC_BO_INVALIDATION_ADX_MAX,
)

# Stage names for logging
STAGE_NAMES = {
    0: 'IDLE',
    1: 'COMPRESSION',
    2: 'LIQUIDITY_BUILT',
    3: 'SWEEP',
    4: 'BOS_CONFIRMED',
    5: 'RETEST_ENTRY',
    6: 'EXPANSION',
}

# Log file
LOG_DIR = Path(__file__).parent.parent.parent / 'ml_system' / 'outputs'


@dataclass
class BreakoutSequenceState:
    """Tracks the state of a structural breakout sequence for one symbol."""
    symbol: str
    stage: int = 0
    direction: Optional[str] = None  # 'buy' or 'sell' — set at stage 3-4

    # Stage 1: Compression
    range_high: float = 0.0
    range_low: float = 0.0
    range_size_pips: float = 0.0
    compression_bar_count: int = 0
    compression_detected_bar: int = 0

    # Stage 2: Liquidity
    liquidity_side: str = ''  # 'above', 'below', 'both'
    eqh_level: Optional[float] = None
    eql_level: Optional[float] = None
    liquidity_detected_bar: int = 0

    # Stage 3: Sweep
    sweep_direction: str = ''  # 'up' (took out highs) or 'down' (took out lows)
    sweep_extreme: float = 0.0
    sweep_bar: int = 0
    sweep_reversal_count: int = 0  # Bars since sweep started

    # Stage 4: BOS
    bos_type: str = ''  # 'bullish' or 'bearish'
    bos_level: float = 0.0
    bos_bar: int = 0

    # Stage 5: Retest
    retest_level: float = 0.0
    retest_type: str = ''  # 'level_retest' or 'fvg_fill'
    fvg_zone: Optional[Tuple[float, float]] = None
    retest_bar: int = 0

    # Timing
    stage_entry_bar: int = 0
    total_bars_in_sequence: int = 0
    last_update_bar: int = 0

    # ADX/ATR at compression (for invalidation reference)
    compression_atr: float = 0.0

    def bars_in_current_stage(self, current_bar: int) -> int:
        return current_bar - self.stage_entry_bar

    def to_dict(self) -> Dict:
        """Convert to JSON-safe dict."""
        d = asdict(self)
        # Convert tuple to list for JSON
        if d.get('fvg_zone') is not None:
            d['fvg_zone'] = list(d['fvg_zone'])
        return d


class BreakoutStateMachine:
    """
    Per-symbol state machine for structural breakout detection.

    Call update() every bar. When stage reaches 5 (RETEST_ENTRY),
    get_entry_signal() returns a trade signal.
    """

    def __init__(self):
        self.states: Dict[str, BreakoutSequenceState] = {}
        self._log_path = LOG_DIR / 'smc_breakout_stages.jsonl'
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _get_state(self, symbol: str) -> BreakoutSequenceState:
        if symbol not in self.states:
            self.states[symbol] = BreakoutSequenceState(symbol=symbol)
        return self.states[symbol]

    def update(
        self,
        symbol: str,
        h1_data: pd.DataFrame,
        d1_data: Optional[pd.DataFrame],
        current_bar_index: int,
    ) -> None:
        """
        Main per-bar update. Advances the state machine for one symbol.
        Should be called every bar, even outside the breakout trading window.
        """
        state = self._get_state(symbol)

        if len(h1_data) < SMC_BO_COMPRESSION_LOOKBACK + 10:
            return

        # Skip if already processed this bar
        if state.last_update_bar >= current_bar_index and current_bar_index > 0:
            return
        state.last_update_bar = current_bar_index

        # Track total bars in sequence
        if state.stage > 0:
            state.total_bars_in_sequence += 1

        # Check invalidation first (any stage -> 0)
        if state.stage > 0 and self._check_invalidation(state, h1_data, current_bar_index):
            return

        # Advance through stages
        if state.stage == 0:
            self._check_compression(state, h1_data, current_bar_index)
        elif state.stage == 1:
            self._check_liquidity_build(state, h1_data, current_bar_index)
        elif state.stage == 2:
            self._check_sweep(state, h1_data, current_bar_index)
        elif state.stage == 3:
            self._check_bos(state, h1_data, current_bar_index)
        elif state.stage == 4:
            self._check_retest_entry(state, h1_data, d1_data, current_bar_index)
        # Stage 5: waiting for get_entry_signal() to be called
        # Stage 6: trade is active (managed externally)

    def get_entry_signal(self, symbol: str) -> Optional[Dict]:
        """
        Returns an entry signal if the state machine is at stage 5 (RETEST_ENTRY).
        Returns None otherwise.
        """
        state = self._get_state(symbol)
        if state.stage != 5:
            return None

        return {
            'type': 'structural_breakout',
            'direction': state.direction,
            'entry_price': state.retest_level,
            'bos_level': state.bos_level,
            'range_high': state.range_high,
            'range_low': state.range_low,
            'range_size_pips': state.range_size_pips,
            'sweep_direction': state.sweep_direction,
            'sweep_extreme': state.sweep_extreme,
            'retest_type': state.retest_type,
            'fvg_zone': state.fvg_zone,
            'total_bars': state.total_bars_in_sequence,
            'stage': 5,
            'confluence_score': 5,  # Base score for structural BO
            'factors': self._build_factors_list(state),
        }

    def advance_to_expansion(self, symbol: str) -> None:
        """Move to stage 6 (EXPANSION) after trade execution."""
        state = self._get_state(symbol)
        if state.stage == 5:
            self._transition(state, 6, "Trade executed, entering expansion")

    def reset(self, symbol: str) -> None:
        """Reset to IDLE."""
        self.states[symbol] = BreakoutSequenceState(symbol=symbol)

    def get_state_summary(self, symbol: str) -> Dict:
        """Get a loggable summary of the current state."""
        state = self._get_state(symbol)
        return {
            'symbol': symbol,
            'stage': state.stage,
            'stage_name': STAGE_NAMES.get(state.stage, '?'),
            'direction': state.direction,
            'total_bars': state.total_bars_in_sequence,
            'range': f"{state.range_low:.5f}-{state.range_high:.5f}" if state.range_high > 0 else None,
        }

    # ── Stage Checks ──────────────────────────────────────────────────

    def _check_compression(
        self,
        state: BreakoutSequenceState,
        h1_data: pd.DataFrame,
        bar_index: int,
    ) -> None:
        """Stage 0 -> 1: Detect compression / tight range."""
        lookback = SMC_BO_COMPRESSION_LOOKBACK
        if len(h1_data) < lookback:
            return

        recent = h1_data.tail(lookback)
        range_high = float(recent['high'].max())
        range_low = float(recent['low'].min())
        range_pips = (range_high - range_low) * 10_000

        # Check 1: Range must be tight enough
        if range_pips > SMC_BO_COMPRESSION_MAX_RANGE_PIPS:
            return

        # Check 2: ATR must be compressed
        if 'atr' in h1_data.columns:
            current_atr = float(h1_data['atr'].iloc[-1])
            # Compute ATR percentile over last 200 bars
            atr_window = h1_data['atr'].tail(min(200, len(h1_data)))
            atr_percentile = (atr_window < current_atr).sum() / len(atr_window)
            if atr_percentile > SMC_BO_COMPRESSION_ATR_PERCENTILE:
                return
        else:
            # Compute ATR manually if column not available
            current_atr = self._compute_atr(h1_data, 14)
            atr_percentile = 0.3  # Assume compressed if we can't check

        # Check 3: ADX should be low (ranging market)
        adx = self._get_current_adx(h1_data)
        if adx > SMC_BO_COMPRESSION_ADX_MAX:
            return

        # Check 4: Minimum bars contained within range
        # Count how many of the last N bars have their body within the range
        bars_in_range = 0
        for i in range(len(recent)):
            bar = recent.iloc[i]
            body_high = max(bar['open'], bar['close'])
            body_low = min(bar['open'], bar['close'])
            # Bar body is within range (with small tolerance)
            tolerance = (range_high - range_low) * 0.1
            if body_high <= range_high + tolerance and body_low >= range_low - tolerance:
                bars_in_range += 1

        if bars_in_range < SMC_BO_COMPRESSION_MIN_BARS:
            return

        # Compression detected!
        state.range_high = range_high
        state.range_low = range_low
        state.range_size_pips = range_pips
        state.compression_bar_count = bars_in_range
        state.compression_detected_bar = bar_index
        state.compression_atr = current_atr
        self._transition(state, 1,
                         f"Range {range_low:.5f}-{range_high:.5f} ({range_pips:.0f} pips), "
                         f"ADX={adx:.1f}, ATR pctl={atr_percentile:.0%}, "
                         f"{bars_in_range} bars in range")

    def _check_liquidity_build(
        self,
        state: BreakoutSequenceState,
        h1_data: pd.DataFrame,
        bar_index: int,
    ) -> None:
        """Stage 1 -> 2: Detect EQH/EQL cluster near range boundary."""
        # Timeout check
        if state.bars_in_current_stage(bar_index) > SMC_BO_LIQUIDITY_TIMEOUT_BARS:
            self._transition(state, 0, "Liquidity build timeout")
            return

        # Use LiquidityLevels to find EQH/EQL
        from indicators.liquidity_levels import LiquidityLevels
        liq = LiquidityLevels()
        eq_levels = liq.detect_equal_levels(
            h1_data,
            lookback=SMC_BO_COMPRESSION_LOOKBACK + 10,
            tolerance_pips=SMC_BO_EQL_EQH_TOLERANCE_PIPS,
        )

        eqh_near_range = None
        eql_near_range = None
        range_tolerance = SMC_BO_EQL_EQH_TOLERANCE_PIPS * 0.0001

        # Check for EQH near range high
        for eqh_price, touch_count, _ in eq_levels.get('equal_highs', []):
            if touch_count >= SMC_BO_EQL_EQH_MIN_TOUCHES:
                if abs(eqh_price - state.range_high) <= range_tolerance:
                    eqh_near_range = eqh_price

        # Check for EQL near range low
        for eql_price, touch_count, _ in eq_levels.get('equal_lows', []):
            if touch_count >= SMC_BO_EQL_EQH_MIN_TOUCHES:
                if abs(eql_price - state.range_low) <= range_tolerance:
                    eql_near_range = eql_price

        if eqh_near_range is None and eql_near_range is None:
            return

        # Liquidity build detected
        state.eqh_level = eqh_near_range
        state.eql_level = eql_near_range
        state.liquidity_detected_bar = bar_index

        if eqh_near_range and eql_near_range:
            state.liquidity_side = 'both'
        elif eqh_near_range:
            state.liquidity_side = 'above'
        else:
            state.liquidity_side = 'below'

        self._transition(state, 2,
                         f"Liquidity {state.liquidity_side}: "
                         f"EQH={eqh_near_range}, EQL={eql_near_range}")

    def _check_sweep(
        self,
        state: BreakoutSequenceState,
        h1_data: pd.DataFrame,
        bar_index: int,
    ) -> None:
        """Stage 2 -> 3: Detect false break / liquidity sweep."""
        # Timeout check
        if state.bars_in_current_stage(bar_index) > SMC_BO_SWEEP_TIMEOUT_BARS:
            self._transition(state, 0, "Sweep timeout")
            return

        current_bar = h1_data.iloc[-1]
        current_high = float(current_bar['high'])
        current_low = float(current_bar['low'])
        current_close = float(current_bar['close'])

        sweep_min = SMC_BO_SWEEP_MIN_PIPS * 0.0001
        sweep_max = SMC_BO_SWEEP_MAX_PIPS * 0.0001

        # Check for upward sweep (took out EQH above range)
        if state.eqh_level is not None:
            pierce_above = current_high - state.eqh_level
            if sweep_min <= pierce_above <= sweep_max:
                # Price pierced above EQH but needs to close back inside range
                if current_close <= state.range_high:
                    state.sweep_direction = 'up'
                    state.sweep_extreme = current_high
                    state.sweep_bar = bar_index
                    state.sweep_reversal_count = 0
                    # Sweep up then reverse = sell setup
                    state.direction = 'sell'
                    self._transition(state, 3,
                                     f"Sweep UP: pierced EQH {state.eqh_level:.5f} "
                                     f"by {pierce_above*10000:.1f} pips, "
                                     f"closed back at {current_close:.5f}")
                    return
                elif pierce_above > sweep_max:
                    # Too far — real breakout, not a sweep
                    self._transition(state, 0, f"Break above EQH too far ({pierce_above*10000:.0f} pips) — real breakout")
                    return

        # Check for downward sweep (took out EQL below range)
        if state.eql_level is not None:
            pierce_below = state.eql_level - current_low
            if sweep_min <= pierce_below <= sweep_max:
                if current_close >= state.range_low:
                    state.sweep_direction = 'down'
                    state.sweep_extreme = current_low
                    state.sweep_bar = bar_index
                    state.sweep_reversal_count = 0
                    # Sweep down then reverse = buy setup
                    state.direction = 'buy'
                    self._transition(state, 3,
                                     f"Sweep DOWN: pierced EQL {state.eql_level:.5f} "
                                     f"by {pierce_below*10000:.1f} pips, "
                                     f"closed back at {current_close:.5f}")
                    return
                elif pierce_below > sweep_max:
                    self._transition(state, 0, f"Break below EQL too far ({pierce_below*10000:.0f} pips) — real breakout")
                    return

        # Also check if a sweep happened in the last few bars (multi-bar sweep)
        for lookback_i in range(1, min(SMC_BO_SWEEP_REVERSAL_BARS + 1, len(h1_data))):
            past_bar = h1_data.iloc[-(lookback_i + 1)]
            past_high = float(past_bar['high'])
            past_low = float(past_bar['low'])

            # Upward sweep in past bar, now reversing
            if state.eqh_level and past_high > state.eqh_level + sweep_min:
                pierce = past_high - state.eqh_level
                if pierce <= sweep_max and current_close <= state.range_high:
                    state.sweep_direction = 'up'
                    state.sweep_extreme = past_high
                    state.sweep_bar = bar_index - lookback_i
                    state.direction = 'sell'
                    self._transition(state, 3,
                                     f"Sweep UP (multi-bar): pierced EQH {lookback_i} bars ago, "
                                     f"now reversed to {current_close:.5f}")
                    return

            # Downward sweep in past bar
            if state.eql_level and past_low < state.eql_level - sweep_min:
                pierce = state.eql_level - past_low
                if pierce <= sweep_max and current_close >= state.range_low:
                    state.sweep_direction = 'down'
                    state.sweep_extreme = past_low
                    state.sweep_bar = bar_index - lookback_i
                    state.direction = 'buy'
                    self._transition(state, 3,
                                     f"Sweep DOWN (multi-bar): pierced EQL {lookback_i} bars ago, "
                                     f"now reversed to {current_close:.5f}")
                    return

    def _check_bos(
        self,
        state: BreakoutSequenceState,
        h1_data: pd.DataFrame,
        bar_index: int,
    ) -> None:
        """Stage 3 -> 4: Detect real Break of Structure (BOS) using SMC library."""
        # Timeout check
        if state.bars_in_current_stage(bar_index) > SMC_BO_BOS_TIMEOUT_BARS:
            self._transition(state, 0, "BOS timeout after sweep")
            return

        if not SMC_AVAILABLE:
            # Fallback: simple structure break detection without SMC library
            self._check_bos_simple(state, h1_data, bar_index)
            return

        # Use smartmoneyconcepts library
        lookback = min(200, len(h1_data))
        h1_slice = h1_data.tail(lookback).copy()
        ohlc = h1_slice[['open', 'high', 'low', 'close']].copy().reset_index(drop=True)

        try:
            swing_hl = smc_lib.swing_highs_lows(ohlc, swing_length=SMC_BO_BOS_SWING_LENGTH)
            bos_choch = smc_lib.bos_choch(ohlc, swing_hl)

            if 'BOS' not in bos_choch.columns:
                return

            # Find BOS events in the last few bars (since sweep)
            bars_since_sweep = bar_index - state.sweep_bar
            bos_events = bos_choch[bos_choch['BOS'].notna()]

            for idx in reversed(bos_events.index.tolist()):
                bars_ago = len(ohlc) - 1 - idx
                if bars_ago > bars_since_sweep + 2:
                    continue  # Too old — before the sweep

                bos_val = bos_events.loc[idx, 'BOS']
                bos_direction = 'bullish' if bos_val == 1 else 'bearish'

                # BOS must be opposite to sweep direction
                # Sweep up (took highs) -> expect bearish BOS (sell setup)
                # Sweep down (took lows) -> expect bullish BOS (buy setup)
                expected_bos = 'bearish' if state.sweep_direction == 'up' else 'bullish'

                if bos_direction != expected_bos:
                    continue

                # Optional: require candle close confirmation
                if SMC_BO_BOS_REQUIRE_CANDLE_CLOSE:
                    bos_candle = ohlc.iloc[idx]
                    if bos_direction == 'bullish' and bos_candle['close'] <= bos_candle['open']:
                        continue  # Bullish BOS should close bullish
                    if bos_direction == 'bearish' and bos_candle['close'] >= bos_candle['open']:
                        continue  # Bearish BOS should close bearish

                # Determine BOS level (the structure that was broken)
                # For bullish BOS: the swing high that was broken
                # For bearish BOS: the swing low that was broken
                if bos_direction == 'bullish':
                    # Find the nearest swing high before this BOS
                    swing_highs = swing_hl[swing_hl['HighLow'] == 1]
                    prior_swings = swing_highs[swing_highs.index < idx]
                    if len(prior_swings) > 0:
                        state.bos_level = float(ohlc.loc[prior_swings.index[-1], 'high'])
                    else:
                        state.bos_level = float(ohlc.iloc[idx]['high'])
                else:
                    swing_lows = swing_hl[swing_hl['HighLow'] == -1]
                    prior_swings = swing_lows[swing_lows.index < idx]
                    if len(prior_swings) > 0:
                        state.bos_level = float(ohlc.loc[prior_swings.index[-1], 'low'])
                    else:
                        state.bos_level = float(ohlc.iloc[idx]['low'])

                state.bos_type = bos_direction
                state.bos_bar = bar_index - bars_ago
                self._transition(state, 4,
                                 f"BOS {bos_direction} confirmed at level {state.bos_level:.5f}, "
                                 f"{bars_ago} bars ago")
                return

        except Exception as e:
            print(f"[SMC_BO] BOS detection error: {e}")

    def _check_bos_simple(
        self,
        state: BreakoutSequenceState,
        h1_data: pd.DataFrame,
        bar_index: int,
    ) -> None:
        """Fallback BOS detection without smartmoneyconcepts library."""
        current = h1_data.iloc[-1]
        current_close = float(current['close'])

        if state.direction == 'buy':
            # For buy setup (sweep down), BOS = price breaks above range high
            if current_close > state.range_high:
                state.bos_type = 'bullish'
                state.bos_level = state.range_high
                state.bos_bar = bar_index
                self._transition(state, 4,
                                 f"Simple BOS bullish: close {current_close:.5f} > "
                                 f"range high {state.range_high:.5f}")
        elif state.direction == 'sell':
            # For sell setup (sweep up), BOS = price breaks below range low
            if current_close < state.range_low:
                state.bos_type = 'bearish'
                state.bos_level = state.range_low
                state.bos_bar = bar_index
                self._transition(state, 4,
                                 f"Simple BOS bearish: close {current_close:.5f} < "
                                 f"range low {state.range_low:.5f}")

    def _check_retest_entry(
        self,
        state: BreakoutSequenceState,
        h1_data: pd.DataFrame,
        d1_data: Optional[pd.DataFrame],
        bar_index: int,
    ) -> None:
        """Stage 4 -> 5: Detect retest of BOS level or FVG fill."""
        # Timeout check
        if state.bars_in_current_stage(bar_index) > SMC_BO_RETEST_TIMEOUT_BARS:
            self._transition(state, 0, "Retest timeout after BOS")
            return

        current = h1_data.iloc[-1]
        current_close = float(current['close'])
        current_low = float(current['low'])
        current_high = float(current['high'])

        retest_tolerance = state.bos_level * SMC_BO_RETEST_TOLERANCE_PCT

        # Check 1: Level retest — price pulls back to BOS level
        if state.direction == 'buy':
            # Buy setup: BOS was bullish (broke above), retest = price dips back toward level
            if current_low <= state.bos_level + retest_tolerance:
                # Confirm: must bounce off (close above BOS level)
                if current_close > state.bos_level:
                    # ADX check at entry
                    adx = self._get_current_adx(h1_data)
                    if SMC_BO_ENTRY_ADX_MIN <= adx <= SMC_BO_ENTRY_ADX_MAX:
                        state.retest_level = current_close
                        state.retest_type = 'level_retest'
                        state.retest_bar = bar_index
                        self._transition(state, 5,
                                         f"Retest BUY: price dipped to {current_low:.5f}, "
                                         f"bounced to {current_close:.5f}, "
                                         f"BOS level={state.bos_level:.5f}, ADX={adx:.1f}")
                        return

        elif state.direction == 'sell':
            # Sell setup: BOS was bearish (broke below), retest = price rises back toward level
            if current_high >= state.bos_level - retest_tolerance:
                if current_close < state.bos_level:
                    adx = self._get_current_adx(h1_data)
                    if SMC_BO_ENTRY_ADX_MIN <= adx <= SMC_BO_ENTRY_ADX_MAX:
                        state.retest_level = current_close
                        state.retest_type = 'level_retest'
                        state.retest_bar = bar_index
                        self._transition(state, 5,
                                         f"Retest SELL: price rose to {current_high:.5f}, "
                                         f"rejected at {current_close:.5f}, "
                                         f"BOS level={state.bos_level:.5f}, ADX={adx:.1f}")
                        return

        # Check 2: FVG fill entry (if enabled)
        if SMC_BO_FVG_ENTRY_ENABLED and d1_data is not None and len(d1_data) >= 10:
            fvg = self._detect_fvg(h1_data, current_close)

            if state.direction == 'buy' and fvg.get('near_bullish_fvg'):
                adx = self._get_current_adx(h1_data)
                if SMC_BO_ENTRY_ADX_MIN <= adx <= SMC_BO_ENTRY_ADX_MAX:
                    state.retest_level = current_close
                    state.retest_type = 'fvg_fill'
                    state.retest_bar = bar_index
                    self._transition(state, 5,
                                     f"FVG fill BUY at {current_close:.5f}, ADX={adx:.1f}")
                    return

            elif state.direction == 'sell' and fvg.get('near_bearish_fvg'):
                adx = self._get_current_adx(h1_data)
                if SMC_BO_ENTRY_ADX_MIN <= adx <= SMC_BO_ENTRY_ADX_MAX:
                    state.retest_level = current_close
                    state.retest_type = 'fvg_fill'
                    state.retest_bar = bar_index
                    self._transition(state, 5,
                                     f"FVG fill SELL at {current_close:.5f}, ADX={adx:.1f}")
                    return

    # ── Invalidation ──────────────────────────────────────────────────

    def _check_invalidation(
        self,
        state: BreakoutSequenceState,
        h1_data: pd.DataFrame,
        bar_index: int,
    ) -> bool:
        """Check if the current sequence should be invalidated. Returns True if reset."""
        # 1. Total sequence too long
        if state.total_bars_in_sequence > SMC_BO_MAX_SEQUENCE_BARS:
            self._transition(state, 0,
                             f"Sequence timeout: {state.total_bars_in_sequence} bars")
            return True

        # 2. ADX too high during compression/liquidity stages
        if state.stage in (1, 2):
            adx = self._get_current_adx(h1_data)
            if adx > SMC_BO_INVALIDATION_ADX_MAX:
                self._transition(state, 0,
                                 f"ADX too high ({adx:.1f}) during compression")
                return True

        # 3. Price moved too far from range (wrong direction escape)
        if state.stage >= 1 and state.compression_atr > 0:
            current_close = float(h1_data.iloc[-1]['close'])
            max_deviation = state.compression_atr * SMC_BO_INVALIDATION_ATR_MULTIPLE

            if state.direction == 'buy' and current_close < state.range_low - max_deviation:
                self._transition(state, 0,
                                 f"Price collapsed below range: {current_close:.5f} vs "
                                 f"range_low {state.range_low:.5f}")
                return True
            elif state.direction == 'sell' and current_close > state.range_high + max_deviation:
                self._transition(state, 0,
                                 f"Price expanded above range: {current_close:.5f} vs "
                                 f"range_high {state.range_high:.5f}")
                return True

        return False

    # ── Helper Methods ────────────────────────────────────────────────

    def _transition(
        self,
        state: BreakoutSequenceState,
        to_stage: int,
        reason: str,
    ) -> None:
        """Transition to a new stage with logging."""
        from_stage = state.stage
        from_name = STAGE_NAMES.get(from_stage, '?')
        to_name = STAGE_NAMES.get(to_stage, '?')

        print(f"[SMC_BO] {state.symbol}: {from_name}({from_stage}) -> "
              f"{to_name}({to_stage}): {reason}")

        # Log to JSONL
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': state.symbol,
            'from_stage': from_stage,
            'from_name': from_name,
            'to_stage': to_stage,
            'to_name': to_name,
            'reason': reason,
            'state': state.to_dict(),
        }
        try:
            with open(self._log_path, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
        except Exception:
            pass  # Non-critical logging

        if to_stage == 0:
            # Reset — preserve symbol
            symbol = state.symbol
            self.states[symbol] = BreakoutSequenceState(symbol=symbol)
        else:
            state.stage = to_stage
            state.stage_entry_bar = state.last_update_bar

    def _get_current_adx(self, h1_data: pd.DataFrame) -> float:
        """Get current ADX value from h1 data."""
        if 'adx' in h1_data.columns and len(h1_data) > 0:
            val = h1_data['adx'].iloc[-1]
            if pd.notna(val):
                return float(val)

        # Compute on the fly
        try:
            from indicators.adx import calculate_adx
            h1_copy = h1_data.copy()
            h1_with_adx = calculate_adx(h1_copy, period=14)
            if 'adx' in h1_with_adx.columns:
                return float(h1_with_adx['adx'].iloc[-1])
        except Exception:
            pass
        return 0.0

    def _compute_atr(self, h1_data: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR manually."""
        if len(h1_data) < period + 1:
            return 0.0
        tr = pd.concat([
            h1_data['high'] - h1_data['low'],
            (h1_data['high'] - h1_data['close'].shift(1)).abs(),
            (h1_data['low'] - h1_data['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def _detect_fvg(self, data: pd.DataFrame, price: float) -> Dict:
        """Detect Fair Value Gaps near current price (reuse pattern from signal_detector)."""
        tolerance_pct = SMC_BO_RETEST_TOLERANCE_PCT
        if len(data) < 10:
            return {'near_bullish_fvg': False, 'near_bearish_fvg': False}

        lookback = min(20, len(data) - 3)
        near_bullish = False
        near_bearish = False

        for i in range(len(data) - 3, len(data) - 3 - lookback, -1):
            if i < 0:
                break
            candle1 = data.iloc[i]
            candle3 = data.iloc[i + 2]

            # Bullish FVG
            if candle1['high'] < candle3['low']:
                gap_low = candle1['high']
                gap_high = candle3['low']
                if gap_low * (1 - tolerance_pct) <= price <= gap_high * (1 + tolerance_pct):
                    near_bullish = True
                    break

            # Bearish FVG
            elif candle1['low'] > candle3['high']:
                gap_high = candle1['low']
                gap_low = candle3['high']
                if gap_low * (1 - tolerance_pct) <= price <= gap_high * (1 + tolerance_pct):
                    near_bearish = True
                    break

        return {'near_bullish_fvg': near_bullish, 'near_bearish_fvg': near_bearish}

    def _build_factors_list(self, state: BreakoutSequenceState) -> List[str]:
        """Build human-readable factors list for the signal."""
        factors = [
            f"Compression: {state.range_size_pips:.0f} pip range "
            f"({state.compression_bar_count} bars)",
            f"Liquidity: {state.liquidity_side}",
        ]
        if state.eqh_level:
            factors.append(f"EQH: {state.eqh_level:.5f}")
        if state.eql_level:
            factors.append(f"EQL: {state.eql_level:.5f}")
        factors.append(f"Sweep {state.sweep_direction} to {state.sweep_extreme:.5f}")
        factors.append(f"BOS {state.bos_type} at {state.bos_level:.5f}")
        factors.append(f"Retest: {state.retest_type} at {state.retest_level:.5f}")
        factors.append(f"Sequence: {state.total_bars_in_sequence} bars total")
        return factors
