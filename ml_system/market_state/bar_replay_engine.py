#!/usr/bin/env python3
"""
Bar Replay Engine

Replays historical H1 bars one-by-one, computing ALL indicators at each bar
using only data available at that point in time (no lookahead).

This is the core of the Market State Modelling Engine.

Usage:
    python bar_replay_engine.py                  # Replay all symbols
    python bar_replay_engine.py --symbol EURUSD  # Replay one symbol
"""

import sys
import json
import time as time_mod
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Path setup (matches continuous_logger.py pattern)
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / 'trading_bot'))

# Import existing indicator modules
from indicators.vwap import VWAP
from indicators.volume_profile import VolumeProfile
from indicators.htf_levels import HTFLevels
from indicators.adx import (
    calculate_adx, interpret_adx,
    analyze_candle_direction, analyze_candle_momentum,
)
from indicators.technical import calculate_rsi, calculate_macd
from indicators.liquidity_levels import LiquidityLevels
from config.strategy_config import LEVEL_TOLERANCE_PCT

# Import market state modules
from ml_system.market_state.config import (
    SYMBOLS, MIN_WARMUP_BARS_H1, SNAPSHOT_DIR, BROKER_GMT_OFFSET,
    FORWARD_HORIZONS, PIP_FACTOR,
)
from ml_system.market_state.snapshot_schema import MarketStateSnapshot
from ml_system.market_state.forward_outcomes import ForwardOutcomeMeasurer
from ml_system.market_state.data_fetcher import HistoricalDataFetcher

# Optional: SMC library
try:
    import io as _io
    _old_stdout = sys.stdout
    sys.stdout = _io.StringIO()
    from smartmoneyconcepts import smc as smc_lib
    sys.stdout = _old_stdout
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
except Exception:
    sys.stdout = _old_stdout if '_old_stdout' in dir() else sys.stdout
    SMC_AVAILABLE = False


# ── Session classification ────────────────────────────────────────
def _classify_session(hour_utc: int) -> str:
    """Classify hour into trading session."""
    if 0 <= hour_utc <= 3:
        return 'asian_early'
    elif 4 <= hour_utc <= 7:
        return 'asian_late'
    elif 8 <= hour_utc <= 9:
        return 'london_open'
    elif 10 <= hour_utc <= 12:
        return 'london_mid'
    elif 13 <= hour_utc <= 16:
        return 'overlap'
    elif 17 <= hour_utc <= 20:
        return 'ny_late'
    else:
        return 'close'


class BarReplayEngine:
    """
    Replays historical H1 bars, computing all indicators at each bar.

    Key optimization: ADX, VWAP, RSI, MACD are pre-computed once on the full
    H1 DataFrame (they use rolling/ewm which is inherently backward-looking).
    Only Volume Profile, HTF levels, and liquidity levels are computed per-bar.
    """

    def __init__(self, include_smc: bool = True):
        """
        Initialize with indicator instances.

        Args:
            include_smc: Whether to include SMC structure analysis (requires library)
        """
        self.vwap = VWAP()
        self.volume_profile = VolumeProfile()
        self.htf_levels = HTFLevels()
        self.liquidity_levels = LiquidityLevels()
        self.outcome_measurer = ForwardOutcomeMeasurer()
        self.include_smc = include_smc and SMC_AVAILABLE

        if include_smc and not SMC_AVAILABLE:
            print("[WARN] SMC library not available — SMC fields will be empty")

    def replay(
        self,
        symbol: str,
        h1_data: pd.DataFrame,
        d1_data: pd.DataFrame,
        w1_data: pd.DataFrame,
        start_index: int = MIN_WARMUP_BARS_H1,
        end_index: int = None,
        progress_interval: int = 1000,
    ) -> str:
        """
        Bar-by-bar replay of the entire H1 history. Writes JSONL incrementally.

        Args:
            symbol: Trading symbol
            h1_data: Full H1 DataFrame (DatetimeIndex)
            d1_data: Full D1 DataFrame (DatetimeIndex)
            w1_data: Full W1 DataFrame (DatetimeIndex)
            start_index: First bar to process (after warmup)
            end_index: Last bar to process (None = all)
            progress_interval: Print progress every N bars

        Returns:
            Path to the output JSONL file.
        """
        if end_index is None:
            end_index = len(h1_data) - 1

        output_path = SNAPSHOT_DIR / f"{symbol}_snapshots.jsonl"
        total_bars = end_index - start_index + 1

        print(f"\n{'='*60}")
        print(f"  REPLAY: {symbol} | {total_bars:,} bars")
        print(f"  Range: bar {start_index} to {end_index}")
        print(f"  Output: {output_path}")
        print(f"{'='*60}")

        # ── Step 1: Pre-compute rolling indicators on full H1 data ──
        print("  Pre-computing rolling indicators...")
        t0 = time_mod.time()

        h1_with_adx = calculate_adx(h1_data.copy(), period=14)
        h1_with_vwap = self.vwap.calculate(h1_data.copy())
        rsi_series = calculate_rsi(h1_data.copy(), period=14)
        macd_line, macd_signal, macd_histogram = calculate_macd(h1_data.copy())

        # ATR for volatility analysis (14-period)
        high_low = h1_data['high'] - h1_data['low']
        high_close = np.abs(h1_data['high'] - h1_data['close'].shift())
        low_close = np.abs(h1_data['low'] - h1_data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_series = true_range.rolling(window=14).mean()

        print(f"  Pre-computation done in {time_mod.time() - t0:.1f}s")

        # ── Step 2: Prepare H1 data with 'time' column for liquidity ──
        # LiquidityLevels.calculate_session_levels() needs a 'time' column
        h1_with_time = h1_data.copy()
        if 'time' not in h1_with_time.columns:
            h1_with_time['time'] = h1_with_time.index

        # ── Step 3: Bar-by-bar replay ────────────────────────────────
        print(f"  Starting bar-by-bar replay...")
        t0 = time_mod.time()
        snapshot_count = 0
        errors = 0

        with open(output_path, 'w') as f:
            for i in range(start_index, end_index + 1):
                try:
                    snapshot = self._process_bar(
                        symbol=symbol,
                        bar_index=i,
                        h1_data=h1_data,
                        h1_with_adx=h1_with_adx,
                        h1_with_vwap=h1_with_vwap,
                        h1_with_time=h1_with_time,
                        rsi_series=rsi_series,
                        macd_line=macd_line,
                        macd_signal=macd_signal,
                        macd_histogram=macd_histogram,
                        atr_series=atr_series,
                        d1_data=d1_data,
                        w1_data=w1_data,
                    )

                    if snapshot is not None:
                        f.write(snapshot.to_json() + '\n')
                        snapshot_count += 1

                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  [ERROR] Bar {i}: {e}")
                    elif errors == 6:
                        print(f"  [ERROR] Suppressing further errors...")

                # Progress reporting
                processed = i - start_index + 1
                if processed % progress_interval == 0:
                    elapsed = time_mod.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total_bars - processed) / rate if rate > 0 else 0
                    print(f"    {processed:,}/{total_bars:,} bars "
                          f"({processed/total_bars*100:.1f}%) | "
                          f"{rate:.0f} bars/s | "
                          f"ETA {eta:.0f}s | "
                          f"Errors: {errors}")

        elapsed = time_mod.time() - t0
        print(f"\n  COMPLETE: {snapshot_count:,} snapshots written in {elapsed:.1f}s "
              f"({snapshot_count/elapsed:.0f} bars/s)")
        if errors > 0:
            print(f"  WARNING: {errors} errors encountered")

        return str(output_path)

    def _process_bar(
        self,
        symbol: str,
        bar_index: int,
        h1_data: pd.DataFrame,
        h1_with_adx: pd.DataFrame,
        h1_with_vwap: pd.DataFrame,
        h1_with_time: pd.DataFrame,
        rsi_series: pd.Series,
        macd_line: pd.Series,
        macd_signal: pd.Series,
        macd_histogram: pd.Series,
        atr_series: pd.Series,
        d1_data: pd.DataFrame,
        w1_data: pd.DataFrame,
    ) -> Optional[MarketStateSnapshot]:
        """
        Process a single bar: compute all indicators and build snapshot.

        Uses pre-computed rolling indicators (ADX, VWAP, RSI, MACD, ATR)
        and computes per-bar indicators (VP, HTF, liquidity, SMC).
        """
        bar = h1_data.iloc[bar_index]
        bar_time = h1_data.index[bar_index]
        price = bar['close']

        # ── Identity + OHLCV + Temporal ───────────────────────────
        if isinstance(bar_time, pd.Timestamp):
            hour_utc = bar_time.hour
            dow = bar_time.weekday()
            dom = bar_time.day
            woy = bar_time.isocalendar()[1]
            month = bar_time.month
            ts_str = bar_time.isoformat()
        else:
            hour_utc = 0
            dow = 0
            dom = 0
            woy = 0
            month = 0
            ts_str = str(bar_time)

        snap = MarketStateSnapshot(
            symbol=symbol,
            timestamp=ts_str,
            bar_index=bar_index,
            open=float(bar['open']),
            high=float(bar['high']),
            low=float(bar['low']),
            close=float(bar['close']),
            volume=float(bar.get('volume', bar.get('tick_volume', 0))),
            hour_utc=hour_utc,
            day_of_week=dow,
            day_of_month=dom,
            week_of_year=woy,
            month=month,
            session=_classify_session(hour_utc),
        )

        # ── ADX / Trend (pre-computed) ────────────────────────────
        adx_row = h1_with_adx.iloc[bar_index]
        adx_val = float(adx_row.get('adx', 0))
        plus_di = float(adx_row.get('plus_di', 0))
        minus_di = float(adx_row.get('minus_di', 0))

        if not np.isnan(adx_val):
            interp = interpret_adx(adx_val, plus_di, minus_di)
            snap.adx = adx_val
            snap.plus_di = plus_di
            snap.minus_di = minus_di
            snap.di_separation = abs(plus_di - minus_di)
            snap.atr = float(adx_row.get('atr', 0)) if 'atr' in adx_row.index else 0.0
            snap.adx_strength = interp['strength']
            snap.adx_direction = interp['direction']
            snap.adx_is_ranging = interp['is_ranging']

        # ── Candle Direction (computed on slice) ──────────────────
        if bar_index >= 5:
            h1_slice_small = h1_data.iloc[max(0, bar_index - 10):bar_index + 1]
            candle_dir = analyze_candle_direction(h1_slice_small, lookback=5)
            snap.candle_alignment = candle_dir['alignment']
            snap.candle_bullish_pct = candle_dir['bullish_pct']
            snap.candle_bearish_pct = candle_dir['bearish_pct']
            snap.candle_avg_body = candle_dir['avg_body']

        # ── Candle Momentum ───────────────────────────────────────
        if bar_index >= 6:
            h1_slice_mom = h1_data.iloc[max(0, bar_index - 10):bar_index + 1]
            try:
                buy_mom = analyze_candle_momentum(h1_slice_mom, 'buy', lookback=4, shrink_ratio=0.85)
                sell_mom = analyze_candle_momentum(h1_slice_mom, 'sell', lookback=4, shrink_ratio=0.85)
                snap.momentum_buy_shrinking = buy_mom.get('shrinking_pairs', 0)
                snap.momentum_sell_shrinking = sell_mom.get('shrinking_pairs', 0)
                snap.momentum_buy_is_fading = buy_mom.get('is_fading', False)
                snap.momentum_sell_is_fading = sell_mom.get('is_fading', False)
            except Exception:
                pass

        # ── VWAP (pre-computed) ───────────────────────────────────
        vwap_row = h1_with_vwap.iloc[bar_index]
        vwap_val = vwap_row.get('vwap', 0)
        if not np.isnan(vwap_val) and vwap_val > 0:
            snap.vwap_value = float(vwap_val)
            snap.vwap_distance_pct = float((price - vwap_val) / vwap_val * 100) if vwap_val else 0
            snap.vwap_direction = 'above' if price > vwap_val else 'below'
            snap.vwap_std = float(vwap_row.get('vwap_std', 0))

            upper_1 = vwap_row.get('vwap_upper_1', 0)
            lower_1 = vwap_row.get('vwap_lower_1', 0)
            upper_2 = vwap_row.get('vwap_upper_2', 0)
            lower_2 = vwap_row.get('vwap_lower_2', 0)
            upper_3 = vwap_row.get('vwap_upper_3', 0)
            lower_3 = vwap_row.get('vwap_lower_3', 0)

            snap.vwap_in_band_1 = bool(
                (not np.isnan(lower_1)) and lower_1 <= price <= upper_1
            ) if not np.isnan(upper_1) else False
            snap.vwap_in_band_2 = bool(
                (not np.isnan(lower_2)) and (lower_2 <= price <= lower_1 or upper_1 <= price <= upper_2)
            ) if not np.isnan(upper_2) else False
            snap.vwap_in_band_3 = bool(
                (not np.isnan(lower_3)) and (lower_3 <= price <= lower_2 or upper_2 <= price <= upper_3)
            ) if not np.isnan(upper_3) else False

            # VWAP stretch (distance in ATR units)
            atr_val = float(atr_series.iloc[bar_index]) if not np.isnan(atr_series.iloc[bar_index]) else 0
            if atr_val > 0:
                snap.vwap_stretch = abs(price - vwap_val) / atr_val

        # ── Volume Profile (per-bar, lookback 50 for speed) ───────
        lookback_start = max(0, bar_index - 49)
        h1_slice_vp = h1_data.iloc[lookback_start:bar_index + 1]
        try:
            vp_signals = self.volume_profile.get_signals(h1_slice_vp, price, lookback=50)
            snap.vp_poc = float(vp_signals.get('poc', 0) or 0)
            snap.vp_vah = float(vp_signals.get('vah', 0) or 0)
            snap.vp_val = float(vp_signals.get('val', 0) or 0)
            snap.vp_at_poc = bool(vp_signals.get('at_poc', False))
            snap.vp_above_vah = bool(vp_signals.get('above_vah', False))
            snap.vp_below_val = bool(vp_signals.get('below_val', False))
            snap.vp_at_lvn = bool(vp_signals.get('at_lvn', False))
            snap.vp_at_swing_high = bool(vp_signals.get('at_swing_high', False))
            snap.vp_at_swing_low = bool(vp_signals.get('at_swing_low', False))
            snap.vp_swing_high_price = float(vp_signals.get('swing_high_price', 0) or 0)
            snap.vp_swing_low_price = float(vp_signals.get('swing_low_price', 0) or 0)
        except Exception:
            pass

        # ── HTF Levels (per-bar, filtered for lookahead) ──────────
        d1_available = self._filter_htf_data(d1_data, bar_time)
        w1_available = self._filter_htf_data(w1_data, bar_time)

        if len(d1_available) >= 2 and len(w1_available) >= 2:
            try:
                all_levels = self.htf_levels.get_all_levels(d1_available, w1_available)
                htf_confluence = self.htf_levels.check_confluence(
                    price, all_levels, LEVEL_TOLERANCE_PCT
                )
                snap.htf_confluence_score = htf_confluence.get('score', 0)
                snap.htf_factors = htf_confluence.get('factors', [])
                snap.htf_factor_count = len(snap.htf_factors)

                # Individual levels from daily
                daily_levels = self.htf_levels.calculate_daily_levels(d1_available)
                snap.prev_day_high = float(daily_levels.get('prev_day_high', 0) or 0)
                snap.prev_day_low = float(daily_levels.get('prev_day_low', 0) or 0)
                snap.prev_day_poc = float(daily_levels.get('prev_day_poc', 0) or 0)
                snap.prev_day_vah = float(daily_levels.get('prev_day_vah', 0) or 0)
                snap.prev_day_val = float(daily_levels.get('prev_day_val', 0) or 0)
                snap.prev_day_close = float(daily_levels.get('prev_day_close', 0) or 0)

                # Individual levels from weekly
                weekly_levels = self.htf_levels.calculate_weekly_levels(w1_available)
                snap.prev_week_high = float(weekly_levels.get('prev_week_high', 0) or 0)
                snap.prev_week_low = float(weekly_levels.get('prev_week_low', 0) or 0)
                snap.weekly_poc = float(weekly_levels.get('weekly_poc', 0) or 0)
            except Exception:
                pass

        # ── Technical Indicators (pre-computed) ───────────────────
        rsi_val = rsi_series.iloc[bar_index] if bar_index < len(rsi_series) else 50
        snap.rsi_14 = float(rsi_val) if not np.isnan(rsi_val) else 50.0

        ml = macd_line.iloc[bar_index] if bar_index < len(macd_line) else 0
        ms = macd_signal.iloc[bar_index] if bar_index < len(macd_signal) else 0
        mh = macd_histogram.iloc[bar_index] if bar_index < len(macd_histogram) else 0
        snap.macd_line = float(ml) if not np.isnan(ml) else 0.0
        snap.macd_signal = float(ms) if not np.isnan(ms) else 0.0
        snap.macd_histogram = float(mh) if not np.isnan(mh) else 0.0

        # ── Volatility ────────────────────────────────────────────
        atr_val = float(atr_series.iloc[bar_index]) if not np.isnan(atr_series.iloc[bar_index]) else 0
        snap.atr_14 = atr_val

        # ATR percentile vs last 50 bars
        if bar_index >= 50:
            atr_window = atr_series.iloc[bar_index - 49:bar_index + 1]
            valid = atr_window.dropna()
            if len(valid) > 0:
                snap.atr_percentile_50 = float((valid < atr_val).sum() / len(valid))

        # Recent range (10-bar)
        if bar_index >= 10:
            recent = h1_data.iloc[bar_index - 9:bar_index + 1]
            snap.recent_range_pips = float((recent['high'].max() - recent['low'].min()) * PIP_FACTOR)
            snap.distance_from_high_pct = float(
                (price - recent['high'].max()) / price * 100
            )
            snap.distance_from_low_pct = float(
                (price - recent['low'].min()) / price * 100
            )

        # Momentum (20-bar ROC)
        if bar_index >= 20:
            close_20_ago = h1_data.iloc[bar_index - 20]['close']
            snap.momentum_20_pct = float((price - close_20_ago) / close_20_ago * 100)

        # ── Liquidity Levels (per-bar) ────────────────────────────
        if bar_index >= 48:  # Needs 48+ bars
            h1_slice_liq = h1_with_time.iloc[max(0, bar_index - 99):bar_index + 1].copy()
            try:
                session_levels = self.liquidity_levels.calculate_session_levels(
                    h1_slice_liq, BROKER_GMT_OFFSET
                )
                equal_levels = self.liquidity_levels.detect_equal_levels(h1_slice_liq)

                # Check proximity for buy direction
                buy_confluence = self.liquidity_levels.check_confluence(
                    current_price=price, direction='buy',
                    session_levels=session_levels,
                    equal_levels=equal_levels,
                    order_blocks={},
                    tolerance_pct=LEVEL_TOLERANCE_PCT,
                )
                sell_confluence = self.liquidity_levels.check_confluence(
                    current_price=price, direction='sell',
                    session_levels=session_levels,
                    equal_levels=equal_levels,
                    order_blocks={},
                    tolerance_pct=LEVEL_TOLERANCE_PCT,
                )

                snap.at_session_low = bool(buy_confluence.get('at_session_low', False))
                snap.at_session_high = bool(sell_confluence.get('at_session_high', False))
                snap.near_eql = bool(buy_confluence.get('near_eql', False))
                snap.near_eqh = bool(sell_confluence.get('near_eqh', False))
            except Exception:
                pass

        # ── FVG Detection ─────────────────────────────────────────
        if len(d1_available) >= 10:
            try:
                fvg = self._detect_fvgs(d1_available, price)
                snap.near_bullish_fvg = fvg.get('near_bullish_fvg', False)
                snap.near_bearish_fvg = fvg.get('near_bearish_fvg', False)
            except Exception:
                pass

        # ── SMC Structure ─────────────────────────────────────────
        if self.include_smc and bar_index >= 50:
            try:
                h1_slice_smc = h1_data.iloc[max(0, bar_index - 199):bar_index + 1]
                smc_state = self._compute_smc(h1_slice_smc)
                snap.smc_available = smc_state.get('available', False)
                snap.smc_htf_bias = smc_state.get('bias', '')
                snap.smc_alignment = smc_state.get('alignment', '')
                snap.smc_last_bos_type = smc_state.get('last_bos_type', '')
                snap.smc_last_bos_bars_ago = smc_state.get('last_bos_bars_ago', 999)
            except Exception:
                pass

        # ── Forward Outcomes ──────────────────────────────────────
        outcomes = self.outcome_measurer.measure_outcomes(h1_data, bar_index)
        for key, value in outcomes.items():
            if hasattr(snap, key):
                setattr(snap, key, value)

        return snap

    def _filter_htf_data(
        self, htf_data: pd.DataFrame, current_h1_time: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Filter D1 or W1 data to only include bars completed BEFORE current_h1_time.

        A D1 bar timestamped 2024-03-15 00:00 represents data from that trading day.
        It's only fully "available" once that day is complete, meaning the first H1
        bar of the NEXT day. So we include D1 bars where index < current_h1_time's date.
        """
        if len(htf_data) == 0:
            return htf_data

        # Normalize current_h1_time to start of day
        current_date = current_h1_time.normalize()

        # Only include bars strictly before current date
        # This prevents lookahead: today's D1 bar isn't complete yet
        return htf_data[htf_data.index < current_date]

    def _detect_fvgs(
        self, data: pd.DataFrame, price: float, tolerance_pct: float = 0.003
    ) -> Dict:
        """
        Detect Fair Value Gaps near current price.
        (Replicates SignalDetector.detect_fair_value_gaps logic)
        """
        if len(data) < 10:
            return {'near_bullish_fvg': False, 'near_bearish_fvg': False}

        lookback = min(20, len(data) - 3)
        near_bullish = False
        near_bearish = False

        for i in range(len(data) - 3, len(data) - 3 - lookback, -1):
            if i < 0:
                break

            c1 = data.iloc[i]
            c3 = data.iloc[i + 2]

            # Bullish FVG
            if c1['high'] < c3['low']:
                if c1['high'] * (1 - tolerance_pct) <= price <= c3['low'] * (1 + tolerance_pct):
                    near_bullish = True
                    break

            # Bearish FVG
            elif c1['low'] > c3['high']:
                if c3['high'] * (1 - tolerance_pct) <= price <= c1['low'] * (1 + tolerance_pct):
                    near_bearish = True
                    break

        return {'near_bullish_fvg': near_bullish, 'near_bearish_fvg': near_bearish}

    def _compute_smc(self, h1_slice: pd.DataFrame) -> Dict:
        """
        Compute SMC structure (BOS/CHOCH) on the given H1 slice.
        Uses the smartmoneyconcepts library directly.
        """
        if not SMC_AVAILABLE or len(h1_slice) < 50:
            return {'available': False}

        try:
            # SMC library expects columns: open, high, low, close
            ohlc = h1_slice[['open', 'high', 'low', 'close']].copy()
            ohlc = ohlc.reset_index(drop=True)

            swing_hl = smc_lib.swing_highs_lows(ohlc, swing_length=10)
            bos_choch = smc_lib.bos_choch(ohlc, swing_hl)

            # Get latest BOS
            last_bos_type = ''
            last_bos_bars_ago = 999
            if 'BOS' in bos_choch.columns:
                bos_events = bos_choch[bos_choch['BOS'].notna()]
                if len(bos_events) > 0:
                    last_bos = bos_events.iloc[-1]
                    last_bos_type = 'bullish' if last_bos['BOS'] == 1 else 'bearish'
                    last_bos_bars_ago = len(bos_choch) - 1 - bos_events.index[-1]

            # Get latest CHOCH
            last_choch_type = ''
            last_choch_bars_ago = 999
            if 'CHOCH' in bos_choch.columns:
                choch_events = bos_choch[bos_choch['CHOCH'].notna()]
                if len(choch_events) > 0:
                    last_choch = choch_events.iloc[-1]
                    last_choch_type = 'bullish' if last_choch['CHOCH'] == 1 else 'bearish'
                    last_choch_bars_ago = len(bos_choch) - 1 - choch_events.index[-1]

            # Determine bias
            bias = self._determine_smc_bias(
                last_bos_type, last_bos_bars_ago,
                last_choch_type, last_choch_bars_ago
            )

            return {
                'available': True,
                'bias': bias,
                'alignment': bias,  # Simplified — full alignment needs ETF+HTF
                'last_bos_type': last_bos_type,
                'last_bos_bars_ago': int(last_bos_bars_ago),
                'last_choch_type': last_choch_type,
                'last_choch_bars_ago': int(last_choch_bars_ago),
            }

        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _determine_smc_bias(
        self,
        bos_type: str, bos_bars: int,
        choch_type: str, choch_bars: int,
    ) -> str:
        """Determine market structure bias from BOS/CHOCH events."""
        # CHOCH is more significant (trend change), BOS confirms existing trend
        if choch_bars < bos_bars and choch_type:
            return f'potential_{choch_type}'  # Recent CHOCH = potential change
        elif bos_type:
            return f'established_{bos_type}'  # BOS confirms current trend
        else:
            return 'neutral'


def load_snapshot_data(symbol: str) -> List[Dict]:
    """Load snapshots from JSONL file."""
    filepath = SNAPSHOT_DIR / f"{symbol}_snapshots.jsonl"
    if not filepath.exists():
        print(f"[ERROR] Snapshot file not found: {filepath}")
        return []

    snapshots = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    snapshots.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"[OK] Loaded {len(snapshots):,} snapshots from {filepath}")
    return snapshots


def main():
    """CLI entry point for bar replay."""
    import argparse

    parser = argparse.ArgumentParser(description='Bar-by-bar replay engine')
    parser.add_argument('--symbol', type=str, help='Replay specific symbol only')
    parser.add_argument('--no-smc', action='store_true', help='Disable SMC analysis')
    parser.add_argument('--start', type=int, default=MIN_WARMUP_BARS_H1, help='Start bar index')
    parser.add_argument('--end', type=int, default=None, help='End bar index')
    parser.add_argument('--limit', type=int, default=None, help='Max bars to process')
    args = parser.parse_args()

    fetcher = HistoricalDataFetcher()
    engine = BarReplayEngine(include_smc=not args.no_smc)

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    for symbol in symbols:
        # Load data from CSVs
        h1 = fetcher.load_for_replay(symbol, 'H1')
        d1 = fetcher.load_for_replay(symbol, 'D1')
        w1 = fetcher.load_for_replay(symbol, 'W1')

        if h1 is None or d1 is None or w1 is None:
            print(f"[ERROR] Missing data for {symbol}. Run data_fetcher.py first.")
            continue

        print(f"\n  Data loaded: H1={len(h1):,} bars, D1={len(d1):,} bars, W1={len(w1):,} bars")

        end_idx = args.end
        if args.limit and end_idx is None:
            end_idx = args.start + args.limit - 1

        output = engine.replay(
            symbol=symbol,
            h1_data=h1,
            d1_data=d1,
            w1_data=w1,
            start_index=args.start,
            end_index=end_idx,
        )

        print(f"  Output: {output}")


if __name__ == '__main__':
    main()
