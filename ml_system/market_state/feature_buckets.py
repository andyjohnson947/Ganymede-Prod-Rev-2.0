#!/usr/bin/env python3
"""
Feature Bucket Normalisation (Phase 2)

Converts continuous indicator values from market state snapshots
into categorical bins for clustering. This enables k-means to work
on discrete market states rather than raw floats.

Usage:
    python feature_buckets.py                  # Bucketize all symbols
    python feature_buckets.py --symbol EURUSD  # Bucketize one symbol
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ml_system.market_state.config import (
    SYMBOLS, SNAPSHOT_DIR, BUCKETED_DIR,
)


class FeatureBucketizer:
    """
    Convert continuous indicator values to categorical bins.

    Produces ~15 categorical features from each snapshot, which are then
    one-hot encoded (~60 binary columns) for k-means clustering.
    """

    # ── ADX Strength Buckets ──────────────────────────────────────
    ADX_BINS = [
        ('WEAK', 0, 20),
        ('EMERGING', 20, 25),
        ('MODERATE', 25, 35),
        ('STRONG', 35, 50),
        ('EXTREME', 50, 200),
    ]

    # ── VWAP Position Buckets (based on distance_pct) ─────────────
    VWAP_BINS = [
        ('FAR_BELOW', -100, -0.3),
        ('BELOW_BAND2', -0.3, -0.15),
        ('BELOW_BAND1', -0.15, -0.05),
        ('AT_VWAP', -0.05, 0.05),
        ('ABOVE_BAND1', 0.05, 0.15),
        ('ABOVE_BAND2', 0.15, 0.3),
        ('FAR_ABOVE', 0.3, 100),
    ]

    # ── RSI Buckets ───────────────────────────────────────────────
    RSI_BINS = [
        ('OVERSOLD', 0, 30),
        ('WEAK', 30, 40),
        ('NEUTRAL', 40, 60),
        ('STRONG', 60, 70),
        ('OVERBOUGHT', 70, 100),
    ]

    # ── ATR Percentile Buckets (volatility regime) ────────────────
    ATR_BINS = [
        ('COMPRESSED', 0, 0.25),
        ('NORMAL', 0.25, 0.75),
        ('EXPANDED', 0.75, 0.9),
        ('EXTREME', 0.9, 1.01),
    ]

    # ── MACD Histogram Buckets ────────────────────────────────────
    MACD_BINS = [
        ('STRONG_BEARISH', -1, -0.0005),
        ('WEAK_BEARISH', -0.0005, -0.0001),
        ('NEUTRAL', -0.0001, 0.0001),
        ('WEAK_BULLISH', 0.0001, 0.0005),
        ('STRONG_BULLISH', 0.0005, 1),
    ]

    # ── HTF Score Buckets ─────────────────────────────────────────
    HTF_BINS = [
        ('NONE', -1, 0.5),
        ('WEAK', 0.5, 3.5),
        ('MODERATE', 3.5, 6.5),
        ('STRONG', 6.5, 10.5),
        ('VERY_STRONG', 10.5, 100),
    ]

    # ── Momentum (20-bar ROC) Buckets ─────────────────────────────
    MOMENTUM_BINS = [
        ('STRONG_DOWN', -100, -0.3),
        ('WEAK_DOWN', -0.3, -0.05),
        ('FLAT', -0.05, 0.05),
        ('WEAK_UP', 0.05, 0.3),
        ('STRONG_UP', 0.3, 100),
    ]

    # ── Session Buckets (hour-based) ──────────────────────────────
    SESSION_MAP = {
        'asian_early': 'ASIAN_EARLY',
        'asian_late': 'ASIAN_LATE',
        'london_open': 'LONDON_OPEN',
        'london_mid': 'LONDON_MID',
        'overlap': 'OVERLAP',
        'ny_late': 'NY_LATE',
        'close': 'CLOSE',
    }

    # ── Day of Week Buckets ───────────────────────────────────────
    DAY_MAP = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}

    def _bin_value(self, value: float, bins: List) -> str:
        """Classify a value into the correct bin."""
        for label, low, high in bins:
            if low <= value < high:
                return label
        # Fallback to last bin
        return bins[-1][0]

    def bucketize_snapshot(self, snap: Dict) -> Dict:
        """
        Convert a raw snapshot dict to bucketed categorical features.

        Args:
            snap: Raw MarketStateSnapshot dict

        Returns:
            Dict with categorical string values for each dimension
        """
        b = {}

        # ADX regime
        b['adx_regime'] = self._bin_value(snap.get('adx', 0), self.ADX_BINS)

        # ADX direction
        b['adx_direction'] = snap.get('adx_direction', 'neutral') or 'neutral'

        # VWAP position
        b['vwap_position'] = self._bin_value(
            snap.get('vwap_distance_pct', 0), self.VWAP_BINS
        )

        # RSI state
        b['rsi_state'] = self._bin_value(snap.get('rsi_14', 50), self.RSI_BINS)

        # Volatility regime (ATR percentile)
        b['volatility_regime'] = self._bin_value(
            snap.get('atr_percentile_50', 0.5), self.ATR_BINS
        )

        # MACD state
        b['macd_state'] = self._bin_value(
            snap.get('macd_histogram', 0), self.MACD_BINS
        )

        # Volume Profile position
        if snap.get('vp_at_poc'):
            b['vp_position'] = 'AT_POC'
        elif snap.get('vp_above_vah'):
            b['vp_position'] = 'ABOVE_VAH'
        elif snap.get('vp_below_val'):
            b['vp_position'] = 'BELOW_VAL'
        else:
            b['vp_position'] = 'IN_VALUE_AREA'

        # HTF strength
        b['htf_strength'] = self._bin_value(
            snap.get('htf_confluence_score', 0), self.HTF_BINS
        )

        # Candle alignment (already categorical from ADX module)
        alignment = snap.get('candle_alignment', 'mixed') or 'mixed'
        b['candle_alignment'] = alignment.upper() if alignment else 'MIXED'

        # Session
        session = snap.get('session', 'close') or 'close'
        b['session_block'] = self.SESSION_MAP.get(session, 'CLOSE')

        # Day of week
        dow = snap.get('day_of_week', 0)
        b['day_of_week'] = self.DAY_MAP.get(dow, 'MON')

        # Momentum (20-bar ROC)
        b['momentum_regime'] = self._bin_value(
            snap.get('momentum_20_pct', 0), self.MOMENTUM_BINS
        )

        # FVG proximity
        if snap.get('near_bullish_fvg'):
            b['fvg_proximity'] = 'BULL_FVG'
        elif snap.get('near_bearish_fvg'):
            b['fvg_proximity'] = 'BEAR_FVG'
        else:
            b['fvg_proximity'] = 'NONE'

        # Swing proximity
        if snap.get('vp_at_swing_high'):
            b['swing_proximity'] = 'AT_SWING_HIGH'
        elif snap.get('vp_at_swing_low'):
            b['swing_proximity'] = 'AT_SWING_LOW'
        else:
            b['swing_proximity'] = 'BETWEEN'

        # SMC bias (if available)
        smc_bias = snap.get('smc_htf_bias', '') or ''
        if 'bullish' in smc_bias:
            b['smc_bias'] = 'BULLISH'
        elif 'bearish' in smc_bias:
            b['smc_bias'] = 'BEARISH'
        else:
            b['smc_bias'] = 'NEUTRAL'

        # DI direction (who's dominant)
        plus_di = snap.get('plus_di', 0)
        minus_di = snap.get('minus_di', 0)
        di_sep = abs(plus_di - minus_di)
        if di_sep < 3:
            b['di_dominance'] = 'BALANCED'
        elif plus_di > minus_di:
            b['di_dominance'] = 'BULLS'
        else:
            b['di_dominance'] = 'BEARS'

        return b

    def bucketize_batch(self, snapshots: List[Dict]) -> pd.DataFrame:
        """
        Bucketize all snapshots into a DataFrame of categorical features.

        Args:
            snapshots: List of raw snapshot dicts

        Returns:
            DataFrame with categorical string columns
        """
        rows = []
        for snap in snapshots:
            row = self.bucketize_snapshot(snap)
            # Preserve key fields for joining later
            row['bar_index'] = snap.get('bar_index', 0)
            row['timestamp'] = snap.get('timestamp', '')
            row['symbol'] = snap.get('symbol', '')
            rows.append(row)

        df = pd.DataFrame(rows)
        print(f"  Bucketized {len(df):,} snapshots -> {len(df.columns)} columns")
        return df

    def one_hot_encode(self, bucketed_df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical columns for k-means input.

        Excludes non-feature columns (bar_index, timestamp, symbol).

        Returns:
            DataFrame of 0/1 binary columns
        """
        # Columns to encode (exclude identity cols + noise features)
        from ml_system.market_state.config import CLUSTER_EXCLUDE_FEATURES
        exclude = {'bar_index', 'timestamp', 'symbol'} | CLUSTER_EXCLUDE_FEATURES
        feature_cols = [c for c in bucketed_df.columns if c not in exclude]

        encoded = pd.get_dummies(bucketed_df[feature_cols], dtype=int)
        print(f"  One-hot encoded: {len(feature_cols)} categories -> {len(encoded.columns)} binary columns")
        return encoded

    def save_bucketed(self, df: pd.DataFrame, symbol: str) -> str:
        """Save bucketed DataFrame to CSV."""
        filepath = BUCKETED_DIR / f"{symbol}_bucketed.csv"
        df.to_csv(filepath, index=False)
        print(f"  [SAVED] {filepath}")
        return str(filepath)

    def load_bucketed(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load bucketed DataFrame from CSV."""
        filepath = BUCKETED_DIR / f"{symbol}_bucketed.csv"
        if not filepath.exists():
            return None
        return pd.read_csv(filepath)


def main():
    """CLI entry point for bucketization."""
    import argparse
    from ml_system.market_state.bar_replay_engine import load_snapshot_data

    parser = argparse.ArgumentParser(description='Feature bucketization')
    parser.add_argument('--symbol', type=str, help='Process specific symbol only')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS
    bucketizer = FeatureBucketizer()

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  BUCKETIZING: {symbol}")
        print(f"{'='*60}")

        snapshots = load_snapshot_data(symbol)
        if not snapshots:
            print(f"  [SKIP] No snapshots for {symbol}")
            continue

        # Bucketize
        bucketed_df = bucketizer.bucketize_batch(snapshots)

        # Save
        bucketizer.save_bucketed(bucketed_df, symbol)

        # Show distribution summary
        print(f"\n  Feature distributions:")
        for col in bucketed_df.columns:
            if col not in ('bar_index', 'timestamp', 'symbol'):
                counts = bucketed_df[col].value_counts()
                top = counts.index[0]
                top_pct = counts.iloc[0] / len(bucketed_df) * 100
                print(f"    {col}: {len(counts)} categories, "
                      f"most common = {top} ({top_pct:.0f}%)")


if __name__ == '__main__':
    main()
