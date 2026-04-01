#!/usr/bin/env python3
"""
H4 Market Regime Classifier using Gaussian Hidden Markov Model

Standalone tool that identifies market regimes (trending, ranging, compression,
volatility spikes, etc.) from H4 GBPUSD price data using unsupervised HMM.

Usage:
    # Full training pipeline (run once, or periodically to retrain)
    python regime_hmm.py --train

    # Predict current regime (called by bot each cycle)
    python regime_hmm.py --predict

    # Analyse historical state distribution
    python regime_hmm.py --analyse

Architecture:
    - Standalone tool: bot reads regime from JSON file
    - Retrains offline, predicts in <50ms
    - Outputs regime_state.json for bot consumption
    - Console output shows alignment/misalignment with current strategy
"""

import sys
import os
import json
import pickle
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'trading_bot'))

# Paths
DATA_DIR = Path(__file__).parent / 'data'
MODEL_DIR = Path(__file__).parent / 'models'
OUTPUT_DIR = Path(__file__).parent / 'outputs'

# Supported symbols
SUPPORTED_SYMBOLS = ['GBPUSD', 'EURUSD']

# Default retrain interval (days)
RETRAIN_INTERVAL_DAYS = 3

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _symbol_paths(symbol: str) -> Dict:
    """Get per-symbol file paths for model artifacts."""
    sym = symbol.upper()
    return {
        'model': MODEL_DIR / f'regime_hmm_model_{sym}.pkl',
        'scaler': MODEL_DIR / f'regime_hmm_scaler_{sym}.pkl',
        'state_map': MODEL_DIR / f'regime_hmm_state_map_{sym}.json',
        'regime_state': OUTPUT_DIR / f'regime_state_{sym}.json',
        'analysis': OUTPUT_DIR / f'regime_analysis_{sym}.json',
        'data': DATA_DIR / f'h4_{sym.lower()}_raw.csv',
        'train_meta': MODEL_DIR / f'regime_hmm_train_meta_{sym}.json',
    }


# Backward compat: old single-file paths (GBPUSD only)
REGIME_STATE_FILE = OUTPUT_DIR / 'regime_state.json'
MODEL_FILE = MODEL_DIR / 'regime_hmm_model.pkl'
SCALER_FILE = MODEL_DIR / 'regime_hmm_scaler.pkl'
STATE_MAP_FILE = MODEL_DIR / 'regime_hmm_state_map.json'


# ============================================================================
# FEATURE ENGINE
# ============================================================================

class H4FeatureEngine:
    """
    Builds DIRECTION-AGNOSTIC features from H4 OHLCV data for regime detection.

    The HMM must cluster by REGIME (trending vs ranging vs compression),
    NOT by direction (bullish vs bearish). All directional features use
    absolute values so the HMM cannot split on up vs down.

    Features (per bar):
        1. body_ratio         - |close-open| / (high-low), body dominance
        2. wick_ratio         - (upper+lower wick) / (high-low), wick dominance
        3. atr_normalised     - ATR14 / ATR14_rolling50, relative volatility
        4. atr_change         - ATR14 pct change, expanding/contracting
        5. volatility_ratio   - current range / 20-bar avg range
        6. abs_slope_4        - |slope| over 4 bars / ATR, short-term trending
        7. abs_slope_12       - |slope| over 12 bars / ATR, sustained trending
        8. efficiency_12      - |net close change| / sum(ranges), path efficiency
        9. candle_consistency - % of last 6 bars in same direction
        10. volume_ratio      - tick_volume / 20-bar avg volume
        11. hh_ll_imbalance   - |count(HH) - count(LL)| / 6, structure trend
        12. range_percentile  - current range vs 50-bar range percentile
    """

    def __init__(self, pip_size: float = 0.0001):
        self.pip_size = pip_size

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all direction-agnostic features from raw OHLCV DataFrame."""
        df = df.copy()

        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Basic derived columns
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        range_safe = df['range'].replace(0, np.nan)

        # 1. Body ratio (body dominance — high in trends, low in indecision)
        df['body_ratio'] = df['body'] / range_safe

        # 2. Wick ratio (wick dominance — high in ranging/exhaustion)
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / range_safe

        # 3. ATR normalised (relative volatility level)
        df['tr'] = np.maximum(
            df['range'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        atr_norm = df['atr_14'].rolling(50).mean()
        df['atr_normalised'] = df['atr_14'] / atr_norm.replace(0, np.nan)

        # 4. ATR change (expansion/contraction rate)
        df['atr_change'] = df['atr_14'].pct_change()

        # 5. Volatility ratio (current bar vs recent average)
        avg_range_20 = df['range'].rolling(20).mean()
        df['volatility_ratio'] = df['range'] / avg_range_20.replace(0, np.nan)

        # 6-7. ABSOLUTE slopes (direction removed — purely about trending strength)
        slope_4_raw = self._rolling_slope(df['close'], 4)
        slope_12_raw = self._rolling_slope(df['close'], 12)
        df['abs_slope_4'] = slope_4_raw.abs() / df['atr_14'].replace(0, np.nan)
        df['abs_slope_12'] = slope_12_raw.abs() / df['atr_14'].replace(0, np.nan)

        # 8. Directional efficiency (THE key regime feature)
        # Net move / total path. High = trending (price going somewhere).
        # Low = ranging (price churning back and forth).
        net_move_12 = (df['close'] - df['close'].shift(12)).abs()
        total_path_12 = df['range'].rolling(12).sum()
        df['efficiency_12'] = net_move_12 / total_path_12.replace(0, np.nan)

        # 9. Candle direction consistency (6-bar window)
        body_dir = np.where(df['close'] > df['open'], 1, -1)
        df['_body_dir'] = body_dir
        df['candle_consistency'] = df['_body_dir'].rolling(6).apply(
            lambda x: max(sum(x > 0), sum(x < 0)) / len(x) if len(x) == 6 else 0.5,
            raw=True
        )

        # 10. Volume ratio
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        if vol_col in df.columns:
            avg_vol_20 = df[vol_col].rolling(20).mean()
            df['volume_ratio'] = df[vol_col] / avg_vol_20.replace(0, np.nan)
        else:
            df['volume_ratio'] = 1.0

        # 11. HH/LL imbalance (market structure trending indicator)
        # High = sustained HHs or LLs (trending). Low = mixed (ranging).
        hh = (df['high'] > df['high'].shift(1)).astype(float)
        ll = (df['low'] < df['low'].shift(1)).astype(float)
        df['hh_ll_imbalance'] = (hh.rolling(6).sum() - ll.rolling(6).sum()).abs() / 6.0

        # 12. Range percentile (where current range sits vs recent history)
        df['range_percentile'] = df['range'].rolling(50).rank(pct=True)

        # Select feature columns (ALL direction-agnostic)
        feature_cols = [
            'body_ratio', 'wick_ratio',
            'atr_normalised', 'atr_change', 'volatility_ratio',
            'abs_slope_4', 'abs_slope_12', 'efficiency_12',
            'candle_consistency', 'volume_ratio',
            'hh_ll_imbalance', 'range_percentile'
        ]

        # Drop NaN rows (from rolling calculations)
        features = df[['time'] + feature_cols].dropna().copy()
        features = features.reset_index(drop=True)

        return features, feature_cols

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear regression slope."""
        def _slope(y):
            if len(y) < window:
                return np.nan
            x = np.arange(len(y))
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except (np.linalg.LinAlgError, ValueError):
                return np.nan

        return series.rolling(window).apply(_slope, raw=True)


# ============================================================================
# HMM TRAINER
# ============================================================================

class RegimeHMMTrainer:
    """
    Trains Gaussian HMM on H4 features.
    Uses BIC to select optimal number of states (3-7).
    """

    def __init__(self, min_states: int = 3, max_states: int = 7, n_iter: int = 200,
                 random_state: int = 42, force_states: int = None):
        self.min_states = min_states
        self.max_states = max_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.force_states = force_states  # Override BIC selection
        self.best_model = None
        self.scaler = None
        self.feature_cols = None
        self.bic_scores = {}

    def train(self, features: pd.DataFrame, feature_cols: List[str],
              train_ratio: float = 0.8, raw_df: pd.DataFrame = None) -> Dict:
        """
        Train HMM with BIC model selection.

        Args:
            features: DataFrame with features + 'time' column
            feature_cols: List of feature column names
            train_ratio: Fraction for training (rest = out-of-sample)
            raw_df: Optional raw OHLCV DataFrame for move magnitude analysis

        Returns:
            Dict with training results and state analysis
        """
        from hmmlearn.hmm import GaussianHMM

        self.feature_cols = feature_cols

        # Split train/test
        split_idx = int(len(features) * train_ratio)
        train_df = features.iloc[:split_idx]
        test_df = features.iloc[split_idx:]

        print(f"\n{'='*60}")
        print(f"  HMM REGIME CLASSIFIER - TRAINING")
        print(f"{'='*60}")
        print(f"  Total bars: {len(features):,}")
        print(f"  Training:   {len(train_df):,} bars ({train_df['time'].iloc[0].strftime('%Y-%m-%d')} to {train_df['time'].iloc[-1].strftime('%Y-%m-%d')})")
        print(f"  Test:       {len(test_df):,} bars ({test_df['time'].iloc[0].strftime('%Y-%m-%d')} to {test_df['time'].iloc[-1].strftime('%Y-%m-%d')})")
        print(f"  Features:   {len(feature_cols)}")

        # Standardize features
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Clip extreme values (HMM sensitive to outliers)
        X_train_scaled = np.clip(X_train_scaled, -4, 4)
        X_test_scaled = np.clip(X_test_scaled, -4, 4)

        # BIC model selection (or forced state count)
        if self.force_states:
            state_range = [self.force_states]
            print(f"\n  Forced state count: {self.force_states}")
        else:
            state_range = range(self.min_states, self.max_states + 1)
            print(f"\n  BIC Model Selection (states {self.min_states}-{self.max_states}):")

        best_bic = np.inf
        best_n = None

        for n_states in state_range:
            try:
                # Use sticky transition prior to reduce oscillation
                # Higher kappa = more likely to stay in same state
                # Initialize with sticky transition matrix (diagonal bias)
                # This encourages the model to persist in states longer
                sticky_alpha = 10.0  # How sticky (higher = stickier)
                init_transmat = np.full((n_states, n_states), 1.0)
                np.fill_diagonal(init_transmat, sticky_alpha)
                init_transmat = init_transmat / init_transmat.sum(axis=1, keepdims=True)

                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type='full',
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                    verbose=False,
                    params='stmc',  # Train all: startprob, transmat, means, covars
                    init_params='smc'  # Don't re-init transmat (we set it manually)
                )
                model.transmat_ = init_transmat

                model.fit(X_train_scaled)

                # Calculate BIC
                log_likelihood = model.score(X_train_scaled)
                n_params = n_states * (n_states - 1) + n_states * len(feature_cols) * 2
                bic = -2 * log_likelihood * len(X_train_scaled) + n_params * np.log(len(X_train_scaled))

                # Also score on test set
                test_ll = model.score(X_test_scaled)

                self.bic_scores[n_states] = {
                    'bic': bic,
                    'train_ll': log_likelihood,
                    'test_ll': test_ll,
                    'converged': model.monitor_.converged
                }

                marker = ''
                if bic < best_bic:
                    best_bic = bic
                    best_n = n_states
                    self.best_model = model
                    marker = ' <-- BEST'

                # Calculate state persistence
                states_tmp = model.predict(X_train_scaled)
                changes = np.sum(np.diff(states_tmp) != 0)
                persistence = len(states_tmp) / (changes + 1)

                print(f"    {n_states} states: BIC={bic:,.0f} | Train LL={log_likelihood:.4f} | Test LL={test_ll:.4f} | Persistence={persistence:.1f} bars | Converged={model.monitor_.converged}{marker}")

            except Exception as e:
                print(f"    {n_states} states: FAILED ({e})")
                self.bic_scores[n_states] = {'error': str(e)}

        print(f"\n  Selected: {best_n} states (BIC={best_bic:,.0f})")

        # Decode states for full dataset
        X_full_scaled = self.scaler.transform(features[feature_cols].values)
        X_full_scaled = np.clip(X_full_scaled, -4, 4)
        states = self.best_model.predict(X_full_scaled)
        state_probs = self.best_model.predict_proba(X_full_scaled)

        # Analyse states
        analysis = self._analyse_states(features, feature_cols, states, state_probs, raw_df=raw_df)
        analysis['n_states'] = best_n
        analysis['bic_scores'] = self.bic_scores
        analysis['split_idx'] = split_idx

        # Out-of-sample validation
        test_states = states[split_idx:]
        test_probs = state_probs[split_idx:]
        oos = self._validate_oos(test_df, feature_cols, test_states, test_probs, analysis['state_profiles'])
        analysis['oos_validation'] = oos

        return analysis

    def _analyse_states(self, features: pd.DataFrame, feature_cols: List[str],
                        states: np.ndarray, state_probs: np.ndarray,
                        raw_df: pd.DataFrame = None) -> Dict:
        """Analyse each state cluster's characteristics."""
        df = features.copy()
        df['state'] = states
        df['state_confidence'] = state_probs.max(axis=1)

        n_states = self.best_model.n_components
        state_profiles = {}

        print(f"\n{'='*60}")
        print(f"  STATE CLUSTER ANALYSIS ({n_states} states)")
        print(f"{'='*60}")

        for state in range(n_states):
            mask = df['state'] == state
            state_data = df[mask]
            pct = len(state_data) / len(df) * 100

            profile = {
                'count': len(state_data),
                'pct': round(pct, 1),
                'avg_confidence': round(state_data['state_confidence'].mean(), 3),
            }

            # Feature means for this state
            for col in feature_cols:
                if col in state_data.columns:
                    profile[f'mean_{col}'] = round(state_data[col].mean(), 4)
                    profile[f'std_{col}'] = round(state_data[col].std(), 4)

            # Interpret the state
            regime, description = self._interpret_state(profile, feature_cols)
            profile['regime'] = regime
            profile['description'] = description

            state_profiles[state] = profile

            print(f"\n  State {state}: {regime}")
            print(f"  {'-'*50}")
            print(f"  Bars: {len(state_data):,} ({pct:.1f}%) | Confidence: {profile['avg_confidence']:.3f}")
            print(f"  Description: {description}")
            print(f"  Key features:")
            print(f"    Body ratio:    {profile.get('mean_body_ratio', 0):.3f}")
            print(f"    Wick ratio:    {profile.get('mean_wick_ratio', 0):.3f}")
            print(f"    ATR normalised:{profile.get('mean_atr_normalised', 0):.3f}")
            print(f"    ATR change:    {profile.get('mean_atr_change', 0):+.4f}")
            print(f"    Vol ratio:     {profile.get('mean_volatility_ratio', 0):.3f}")
            print(f"    |Slope| (4):   {profile.get('mean_abs_slope_4', 0):.4f}")
            print(f"    |Slope| (12):  {profile.get('mean_abs_slope_12', 0):.4f}")
            print(f"    Efficiency:    {profile.get('mean_efficiency_12', 0):.4f}")
            print(f"    Consistency:   {profile.get('mean_candle_consistency', 0):.3f}")
            print(f"    Volume ratio:  {profile.get('mean_volume_ratio', 0):.3f}")
            print(f"    HH/LL imbal:   {profile.get('mean_hh_ll_imbalance', 0):.3f}")
            print(f"    Range %tile:   {profile.get('mean_range_percentile', 0):.3f}")

        # Transition matrix
        trans_mat = self.best_model.transmat_
        print(f"\n  Transition Matrix:")
        header = "       " + "  ".join([f"  S{i}" for i in range(n_states)])
        print(f"  {header}")
        for i in range(n_states):
            row = f"  S{i}: " + "  ".join([f"{trans_mat[i,j]:.2f}" for j in range(n_states)])
            regime = state_profiles[i]['regime'][:12]
            print(f"  {row}  ({regime})")

        # --- Improvement A: Trend Cluster Analysis ---
        cluster_analysis = self._analyse_clusters(df, states, state_profiles)

        # --- Improvement C: Move Magnitude per Regime ---
        move_analysis = None
        if raw_df is not None:
            move_analysis = self._analyse_move_magnitude(df, states, state_profiles, raw_df)

        result = {
            'state_profiles': state_profiles,
            'transition_matrix': trans_mat.tolist(),
            'all_states': states.tolist(),
            'all_confidences': state_probs.max(axis=1).tolist(),
            'cluster_analysis': cluster_analysis,
        }
        if move_analysis:
            result['move_magnitude'] = move_analysis

        return result

    def _interpret_state(self, profile: Dict, feature_cols: List[str]) -> Tuple[str, str]:
        """
        Interpret a state cluster based on its feature means.
        Returns (regime_name, description).

        Uses a scoring system across multiple dimensions rather than hard cutoffs.
        Each state gets scored for each regime type, highest score wins.
        """
        body = profile.get('mean_body_ratio', 0.5)
        wick = profile.get('mean_wick_ratio', 0.5)
        atr_n = profile.get('mean_atr_normalised', 1.0)
        atr_chg = profile.get('mean_atr_change', 0)
        vol_ratio = profile.get('mean_volatility_ratio', 1.0)
        slope4 = profile.get('mean_abs_slope_4', 0)
        slope12 = profile.get('mean_abs_slope_12', 0)
        efficiency = profile.get('mean_efficiency_12', 0.1)
        consistency = profile.get('mean_candle_consistency', 0.5)
        volume = profile.get('mean_volume_ratio', 1.0)
        hh_ll = profile.get('mean_hh_ll_imbalance', 0.3)
        range_pct = profile.get('mean_range_percentile', 0.5)
        pct = profile.get('pct', 0)

        # Score each regime type (0-100 scale)
        scores = {}

        # VOLATILITY_SPIKE: Very high ATR, high volume, large range bars
        scores['VOLATILITY_SPIKE'] = (
            min(max((atr_n - 1.1) * 60, 0), 25) +
            min(max((vol_ratio - 1.2) * 40, 0), 25) +
            min(max((volume - 1.1) * 30, 0), 20) +
            min(max((range_pct - 0.7) * 50, 0), 15) +
            min(max(atr_chg * 80, 0), 15)
        )

        # TREND_EXPANSION: High efficiency, strong slope, big bodies, consistent direction
        # Lowered thresholds to capture slower, sustained trends (not just fast spikes)
        # Multi-candle slope (slope12) is weighted more than single-candle (slope4)
        scores['TREND_EXPANSION'] = (
            min(max((efficiency - 0.12) * 250, 0), 30) +       # Path efficiency (lowered from 0.15)
            min(max((slope12 - 0.14) * 130, 0), 25) +          # Sustained slope (lowered from 0.18)
            min(slope4 * 50, 12) +                              # Short-term slope (reduced weight)
            min(max((body - 0.38) * 50, 0), 10) +              # Body-dominant candles (slightly lower)
            min(max((consistency - 0.58) * 45, 0), 12) +       # Directional consistency (slightly lower)
            min(max((hh_ll - 0.28) * 45, 0), 11)               # HH/LL structure (slightly lower)
        )

        # COMPRESSION: Low ATR, small bodies, tight range, low volume
        scores['COMPRESSION'] = (
            min(max((1.0 - atr_n) * 60, 0), 25) +
            min(max((1.0 - vol_ratio) * 50, 0), 25) +
            min(max((0.45 - body) * 60, 0), 15) +
            min(max((0.4 - range_pct) * 50, 0), 15) +
            min(max((1.0 - volume) * 40, 0), 10) +
            min(max(-atr_chg * 60, 0), 10)
        )

        # MEAN_REVERSION: Low efficiency, flat slope, mixed direction, wicks dominate
        scores['MEAN_REVERSION'] = (
            min(max((0.10 - efficiency) * 400, 0), 30) +      # Very low path efficiency
            min(max((0.10 - slope12) * 250, 0), 25) +         # Very flat sustained slope
            min(max((wick - 0.45) * 80, 0), 15) +             # Wicks dominate
            min(max((0.62 - consistency) * 80, 0), 15) +      # Mixed direction
            min(max((0.25 - hh_ll) * 60, 0), 10) +            # No HH/LL structure
            min(max((0.45 - body) * 40, 0), 5)                # Smaller bodies
        )

        # TREND_EXHAUSTION: Was trending (some slope), but efficiency dropping, wicks growing
        scores['TREND_EXHAUSTION'] = (
            min(slope12 * 80, 20) +                            # Still some slope
            min(max((wick - 0.4) * 60, 0), 20) +              # Growing wicks
            min(max(-atr_chg * 100, 0), 20) +                 # ATR contracting
            min(max((0.15 - efficiency) * 200, 0), 15) +      # Efficiency dropping
            min(max((0.55 - body) * 50, 0), 15) +             # Bodies shrinking
            min(max((1.0 - vol_ratio) * 20, 0), 10)           # Volatility fading
        )

        # TRANSITIONAL: Baseline score other regimes must beat
        scores['TRANSITIONAL'] = 18

        best_regime = max(scores, key=scores.get)

        # Generate description using actual feature names
        descriptions = {
            'VOLATILITY_SPIKE': f"ATR {atr_n:.2f}x, vol_ratio={vol_ratio:.2f}, volume={volume:.2f} -- event-driven",
            'TREND_EXPANSION': f"Efficiency={efficiency:.3f}, |slope12|={slope12:.3f}, body={body:.2f}, consistency={consistency:.2f}",
            'COMPRESSION': f"Low ATR ({atr_n:.2f}x), range_pct={range_pct:.2f}, small bodies={body:.2f} -- pre-breakout",
            'MEAN_REVERSION': f"Low efficiency ({efficiency:.3f}), |slope12|={slope12:.3f}, wicks={wick:.2f}, mixed dir={consistency:.2f}",
            'TREND_EXHAUSTION': f"|slope12|={slope12:.3f} but wicks={wick:.2f}, ATR change={atr_chg:+.3f}, efficiency={efficiency:.3f}",
            'TRANSITIONAL': f"Mixed (top: {', '.join(f'{k[:5]}={v:.0f}' for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3])})",
        }

        return best_regime, descriptions.get(best_regime, "Unknown")

    def _validate_oos(self, test_df: pd.DataFrame, feature_cols: List[str],
                      test_states: np.ndarray, test_probs: np.ndarray,
                      state_profiles: Dict) -> Dict:
        """Validate out-of-sample state distribution."""
        print(f"\n{'='*60}")
        print(f"  OUT-OF-SAMPLE VALIDATION")
        print(f"{'='*60}")

        n_states = self.best_model.n_components

        # State distribution in test set
        oos_dist = {}
        for state in range(n_states):
            count = int(np.sum(test_states == state))
            pct = count / len(test_states) * 100
            regime = state_profiles[state]['regime']
            train_pct = state_profiles[state]['pct']
            drift = abs(pct - train_pct)
            oos_dist[state] = {
                'count': count,
                'pct': round(pct, 1),
                'train_pct': train_pct,
                'drift': round(drift, 1),
                'regime': regime
            }
            drift_flag = " !! DRIFT" if drift > 10 else ""
            print(f"  State {state} ({regime:20s}): {pct:5.1f}% (train: {train_pct:.1f}%) drift={drift:.1f}%{drift_flag}")

        # Average confidence
        avg_conf = float(test_probs.max(axis=1).mean())
        low_conf = float(np.sum(test_probs.max(axis=1) < 0.5)) / len(test_probs) * 100
        print(f"\n  Avg confidence: {avg_conf:.3f}")
        print(f"  Low confidence (<50%): {low_conf:.1f}% of bars")

        # Regime persistence (avg consecutive bars in same state)
        changes = np.sum(np.diff(test_states) != 0)
        avg_persistence = len(test_states) / (changes + 1)
        print(f"  State changes: {changes} ({changes/len(test_states)*100:.1f}% of bars)")
        print(f"  Avg persistence: {avg_persistence:.1f} bars ({avg_persistence*4:.0f} hours)")

        return {
            'distribution': oos_dist,
            'avg_confidence': avg_conf,
            'low_confidence_pct': low_conf,
            'state_changes': int(changes),
            'avg_persistence_bars': round(avg_persistence, 1)
        }

    def _analyse_clusters(self, df: pd.DataFrame, states: np.ndarray,
                          state_profiles: Dict) -> Dict:
        """
        Improvement A: Analyse consecutive H4 sequences (clusters/episodes).

        Instead of counting individual candles, groups consecutive same-state
        bars into episodes. A trend is better measured as "12 consecutive H4 bars
        in TREND_EXPANSION = 48 hours of sustained trending" rather than
        individual bar counts.

        Returns dict with cluster statistics per state.
        """
        print(f"\n{'='*60}")
        print(f"  CLUSTER ANALYSIS (consecutive H4 sequences)")
        print(f"{'='*60}")

        n_states = self.best_model.n_components
        cluster_stats = {}

        for state in range(n_states):
            regime = state_profiles[state]['regime']

            # Find consecutive runs of this state
            runs = []
            current_run = 0
            for s in states:
                if s == state:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)

            if not runs:
                cluster_stats[state] = {
                    'regime': regime,
                    'num_episodes': 0,
                    'avg_duration_bars': 0,
                    'avg_duration_hours': 0,
                    'max_duration_bars': 0,
                    'max_duration_hours': 0,
                    'median_duration_bars': 0,
                    'pct_in_long_episodes': 0,
                    'episodes_over_6_bars': 0,
                }
                continue

            runs_arr = np.array(runs)
            total_bars = int(runs_arr.sum())
            long_episodes = runs_arr[runs_arr >= 6]  # 6+ bars = 24+ hours
            bars_in_long = int(long_episodes.sum()) if len(long_episodes) > 0 else 0

            stats = {
                'regime': regime,
                'num_episodes': len(runs),
                'avg_duration_bars': round(float(runs_arr.mean()), 1),
                'avg_duration_hours': round(float(runs_arr.mean()) * 4, 0),
                'max_duration_bars': int(runs_arr.max()),
                'max_duration_hours': int(runs_arr.max()) * 4,
                'median_duration_bars': round(float(np.median(runs_arr)), 1),
                'pct_in_long_episodes': round(bars_in_long / total_bars * 100, 1) if total_bars > 0 else 0,
                'episodes_over_6_bars': int(len(long_episodes)),
                'duration_distribution': {
                    '1_bar': int(np.sum(runs_arr == 1)),
                    '2_3_bars': int(np.sum((runs_arr >= 2) & (runs_arr <= 3))),
                    '4_6_bars': int(np.sum((runs_arr >= 4) & (runs_arr <= 6))),
                    '7_12_bars': int(np.sum((runs_arr >= 7) & (runs_arr <= 12))),
                    '13_plus_bars': int(np.sum(runs_arr >= 13)),
                }
            }
            cluster_stats[state] = stats

            print(f"\n  State {state} ({regime}):")
            print(f"    Episodes: {len(runs)} | Avg: {stats['avg_duration_bars']:.1f} bars ({stats['avg_duration_hours']:.0f}h)")
            print(f"    Max: {stats['max_duration_bars']} bars ({stats['max_duration_hours']}h) | Median: {stats['median_duration_bars']:.1f} bars")
            print(f"    Long episodes (24h+): {stats['episodes_over_6_bars']} ({stats['pct_in_long_episodes']:.1f}% of bars)")
            dd = stats['duration_distribution']
            print(f"    Distribution: 1bar={dd['1_bar']} | 2-3={dd['2_3_bars']} | 4-6={dd['4_6_bars']} | 7-12={dd['7_12_bars']} | 13+={dd['13_plus_bars']}")

        return cluster_stats

    def _analyse_move_magnitude(self, df: pd.DataFrame, states: np.ndarray,
                                state_profiles: Dict, raw_df: pd.DataFrame) -> Dict:
        """
        Improvement C: Calculate net directional move (pips) per regime state.

        Validates the hypothesis: "3% of trend candles produce 80-90% of PnL."
        For each regime state, calculates:
        - Total net directional move (sum of |close - open| in pips)
        - Average move per bar
        - Percentage of total market movement attributed to each state

        Uses raw OHLCV data aligned by time with the feature DataFrame.
        """
        print(f"\n{'='*60}")
        print(f"  MOVE MAGNITUDE ANALYSIS (pip moves per regime)")
        print(f"{'='*60}")

        # Align raw data with features by time
        pip_size = 0.0001  # GBPUSD

        # Merge feature times with raw close prices
        aligned = df[['time']].copy()
        aligned['state'] = states

        # Match raw_df by time
        raw_lookup = raw_df.set_index('time')[['open', 'close', 'high', 'low']].copy()
        aligned = aligned.merge(raw_lookup, left_on='time', right_index=True, how='left')
        aligned = aligned.dropna(subset=['close'])

        if len(aligned) == 0:
            print("  [WARN] No aligned data for move magnitude analysis")
            return {}

        # Calculate per-bar directional move (in pips)
        aligned['body_pips'] = abs(aligned['close'] - aligned['open']) / pip_size
        aligned['range_pips'] = (aligned['high'] - aligned['low']) / pip_size

        # Calculate close-to-close moves for net displacement
        aligned['net_move_pips'] = (aligned['close'] - aligned['close'].shift(1)).abs() / pip_size
        aligned = aligned.dropna(subset=['net_move_pips'])

        n_states = self.best_model.n_components
        move_stats = {}
        total_body_pips = aligned['body_pips'].sum()
        total_net_pips = aligned['net_move_pips'].sum()
        total_bars = len(aligned)

        for state in range(n_states):
            regime = state_profiles[state]['regime']
            mask = aligned['state'] == state
            state_data = aligned[mask]

            if len(state_data) == 0:
                continue

            state_body = state_data['body_pips'].sum()
            state_net = state_data['net_move_pips'].sum()
            state_range = state_data['range_pips'].sum()
            bars_pct = len(state_data) / total_bars * 100

            stats = {
                'regime': regime,
                'bars': len(state_data),
                'bars_pct': round(bars_pct, 1),
                'total_body_pips': round(float(state_body), 1),
                'total_net_move_pips': round(float(state_net), 1),
                'avg_body_pips': round(float(state_data['body_pips'].mean()), 1),
                'avg_net_move_pips': round(float(state_data['net_move_pips'].mean()), 1),
                'avg_range_pips': round(float(state_data['range_pips'].mean()), 1),
                'pct_of_total_body': round(float(state_body / total_body_pips * 100), 1) if total_body_pips > 0 else 0,
                'pct_of_total_net': round(float(state_net / total_net_pips * 100), 1) if total_net_pips > 0 else 0,
                'move_concentration': round(float(state_body / total_body_pips * 100) / bars_pct, 2) if bars_pct > 0 and total_body_pips > 0 else 0,
            }
            move_stats[state] = stats

            # Move concentration > 1 = this state produces MORE than its fair share of movement
            conc_marker = " ** HIGH" if stats['move_concentration'] > 1.3 else ""
            print(f"\n  State {state} ({regime}):")
            print(f"    Bars: {len(state_data):,} ({bars_pct:.1f}%)")
            print(f"    Body pips: {state_body:,.0f} ({stats['pct_of_total_body']:.1f}% of total) | Avg: {stats['avg_body_pips']:.1f}/bar")
            print(f"    Net move:  {state_net:,.0f} ({stats['pct_of_total_net']:.1f}% of total) | Avg: {stats['avg_net_move_pips']:.1f}/bar")
            print(f"    Concentration: {stats['move_concentration']:.2f}x (pct move / pct bars){conc_marker}")

        # Summary: validate "3% trend = 80% PnL" hypothesis
        print(f"\n  --- Move Concentration Summary ---")
        for state in sorted(move_stats.keys(), key=lambda s: move_stats[s]['move_concentration'], reverse=True):
            s = move_stats[state]
            print(f"  {s['regime']:20s}: {s['bars_pct']:5.1f}% of bars -> {s['pct_of_total_body']:5.1f}% of body movement ({s['move_concentration']:.2f}x)")

        return move_stats

    def save(self, symbol: str = 'GBPUSD'):
        """Save trained model, scaler, and state map."""
        if self.best_model is None:
            raise ValueError("No trained model to save")

        paths = _symbol_paths(symbol)

        with open(paths['model'], 'wb') as f:
            pickle.dump(self.best_model, f)

        with open(paths['scaler'], 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_cols': self.feature_cols
            }, f)

        # Also write to legacy paths for GBPUSD backward compat
        if symbol.upper() == 'GBPUSD':
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(self.best_model, f)
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'feature_cols': self.feature_cols
                }, f)

        print(f"\n  Model saved to: {paths['model']}")
        print(f"  Scaler saved to: {paths['scaler']}")

    @staticmethod
    def load(symbol: str = 'GBPUSD') -> Tuple:
        """Load trained model and scaler for a symbol."""
        paths = _symbol_paths(symbol)

        # Try per-symbol path first, fall back to legacy for GBPUSD
        model_path = paths['model']
        scaler_path = paths['scaler']
        if not model_path.exists() and symbol.upper() == 'GBPUSD':
            model_path = MODEL_FILE
            scaler_path = SCALER_FILE

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)

        return model, scaler_data['scaler'], scaler_data['feature_cols']


# ============================================================================
# REGIME PREDICTOR (called by bot)
# ============================================================================

class RegimePredictor:
    """
    Predicts current H4 regime from live MT5 data.
    Writes per-symbol regime_state_{SYMBOL}.json for bot consumption.
    Supports multiple symbols with separate models.
    """

    def __init__(self):
        self.models = {}       # {symbol: model}
        self.scalers = {}      # {symbol: scaler}
        self.feature_cols_map = {}  # {symbol: feature_cols}
        self.state_maps = {}   # {symbol: state_map}
        self.feature_engine = H4FeatureEngine()
        self._load_all()

    def _load_all(self):
        """Load models for all available symbols."""
        for symbol in SUPPORTED_SYMBOLS:
            self._load_symbol(symbol)

    def _load_symbol(self, symbol: str):
        """Load model, scaler, and state map for one symbol."""
        try:
            model, scaler, feature_cols = RegimeHMMTrainer.load(symbol)
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.feature_cols_map[symbol] = feature_cols

            paths = _symbol_paths(symbol)
            state_map_file = paths['state_map']
            # Fall back to legacy for GBPUSD
            if not state_map_file.exists() and symbol == 'GBPUSD':
                state_map_file = STATE_MAP_FILE

            if state_map_file.exists():
                with open(state_map_file, 'r') as f:
                    self.state_maps[symbol] = json.load(f)

            print(f"[OK] Regime HMM model loaded for {symbol}")
        except FileNotFoundError:
            print(f"[WARN] Regime HMM model not found for {symbol} - run --train --symbol {symbol}")

    def predict_current(self, symbol: str = 'GBPUSD') -> Optional[Dict]:
        """
        Predict current H4 regime from live MT5 data.

        Returns dict with regime, confidence, allowed strategies, etc.
        """
        if symbol not in self.models:
            print(f"[WARN] No model for {symbol}")
            return None

        model = self.models[symbol]
        scaler = self.scalers[symbol]
        feature_cols = self.feature_cols_map[symbol]
        state_map = self.state_maps.get(symbol)

        try:
            import MetaTrader5 as mt5
            mt5.initialize()

            # Get last 120 H4 bars (need 50+14+12=76 minimum for rolling features)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 120)
            mt5.shutdown()

            if rates is None or len(rates) < 80:
                print(f"[WARN] Insufficient H4 data for {symbol} regime prediction")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Build features
            features, _ = self.feature_engine.build_features(df)
            if len(features) == 0:
                return None

            # Take last row (current bar)
            X = features[feature_cols].iloc[[-1]].values
            X_scaled = scaler.transform(X)
            X_scaled = np.clip(X_scaled, -4, 4)

            # Predict
            state = int(model.predict(X_scaled)[0])
            probs = model.predict_proba(X_scaled)[0]
            confidence = float(probs.max())

            # Get regime info
            if state_map and str(state) in state_map:
                state_info = state_map[str(state)]
                regime = state_info.get('regime', 'UNKNOWN')
                description = state_info.get('description', '')
                allowed = state_info.get('allowed_strategies', ['MR', 'BO'])
            else:
                regime = 'UNKNOWN'
                description = 'State map not loaded'
                allowed = ['MR', 'BO']

            # Strategy mapping
            strategy_map = {
                'COMPRESSION':      {'allowed': ['BO'], 'size_mult': 0.5, 'note': 'Wait for breakout confirmation'},
                'TREND_EXPANSION':  {'allowed': ['BO'], 'size_mult': 1.0, 'note': 'Strong trend, BO preferred'},
                'TREND_EXHAUSTION': {'allowed': ['MR'], 'size_mult': 0.75, 'note': 'Fading trend, MR on edges only'},
                'MEAN_REVERSION':   {'allowed': ['MR'], 'size_mult': 1.0, 'note': 'Ranging market, MR territory'},
                'VOLATILITY_SPIKE': {'allowed': [],      'size_mult': 0.25, 'note': 'High risk, reduce or avoid'},
                'TRANSITIONAL':     {'allowed': ['MR', 'BO'], 'size_mult': 0.5, 'note': 'Mixed signals, reduce size'},
            }

            mapping = strategy_map.get(regime, {'allowed': ['MR', 'BO'], 'size_mult': 0.5, 'note': 'Unknown'})

            # Build current bar feature summary
            current_features = {}
            for col in feature_cols:
                current_features[col] = round(float(features[col].iloc[-1]), 4)

            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'h4_bar_time': features['time'].iloc[-1].isoformat(),
                'state': state,
                'regime': regime,
                'confidence': round(confidence, 4),
                'all_probs': {str(i): round(float(p), 4) for i, p in enumerate(probs)},
                'description': description,
                'allowed_strategies': mapping['allowed'],
                'size_multiplier': mapping['size_mult'],
                'note': mapping['note'],
                'current_features': current_features,
            }

            # Save to per-symbol file for bot consumption
            paths = _symbol_paths(symbol)
            with open(paths['regime_state'], 'w') as f:
                json.dump(result, f, indent=2, default=str)

            # Also write legacy file for GBPUSD backward compat
            if symbol == 'GBPUSD':
                with open(REGIME_STATE_FILE, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

            return result

        except Exception as e:
            print(f"[ERROR] Regime prediction failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_regime(self, result: Dict):
        """Print regime status to console (for bot integration)."""
        if result is None:
            print("[REGIME] No prediction available")
            return

        regime = result['regime']
        confidence = result['confidence']
        allowed = result['allowed_strategies']

        # Colour-coded regime indicator
        regime_icons = {
            'COMPRESSION':      '[==]',
            'TREND_EXPANSION':  '[>>]',
            'TREND_EXHAUSTION': '[><]',
            'MEAN_REVERSION':   '[<>]',
            'VOLATILITY_SPIKE': '[!!]',
            'TRANSITIONAL':     '[??]',
        }
        icon = regime_icons.get(regime, '[??]')
        strat_str = '+'.join(allowed) if allowed else 'NONE'

        print(f"\n  {icon} REGIME: {regime} ({confidence:.0%} conf)")
        print(f"      Allowed: {strat_str} | Size: {result['size_multiplier']:.0%}")
        print(f"      {result['note']}")


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def pull_data(symbol: str = 'GBPUSD', years: int = 8) -> pd.DataFrame:
    """Pull H4 data from MT5."""
    import MetaTrader5 as mt5
    mt5.initialize()

    start = datetime.now() - timedelta(days=years * 365)
    end = datetime.now()

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H4, start, end)
    mt5.shutdown()

    if rates is None:
        raise ValueError(f"No data for {symbol}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def run_training(symbol: str = 'GBPUSD'):
    """Full training pipeline for one symbol."""
    paths = _symbol_paths(symbol)

    print(f"\n  Pulling H4 {symbol} data...")
    df = pull_data(symbol)
    print(f"  Got {len(df):,} bars ({df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')})")

    # Build features
    engine = H4FeatureEngine()
    features, feature_cols = engine.build_features(df)
    print(f"  Built {len(feature_cols)} features for {len(features):,} bars")

    # Train HMM (force 5 states: compression, trend, exhaustion, MR, volatility)
    trainer = RegimeHMMTrainer(min_states=3, max_states=7, n_iter=300, force_states=5)
    analysis = trainer.train(features, feature_cols, train_ratio=0.8, raw_df=df)

    # Save model (per-symbol)
    trainer.save(symbol)

    # Save state map (per-symbol)
    state_map = {}
    for state, profile in analysis['state_profiles'].items():
        state_map[str(state)] = {
            'regime': profile['regime'],
            'description': profile['description'],
            'pct': profile['pct'],
            'allowed_strategies': {
                'COMPRESSION':      ['BO'],
                'TREND_EXPANSION':  ['BO'],
                'TREND_EXHAUSTION': ['MR'],
                'MEAN_REVERSION':   ['MR'],
                'VOLATILITY_SPIKE': [],
                'TRANSITIONAL':     ['MR', 'BO'],
            }.get(profile['regime'], ['MR', 'BO'])
        }

    with open(paths['state_map'], 'w') as f:
        json.dump(state_map, f, indent=2)
    # Legacy compat for GBPUSD
    if symbol.upper() == 'GBPUSD':
        with open(STATE_MAP_FILE, 'w') as f:
            json.dump(state_map, f, indent=2)
    print(f"  State map saved to: {paths['state_map']}")

    # Save full analysis (per-symbol)
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(paths['analysis'], 'w') as f:
        json.dump(analysis, f, indent=2, default=convert)
    print(f"  Analysis saved to: {paths['analysis']}")

    # Save training metadata (for retrain scheduling)
    train_meta = {
        'symbol': symbol,
        'trained_at': datetime.utcnow().isoformat(),
        'bars_used': len(features),
        'data_range': f"{df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}",
        'n_states': analysis.get('n_states', 5),
    }
    with open(paths['train_meta'], 'w') as f:
        json.dump(train_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE - {symbol}")
    print(f"{'='*60}")

    return analysis


def run_training_all():
    """Train regime models for ALL supported symbols."""
    print(f"\n{'='*60}")
    print(f"  TRAINING ALL SYMBOLS: {', '.join(SUPPORTED_SYMBOLS)}")
    print(f"{'='*60}")

    results = {}
    for symbol in SUPPORTED_SYMBOLS:
        try:
            results[symbol] = run_training(symbol)
        except Exception as e:
            print(f"\n  [ERROR] Training failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = None

    # Summary
    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"{'='*60}")
    for sym, result in results.items():
        if result:
            states = result.get('n_states', '?')
            print(f"  {sym}: {states} states trained OK")
        else:
            print(f"  {sym}: FAILED")

    return results


def run_prediction(symbol: str = 'GBPUSD'):
    """Predict current regime for one symbol."""
    predictor = RegimePredictor()
    result = predictor.predict_current(symbol)
    predictor.print_regime(result)
    return result


def run_prediction_all():
    """Predict current regime for ALL supported symbols."""
    predictor = RegimePredictor()
    results = {}
    for symbol in SUPPORTED_SYMBOLS:
        result = predictor.predict_current(symbol)
        if result:
            predictor.print_regime(result)
        results[symbol] = result
    return results


def check_retrain_needed() -> List[str]:
    """
    Check which symbols need retraining based on RETRAIN_INTERVAL_DAYS.
    Returns list of symbols needing retrain.
    """
    needs_retrain = []

    for symbol in SUPPORTED_SYMBOLS:
        paths = _symbol_paths(symbol)

        # No model at all -> needs training
        if not paths['model'].exists():
            print(f"  {symbol}: No model found - needs training")
            needs_retrain.append(symbol)
            continue

        # Check training metadata age
        if paths['train_meta'].exists():
            try:
                with open(paths['train_meta'], 'r') as f:
                    meta = json.load(f)
                trained_at = datetime.fromisoformat(meta['trained_at'])
                age_days = (datetime.utcnow() - trained_at).total_seconds() / 86400
                if age_days >= RETRAIN_INTERVAL_DAYS:
                    print(f"  {symbol}: Last trained {age_days:.1f} days ago (threshold: {RETRAIN_INTERVAL_DAYS}d) - needs retrain")
                    needs_retrain.append(symbol)
                else:
                    print(f"  {symbol}: Last trained {age_days:.1f} days ago - OK")
            except Exception:
                print(f"  {symbol}: Cannot read training metadata - needs retrain")
                needs_retrain.append(symbol)
        else:
            print(f"  {symbol}: No training metadata - needs retrain")
            needs_retrain.append(symbol)

    return needs_retrain


def run_auto_retrain():
    """
    Auto-retrain symbols that are overdue. Called by scheduler.
    Retrains only what's needed, then refreshes predictions.
    """
    print(f"\n{'='*60}")
    print(f"  AUTO RETRAIN CHECK ({datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC)")
    print(f"{'='*60}")

    symbols_to_retrain = check_retrain_needed()

    if not symbols_to_retrain:
        print(f"\n  All models up to date. Refreshing predictions...")
        run_prediction_all()
        return

    print(f"\n  Retraining: {', '.join(symbols_to_retrain)}")
    for symbol in symbols_to_retrain:
        try:
            run_training(symbol)
        except Exception as e:
            print(f"  [ERROR] Retrain failed for {symbol}: {e}")

    # Refresh predictions for all symbols after retrain
    print(f"\n  Refreshing predictions...")
    run_prediction_all()


def run_analysis(symbol: str = 'GBPUSD'):
    """Load and display saved analysis for a symbol."""
    paths = _symbol_paths(symbol)
    analysis_file = paths['analysis']

    # Fall back to legacy for GBPUSD
    if not analysis_file.exists() and symbol == 'GBPUSD':
        analysis_file = OUTPUT_DIR / 'regime_analysis.json'

    if not analysis_file.exists():
        print(f"No analysis found for {symbol}. Run --train --symbol {symbol} first.")
        return

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    print(f"\n{'='*60}")
    print(f"  SAVED REGIME ANALYSIS - {symbol}")
    print(f"{'='*60}")

    for state, profile in analysis['state_profiles'].items():
        print(f"\n  State {state}: {profile['regime']}")
        print(f"  Bars: {profile['count']:,} ({profile['pct']:.1f}%)")
        print(f"  {profile['description']}")

    if 'oos_validation' in analysis:
        oos = analysis['oos_validation']
        print(f"\n  OOS Validation:")
        print(f"  Avg confidence: {oos['avg_confidence']:.3f}")
        print(f"  Avg persistence: {oos['avg_persistence_bars']:.1f} bars")

    if 'cluster_analysis' in analysis:
        print(f"\n  Cluster Analysis:")
        for state, stats in analysis['cluster_analysis'].items():
            regime = stats.get('regime', '?')
            avg = stats.get('avg_duration_bars', 0)
            mx = stats.get('max_duration_bars', 0)
            print(f"    S{state} ({regime}): avg={avg:.1f} bars, max={mx} bars")

    if 'move_magnitude' in analysis:
        print(f"\n  Move Concentration:")
        for state in sorted(analysis['move_magnitude'].keys(),
                          key=lambda s: analysis['move_magnitude'][s].get('move_concentration', 0),
                          reverse=True):
            s = analysis['move_magnitude'][state]
            print(f"    {s['regime']:20s}: {s['bars_pct']:5.1f}% bars -> {s['pct_of_total_body']:5.1f}% movement ({s['move_concentration']:.2f}x)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='H4 Market Regime HMM Classifier')
    parser.add_argument('--train', action='store_true', help='Train model for one symbol')
    parser.add_argument('--train-all', action='store_true', help='Train models for ALL symbols')
    parser.add_argument('--predict', action='store_true', help='Predict regime for one symbol')
    parser.add_argument('--predict-all', action='store_true', help='Predict regime for ALL symbols')
    parser.add_argument('--analyse', action='store_true', help='Show saved analysis')
    parser.add_argument('--auto-retrain', action='store_true', help='Check & retrain overdue models')
    parser.add_argument('--symbol', default='GBPUSD', help='Symbol (default: GBPUSD)')

    args = parser.parse_args()

    if args.train_all:
        run_training_all()
    elif args.train:
        run_training(args.symbol)
    elif args.predict_all:
        run_prediction_all()
    elif args.predict:
        run_prediction(args.symbol)
    elif args.auto_retrain:
        run_auto_retrain()
    elif args.analyse:
        run_analysis(args.symbol)
    else:
        # Default: auto-retrain check (trains if needed, predicts all)
        run_auto_retrain()
