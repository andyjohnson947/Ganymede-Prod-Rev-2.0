"""
ML-Driven Confluence Weights Optimizer
Analyzes feature importance and updates confluence weights dynamically
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class ConfluenceWeightsOptimizer:
    """
    Analyzes ML feature importance and recommends optimal confluence weights.
    Writes recommendations to JSON file for strategy to read.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize weights optimizer

        Args:
            output_dir: Directory to write weights JSON (default: ml_system/output)
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / 'ml_system' / 'output'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.weights_file = self.output_dir / 'confluence_weights_optimized.json'

    def analyze_feature_importance(self, feature_importance_df: pd.DataFrame) -> Dict[str, float]:
        """
        Convert feature importance to confluence weights.

        Args:
            feature_importance_df: DataFrame with 'feature' and 'importance' columns

        Returns:
            Dict mapping confluence factors to weights
        """
        if feature_importance_df is None or len(feature_importance_df) == 0:
            return {}

        # Map ML features to confluence factors
        # Look for features that match confluence factor names
        confluence_weights = {}

        # Known confluence factor patterns
        confluence_patterns = {
            'vwap_band_1': ['vwap_band_1', 'vwap_below_band1', 'vwap_above_band1'],
            'vwap_band_2': ['vwap_band_2', 'vwap_below_band2', 'vwap_above_band2'],
            'volume_profile_poi': ['volume_profile_poi', 'vol_profile_poi', 'vp_poi'],
            'htf_support_resist': ['htf_support', 'htf_resistance', 'htf_level'],
            'fvg_present': ['fvg_present', 'fvg_gap', 'fair_value_gap'],
            'wick_rejection': ['wick_rejection', 'wick_size', 'rejection_wick'],
            'candlestick_pattern': ['candlestick_pattern', 'candle_pattern'],
            'adx_filter': ['adx', 'adx_value'],
            'session_alignment': ['session', 'trading_session'],
            'spread_acceptable': ['spread', 'spread_pips'],
        }

        # Normalize importance to 0-10 scale (confluence weights typically 0-3)
        max_importance = feature_importance_df['importance'].max()
        if max_importance == 0:
            return {}

        # Find matching features and aggregate importance
        for factor, patterns in confluence_patterns.items():
            total_importance = 0
            count = 0

            for pattern in patterns:
                matching = feature_importance_df[
                    feature_importance_df['feature'].str.contains(pattern, case=False, na=False)
                ]
                if len(matching) > 0:
                    total_importance += matching['importance'].sum()
                    count += len(matching)

            if count > 0:
                # Average importance, then scale to 0-3 range
                avg_importance = total_importance / count
                weight = (avg_importance / max_importance) * 3.0
                # Round to 1 decimal place
                confluence_weights[factor] = round(weight, 1)

        return confluence_weights

    def update_weights(self, feature_importance_path: str = None, min_features: int = 10):
        """
        Read feature importance and update confluence weights file.

        Args:
            feature_importance_path: Path to feature_importance CSV
            min_features: Minimum features required to update (safety check)

        Returns:
            Dict with updated weights or None if update failed
        """
        # Load feature importance
        if feature_importance_path is None:
            project_root = Path(__file__).parent.parent
            feature_importance_path = project_root / 'ml_system' / 'models' / 'feature_importance_baseline.csv'

        importance_path = Path(feature_importance_path)
        if not importance_path.exists():
            print(f"[WEIGHTS] Feature importance file not found: {importance_path}")
            return None

        try:
            feature_importance_df = pd.read_csv(importance_path)
        except Exception as e:
            print(f"[WEIGHTS] Failed to read feature importance: {e}")
            return None

        if len(feature_importance_df) < min_features:
            print(f"[WEIGHTS] Too few features ({len(feature_importance_df)}) to update weights")
            return None

        # Analyze and generate weights
        optimized_weights = self.analyze_feature_importance(feature_importance_df)

        if len(optimized_weights) == 0:
            print("[WEIGHTS] No confluence factors found in feature importance")
            return None

        # Prepare output structure
        weights_data = {
            'last_updated': datetime.now().isoformat(),
            'source': str(feature_importance_path),
            'total_features_analyzed': len(feature_importance_df),
            'confluence_factors_updated': len(optimized_weights),
            'weights': optimized_weights,
            'note': 'These weights override strategy_config.py CONFLUENCE_WEIGHTS when present'
        }

        # Write to JSON file (UTF-8 encoding to avoid issues)
        try:
            with open(self.weights_file, 'w', encoding='utf-8') as f:
                json.dump(weights_data, f, indent=2, ensure_ascii=False)

            print(f"[WEIGHTS] Updated confluence weights: {self.weights_file}")
            print(f"[WEIGHTS] Optimized {len(optimized_weights)} factors from {len(feature_importance_df)} features")

            return weights_data

        except Exception as e:
            print(f"[WEIGHTS] Failed to write weights file: {e}")
            return None

    def get_current_weights(self) -> Optional[Dict[str, float]]:
        """
        Read current optimized weights from file.

        Returns:
            Dict of weights or None if file doesn't exist
        """
        if not self.weights_file.exists():
            return None

        try:
            with open(self.weights_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('weights', {})
        except Exception as e:
            print(f"[WEIGHTS] Failed to read weights file: {e}")
            return None


def load_optimized_weights(weights_dir: str = None) -> Optional[Dict[str, float]]:
    """
    Convenience function to load optimized weights.
    Strategy can use this to override default weights.

    Args:
        weights_dir: Directory containing confluence_weights_optimized.json

    Returns:
        Dict of weights or None if file doesn't exist
    """
    if weights_dir is None:
        project_root = Path(__file__).parent.parent
        weights_dir = project_root / 'ml_system' / 'output'

    weights_file = Path(weights_dir) / 'confluence_weights_optimized.json'

    if not weights_file.exists():
        return None

    try:
        with open(weights_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('weights', {})
    except Exception as e:
        print(f"[WEIGHTS] Failed to load optimized weights: {e}")
        return None


if __name__ == '__main__':
    # Test the optimizer
    optimizer = ConfluenceWeightsOptimizer()
    result = optimizer.update_weights()

    if result:
        print("\n=== OPTIMIZED WEIGHTS ===")
        for factor, weight in result['weights'].items():
            print(f"  {factor}: {weight}")
    else:
        print("[ERROR] Failed to generate optimized weights")
