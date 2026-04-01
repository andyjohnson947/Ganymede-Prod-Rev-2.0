#!/usr/bin/env python3
"""
Market Environment Clustering (Phase 3)

Clusters market state snapshots into recurring environment types using k-means
on one-hot encoded categorical features. Each cluster represents a distinct
market environment with characteristic behaviour patterns.

Usage:
    python environment_clustering.py                  # Cluster all symbols
    python environment_clustering.py --symbol EURUSD  # Cluster one symbol
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ml_system.market_state.config import (
    SYMBOLS, MODEL_DIR, DEFAULT_N_CLUSTERS, CLUSTER_K_RANGE,
    MIN_CLUSTER_SAMPLES,
)
from ml_system.market_state.feature_buckets import FeatureBucketizer


class EnvironmentClusterer:
    """
    Cluster market states into recurring environment types using k-means.

    Workflow:
    1. Load bucketed features (categorical)
    2. One-hot encode → binary matrix
    3. Standardize (optional but helps with mixed sparsity)
    4. Find optimal k via elbow + silhouette
    5. Fit k-means → cluster labels
    6. Describe clusters (dominant features per cluster)
    """

    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None

    def find_optimal_clusters(
        self,
        encoded_features: pd.DataFrame,
        k_range: range = CLUSTER_K_RANGE,
    ) -> int:
        """
        Use elbow method + silhouette score to find optimal k.

        Args:
            encoded_features: One-hot encoded feature matrix
            k_range: Range of k values to test

        Returns:
            Optimal number of clusters
        """
        X = self.scaler.fit_transform(encoded_features.values)

        inertias = []
        silhouettes = []
        k_values = list(k_range)

        print(f"  Testing k = {k_values[0]} to {k_values[-1]}...")
        for k in k_values:
            km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)

            if k >= 2:
                # Subsample for silhouette if dataset is large
                n = len(X)
                if n > 10000:
                    idx = np.random.RandomState(42).choice(n, 10000, replace=False)
                    sil = silhouette_score(X[idx], labels[idx])
                else:
                    sil = silhouette_score(X, labels)
                silhouettes.append(sil)
                print(f"    k={k:2d}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")
            else:
                silhouettes.append(0)
                print(f"    k={k:2d}: inertia={km.inertia_:.0f}")

        # Find best k by silhouette score (skip k=1 if in range)
        valid_sils = [(k, s) for k, s in zip(k_values, silhouettes) if k >= 2]
        if valid_sils:
            best_k, best_sil = max(valid_sils, key=lambda x: x[1])
        else:
            best_k = DEFAULT_N_CLUSTERS

        print(f"\n  Optimal k = {best_k} (silhouette = {best_sil:.4f})")
        self.n_clusters = best_k
        return best_k

    def fit(self, encoded_features: pd.DataFrame) -> np.ndarray:
        """
        Fit k-means on one-hot encoded feature matrix.

        Args:
            encoded_features: Binary feature matrix from one_hot_encode()

        Returns:
            Array of cluster labels (one per snapshot)
        """
        self.feature_columns = list(encoded_features.columns)
        X = self.scaler.fit_transform(encoded_features.values)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=42,
            max_iter=300,
        )
        labels = self.model.fit_predict(X)

        # Report cluster sizes
        counts = Counter(labels)
        print(f"\n  Fitted {self.n_clusters} clusters on {len(X):,} samples:")
        for cid in sorted(counts.keys()):
            pct = counts[cid] / len(labels) * 100
            print(f"    Cluster {cid:2d}: {counts[cid]:,} samples ({pct:.1f}%)")

        return labels

    def predict(self, encoded_features: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new snapshots.

        Args:
            encoded_features: One-hot encoded features (same columns as training)

        Returns:
            Array of cluster labels
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Ensure same columns (add missing with 0, drop extra)
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in encoded_features.columns:
                    encoded_features[col] = 0
            encoded_features = encoded_features[self.feature_columns]

        X = self.scaler.transform(encoded_features.values)
        return self.model.predict(X)

    def describe_clusters(
        self,
        bucketed_df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Dict[int, Dict]:
        """
        For each cluster, compute the mode of each categorical feature.
        Returns human-readable description of each cluster.

        Args:
            bucketed_df: DataFrame with categorical columns
            labels: Cluster labels array

        Returns:
            Dict[cluster_id -> {name, features, count, pct}]
        """
        exclude = {'bar_index', 'timestamp', 'symbol'}
        feature_cols = [c for c in bucketed_df.columns if c not in exclude]

        df = bucketed_df.copy()
        df['cluster'] = labels

        descriptions = {}

        for cid in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cid]
            count = len(cluster_df)
            pct = count / len(df) * 100

            # Get mode of each feature
            modes = {}
            for col in feature_cols:
                mode_val = cluster_df[col].mode()
                if len(mode_val) > 0:
                    modes[col] = str(mode_val.iloc[0])
                    # Also get % of dominant value
                    mode_pct = (cluster_df[col] == mode_val.iloc[0]).sum() / count * 100
                    modes[f'{col}_pct'] = round(mode_pct, 1)

            # Generate descriptive name from top clustering features
            # (session_block and volatility_regime excluded — they're noise)
            name_parts = []
            if modes.get('adx_regime'):
                name_parts.append(modes['adx_regime'].title())
            if modes.get('di_dominance'):
                name_parts.append(modes['di_dominance'].title())
            if modes.get('momentum_regime'):
                name_parts.append(modes['momentum_regime'].replace('_', ' ').title())

            name = ' '.join(name_parts) if name_parts else f'Cluster {cid}'

            descriptions[cid] = {
                'name': name,
                'count': count,
                'pct': round(pct, 1),
                'features': modes,
            }

        return descriptions

    def save_model(self, symbol: str) -> str:
        """Save scaler + k-means model + metadata to pickle."""
        filepath = MODEL_DIR / f"cluster_model_{symbol}.pkl"
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'feature_columns': self.feature_columns,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"  [SAVED] {filepath}")
        return str(filepath)

    def load_model(self, symbol: str) -> bool:
        """Load previously saved model."""
        filepath = MODEL_DIR / f"cluster_model_{symbol}.pkl"
        if not filepath.exists():
            return False

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.scaler = data['scaler']
        self.n_clusters = data['n_clusters']
        self.feature_columns = data['feature_columns']
        print(f"  [LOADED] {filepath} (k={self.n_clusters})")
        return True

    def save_descriptions(self, descriptions: Dict, symbol: str) -> str:
        """Save cluster descriptions to JSON."""
        filepath = MODEL_DIR / f"cluster_descriptions_{symbol}.json"
        # Convert numpy int keys to str for JSON serialization
        json_safe = {str(int(k)): v for k, v in descriptions.items()}
        with open(filepath, 'w') as f:
            json.dump(json_safe, f, indent=2)
        print(f"  [SAVED] {filepath}")
        return str(filepath)


def main():
    """CLI entry point for clustering."""
    import argparse
    from ml_system.market_state.bar_replay_engine import load_snapshot_data

    parser = argparse.ArgumentParser(description='Environment clustering')
    parser.add_argument('--symbol', type=str, help='Cluster specific symbol')
    parser.add_argument('--k', type=int, default=None, help='Force specific k')
    parser.add_argument('--auto-k', action='store_true', help='Find optimal k')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS
    bucketizer = FeatureBucketizer()

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  CLUSTERING: {symbol}")
        print(f"{'='*60}")

        # Load snapshots and bucketize
        snapshots = load_snapshot_data(symbol)
        if not snapshots:
            print(f"  [SKIP] No snapshots for {symbol}")
            continue

        bucketed_df = bucketizer.bucketize_batch(snapshots)
        encoded = bucketizer.one_hot_encode(bucketed_df)

        # Create clusterer
        if args.k:
            clusterer = EnvironmentClusterer(n_clusters=args.k)
        else:
            clusterer = EnvironmentClusterer()

        # Find optimal k if requested
        if args.auto_k and not args.k:
            optimal_k = clusterer.find_optimal_clusters(encoded)
            print(f"\n  Using optimal k = {optimal_k}")

        # Fit
        labels = clusterer.fit(encoded)

        # Describe clusters
        descriptions = clusterer.describe_clusters(bucketed_df, labels)

        print(f"\n  Cluster Descriptions:")
        for cid, desc in sorted(descriptions.items()):
            print(f"    {cid:2d}: {desc['name']} ({desc['count']:,} samples, {desc['pct']:.1f}%)")

        # Check minimum sample requirement
        small_clusters = [
            cid for cid, desc in descriptions.items()
            if desc['count'] < MIN_CLUSTER_SAMPLES
        ]
        if small_clusters:
            print(f"\n  [WARN] {len(small_clusters)} clusters have < {MIN_CLUSTER_SAMPLES} samples: {small_clusters}")

        # Save
        clusterer.save_model(symbol)
        clusterer.save_descriptions(descriptions, symbol)

        # Save labels with bucketed data
        bucketed_df['cluster'] = labels
        bucketizer.save_bucketed(bucketed_df, symbol)


if __name__ == '__main__':
    main()
