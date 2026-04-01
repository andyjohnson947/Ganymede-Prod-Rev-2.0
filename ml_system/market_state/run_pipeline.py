#!/usr/bin/env python3
"""
Full Pipeline Runner

Runs the complete market state pipeline after snapshots are generated:
1. Feature bucketization (Phase 2)
2. Environment clustering (Phase 3)
3. Strategy dominance analysis (Phase 4)
4. Reporting (Phase 7)

Usage:
    python run_pipeline.py                  # All symbols
    python run_pipeline.py --symbol EURUSD  # One symbol
"""

import sys
import json
from pathlib import Path

# Path setup
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / 'trading_bot'))

from ml_system.market_state.config import SYMBOLS, MODEL_DIR
from ml_system.market_state.bar_replay_engine import load_snapshot_data
from ml_system.market_state.feature_buckets import FeatureBucketizer
from ml_system.market_state.environment_clustering import EnvironmentClusterer
from ml_system.market_state.strategy_dominance import StrategyDominanceModel
from ml_system.market_state.reporting import MarketStateReporter


def run_pipeline(symbol: str, auto_k: bool = True):
    """Run phases 2-4 for a single symbol."""

    print(f"\n{'='*70}")
    print(f"  PIPELINE: {symbol}")
    print(f"{'='*70}")

    # Load snapshots
    snapshots = load_snapshot_data(symbol)
    if not snapshots:
        print(f"  [SKIP] No snapshots for {symbol}")
        return

    # ── Phase 2: Bucketize ────────────────────────────────────────
    print(f"\n  --- PHASE 2: Feature Bucketization ---")
    bucketizer = FeatureBucketizer()
    bucketed_df = bucketizer.bucketize_batch(snapshots)
    encoded = bucketizer.one_hot_encode(bucketed_df)
    bucketizer.save_bucketed(bucketed_df, symbol)

    # Show distribution
    print(f"\n  Feature distributions:")
    exclude = {'bar_index', 'timestamp', 'symbol'}
    for col in bucketed_df.columns:
        if col not in exclude:
            counts = bucketed_df[col].value_counts()
            top = counts.index[0]
            top_pct = counts.iloc[0] / len(bucketed_df) * 100
            print(f"    {col}: {len(counts)} categories, "
                  f"most common = {top} ({top_pct:.0f}%)")

    # ── Phase 3: Clustering ───────────────────────────────────────
    print(f"\n  --- PHASE 3: Environment Clustering ---")
    clusterer = EnvironmentClusterer()

    if auto_k:
        optimal_k = clusterer.find_optimal_clusters(encoded)
        print(f"\n  Using optimal k = {optimal_k}")
    else:
        print(f"  Using default k = {clusterer.n_clusters}")

    labels = clusterer.fit(encoded)
    descriptions = clusterer.describe_clusters(bucketed_df, labels)

    print(f"\n  Cluster Descriptions:")
    for cid, desc in sorted(descriptions.items()):
        print(f"    {cid:2d}: {desc['name']} ({desc['count']:,} samples, {desc['pct']:.1f}%)")

    clusterer.save_model(symbol)
    clusterer.save_descriptions(descriptions, symbol)

    # Save labels with bucketed data
    bucketed_df['cluster'] = labels
    bucketizer.save_bucketed(bucketed_df, symbol)

    # ── Phase 4: Strategy Dominance ───────────────────────────────
    print(f"\n  --- PHASE 4: Strategy Dominance ---")
    dominance_model = StrategyDominanceModel()
    dominance = dominance_model.build_dominance_table(
        snapshots, labels.tolist(), descriptions
    )
    dominance_model.print_dominance_summary(dominance)
    dominance_model.save_dominance_table(dominance, symbol)

    # ── Phase 7: Report ───────────────────────────────────────────
    print(f"\n  --- PHASE 7: Reporting ---")
    reporter = MarketStateReporter(symbol)
    reporter.generate_full_report()

    print(f"\n  {'='*70}")
    print(f"  PIPELINE COMPLETE: {symbol}")
    print(f"  {'='*70}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run full pipeline (phases 2-4+7)')
    parser.add_argument('--symbol', type=str, help='Process specific symbol only')
    parser.add_argument('--no-auto-k', action='store_true', help='Use default k instead of auto')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    for symbol in symbols:
        run_pipeline(symbol, auto_k=not args.no_auto_k)


if __name__ == '__main__':
    main()
