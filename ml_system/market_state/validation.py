#!/usr/bin/env python3
"""
Walk-Forward Validation (Phase 8)

Validates the market state modelling engine by:
1. Training clusters + dominance on a training window
2. Scoring alignment on an out-of-sample test window
3. Sliding the window and repeating
4. Reporting stability across folds

Usage:
    python validation.py --symbol EURUSD
    python validation.py --symbol EURUSD --train-months 24 --test-months 6
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

from ml_system.market_state.config import (
    SYMBOLS, MODEL_DIR, SNAPSHOT_DIR,
    DEFAULT_N_CLUSTERS, MIN_CLUSTER_SAMPLES,
    MR_1R_PIPS, BO_1R_PIPS,
    MIN_DOMINANCE_CONFIDENCE,
)
from ml_system.market_state.feature_buckets import FeatureBucketizer
from ml_system.market_state.environment_clustering import EnvironmentClusterer
from ml_system.market_state.strategy_dominance import StrategyDominanceModel


class WalkForwardValidator:
    """
    Walk-forward validation of the market state modelling approach.

    Splits data chronologically into train/test windows,
    trains the full pipeline on each train set, then evaluates
    on the test set. Slides forward and repeats.
    """

    def __init__(
        self,
        train_months: int = 24,
        test_months: int = 6,
        step_months: int = 3,
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.bucketizer = FeatureBucketizer()
        self.results = []

    def walk_forward_test(
        self,
        snapshots: List[Dict],
        symbol: str,
    ) -> Dict:
        """
        Run walk-forward validation on the full snapshot dataset.

        Args:
            snapshots: All snapshots (chronological order)
            symbol: Symbol name

        Returns:
            Summary dict with per-fold and aggregate metrics
        """
        if not snapshots:
            return {'error': 'No snapshots provided'}

        # Parse timestamps
        timestamps = []
        for s in snapshots:
            ts = s.get('timestamp', '')
            try:
                if 'T' in ts:
                    timestamps.append(pd.Timestamp(ts))
                else:
                    timestamps.append(pd.Timestamp(ts))
            except Exception:
                timestamps.append(pd.NaT)

        ts_series = pd.Series(timestamps)
        min_ts = ts_series.min()
        max_ts = ts_series.max()

        print(f"\n  Walk-forward test: {symbol}")
        print(f"  Date range: {min_ts} to {max_ts}")
        print(f"  Train: {self.train_months}mo | Test: {self.test_months}mo | Step: {self.step_months}mo")

        # Generate fold boundaries
        folds = []
        train_start = min_ts
        while True:
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_end = train_end + pd.DateOffset(months=self.test_months)

            if train_end >= max_ts:
                break

            if test_end > max_ts:
                test_end = max_ts

            folds.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
            })

            train_start = train_start + pd.DateOffset(months=self.step_months)

        print(f"  Generated {len(folds)} folds")

        # Run each fold
        fold_results = []
        for fold_idx, fold in enumerate(folds):
            print(f"\n  --- Fold {fold_idx + 1}/{len(folds)} ---")
            print(f"  Train: {fold['train_start'].strftime('%Y-%m-%d')} to "
                  f"{fold['train_end'].strftime('%Y-%m-%d')}")
            print(f"  Test:  {fold['test_start'].strftime('%Y-%m-%d')} to "
                  f"{fold['test_end'].strftime('%Y-%m-%d')}")

            # Split snapshots
            train_snaps = []
            test_snaps = []
            for s, ts in zip(snapshots, timestamps):
                if pd.isna(ts):
                    continue
                if fold['train_start'] <= ts < fold['train_end']:
                    train_snaps.append(s)
                elif fold['test_start'] <= ts < fold['test_end']:
                    test_snaps.append(s)

            print(f"  Train size: {len(train_snaps):,} | Test size: {len(test_snaps):,}")

            if len(train_snaps) < 1000 or len(test_snaps) < 100:
                print(f"  [SKIP] Insufficient data for this fold")
                continue

            fold_result = self._evaluate_fold(
                train_snaps, test_snaps, fold_idx, symbol
            )
            fold_results.append(fold_result)

        # Aggregate results
        summary = self._aggregate_results(fold_results, symbol)
        self.results = fold_results

        return summary

    def _evaluate_fold(
        self,
        train_snaps: List[Dict],
        test_snaps: List[Dict],
        fold_idx: int,
        symbol: str,
    ) -> Dict:
        """
        Train on train_snaps, evaluate on test_snaps.

        Returns metrics for this fold.
        """
        # 1. Bucketize train data
        train_bucketed = self.bucketizer.bucketize_batch(train_snaps)
        train_encoded = self.bucketizer.one_hot_encode(train_bucketed)

        # 2. Cluster train data
        n_clusters = min(DEFAULT_N_CLUSTERS, len(train_snaps) // MIN_CLUSTER_SAMPLES)
        n_clusters = max(3, n_clusters)

        clusterer = EnvironmentClusterer(n_clusters=n_clusters)
        train_labels = clusterer.fit(train_encoded)

        # 3. Build dominance table on train data
        dominance_model = StrategyDominanceModel()
        train_descriptions = clusterer.describe_clusters(train_bucketed, train_labels)
        dominance = dominance_model.build_dominance_table(
            train_snaps, train_labels.tolist(), train_descriptions
        )

        # 4. Bucketize + predict clusters for test data
        test_bucketed = self.bucketizer.bucketize_batch(test_snaps)
        test_encoded = self.bucketizer.one_hot_encode(test_bucketed)
        test_labels = clusterer.predict(test_encoded)

        # 5. Evaluate test outcomes
        results = self._evaluate_test_outcomes(
            test_snaps, test_labels, dominance
        )

        results['fold_idx'] = fold_idx
        results['train_size'] = len(train_snaps)
        results['test_size'] = len(test_snaps)
        results['n_clusters'] = n_clusters

        return results

    def _evaluate_test_outcomes(
        self,
        test_snaps: List[Dict],
        test_labels: np.ndarray,
        dominance: Dict,
    ) -> Dict:
        """
        For each test snapshot, evaluate three scenarios:
        1. Aligned: follow the model's recommended strategy + direction
        2. Counter: use the OPPOSITE strategy in the SAME direction
        3. Baseline: unconditional MR/BO win rates across all bars

        The true edge = aligned WR vs baseline WR (not vs counter).

        Uses MIN_DOMINANCE_CONFIDENCE threshold: when confidence is low,
        the dominant strategy becomes 'BOTH' and the win-rate tiebreaker
        selects which strategy to follow.
        """
        aligned_trades = 0
        aligned_wins = 0
        aligned_pnl = 0.0

        counter_trades = 0
        counter_wins = 0
        counter_pnl = 0.0

        # Baseline: unconditional rates
        baseline_mr_buy_wins = 0
        baseline_mr_sell_wins = 0
        baseline_bo_buy_wins = 0
        baseline_bo_sell_wins = 0
        baseline_total = 0

        skip_count = 0
        no_outcome_count = 0

        for snap, cluster_id in zip(test_snaps, test_labels):
            cluster_id = int(cluster_id)
            entry = dominance.get(cluster_id, {})
            raw_strat = entry.get('dominant', 'SKIP')
            dom_dir = entry.get('dominant_dir', 'none')
            confidence = entry.get('confidence', 0)

            # Apply confidence threshold (same logic as get_cluster_strategy)
            if raw_strat != 'SKIP' and confidence < MIN_DOMINANCE_CONFIDENCE:
                # Low confidence -- pick by win rate (same as _compute_priority)
                mr_wr = max(entry.get('mr_buy_wr', 0), entry.get('mr_sell_wr', 0))
                bo_wr = max(entry.get('bo_buy_wr', 0), entry.get('bo_sell_wr', 0))
                dom_strat = 'MR' if mr_wr >= bo_wr else 'BO'
            else:
                dom_strat = raw_strat

            # Check if we have forward outcome data
            if snap.get('fwd_mr_buy_1r') is None:
                no_outcome_count += 1
                continue

            # Accumulate baseline stats (unconditional)
            baseline_total += 1
            if snap.get('fwd_mr_buy_1r'):
                baseline_mr_buy_wins += 1
            if snap.get('fwd_mr_sell_1r'):
                baseline_mr_sell_wins += 1
            if snap.get('fwd_bo_buy_1r'):
                baseline_bo_buy_wins += 1
            if snap.get('fwd_bo_sell_1r'):
                baseline_bo_sell_wins += 1

            if dom_strat == 'SKIP':
                skip_count += 1
                continue

            # Determine aligned and counter outcomes for each direction
            directions = []
            if dom_dir == 'buy':
                directions = ['buy']
            elif dom_dir == 'sell':
                directions = ['sell']
            elif dom_dir == 'both':
                directions = ['buy', 'sell']
            else:
                skip_count += 1
                continue

            for direction in directions:
                # --- Aligned: follow model's strategy + direction ---
                if dom_strat == 'MR':
                    aligned_key = f'fwd_mr_{direction}_1r'
                    counter_key = f'fwd_bo_{direction}_1r'
                    aligned_r = MR_1R_PIPS
                    counter_r = BO_1R_PIPS
                else:  # BO
                    aligned_key = f'fwd_bo_{direction}_1r'
                    counter_key = f'fwd_mr_{direction}_1r'
                    aligned_r = BO_1R_PIPS
                    counter_r = MR_1R_PIPS

                aligned_trades += 1
                if snap.get(aligned_key):
                    aligned_wins += 1
                    aligned_pnl += aligned_r
                else:
                    aligned_pnl -= aligned_r

                # --- Counter: opposite strategy, same direction ---
                counter_trades += 1
                if snap.get(counter_key):
                    counter_wins += 1
                    counter_pnl += counter_r
                else:
                    counter_pnl -= counter_r

        aligned_wr = aligned_wins / aligned_trades if aligned_trades > 0 else 0
        counter_wr = counter_wins / counter_trades if counter_trades > 0 else 0

        # Baseline rates (unconditional)
        if baseline_total > 0:
            base_mr_wr = (baseline_mr_buy_wins + baseline_mr_sell_wins) / (baseline_total * 2)
            base_bo_wr = (baseline_bo_buy_wins + baseline_bo_sell_wins) / (baseline_total * 2)
            baseline_wr = (base_mr_wr + base_bo_wr) / 2  # avg of all strategies
        else:
            base_mr_wr = base_bo_wr = baseline_wr = 0

        # Edge vs counter (same direction, wrong strategy)
        edge_vs_counter = aligned_wr - counter_wr
        # Edge vs baseline (model selection vs random strategy)
        edge_vs_baseline = aligned_wr - baseline_wr

        if aligned_trades > 0:
            avg_pnl = aligned_pnl / aligned_trades
        else:
            avg_pnl = 0

        print(f"    Aligned: {aligned_trades} trades, {aligned_wr:.1%} WR, "
              f"{aligned_pnl:.0f} pips")
        print(f"    Counter: {counter_trades} trades, {counter_wr:.1%} WR, "
              f"{counter_pnl:.0f} pips")
        print(f"    Baseline: MR={base_mr_wr:.1%} BO={base_bo_wr:.1%} avg={baseline_wr:.1%}")
        print(f"    Edge vs counter: {edge_vs_counter:+.1%} | "
              f"Edge vs baseline: {edge_vs_baseline:+.1%}")
        print(f"    Skipped: {skip_count} | No data: {no_outcome_count}")

        return {
            'aligned_trades': aligned_trades,
            'aligned_wins': aligned_wins,
            'aligned_wr': round(aligned_wr, 4),
            'aligned_pnl': round(aligned_pnl, 2),
            'counter_trades': counter_trades,
            'counter_wins': counter_wins,
            'counter_wr': round(counter_wr, 4),
            'counter_pnl': round(counter_pnl, 2),
            'edge': round(edge_vs_counter, 4),
            'edge_vs_baseline': round(edge_vs_baseline, 4),
            'baseline_mr_wr': round(base_mr_wr, 4),
            'baseline_bo_wr': round(base_bo_wr, 4),
            'skip_count': skip_count,
            'avg_pnl_per_trade': round(avg_pnl, 2),
        }

    def _aggregate_results(self, fold_results: List[Dict], symbol: str) -> Dict:
        """Aggregate metrics across all folds."""
        if not fold_results:
            return {'error': 'No valid folds'}

        print(f"\n  {'='*60}")
        print(f"  WALK-FORWARD SUMMARY: {symbol}")
        print(f"  {'='*60}")

        total_aligned = sum(r['aligned_trades'] for r in fold_results)
        total_aligned_wins = sum(r['aligned_wins'] for r in fold_results)
        total_aligned_pnl = sum(r['aligned_pnl'] for r in fold_results)

        total_counter = sum(r['counter_trades'] for r in fold_results)
        total_counter_wins = sum(r['counter_wins'] for r in fold_results)
        total_counter_pnl = sum(r['counter_pnl'] for r in fold_results)

        agg_aligned_wr = total_aligned_wins / total_aligned if total_aligned > 0 else 0
        agg_counter_wr = total_counter_wins / total_counter if total_counter > 0 else 0
        agg_edge = agg_aligned_wr - agg_counter_wr

        edges = [r['edge'] for r in fold_results]
        baseline_edges = [r.get('edge_vs_baseline', 0) for r in fold_results]
        wrs = [r['aligned_wr'] for r in fold_results]

        # Average baseline rates
        avg_baseline_mr = np.mean([r.get('baseline_mr_wr', 0) for r in fold_results])
        avg_baseline_bo = np.mean([r.get('baseline_bo_wr', 0) for r in fold_results])

        print(f"\n  Folds: {len(fold_results)}")

        print(f"\n  Baseline (unconditional):")
        print(f"    MR 1R avg hit rate: {avg_baseline_mr:.1%}")
        print(f"    BO 1R avg hit rate: {avg_baseline_bo:.1%}")

        print(f"\n  Aligned (following model):")
        print(f"    Total trades: {total_aligned:,}")
        print(f"    Win rate: {agg_aligned_wr:.1%}")
        print(f"    Total PnL: {total_aligned_pnl:+.0f} pips")
        if total_aligned > 0:
            print(f"    Avg PnL/trade: {total_aligned_pnl/total_aligned:+.1f} pips")

        print(f"\n  Counter (opposite strategy, same direction):")
        print(f"    Total trades: {total_counter:,}")
        print(f"    Win rate: {agg_counter_wr:.1%}")
        print(f"    Total PnL: {total_counter_pnl:+.0f} pips")

        print(f"\n  Model Edge:")
        print(f"    WR edge vs counter: {agg_edge:+.1%}")
        print(f"    WR edge vs baseline: {np.mean(baseline_edges):+.1%}")
        print(f"    PnL edge: {total_aligned_pnl - total_counter_pnl:+.0f} pips")
        print(f"    Edge stability (std): {np.std(edges):.3f}")
        print(f"    WR stability (std): {np.std(wrs):.3f}")
        print(f"    Positive edge folds (vs counter): {sum(1 for e in edges if e > 0)}/{len(edges)}")
        print(f"    Positive edge folds (vs baseline): {sum(1 for e in baseline_edges if e > 0)}/{len(baseline_edges)}")

        # Per-fold detail
        print(f"\n  Per-fold detail:")
        print(f"  {'Fold':>5} | {'Aligned WR':>10} | {'Counter WR':>10} | {'vs Ctr':>7} | {'vs Base':>7} | {'PnL':>8}")
        print(f"  {'-'*65}")
        for r in fold_results:
            print(f"  {r['fold_idx']+1:>5} | {r['aligned_wr']:>9.1%} | "
                  f"{r['counter_wr']:>9.1%} | {r['edge']:>+6.1%} | "
                  f"{r.get('edge_vs_baseline', 0):>+6.1%} | "
                  f"{r['aligned_pnl']:>+7.0f}")

        summary = {
            'symbol': symbol,
            'folds': len(fold_results),
            'aligned_total_trades': total_aligned,
            'aligned_wr': round(agg_aligned_wr, 4),
            'aligned_total_pnl': round(total_aligned_pnl, 2),
            'counter_total_trades': total_counter,
            'counter_wr': round(agg_counter_wr, 4),
            'counter_total_pnl': round(total_counter_pnl, 2),
            'edge_vs_counter': round(agg_edge, 4),
            'edge_vs_baseline': round(np.mean(baseline_edges), 4),
            'baseline_mr_wr': round(avg_baseline_mr, 4),
            'baseline_bo_wr': round(avg_baseline_bo, 4),
            'edge_std': round(np.std(edges), 4),
            'wr_std': round(np.std(wrs), 4),
            'positive_edge_folds_vs_counter': sum(1 for e in edges if e > 0),
            'positive_edge_folds_vs_baseline': sum(1 for e in baseline_edges if e > 0),
            'fold_details': fold_results,
        }

        # Save results
        results_path = MODEL_DIR / f"validation_{symbol}.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  [SAVED] {results_path}")

        return summary


def main():
    """CLI entry point."""
    import argparse
    from ml_system.market_state.bar_replay_engine import load_snapshot_data

    parser = argparse.ArgumentParser(description='Walk-forward validation')
    parser.add_argument('--symbol', type=str, help='Validate specific symbol')
    parser.add_argument('--train-months', type=int, default=24, help='Training window (months)')
    parser.add_argument('--test-months', type=int, default=6, help='Test window (months)')
    parser.add_argument('--step-months', type=int, default=3, help='Step size (months)')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD VALIDATION: {symbol}")
        print(f"{'='*60}")

        snapshots = load_snapshot_data(symbol)
        if not snapshots:
            print(f"  [SKIP] No snapshots for {symbol}")
            continue

        validator = WalkForwardValidator(
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
        )
        summary = validator.walk_forward_test(snapshots, symbol)


if __name__ == '__main__':
    main()
