#!/usr/bin/env python3
"""
Strategy Dominance Model (Phase 4)

For each cluster, compute MR vs BO expectancy from forward outcomes
to determine which strategy dominates in each environment.

Usage:
    python strategy_dominance.py                  # All symbols
    python strategy_dominance.py --symbol EURUSD  # One symbol
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ml_system.market_state.config import (
    SYMBOLS, MODEL_DIR, SNAPSHOT_DIR, BUCKETED_DIR,
    MR_1R_PIPS, MR_2R_PIPS, BO_1R_PIPS, BO_2R_PIPS,
    MIN_CLUSTER_SAMPLES,
)


class StrategyDominanceModel:
    """
    Determine which strategy (MR or BO) dominates in each environment cluster.

    For each cluster, compute:
    - MR buy/sell win rates (from forward MR 1R target hits)
    - BO buy/sell win rates (from forward BO 1R target hits)
    - MR/BO expectancy (win_rate * avg_win - loss_rate * avg_loss)
    - Dominant strategy: higher expectancy wins; SKIP if both negative
    - Confidence: abs(mr_expectancy - bo_expectancy)
    """

    def __init__(
        self,
        mr_1r: float = MR_1R_PIPS,
        mr_2r: float = MR_2R_PIPS,
        bo_1r: float = BO_1R_PIPS,
        bo_2r: float = BO_2R_PIPS,
    ):
        self.mr_1r = mr_1r
        self.mr_2r = mr_2r
        self.bo_1r = bo_1r
        self.bo_2r = bo_2r
        self.dominance_table = None

    def build_dominance_table(
        self,
        snapshots: List[Dict],
        cluster_labels: List[int],
        cluster_descriptions: Dict = None,
    ) -> Dict[int, Dict]:
        """
        Build the strategy dominance table from snapshots + cluster labels.

        Args:
            snapshots: Raw snapshot dicts (must include forward outcome fields)
            cluster_labels: Cluster label for each snapshot
            cluster_descriptions: Optional cluster name/desc from clustering

        Returns:
            Dict[cluster_id -> dominance metrics]
        """
        if len(snapshots) != len(cluster_labels):
            raise ValueError(
                f"Snapshots ({len(snapshots)}) and labels ({len(cluster_labels)}) "
                "must have same length"
            )

        # Build DataFrame for vectorized computation
        records = []
        for snap, label in zip(snapshots, cluster_labels):
            records.append({
                'cluster': label,
                # MR forward outcomes
                'fwd_mr_buy_1r': snap.get('fwd_mr_buy_1r'),
                'fwd_mr_buy_2r': snap.get('fwd_mr_buy_2r'),
                'fwd_mr_sell_1r': snap.get('fwd_mr_sell_1r'),
                'fwd_mr_sell_2r': snap.get('fwd_mr_sell_2r'),
                # BO forward outcomes
                'fwd_bo_buy_1r': snap.get('fwd_bo_buy_1r'),
                'fwd_bo_buy_2r': snap.get('fwd_bo_buy_2r'),
                'fwd_bo_sell_1r': snap.get('fwd_bo_sell_1r'),
                'fwd_bo_sell_2r': snap.get('fwd_bo_sell_2r'),
                # MFE/MAE for expectancy calculation
                'fwd_24h_mfe_up_pips': snap.get('fwd_24h_mfe_up_pips'),
                'fwd_24h_mfe_down_pips': snap.get('fwd_24h_mfe_down_pips'),
                'fwd_24h_mae_up_pips': snap.get('fwd_24h_mae_up_pips'),
                'fwd_24h_mae_down_pips': snap.get('fwd_24h_mae_down_pips'),
                'fwd_24h_net_direction': snap.get('fwd_24h_net_direction'),
            })

        df = pd.DataFrame(records)
        dominance = {}

        for cid in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cid]
            count = len(cluster_df)

            # Get cluster name from descriptions if available
            name = 'Unknown'
            if cluster_descriptions and cid in cluster_descriptions:
                name = cluster_descriptions[cid].get('name', f'Cluster {cid}')
            elif cluster_descriptions and str(cid) in cluster_descriptions:
                name = cluster_descriptions[str(cid)].get('name', f'Cluster {cid}')

            # Filter rows with valid forward outcome data
            valid = cluster_df.dropna(subset=['fwd_mr_buy_1r'])
            valid_count = len(valid)

            if valid_count < 10:
                dominance[cid] = {
                    'name': name,
                    'sample_count': count,
                    'valid_count': valid_count,
                    'dominant': 'SKIP',
                    'dominant_dir': 'none',
                    'confidence': 0,
                    'mr_buy_wr': 0, 'mr_sell_wr': 0, 'mr_expectancy': 0,
                    'bo_buy_wr': 0, 'bo_sell_wr': 0, 'bo_expectancy': 0,
                    'insufficient_data': True,
                }
                continue

            # ── MR Metrics ───────────────────────────────────────────
            mr_buy_wins = valid['fwd_mr_buy_1r'].sum()
            mr_sell_wins = valid['fwd_mr_sell_1r'].sum()
            mr_buy_wr = mr_buy_wins / valid_count
            mr_sell_wr = mr_sell_wins / valid_count
            mr_buy_2r_wr = valid['fwd_mr_buy_2r'].sum() / valid_count
            mr_sell_2r_wr = valid['fwd_mr_sell_2r'].sum() / valid_count

            # MR expectancy: avg win * WR - avg loss * LR
            # Win = 1R pips, Loss = 1R pips (assuming 1:1 SL)
            mr_best_wr = max(mr_buy_wr, mr_sell_wr)
            mr_expectancy = (mr_best_wr * self.mr_1r) - ((1 - mr_best_wr) * self.mr_1r)

            # ── BO Metrics ───────────────────────────────────────────
            bo_buy_wins = valid['fwd_bo_buy_1r'].sum()
            bo_sell_wins = valid['fwd_bo_sell_1r'].sum()
            bo_buy_wr = bo_buy_wins / valid_count
            bo_sell_wr = bo_sell_wins / valid_count
            bo_buy_2r_wr = valid['fwd_bo_buy_2r'].sum() / valid_count
            bo_sell_2r_wr = valid['fwd_bo_sell_2r'].sum() / valid_count

            bo_best_wr = max(bo_buy_wr, bo_sell_wr)
            bo_expectancy = (bo_best_wr * self.bo_1r) - ((1 - bo_best_wr) * self.bo_1r)

            # ── Dominant Direction ────────────────────────────────────
            mr_dir = 'buy' if mr_buy_wr > mr_sell_wr else 'sell'
            if abs(mr_buy_wr - mr_sell_wr) < 0.03:
                mr_dir = 'both'

            bo_dir = 'buy' if bo_buy_wr > bo_sell_wr else 'sell'
            if abs(bo_buy_wr - bo_sell_wr) < 0.03:
                bo_dir = 'both'

            # ── Net Direction Bias ────────────────────────────────────
            dir_counts = valid['fwd_24h_net_direction'].value_counts()
            up_pct = dir_counts.get('up', 0) / valid_count
            down_pct = dir_counts.get('down', 0) / valid_count

            # ── Determine Dominant Strategy ───────────────────────────
            if mr_expectancy > 0 and bo_expectancy > 0:
                dominant = 'MR' if mr_expectancy >= bo_expectancy else 'BO'
            elif mr_expectancy > 0:
                dominant = 'MR'
            elif bo_expectancy > 0:
                dominant = 'BO'
            else:
                dominant = 'SKIP'  # Both negative expectancy

            dominant_dir = mr_dir if dominant == 'MR' else bo_dir if dominant == 'BO' else 'none'
            confidence = abs(mr_expectancy - bo_expectancy)

            # ── Average MFE/MAE ───────────────────────────────────────
            avg_mfe_up = valid['fwd_24h_mfe_up_pips'].mean()
            avg_mfe_down = valid['fwd_24h_mfe_down_pips'].mean()
            avg_mae_up = valid['fwd_24h_mae_up_pips'].mean()
            avg_mae_down = valid['fwd_24h_mae_down_pips'].mean()

            dominance[cid] = {
                'name': name,
                'sample_count': count,
                'valid_count': valid_count,
                # MR
                'mr_buy_wr': round(mr_buy_wr, 4),
                'mr_sell_wr': round(mr_sell_wr, 4),
                'mr_buy_2r_wr': round(mr_buy_2r_wr, 4),
                'mr_sell_2r_wr': round(mr_sell_2r_wr, 4),
                'mr_expectancy': round(mr_expectancy, 2),
                'mr_best_dir': mr_dir,
                # BO
                'bo_buy_wr': round(bo_buy_wr, 4),
                'bo_sell_wr': round(bo_sell_wr, 4),
                'bo_buy_2r_wr': round(bo_buy_2r_wr, 4),
                'bo_sell_2r_wr': round(bo_sell_2r_wr, 4),
                'bo_expectancy': round(bo_expectancy, 2),
                'bo_best_dir': bo_dir,
                # Dominant
                'dominant': dominant,
                'dominant_dir': dominant_dir,
                'confidence': round(confidence, 2),
                # Extra metrics
                'net_up_pct': round(up_pct, 4),
                'net_down_pct': round(down_pct, 4),
                'avg_mfe_up': round(float(avg_mfe_up), 2) if not pd.isna(avg_mfe_up) else 0,
                'avg_mfe_down': round(float(avg_mfe_down), 2) if not pd.isna(avg_mfe_down) else 0,
                'avg_mae_up': round(float(avg_mae_up), 2) if not pd.isna(avg_mae_up) else 0,
                'avg_mae_down': round(float(avg_mae_down), 2) if not pd.isna(avg_mae_down) else 0,
                'insufficient_data': count < MIN_CLUSTER_SAMPLES,
            }

        self.dominance_table = dominance
        return dominance

    def save_dominance_table(self, table: Dict, symbol: str) -> str:
        """Save dominance table to JSON."""
        filepath = MODEL_DIR / f"dominance_{symbol}.json"

        # Convert int keys to str for JSON
        json_table = {str(k): v for k, v in table.items()}

        with open(filepath, 'w') as f:
            json.dump(json_table, f, indent=2)
        print(f"  [SAVED] {filepath}")
        return str(filepath)

    def load_dominance_table(self, symbol: str) -> Optional[Dict]:
        """Load dominance table from JSON."""
        filepath = MODEL_DIR / f"dominance_{symbol}.json"
        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert str keys back to int
        self.dominance_table = {int(k): v for k, v in data.items()}
        print(f"  [LOADED] {filepath}")
        return self.dominance_table

    def get_cluster_strategy(self, cluster_id: int, min_confidence: float = None) -> Dict:
        """
        Get the dominant strategy for a given cluster.

        Args:
            cluster_id: Cluster to look up.
            min_confidence: Minimum confidence (pip gap) to declare a single
                winner. Below this, returns 'BOTH'. Defaults to config value.

        Returns:
            Dict with strategy info including both MR/BO metrics.
        """
        _empty = {
            'strategy': 'SKIP', 'direction': 'none', 'confidence': 0,
            'expectancy': 0, 'mr_expectancy': 0, 'bo_expectancy': 0,
            'mr_best_wr': 0, 'bo_best_wr': 0, 'raw_dominant': 'SKIP',
        }
        if self.dominance_table is None:
            return _empty

        entry = self.dominance_table.get(cluster_id)
        if entry is None:
            return _empty

        if min_confidence is None:
            from ml_system.market_state.config import MIN_DOMINANCE_CONFIDENCE
            min_confidence = MIN_DOMINANCE_CONFIDENCE

        raw_dominant = entry['dominant']
        confidence = entry['confidence']
        mr_exp = entry.get('mr_expectancy', 0)
        bo_exp = entry.get('bo_expectancy', 0)
        mr_best_wr = max(entry.get('mr_buy_wr', 0), entry.get('mr_sell_wr', 0))
        bo_best_wr = max(entry.get('bo_buy_wr', 0), entry.get('bo_sell_wr', 0))

        # If confidence below threshold, both strategies are viable
        if raw_dominant != 'SKIP' and confidence < min_confidence:
            strategy = 'BOTH'
        else:
            strategy = raw_dominant

        return {
            'strategy': strategy,
            'direction': entry['dominant_dir'],
            'confidence': confidence,
            'expectancy': mr_exp if raw_dominant == 'MR' else bo_exp if raw_dominant == 'BO' else 0,
            'mr_expectancy': mr_exp,
            'bo_expectancy': bo_exp,
            'mr_best_wr': mr_best_wr,
            'bo_best_wr': bo_best_wr,
            'raw_dominant': raw_dominant,
        }

    def print_dominance_summary(self, table: Dict = None):
        """Print a formatted summary of the dominance table."""
        table = table or self.dominance_table
        if table is None:
            print("  No dominance table loaded.")
            return

        print(f"\n  {'ID':>3} | {'Name':<35} | {'Dom':>4} | {'Dir':>5} | "
              f"{'MR WR':>6} | {'BO WR':>6} | {'MR Exp':>7} | {'BO Exp':>7} | "
              f"{'Conf':>5} | {'N':>6}")
        print(f"  {'-'*110}")

        mr_count = bo_count = skip_count = 0

        for cid in sorted(table.keys()):
            entry = table[cid]
            name = entry['name'][:35]
            dom = entry['dominant']

            # Use best direction WR for display
            mr_wr = max(entry.get('mr_buy_wr', 0), entry.get('mr_sell_wr', 0))
            bo_wr = max(entry.get('bo_buy_wr', 0), entry.get('bo_sell_wr', 0))

            print(f"  {cid:3d} | {name:<35} | {dom:>4} | {entry['dominant_dir']:>5} | "
                  f"{mr_wr:>5.1%} | {bo_wr:>5.1%} | "
                  f"{entry.get('mr_expectancy', 0):>7.2f} | {entry.get('bo_expectancy', 0):>7.2f} | "
                  f"{entry.get('confidence', 0):>5.2f} | {entry.get('valid_count', 0):>6,}")

            if dom == 'MR':
                mr_count += 1
            elif dom == 'BO':
                bo_count += 1
            else:
                skip_count += 1

        print(f"\n  Summary: {mr_count} MR-dominant, {bo_count} BO-dominant, {skip_count} SKIP")


def main():
    """CLI entry point for dominance analysis."""
    import argparse
    from ml_system.market_state.bar_replay_engine import load_snapshot_data
    from ml_system.market_state.feature_buckets import FeatureBucketizer
    from ml_system.market_state.environment_clustering import EnvironmentClusterer

    parser = argparse.ArgumentParser(description='Strategy dominance analysis')
    parser.add_argument('--symbol', type=str, help='Analyze specific symbol only')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  DOMINANCE ANALYSIS: {symbol}")
        print(f"{'='*60}")

        # Load snapshots
        snapshots = load_snapshot_data(symbol)
        if not snapshots:
            print(f"  [SKIP] No snapshots for {symbol}")
            continue

        # Load cluster model
        clusterer = EnvironmentClusterer()
        if not clusterer.load_model(symbol):
            print(f"  [ERROR] No cluster model for {symbol}. Run environment_clustering.py first.")
            continue

        # Load cluster descriptions
        desc_path = MODEL_DIR / f"cluster_descriptions_{symbol}.json"
        cluster_descriptions = {}
        if desc_path.exists():
            with open(desc_path, 'r') as f:
                cluster_descriptions = json.load(f)
            # Convert string keys to int
            cluster_descriptions = {int(k): v for k, v in cluster_descriptions.items()}

        # Bucketize + predict clusters for all snapshots
        bucketizer = FeatureBucketizer()
        bucketed_df = bucketizer.bucketize_batch(snapshots)
        encoded = bucketizer.one_hot_encode(bucketed_df)
        labels = clusterer.predict(encoded)

        # Build dominance table
        model = StrategyDominanceModel()
        dominance = model.build_dominance_table(snapshots, labels.tolist(), cluster_descriptions)

        # Print summary
        model.print_dominance_summary(dominance)

        # Save
        model.save_dominance_table(dominance, symbol)


if __name__ == '__main__':
    main()
