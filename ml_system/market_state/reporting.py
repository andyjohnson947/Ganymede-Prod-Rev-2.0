#!/usr/bin/env python3
"""
Reporting Module (Phase 7)

Generates analysis reports for the market state modelling engine:
- Cluster distribution (count, %, dominant features)
- Strategy dominance per cluster (MR/BO expectancy, win rates)
- Alignment score distribution vs outcomes
- Temporal cluster heatmap (hour x cluster frequency)

Usage:
    python reporting.py                  # All symbols
    python reporting.py --symbol EURUSD  # One symbol
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from ml_system.market_state.config import (
    SYMBOLS, MODEL_DIR, SNAPSHOT_DIR, BUCKETED_DIR,
)


class MarketStateReporter:
    """Generate analysis reports for the market state modelling engine."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.snapshots = None
        self.bucketed_df = None
        self.cluster_descriptions = None
        self.dominance_table = None

    def load_data(self) -> bool:
        """Load all data needed for reporting."""
        from ml_system.market_state.bar_replay_engine import load_snapshot_data

        # Load snapshots
        self.snapshots = load_snapshot_data(self.symbol)
        if not self.snapshots:
            return False

        # Load bucketed data
        bucketed_path = BUCKETED_DIR / f"{self.symbol}_bucketed.csv"
        if bucketed_path.exists():
            self.bucketed_df = pd.read_csv(bucketed_path)

        # Load cluster descriptions
        desc_path = MODEL_DIR / f"cluster_descriptions_{self.symbol}.json"
        if desc_path.exists():
            with open(desc_path, 'r') as f:
                self.cluster_descriptions = json.load(f)

        # Load dominance table
        dom_path = MODEL_DIR / f"dominance_{self.symbol}.json"
        if dom_path.exists():
            with open(dom_path, 'r') as f:
                self.dominance_table = json.load(f)

        return True

    def report_overview(self) -> str:
        """Generate overview report."""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"  MARKET STATE ENGINE REPORT: {self.symbol}")
        lines.append(f"{'='*70}")

        if self.snapshots:
            lines.append(f"\n  Total snapshots: {len(self.snapshots):,}")
            first = self.snapshots[0]
            last = self.snapshots[-1]
            lines.append(f"  Date range: {first.get('timestamp', 'N/A')[:10]} to "
                        f"{last.get('timestamp', 'N/A')[:10]}")

            # Bar count
            lines.append(f"  Bar range: {first.get('bar_index', 0)} to {last.get('bar_index', 0)}")

        return '\n'.join(lines)

    def report_cluster_distribution(self) -> str:
        """Report cluster sizes and descriptions."""
        lines = []
        lines.append(f"\n  {'='*60}")
        lines.append(f"  CLUSTER DISTRIBUTION")
        lines.append(f"  {'='*60}")

        if not self.cluster_descriptions:
            lines.append("  No cluster data available.")
            return '\n'.join(lines)

        lines.append(f"\n  {'ID':>3} | {'Name':<40} | {'Count':>7} | {'Pct':>5}")
        lines.append(f"  {'-'*65}")

        total = sum(v.get('count', 0) for v in self.cluster_descriptions.values())

        for cid in sorted(self.cluster_descriptions.keys(), key=lambda x: int(x)):
            desc = self.cluster_descriptions[cid]
            name = desc.get('name', f'Cluster {cid}')[:40]
            count = desc.get('count', 0)
            pct = desc.get('pct', 0)
            lines.append(f"  {cid:>3} | {name:<40} | {count:>7,} | {pct:>4.1f}%")

        lines.append(f"  {'-'*65}")
        lines.append(f"  {'':>3} | {'TOTAL':<40} | {total:>7,} | 100.0%")

        return '\n'.join(lines)

    def report_strategy_dominance(self) -> str:
        """Report strategy dominance per cluster."""
        lines = []
        lines.append(f"\n  {'='*60}")
        lines.append(f"  STRATEGY DOMINANCE BY CLUSTER")
        lines.append(f"  {'='*60}")

        if not self.dominance_table:
            lines.append("  No dominance data available.")
            return '\n'.join(lines)

        lines.append(f"\n  {'ID':>3} | {'Name':<30} | {'Dom':>4} | {'Dir':>5} | "
                     f"{'MR WR':>6} | {'BO WR':>6} | {'MR Exp':>7} | {'BO Exp':>7} | {'Conf':>5}")
        lines.append(f"  {'-'*100}")

        mr_count = bo_count = skip_count = 0

        for cid in sorted(self.dominance_table.keys(), key=lambda x: int(x)):
            entry = self.dominance_table[cid]
            name = entry.get('name', f'Cluster {cid}')[:30]
            dom = entry.get('dominant', '?')

            mr_wr = max(entry.get('mr_buy_wr', 0), entry.get('mr_sell_wr', 0))
            bo_wr = max(entry.get('bo_buy_wr', 0), entry.get('bo_sell_wr', 0))

            lines.append(f"  {cid:>3} | {name:<30} | {dom:>4} | "
                        f"{entry.get('dominant_dir', '?'):>5} | "
                        f"{mr_wr:>5.1%} | {bo_wr:>5.1%} | "
                        f"{entry.get('mr_expectancy', 0):>7.2f} | "
                        f"{entry.get('bo_expectancy', 0):>7.2f} | "
                        f"{entry.get('confidence', 0):>5.2f}")

            if dom == 'MR':
                mr_count += 1
            elif dom == 'BO':
                bo_count += 1
            else:
                skip_count += 1

        lines.append(f"\n  Summary: {mr_count} MR-dominant | {bo_count} BO-dominant | {skip_count} SKIP")

        return '\n'.join(lines)

    def report_temporal_patterns(self) -> str:
        """Report cluster frequency by hour and day of week."""
        lines = []
        lines.append(f"\n  {'='*60}")
        lines.append(f"  TEMPORAL PATTERNS")
        lines.append(f"  {'='*60}")

        if not self.bucketed_df is not None or 'cluster' not in self.bucketed_df.columns:
            lines.append("  No bucketed data with clusters available.")
            return '\n'.join(lines)

        if self.snapshots is None:
            return '\n'.join(lines)

        # Build hour-cluster frequency table
        hours = [s.get('hour_utc', 0) for s in self.snapshots]
        clusters = self.bucketed_df['cluster'].tolist() if 'cluster' in self.bucketed_df.columns else []

        if len(hours) != len(clusters):
            lines.append("  Data length mismatch between snapshots and clusters.")
            return '\n'.join(lines)

        # Session distribution
        lines.append(f"\n  Session-cluster heatmap (top 3 clusters per session):")
        session_clusters = defaultdict(lambda: Counter())
        sessions = [s.get('session', 'unknown') for s in self.snapshots]

        for sess, clust in zip(sessions, clusters):
            session_clusters[sess][clust] += 1

        for session in ['asian_early', 'asian_late', 'london_open', 'london_mid',
                        'overlap', 'ny_late', 'close']:
            counts = session_clusters.get(session, Counter())
            total = sum(counts.values())
            if total == 0:
                continue
            top3 = counts.most_common(3)
            top_str = ', '.join([f"C{c}({n/total*100:.0f}%)" for c, n in top3])
            lines.append(f"    {session:<15}: {total:>5} bars | {top_str}")

        # Day of week distribution
        lines.append(f"\n  Day-of-week distribution:")
        day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
        days = [s.get('day_of_week', 0) for s in self.snapshots]
        day_counts = Counter(days)
        for d in range(5):
            count = day_counts.get(d, 0)
            pct = count / len(days) * 100 if len(days) > 0 else 0
            lines.append(f"    {day_names.get(d, '?')}: {count:>5} bars ({pct:.1f}%)")

        return '\n'.join(lines)

    def report_forward_outcomes_summary(self) -> str:
        """Report forward outcome statistics."""
        lines = []
        lines.append(f"\n  {'='*60}")
        lines.append(f"  FORWARD OUTCOME STATISTICS")
        lines.append(f"  {'='*60}")

        if not self.snapshots:
            lines.append("  No snapshot data available.")
            return '\n'.join(lines)

        # Collect forward outcome data
        horizons = [1, 2, 4, 6, 8, 12, 24]

        lines.append(f"\n  Returns by horizon (pips):")
        lines.append(f"  {'Horizon':>8} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Pos%':>6}")
        lines.append(f"  {'-'*55}")

        for h in horizons:
            key = f'fwd_{h}h_return_pips'
            values = [s.get(key) for s in self.snapshots if s.get(key) is not None]
            if not values:
                continue
            arr = np.array(values)
            pos_pct = (arr > 0).sum() / len(arr) * 100
            lines.append(f"  {h:>5}h   | {arr.mean():>8.1f} | {arr.std():>8.1f} | "
                        f"{arr.min():>8.1f} | {arr.max():>8.1f} | {pos_pct:>5.1f}%")

        # R-target hit rates
        lines.append(f"\n  R-target hit rates (24h window):")
        r_targets = [
            ('MR Buy 1R', 'fwd_mr_buy_1r'), ('MR Buy 2R', 'fwd_mr_buy_2r'),
            ('MR Sell 1R', 'fwd_mr_sell_1r'), ('MR Sell 2R', 'fwd_mr_sell_2r'),
            ('BO Buy 1R', 'fwd_bo_buy_1r'), ('BO Buy 2R', 'fwd_bo_buy_2r'),
            ('BO Sell 1R', 'fwd_bo_sell_1r'), ('BO Sell 2R', 'fwd_bo_sell_2r'),
        ]

        for label, key in r_targets:
            values = [s.get(key) for s in self.snapshots if s.get(key) is not None]
            if not values:
                continue
            hit_rate = sum(1 for v in values if v) / len(values) * 100
            lines.append(f"    {label:<15}: {hit_rate:>5.1f}% ({sum(1 for v in values if v):,} / {len(values):,})")

        # MFE/MAE
        lines.append(f"\n  MFE/MAE statistics (pips):")
        for key, label in [
            ('fwd_24h_mfe_up_pips', 'MFE Up'),
            ('fwd_24h_mfe_down_pips', 'MFE Down'),
            ('fwd_24h_mae_up_pips', 'MAE Up (buy DD)'),
            ('fwd_24h_mae_down_pips', 'MAE Down (sell DD)'),
        ]:
            values = [s.get(key) for s in self.snapshots if s.get(key) is not None]
            if not values:
                continue
            arr = np.array(values)
            lines.append(f"    {label:<20}: mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
                        f"p75={np.percentile(arr, 75):.1f}, p90={np.percentile(arr, 90):.1f}")

        return '\n'.join(lines)

    def report_smc_summary(self) -> str:
        """Report SMC structure statistics."""
        lines = []
        lines.append(f"\n  {'='*60}")
        lines.append(f"  SMC STRUCTURE SUMMARY")
        lines.append(f"  {'='*60}")

        if not self.snapshots:
            return '\n'.join(lines)

        smc_available = [s for s in self.snapshots if s.get('smc_available')]
        lines.append(f"  SMC data available: {len(smc_available):,} / {len(self.snapshots):,} bars "
                    f"({len(smc_available)/len(self.snapshots)*100:.1f}%)")

        if smc_available:
            # Bias distribution
            biases = Counter(s.get('smc_htf_bias', 'neutral') for s in smc_available)
            lines.append(f"\n  HTF Bias distribution:")
            for bias, count in biases.most_common():
                pct = count / len(smc_available) * 100
                lines.append(f"    {bias:<30}: {count:>5} ({pct:.1f}%)")

            # BOS type distribution
            bos_types = Counter(s.get('smc_last_bos_type', 'none') for s in smc_available)
            lines.append(f"\n  Last BOS type:")
            for btype, count in bos_types.most_common():
                pct = count / len(smc_available) * 100
                lines.append(f"    {btype:<20}: {count:>5} ({pct:.1f}%)")

        return '\n'.join(lines)

    def generate_full_report(self) -> str:
        """Generate complete report."""
        if not self.load_data():
            return f"[ERROR] Could not load data for {self.symbol}"

        sections = [
            self.report_overview(),
            self.report_cluster_distribution(),
            self.report_strategy_dominance(),
            self.report_forward_outcomes_summary(),
            self.report_smc_summary(),
            self.report_temporal_patterns(),
        ]

        report = '\n'.join(sections)
        print(report)

        # Save report
        report_path = MODEL_DIR / f"report_{self.symbol}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n  [SAVED] Report: {report_path}")

        return report


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Market state reports')
    parser.add_argument('--symbol', type=str, help='Report for specific symbol')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    for symbol in symbols:
        reporter = MarketStateReporter(symbol)
        reporter.generate_full_report()


if __name__ == '__main__':
    main()
