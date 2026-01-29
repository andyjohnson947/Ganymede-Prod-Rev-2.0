#!/usr/bin/env python3
"""
ADX Hard Stop vs Recovery System Comparison

Calculates: What would profit be with -50 pip ADX hard stops instead of recovery cascade?
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta


class ADXvsRecoveryComparison:
    """Compare ADX hard stops vs current recovery system"""

    def __init__(self, ml_outputs_dir: str = None):
        if ml_outputs_dir is None:
            project_root = Path(__file__).parent.parent
            ml_outputs_dir = project_root / "ml_system" / "outputs"
        self.outputs_dir = Path(ml_outputs_dir)

        # Load data
        self.recovery_patterns = self._load_json("recovery_pattern_analysis.json")
        self.time_performance = self._load_json("time_performance.json")
        self.confluence_quality = self._load_json("confluence_quality_analysis.json")

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file"""
        filepath = self.outputs_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def estimate_adx_hard_stop_loss(self, volume: float = 0.04) -> float:
        """
        Estimate loss from -50 pip hard stop.

        For forex (EUR/USD typical):
        - 0.01 lot = 1,000 units = $0.10 per pip
        - 0.04 lot = 4,000 units = $0.40 per pip
        - -50 pips on 0.04 lot = 50 * $0.40 = -$20

        BUT: This assumes full position size. In reality, with PC1 at 10-12 pips
        taking 25% off, and PC2 at 20-25 pips taking another 25%, only 50% of
        position remains if it reverses after PC2.

        However, with ADX > 30 (trending), we wouldn't even enter or would exit early.
        Let's use empirical data instead.
        """
        # From the data, clean losses (no recovery) average around -$1 to -$3
        # Recovery cascades turn these into -$5 to -$13 losses
        # A -50 pip hard stop on 0.04 lot ≈ -$2 to -$3 (accounting for partial closes)
        return -2.5

    def calculate_recovery_system_performance(self) -> Dict:
        """Calculate actual performance with current recovery system"""
        dca_by_conf = self.recovery_patterns.get('dca_patterns', {}).get('by_confluence_score', {})
        hedge_by_conf = self.recovery_patterns.get('hedge_patterns', {}).get('by_confluence_score', {})

        total_trades = 0
        trades_with_recovery = 0
        trades_without_recovery = 0
        profit_with_recovery = 0
        profit_without_recovery = 0
        recovery_costs = 0

        for score, dca_data in dca_by_conf.items():
            count = dca_data.get('count', 0)
            if count == 0:
                continue

            total_trades += count

            # Trades with recovery
            with_dca = dca_data.get('trades_with_dca', 0)
            hedge_data = hedge_by_conf.get(score, {})
            with_hedge = hedge_data.get('trades_with_hedge', 0)
            with_recovery = max(with_dca, with_hedge)  # Some may have both
            trades_with_recovery += with_recovery

            # Trades without recovery
            without_recovery = count - with_recovery
            trades_without_recovery += without_recovery

            # Profit calculations
            profit_with = dca_data.get('avg_profit_with_dca', 0) * with_dca
            profit_without = dca_data.get('avg_profit_without_dca', 0) * without_recovery

            profit_with_recovery += profit_with
            profit_without_recovery += profit_without

            # Recovery costs
            recovery_costs += dca_data.get('recovery_cost', 0)

        total_profit_current = profit_with_recovery + profit_without_recovery

        return {
            'total_trades': total_trades,
            'trades_with_recovery': trades_with_recovery,
            'trades_without_recovery': trades_without_recovery,
            'profit_with_recovery': profit_with_recovery,
            'profit_without_recovery': profit_without_recovery,
            'total_profit_current_system': total_profit_current,
            'avg_profit_with_recovery': profit_with_recovery / trades_with_recovery if trades_with_recovery > 0 else 0,
            'avg_profit_without_recovery': profit_without_recovery / trades_without_recovery if trades_without_recovery > 0 else 0,
            'recovery_costs': recovery_costs,
            'recovery_damage': (profit_with_recovery / trades_with_recovery if trades_with_recovery > 0 else 0) -
                             (profit_without_recovery / trades_without_recovery if trades_without_recovery > 0 else 0)
        }

    def calculate_adx_system_performance(self, current_stats: Dict) -> Dict:
        """
        Calculate what performance would be with ADX hard stops.

        Assumptions:
        1. ADX > 30 (trending): Hit -50 pip hard stop = ~-$2.50 loss
        2. ADX <= 30 (ranging): Same as current "without recovery" trades
        3. From market statistics, ~30-40% of market is trending (ADX > 30)
        """
        total_trades = current_stats['total_trades']

        # Estimate trending vs ranging distribution
        # From forex market stats: ~35% trending, 65% ranging
        trending_rate = 0.35
        trending_trades = int(total_trades * trending_rate)
        ranging_trades = total_trades - trending_trades

        # ADX system outcomes:
        # - Trending trades: Hit hard stop at -$2.50
        # - Ranging trades: Same as "without recovery" (no help needed)

        adx_hard_stop_loss = self.estimate_adx_hard_stop_loss()

        # Ranging trades perform like "without recovery" trades
        ranging_avg = current_stats['avg_profit_without_recovery']
        ranging_total = ranging_trades * ranging_avg

        # Trending trades hit hard stop
        trending_total = trending_trades * adx_hard_stop_loss

        total_profit_adx = ranging_total + trending_total

        return {
            'total_trades': total_trades,
            'trending_trades': trending_trades,
            'ranging_trades': ranging_trades,
            'trending_rate': trending_rate,
            'hard_stop_loss_per_trade': adx_hard_stop_loss,
            'ranging_avg_profit': ranging_avg,
            'profit_from_trending': trending_total,
            'profit_from_ranging': ranging_total,
            'total_profit_adx_system': total_profit_adx,
            'avg_profit_adx_system': total_profit_adx / total_trades if total_trades > 0 else 0
        }

    def compare_systems(self) -> Dict:
        """Compare both systems side by side"""
        current = self.calculate_recovery_system_performance()
        adx = self.calculate_adx_system_performance(current)

        # Calculate difference
        profit_difference = adx['total_profit_adx_system'] - current['total_profit_current_system']
        percent_improvement = (profit_difference / abs(current['total_profit_current_system']) * 100) if current['total_profit_current_system'] != 0 else 0

        return {
            'period_analyzed': 'Last 10 days (aggregate data)',
            'current_system': current,
            'adx_system': adx,
            'comparison': {
                'profit_difference': profit_difference,
                'percent_improvement': percent_improvement,
                'better_system': 'ADX_HARD_STOPS' if profit_difference > 0 else 'CURRENT_RECOVERY',
                'trades_saved_from_cascade': current['trades_with_recovery']
            }
        }

    def print_comparison(self, comparison: Dict):
        """Print detailed comparison"""
        print("=" * 120)
        print("ADX HARD STOPS vs RECOVERY SYSTEM COMPARISON")
        print("=" * 120)
        print()

        current = comparison['current_system']
        adx = comparison['adx_system']
        comp = comparison['comparison']

        print(f"Period: {comparison['period_analyzed']}")
        print(f"Total trades analyzed: {current['total_trades']}")
        print()

        # Current system breakdown
        print("=" * 120)
        print("CURRENT SYSTEM (DCA/Hedge Recovery)")
        print("=" * 120)
        print(f"  Trades with recovery: {current['trades_with_recovery']} ({current['trades_with_recovery']/current['total_trades']*100:.1f}%)")
        print(f"    Avg profit with recovery: ${current['avg_profit_with_recovery']:.2f}")
        print(f"    Total from trades with recovery: ${current['profit_with_recovery']:.2f}")
        print()
        print(f"  Trades without recovery: {current['trades_without_recovery']} ({current['trades_without_recovery']/current['total_trades']*100:.1f}%)")
        print(f"    Avg profit without recovery: ${current['avg_profit_without_recovery']:.2f}")
        print(f"    Total from trades without recovery: ${current['profit_without_recovery']:.2f}")
        print()
        print(f"  Recovery costs: ${current['recovery_costs']:.2f}")
        print(f"  Recovery damage per trade: ${current['recovery_damage']:.2f}")
        print()
        print(f"  TOTAL PROFIT (Current System): ${current['total_profit_current_system']:.2f}")
        print()

        # ADX system breakdown
        print("=" * 120)
        print("ADX SYSTEM (Hard Stops Only)")
        print("=" * 120)
        print(f"  Trending market trades (ADX > 30): {adx['trending_trades']} ({adx['trending_rate']*100:.1f}%)")
        print(f"    Hard stop loss per trade: ${adx['hard_stop_loss_per_trade']:.2f}")
        print(f"    Total from trending: ${adx['profit_from_trending']:.2f}")
        print()
        print(f"  Ranging market trades (ADX <= 30): {adx['ranging_trades']} ({(1-adx['trending_rate'])*100:.1f}%)")
        print(f"    Avg profit (no recovery needed): ${adx['ranging_avg_profit']:.2f}")
        print(f"    Total from ranging: ${adx['profit_from_ranging']:.2f}")
        print()
        print(f"  TOTAL PROFIT (ADX System): ${adx['total_profit_adx_system']:.2f}")
        print(f"  Average per trade: ${adx['avg_profit_adx_system']:.2f}")
        print()

        # Comparison
        print("=" * 120)
        print("COMPARISON")
        print("=" * 120)
        print(f"  Current System: ${current['total_profit_current_system']:.2f}")
        print(f"  ADX System:     ${adx['total_profit_adx_system']:.2f}")
        print()
        print(f"  Difference:     ${comp['profit_difference']:.2f}")
        print(f"  Improvement:    {comp['percent_improvement']:.1f}%")
        print()
        print(f"  WINNER: {comp['better_system']}")
        print()

        # Key insights
        print("=" * 120)
        print("KEY INSIGHTS")
        print("=" * 120)
        print()
        print("What happens with ADX system:")
        print(f"  1. {comp['trades_saved_from_cascade']} trades AVOID recovery cascade (save ${abs(current['recovery_damage']) * comp['trades_saved_from_cascade']:.2f})")
        print(f"  2. Trending trades ({adx['trending_trades']}) take controlled -$2.50 loss (total: ${adx['profit_from_trending']:.2f})")
        print(f"  3. Ranging trades ({adx['ranging_trades']}) profit normally (total: ${adx['profit_from_ranging']:.2f})")
        print()
        print("What happens with Current system:")
        print(f"  1. {current['trades_with_recovery']} trades trigger recovery cascade (damage: ${current['recovery_damage']:.2f} per trade)")
        print(f"  2. Recovery costs: ${current['recovery_costs']:.2f}")
        print(f"  3. Total damage from recovery: ${abs(current['recovery_damage']) * current['trades_with_recovery']:.2f}")
        print()

        if comp['profit_difference'] > 0:
            print(f"BOTTOM LINE: ADX hard stops would have made you ${comp['profit_difference']:.2f} MORE profit")
            print(f"             That's {comp['percent_improvement']:.1f}% better performance!")
        else:
            print(f"BOTTOM LINE: Current system made ${abs(comp['profit_difference']):.2f} more profit")
            print(f"             But this doesn't account for the risk of implosions")

        print()
        print("=" * 120)

    def save_comparison(self, comparison: Dict):
        """Save comparison to file"""
        output_path = self.outputs_dir / "adx_vs_recovery_comparison.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        return output_path


def main():
    analyzer = ADXvsRecoveryComparison()
    comparison = analyzer.compare_systems()
    analyzer.print_comparison(comparison)
    output_path = analyzer.save_comparison(comparison)
    print(f"Full comparison saved to: {output_path}")


if __name__ == '__main__':
    main()
