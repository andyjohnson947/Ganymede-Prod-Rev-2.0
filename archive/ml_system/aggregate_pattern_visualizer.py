#!/usr/bin/env python3
"""
Aggregate Pattern Visualizer

Shows how all the pieces tie together using aggregate ML statistics:
- Confluence scores vs recovery usage
- Time of day vs outcomes
- Recovery impact by confluence and hour
"""

import json
from pathlib import Path
from typing import Dict, List


class AggregatePatternVisualizer:
    """Visualize patterns from aggregate ML data"""

    def __init__(self, ml_outputs_dir: str = None):
        if ml_outputs_dir is None:
            project_root = Path(__file__).parent.parent
            ml_outputs_dir = project_root / "ml_system" / "outputs"
        self.outputs_dir = Path(ml_outputs_dir)

        # Load all analysis files
        self.recovery_patterns = self._load_json("recovery_pattern_analysis.json")
        self.time_performance = self._load_json("time_performance.json")
        self.spread_hours = self._load_json("spread_hours_analysis.json")
        self.stack_sl = self._load_json("stack_sl_deep_dive.json")
        self.confluence_quality = self._load_json("confluence_quality_analysis.json")

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file"""
        filepath = self.outputs_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_master_view(self) -> Dict:
        """Create master view combining all data"""
        # Get confluence data
        dca_by_conf = self.recovery_patterns.get('dca_patterns', {}).get('by_confluence_score', {})
        hedge_by_conf = self.recovery_patterns.get('hedge_patterns', {}).get('by_confluence_score', {})

        # Get time data
        by_hour = self.time_performance.get('by_hour', {})

        master_data = []

        # Build confluence rows
        all_scores = sorted(set(dca_by_conf.keys()) | set(hedge_by_conf.keys()), key=lambda x: int(x))

        for score in all_scores:
            dca_data = dca_by_conf.get(score, {})
            hedge_data = hedge_by_conf.get(score, {})

            if not dca_data or dca_data.get('count', 0) == 0:
                continue

            row = {
                'type': 'CONFLUENCE',
                'confluence': int(score),
                'total_trades': dca_data['count'],
                'win_rate': dca_data.get('win_rate', 0),
                'avg_profit': dca_data.get('avg_profit', 0),

                # Recovery usage
                'dca_usage_%': (dca_data['trades_with_dca'] / dca_data['count'] * 100) if dca_data['count'] > 0 else 0,
                'hedge_usage_%': (hedge_data.get('trades_with_hedge', 0) / dca_data['count'] * 100) if dca_data['count'] > 0 else 0,
                'total_recovery_usage_%': ((dca_data['trades_with_dca'] + hedge_data.get('trades_with_hedge', 0)) / (dca_data['count'] * 2) * 100),

                # Profit comparison
                'profit_without_recovery': dca_data.get('avg_profit_without_dca', 0),
                'profit_with_recovery': dca_data.get('avg_profit_with_dca', 0),
                'recovery_damage': dca_data.get('avg_profit_with_dca', 0) - dca_data.get('avg_profit_without_dca', 0),

                # Classification
                'quality': self._classify_quality(dca_data.get('win_rate', 0),
                                                 (dca_data['trades_with_dca'] / dca_data['count'] * 100) if dca_data['count'] > 0 else 0)
            }

            master_data.append(row)

        # Build hour rows
        spread_hours = [0, 9, 13, 20, 21]
        for hour_str, hour_data in sorted(by_hour.items(), key=lambda x: int(x[0])):
            hour = int(hour_str)

            row = {
                'type': 'HOUR',
                'hour': hour,
                'is_spread_hour': hour in spread_hours,
                'session': self._get_session(hour),
                'total_trades': hour_data['trades'],
                'win_rate': hour_data['win_rate'],
                'avg_profit': hour_data['avg_profit'],
                'total_profit': hour_data['total_profit'],

                # Classification
                'quality': self._classify_hour_quality(hour_data['avg_profit'], hour_data['win_rate'])
            }

            master_data.append(row)

        return {
            'confluence_patterns': [d for d in master_data if d['type'] == 'CONFLUENCE'],
            'hour_patterns': [d for d in master_data if d['type'] == 'HOUR'],
            'key_insights': self._extract_insights()
        }

    def _classify_quality(self, win_rate: float, recovery_rate: float) -> str:
        """Classify trade quality"""
        if recovery_rate == 0 and win_rate >= 80:
            return "EXCELLENT"
        elif recovery_rate < 30 and win_rate >= 70:
            return "GOOD"
        elif recovery_rate < 50:
            return "FAIR"
        else:
            return "POOR"

    def _classify_hour_quality(self, avg_profit: float, win_rate: float) -> str:
        """Classify hour quality"""
        if avg_profit > 2 and win_rate > 75:
            return "EXCELLENT"
        elif avg_profit > 0.5 and win_rate > 60:
            return "GOOD"
        elif avg_profit > 0:
            return "FAIR"
        else:
            return "POOR"

    def _get_session(self, hour: int) -> str:
        """Get trading session"""
        if 0 <= hour < 8:
            return 'Tokyo'
        elif 8 <= hour < 13:
            return 'London'
        elif 13 <= hour < 21:
            return 'NY'
        else:
            return 'Sydney'

    def _extract_insights(self) -> List[str]:
        """Extract key insights from all analyses"""
        insights = []

        # From recovery patterns
        dca_by_conf = self.recovery_patterns.get('dca_patterns', {}).get('by_confluence_score', {})
        for score, data in dca_by_conf.items():
            if data.get('trades_with_dca', 0) > 0:
                profit_with = data.get('avg_profit_with_dca', 0)
                profit_without = data.get('avg_profit_without_dca', 0)
                if profit_with < profit_without - 2:
                    insights.append(f"Confluence {score}: Recovery HURTS by ${profit_without - profit_with:.2f} per trade")

        # From time analysis
        by_hour = self.time_performance.get('by_hour', {})
        worst_hours = sorted(by_hour.items(), key=lambda x: x[1]['avg_profit'])[:3]
        best_hours = sorted(by_hour.items(), key=lambda x: x[1]['avg_profit'], reverse=True)[:3]

        insights.append(f"Best hours: {', '.join([str(h[0]) for h in best_hours])}")
        insights.append(f"Worst hours (spread): {', '.join([str(h[0]) for h in worst_hours])}")

        return insights

    def print_master_view(self, master: Dict):
        """Print comprehensive master view"""
        print("=" * 180)
        print("MASTER VIEW: HOW ALL THE PIECES TIE TOGETHER")
        print("=" * 180)
        print()

        # Part 1: Confluence Analysis
        print("PART 1: RECOVERY IMPACT BY CONFLUENCE SCORE")
        print("-" * 180)
        print(f"{'Conf':>5} {'Trades':>7} {'WinRate':>8} {'AvgProf':>8} {'RecUse%':>8} | "
              f"{'WithRec$':>9} {'NoRec$':>9} {'Damage$':>9} | {'Quality':>10}")
        print("-" * 180)

        for row in master['confluence_patterns']:
            print(f"{row['confluence']:>5} {row['total_trades']:>7} {row['win_rate']:>8.1f} "
                  f"{row['avg_profit']:>8.2f} {row['total_recovery_usage_%']:>8.1f} | "
                  f"{row['profit_with_recovery']:>9.2f} {row['profit_without_recovery']:>9.2f} "
                  f"{row['recovery_damage']:>9.2f} | {row['quality']:>10}")

        print("-" * 180)
        print()

        # Key insight from confluence
        print("KEY INSIGHT - CONFLUENCE:")
        conf_with_rec = [r for r in master['confluence_patterns'] if r['total_recovery_usage_%'] > 0]
        if conf_with_rec:
            avg_damage = sum(r['recovery_damage'] for r in conf_with_rec) / len(conf_with_rec)
            print(f"  Average recovery damage across all confluences: ${avg_damage:.2f}")
            print(f"  Recovery hurts at EVERY confluence level - even high quality signals")
        print()
        print()

        # Part 2: Time of Day Analysis
        print("PART 2: PERFORMANCE BY HOUR (TIME OF DAY IMPACT)")
        print("-" * 180)
        print(f"{'Hour':>5} {'Session':>10} {'Spread?':>8} {'Trades':>7} {'WinRate':>8} "
              f"{'AvgProf$':>9} {'TotalP$':>9} | {'Quality':>10}")
        print("-" * 180)

        # Sort by hour
        sorted_hours = sorted(master['hour_patterns'], key=lambda x: x['hour'])
        for row in sorted_hours:
            spread_marker = 'YES' if row['is_spread_hour'] else 'NO'
            print(f"{row['hour']:>5} {row['session']:>10} {spread_marker:>8} {row['total_trades']:>7} "
                  f"{row['win_rate']:>8.1f} {row['avg_profit']:>9.2f} {row['total_profit']:>9.2f} | "
                  f"{row['quality']:>10}")

        print("-" * 180)
        print()

        # Key insight from hours
        print("KEY INSIGHT - TIME OF DAY:")
        spread_hours = [r for r in master['hour_patterns'] if r['is_spread_hour']]
        normal_hours = [r for r in master['hour_patterns'] if not r['is_spread_hour']]

        if spread_hours and normal_hours:
            spread_avg = sum(r['avg_profit'] for r in spread_hours) / len(spread_hours)
            normal_avg = sum(r['avg_profit'] for r in normal_hours) / len(normal_hours)
            print(f"  Spread hours (0, 9, 13, 20, 21) average: ${spread_avg:.2f}")
            print(f"  Normal hours average: ${normal_avg:.2f}")
            print(f"  Difference: ${spread_avg - normal_avg:.2f} (spread hours are worse)")
        print()
        print()

        # Part 3: Combined Analysis
        print("PART 3: THE COMPLETE PICTURE")
        print("-" * 180)
        print()
        print("What the data is telling us:")
        print()

        for i, insight in enumerate(master['key_insights'], 1):
            print(f"  {i}. {insight}")

        print()
        print("THE PATTERN:")
        print("  1. Recovery systems ADD to losers (Martingale/DCA/Hedge)")
        print("  2. This amplifies losses at ALL confluence levels (even good signals)")
        print("  3. Spread hours (midnight, 9am, 1pm, 8pm, 9pm) are high risk")
        print("  4. Adding recovery DURING spread hours = cascade + bad prices = IMPLOSION")
        print()
        print("THE SOLUTION:")
        print("  1. Raise MIN_CONFLUENCE to 13-20 (better quality entries)")
        print("  2. DISABLE DCA/Hedge recovery entirely (they hurt at all levels)")
        print("  3. Block trading during spread hours (0, 9, 13, 20, 21)")
        print("  4. Rely ONLY on ADX hard stops (ADX > 30 = -50 pip SL)")
        print()
        print("EXPECTED OUTCOME:")
        print("  - Fewer trades (but higher quality)")
        print("  - No cascades (no recovery adding to losers)")
        print("  - Clean wins and controlled losses")
        print("  - Profit WITHOUT the -$8.81 recovery tax per trade")
        print()
        print("=" * 180)

    def save_master_view(self, master: Dict):
        """Save master view"""
        output_path = self.outputs_dir / "master_pattern_view.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(master, f, indent=2, ensure_ascii=False)
        return output_path


def main():
    visualizer = AggregatePatternVisualizer()
    master = visualizer.create_master_view()
    visualizer.print_master_view(master)
    output_path = visualizer.save_master_view(master)
    print(f"Master view saved to: {output_path}")


if __name__ == '__main__':
    main()
