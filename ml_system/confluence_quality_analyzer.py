#!/usr/bin/env python3
"""
Confluence Quality Analyzer

Answers: At what confluence score does recovery become unnecessary?
Which confluence scores have the least drawdown?
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class ConfluenceQualityAnalyzer:
    """Analyze trade quality by confluence score"""

    def __init__(self, ml_outputs_dir: str = None):
        if ml_outputs_dir is None:
            project_root = Path(__file__).parent.parent
            ml_outputs_dir = project_root / "ml_system" / "outputs"
        self.outputs_dir = Path(ml_outputs_dir)

        # Load data
        self.recovery_patterns = self._load_json("recovery_pattern_analysis.json")
        self.signal_quality = self._load_json("signal_quality.json")

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file"""
        filepath = self.outputs_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_by_confluence(self) -> Dict:
        """
        Analyze trade quality by confluence score.

        Key metrics:
        - Recovery usage rate (lower = better quality)
        - Win rate (higher = better)
        - Profit with vs without recovery
        - Average profit
        """
        dca_patterns = self.recovery_patterns.get('dca_patterns', {}).get('by_confluence_score', {})
        hedge_patterns = self.recovery_patterns.get('hedge_patterns', {}).get('by_confluence_score', {})

        analysis = []

        # Get all unique confluence scores
        all_scores = set(dca_patterns.keys()) | set(hedge_patterns.keys())

        for score in sorted(all_scores, key=lambda x: int(x)):
            score_int = int(score)
            dca_data = dca_patterns.get(score, {})
            hedge_data = hedge_patterns.get(score, {})

            if not dca_data or dca_data.get('count', 0) == 0:
                continue

            total_trades = dca_data['count']
            trades_with_dca = dca_data['trades_with_dca']
            trades_without_dca = dca_data['trades_without_dca']
            trades_with_hedge = hedge_data.get('trades_with_hedge', 0)
            trades_without_hedge = hedge_data.get('trades_without_hedge', 0)

            # Key metrics
            dca_usage_rate = (trades_with_dca / total_trades * 100) if total_trades > 0 else 0
            hedge_usage_rate = (trades_with_hedge / total_trades * 100) if total_trades > 0 else 0
            recovery_usage_rate = ((trades_with_dca + trades_with_hedge) / (total_trades * 2) * 100)

            win_rate = dca_data.get('win_rate', 0)
            avg_profit = dca_data.get('avg_profit', 0)

            # Profit comparison
            profit_with_dca = dca_data.get('avg_profit_with_dca', 0)
            profit_without_dca = dca_data.get('avg_profit_without_dca', 0)
            profit_with_hedge = hedge_data.get('avg_profit_with_hedge', 0)
            profit_without_hedge = hedge_data.get('avg_profit_without_hedge', 0)

            # Recovery impact (negative = recovery hurts)
            dca_impact = profit_with_dca - profit_without_dca if trades_with_dca > 0 else 0
            hedge_impact = profit_with_hedge - profit_without_hedge if trades_with_hedge > 0 else 0
            total_recovery_impact = dca_impact + hedge_impact

            # Calculate "quality score" (higher = better)
            # Quality = Win rate + (100 - recovery usage rate) - (recovery impact * 10)
            quality_score = win_rate + (100 - recovery_usage_rate) - (abs(total_recovery_impact) * 10)

            analysis.append({
                'confluence': score_int,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'dca_usage_rate': dca_usage_rate,
                'hedge_usage_rate': hedge_usage_rate,
                'recovery_usage_rate': recovery_usage_rate,
                'trades_with_recovery': trades_with_dca + trades_with_hedge,
                'trades_without_recovery': max(trades_without_dca, trades_without_hedge),
                'profit_with_recovery': (profit_with_dca + profit_with_hedge) / 2,
                'profit_without_recovery': (profit_without_dca + profit_without_hedge) / 2,
                'recovery_impact': total_recovery_impact,
                'quality_score': quality_score,
                'avg_dca_levels': dca_data.get('avg_dca_levels', 0),
                'verdict': self._get_verdict(dca_usage_rate, win_rate, total_recovery_impact)
            })

        # Sort by quality score
        analysis.sort(key=lambda x: x['quality_score'], reverse=True)

        return analysis

    def _get_verdict(self, recovery_rate: float, win_rate: float, recovery_impact: float) -> str:
        """Determine verdict for confluence level"""
        if recovery_rate == 0 and win_rate >= 80:
            return "EXCELLENT - No recovery needed"
        elif recovery_rate < 20 and win_rate >= 70:
            return "VERY GOOD - Rarely needs recovery"
        elif recovery_rate < 40 and win_rate >= 60:
            return "GOOD - Moderate quality"
        elif recovery_rate < 60:
            return "FAIR - Often needs recovery"
        else:
            return "POOR - Frequently needs recovery"

    def find_optimal_threshold(self, analysis: List[Dict]) -> Dict:
        """Find optimal confluence threshold"""

        # Find scores with best characteristics
        excellent_scores = [a for a in analysis if a['recovery_usage_rate'] < 30 and a['win_rate'] >= 70]
        good_scores = [a for a in analysis if a['recovery_usage_rate'] < 50 and a['win_rate'] >= 60]

        if excellent_scores:
            recommended_min = min(a['confluence'] for a in excellent_scores)
            recommended_max = max(a['confluence'] for a in excellent_scores)
        elif good_scores:
            recommended_min = min(a['confluence'] for a in good_scores)
            recommended_max = max(a['confluence'] for a in good_scores)
        else:
            recommended_min = 9
            recommended_max = 20

        # Calculate statistics for different thresholds
        thresholds = {}
        for threshold in [8, 9, 10, 11, 12, 13, 15, 17, 20]:
            above_threshold = [a for a in analysis if a['confluence'] >= threshold]
            if not above_threshold:
                continue

            total_trades = sum(a['total_trades'] for a in above_threshold)
            avg_win_rate = sum(a['win_rate'] * a['total_trades'] for a in above_threshold) / total_trades if total_trades > 0 else 0
            avg_recovery_rate = sum(a['recovery_usage_rate'] * a['total_trades'] for a in above_threshold) / total_trades if total_trades > 0 else 0
            avg_profit = sum(a['avg_profit'] * a['total_trades'] for a in above_threshold) / total_trades if total_trades > 0 else 0

            thresholds[threshold] = {
                'min_confluence': threshold,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'avg_recovery_rate': avg_recovery_rate,
                'avg_profit': avg_profit,
                'score': avg_win_rate - avg_recovery_rate + (avg_profit * 10)
            }

        # Find best threshold
        best_threshold = max(thresholds.items(), key=lambda x: x[1]['score'])

        return {
            'recommended_range': [recommended_min, recommended_max],
            'best_threshold': best_threshold[0],
            'threshold_analysis': thresholds,
            'excellent_scores': [a['confluence'] for a in excellent_scores],
            'good_scores': [a['confluence'] for a in good_scores]
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive confluence quality report"""
        analysis = self.analyze_by_confluence()
        optimal = self.find_optimal_threshold(analysis)

        # Get best and worst performers
        best_3 = analysis[:3]
        worst_3 = analysis[-3:]

        return {
            'summary': {
                'total_confluence_levels_analyzed': len(analysis),
                'best_confluence_score': best_3[0]['confluence'] if best_3 else None,
                'worst_confluence_score': worst_3[-1]['confluence'] if worst_3 else None,
                'recommended_min_threshold': optimal['best_threshold'],
            },
            'question_answer': {
                'question': 'At what confluence does recovery become unnecessary?',
                'answer': f"Confluence {optimal['best_threshold']}+ shows the best quality",
                'excellent_levels': optimal['excellent_scores'],
                'good_levels': optimal['good_scores'],
                'explanation': f"Trades at {optimal['best_threshold']}+ have lower recovery usage and better win rates"
            },
            'by_confluence': analysis,
            'optimal_threshold': optimal,
            'best_performers': best_3,
            'worst_performers': worst_3,
            'recommendations': self._generate_recommendations(analysis, optimal)
        }

    def _generate_recommendations(self, analysis: List[Dict], optimal: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recs = []

        best_threshold = optimal['best_threshold']
        threshold_data = optimal['threshold_analysis'].get(best_threshold, {})

        recs.append(f"Set MIN_CONFLUENCE_SCORE = {best_threshold}")
        recs.append(f"Expected win rate: {threshold_data.get('avg_win_rate', 0):.1f}%")
        recs.append(f"Expected recovery usage: {threshold_data.get('avg_recovery_rate', 0):.1f}%")

        # Analyze specific problem scores
        high_recovery = [a for a in analysis if a['recovery_usage_rate'] > 50]
        if high_recovery:
            scores = [str(a['confluence']) for a in high_recovery]
            recs.append(f"AVOID confluence scores: {', '.join(scores)} (high recovery usage)")

        # Find sweet spot
        sweet_spot = [a for a in analysis if a['recovery_usage_rate'] < 30 and a['win_rate'] >= 75]
        if sweet_spot:
            scores = [str(a['confluence']) for a in sweet_spot]
            recs.append(f"SWEET SPOT: Confluence {', '.join(scores)} (low recovery, high win rate)")

        return recs

    def print_report(self, report: Dict):
        """Print human-readable report"""
        print("=" * 80)
        print("CONFLUENCE QUALITY ANALYSIS")
        print("=" * 80)
        print()

        # Answer the question
        qa = report['question_answer']
        print(f"QUESTION: {qa['question']}")
        print(f"ANSWER: {qa['answer']}")
        print()
        print(f"Explanation: {qa['explanation']}")
        print()

        if qa['excellent_levels']:
            print(f"EXCELLENT confluence scores: {', '.join(map(str, qa['excellent_levels']))}")
        if qa['good_levels']:
            print(f"GOOD confluence scores: {', '.join(map(str, qa['good_levels']))}")
        print()

        # Best performers
        print("=" * 80)
        print("TOP 3 BEST CONFLUENCE SCORES (Highest Quality)")
        print("=" * 80)
        for i, conf in enumerate(report['best_performers'], 1):
            print(f"{i}. Confluence {conf['confluence']} - {conf['verdict']}")
            print(f"   Trades: {conf['total_trades']}")
            print(f"   Win Rate: {conf['win_rate']:.1f}%")
            print(f"   Avg Profit: ${conf['avg_profit']:.2f}")
            print(f"   Recovery Usage: {conf['recovery_usage_rate']:.1f}%")
            print(f"   Profit WITHOUT Recovery: ${conf['profit_without_recovery']:.2f}")
            print(f"   Profit WITH Recovery: ${conf['profit_with_recovery']:.2f}")
            print(f"   Recovery Impact: ${conf['recovery_impact']:.2f}")
            print(f"   Quality Score: {conf['quality_score']:.1f}")
            print()

        # Worst performers
        print("=" * 80)
        print("BOTTOM 3 WORST CONFLUENCE SCORES (Lowest Quality)")
        print("=" * 80)
        for i, conf in enumerate(report['worst_performers'], 1):
            print(f"{i}. Confluence {conf['confluence']} - {conf['verdict']}")
            print(f"   Trades: {conf['total_trades']}")
            print(f"   Win Rate: {conf['win_rate']:.1f}%")
            print(f"   Avg Profit: ${conf['avg_profit']:.2f}")
            print(f"   Recovery Usage: {conf['recovery_usage_rate']:.1f}%")
            print(f"   Profit WITHOUT Recovery: ${conf['profit_without_recovery']:.2f}")
            print(f"   Profit WITH Recovery: ${conf['profit_with_recovery']:.2f}")
            print(f"   Recovery Impact: ${conf['recovery_impact']:.2f}")
            print(f"   Quality Score: {conf['quality_score']:.1f}")
            print()

        # All scores ranked
        print("=" * 80)
        print("ALL CONFLUENCE SCORES (Ranked by Quality)")
        print("=" * 80)
        print(f"{'Conf':>5} {'Trades':>7} {'Win%':>6} {'AvgP$':>8} {'RecUse%':>8} {'WithRec$':>9} {'NoRec$':>9} {'Impact$':>9} {'Quality':>8}")
        print("-" * 80)
        for conf in report['by_confluence']:
            print(f"{conf['confluence']:>5} {conf['total_trades']:>7} {conf['win_rate']:>6.1f} "
                  f"{conf['avg_profit']:>8.2f} {conf['recovery_usage_rate']:>8.1f} "
                  f"{conf['profit_with_recovery']:>9.2f} {conf['profit_without_recovery']:>9.2f} "
                  f"{conf['recovery_impact']:>9.2f} {conf['quality_score']:>8.1f}")
        print()

        # Threshold analysis
        print("=" * 80)
        print("THRESHOLD ANALYSIS (What happens at different MIN_CONFLUENCE settings)")
        print("=" * 80)
        print(f"{'MinConf':>8} {'Trades':>7} {'WinRate%':>9} {'RecRate%':>9} {'AvgProfit$':>11} {'Score':>8}")
        print("-" * 80)
        for threshold, data in sorted(report['optimal_threshold']['threshold_analysis'].items()):
            print(f"{threshold:>8} {data['total_trades']:>7} {data['avg_win_rate']:>9.1f} "
                  f"{data['avg_recovery_rate']:>9.1f} {data['avg_profit']:>11.2f} {data['score']:>8.1f}")
        print()

        # Recommendations
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        print()
        print("=" * 80)

    def save_report(self, report: Dict):
        """Save report to file"""
        output_path = self.outputs_dir / "confluence_quality_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return output_path


def main():
    analyzer = ConfluenceQualityAnalyzer()
    report = analyzer.generate_report()
    analyzer.print_report(report)
    output_path = analyzer.save_report(report)
    print(f"Full report saved to: {output_path}")


if __name__ == '__main__':
    main()
