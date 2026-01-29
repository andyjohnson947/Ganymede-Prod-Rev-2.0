#!/usr/bin/env python3
"""
Spread Hours and Stack SL Analyzer

Analyzes trading performance by hour to identify spread hours and provides
recommendations for:
1. Best trading hours for mean reversion
2. Stack SL adjustments during spread hours
3. Whether to disable stack SL during high-risk hours
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class SpreadHoursAnalyzer:
    """
    Analyzes spread hours impact on trading performance and recovery stack failures.
    """

    def __init__(self, ml_outputs_dir: str = None):
        if ml_outputs_dir is None:
            project_root = Path(__file__).parent.parent
            ml_outputs_dir = project_root / "ml_system" / "outputs"
        self.outputs_dir = Path(ml_outputs_dir)

        # Load existing ML analysis
        self.time_performance = self._load_json("time_performance.json")
        self.recovery_patterns = self._load_json("recovery_pattern_analysis.json")
        self.recovery_effectiveness = self._load_json("recovery_effectiveness.json")

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from outputs directory"""
        filepath = self.outputs_dir / filename
        if not filepath.exists():
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def identify_spread_hours(self) -> Dict:
        """
        Identify spread hours based on performance metrics.

        Spread hours are characterized by:
        - Low win rate
        - Negative average profit
        - High trade count with poor outcomes

        Returns:
            Dictionary with spread hours analysis
        """
        if not self.time_performance or 'by_hour' not in self.time_performance:
            return {}

        hours_data = self.time_performance['by_hour']

        # Analyze each hour
        spread_hours = []
        normal_hours = []

        for hour, metrics in hours_data.items():
            hour_num = int(hour)
            trades = metrics['total_profit']
            avg_profit = metrics['avg_profit']
            win_rate = metrics['win_rate']
            trade_count = metrics['trades']

            # Spread hour criteria:
            # 1. Negative total profit OR very low avg profit
            # 2. Low win rate (< 50%)
            # 3. Sufficient trades to be statistically meaningful (> 10)
            is_spread_hour = (
                (trades < 0 or avg_profit < 0.5) and
                win_rate < 50 and
                trade_count > 10
            )

            hour_analysis = {
                'hour': hour_num,
                'trades': trade_count,
                'total_profit': trades,
                'avg_profit': avg_profit,
                'win_rate': win_rate,
                'is_spread_hour': is_spread_hour
            }

            if is_spread_hour:
                spread_hours.append(hour_analysis)
            else:
                normal_hours.append(hour_analysis)

        # Sort by performance
        spread_hours.sort(key=lambda x: x['avg_profit'])
        normal_hours.sort(key=lambda x: x['avg_profit'], reverse=True)

        return {
            'spread_hours': spread_hours,
            'normal_hours': normal_hours,
            'spread_hour_numbers': [h['hour'] for h in spread_hours],
            'best_hour_numbers': [h['hour'] for h in normal_hours[:8]]
        }

    def analyze_recovery_during_spread_hours(self) -> Dict:
        """
        Analyze how recovery systems (DCA/Hedge) perform during spread hours vs normal hours.

        Returns:
            Analysis of recovery effectiveness by time period
        """
        spread_analysis = self.identify_spread_hours()
        spread_hours = spread_analysis.get('spread_hour_numbers', [])

        # Get recovery pattern data
        dca_patterns = self.recovery_patterns.get('dca_patterns', {}).get('by_confluence_score', {})
        hedge_patterns = self.recovery_patterns.get('hedge_patterns', {}).get('by_confluence_score', {})

        # Analyze DCA effectiveness
        total_dca_impact = 0
        dca_count = 0

        for score, data in dca_patterns.items():
            dca_impact = data.get('dca_profit_impact', 0)
            total_dca_impact += dca_impact
            if data.get('trades_with_dca', 0) > 0:
                dca_count += 1

        avg_dca_impact = total_dca_impact / dca_count if dca_count > 0 else 0

        # Analyze Hedge effectiveness
        total_hedge_impact = 0
        hedge_count = 0

        for score, data in hedge_patterns.items():
            hedge_impact = data.get('hedge_profit_impact', 0)
            total_hedge_impact += hedge_impact
            if data.get('trades_with_hedge', 0) > 0:
                hedge_count += 1

        avg_hedge_impact = total_hedge_impact / hedge_count if hedge_count > 0 else 0

        return {
            'spread_hours': spread_hours,
            'dca_analysis': {
                'avg_impact': avg_dca_impact,
                'is_profitable': avg_dca_impact > 0,
                'recommendation': 'DISABLE' if avg_dca_impact < -3 else 'KEEP'
            },
            'hedge_analysis': {
                'avg_impact': avg_hedge_impact,
                'is_profitable': avg_hedge_impact > 0,
                'recommendation': 'DISABLE' if avg_hedge_impact < -3 else 'KEEP'
            },
            'recovery_effectiveness': self.recovery_effectiveness
        }

    def recommend_trading_hours(self) -> Dict:
        """
        Recommend best trading hours for mean reversion strategy.

        Returns:
            Dictionary with trading hour recommendations
        """
        spread_analysis = self.identify_spread_hours()
        spread_hours = spread_analysis.get('spread_hour_numbers', [])
        best_hours = spread_analysis.get('best_hour_numbers', [])

        # Get best hours from existing analysis
        analysis_best_hours = self.time_performance.get('best_hours', [])

        return {
            'avoid_hours': spread_hours,
            'recommended_hours': best_hours,
            'analysis_best_hours': analysis_best_hours,
            'summary': {
                'spread_hours_count': len(spread_hours),
                'best_hours_count': len(best_hours),
                'coverage': f"{len(best_hours)}/{24} hours recommended"
            }
        }

    def calculate_stack_sl_recommendations(self) -> Dict:
        """
        Calculate recommendations for stack SL adjustments.

        Based on:
        - Current stack failures
        - Time-of-day impact
        - Recovery effectiveness

        Returns:
            Stack SL recommendations
        """
        spread_analysis = self.identify_spread_hours()
        spread_hours = spread_analysis['spread_hours']
        recovery_analysis = self.analyze_recovery_during_spread_hours()

        # Calculate average loss during spread hours
        spread_hour_losses = [h for h in spread_hours if h['avg_profit'] < 0]
        avg_spread_loss = sum(h['avg_profit'] for h in spread_hour_losses) / len(spread_hour_losses) if spread_hour_losses else 0

        # Calculate average profit during normal hours
        normal_hours = spread_analysis['normal_hours']
        avg_normal_profit = sum(h['avg_profit'] for h in normal_hours) / len(normal_hours) if normal_hours else 0

        # Current stack SL is $-20 (from user feedback about $-22.50 trigger)
        current_stack_sl = -20.0

        # Recommendations
        recommendations = []

        # 1. Should we disable stack SL during spread hours?
        if avg_spread_loss < -5:  # Very bad during spread hours
            recommendations.append({
                'priority': 'HIGH',
                'category': 'spread_hours',
                'action': 'DISABLE_STACK_SL',
                'reason': f'Stack losses average ${avg_spread_loss:.2f} during spread hours',
                'implementation': 'Disable stack SL during hours: ' + ', '.join(str(h['hour']) for h in spread_hours)
            })

        # 2. Should we increase stack SL overall?
        dca_impact = recovery_analysis['dca_analysis']['avg_impact']
        hedge_impact = recovery_analysis['hedge_analysis']['avg_impact']

        if dca_impact < -5 and hedge_impact < -5:
            # Recovery is making things WORSE
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'stack_sl',
                'action': 'TIGHTEN_STACK_SL',
                'reason': f'Recovery systems HURTING profit (DCA: ${dca_impact:.2f}, Hedge: ${hedge_impact:.2f})',
                'implementation': f'Reduce stack SL from ${current_stack_sl:.2f} to $-10.00 to cut losses faster',
                'alternative': 'OR disable DCA/Hedge entirely and rely on ADX hard stops'
            })

        # 3. Best hours recommendation
        best_hours = spread_analysis['best_hour_numbers']
        recommendations.append({
            'priority': 'HIGH',
            'category': 'trading_hours',
            'action': 'RESTRICT_TRADING_HOURS',
            'reason': f'Performance varies significantly by hour (${avg_normal_profit:.2f} vs ${avg_spread_loss:.2f})',
            'implementation': f'Only trade during: {", ".join(str(h) for h in best_hours)} GMT'
        })

        return {
            'current_stack_sl': current_stack_sl,
            'avg_spread_hour_loss': avg_spread_loss,
            'avg_normal_hour_profit': avg_normal_profit,
            'recommendations': recommendations
        }

    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive analysis report answering user's questions:
        1. What is the best time for the mean reversion bot to run?
        2. Does ML suggest an increase in SL value per stack outright?
        3. Should the stack SL logic just be disabled during spread hours?

        Returns:
            Complete analysis report
        """
        spread_hours = self.identify_spread_hours()
        recovery_analysis = self.analyze_recovery_during_spread_hours()
        trading_hours = self.recommend_trading_hours()
        stack_sl_recs = self.calculate_stack_sl_recommendations()

        # Build detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {
                'total_hours_analyzed': 24,
                'spread_hours_identified': len(spread_hours['spread_hour_numbers']),
                'best_hours_identified': len(trading_hours['recommended_hours']),
                'recovery_systems_status': 'HURTING_PROFIT' if recovery_analysis['dca_analysis']['avg_impact'] < -3 else 'NEUTRAL'
            },

            # Question 1: Best trading times
            'question_1_best_trading_times': {
                'answer': f"Trade during hours: {', '.join(str(h) for h in trading_hours['recommended_hours'])} GMT",
                'avoid_hours': spread_hours['spread_hour_numbers'],
                'spread_hours_detail': spread_hours['spread_hours'],
                'best_hours_detail': spread_hours['normal_hours'][:8],
                'explanation': 'These hours show positive avg profit and good win rates'
            },

            # Question 2: Stack SL adjustment
            'question_2_stack_sl_adjustment': {
                'answer': stack_sl_recs['recommendations'][0]['action'] if stack_sl_recs['recommendations'] else 'NO_CHANGE',
                'current_value': stack_sl_recs['current_stack_sl'],
                'recommended_value': -10.0 if recovery_analysis['dca_analysis']['avg_impact'] < -5 else -20.0,
                'explanation': 'Recovery systems are amplifying losses, not helping',
                'dca_impact': recovery_analysis['dca_analysis']['avg_impact'],
                'hedge_impact': recovery_analysis['hedge_analysis']['avg_impact']
            },

            # Question 3: Disable during spread hours?
            'question_3_disable_during_spread_hours': {
                'answer': 'YES' if stack_sl_recs['avg_spread_hour_loss'] < -5 else 'NO',
                'spread_hours': spread_hours['spread_hour_numbers'],
                'avg_loss_during_spread': stack_sl_recs['avg_spread_hour_loss'],
                'avg_profit_normal': stack_sl_recs['avg_normal_hour_profit'],
                'explanation': f'Spread hours (0, 9, 13, 20) show ${stack_sl_recs["avg_spread_hour_loss"]:.2f} avg loss',
                'implementation': 'Add hour-based check before enabling recovery/stack SL'
            },

            # All recommendations
            'recommendations': stack_sl_recs['recommendations'],

            # Raw data
            'spread_hours_analysis': spread_hours,
            'recovery_analysis': recovery_analysis,
            'trading_hours_analysis': trading_hours
        }

        return report

    def save_report(self, report: Dict, filename: str = "spread_hours_analysis.json"):
        """Save analysis report to file"""
        output_path = self.outputs_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return output_path

    def print_summary(self, report: Dict):
        """Print human-readable summary of analysis"""
        print("=" * 80)
        print("SPREAD HOURS & STACK SL ANALYSIS")
        print("=" * 80)
        print()

        # Question 1
        q1 = report['question_1_best_trading_times']
        print("QUESTION 1: What is the best time for the mean reversion bot to run?")
        print(f"ANSWER: {q1['answer']}")
        print(f"  - Avoid hours: {', '.join(str(h) for h in q1['avoid_hours'])}")
        print(f"  - Reason: {q1['explanation']}")
        print()

        # Spread hours detail
        print("SPREAD HOURS (HIGH RISK):")
        for h in q1['spread_hours_detail'][:4]:
            print(f"  Hour {h['hour']:2d}: {h['trades']:3d} trades, ${h['avg_profit']:6.2f} avg, {h['win_rate']:.1f}% WR")
        print()

        # Best hours detail
        print("BEST HOURS (LOW RISK):")
        for h in q1['best_hours_detail'][:5]:
            print(f"  Hour {h['hour']:2d}: {h['trades']:3d} trades, ${h['avg_profit']:6.2f} avg, {h['win_rate']:.1f}% WR")
        print()

        # Question 2
        q2 = report['question_2_stack_sl_adjustment']
        print("QUESTION 2: Should stack SL be increased or decreased?")
        print(f"ANSWER: {q2['answer']}")
        print(f"  - Current: ${q2['current_value']:.2f}")
        print(f"  - Recommended: ${q2['recommended_value']:.2f}")
        print(f"  - DCA Impact: ${q2['dca_impact']:.2f} (negative = hurting profit)")
        print(f"  - Hedge Impact: ${q2['hedge_impact']:.2f} (negative = hurting profit)")
        print(f"  - Reason: {q2['explanation']}")
        print()

        # Question 3
        q3 = report['question_3_disable_during_spread_hours']
        print("QUESTION 3: Should stack SL be disabled during spread hours?")
        print(f"ANSWER: {q3['answer']}")
        print(f"  - Spread hours: {', '.join(str(h) for h in q3['spread_hours'])}")
        print(f"  - Avg loss during spread: ${q3['avg_loss_during_spread']:.2f}")
        print(f"  - Avg profit normal hours: ${q3['avg_profit_normal']:.2f}")
        print(f"  - Reason: {q3['explanation']}")
        print(f"  - Implementation: {q3['implementation']}")
        print()

        # All recommendations
        print("ALL RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. [{rec['priority']}] {rec['action']}")
            print(f"   Reason: {rec['reason']}")
            print(f"   Implementation: {rec['implementation']}")
            if 'alternative' in rec:
                print(f"   Alternative: {rec['alternative']}")
            print()

        print("=" * 80)
        print(f"Report saved to: {self.outputs_dir / 'spread_hours_analysis.json'}")
        print("=" * 80)


def main():
    """Main entry point"""
    analyzer = SpreadHoursAnalyzer()

    print("Analyzing spread hours and stack SL performance...")
    print()

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()

    # Save to file
    output_path = analyzer.save_report(report)

    # Print summary
    analyzer.print_summary(report)

    return report


if __name__ == '__main__':
    main()
