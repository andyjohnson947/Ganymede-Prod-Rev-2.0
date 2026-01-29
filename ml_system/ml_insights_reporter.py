#!/usr/bin/env python3
"""
ML Insights Reporter
Provides automatic ML insights for bot startup and daily reports
No separate scripts needed - called directly by the bot
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os


class MLInsightsReporter:
    """Generate automatic ML insights from collected data"""

    def __init__(self):
        # Use absolute path based on this file's location (not current working directory)
        self.outputs_dir = Path(__file__).parent / "outputs"
        self.enhanced_trade_log = self.outputs_dir / "enhanced_trade_log.jsonl"
        self.recovery_log = self.outputs_dir / "recovery_decisions.jsonl"
        self.market_conditions_log = self.outputs_dir / "market_conditions.jsonl"
        self.adaptive_weights = self.outputs_dir / "adaptive_confluence_weights.json"

    def get_data_status(self) -> Dict:
        """Get current ML data collection status"""
        status = {
            'trades_logged': 0,
            'recovery_decisions': 0,
            'has_adaptive_weights': False,
            'data_age_hours': None,
            'ready_for_analysis': False
        }

        # Count trade entries
        if self.enhanced_trade_log.exists():
            with open(self.enhanced_trade_log, 'r', encoding='utf-8') as f:
                status['trades_logged'] = sum(1 for line in f if line.strip())

        # Count recovery decisions
        if self.recovery_log.exists():
            with open(self.recovery_log, 'r', encoding='utf-8') as f:
                status['recovery_decisions'] = sum(1 for line in f if line.strip())

        # Check adaptive weights
        status['has_adaptive_weights'] = self.adaptive_weights.exists()

        # Check data freshness
        if self.enhanced_trade_log.exists():
            age = datetime.now() - datetime.fromtimestamp(self.enhanced_trade_log.stat().st_mtime)
            status['data_age_hours'] = age.total_seconds() / 3600

        # Ready for analysis if we have 50+ trades
        status['ready_for_analysis'] = status['trades_logged'] >= 50

        return status

    def get_quick_insights(self, days: int = 7) -> Dict:
        """Get quick ML insights without running full analysis"""
        insights = {
            'summary': [],
            'winning_factors': [],
            'losing_factors': [],
            'recovery_performance': {},
            'best_hours': [],
            'worst_hours': [],
            'recommendations': []
        }

        if not self.enhanced_trade_log.exists():
            insights['summary'].append("No trade data yet - ML collection starts on first trade")
            return insights

        # Load recent trades
        trades = self._load_recent_trades(days)

        if len(trades) == 0:
            insights['summary'].append(f"No trades in last {days} days")
            return insights

        # Basic stats
        closed = [t for t in trades if t.get('exit_time')]
        if len(closed) == 0:
            insights['summary'].append(f"{len(trades)} trades opened, 0 closed (need closed trades for insights)")
            return insights

        # Win rate
        wins = [t for t in closed if t.get('profit', 0) > 0]
        win_rate = len(wins) / len(closed) * 100
        avg_profit = sum(t.get('profit', 0) for t in closed) / len(closed)

        insights['summary'].append(f"Last {days} days: {len(closed)} trades closed")
        insights['summary'].append(f"   Win Rate: {win_rate:.1f}% ({len(wins)}/{len(closed)})")
        insights['summary'].append(f"   Avg P&L: ${avg_profit:.2f}")

        # Analyze confluence factors (if we have them)
        if len(closed) >= 10:
            factor_analysis = self._analyze_confluence_factors(closed)
            if factor_analysis['winning_factors']:
                insights['winning_factors'] = factor_analysis['winning_factors'][:3]  # Top 3
            if factor_analysis['losing_factors']:
                insights['losing_factors'] = factor_analysis['losing_factors'][:3]  # Top 3

        # Recovery performance
        recovery_stats = self._analyze_recovery(closed)
        insights['recovery_performance'] = recovery_stats

        # Hour analysis
        if len(closed) >= 20:
            hour_stats = self._analyze_hours(closed)
            insights['best_hours'] = hour_stats['best'][:3]
            insights['worst_hours'] = hour_stats['worst'][:3]

        # Generate recommendations
        insights['recommendations'] = self._generate_quick_recommendations(
            closed, factor_analysis if len(closed) >= 10 else None, recovery_stats
        )

        return insights

    def _load_recent_trades(self, days: int) -> List[Dict]:
        """Load trades from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        trades = []

        try:
            with open(self.enhanced_trade_log, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        trade = json.loads(line)
                        entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', ''))
                        if entry_time >= cutoff:
                            trades.append(trade)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception:
            pass

        return trades

    def _analyze_confluence_factors(self, trades: List[Dict]) -> Dict:
        """Analyze which confluence factors are winning/losing"""
        factor_stats = {}

        for trade in trades:
            factors = trade.get('confluence_factors', [])
            is_win = trade.get('profit', 0) > 0

            for factor in factors:
                if factor not in factor_stats:
                    factor_stats[factor] = {'wins': 0, 'total': 0}

                factor_stats[factor]['total'] += 1
                if is_win:
                    factor_stats[factor]['wins'] += 1

        # Calculate win rates
        factor_win_rates = []
        for factor, stats in factor_stats.items():
            if stats['total'] >= 3:  # Need at least 3 occurrences
                win_rate = stats['wins'] / stats['total']
                factor_win_rates.append({
                    'factor': factor,
                    'win_rate': win_rate,
                    'count': stats['total']
                })

        # Sort by win rate
        factor_win_rates.sort(key=lambda x: x['win_rate'], reverse=True)

        return {
            'winning_factors': [f for f in factor_win_rates if f['win_rate'] >= 0.7],
            'losing_factors': [f for f in factor_win_rates if f['win_rate'] <= 0.4]
        }

    def _analyze_recovery(self, trades: List[Dict]) -> Dict:
        """Analyze recovery mechanism performance"""
        stats = {
            'dca_used': 0,
            'dca_wins': 0,
            'hedge_used': 0,
            'hedge_wins': 0,
            'no_recovery': 0,
            'no_recovery_wins': 0
        }

        for trade in trades:
            had_dca = trade.get('had_dca', False)
            had_hedge = trade.get('had_hedge', False)
            is_win = trade.get('profit', 0) > 0

            if had_dca:
                stats['dca_used'] += 1
                if is_win:
                    stats['dca_wins'] += 1

            if had_hedge:
                stats['hedge_used'] += 1
                if is_win:
                    stats['hedge_wins'] += 1

            if not had_dca and not had_hedge:
                stats['no_recovery'] += 1
                if is_win:
                    stats['no_recovery_wins'] += 1

        # Calculate rates
        stats['dca_win_rate'] = (stats['dca_wins'] / stats['dca_used'] * 100) if stats['dca_used'] > 0 else 0
        stats['hedge_win_rate'] = (stats['hedge_wins'] / stats['hedge_used'] * 100) if stats['hedge_used'] > 0 else 0
        stats['clean_win_rate'] = (stats['no_recovery_wins'] / stats['no_recovery'] * 100) if stats['no_recovery'] > 0 else 0

        return stats

    def _analyze_hours(self, trades: List[Dict]) -> Dict:
        """Analyze performance by hour"""
        hour_stats = {}

        for trade in trades:
            hour = trade.get('hour', 0)
            if hour not in hour_stats:
                hour_stats[hour] = {'wins': 0, 'total': 0, 'profit': 0}

            hour_stats[hour]['total'] += 1
            if trade.get('profit', 0) > 0:
                hour_stats[hour]['wins'] += 1
            hour_stats[hour]['profit'] += trade.get('profit', 0)

        # Calculate win rates
        hour_performance = []
        for hour, stats in hour_stats.items():
            if stats['total'] >= 2:  # Need at least 2 trades
                win_rate = stats['wins'] / stats['total']
                avg_profit = stats['profit'] / stats['total']
                hour_performance.append({
                    'hour': hour,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'count': stats['total']
                })

        # Sort by win rate
        hour_performance.sort(key=lambda x: x['win_rate'], reverse=True)

        return {
            'best': hour_performance[:3],
            'worst': hour_performance[-3:]
        }

    def _generate_quick_recommendations(self, trades: List[Dict], factor_analysis: Optional[Dict], recovery_stats: Dict) -> List[str]:
        """Generate quick actionable recommendations"""
        recommendations = []

        # 1. Data collection status
        if len(trades) < 50:
            recommendations.append(f"📈 Keep collecting data ({len(trades)}/50 trades for full analysis)")
        else:
            recommendations.append(f"[OK] Sufficient data for analysis ({len(trades)} trades)")

        # 2. Winning factors
        if factor_analysis and factor_analysis['winning_factors']:
            top_factor = factor_analysis['winning_factors'][0]
            recommendations.append(
                f"[TOP] Best factor: {top_factor['factor']} ({top_factor['win_rate']*100:.0f}% WR, n={top_factor['count']})"
            )

        # 3. Losing factors
        if factor_analysis and factor_analysis['losing_factors']:
            worst_factor = factor_analysis['losing_factors'][0]
            recommendations.append(
                f"[WARN]  Weak factor: {worst_factor['factor']} ({worst_factor['win_rate']*100:.0f}% WR, n={worst_factor['count']})"
            )

        # 4. Recovery recommendations
        if recovery_stats['dca_used'] >= 3:
            if recovery_stats['dca_win_rate'] >= 70:
                recommendations.append(f"[OK] DCA working well ({recovery_stats['dca_win_rate']:.0f}% recovery rate)")
            elif recovery_stats['dca_win_rate'] <= 40:
                recommendations.append(f"[WARN]  DCA struggling ({recovery_stats['dca_win_rate']:.0f}% recovery rate) - review triggers")

        if recovery_stats['hedge_used'] >= 3:
            if recovery_stats['hedge_win_rate'] >= 70:
                recommendations.append(f"[OK] Hedge working well ({recovery_stats['hedge_win_rate']:.0f}% recovery rate)")
            elif recovery_stats['hedge_win_rate'] <= 40:
                recommendations.append(f"[WARN]  Hedge struggling ({recovery_stats['hedge_win_rate']:.0f}% recovery rate) - review triggers")

        # 5. Clean trades (no recovery needed)
        if recovery_stats['no_recovery'] >= 5:
            clean_rate = recovery_stats['clean_win_rate']
            recovery_rate = (recovery_stats['dca_used'] + recovery_stats['hedge_used']) / len(trades) * 100
            recommendations.append(f"Clean trades: {clean_rate:.0f}% WR, Recovery needed: {recovery_rate:.0f}%")

        return recommendations

    def format_startup_report(self) -> str:
        """Format ML insights for bot startup"""
        status = self.get_data_status()
        insights = self.get_quick_insights(days=7)

        report = []
        report.append("")
        report.append("=" * 80)
        report.append(" ML INSIGHTS (Last 7 Days)")
        report.append("=" * 80)

        # Data status
        report.append("Data Collection:")
        report.append(f"   Trades Logged: {status['trades_logged']}")
        report.append(f"   Recovery Decisions: {status['recovery_decisions']}")
        if status['data_age_hours']:
            report.append(f"   Last Update: {status['data_age_hours']:.1f} hours ago")
        trades_needed = 50 - status['trades_logged']
        analysis_status = 'Yes' if status['ready_for_analysis'] else f'No (need {trades_needed} more trades)'
        report.append(f"   Analysis Ready: {analysis_status}")
        report.append("")

        # Summary
        if insights['summary']:
            for line in insights['summary']:
                report.append(line)
            report.append("")

        # Winning factors
        if insights['winning_factors']:
            report.append("[+] Top Performing Factors:")
            for factor in insights['winning_factors']:
                report.append(f"   + {factor['factor']}: {factor['win_rate']*100:.0f}% WR (n={factor['count']})")
            report.append("")

        # Losing factors
        if insights['losing_factors']:
            report.append("[!] Underperforming Factors:")
            for factor in insights['losing_factors']:
                report.append(f"   - {factor['factor']}: {factor['win_rate']*100:.0f}% WR (n={factor['count']})")
            report.append("")

        # Recovery performance
        if insights['recovery_performance']:
            recovery = insights['recovery_performance']
            report.append("[~] Recovery Performance:")
            if recovery['dca_used'] > 0:
                report.append(f"   DCA: {recovery['dca_win_rate']:.0f}% recovery ({recovery['dca_wins']}/{recovery['dca_used']} trades)")
            if recovery['hedge_used'] > 0:
                report.append(f"   Hedge: {recovery['hedge_win_rate']:.0f}% recovery ({recovery['hedge_wins']}/{recovery['hedge_used']} trades)")
            if recovery['no_recovery'] > 0:
                report.append(f"   Clean: {recovery['clean_win_rate']:.0f}% WR ({recovery['no_recovery_wins']}/{recovery['no_recovery']} trades)")
            report.append("")

        # Best/worst hours
        if insights['best_hours']:
            report.append("[*] Best Trading Hours:")
            for hour_stat in insights['best_hours']:
                report.append(f"   {hour_stat['hour']:02d}:00 - {hour_stat['win_rate']*100:.0f}% WR (${hour_stat['avg_profit']:.2f} avg, n={hour_stat['count']})")
            report.append("")

        # Recommendations
        if insights['recommendations']:
            report.append("[>] ML Recommendations:")
            for rec in insights['recommendations']:
                report.append(f"   {rec}")
            report.append("")

        report.append("=" * 80)

        return '\n'.join(report)


def main():
    """Test the reporter"""
    reporter = MLInsightsReporter()
    print(reporter.format_startup_report())


if __name__ == '__main__':
    main()
