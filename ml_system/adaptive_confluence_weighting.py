#!/usr/bin/env python3
"""
Adaptive Confluence Weighting System

Dynamically adjusts confluence factor weights based on historical performance.

Key Features:
1. Track performance by individual confluence factor
2. Track performance by confluence combinations (patterns)
3. Learn which factors/combinations predict wins
4. Categorize setups: EXCELLENT > VERY_GOOD > GOOD > MEDIUM > POOR
5. Recommend optimal weights based on data
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from datetime import datetime


class AdaptiveConfluenceWeighting:
    """Learn optimal confluence weights from trade outcomes"""

    def __init__(self, ml_outputs_dir: str = None):
        if ml_outputs_dir is None:
            project_root = Path(__file__).parent.parent
            ml_outputs_dir = project_root / "ml_system" / "outputs"
        self.outputs_dir = Path(ml_outputs_dir)

        # Load existing data
        self.trade_log = self._load_trade_log()

        # Analysis results
        self.factor_performance = {}  # Individual factor performance
        self.combination_performance = {}  # Confluence combination performance
        self.quality_tiers = {}  # Setup quality categorization

    def _load_trade_log(self) -> List[Dict]:
        """Load enhanced trade log if available"""
        log_file = self.outputs_dir / "enhanced_trade_log.jsonl"

        if not log_file.exists():
            return []

        trades = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        trades.append(json.loads(line))
                    except:
                        continue

        return trades

    def analyze_individual_factors(self) -> Dict:
        """
        Analyze performance of individual confluence factors

        Returns:
            {
                'vwap_band_1': {
                    'trades': 45,
                    'wins': 32,
                    'win_rate': 71.1,
                    'avg_profit': 2.34,
                    'importance_score': 85.2
                }
            }
        """
        # Group trades by factors present
        factor_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_profit': 0,
            'present_in_trades': []
        })

        for trade in self.trade_log:
            # Get confluence factors present in this trade
            factors = trade.get('confluence_factors', [])
            outcome = trade.get('outcome', {})
            profit = outcome.get('net_profit', 0)
            won = profit > 0

            for factor in factors:
                factor_stats[factor]['trades'] += 1
                if won:
                    factor_stats[factor]['wins'] += 1
                factor_stats[factor]['total_profit'] += profit
                factor_stats[factor]['present_in_trades'].append(trade.get('ticket'))

        # Calculate metrics for each factor
        results = {}
        for factor, stats in factor_stats.items():
            if stats['trades'] < 5:  # Need minimum sample
                continue

            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            avg_profit = stats['total_profit'] / stats['trades'] if stats['trades'] > 0 else 0

            # Importance score combines win rate and profitability
            # 70% weight on win rate, 30% on avg profit (normalized to 0-100)
            normalized_profit = min(100, max(0, (avg_profit + 5) * 10))  # -5 to +5 → 0 to 100
            importance_score = (win_rate * 0.7) + (normalized_profit * 0.3)

            results[factor] = {
                'trades': stats['trades'],
                'wins': stats['wins'],
                'win_rate': round(win_rate, 1),
                'avg_profit': round(avg_profit, 2),
                'total_profit': round(stats['total_profit'], 2),
                'importance_score': round(importance_score, 1),
                'recommended_weight': self._calculate_recommended_weight(importance_score)
            }

        # Sort by importance
        self.factor_performance = dict(sorted(
            results.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        ))

        return self.factor_performance

    def analyze_confluence_combinations(self) -> Dict:
        """
        Analyze performance of specific confluence factor combinations

        Identifies patterns like:
        - "vwap_band_2 + poc + swing_low" → 85% win rate
        - "daily_poc + weekly_hvn" → 78% win rate
        """
        combination_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_profit': 0,
            'factor_count': 0
        })

        for trade in self.trade_log:
            factors = trade.get('confluence_factors', [])
            if not factors:
                continue

            outcome = trade.get('outcome', {})
            profit = outcome.get('net_profit', 0)
            won = profit > 0

            # Sort factors for consistent key
            factors_sorted = tuple(sorted(factors))

            combination_stats[factors_sorted]['trades'] += 1
            if won:
                combination_stats[factors_sorted]['wins'] += 1
            combination_stats[factors_sorted]['total_profit'] += profit
            combination_stats[factors_sorted]['factor_count'] = len(factors)

        # Calculate metrics
        results = {}
        for combo, stats in combination_stats.items():
            if stats['trades'] < 3:  # Need minimum sample
                continue

            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            avg_profit = stats['total_profit'] / stats['trades'] if stats['trades'] > 0 else 0

            # Pattern strength score
            pattern_strength = self._calculate_pattern_strength(
                win_rate,
                avg_profit,
                stats['trades'],
                stats['factor_count']
            )

            results[combo] = {
                'factors': list(combo),
                'factor_count': stats['factor_count'],
                'trades': stats['trades'],
                'wins': stats['wins'],
                'win_rate': round(win_rate, 1),
                'avg_profit': round(avg_profit, 2),
                'total_profit': round(stats['total_profit'], 2),
                'pattern_strength': round(pattern_strength, 1),
                'quality_tier': self._assign_quality_tier(win_rate, avg_profit)
            }

        # Sort by pattern strength
        self.combination_performance = dict(sorted(
            results.items(),
            key=lambda x: x[1]['pattern_strength'],
            reverse=True
        ))

        return self.combination_performance

    def categorize_setup_quality(self, confluence_factors: List[str]) -> Dict:
        """
        Categorize a trade setup based on confluence factors present

        Returns:
            {
                'quality_tier': 'EXCELLENT',
                'score': 87.5,
                'win_probability': 85.2,
                'expected_profit': 3.45,
                'recommendation': 'TAKE_FULL_SIZE'
            }
        """
        if not self.factor_performance:
            self.analyze_individual_factors()
        if not self.combination_performance:
            self.analyze_confluence_combinations()

        # Check for exact combination match first
        factors_sorted = tuple(sorted(confluence_factors))
        if factors_sorted in self.combination_performance:
            combo_data = self.combination_performance[factors_sorted]
            return {
                'quality_tier': combo_data['quality_tier'],
                'score': combo_data['pattern_strength'],
                'win_probability': combo_data['win_rate'],
                'expected_profit': combo_data['avg_profit'],
                'recommendation': self._get_recommendation(combo_data['quality_tier']),
                'confidence': 'HIGH',
                'reason': f"Exact pattern match ({combo_data['trades']} historical trades)"
            }

        # If no exact match, score based on individual factors
        total_importance = 0
        factor_count = 0
        win_rate_sum = 0
        profit_sum = 0

        for factor in confluence_factors:
            if factor in self.factor_performance:
                stats = self.factor_performance[factor]
                total_importance += stats['importance_score']
                win_rate_sum += stats['win_rate']
                profit_sum += stats['avg_profit']
                factor_count += 1

        if factor_count == 0:
            # No data on any of these factors
            return {
                'quality_tier': 'UNKNOWN',
                'score': 0,
                'win_probability': 50,
                'expected_profit': 0,
                'recommendation': 'SKIP_INSUFFICIENT_DATA',
                'confidence': 'NONE',
                'reason': 'No historical data on these confluence factors'
            }

        # Calculate composite scores
        avg_importance = total_importance / factor_count
        avg_win_rate = win_rate_sum / factor_count
        avg_profit = profit_sum / factor_count

        # Bonus for more factors
        factor_bonus = min(20, len(confluence_factors) * 3)

        composite_score = avg_importance + factor_bonus

        # Determine quality tier
        if composite_score >= 85:
            tier = 'EXCELLENT'
        elif composite_score >= 70:
            tier = 'VERY_GOOD'
        elif composite_score >= 55:
            tier = 'GOOD'
        elif composite_score >= 40:
            tier = 'MEDIUM'
        else:
            tier = 'POOR'

        return {
            'quality_tier': tier,
            'score': round(composite_score, 1),
            'win_probability': round(avg_win_rate, 1),
            'expected_profit': round(avg_profit, 2),
            'recommendation': self._get_recommendation(tier),
            'confidence': 'MEDIUM',
            'reason': f'Composite score from {factor_count} known factors'
        }

    def generate_optimal_weights(self) -> Dict:
        """
        Generate optimal confluence weights based on learned importance

        Returns config-ready weight dictionary
        """
        if not self.factor_performance:
            self.analyze_individual_factors()

        optimal_weights = {}

        for factor, stats in self.factor_performance.items():
            optimal_weights[factor] = stats['recommended_weight']

        return optimal_weights

    def _calculate_recommended_weight(self, importance_score: float) -> int:
        """Convert importance score to weight (1-5)"""
        if importance_score >= 85:
            return 5
        elif importance_score >= 70:
            return 4
        elif importance_score >= 55:
            return 3
        elif importance_score >= 40:
            return 2
        else:
            return 1

    def _calculate_pattern_strength(self, win_rate: float, avg_profit: float,
                                   trades: int, factor_count: int) -> float:
        """Calculate overall pattern strength score"""
        # Base score from win rate and profit
        base_score = (win_rate * 0.7) + (min(100, (avg_profit + 5) * 10) * 0.3)

        # Confidence boost from sample size
        confidence_multiplier = min(1.2, 1.0 + (trades / 100))

        # Factor diversity bonus
        diversity_bonus = min(10, factor_count * 2)

        return (base_score * confidence_multiplier) + diversity_bonus

    def _assign_quality_tier(self, win_rate: float, avg_profit: float) -> str:
        """Assign quality tier based on metrics"""
        # Excellent: High win rate AND good profit
        if win_rate >= 75 and avg_profit >= 2.0:
            return 'EXCELLENT'
        # Very Good: Good win rate OR very good profit
        elif win_rate >= 70 or avg_profit >= 3.0:
            return 'VERY_GOOD'
        # Good: Decent performance
        elif win_rate >= 60 and avg_profit >= 1.0:
            return 'GOOD'
        # Medium: Marginal
        elif win_rate >= 50 or avg_profit >= 0:
            return 'MEDIUM'
        else:
            return 'POOR'

    def _get_recommendation(self, tier: str) -> str:
        """Get trading recommendation based on tier"""
        recommendations = {
            'EXCELLENT': 'TAKE_FULL_SIZE',
            'VERY_GOOD': 'TAKE_FULL_SIZE',
            'GOOD': 'TAKE_NORMAL_SIZE',
            'MEDIUM': 'TAKE_REDUCED_SIZE',
            'POOR': 'SKIP',
            'UNKNOWN': 'SKIP_INSUFFICIENT_DATA'
        }
        return recommendations.get(tier, 'SKIP')

    def compare_with_current_weights(self, current_weights: Dict) -> Dict:
        """
        Compare current weights with ML-recommended weights

        Shows which weights should be adjusted
        """
        if not self.factor_performance:
            self.analyze_individual_factors()

        optimal_weights = self.generate_optimal_weights()

        comparison = {}
        for factor in set(current_weights.keys()) | set(optimal_weights.keys()):
            current = current_weights.get(factor, 0)
            optimal = optimal_weights.get(factor, 0)

            if factor not in self.factor_performance:
                status = 'NO_DATA'
                change = 0
            elif abs(current - optimal) <= 1:
                status = 'OK'
                change = 0
            elif optimal > current:
                status = 'INCREASE'
                change = optimal - current
            else:
                status = 'DECREASE'
                change = optimal - current

            comparison[factor] = {
                'current_weight': current,
                'optimal_weight': optimal,
                'change': change,
                'status': status,
                'importance_score': self.factor_performance.get(factor, {}).get('importance_score', 0)
            }

        return comparison

    def generate_report(self) -> Dict:
        """Generate comprehensive confluence analysis report"""

        # Analyze everything
        factor_perf = self.analyze_individual_factors()
        combo_perf = self.analyze_confluence_combinations()

        # Get top patterns
        top_patterns = list(combo_perf.items())[:10]

        # Categorize patterns by quality
        excellent = [p for p in combo_perf.values() if p['quality_tier'] == 'EXCELLENT']
        very_good = [p for p in combo_perf.values() if p['quality_tier'] == 'VERY_GOOD']
        good = [p for p in combo_perf.values() if p['quality_tier'] == 'GOOD']
        poor = [p for p in combo_perf.values() if p['quality_tier'] == 'POOR']

        return {
            'timestamp': datetime.now().isoformat(),
            'total_trades_analyzed': len(self.trade_log),

            'factor_performance': factor_perf,
            'combination_performance': combo_perf,

            'summary': {
                'factors_analyzed': len(factor_perf),
                'patterns_identified': len(combo_perf),
                'excellent_patterns': len(excellent),
                'very_good_patterns': len(very_good),
                'good_patterns': len(good),
                'poor_patterns': len(poor)
            },

            'top_individual_factors': list(factor_perf.items())[:10],
            'top_patterns': top_patterns,

            'quality_tiers': {
                'EXCELLENT': excellent,
                'VERY_GOOD': very_good,
                'GOOD': good,
                'POOR': poor
            },

            'optimal_weights': self.generate_optimal_weights()
        }

    def print_report(self, report: Dict):
        """Print human-readable report"""
        print("=" * 100)
        print("ADAPTIVE CONFLUENCE WEIGHTING ANALYSIS")
        print("=" * 100)
        print()

        print(f"Trades analyzed: {report['total_trades_analyzed']}")
        print(f"Factors analyzed: {report['summary']['factors_analyzed']}")
        print(f"Patterns identified: {report['summary']['patterns_identified']}")
        print()

        # Top individual factors
        print("=" * 100)
        print("TOP 10 INDIVIDUAL CONFLUENCE FACTORS (By Importance)")
        print("=" * 100)
        print(f"{'Factor':<30} {'Trades':>7} {'Win%':>7} {'Avg$':>8} {'Score':>7} {'Weight':>7}")
        print("-" * 100)

        for factor, stats in report['top_individual_factors']:
            print(f"{factor:<30} {stats['trades']:>7} {stats['win_rate']:>7.1f} "
                  f"{stats['avg_profit']:>8.2f} {stats['importance_score']:>7.1f} "
                  f"{stats['recommended_weight']:>7}")
        print()

        # Top patterns
        print("=" * 100)
        print("TOP 10 CONFLUENCE PATTERNS (By Pattern Strength)")
        print("=" * 100)

        for i, (combo, stats) in enumerate(report['top_patterns'], 1):
            print(f"{i}. {stats['quality_tier']} (Score: {stats['pattern_strength']:.1f})")
            print(f"   Factors: {' + '.join(stats['factors'])}")
            print(f"   Performance: {stats['trades']} trades, {stats['win_rate']:.1f}% WR, ${stats['avg_profit']:.2f} avg")
            print(f"   Recommendation: {self._get_recommendation(stats['quality_tier'])}")
            print()

        # Quality tier summary
        print("=" * 100)
        print("PATTERN QUALITY DISTRIBUTION")
        print("=" * 100)

        for tier in ['EXCELLENT', 'VERY_GOOD', 'GOOD', 'POOR']:
            patterns = report['quality_tiers'][tier]
            count = len(patterns)
            if count > 0:
                avg_wr = sum(p['win_rate'] for p in patterns) / count
                avg_profit = sum(p['avg_profit'] for p in patterns) / count
                print(f"{tier:12} {count:3} patterns | Avg: {avg_wr:.1f}% WR, ${avg_profit:.2f} profit")
        print()

        print("=" * 100)

    def save_report(self, report: Dict):
        """Save report to file"""
        output_path = self.outputs_dir / "adaptive_confluence_weights.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return output_path


def main():
    """Generate adaptive confluence weighting analysis"""
    analyzer = AdaptiveConfluenceWeighting()

    if not analyzer.trade_log:
        print("=" * 100)
        print("NO TRADE DATA AVAILABLE")
        print("=" * 100)
        print()
        print("The adaptive confluence weighting system needs trade data to analyze.")
        print()
        print("To generate data:")
        print("  1. Integrate enhanced_trade_logger.py into your trading bot")
        print("  2. Ensure trades log 'confluence_factors' field")
        print("  3. Run bot for 50+ trades")
        print("  4. Re-run this analysis")
        print()
        print("Example confluence_factors field:")
        print("  'confluence_factors': ['vwap_band_2', 'poc', 'swing_low', 'daily_hvn']")
        print()
        return

    report = analyzer.generate_report()
    analyzer.print_report(report)
    output_path = analyzer.save_report(report)
    print(f"Full report saved to: {output_path}")


if __name__ == '__main__':
    main()
