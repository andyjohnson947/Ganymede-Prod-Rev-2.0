#!/usr/bin/env python3
"""
Factor Effectiveness Analyzer

Tracks actual win rate per confluence factor (not ML importance).
Provides per-symbol breakdown and generates effectiveness reports.
Used by auto-tuner to adjust weights and by tier classifier for quality scoring.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class FactorEffectivenessAnalyzer:
    """
    Analyzes actual trade outcomes to determine which confluence factors
    are most effective at predicting winning trades.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent / "outputs"
        else:
            self.data_dir = Path(data_dir)

        self.continuous_log = self.data_dir / "continuous_trade_log.jsonl"
        self.effectiveness_file = self.data_dir / "factor_effectiveness.json"

        # Cache for factor stats
        self.factor_stats: Dict = {}
        self.last_analysis: datetime = None

    def analyze_factors(self, days: int = 30, force: bool = False) -> Dict:
        """
        Analyze factor effectiveness from trade history.

        Args:
            days: Number of days to analyze
            force: Force re-analysis even if cache is fresh

        Returns:
            Dict with factor effectiveness stats
        """
        # Check if we have recent analysis
        if not force and self.factor_stats and self.last_analysis:
            if (datetime.now() - self.last_analysis).total_seconds() < 3600:  # 1 hour cache
                return self.factor_stats

        if not self.continuous_log.exists():
            return {}

        cutoff = datetime.now() - timedelta(days=days)

        # Initialize tracking structures
        # Global stats
        global_factor_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0,
            'profit': 0.0, 'avg_profit': 0.0, 'win_rate': 0.0
        })

        # Per-symbol stats
        symbol_factor_stats = defaultdict(lambda: defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0,
            'profit': 0.0, 'avg_profit': 0.0, 'win_rate': 0.0
        }))

        # Per-direction stats (buy vs sell)
        direction_factor_stats = defaultdict(lambda: defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0,
            'profit': 0.0, 'win_rate': 0.0
        }))

        # Factor combinations that work well together
        factor_combos = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0.0
        })

        trades_analyzed = 0

        # Load closed trades - try SQLite first, then JSONL fallback
        closed_trades = None
        try:
            from ml_system.trade_db import get_trade_db
            db = get_trade_db()
            if db:
                closed_trades = db.get_closed_trades()
        except Exception:
            pass

        if closed_trades is None:
            # Fallback: parse JSONL
            closed_trades = []
            try:
                with open(self.continuous_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            trade = json.loads(line)
                            if 'outcome' in trade and trade['outcome'].get('status') == 'closed':
                                closed_trades.append(trade)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except Exception as e:
                print(f"[FACTOR ANALYSIS] Error loading trades: {e}")
                return {}

        try:
            for trade in closed_trades:
                try:
                    # Check date - handle both old and new log formats
                    entry_time_str = trade.get('entry_time')
                    if not entry_time_str:
                        # Try execution_quality timestamp for new format
                        exec_quality = trade.get('execution_quality', {})
                        entry_time_str = exec_quality.get('timestamp')
                    if not entry_time_str:
                        continue

                    entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                    if entry_time < cutoff:
                        continue

                    # Get trade data - handle both old and new log formats
                    factors = trade.get('confluence_factors', [])
                    if not factors:
                        # Try old format: htf_levels.factors_matched
                        htf_levels = trade.get('htf_levels', {})
                        factors = htf_levels.get('factors_matched', [])
                    if not factors:
                        continue

                    symbol = trade.get('symbol', 'UNKNOWN')
                    direction = trade.get('direction', 'buy').lower()
                    profit = trade.get('outcome', {}).get('profit', 0)
                    is_win = profit > 0

                    trades_analyzed += 1

                    # Update stats for each factor
                    for factor in factors:
                        # Global
                        global_factor_stats[factor]['total'] += 1
                        global_factor_stats[factor]['profit'] += profit
                        if is_win:
                            global_factor_stats[factor]['wins'] += 1
                        else:
                            global_factor_stats[factor]['losses'] += 1

                        # Per-symbol
                        symbol_factor_stats[symbol][factor]['total'] += 1
                        symbol_factor_stats[symbol][factor]['profit'] += profit
                        if is_win:
                            symbol_factor_stats[symbol][factor]['wins'] += 1
                        else:
                            symbol_factor_stats[symbol][factor]['losses'] += 1

                        # Per-direction
                        direction_factor_stats[direction][factor]['total'] += 1
                        direction_factor_stats[direction][factor]['profit'] += profit
                        if is_win:
                            direction_factor_stats[direction][factor]['wins'] += 1
                        else:
                            direction_factor_stats[direction][factor]['losses'] += 1

                    # Track factor combinations (pairs)
                    sorted_factors = sorted(factors)
                    for i, f1 in enumerate(sorted_factors):
                        for f2 in sorted_factors[i+1:]:
                            combo_key = f"{f1} + {f2}"
                            factor_combos[combo_key]['total'] += 1
                            if is_win:
                                factor_combos[combo_key]['wins'] += 1
                            else:
                                factor_combos[combo_key]['losses'] += 1

                except (KeyError, ValueError):
                    continue

        except Exception as e:
            print(f"[FACTOR ANALYSIS] Error: {e}")
            return {}

        # Calculate win rates and averages
        def calc_rates(stats_dict):
            for factor, stats in stats_dict.items():
                if isinstance(stats, dict) and 'total' in stats:
                    if stats['total'] > 0:
                        stats['win_rate'] = round((stats['wins'] / stats['total']) * 100, 1)
                        stats['avg_profit'] = round(stats['profit'] / stats['total'], 2)

        calc_rates(global_factor_stats)
        for symbol_stats in symbol_factor_stats.values():
            calc_rates(symbol_stats)
        for dir_stats in direction_factor_stats.values():
            calc_rates(dir_stats)
        for combo_stats in factor_combos.values():
            if combo_stats['total'] > 0:
                combo_stats['win_rate'] = round((combo_stats['wins'] / combo_stats['total']) * 100, 1)

        # Build result
        self.factor_stats = {
            'analysis_date': datetime.now().isoformat(),
            'days_analyzed': days,
            'trades_analyzed': trades_analyzed,
            'global': dict(global_factor_stats),
            'by_symbol': {sym: dict(stats) for sym, stats in symbol_factor_stats.items()},
            'by_direction': {dir: dict(stats) for dir, stats in direction_factor_stats.items()},
            'best_combos': self._get_top_combos(factor_combos, top_n=10),
            'worst_combos': self._get_worst_combos(factor_combos, top_n=5),
            'rankings': self._rank_factors(global_factor_stats),
            'symbol_rankings': {
                sym: self._rank_factors(stats)
                for sym, stats in symbol_factor_stats.items()
            }
        }

        self.last_analysis = datetime.now()

        # Save to file
        self._save_analysis()

        return self.factor_stats

    def _rank_factors(self, stats_dict: Dict, min_trades: int = 3) -> Dict:
        """Rank factors by win rate (with minimum trade threshold)"""
        qualified = [
            {'factor': factor, **stats}
            for factor, stats in stats_dict.items()
            if stats['total'] >= min_trades
        ]

        # Sort by win rate descending
        sorted_factors = sorted(qualified, key=lambda x: x['win_rate'], reverse=True)

        return {
            'best': sorted_factors[:5],
            'worst': sorted_factors[-5:] if len(sorted_factors) >= 5 else sorted_factors,
            'all': sorted_factors
        }

    def _get_top_combos(self, combos: Dict, top_n: int = 10, min_trades: int = 3) -> List[Dict]:
        """Get best factor combinations"""
        qualified = [
            {'combo': combo, **stats}
            for combo, stats in combos.items()
            if stats['total'] >= min_trades
        ]
        return sorted(qualified, key=lambda x: x['win_rate'], reverse=True)[:top_n]

    def _get_worst_combos(self, combos: Dict, top_n: int = 5, min_trades: int = 3) -> List[Dict]:
        """Get worst factor combinations"""
        qualified = [
            {'combo': combo, **stats}
            for combo, stats in combos.items()
            if stats['total'] >= min_trades
        ]
        return sorted(qualified, key=lambda x: x['win_rate'])[:top_n]

    def _save_analysis(self):
        """Save analysis to file"""
        try:
            with open(self.effectiveness_file, 'w', encoding='utf-8') as f:
                json.dump(self.factor_stats, f, indent=2)
        except Exception as e:
            print(f"[FACTOR ANALYSIS] Error saving: {e}")

    def load_analysis(self) -> Dict:
        """Load cached analysis from file"""
        if self.effectiveness_file.exists():
            try:
                with open(self.effectiveness_file, 'r', encoding='utf-8') as f:
                    self.factor_stats = json.load(f)
                    return self.factor_stats
            except:
                pass
        return {}

    def get_factor_score(self, factor: str, symbol: str = None) -> float:
        """
        Get effectiveness score for a factor (0-100).

        Args:
            factor: Factor name
            symbol: Optional symbol for symbol-specific score

        Returns:
            Effectiveness score (win rate adjusted for sample size)
        """
        if not self.factor_stats:
            self.load_analysis()

        if not self.factor_stats:
            return 50.0  # Default neutral score

        # Try symbol-specific first
        if symbol and symbol in self.factor_stats.get('by_symbol', {}):
            sym_stats = self.factor_stats['by_symbol'][symbol]
            if factor in sym_stats and sym_stats[factor]['total'] >= 3:
                return sym_stats[factor]['win_rate']

        # Fall back to global
        global_stats = self.factor_stats.get('global', {})
        if factor in global_stats and global_stats[factor]['total'] >= 3:
            return global_stats[factor]['win_rate']

        return 50.0  # Default neutral

    def get_setup_quality_score(self, factors: List[str], symbol: str = None) -> Dict:
        """
        Calculate quality score for a setup based on its factors' effectiveness.

        Args:
            factors: List of confluence factors
            symbol: Optional symbol for symbol-specific scoring

        Returns:
            Dict with quality metrics
        """
        if not factors:
            return {'score': 0, 'avg_win_rate': 0, 'best_factor': None, 'worst_factor': None}

        factor_scores = []
        for factor in factors:
            score = self.get_factor_score(factor, symbol)
            factor_scores.append({'factor': factor, 'score': score})

        # Sort by score
        factor_scores.sort(key=lambda x: x['score'], reverse=True)

        avg_score = sum(f['score'] for f in factor_scores) / len(factor_scores)

        # Weight towards best factors (top factors matter more)
        if len(factor_scores) >= 2:
            weighted_score = (factor_scores[0]['score'] * 0.4 +
                            factor_scores[1]['score'] * 0.3 +
                            avg_score * 0.3)
        else:
            weighted_score = avg_score

        return {
            'score': round(weighted_score, 1),
            'avg_win_rate': round(avg_score, 1),
            'best_factor': factor_scores[0] if factor_scores else None,
            'worst_factor': factor_scores[-1] if factor_scores else None,
            'factor_breakdown': factor_scores,
            'has_winning_factor': any(f['score'] >= 60 for f in factor_scores),
            'has_losing_factor': any(f['score'] <= 40 for f in factor_scores),
        }

    def get_recommended_weights(self, min_trades: int = 5) -> Dict[str, int]:
        """
        Generate recommended weights based on actual effectiveness.

        Returns:
            Dict mapping factor names to recommended weights (1-5)
        """
        if not self.factor_stats:
            self.analyze_factors()

        global_stats = self.factor_stats.get('global', {})
        recommendations = {}

        for factor, stats in global_stats.items():
            if stats['total'] < min_trades:
                recommendations[factor] = 2  # Default for insufficient data
                continue

            win_rate = stats['win_rate']

            # Map win rate to weight (1-5)
            if win_rate >= 70:
                weight = 5
            elif win_rate >= 60:
                weight = 4
            elif win_rate >= 50:
                weight = 3
            elif win_rate >= 40:
                weight = 2
            else:
                weight = 1

            recommendations[factor] = weight

        return recommendations

    def print_report(self, symbol: str = None):
        """Print formatted effectiveness report"""
        if not self.factor_stats:
            self.analyze_factors()

        print()
        print("=" * 80)
        print("FACTOR EFFECTIVENESS REPORT")
        print(f"Based on {self.factor_stats.get('trades_analyzed', 0)} trades "
              f"(last {self.factor_stats.get('days_analyzed', 30)} days)")
        print("=" * 80)

        # Global rankings
        rankings = self.factor_stats.get('rankings', {})

        print("\n[+] TOP 5 WINNING FACTORS (Global):")
        print("-" * 50)
        for i, f in enumerate(rankings.get('best', [])[:5], 1):
            print(f"  {i}. {f['factor']:25} | {f['win_rate']:5.1f}% WR | "
                  f"{f['total']:3} trades | ${f['avg_profit']:.2f} avg")

        print("\n[-] TOP 5 LOSING FACTORS (Global):")
        print("-" * 50)
        for i, f in enumerate(rankings.get('worst', [])[:5], 1):
            print(f"  {i}. {f['factor']:25} | {f['win_rate']:5.1f}% WR | "
                  f"{f['total']:3} trades | ${f['avg_profit']:.2f} avg")

        # Per-symbol if available
        symbol_rankings = self.factor_stats.get('symbol_rankings', {})
        for sym, sym_ranks in symbol_rankings.items():
            print(f"\n[{sym}] TOP 3 WINNING FACTORS:")
            print("-" * 50)
            for i, f in enumerate(sym_ranks.get('best', [])[:3], 1):
                print(f"  {i}. {f['factor']:25} | {f['win_rate']:5.1f}% WR | {f['total']:3} trades")

        # Best combinations
        print("\n[*] BEST FACTOR COMBINATIONS:")
        print("-" * 50)
        for combo in self.factor_stats.get('best_combos', [])[:5]:
            print(f"  {combo['combo']:40} | {combo['win_rate']:5.1f}% WR | {combo['total']:3} trades")

        # Recommendations
        print("\n[>] WEIGHT RECOMMENDATIONS:")
        print("-" * 50)
        recs = self.get_recommended_weights()
        sorted_recs = sorted(recs.items(), key=lambda x: x[1], reverse=True)
        for factor, weight in sorted_recs[:10]:
            stars = "*" * weight + "-" * (5 - weight)
            print(f"  {factor:25} | Weight: {weight} {stars}")

        print()
        print("=" * 80)


# Global instance
_analyzer_instance: Optional[FactorEffectivenessAnalyzer] = None


def get_factor_analyzer() -> FactorEffectivenessAnalyzer:
    """Get or create global analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = FactorEffectivenessAnalyzer()
    return _analyzer_instance


def get_factor_score(factor: str, symbol: str = None) -> float:
    """Convenience function to get factor effectiveness score"""
    return get_factor_analyzer().get_factor_score(factor, symbol)


def get_setup_quality(factors: List[str], symbol: str = None) -> Dict:
    """Convenience function to get setup quality score"""
    return get_factor_analyzer().get_setup_quality_score(factors, symbol)


if __name__ == '__main__':
    analyzer = FactorEffectivenessAnalyzer()
    analyzer.analyze_factors(days=30)
    analyzer.print_report()
