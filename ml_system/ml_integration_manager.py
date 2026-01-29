#!/usr/bin/env python3
"""
ML Integration Manager

Unified interface for all ML functionality (except full automation).
Integrates:
- Enhanced trade logging (execution quality, market conditions)
- Adaptive confluence weighting
- Recovery decision tracking
- Near-miss signal tracking
- Real-time analysis and recommendations

User keeps control, ML provides insights.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_system.enhanced_trade_logger import EnhancedTradeLogger
from ml_system.adaptive_confluence_weighting import AdaptiveConfluenceWeighting


class MLIntegrationManager:
    """
    Unified ML manager - connects all ML components to trading bot

    Features:
    - Enhanced trade logging with execution quality
    - Market conditions tracking
    - Recovery decision logging
    - Near-miss signal tracking
    - Adaptive confluence scoring (setup quality)
    - Real-time recommendations

    Does NOT:
    - Make autonomous trading decisions
    - Modify parameters without user approval
    - Execute trades automatically
    """

    def __init__(self, enable_adaptive_weighting: bool = True):
        """
        Initialize ML integration

        Args:
            enable_adaptive_weighting: Use adaptive confluence scoring (requires 50+ trades)
        """
        print("[ML INTEGRATION] Initializing enhanced ML system...")

        # Enhanced logger
        self.enhanced_logger = EnhancedTradeLogger()
        print("[ML] [OK] Enhanced trade logger active")

        # Adaptive confluence (requires data)
        self.enable_adaptive = enable_adaptive_weighting
        if self.enable_adaptive:
            try:
                self.confluence_analyzer = AdaptiveConfluenceWeighting()
                if self.confluence_analyzer.trade_log and len(self.confluence_analyzer.trade_log) >= 10:
                    print(f"[ML] [OK] Adaptive confluence active ({len(self.confluence_analyzer.trade_log)} trades)")
                else:
                    print(f"[ML] [WARN] Adaptive confluence limited (only {len(self.confluence_analyzer.trade_log)} trades, need 50+)")
                    self.enable_adaptive = False
            except Exception as e:
                print(f"[ML] [WARN] Adaptive confluence disabled: {e}")
                self.enable_adaptive = False

        # Stats
        self.trades_logged = 0
        self.signals_logged = 0
        self.recovery_decisions_logged = 0

        print("[ML INTEGRATION] [OK] Ready (all encoding UTF-8)")

    # ============================================================================
    # TRADE LOGGING
    # ============================================================================

    def log_trade_entry(self, trade_data: Dict) -> Dict:
        """
        Log trade entry with full context

        Required fields:
            ticket, symbol, direction, entry_price, volume, confluence_score

        Optional fields:
            expected_price, spread_at_entry_pips, fill_time_ms, requotes,
            confluence_factors, adx, atr_pips, hour

        Returns:
            Enhanced trade record with execution quality
        """
        try:
            # Log to enhanced logger
            self.enhanced_logger.log_trade_with_execution(trade_data)

            # Log market conditions if provided
            if 'adx' in trade_data or 'atr_pips' in trade_data:
                conditions = {
                    'adx': trade_data.get('adx'),
                    'atr_pips': trade_data.get('atr_pips'),
                    'spread_pips': trade_data.get('spread_at_entry_pips'),
                    'hour': trade_data.get('hour'),
                    'confluence_score': trade_data.get('confluence_score'),
                    'distance_to_level_pips': trade_data.get('distance_to_level_pips'),
                    'at_hvn': trade_data.get('at_hvn'),
                    'at_lvn': trade_data.get('at_lvn'),
                    'at_poc': trade_data.get('at_poc')
                }
                self.enhanced_logger.log_market_conditions(
                    trade_data.get('symbol'),
                    conditions
                )

            self.trades_logged += 1

            return {
                'success': True,
                'execution_quality': trade_data.get('execution_quality', {}),
                'trades_logged': self.trades_logged
            }

        except Exception as e:
            print(f"[ML ERROR] Failed to log trade: {e}")
            return {'success': False, 'error': str(e)}

    def log_signal_detected(self, signal: Dict, blocked: bool = False, block_reason: str = None):
        """
        Log signal detection (taken or blocked)

        Args:
            signal: Signal data (symbol, direction, confluence_score, factors, etc.)
            blocked: Was signal blocked?
            block_reason: Why was it blocked?
        """
        try:
            if blocked:
                # Log near-miss (what we didn't trade)
                near_miss = {
                    'symbol': signal.get('symbol'),
                    'confluence_score': signal.get('confluence_score'),
                    'direction': signal.get('direction'),
                    'block_reason': block_reason,
                    'adx': signal.get('adx'),
                    'hour': signal.get('hour'),
                    'spread_pips': signal.get('spread_pips'),
                    'price': signal.get('price')
                }
                self.enhanced_logger.log_near_miss_signal(near_miss)
                print(f"[ML] Near-miss logged: {signal.get('symbol')} {block_reason}")

            self.signals_logged += 1

        except Exception as e:
            print(f"[ML ERROR] Failed to log signal: {e}")

    def log_recovery_decision(self, decision_data: Dict):
        """
        Log recovery trigger decision (DCA/Hedge/Grid)

        Required fields:
            ticket, type (DCA/Hedge/Grid), pips_underwater, unrealized_pnl

        Optional fields:
            was_blocked, block_reason, adx_at_entry, adx_at_trigger, etc.
        """
        try:
            self.enhanced_logger.log_recovery_decision(decision_data)
            self.recovery_decisions_logged += 1

            if decision_data.get('was_blocked'):
                print(f"[ML] Recovery blocked: {decision_data.get('type')} for #{decision_data.get('ticket')} - {decision_data.get('block_reason')}")

        except Exception as e:
            print(f"[ML ERROR] Failed to log recovery decision: {e}")

    # ============================================================================
    # ADAPTIVE CONFLUENCE SCORING
    # ============================================================================

    def score_setup_quality(self, confluence_factors: List[str]) -> Dict:
        """
        Score a trade setup BEFORE entry

        Args:
            confluence_factors: List of factors present (e.g., ['vwap_band_2', 'poc', 'swing_low'])

        Returns:
            {
                'quality_tier': 'EXCELLENT' | 'VERY_GOOD' | 'GOOD' | 'MEDIUM' | 'POOR' | 'UNKNOWN',
                'score': 85.2,  # 0-100
                'win_probability': 82.4,  # % chance of winning
                'expected_profit': 3.45,  # Expected $ profit
                'recommendation': 'TAKE_FULL_SIZE' | 'TAKE_NORMAL_SIZE' | 'TAKE_REDUCED_SIZE' | 'SKIP',
                'confidence': 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE',
                'reason': 'Exact pattern match (14 historical trades)'
            }
        """
        if not self.enable_adaptive:
            return {
                'quality_tier': 'UNKNOWN',
                'score': 0,
                'win_probability': 50,
                'expected_profit': 0,
                'recommendation': 'TAKE_NORMAL_SIZE',
                'confidence': 'NONE',
                'reason': 'Adaptive confluence disabled (need 50+ trades)'
            }

        try:
            return self.confluence_analyzer.categorize_setup_quality(confluence_factors)
        except Exception as e:
            print(f"[ML ERROR] Setup scoring failed: {e}")
            return {
                'quality_tier': 'UNKNOWN',
                'score': 0,
                'win_probability': 50,
                'expected_profit': 0,
                'recommendation': 'TAKE_NORMAL_SIZE',
                'confidence': 'NONE',
                'reason': f'Error: {e}'
            }

    def get_optimal_weights(self) -> Dict:
        """
        Get ML-recommended confluence weights

        Returns:
            {
                'daily_hvn': 5,
                'weekly_poc': 5,
                'poc': 4,
                ...
            }
        """
        if not self.enable_adaptive:
            return {}

        try:
            return self.confluence_analyzer.generate_optimal_weights()
        except Exception as e:
            print(f"[ML ERROR] Failed to get optimal weights: {e}")
            return {}

    # ============================================================================
    # ANALYSIS & REPORTS
    # ============================================================================

    def generate_session_summary(self) -> Dict:
        """Generate summary of current session"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'trades_logged': self.trades_logged,
            'signals_logged': self.signals_logged,
            'recovery_decisions_logged': self.recovery_decisions_logged,
            'adaptive_confluence_enabled': self.enable_adaptive
        }

        if self.enable_adaptive:
            try:
                # Get factor performance
                self.confluence_analyzer.analyze_individual_factors()
                top_factors = list(self.confluence_analyzer.factor_performance.items())[:5]

                summary['top_confluence_factors'] = [
                    {
                        'factor': factor,
                        'win_rate': stats['win_rate'],
                        'avg_profit': stats['avg_profit'],
                        'importance': stats['importance_score']
                    }
                    for factor, stats in top_factors
                ]
            except:
                pass

        return summary

    def print_status(self):
        """Print current ML status"""
        print()
        print("=" * 80)
        print("ML INTEGRATION STATUS")
        print("=" * 80)
        print(f"  Trades logged: {self.trades_logged}")
        print(f"  Signals logged: {self.signals_logged}")
        print(f"  Recovery decisions logged: {self.recovery_decisions_logged}")
        print(f"  Adaptive confluence: {'ENABLED' if self.enable_adaptive else 'DISABLED (need 50+ trades)'}")
        print()

        if self.enable_adaptive:
            try:
                report = self.confluence_analyzer.generate_report()
                print(f"  Patterns identified: {report['summary']['patterns_identified']}")
                print(f"  Excellent patterns: {report['summary']['excellent_patterns']}")
                print(f"  Very good patterns: {report['summary']['very_good_patterns']}")
                print()

                # Top 3 factors
                if report['top_individual_factors']:
                    print("  Top Confluence Factors:")
                    for i, (factor, stats) in enumerate(report['top_individual_factors'][:3], 1):
                        print(f"    {i}. {factor}: {stats['win_rate']:.1f}% WR, ${stats['avg_profit']:.2f} avg, weight {stats['recommended_weight']}")
            except:
                pass

        print("=" * 80)
        print()

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def should_filter_setup(self, quality: Dict, min_quality: str = 'MEDIUM') -> bool:
        """
        Check if setup should be filtered out

        Args:
            quality: Output from score_setup_quality()
            min_quality: Minimum acceptable quality tier

        Returns:
            True if setup should be skipped, False if it's acceptable
        """
        quality_tiers = ['UNKNOWN', 'POOR', 'MEDIUM', 'GOOD', 'VERY_GOOD', 'EXCELLENT']

        if quality['quality_tier'] not in quality_tiers:
            return False  # Unknown quality, let it through

        setup_level = quality_tiers.index(quality['quality_tier'])
        min_level = quality_tiers.index(min_quality)

        return setup_level < min_level

    def adjust_position_size(self, base_size: float, quality: Dict) -> float:
        """
        Adjust position size based on setup quality

        Args:
            base_size: Base lot size (e.g., 0.04)
            quality: Output from score_setup_quality()

        Returns:
            Adjusted lot size
        """
        multipliers = {
            'EXCELLENT': 1.5,
            'VERY_GOOD': 1.25,
            'GOOD': 1.0,
            'MEDIUM': 0.75,
            'POOR': 0.5,
            'UNKNOWN': 1.0
        }

        multiplier = multipliers.get(quality['quality_tier'], 1.0)
        return round(base_size * multiplier, 2)


# Example usage
if __name__ == '__main__':
    # Initialize
    ml_manager = MLIntegrationManager(enable_adaptive_weighting=True)

    # Example: Log a trade
    trade = {
        'ticket': 12345678,
        'symbol': 'EURUSD',
        'direction': 'buy',
        'expected_price': 1.10500,
        'entry_price': 1.10508,
        'volume': 0.04,
        'spread_at_entry_pips': 1.2,
        'fill_time_ms': 450,
        'confluence_score': 12,
        'confluence_factors': ['vwap_band_2', 'poc', 'swing_low', 'daily_hvn'],
        'adx': 28.5,
        'atr_pips': 85.3,
        'hour': 12
    }
    ml_manager.log_trade_entry(trade)
    print("[OK] Trade logged")

    # Example: Score a setup
    factors = ['vwap_band_2', 'poc', 'swing_low', 'daily_hvn']
    quality = ml_manager.score_setup_quality(factors)
    print(f"\n[OK] Setup scored: {quality['quality_tier']} ({quality['score']:.1f}/100)")
    print(f"  Win probability: {quality['win_probability']:.1f}%")
    print(f"  Expected profit: ${quality['expected_profit']:.2f}")
    print(f"  Recommendation: {quality['recommendation']}")

    # Example: Log recovery decision
    recovery = {
        'ticket': 12345678,
        'type': 'DCA',
        'pips_underwater': 35,
        'unrealized_pnl': -14.00,
        'was_blocked': True,
        'block_reason': 'SPREAD_HOUR'
    }
    ml_manager.log_recovery_decision(recovery)
    print("\n[OK] Recovery decision logged")

    # Status
    ml_manager.print_status()
