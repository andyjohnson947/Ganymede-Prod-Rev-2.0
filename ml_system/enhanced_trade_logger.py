#!/usr/bin/env python3
"""
Enhanced Trade Logger - Phase 1 Implementation

Immediate improvements:
1. Execution quality tracking (slippage, spread, fill time)
2. Market conditions at entry (ADX, ATR, session, spread status)
3. Continuous logging with enhanced metadata

Usage:
    python3 ml_system/enhanced_trade_logger.py &
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class EnhancedTradeLogger:
    """Enhanced continuous logger with execution quality and market condition tracking"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "ml_system" / "outputs"
        self.output_dir.mkdir(exist_ok=True)

        self.log_file = self.output_dir / "enhanced_trade_log.jsonl"
        self.execution_log = self.output_dir / "execution_quality.jsonl"
        self.market_conditions_log = self.output_dir / "market_conditions.jsonl"

        print(f"[ENHANCED LOGGER] Starting...")
        print(f"  Trade log: {self.log_file}")
        print(f"  Execution log: {self.execution_log}")
        print(f"  Market conditions log: {self.market_conditions_log}")

    def log_trade_with_execution(self, trade_data: Dict):
        """
        Log trade with enhanced execution quality data

        Enhanced fields:
        - slippage (expected vs actual entry)
        - spread_at_entry
        - fill_time_ms
        - execution_quality_score (0-100)
        """
        # Add execution quality metrics
        execution_quality = {
            'timestamp': datetime.now().isoformat(),
            'ticket': trade_data.get('ticket'),
            'symbol': trade_data.get('symbol'),

            # Execution quality
            'expected_price': trade_data.get('expected_price'),
            'actual_price': trade_data.get('entry_price'),
            'slippage_pips': self._calculate_slippage(
                trade_data.get('expected_price'),
                trade_data.get('entry_price'),
                trade_data.get('symbol')
            ),
            'spread_at_entry_pips': trade_data.get('spread_at_entry_pips', 0),
            'fill_time_ms': trade_data.get('fill_time_ms', 0),
            'requotes': trade_data.get('requotes', 0),

            # Quality score (0-100)
            'execution_quality_score': self._calculate_execution_quality(trade_data)
        }

        # Log to execution quality file
        with open(self.execution_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(execution_quality, ensure_ascii=False) + '\n')

        # Add to main trade data
        trade_data['execution_quality'] = execution_quality

        # Log to main trade log
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trade_data, ensure_ascii=False) + '\n')

    def log_market_conditions(self, symbol: str, conditions: Dict):
        """
        Log market conditions at signal detection time

        Conditions to track:
        - ADX (trend strength)
        - ATR (volatility)
        - Spread status (normal/widened)
        - Session (Tokyo/London/NY/Sydney)
        - Distance to nearest level
        - Volume profile position
        """
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,

            # Trend and volatility
            'adx': conditions.get('adx'),
            'adx_classification': self._classify_adx(conditions.get('adx', 0)),
            'atr_pips': conditions.get('atr_pips'),
            'volatility_regime': self._classify_volatility(conditions.get('atr_pips', 0)),

            # Spread
            'spread_pips': conditions.get('spread_pips'),
            'spread_status': 'WIDENED' if conditions.get('spread_pips', 0) > 2 else 'NORMAL',

            # Time context
            'hour': conditions.get('hour'),
            'session': self._get_session(conditions.get('hour', 0)),
            'is_spread_hour': conditions.get('hour', 0) in [0, 9, 13, 20, 21],

            # Technical context
            'distance_to_level_pips': conditions.get('distance_to_level_pips'),
            'at_hvn': conditions.get('at_hvn', False),
            'at_lvn': conditions.get('at_lvn', False),
            'at_poc': conditions.get('at_poc', False),

            # Signal quality
            'confluence_score': conditions.get('confluence_score'),
            'signal_strength': self._classify_signal_strength(conditions.get('confluence_score', 0))
        }

        # Log to market conditions file
        with open(self.market_conditions_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(market_data, ensure_ascii=False) + '\n')

        return market_data

    def log_recovery_decision(self, decision_data: Dict):
        """
        Log recovery trigger decision with context

        Captures WHY recovery was triggered and surrounding conditions
        """
        recovery_log = self.output_dir / "recovery_decisions.jsonl"

        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'ticket': decision_data.get('ticket'),
            'recovery_type': decision_data.get('type'),  # DCA/Hedge/Grid

            # Trigger conditions
            'price_at_trigger': decision_data.get('price_at_trigger'),
            'pips_underwater': decision_data.get('pips_underwater'),
            'unrealized_pnl': decision_data.get('unrealized_pnl'),
            'time_underwater_minutes': decision_data.get('time_underwater_minutes'),

            # Market conditions at trigger
            'adx_at_entry': decision_data.get('adx_at_entry'),
            'adx_at_trigger': decision_data.get('adx_at_trigger'),
            'spread_at_trigger': decision_data.get('spread_at_trigger'),
            'hour_at_trigger': decision_data.get('hour_at_trigger'),

            # Decision outcome
            'was_blocked': decision_data.get('was_blocked', False),
            'block_reason': decision_data.get('block_reason'),
            'recovery_placed': decision_data.get('recovery_placed', False)
        }

        with open(recovery_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(decision_record, ensure_ascii=False) + '\n')

    def log_near_miss_signal(self, signal_data: Dict):
        """
        Log signals that were blocked/skipped

        Tracks what we DIDN'T trade and validates blocking rules
        """
        near_miss_log = self.output_dir / "near_miss_signals.jsonl"

        near_miss = {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal_data.get('symbol'),
            'confluence_score': signal_data.get('confluence_score'),
            'direction': signal_data.get('direction'),

            # Why was it blocked?
            'block_reason': signal_data.get('block_reason'),
            'adx': signal_data.get('adx'),
            'hour': signal_data.get('hour'),
            'spread_pips': signal_data.get('spread_pips'),

            # What happened after? (to be filled in later)
            'price_at_signal': signal_data.get('price'),
            'price_1h_later': None,  # TODO: Track this
            'price_4h_later': None,  # TODO: Track this
            'dodged_bullet': None  # True if price went against signal
        }

        with open(near_miss_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(near_miss, ensure_ascii=False) + '\n')

    # Helper methods

    def _calculate_slippage(self, expected: Optional[float], actual: Optional[float], symbol: str) -> float:
        """Calculate slippage in pips"""
        if not expected or not actual:
            return 0.0

        # Assuming 4/5 digit pricing
        point = 0.0001 if 'JPY' not in symbol else 0.01
        pip_diff = abs(actual - expected) / point
        return round(pip_diff, 2)

    def _calculate_execution_quality(self, trade_data: Dict) -> int:
        """Calculate execution quality score (0-100)"""
        score = 100

        # Penalize slippage
        slippage = abs(trade_data.get('slippage_pips', 0))
        if slippage > 2:
            score -= min(30, slippage * 5)

        # Penalize wide spreads
        spread = trade_data.get('spread_at_entry_pips', 0)
        if spread > 2:
            score -= min(20, (spread - 2) * 10)

        # Penalize slow fills
        fill_time = trade_data.get('fill_time_ms', 0)
        if fill_time > 1000:  # > 1 second
            score -= 20

        # Penalize requotes
        requotes = trade_data.get('requotes', 0)
        score -= requotes * 10

        return max(0, score)

    def _classify_adx(self, adx: float) -> str:
        """Classify ADX value"""
        if adx < 20:
            return 'WEAK_NO_TREND'
        elif adx < 25:
            return 'DEVELOPING_TREND'
        elif adx < 30:
            return 'MODERATE_TREND'
        elif adx < 40:
            return 'STRONG_TREND'
        else:
            return 'VERY_STRONG_TREND'

    def _classify_volatility(self, atr_pips: float) -> str:
        """Classify volatility regime"""
        if atr_pips < 50:
            return 'LOW'
        elif atr_pips < 100:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _get_session(self, hour: int) -> str:
        """Determine trading session"""
        if 0 <= hour < 8:
            return 'Tokyo'
        elif 8 <= hour < 13:
            return 'London'
        elif 13 <= hour < 21:
            return 'NY'
        else:
            return 'Sydney'

    def _classify_signal_strength(self, confluence: int) -> str:
        """Classify signal strength"""
        if confluence < 9:
            return 'WEAK'
        elif confluence < 13:
            return 'MODERATE'
        elif confluence < 17:
            return 'STRONG'
        else:
            return 'VERY_STRONG'


# Example usage demonstrating the enhanced logger
def example_usage():
    """Show how to use the enhanced logger"""
    logger = EnhancedTradeLogger()

    # Example 1: Log a trade with execution quality
    trade = {
        'ticket': 12345678,
        'symbol': 'EURUSD',
        'direction': 'buy',
        'expected_price': 1.10500,
        'entry_price': 1.10508,  # 0.8 pip slippage
        'volume': 0.04,
        'spread_at_entry_pips': 1.2,
        'fill_time_ms': 450,
        'requotes': 0,
        'confluence_score': 12
    }
    logger.log_trade_with_execution(trade)
    print("[EXAMPLE] Logged trade with execution quality")

    # Example 2: Log market conditions at signal time
    conditions = {
        'adx': 28.5,
        'atr_pips': 85.3,
        'spread_pips': 1.5,
        'hour': 12,
        'distance_to_level_pips': 5.2,
        'at_poc': True,
        'confluence_score': 12
    }
    logger.log_market_conditions('EURUSD', conditions)
    print("[EXAMPLE] Logged market conditions")

    # Example 3: Log a recovery decision
    recovery_decision = {
        'ticket': 12345678,
        'type': 'DCA',
        'price_at_trigger': 1.10400,
        'pips_underwater': 35,
        'unrealized_pnl': -14.00,
        'time_underwater_minutes': 45,
        'adx_at_entry': 28.5,
        'adx_at_trigger': 32.1,
        'spread_at_trigger': 2.5,
        'hour_at_trigger': 13,
        'was_blocked': True,
        'block_reason': 'ADX_HARD_STOPS_ENABLED'
    }
    logger.log_recovery_decision(recovery_decision)
    print("[EXAMPLE] Logged recovery decision")

    # Example 4: Log a near-miss signal
    near_miss = {
        'symbol': 'GBPUSD',
        'confluence_score': 11,
        'direction': 'sell',
        'block_reason': 'SPREAD_HOUR',
        'adx': 22.3,
        'hour': 0,
        'spread_pips': 3.8,
        'price': 1.27500
    }
    logger.log_near_miss_signal(near_miss)
    print("[EXAMPLE] Logged near-miss signal")

    print()
    print("=" * 80)
    print("Enhanced logger examples complete!")
    print("Check ml_system/outputs/ for generated files:")
    print("  - enhanced_trade_log.jsonl")
    print("  - execution_quality.jsonl")
    print("  - market_conditions.jsonl")
    print("  - recovery_decisions.jsonl")
    print("  - near_miss_signals.jsonl")
    print("=" * 80)


if __name__ == '__main__':
    # Run example to demonstrate functionality
    example_usage()
