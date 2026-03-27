#!/usr/bin/env python3
"""
Setup Classifier - Learns Best Trade Setups Over Time

Classifies trade setups into tiers based on historical performance:
- PLATINUM: 80%+ win rate, 20+ trades - best setups, full size
- GOLD: 70-79% win rate, 15+ trades - strong setups, full size
- SILVER: 60-69% win rate, 10+ trades - decent setups, normal size
- BRONZE: 50-59% win rate, 10+ trades - marginal setups, reduced size
- UNPROVEN: <10 trades - not enough data, cautious size

The classifier:
1. Tracks every trade's confluence factor combination
2. Calculates win rate for each unique setup (factor combo)
3. Learns which combinations perform best over time
4. Provides tier classification for new signals
5. Auto-updates after each closed trade
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib


class SetupClassifier:
    """
    Learns and classifies trade setups based on historical performance.
    """

    # Classification tiers with thresholds
    TIERS = {
        'PLATINUM': {'min_win_rate': 80, 'min_trades': 20, 'size_multiplier': 1.0, 'emoji': '💎'},
        'GOLD': {'min_win_rate': 70, 'min_trades': 15, 'size_multiplier': 1.0, 'emoji': '🥇'},
        'SILVER': {'min_win_rate': 60, 'min_trades': 10, 'size_multiplier': 1.0, 'emoji': '🥈'},
        'BRONZE': {'min_win_rate': 50, 'min_trades': 10, 'size_multiplier': 0.75, 'emoji': '🥉'},
        'UNPROVEN': {'min_win_rate': 0, 'min_trades': 0, 'size_multiplier': 0.5, 'emoji': '❓'},
    }

    def __init__(self, data_dir: str = None):
        """
        Initialize the setup classifier.

        Args:
            data_dir: Directory to store setup performance data
        """
        if data_dir is None:
            # Default to ml_system/outputs
            self.data_dir = Path(__file__).parent.parent.parent / 'ml_system' / 'outputs'
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Setup performance tracking file
        self.setups_file = self.data_dir / 'setup_performance.json'
        self.history_file = self.data_dir / 'setup_classification_history.jsonl'

        # Load existing setup data
        self.setups: Dict[str, Dict] = {}
        self._load_setups()

    def _load_setups(self):
        """Load setup performance data from file"""
        if self.setups_file.exists():
            try:
                with open(self.setups_file, 'r', encoding='utf-8') as f:
                    self.setups = json.load(f)
                print(f"[SETUP CLASSIFIER] Loaded {len(self.setups)} tracked setups")
            except Exception as e:
                print(f"[SETUP CLASSIFIER] Error loading setups: {e}")
                self.setups = {}
        else:
            self.setups = {}

    def _save_setups(self):
        """Save setup performance data to file"""
        try:
            with open(self.setups_file, 'w', encoding='utf-8') as f:
                json.dump(self.setups, f, indent=2)
        except Exception as e:
            print(f"[SETUP CLASSIFIER] Error saving setups: {e}")

    def _get_setup_key(self, factors: List[str], strategy_type: str = None,
                       hour: int = None, day_of_week: str = None) -> str:
        """
        Generate a unique key for a setup based on factors AND context.

        Args:
            factors: List of confluence factors
            strategy_type: 'mean_reversion', 'breakout', 'vwap', etc.
            hour: Hour of entry (0-23)
            day_of_week: Day of week ('Monday', 'Tuesday', etc.)

        Returns:
            Unique hash key for this factor + context combination
        """
        # Sort factors for consistent ordering
        sorted_factors = sorted([f.lower().replace(' ', '_') for f in factors])
        factor_str = '|'.join(sorted_factors)

        # Include strategy, hour, day in the hash for granular tracking
        context_str = f"{strategy_type or 'any'}|h{hour if hour is not None else 'X'}|{(day_of_week or 'any')[:3]}"
        full_str = f"{factor_str}|{context_str}"
        hash_part = hashlib.md5(full_str.encode()).hexdigest()[:8]

        # Create readable key with strategy abbreviation
        first_factor = sorted_factors[0] if sorted_factors else 'empty'
        strat_abbrev = {'mean_reversion': 'MR', 'breakout': 'BO', 'vwap': 'MR'}.get(strategy_type, '?')
        hour_str = f"h{hour}" if hour is not None else "hX"

        return f"{len(sorted_factors)}f_{strat_abbrev}_{hour_str}_{first_factor[:6]}_{hash_part}"

    def _get_setup_signature(self, factors: List[str]) -> str:
        """Get human-readable setup signature"""
        sorted_factors = sorted(factors)
        return ' + '.join(sorted_factors[:5])  # Show first 5 factors

    def record_trade(
        self,
        factors: List[str],
        direction: str,
        symbol: str,
        is_win: bool,
        profit: float = 0.0,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        htf_score: int = 0,
        entry_score: int = 0,
        ticket: int = None,
        strategy_type: str = None,
        hour: int = None,
        day_of_week: str = None
    ):
        """
        Record a trade result for a setup.

        Args:
            factors: List of confluence factors that triggered the trade
            direction: 'buy' or 'sell'
            symbol: Trading symbol
            is_win: Whether the trade was profitable
            profit: Profit amount
            entry_price: Entry price
            exit_price: Exit price
            htf_score: HTF confluence score
            entry_score: Entry confluence score
            ticket: Trade ticket number
            strategy_type: 'mean_reversion', 'breakout', etc. (NEW)
            hour: Hour of entry 0-23 (NEW)
            day_of_week: Day of week e.g. 'Monday' (NEW)
        """
        # Normalize direction to lowercase
        direction = direction.lower() if direction else 'buy'

        setup_key = self._get_setup_key(factors, strategy_type, hour, day_of_week)

        if setup_key not in self.setups:
            self.setups[setup_key] = {
                'factors': sorted(factors),
                'signature': self._get_setup_signature(factors),
                'strategy_type': strategy_type,  # NEW: Track strategy context
                'hour': hour,                    # NEW: Track hour context
                'day_of_week': day_of_week,      # NEW: Track day context
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_profit': 0.0,
                'avg_profit': 0.0,
                'win_rate': 0.0,
                'tier': 'UNPROVEN',
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'symbols': {},
                'directions': {'buy': {'wins': 0, 'losses': 0}, 'sell': {'wins': 0, 'losses': 0}},
                'htf_scores': [],
                'entry_scores': [],
                'recent_results': [],  # Last 10 results for trend analysis
            }

        setup = self.setups[setup_key]

        # Update stats
        setup['total_trades'] += 1
        setup['total_profit'] += profit
        setup['last_seen'] = datetime.now().isoformat()

        if is_win:
            setup['wins'] += 1
            setup['directions'][direction]['wins'] += 1
        else:
            setup['losses'] += 1
            setup['directions'][direction]['losses'] += 1

        # Update symbol tracking
        if symbol not in setup['symbols']:
            setup['symbols'][symbol] = {'wins': 0, 'losses': 0, 'profit': 0.0}
        setup['symbols'][symbol]['wins' if is_win else 'losses'] += 1
        setup['symbols'][symbol]['profit'] += profit

        # Track scores
        if htf_score > 0:
            setup['htf_scores'].append(htf_score)
            # Keep only last 50 scores
            setup['htf_scores'] = setup['htf_scores'][-50:]
        if entry_score > 0:
            setup['entry_scores'].append(entry_score)
            setup['entry_scores'] = setup['entry_scores'][-50:]

        # Track recent results for trend
        setup['recent_results'].append({
            'is_win': is_win,
            'profit': profit,
            'timestamp': datetime.now().isoformat()
        })
        setup['recent_results'] = setup['recent_results'][-10:]

        # Recalculate metrics
        setup['win_rate'] = (setup['wins'] / setup['total_trades']) * 100
        setup['avg_profit'] = setup['total_profit'] / setup['total_trades']

        # Reclassify tier
        setup['tier'] = self._classify_tier(setup)

        # Save updated data
        self._save_setups()

        # Log to history
        self._log_classification(setup_key, setup, ticket)

        return setup

    def _classify_tier(self, setup: Dict) -> str:
        """
        Classify a setup into a tier based on performance.

        Args:
            setup: Setup performance data

        Returns:
            Tier name (PLATINUM, GOLD, SILVER, BRONZE, UNPROVEN)
        """
        trades = setup['total_trades']
        win_rate = setup['win_rate']

        # Check tiers in order (best to worst)
        for tier_name, thresholds in self.TIERS.items():
            if trades >= thresholds['min_trades'] and win_rate >= thresholds['min_win_rate']:
                return tier_name

        return 'UNPROVEN'

    def _log_classification(self, setup_key: str, setup: Dict, ticket: int = None):
        """Log classification to history file"""
        try:
            record = {
                'timestamp': datetime.now().isoformat(),
                'setup_key': setup_key,
                'signature': setup['signature'],
                'tier': setup['tier'],
                'win_rate': round(setup['win_rate'], 1),
                'trades': setup['total_trades'],
                'ticket': ticket
            }
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception:
            pass  # Non-critical, don't fail on logging

    def classify_signal(self, factors: List[str], symbol: str = None) -> Dict:
        """
        Classify a new signal based on historical performance of similar setups.

        Args:
            factors: List of confluence factors for the signal
            symbol: Optional symbol to get symbol-specific stats (EURUSD, GBPUSD)

        Returns:
            Dict with classification info:
            {
                'tier': str,
                'emoji': str,
                'win_rate': float,
                'trades': int,
                'size_multiplier': float,
                'confidence': str,
                'is_proven': bool,
                'recommendation': str,
                'trade_count': int,  # Number of trades to open based on tier
                'symbol_stats': dict,  # Symbol-specific performance
                'factor_quality': dict,  # Individual factor effectiveness
            }
        """
        # Get factor quality scores
        factor_quality = self._get_factor_quality(factors, symbol)

        setup_key = self._get_setup_key(factors)

        if setup_key in self.setups:
            setup = self.setups[setup_key]

            # Get symbol-specific stats if symbol provided
            symbol_stats = None
            symbol_tier = None
            symbol_win_rate = 0.0
            symbol_trades = 0

            if symbol and symbol in setup.get('symbols', {}):
                sym_data = setup['symbols'][symbol]
                symbol_trades = sym_data['wins'] + sym_data['losses']
                if symbol_trades > 0:
                    symbol_win_rate = (sym_data['wins'] / symbol_trades) * 100
                    symbol_stats = {
                        'symbol': symbol,
                        'wins': sym_data['wins'],
                        'losses': sym_data['losses'],
                        'trades': symbol_trades,
                        'win_rate': round(symbol_win_rate, 1),
                        'profit': round(sym_data['profit'], 2)
                    }
                    # Calculate symbol-specific tier
                    symbol_tier = self._classify_tier_from_stats(symbol_win_rate, symbol_trades)

            # Use symbol-specific tier if available and has enough data, else use global
            if symbol_tier and symbol_trades >= 5:
                tier = symbol_tier
                win_rate = symbol_win_rate
                trades = symbol_trades
            else:
                tier = setup['tier']
                win_rate = setup['win_rate']
                trades = setup['total_trades']

            tier_info = self.TIERS[tier]

            # Calculate recent trend (last 5 trades)
            recent = setup.get('recent_results', [])[-5:]
            recent_win_rate = 0
            if recent:
                recent_wins = sum(1 for r in recent if r['is_win'])
                recent_win_rate = (recent_wins / len(recent)) * 100

            # Determine confidence based on trade count
            if trades >= 30:
                confidence = 'HIGH'
            elif trades >= 15:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            # Determine trade count based on tier
            # PLATINUM: 3 trades (max confidence)
            # GOLD: 2 trades (strong)
            # SILVER: 2 trades (decent)
            # BRONZE: 1 trade (reduced risk)
            # UNPROVEN: 1 trade (minimal risk)
            trade_counts = {
                'PLATINUM': 3,
                'GOLD': 2,
                'SILVER': 2,
                'BRONZE': 1,
                'UNPROVEN': 1
            }
            trade_count = trade_counts.get(tier, 1)

            # Generate recommendation
            if tier == 'PLATINUM':
                recommendation = f'STRONG BUY - Proven setup ({trades} trades, {win_rate:.0f}% WR)'
            elif tier == 'GOLD':
                recommendation = f'BUY - Strong setup ({trades} trades, {win_rate:.0f}% WR)'
            elif tier == 'SILVER':
                recommendation = f'CAUTIOUS BUY - Decent setup ({trades} trades, {win_rate:.0f}% WR)'
            elif tier == 'BRONZE':
                recommendation = f'REDUCE SIZE - Marginal setup ({trades} trades, {win_rate:.0f}% WR)'
            else:
                recommendation = 'CAUTION - Unproven setup, use minimal size'

            return {
                'tier': tier,
                'emoji': tier_info['emoji'],
                'win_rate': round(win_rate, 1),
                'trades': trades,
                'size_multiplier': tier_info['size_multiplier'],
                'trade_count': trade_count,
                'confidence': confidence,
                'is_proven': trades >= 10,
                'recommendation': recommendation,
                'recent_win_rate': round(recent_win_rate, 1),
                'avg_profit': round(setup['avg_profit'], 2),
                'signature': setup['signature'],
                'symbol_stats': symbol_stats,
                'factor_quality': factor_quality,  # Individual factor effectiveness scores
                'global_tier': setup['tier'],  # Include global tier for comparison
                'global_win_rate': round(setup['win_rate'], 1),
                'global_trades': setup['total_trades'],
            }
        else:
            # New setup - unproven
            tier_info = self.TIERS['UNPROVEN']
            return {
                'tier': 'UNPROVEN',
                'emoji': tier_info['emoji'],
                'win_rate': 0.0,
                'trades': 0,
                'size_multiplier': tier_info['size_multiplier'],
                'trade_count': 1,  # Minimal for unproven
                'confidence': 'NONE',
                'is_proven': False,
                'recommendation': 'NEW SETUP - First occurrence, use minimal size',
                'recent_win_rate': 0.0,
                'avg_profit': 0.0,
                'signature': self._get_setup_signature(factors),
                'symbol_stats': None,
                'factor_quality': factor_quality,  # Individual factor effectiveness scores
                'global_tier': 'UNPROVEN',
                'global_win_rate': 0.0,
                'global_trades': 0,
            }

    def _classify_tier_from_stats(self, win_rate: float, trades: int) -> str:
        """Classify tier from win rate and trade count"""
        for tier_name, thresholds in self.TIERS.items():
            if trades >= thresholds['min_trades'] and win_rate >= thresholds['min_win_rate']:
                return tier_name
        return 'UNPROVEN'

    def _get_factor_quality(self, factors: List[str], symbol: str = None) -> Dict:
        """
        Get quality scores for individual factors using factor effectiveness analyzer.

        Args:
            factors: List of confluence factors
            symbol: Optional symbol for symbol-specific scoring

        Returns:
            Dict with factor quality metrics
        """
        try:
            # Import factor effectiveness analyzer
            import sys
            from pathlib import Path
            ml_system_path = Path(__file__).parent.parent.parent / 'ml_system'
            if str(ml_system_path) not in sys.path:
                sys.path.insert(0, str(ml_system_path))

            from factor_effectiveness import get_setup_quality
            return get_setup_quality(factors, symbol)
        except Exception as e:
            # If factor analyzer not available, return neutral scores
            return {
                'score': 50.0,
                'avg_win_rate': 50.0,
                'best_factor': None,
                'worst_factor': None,
                'factor_breakdown': [],
                'has_winning_factor': False,
                'has_losing_factor': False,
                'error': str(e)
            }

    def get_best_setups(self, min_trades: int = 10, top_n: int = 10) -> List[Dict]:
        """
        Get the best performing setups.

        Args:
            min_trades: Minimum trades required
            top_n: Number of top setups to return

        Returns:
            List of top setups sorted by win rate
        """
        qualified = [
            {**setup, 'key': key}
            for key, setup in self.setups.items()
            if setup['total_trades'] >= min_trades
        ]

        # Sort by win rate, then by total trades
        sorted_setups = sorted(
            qualified,
            key=lambda x: (x['win_rate'], x['total_trades']),
            reverse=True
        )

        return sorted_setups[:top_n]

    def get_worst_setups(self, min_trades: int = 10, top_n: int = 10) -> List[Dict]:
        """
        Get the worst performing setups (to avoid).

        Args:
            min_trades: Minimum trades required
            top_n: Number of worst setups to return

        Returns:
            List of worst setups sorted by win rate (ascending)
        """
        qualified = [
            {**setup, 'key': key}
            for key, setup in self.setups.items()
            if setup['total_trades'] >= min_trades
        ]

        # Sort by win rate ascending (worst first)
        sorted_setups = sorted(
            qualified,
            key=lambda x: (x['win_rate'], -x['total_trades'])
        )

        return sorted_setups[:top_n]

    def get_tier_summary(self) -> Dict:
        """
        Get summary of setups by tier.

        Returns:
            Dict with tier counts and stats
        """
        summary = {tier: {'count': 0, 'total_trades': 0, 'total_profit': 0.0}
                   for tier in self.TIERS.keys()}

        for setup in self.setups.values():
            tier = setup['tier']
            summary[tier]['count'] += 1
            summary[tier]['total_trades'] += setup['total_trades']
            summary[tier]['total_profit'] += setup['total_profit']

        return summary

    def print_report(self):
        """Print a formatted report of setup classifications"""
        print()
        print("=" * 80)
        print("SETUP CLASSIFICATION REPORT")
        print("=" * 80)

        # Tier summary
        summary = self.get_tier_summary()
        print("\nTIER SUMMARY:")
        print("-" * 40)
        for tier, info in self.TIERS.items():
            stats = summary[tier]
            print(f"  {info['emoji']} {tier:10} | Setups: {stats['count']:3} | "
                  f"Trades: {stats['total_trades']:4} | Profit: ${stats['total_profit']:.2f}")

        # Best setups
        print("\nTOP 5 BEST SETUPS (PLATINUM/GOLD):")
        print("-" * 60)
        best = self.get_best_setups(min_trades=10, top_n=5)
        for i, setup in enumerate(best, 1):
            tier_info = self.TIERS[setup['tier']]
            print(f"  {i}. {tier_info['emoji']} [{setup['tier']}] {setup['win_rate']:.1f}% "
                  f"({setup['wins']}/{setup['total_trades']}) - {setup['signature']}")

        # Worst setups
        print("\nBOTTOM 5 WORST SETUPS (AVOID):")
        print("-" * 60)
        worst = self.get_worst_setups(min_trades=10, top_n=5)
        for i, setup in enumerate(worst, 1):
            tier_info = self.TIERS[setup['tier']]
            print(f"  {i}. {tier_info['emoji']} [{setup['tier']}] {setup['win_rate']:.1f}% "
                  f"({setup['wins']}/{setup['total_trades']}) - {setup['signature']}")

        print()
        print("=" * 80)


# Global instance
_classifier_instance: Optional[SetupClassifier] = None


def get_setup_classifier() -> SetupClassifier:
    """Get or create the global SetupClassifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SetupClassifier()
    return _classifier_instance


def classify_signal(factors: List[str], symbol: str = None) -> Dict:
    """Convenience function to classify a signal"""
    classifier = get_setup_classifier()
    return classifier.classify_signal(factors, symbol=symbol)


def record_trade_result(
    factors: List[str],
    direction: str,
    symbol: str,
    is_win: bool,
    profit: float = 0.0,
    **kwargs
) -> Dict:
    """Convenience function to record a trade result"""
    classifier = get_setup_classifier()
    return classifier.record_trade(
        factors=factors,
        direction=direction,
        symbol=symbol,
        is_win=is_win,
        profit=profit,
        **kwargs
    )


if __name__ == '__main__':
    # Test the classifier
    print("Setup Classifier Test")
    print("=" * 50)

    classifier = SetupClassifier()

    # Simulate some trades
    test_setups = [
        # Platinum setup - high win rate
        (['VWAP Band 1', 'POC', 'Daily HVN', 'Weekly POC'], True),
        (['VWAP Band 1', 'POC', 'Daily HVN', 'Weekly POC'], True),
        (['VWAP Band 1', 'POC', 'Daily HVN', 'Weekly POC'], True),
        (['VWAP Band 1', 'POC', 'Daily HVN', 'Weekly POC'], True),
        (['VWAP Band 1', 'POC', 'Daily HVN', 'Weekly POC'], False),

        # Gold setup
        (['VWAP Band 2', 'Swing Low', 'Prev Day POC'], True),
        (['VWAP Band 2', 'Swing Low', 'Prev Day POC'], True),
        (['VWAP Band 2', 'Swing Low', 'Prev Day POC'], True),
        (['VWAP Band 2', 'Swing Low', 'Prev Day POC'], False),

        # Bronze setup
        (['Below VAL', 'LVN'], True),
        (['Below VAL', 'LVN'], False),
        (['Below VAL', 'LVN'], True),
        (['Below VAL', 'LVN'], False),
    ]

    for factors, is_win in test_setups:
        classifier.record_trade(
            factors=factors,
            direction='buy',
            symbol='EURUSD',
            is_win=is_win,
            profit=10.0 if is_win else -5.0
        )

    # Test classification
    print("\nClassifying new signal:")
    result = classifier.classify_signal(['VWAP Band 1', 'POC', 'Daily HVN', 'Weekly POC'])
    print(f"  Tier: {result['emoji']} {result['tier']}")
    print(f"  Win Rate: {result['win_rate']}%")
    print(f"  Trades: {result['trades']}")
    print(f"  Recommendation: {result['recommendation']}")

    # Print report
    classifier.print_report()
