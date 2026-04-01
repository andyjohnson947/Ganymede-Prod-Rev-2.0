"""
Q-Table Implementation

Simple tabular Q-learning for trade/no-trade decisions.
Stores Q-values for (state, action) pairs and supports save/load as JSON.
"""

import json
from typing import Dict, Tuple, Optional
from collections import defaultdict


ACTIONS = ('TRADE', 'NO_TRADE')


class QTable:
    """Tabular Q-table for trade signal filtering."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.0):
        """
        Args:
            alpha: Learning rate (0.1 = moderate learning speed)
            gamma: Discount factor (0.0 = single-step, no future discounting)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.version = 1  # v1=no direction, v2=direction-aware
        # q_values[state_key] = {'TRADE': float, 'NO_TRADE': float}
        self.q_values: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'TRADE': 0.0, 'NO_TRADE': 0.0}
        )
        self.visit_counts: Dict[str, int] = defaultdict(int)

    def _state_key(self, state: Tuple) -> str:
        """Convert state tuple to a hashable string key for JSON serialization."""
        return '|'.join(str(s) for s in state)

    def update(self, state: Tuple, action: str, reward: float):
        """
        Q-learning update: Q(s,a) += alpha * (reward - Q(s,a))

        Single-step (gamma=0), so no next-state max needed.
        """
        key = self._state_key(state)
        old_q = self.q_values[key][action]
        self.q_values[key][action] = old_q + self.alpha * (reward - old_q)
        self.visit_counts[key] += 1

    def get_action(self, state: Tuple) -> str:
        """Return the best action for a given state."""
        key = self._state_key(state)
        qv = self.q_values[key]
        return 'TRADE' if qv['TRADE'] >= qv['NO_TRADE'] else 'NO_TRADE'

    def get_q_values(self, state: Tuple) -> Dict[str, float]:
        """Return Q-values for both actions at a given state."""
        key = self._state_key(state)
        return dict(self.q_values[key])

    def get_visit_count(self, state: Tuple) -> int:
        """Return how many times a state has been visited during training."""
        key = self._state_key(state)
        return self.visit_counts.get(key, 0)

    def save(self, path: str):
        """Save Q-table to JSON file."""
        data = {
            'version': 2,  # v2 = direction-aware encoding
            'alpha': self.alpha,
            'gamma': self.gamma,
            'q_values': dict(self.q_values),
            'visit_counts': dict(self.visit_counts),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load Q-table from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.version = data.get('version', 1)
        self.alpha = data.get('alpha', 0.1)
        self.gamma = data.get('gamma', 0.0)

        self.q_values = defaultdict(lambda: {'TRADE': 0.0, 'NO_TRADE': 0.0})
        for key, vals in data.get('q_values', {}).items():
            self.q_values[key] = vals

        self.visit_counts = defaultdict(int)
        for key, count in data.get('visit_counts', {}).items():
            self.visit_counts[key] = count

    def get_stats(self) -> Dict:
        """Return summary statistics about the Q-table."""
        total_states = len(self.q_values)
        if total_states == 0:
            return {'total_states': 0}

        trade_q = [v['TRADE'] for v in self.q_values.values()]
        skip_q = [v['NO_TRADE'] for v in self.q_values.values()]
        visits = list(self.visit_counts.values())

        trade_favored = sum(1 for v in self.q_values.values() if v['TRADE'] > v['NO_TRADE'])
        skip_favored = total_states - trade_favored

        return {
            'total_states': total_states,
            'trade_favored': trade_favored,
            'skip_favored': skip_favored,
            'trade_pct': trade_favored / total_states * 100,
            'avg_trade_q': sum(trade_q) / total_states,
            'avg_skip_q': sum(skip_q) / total_states,
            'max_trade_q': max(trade_q),
            'min_trade_q': min(trade_q),
            'avg_visits': sum(visits) / len(visits) if visits else 0,
            'max_visits': max(visits) if visits else 0,
            'min_visits': min(visits) if visits else 0,
        }
