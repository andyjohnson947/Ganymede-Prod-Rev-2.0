#!/usr/bin/env python3
"""
Q-Table Training Script

Trains per-symbol Q-tables from 6-year hourly snapshot data.
Uses structural direction (Swing High=SELL, Swing Low=BUY, HTF overrides)
and MFE-based rewards matched to per-instrument PC pip targets.

Direction: Structural levels override VWAP (matches production signal_detector.py)
Rewards: MFE >= PC2 pips = +2.0, MFE >= PC1 pips = +0.5, else = -1.0
PC targets: EURUSD 10/20p, GBPUSD 12/25p (from instruments_config.py)

Usage:
    python -m ml_system.qtable.train_qtable
    python -m ml_system.qtable.train_qtable --symbol EURUSD
    python -m ml_system.qtable.train_qtable --passes 5
"""

import json
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_system.qtable.state_encoder import StateEncoder, SNAPSHOT_TO_FACTOR, FACTOR_WEIGHTS
from ml_system.qtable.q_table import QTable

SNAPSHOT_DIR = os.path.join(project_root, 'ml_system', 'market_state', 'data', 'snapshots')
OUTPUT_DIR = os.path.join(project_root, 'ml_system', 'qtable')

MIN_CONFLUENCE_SCORE = 7  # Must match strategy_config.py

# Per-instrument PC pip targets (must match instruments_config.py)
TARGETS = {
    'EURUSD': {'pc1': 10, 'pc2': 20},
    'GBPUSD': {'pc1': 12, 'pc2': 25},
}

# Structural direction factor sets (must match signal_detector.py / confluence_strategy.py)
RESISTANCE_FACTORS = {'Swing High', 'Prev Day VAH', 'Prev Day High', 'Daily Swing High',
                      'Prev Week High', 'Prev Week Swing High', 'Above VAH'}
SUPPORT_FACTORS = {'Swing Low', 'Prev Day VAL', 'Prev Day Low', 'Daily Swing Low',
                   'Prev Week Low', 'Prev Week Swing Low', 'Below VAL'}


def has_signal(snap: dict, encoder: StateEncoder) -> bool:
    """Check if a snapshot would have triggered a confluence signal."""
    factors = encoder._extract_factors_from_snapshot(snap)
    score = encoder.calculate_confluence_score(factors)
    return score >= MIN_CONFLUENCE_SCORE


def get_factors(snap: dict, encoder: StateEncoder) -> list:
    """Extract active confluence factors from snapshot."""
    return encoder._extract_factors_from_snapshot(snap)


def get_direction(snap: dict, factors: list) -> str:
    """
    Structural direction: levels override VWAP.
    Matches production logic in signal_detector.py.
    """
    factor_set = set(factors)
    resistance = factor_set & RESISTANCE_FACTORS
    support = factor_set & SUPPORT_FACTORS

    # Structural direction takes priority
    if resistance and not support:
        return 'sell'
    elif support and not resistance:
        return 'buy'

    # Fallback to VWAP
    vwap_dir = snap.get('vwap_direction', '')
    if vwap_dir == 'below':
        return 'buy'
    elif vwap_dir == 'above':
        return 'sell'

    # Last resort: VWAP distance
    dist = snap.get('vwap_distance_pct', 0) or 0
    return 'buy' if dist < 0 else 'sell'


def get_reward(snap: dict, direction: str, symbol: str) -> float:
    """
    MFE-based reward using per-instrument PC pip targets.

    Uses raw MFE (max favorable excursion) instead of hardcoded boolean fields.
    This matches the actual PC1/PC2 pip targets in instruments_config.py:
      EURUSD: PC1=10p, PC2=20p
      GBPUSD: PC1=12p, PC2=25p

    50/25/25 PC structure rewards:
      - Hit PC2 (2R): PC1 fires at 1R + PC2 fires at 2R + trail = +2.0
      - Hit PC1 only (1R): PC1 fires, rest at BE = +0.5
      - Miss: full SL hit = -1.0
      - No MFE data: skip = 0.0
    """
    t = TARGETS.get(symbol, TARGETS['EURUSD'])

    if direction == 'buy':
        mfe = snap.get('fwd_24h_mfe_up_pips', None)
    else:
        mfe = snap.get('fwd_24h_mfe_down_pips', None)

    if mfe is None:
        return 0.0

    mfe = mfe or 0

    if mfe >= t['pc2']:
        return 2.0   # Hit 2R: full PC treatment
    elif mfe >= t['pc1']:
        return 0.5   # Hit 1R: PC1 only, rest at BE
    else:
        return -1.0  # Miss: SL hit


def train_symbol(symbol: str, passes: int = 3, alpha: float = 0.1):
    """Train Q-table for a single symbol."""
    snapshot_file = os.path.join(SNAPSHOT_DIR, f'{symbol}_snapshots.jsonl')
    if not os.path.exists(snapshot_file):
        print(f"  [SKIP] No snapshot file: {snapshot_file}")
        return None

    file_size = os.path.getsize(snapshot_file) / (1024 * 1024)
    print(f"  Reading {snapshot_file} ({file_size:.1f} MB)")

    encoder = StateEncoder()
    qt = QTable(alpha=alpha)

    total_snapshots = 0
    signal_snapshots = 0
    skipped_no_outcome = 0

    for pass_num in range(1, passes + 1):
        # Decay learning rate across passes
        qt.alpha = alpha / pass_num

        pass_signals = 0
        pass_trades = 0
        pass_reward_sum = 0

        with open(snapshot_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    snap = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if pass_num == 1:
                    total_snapshots += 1

                # Check if this snapshot would trigger a signal
                if not has_signal(snap, encoder):
                    continue

                pass_signals += 1
                if pass_num == 1:
                    signal_snapshots += 1

                # Get structural direction and MFE-based reward
                factors = get_factors(snap, encoder)
                direction = get_direction(snap, factors)
                reward = get_reward(snap, direction, symbol)

                if reward == 0.0:
                    if pass_num == 1:
                        skipped_no_outcome += 1
                    continue

                pass_trades += 1
                pass_reward_sum += reward

                # Encode state (include direction for v2 direction-aware encoding)
                snap['direction'] = direction
                state = encoder.encode(snap)

                # Update Q-table
                # TRADE action gets the actual reward
                qt.update(state, 'TRADE', reward)
                # NO_TRADE action always gets 0 (no risk, no reward)
                qt.update(state, 'NO_TRADE', 0.0)

        avg_reward = pass_reward_sum / pass_trades if pass_trades > 0 else 0
        print(f"  Pass {pass_num}/{passes}: {pass_signals} signals, "
              f"{pass_trades} with outcomes, avg reward: {avg_reward:+.3f}, "
              f"alpha: {qt.alpha:.4f}")

    # Save Q-table
    output_path = os.path.join(OUTPUT_DIR, f'q_table_{symbol}.json')
    qt.save(output_path)

    stats = qt.get_stats()
    print(f"\n  {symbol} Q-Table Stats:")
    print(f"    Total snapshots: {total_snapshots:,}")
    print(f"    Signal snapshots: {signal_snapshots:,} ({signal_snapshots/max(total_snapshots,1)*100:.1f}%)")
    print(f"    Skipped (no outcome): {skipped_no_outcome:,}")
    print(f"    Unique states: {stats['total_states']}")
    print(f"    Trade-favored states: {stats['trade_favored']} ({stats['trade_pct']:.1f}%)")
    print(f"    Skip-favored states: {stats['skip_favored']} ({100-stats['trade_pct']:.1f}%)")
    print(f"    Avg Q(TRADE): {stats['avg_trade_q']:.4f}")
    print(f"    Avg Q(NO_TRADE): {stats['avg_skip_q']:.4f}")
    print(f"    Avg visits/state: {stats['avg_visits']:.1f}")
    print(f"    Saved to: {output_path}")

    return qt


def main():
    parser = argparse.ArgumentParser(description='Train Q-tables from snapshot data')
    parser.add_argument('--symbol', type=str, help='Train specific symbol only')
    parser.add_argument('--passes', type=int, default=3, help='Training passes (default: 3)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Initial learning rate (default: 0.1)')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else ['EURUSD', 'GBPUSD']

    print("=" * 60)
    print("  Q-TABLE TRAINING (Structural Direction + MFE Rewards)")
    print(f"  Symbols: {symbols}")
    print(f"  Passes: {args.passes}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Targets: {TARGETS}")
    print("=" * 60)

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {symbol}")
        print(f"{'='*60}")
        train_symbol(symbol, passes=args.passes, alpha=args.alpha)

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
