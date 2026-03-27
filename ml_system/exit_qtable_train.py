"""
Exit Q-Table Trainer — Step 2 of 2
=====================================
Reads simulated trade snapshots from exit_qtable_training.db,
discretises state variables into buckets, counts PC1 vs SL outcomes
per state, and saves the Q-table as JSON.

Run AFTER exit_qtable_builder.py has completed.

Input:  ml_system/data/exit_qtable_training.db
Output: ml_system/models/exit_qtable.json

Does NOT touch any live bot files.
The bot loads the JSON only when explicitly wired in.
"""

import sys
import os
import sqlite3
import json
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(SCRIPT_DIR, 'data',   'exit_qtable_training.db')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'models')
OUTPUT_PATH = os.path.join(MODEL_DIR,  'exit_qtable.json')

# =============================================================================
# STATE BUCKETS
# Keeping it to 3 variables x small buckets = 48 states total.
# More variables = sparser data = less reliable Q-values.
# =============================================================================

# Elapsed time buckets (minutes)
ELAPSED_BUCKETS = [
    (0,    60,  '0-60m'),
    (60,   120, '60-120m'),
    (120,  240, '120-240m'),
    (240,  999, '240m+'),
]

# Current pip position buckets
PIP_BUCKETS = [
    (-999, -10, 'deep_neg'),    # > 10 pips underwater
    (-10,   -3, 'shallow_neg'), # 3-10 pips underwater
    (-3,     3, 'near_zero'),   # within 3 pips of entry
    (3,    999, 'positive'),    # more than 3 pips ahead
]

# MAE (max adverse excursion so far) buckets
# Note: mae_pips is stored as negative (below entry), we use abs value here
MAE_BUCKETS = [
    (0,   5,  'mild'),      # worst dip < 5 pips
    (5,  15,  'moderate'),  # worst dip 5-15 pips
    (15, 999, 'deep'),      # worst dip > 15 pips
]


def get_elapsed_bucket(mins):
    for lo, hi, label in ELAPSED_BUCKETS:
        if lo <= mins < hi:
            return label
    return ELAPSED_BUCKETS[-1][2]


def get_pip_bucket(pips):
    for lo, hi, label in PIP_BUCKETS:
        if lo <= pips < hi:
            return label
    return PIP_BUCKETS[-1][2]


def get_mae_bucket(mae_pips):
    # mae_pips is negative (stored as signed), use absolute value
    abs_mae = abs(mae_pips)
    for lo, hi, label in MAE_BUCKETS:
        if lo <= abs_mae < hi:
            return label
    return MAE_BUCKETS[-1][2]


def state_key(elapsed, pips, mae):
    return f"{get_elapsed_bucket(elapsed)}|{get_pip_bucket(pips)}|{get_mae_bucket(mae)}"


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("Exit Q-Table Trainer")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Training DB not found: {DB_PATH}")
        print("Run exit_qtable_builder.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Count total snapshots
    c.execute("SELECT COUNT(*) as n FROM trade_snapshots")
    total = c.fetchone()['n']
    print(f"Loading {total:,} snapshots from DB ...")

    # state_key -> {'hold_pc1': n, 'hold_sl': n, 'hold_timeout': n}
    state_counts = defaultdict(lambda: {'hold_pc1': 0, 'hold_sl': 0, 'hold_timeout': 0})

    # Also track per-symbol breakdown
    symbol_counts = defaultdict(lambda: defaultdict(lambda: {'hold_pc1': 0, 'hold_sl': 0, 'hold_timeout': 0}))

    # Pull snapshots joined with trade outcome
    c.execute("""
        SELECT
            s.elapsed_mins,
            s.pips_now,
            s.mae_pips,
            s.mfe_pips,
            s.future_pc1,
            t.outcome,
            t.symbol
        FROM trade_snapshots s
        JOIN trade_simulations t ON s.trade_id = t.id
    """)

    rows = c.fetchall()
    conn.close()

    print(f"Processing {len(rows):,} rows ...")

    for row in rows:
        key = state_key(row['elapsed_mins'], row['pips_now'], row['mae_pips'])

        if row['future_pc1'] == 1:
            state_counts[key]['hold_pc1'] += 1
            symbol_counts[row['symbol']][key]['hold_pc1'] += 1
        elif row['outcome'] == 'sl':
            state_counts[key]['hold_sl'] += 1
            symbol_counts[row['symbol']][key]['hold_sl'] += 1
        else:
            state_counts[key]['hold_timeout'] += 1
            symbol_counts[row['symbol']][key]['hold_timeout'] += 1

    # Build Q-table: for each state, Q(HOLD) = P(eventually reach PC1)
    qtable = {}
    print(f"\n{'State':45}  {'Samples':>8}  {'PC1%':>6}  {'Q(HOLD)':>8}  {'Action':>8}")
    print("-" * 85)

    all_states = sorted(state_counts.keys())
    for key in all_states:
        counts = state_counts[key]
        n_pc1     = counts['hold_pc1']
        n_sl      = counts['hold_sl']
        n_timeout = counts['hold_timeout']
        n_total   = n_pc1 + n_sl + n_timeout

        if n_total == 0:
            continue

        # Q(HOLD) = probability of eventual PC1 if we hold
        q_hold = n_pc1 / n_total

        # Simple decision: HOLD if P(PC1) >= 0.30, else CLOSE
        # Threshold 0.30 = we need at least 30% chance to stay in
        action = 'HOLD' if q_hold >= 0.30 else 'CLOSE'

        qtable[key] = {
            'q_hold':   round(q_hold, 4),
            'action':   action,
            'n_total':  n_total,
            'n_pc1':    n_pc1,
            'n_sl':     n_sl,
            'n_timeout': n_timeout,
        }

        print(f"{key:45}  {n_total:>8,}  {100*q_hold:>5.1f}%  {q_hold:>8.4f}  {action:>8}")

    # Summary stats
    n_close_states = sum(1 for v in qtable.values() if v['action'] == 'CLOSE')
    n_hold_states  = sum(1 for v in qtable.values() if v['action'] == 'HOLD')
    print(f"\nStates defined: {len(qtable)}")
    print(f"  HOLD:  {n_hold_states} states")
    print(f"  CLOSE: {n_close_states} states")

    # Save Q-table
    os.makedirs(MODEL_DIR, exist_ok=True)
    output = {
        'version':   '1.0',
        'built_at':  __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'),
        'description': (
            'Exit Q-table: state = (elapsed_bucket, pip_bucket, mae_bucket). '
            'Q(HOLD) = probability of reaching PC1 if held. '
            'Action = CLOSE if Q(HOLD) < 0.30.'
        ),
        'buckets': {
            'elapsed': [[lo, hi, lbl] for lo, hi, lbl in ELAPSED_BUCKETS],
            'pips':    [[lo, hi, lbl] for lo, hi, lbl in PIP_BUCKETS],
            'mae':     [[lo, hi, lbl] for lo, hi, lbl in MAE_BUCKETS],
        },
        'threshold': 0.30,
        'states':    qtable,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nQ-table saved to: {OUTPUT_PATH}")
    print("\nNext step: review the table, then wire into confluence_strategy.py when ready.")
    print("=" * 60)


if __name__ == '__main__':
    main()
