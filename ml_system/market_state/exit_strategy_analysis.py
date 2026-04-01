#!/usr/bin/env python3
"""
Exit Strategy Analysis — Per-Symbol MR/BO Target Optimisation

Compares the CURRENT partial close system vs alternative exit strategies
using 6 years of MFE/MAE data already in snapshots.

For each directional trade entry, simulates:
  A) Current PC system:  50% at PC1, 25% at PC2, 25% trailing
  B) Flat 1R exit:       100% at 1R (15 pips)
  C) Scaled 1R/2R:       50% at 1R + 50% trail to 2R
  D) Scaled 1R/2R/3R:    33% at 1R + 33% at 2R + 34% trail to 3R
  E) Wide 2R exit:       100% at 2R (30 pips)

All strategies use the same per-symbol SL.

Usage:
    python -m ml_system.market_state.exit_strategy_analysis
    python -m ml_system.market_state.exit_strategy_analysis --symbol GBPUSD
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from ml_system.market_state.config import SYMBOLS, SNAPSHOT_DIR
from ml_system.market_state.bar_replay_engine import load_snapshot_data


# ── R-value targets (pips) ──────────────────────────────────────────────
MR_1R = 15
MR_2R = 30
MR_3R = 45

BO_1R = 20
BO_2R = 40
BO_3R = 60

# ── Current PC settings per symbol (from instruments_config.py) ─────────
CURRENT_PC = {
    'EURUSD': {
        'pc1_pips': 7,  'pc1_pct': 0.50,
        'pc2_pips': 14, 'pc2_pct': 0.25,
        'trail_pct': 0.25, 'trail_give_back_pips': 15,
        'sl_pips': 20,
    },
    'GBPUSD': {
        'pc1_pips': 8,  'pc1_pct': 0.50,
        'pc2_pips': 16, 'pc2_pct': 0.25,
        'trail_pct': 0.25, 'trail_give_back_pips': 15,
        'sl_pips': 28,
    },
    'AUDUSD': {
        'pc1_pips': 7,  'pc1_pct': 0.50,
        'pc2_pips': 14, 'pc2_pct': 0.25,
        'trail_pct': 0.25, 'trail_give_back_pips': 15,
        'sl_pips': 20,
    },
}


def get_direction(snap: dict) -> str:
    """Determine trade direction from VWAP or DI fallback. Returns 'buy', 'sell', or 'skip'."""
    vwap_dir = snap.get('vwap_direction', '')
    if vwap_dir == 'below':
        return 'buy'
    elif vwap_dir == 'above':
        return 'sell'
    else:
        plus_di = snap.get('plus_di', 0) or 0
        minus_di = snap.get('minus_di', 0) or 0
        di_gap = plus_di - minus_di
        if di_gap > 1.0:
            return 'buy'
        elif di_gap < -1.0:
            return 'sell'
        return 'skip'


def simulate_current_pc(mfe: float, mae: float, pc: dict) -> float:
    """
    Simulate current PC system profit in pips.

    Timeline logic:
    - If MAE >= SL first (approximation: MAE >= SL AND MFE < PC1) → full SL loss
    - If MFE >= PC1: bank pc1_pct at pc1_pips, SL→BE
      - If MFE >= PC2: bank pc2_pct at pc2_pips, trail remainder
        - Trail captures: max(0, MFE - trail_give_back) on trail_pct
      - If MFE < PC2: remaining stops at BE (0 pips)
    - If MFE < PC1 and MAE >= SL: full SL loss
    - If MFE < PC1 and MAE < SL: estimate small loss (average of MFE and -MAE) / 2
    """
    sl = pc['sl_pips']
    pc1 = pc['pc1_pips']
    pc2 = pc['pc2_pips']
    pc1_pct = pc['pc1_pct']
    pc2_pct = pc['pc2_pct']
    trail_pct = pc['trail_pct']
    trail_give_back = pc['trail_give_back_pips']

    if mfe >= pc1:
        # PC1 fires — bank 50% at PC1, SL→BE
        profit = pc1_pct * pc1

        if mfe >= pc2:
            # PC2 fires — bank 25% at PC2
            profit += pc2_pct * pc2
            # Trail: remaining 25% captures (MFE - give_back), min 0 (BE)
            trail_capture = max(0, mfe - trail_give_back)
            profit += trail_pct * trail_capture
        else:
            # Never hit PC2 — remaining 50% stops at BE (0 pips)
            profit += 0  # (pc2_pct + trail_pct) * 0
        return profit
    else:
        # Never hit PC1
        if mae >= sl:
            # SL hit — full loss
            return -sl
        else:
            # Small move, closes near entry (VWAP exit, time limit, etc.)
            # Approximate: average of small gain and small drawdown
            return (mfe - mae) / 2


def simulate_flat_exit(mfe: float, mae: float, target: float, sl: float) -> float:
    """Simulate flat exit: 100% at target or SL."""
    if mfe >= target:
        return target
    elif mae >= sl:
        return -sl
    else:
        return (mfe - mae) / 2


def simulate_scaled_2(mfe: float, mae: float, t1: float, t2: float, sl: float,
                       trail_give_back: float = 15) -> float:
    """Simulate 50% at T1 + 50% trail to T2."""
    if mfe >= t1:
        profit = 0.50 * t1  # Bank 50% at T1
        if mfe >= t2:
            profit += 0.50 * t2
        else:
            # Trail: capture (MFE - give_back), min BE (0)
            trail_capture = max(0, mfe - trail_give_back)
            profit += 0.50 * trail_capture
        return profit
    elif mae >= sl:
        return -sl
    else:
        return (mfe - mae) / 2


def simulate_scaled_3(mfe: float, mae: float, t1: float, t2: float, t3: float, sl: float,
                       trail_give_back: float = 15) -> float:
    """Simulate 33% at T1 + 33% at T2 + 34% trail to T3."""
    if mfe >= t1:
        profit = 0.33 * t1  # Bank 33% at T1, SL→BE

        if mfe >= t2:
            profit += 0.33 * t2  # Bank 33% at T2

            if mfe >= t3:
                profit += 0.34 * t3
            else:
                trail_capture = max(0, mfe - trail_give_back)
                profit += 0.34 * trail_capture
        else:
            # T2 never hit — remaining 67% at BE
            trail_capture = max(0, mfe - trail_give_back)
            profit += 0.67 * trail_capture  # Small trail or 0
        return profit
    elif mae >= sl:
        return -sl
    else:
        return (mfe - mae) / 2


def analyze_symbol(symbol: str, snapshots: list) -> dict:
    """Run full exit strategy comparison for one symbol."""

    pc = CURRENT_PC.get(symbol, CURRENT_PC['EURUSD'])
    sl = pc['sl_pips']

    # MFE distribution buckets
    mfe_buckets = defaultdict(int)

    # Strategy results
    strategies = {
        'A_current_pc': [],
        'B_flat_1r': [],
        'C_scaled_1r_2r': [],
        'D_scaled_1r_2r_3r': [],
        'E_flat_2r': [],
        'F_flat_3r': [],
    }

    # Hit rate tracking
    hits = {5: 0, 7: 0, 8: 0, 10: 0, 14: 0, 15: 0, 16: 0, 20: 0, 25: 0, 30: 0, 35: 0, 40: 0, 45: 0}

    total = 0
    skipped = 0

    for snap in snapshots:
        direction = get_direction(snap)
        if direction == 'skip':
            skipped += 1
            continue

        # Get MFE/MAE in the correct direction
        if direction == 'buy':
            mfe = snap.get('fwd_24h_mfe_up_pips', 0) or 0
            mae = snap.get('fwd_24h_mae_down_pips', 0) or 0
        else:
            mfe = snap.get('fwd_24h_mfe_down_pips', 0) or 0
            mae = snap.get('fwd_24h_mae_up_pips', 0) or 0

        if mfe == 0 and mae == 0:
            skipped += 1
            continue

        total += 1

        # MFE distribution
        for level in hits.keys():
            if mfe >= level:
                hits[level] += 1

        bucket = int(mfe // 5) * 5
        mfe_buckets[bucket] += 1

        # Simulate each strategy
        strategies['A_current_pc'].append(simulate_current_pc(mfe, mae, pc))
        strategies['B_flat_1r'].append(simulate_flat_exit(mfe, mae, MR_1R, sl))
        strategies['C_scaled_1r_2r'].append(simulate_scaled_2(mfe, mae, MR_1R, MR_2R, sl))
        strategies['D_scaled_1r_2r_3r'].append(simulate_scaled_3(mfe, mae, MR_1R, MR_2R, MR_3R, sl))
        strategies['E_flat_2r'].append(simulate_flat_exit(mfe, mae, MR_2R, sl))
        strategies['F_flat_3r'].append(simulate_flat_exit(mfe, mae, MR_3R, sl))

    return {
        'symbol': symbol,
        'total': total,
        'skipped': skipped,
        'sl_pips': sl,
        'hits': hits,
        'mfe_buckets': dict(mfe_buckets),
        'strategies': {
            name: {
                'trades': len(results),
                'total_pips': sum(results),
                'avg_pips': sum(results) / len(results) if results else 0,
                'winners': sum(1 for r in results if r > 0),
                'losers': sum(1 for r in results if r < 0),
                'breakeven': sum(1 for r in results if r == 0),
                'win_rate': sum(1 for r in results if r > 0) / len(results) * 100 if results else 0,
                'avg_win': (sum(r for r in results if r > 0) / max(1, sum(1 for r in results if r > 0))),
                'avg_loss': (sum(r for r in results if r < 0) / max(1, sum(1 for r in results if r < 0))),
                'profit_factor': (
                    sum(r for r in results if r > 0) / abs(sum(r for r in results if r < 0))
                    if sum(r for r in results if r < 0) != 0 else float('inf')
                ),
            }
            for name, results in strategies.items()
        },
    }


def print_results(results: dict):
    symbol = results['symbol']
    total = results['total']
    sl = results['sl_pips']

    print(f"\n{'='*80}")
    print(f"  EXIT STRATEGY ANALYSIS: {symbol}  ({total:,} directional trades, SL={sl} pips)")
    print(f"{'='*80}")

    # MFE hit rates
    print(f"\n  --- MFE Hit Rates (how far price goes in your direction) ---")
    print(f"  {'Target':>10} | {'Hit Rate':>8} | {'Count':>8} | {'Bar'}")
    print(f"  {'-'*55}")
    for level in sorted(results['hits'].keys()):
        count = results['hits'][level]
        pct = count / total * 100 if total else 0
        bar = '#' * int(pct / 2)
        label = ''
        if level == CURRENT_PC[symbol]['pc1_pips']:
            label = ' <-- PC1'
        elif level == CURRENT_PC[symbol]['pc2_pips']:
            label = ' <-- PC2'
        elif level == MR_1R:
            label = ' <-- 1R'
        elif level == MR_2R:
            label = ' <-- 2R'
        elif level == MR_3R:
            label = ' <-- 3R'
        print(f"  {level:>7} pip | {pct:>6.1f}% | {count:>7,} | {bar}{label}")

    # Strategy comparison
    print(f"\n  --- Strategy Comparison (pips per trade, averaged over {total:,} trades) ---")
    print(f"  {'Strategy':<25} | {'Avg P/L':>8} | {'WR':>6} | {'Avg Win':>8} | {'Avg Loss':>9} | {'PF':>5} | {'Total Pips':>11}")
    print(f"  {'-'*88}")

    labels = {
        'A_current_pc':       f'A) Current PC ({CURRENT_PC[symbol]["pc1_pips"]}/{CURRENT_PC[symbol]["pc2_pips"]} pip)',
        'B_flat_1r':          f'B) Flat 1R ({MR_1R} pip)',
        'C_scaled_1r_2r':     f'C) 50/50 @ 1R/2R',
        'D_scaled_1r_2r_3r':  f'D) 33/33/34 @ 1R/2R/3R',
        'E_flat_2r':          f'E) Flat 2R ({MR_2R} pip)',
        'F_flat_3r':          f'F) Flat 3R ({MR_3R} pip)',
    }

    best_avg = -999
    best_name = ''
    for name in ['A_current_pc', 'B_flat_1r', 'C_scaled_1r_2r', 'D_scaled_1r_2r_3r', 'E_flat_2r', 'F_flat_3r']:
        s = results['strategies'][name]
        avg = s['avg_pips']
        wr = s['win_rate']
        avg_w = s['avg_win']
        avg_l = s['avg_loss']
        pf = s['profit_factor']
        total_pips = s['total_pips']

        marker = ''
        if avg > best_avg:
            best_avg = avg
            best_name = name

        pf_str = f"{pf:.2f}" if pf < 100 else "INF"
        print(f"  {labels[name]:<25} | {avg:>+7.2f} | {wr:>5.1f}% | {avg_w:>+7.2f} | {avg_l:>+8.2f} | {pf_str:>5} | {total_pips:>+10,.0f}")

    print(f"\n  >> BEST: {labels[best_name]} ({best_avg:+.2f} pips/trade)")

    # Improvement over current
    current_avg = results['strategies']['A_current_pc']['avg_pips']
    if best_name != 'A_current_pc':
        improvement = ((best_avg - current_avg) / abs(current_avg)) * 100 if current_avg != 0 else 0
        print(f"  >> vs Current: +{best_avg - current_avg:.2f} pips/trade ({improvement:+.0f}% improvement)")

    # Per-$10/pip calculation (0.10 lot)
    print(f"\n  --- Projected $ Impact (at $10/pip per 0.10 lot) ---")
    for name in ['A_current_pc', 'B_flat_1r', 'C_scaled_1r_2r', 'D_scaled_1r_2r_3r']:
        s = results['strategies'][name]
        avg_dollar = s['avg_pips'] * 10  # $10/pip at 0.10 lot
        print(f"  {labels[name]:<25}:  ${avg_dollar:>+.2f}/trade")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Exit strategy comparison')
    parser.add_argument('--symbol', type=str, help='Analyze specific symbol')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS
    all_results = {}

    for symbol in symbols:
        print(f"\nLoading {symbol} snapshots...")
        snapshots = load_snapshot_data(symbol)
        if not snapshots:
            print(f"  [SKIP] No snapshots for {symbol}")
            continue

        print(f"  Analyzing {len(snapshots):,} snapshots...")
        results = analyze_symbol(symbol, snapshots)
        all_results[symbol] = results
        print_results(results)

    # Cross-symbol summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"  CROSS-SYMBOL SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Symbol':<8} | {'Current':>8} | {'Flat 1R':>8} | {'50/50':>8} | {'33/33/34':>8} | {'Flat 2R':>8} | {'Best':>12}")
        print(f"  {'-'*72}")
        for sym, r in all_results.items():
            strats = r['strategies']
            a = strats['A_current_pc']['avg_pips']
            b = strats['B_flat_1r']['avg_pips']
            c = strats['C_scaled_1r_2r']['avg_pips']
            d = strats['D_scaled_1r_2r_3r']['avg_pips']
            e = strats['E_flat_2r']['avg_pips']
            best_val = max(a, b, c, d, e)
            best_label = ['Current', 'Flat 1R', '50/50', '33/33/34', 'Flat 2R'][[a, b, c, d, e].index(best_val)]
            print(f"  {sym:<8} | {a:>+7.2f} | {b:>+7.2f} | {c:>+7.2f} | {d:>+7.2f} | {e:>+7.2f} | {best_label:>12}")

    # Save results
    output_path = Path(__file__).parent / 'data' / 'models' / 'exit_strategy_analysis.json'
    save_data = {}
    for sym, r in all_results.items():
        save_data[sym] = {
            'total': r['total'],
            'sl_pips': r['sl_pips'],
            'hits': r['hits'],
            'strategies': r['strategies'],
        }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[SAVED] {output_path}")


if __name__ == '__main__':
    main()
