#!/usr/bin/env python3
"""
Sweep LEVEL_TOLERANCE_PCT to find optimal value.
Tests: 0.001 (10p), 0.0015 (15p), 0.002 (20p), 0.0025 (25p), 0.003 (30p)
All other settings: MIN_CONFLUENCE_SCORE=2, K filter (4 bar, 25% wick), Q-table active.
"""

import json
import os
import sys
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_system.qtable.state_encoder import StateEncoder, SNAPSHOT_TO_FACTOR, FACTOR_WEIGHTS
from ml_system.qtable.q_table import QTable

SNAPSHOT_DIR = os.path.join(project_root, 'ml_system', 'market_state', 'data', 'snapshots')

PIP = 0.0001
MIN_CONFLUENCE_SCORE = 2
COOLDOWN_BARS = 4
USE_TIME_FILTERS = False

RESISTANCE_FACTORS = {'Swing High', 'Prev Day VAH', 'Prev Day High', 'Daily Swing High',
                      'Prev Week High', 'Prev Week Swing High', 'Above VAH'}
SUPPORT_FACTORS = {'Swing Low', 'Prev Day VAL', 'Prev Day Low', 'Daily Swing Low',
                   'Prev Week Low', 'Prev Week Swing Low', 'Below VAL'}

INSTRUMENT_CONFIG = {
    'EURUSD': {
        'pc1_pips': 10, 'pc1_pct': 0.50,
        'pc2_pips': 20, 'pc2_pct': 0.25,
        'full_tp_pips': 40, 'trail_pct': 0.25, 'sl_pips': 20,
    },
    'GBPUSD': {
        'pc1_pips': 12, 'pc1_pct': 0.50,
        'pc2_pips': 25, 'pc2_pct': 0.25,
        'full_tp_pips': 50, 'trail_pct': 0.25, 'sl_pips': 25,
    },
}

LOT_PER_TRADE = 0.16
TRADES_PER_SIGNAL = 4
TOTAL_LOTS = LOT_PER_TRADE * TRADES_PER_SIGNAL
DOLLAR_PER_PIP_PER_LOT = 10.0


def extract_factors(snap, tolerance):
    price = snap.get('close', 0)
    if not price:
        return [], 0, None

    factors = []

    if snap.get('vwap_in_band_1'):
        factors.append('VWAP Band 1')
    if snap.get('vwap_in_band_2'):
        factors.append('VWAP Band 2')

    poc = snap.get('vp_poc', 0) or 0
    if poc and abs(price - poc) <= abs(poc * tolerance):
        factors.append('POC')

    swing_high = snap.get('vp_swing_high_price', 0) or 0
    swing_low = snap.get('vp_swing_low_price', 0) or 0
    if swing_high and abs(price - swing_high) <= abs(swing_high * tolerance):
        factors.append('Swing High')
    if swing_low and abs(price - swing_low) <= abs(swing_low * tolerance):
        factors.append('Swing Low')

    if snap.get('vp_above_vah'):
        factors.append('Above VAH')
    if snap.get('vp_below_val'):
        factors.append('Below VAL')
    if snap.get('vp_at_lvn'):
        factors.append('Low Volume Node')

    htf_checks = {
        'prev_day_vah': ('Prev Day VAH', 2),
        'prev_day_val': ('Prev Day VAL', 2),
        'prev_day_poc': ('Prev Day POC', 2),
        'prev_day_high': ('Prev Day High', 2),
        'prev_day_low': ('Prev Day Low', 2),
        'weekly_poc': ('Weekly POC', 3),
    }
    for field, (name, weight) in htf_checks.items():
        level = snap.get(field, 0) or 0
        if level and abs(price - level) <= abs(price * tolerance):
            factors.append(name)

    htf_list = snap.get('htf_factors', [])
    if htf_list:
        for f in htf_list:
            normalized = f.strip()
            if normalized and normalized not in factors:
                factors.append(normalized)

    score = 0
    for f in factors:
        key = f.lower().replace(' ', '_').replace("'s", '')
        score += FACTOR_WEIGHTS.get(key, 1)

    vwap_dir = snap.get('vwap_direction', '')
    return factors, score, vwap_dir


def get_direction(factors, vwap_dir):
    factor_set = set(factors)
    resistance = factor_set & RESISTANCE_FACTORS
    support = factor_set & SUPPORT_FACTORS

    if resistance and not support:
        return 'sell'
    elif support and not resistance:
        return 'buy'

    if vwap_dir == 'below':
        return 'buy'
    elif vwap_dir == 'above':
        return 'sell'
    return None


def check_confirmation(snap, direction):
    o = snap.get('open', 0)
    c = snap.get('close', 0)
    h = snap.get('high', 0)
    l = snap.get('low', 0)
    rng = h - l
    if rng == 0:
        return False

    if direction == 'buy':
        wick = min(o, c) - l
        return (c > o) and (wick / rng >= 0.25)
    elif direction == 'sell':
        wick = h - max(o, c)
        return (c < o) and (wick / rng >= 0.25)
    return False


def calculate_pnl(snap, direction, cfg):
    if direction == 'buy':
        hit_1r = snap.get('fwd_mr_buy_1r')
        hit_2r = snap.get('fwd_mr_buy_2r')
        mfe = snap.get('fwd_24h_mfe_up_pips', 0) or 0
        mae = snap.get('fwd_24h_mae_down_pips', 0) or 0
    else:
        hit_1r = snap.get('fwd_mr_sell_1r')
        hit_2r = snap.get('fwd_mr_sell_2r')
        mfe = snap.get('fwd_24h_mfe_down_pips', 0) or 0
        mae = snap.get('fwd_24h_mae_up_pips', 0) or 0

    if hit_1r is None:
        return None, None, 'no_data'

    pc1_pips = cfg['pc1_pips']
    pc2_pips = cfg['pc2_pips']
    pc1_pct = cfg['pc1_pct']
    pc2_pct = cfg['pc2_pct']
    trail_pct = cfg['trail_pct']
    sl_pips = cfg['sl_pips']

    if hit_2r:
        trail_capture = min(mfe * 0.6, cfg['full_tp_pips'])
        trail_capture = max(trail_capture, pc2_pips)
        pnl_dollar = (
            (TOTAL_LOTS * pc1_pct) * pc1_pips * DOLLAR_PER_PIP_PER_LOT +
            (TOTAL_LOTS * pc2_pct) * pc2_pips * DOLLAR_PER_PIP_PER_LOT +
            (TOTAL_LOTS * trail_pct) * trail_capture * DOLLAR_PER_PIP_PER_LOT
        )
        pnl_pips = (pc1_pct * pc1_pips) + (pc2_pct * pc2_pips) + (trail_pct * trail_capture)
        return pnl_pips, pnl_dollar, '2r_hit'

    elif hit_1r:
        pnl_pips = pc1_pct * pc1_pips
        pnl_dollar = (TOTAL_LOTS * pc1_pct) * pc1_pips * DOLLAR_PER_PIP_PER_LOT
        return pnl_pips, pnl_dollar, '1r_hit'

    else:
        loss = min(mae, 50)
        if loss < sl_pips:
            loss = sl_pips
        pnl_pips = -loss
        pnl_dollar = -(TOTAL_LOTS * loss * DOLLAR_PER_PIP_PER_LOT)
        return pnl_pips, pnl_dollar, 'loss'


def run_backtest(symbol, tolerance):
    path = os.path.join(SNAPSHOT_DIR, f'{symbol}_snapshots.jsonl')
    cfg = INSTRUMENT_CONFIG[symbol]

    snapshots = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(snapshots)
    start_index = max(0, total - 8760)

    q_table = None
    encoder = None
    qt_path = os.path.join(project_root, 'ml_system', 'qtable', f'q_table_{symbol}.json')
    if os.path.exists(qt_path):
        q_table = QTable()
        q_table.load(qt_path)
        encoder = StateEncoder()

    trades = []
    wrong_direction = 0  # Track trades where direction contradicts structure
    pending_signal = None
    last_trade_idx = -COOLDOWN_BARS - 1

    for i in range(start_index, total):
        snap = snapshots[i]

        if pending_signal is not None:
            ps_dir = pending_signal['direction']
            confirmed = check_confirmation(snap, ps_dir)
            pending_signal['bars_waited'] += 1

            if confirmed:
                pnl_pips, pnl_dollar, outcome = calculate_pnl(snap, ps_dir, cfg)
                if pnl_pips is not None:
                    trades.append({
                        'direction': ps_dir,
                        'factors': pending_signal['factors'],
                        'score': pending_signal['score'],
                        'pnl_pips': pnl_pips,
                        'pnl_dollar': pnl_dollar,
                        'outcome': outcome,
                    })
                    last_trade_idx = i
                pending_signal = None
                continue
            elif pending_signal['bars_waited'] >= 4:
                pending_signal = None
                continue
            else:
                continue

        if i - last_trade_idx < COOLDOWN_BARS:
            continue

        factors, score, vwap_dir = extract_factors(snap, tolerance)
        if score < MIN_CONFLUENCE_SCORE:
            continue

        direction = get_direction(factors, vwap_dir)
        if not direction:
            continue

        if q_table and encoder:
            hour = snap.get('hour_utc', 0)
            day = snap.get('day_of_week', 0)
            adx = snap.get('adx', 0) or 0
            atr_pct = snap.get('atr_percentile_50', 0.5) or 0.5
            vwap_dist = snap.get('vwap_distance_pct', 0) or 0

            market_state = {
                'hour_utc': hour, 'day_of_week': day, 'adx': adx,
                'atr_percentile_50': atr_pct, 'vwap_distance_pct': vwap_dist,
                'active_factors': factors,
            }
            state = encoder.encode(market_state)
            q_vals = q_table.get_q_values(state)
            visits = q_table.get_visit_count(state)

            if q_vals['NO_TRADE'] > q_vals['TRADE'] and visits >= 5:
                continue

        pending_signal = {
            'direction': direction,
            'factors': factors,
            'score': score,
            'bars_waited': 0,
        }

    return trades


def main():
    tolerances = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]

    print("=" * 100)
    print("TOLERANCE SWEEP — LEVEL_TOLERANCE_PCT")
    print(f"Settings: MIN_CONFLUENCE_SCORE={MIN_CONFLUENCE_SCORE}, K=4 bars/25% wick, Q-table active")
    print("=" * 100)

    results = []

    for tol in tolerances:
        pips_approx = tol * 10000  # Rough pips equivalent
        all_trades = []

        for symbol in ['EURUSD', 'GBPUSD']:
            trades = run_backtest(symbol, tol)
            all_trades.extend(trades)

        total = len(all_trades)
        if total == 0:
            results.append({'tol': tol, 'pips': pips_approx, 'trades': 0})
            continue

        wins = [t for t in all_trades if t['pnl_dollar'] > 0]
        losses = [t for t in all_trades if t['pnl_dollar'] <= 0]
        total_pnl = sum(t['pnl_dollar'] for t in all_trades)
        wr = len(wins) / total * 100
        avg_trade = total_pnl / total

        # Max drawdown
        equity = 0
        peak = 0
        max_dd = 0
        for t in all_trades:
            equity += t['pnl_dollar']
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        gross_win = sum(t['pnl_dollar'] for t in wins)
        gross_loss = abs(sum(t['pnl_dollar'] for t in losses)) if losses else 0.01
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

        results.append({
            'tol': tol,
            'pips': pips_approx,
            'trades': total,
            'wr': wr,
            'total_pnl': total_pnl,
            'avg_trade': avg_trade,
            'max_dd': max_dd,
            'pf': pf,
        })

    # Print comparison table
    print(f"\n{'Tolerance':>10} | {'~Pips':>5} | {'Trades':>6} | {'WR%':>6} | {'Total P/L':>10} | {'Avg/Trade':>9} | {'Max DD':>8} | {'PF':>5}")
    print("-" * 80)

    for r in results:
        if r['trades'] == 0:
            print(f"{r['tol']:>10.4f} | {r['pips']:>5.0f} | {r['trades']:>6} | {'N/A':>6} | {'N/A':>10} | {'N/A':>9} | {'N/A':>8} | {'N/A':>5}")
        else:
            marker = " <-- BEST" if r['total_pnl'] == max(x['total_pnl'] for x in results if x['trades'] > 0) else ""
            print(f"{r['tol']:>10.4f} | {r['pips']:>5.0f} | {r['trades']:>6} | {r['wr']:>5.1f}% | ${r['total_pnl']:>9,.0f} | ${r['avg_trade']:>8.2f} | ${r['max_dd']:>7.0f} | {r['pf']:>5.2f}{marker}")

    # Per-symbol breakdown for top 3
    print("\n--- Per-Symbol Breakdown (Top 3 by P/L) ---")
    top3 = sorted([r for r in results if r['trades'] > 0], key=lambda x: x['total_pnl'], reverse=True)[:3]

    for r in top3:
        print(f"\nTolerance: {r['tol']} (~{r['pips']:.0f} pips)")
        for symbol in ['EURUSD', 'GBPUSD']:
            trades = run_backtest(symbol, r['tol'])
            if not trades:
                print(f"  {symbol}: 0 trades")
                continue
            wins = [t for t in trades if t['pnl_dollar'] > 0]
            pnl = sum(t['pnl_dollar'] for t in trades)
            wr = len(wins) / len(trades) * 100
            print(f"  {symbol}: {len(trades)} trades, {wr:.1f}% WR, ${pnl:,.0f}")


if __name__ == '__main__':
    main()
