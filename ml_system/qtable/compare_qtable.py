#!/usr/bin/env python3
"""Compare backtest results: Q-table ON vs OFF."""
import sys, os, json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
worktree_root = os.path.join(project_root, '.claude', 'worktrees', 'stupefied-mccarthy')
sys.path.insert(0, worktree_root)
sys.path.insert(0, project_root)

from ml_system.qtable.q_table import QTable
from ml_system.qtable.state_encoder import StateEncoder

SNAPSHOT_DIR = os.path.join(project_root, 'ml_system', 'market_state', 'data', 'snapshots')

TOLERANCE = 0.001
MIN_CONFLUENCE_SCORE = 7
COOLDOWN_BARS = 4
CONFIRM_MAX_BARS = 4

CONFIGS = {
    'EURUSD': {'pc1': 10, 'pc2': 20, 'sl': 20, 'full_tp': 40, 'lot_value': 6.4},
    'GBPUSD': {'pc1': 12, 'pc2': 25, 'sl': 25, 'full_tp': 50, 'lot_value': 6.4},
}

RESISTANCE_FACTORS = {'Swing High', 'Prev Day VAH', 'Prev Day High', 'Daily Swing High',
                      'Prev Week High', 'Prev Week Swing High', 'Above VAH'}
SUPPORT_FACTORS = {'Swing Low', 'Prev Day VAL', 'Prev Day Low', 'Daily Swing Low',
                   'Prev Week Low', 'Prev Week Swing Low', 'Below VAL'}

SNAPSHOT_TO_FACTOR = {
    'vwap_in_band_1': 'vwap_band_1', 'vwap_in_band_2': 'vwap_band_2',
    'vp_at_poc': 'poc', 'vp_at_swing_high': 'swing_high', 'vp_at_swing_low': 'swing_low',
    'vp_above_vah': 'above_vah', 'vp_below_val': 'below_val', 'vp_at_lvn': 'lvn',
}
FACTOR_WEIGHTS = {
    'vwap_band_1': 1, 'vwap_band_2': 1, 'poc': 1, 'swing_high': 1, 'swing_low': 1,
    'above_vah': 1, 'below_val': 1, 'lvn': 1,
    'prev_day_vah': 2, 'prev_day_val': 2, 'prev_day_poc': 2, 'daily_hvn': 2, 'daily_poc': 2,
    'weekly_hvn': 3, 'weekly_poc': 3,
    'prev_week_swing_low': 2, 'prev_week_swing_high': 2, 'prev_week_vwap': 2,
}


def extract_factors(snap):
    factors = []
    for sf, fn in SNAPSHOT_TO_FACTOR.items():
        if snap.get(sf):
            factors.append(fn)
    for f in snap.get('htf_factors', []):
        factors.append(f.lower().replace(' ', '_').replace("'s", ''))
    score = sum(FACTOR_WEIGHTS.get(f, 0) for f in factors)
    vwap_dir = snap.get('vwap_direction', '')
    return factors, score, vwap_dir


def get_direction(factors, vwap_dir):
    fs = set(f.lower().replace(' ', '_') for f in factors)
    r = bool(fs & {f.lower().replace(' ', '_') for f in RESISTANCE_FACTORS})
    s = bool(fs & {f.lower().replace(' ', '_') for f in SUPPORT_FACTORS})
    if r and not s:
        return 'sell'
    if s and not r:
        return 'buy'
    if vwap_dir == 'below':
        return 'buy'
    if vwap_dir == 'above':
        return 'sell'
    return None


def run_bt(symbol, use_qtable=True):
    cfg = CONFIGS[symbol]
    sf = os.path.join(SNAPSHOT_DIR, f'{symbol}_snapshots.jsonl')
    snaps = []
    with open(sf) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    snaps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total = len(snaps)
    start = max(0, total - 8760)

    q_table = None
    encoder = None
    if use_qtable:
        qt_path = os.path.join(project_root, 'ml_system', 'qtable', f'q_table_{symbol}.json')
        if os.path.exists(qt_path):
            q_table = QTable()
            q_table.load(qt_path)
            encoder = StateEncoder()

    trades = []
    sig_gen = 0
    sig_blocked = 0
    sig_noconf = 0
    pending = None
    last_trade_idx = -COOLDOWN_BARS - 1

    for i in range(start, total):
        snap = snaps[i]

        # Check pending confirmation
        if pending:
            pending['bars_waited'] += 1
            d = pending['direction']
            o = snap.get('open', 0)
            h = snap.get('high', 0)
            l = snap.get('low', 0)
            c = snap.get('close', 0)
            rng = h - l if h > l else 0.0001
            confirmed = False
            if d == 'buy' and c > o and (o - l) / rng >= 0.25:
                confirmed = True
            elif d == 'sell' and c < o and (h - c) / rng >= 0.25:
                confirmed = True

            if confirmed:
                mfe_key = f'fwd_24h_mfe_{"up" if d == "buy" else "down"}_pips'
                mfe = snap.get(mfe_key, 0) or 0
                if mfe >= cfg['pc2']:
                    pnl = cfg['pc2'] * cfg['lot_value']
                elif mfe >= cfg['pc1']:
                    pnl = cfg['pc1'] * 0.5 * cfg['lot_value']
                else:
                    pnl = -cfg['sl'] * cfg['lot_value']
                trades.append(pnl)
                last_trade_idx = i
                pending = None
                continue
            elif pending['bars_waited'] >= CONFIRM_MAX_BARS:
                sig_noconf += 1
                pending = None

        if i - last_trade_idx < COOLDOWN_BARS:
            continue
        if pending:
            continue

        factors, score, vwap_dir = extract_factors(snap)
        if score < MIN_CONFLUENCE_SCORE:
            continue

        sig_gen += 1
        direction = get_direction(factors, vwap_dir)
        if not direction:
            continue

        # Q-table gate
        if q_table and encoder:
            hour = snap.get('hour_utc', 0)
            day = snap.get('day_of_week', 0)
            adx = snap.get('adx', 0) or 0
            atr_pct = snap.get('atr_percentile_50', 0.5) or 0.5
            vwap_dist = snap.get('vwap_distance_pct', 0) or 0
            ms = {
                'hour_utc': hour, 'day_of_week': day, 'adx': adx,
                'atr_percentile_50': atr_pct, 'vwap_distance_pct': vwap_dist,
                'active_factors': factors, 'direction': direction,
            }
            state = encoder.encode(ms)
            qv = q_table.get_q_values(state)
            vis = q_table.get_visit_count(state)
            if qv['NO_TRADE'] > qv['TRADE'] and vis >= 5:
                sig_blocked += 1
                continue

        pending = {'direction': direction, 'factors': factors, 'score': score, 'bars_waited': 0}

    wins = sum(1 for t in trades if t > 0)
    losses = len(trades) - wins
    wr = wins / len(trades) * 100 if trades else 0
    total_pnl = sum(trades)
    win_pnl = sum(t for t in trades if t > 0)
    loss_pnl = sum(t for t in trades if t < 0)
    pf = win_pnl / abs(loss_pnl) if loss_pnl else 999
    return {
        'trades': len(trades), 'wins': wins, 'losses': losses, 'wr': wr,
        'pnl': total_pnl, 'blocked': sig_blocked, 'signals': sig_gen,
        'avg': total_pnl / len(trades) if trades else 0, 'pf': pf,
    }


if __name__ == '__main__':
    print('=' * 72)
    print('  Q-TABLE DIRECTION FILTER: ON vs OFF  (last 12 months)')
    print('=' * 72)

    all_on = {}
    all_off = {}

    for sym in ['EURUSD', 'GBPUSD']:
        on = run_bt(sym, use_qtable=True)
        off = run_bt(sym, use_qtable=False)
        all_on[sym] = on
        all_off[sym] = off

        print(f'\n  {sym}')
        print(f'  {"":22} {"Q-TABLE ON":>14} {"Q-TABLE OFF":>14} {"DIFF":>12}')
        print(f'  {"-" * 64}')
        print(f'  {"Signals":22} {on["signals"]:>14,} {off["signals"]:>14,}')
        print(f'  {"Q-table blocked":22} {on["blocked"]:>14,} {off["blocked"]:>14,}')
        print(f'  {"Trades executed":22} {on["trades"]:>14,} {off["trades"]:>14,} {on["trades"]-off["trades"]:>+12,}')
        print(f'  {"Win Rate":22} {on["wr"]:>13.1f}% {off["wr"]:>13.1f}% {on["wr"]-off["wr"]:>+11.1f}%')
        print(f'  {"Profit Factor":22} {on["pf"]:>14.2f} {off["pf"]:>14.2f} {on["pf"]-off["pf"]:>+12.2f}')
        print(f'  {"Total P&L":22} ${on["pnl"]:>12,.2f} ${off["pnl"]:>12,.2f} ${on["pnl"]-off["pnl"]:>+10,.2f}')
        print(f'  {"Avg P&L/trade":22} ${on["avg"]:>12,.2f} ${off["avg"]:>12,.2f} ${on["avg"]-off["avg"]:>+10,.2f}')

    # Combined
    on_t = sum(v['trades'] for v in all_on.values())
    off_t = sum(v['trades'] for v in all_off.values())
    on_w = sum(v['wins'] for v in all_on.values())
    off_w = sum(v['wins'] for v in all_off.values())
    on_pnl = sum(v['pnl'] for v in all_on.values())
    off_pnl = sum(v['pnl'] for v in all_off.values())
    on_blk = sum(v['blocked'] for v in all_on.values())

    print(f'\n  {"COMBINED":22}')
    print(f'  {"-" * 64}')
    print(f'  {"Q-table blocked":22} {on_blk:>14,}')
    print(f'  {"Trades":22} {on_t:>14,} {off_t:>14,} {on_t-off_t:>+12,}')
    print(f'  {"Win Rate":22} {on_w/on_t*100:>13.1f}% {off_w/off_t*100:>13.1f}% {on_w/on_t*100-off_w/off_t*100:>+11.1f}%')
    print(f'  {"Total P&L":22} ${on_pnl:>12,.2f} ${off_pnl:>12,.2f} ${on_pnl-off_pnl:>+10,.2f}')
    print(f'  {"Avg P&L/trade":22} ${on_pnl/on_t:>12,.2f} ${off_pnl/off_t:>12,.2f} ${on_pnl/on_t-off_pnl/off_t:>+10,.2f}')
    print(f'\n  Net impact of Q-table filter: ${on_pnl - off_pnl:+,.2f}')
    print(f'  Blocked {on_blk} trades, improved avg P&L by ${on_pnl/on_t - off_pnl/off_t:+.2f}/trade')
