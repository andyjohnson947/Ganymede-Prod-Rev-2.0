"""
Exit Q-Table Builder — Step 1 of 2
====================================
Reads local H1 CSV files (16 years of EURUSD + GBPUSD), simulates
mean-reversion entries, tracks pip position at hourly intervals, and
writes results to SQLite for Q-table training.

CSV format (tab-separated, no header required):
  datetime | open | high | low | close | volume
  e.g. 2010-03-15 22:00  1.36751  1.36801  1.36697  1.36712  1901

Run this ONCE to generate training data.
Then run exit_qtable_train.py to build the Q-table.

Output: ml_system/data/exit_qtable_training.db

Does NOT touch any live bot files.
"""

import sys
import os
import csv
import sqlite3
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
IMPORT_DIR = os.path.join(SCRIPT_DIR, 'import')
DB_PATH    = os.path.join(DATA_DIR, 'exit_qtable_training.db')

# =============================================================================
# CONFIG
# =============================================================================
SYMBOLS = {
    'EURUSD': {
        'csv':       'EURUSD60.csv',
        'pc1_pips':  10,    # PC1 target: +10 pips
        'sl_pips':   35,    # Full SL: -35 pips
        'pip_value': 0.0001,
    },
    'GBPUSD': {
        'csv':       'GBPUSD60.csv',
        'pc1_pips':  12,    # PC1 target: +12 pips
        'sl_pips':   38,    # Full SL: -38 pips
        'pip_value': 0.0001,
    },
}

# Each H1 bar = 60 minutes, so snapshot marks align exactly with bar boundaries
SNAPSHOT_HOURS  = [1, 2, 3, 4, 5, 6, 7, 8]   # hours after entry
MAX_TRADE_HOURS = 8                             # 8-hour max (covers all live trade outliers)

# Entry signal: simplified mean-reversion proxy
SMA_PERIOD     = 20    # 20-bar SMA on H1
ATR_PERIOD     = 14    # ATR(14)
ENTRY_ATR_MULT = 0.4   # Price must be >= 0.4x ATR beyond SMA to qualify


# =============================================================================
# CSV LOADER
# =============================================================================
def load_csv(path):
    """
    Load H1 CSV file. Handles tab or comma separated.
    Expected columns: datetime, open, high, low, close, volume
    Returns list of dicts sorted by time.
    """
    bars = []
    with open(path, 'r', encoding='utf-8') as fh:
        # Detect delimiter
        sample = fh.read(512)
        fh.seek(0)
        delim = '\t' if '\t' in sample else ','

        reader = csv.reader(fh, delimiter=delim)
        for row in reader:
            if not row or len(row) < 5:
                continue
            # Skip header rows
            if row[0].strip().lower() in ('date', 'datetime', 'time', '<date>'):
                continue
            try:
                dt_str = row[0].strip()
                # Handle "YYYY.MM.DD HH:MM" or "YYYY-MM-DD HH:MM"
                dt_str = dt_str.replace('.', '-').replace('/', '-')
                if len(dt_str) == 10:          # date only, no time
                    dt_str += ' 00:00'
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
                bars.append({
                    'time':  dt,
                    'open':  float(row[1]),
                    'high':  float(row[2]),
                    'low':   float(row[3]),
                    'close': float(row[4]),
                })
            except (ValueError, IndexError):
                continue

    bars.sort(key=lambda b: b['time'])
    return bars


# =============================================================================
# INDICATORS
# =============================================================================
def calc_sma(closes, period):
    out = np.full(len(closes), np.nan)
    cs  = np.cumsum(closes)
    for i in range(period - 1, len(closes)):
        out[i] = (cs[i] - (cs[i - period] if i >= period else 0)) / period
    return out


def calc_atr(highs, lows, closes, period):
    tr  = np.maximum(highs[1:] - lows[1:],
          np.maximum(np.abs(highs[1:] - closes[:-1]),
                     np.abs(lows[1:]  - closes[:-1])))
    atr = np.full(len(closes), np.nan)
    if len(tr) >= period:
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(closes)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr


# =============================================================================
# ENTRY SIGNAL DETECTION
# =============================================================================
def find_entries(bars):
    """
    Simplified mean-reversion signal on H1:
      BUY  — previous close below SMA - ATR*mult, current bar closes higher
      SELL — previous close above SMA + ATR*mult, current bar closes lower

    Returns list of (bar_index, direction) tuples.
    """
    closes = np.array([b['close'] for b in bars])
    highs  = np.array([b['high']  for b in bars])
    lows   = np.array([b['low']   for b in bars])

    sma  = calc_sma(closes, SMA_PERIOD)
    atr  = calc_atr(highs, lows, closes, ATR_PERIOD)

    entries = []
    warmup  = max(SMA_PERIOD, ATR_PERIOD) + 2

    for i in range(warmup, len(bars) - MAX_TRADE_HOURS - 1):
        if np.isnan(sma[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue

        prev_c = closes[i - 1]
        curr_c = closes[i]
        s      = sma[i]
        a      = atr[i]

        if prev_c < s - ENTRY_ATR_MULT * a and curr_c > prev_c:
            entries.append((i, 'buy'))
        elif prev_c > s + ENTRY_ATR_MULT * a and curr_c < prev_c:
            entries.append((i, 'sell'))

    return entries


# =============================================================================
# TRADE SIMULATION  (H1 bars — each bar = 1 snapshot mark)
# =============================================================================
def simulate_trade(bars, entry_idx, direction, pc1_pips, sl_pips, pip_value):
    """
    Forward-simulate from entry_idx using H1 bars.
    Each bar forward = 1 hour elapsed.
    Snapshots recorded at each hour mark (1h, 2h, ... 8h).
    Returns dict with outcome and snapshots.
    """
    entry_price = bars[entry_idx]['close']
    sign        = 1 if direction == 'buy' else -1

    mae_pips = 0.0
    mfe_pips = 0.0
    snapshots     = []
    outcome       = 'timeout'
    outcome_mins  = MAX_TRADE_HOURS * 60

    end_idx = min(entry_idx + MAX_TRADE_HOURS + 1, len(bars))

    for j in range(entry_idx + 1, end_idx):
        bar     = bars[j]
        elapsed_hours = j - entry_idx   # each bar = 1 hour

        # Directional high/low
        if direction == 'buy':
            pips_high  = (bar['high']  - entry_price) / pip_value
            pips_low   = (bar['low']   - entry_price) / pip_value
        else:
            pips_high  = (entry_price - bar['low'])   / pip_value
            pips_low   = (entry_price - bar['high'])  / pip_value

        pips_close = sign * (bar['close'] - entry_price) / pip_value

        # Update excursions
        if pips_high > mfe_pips:
            mfe_pips = pips_high
        if pips_low < mae_pips:
            mae_pips = pips_low

        # Check PC1 hit (intra-bar high)
        if pips_high >= pc1_pips:
            outcome      = 'pc1'
            outcome_mins = elapsed_hours * 60
            break

        # Check SL hit (intra-bar low)
        if pips_low <= -sl_pips:
            outcome      = 'sl'
            outcome_mins = elapsed_hours * 60
            break

        # Record snapshot at each completed hour
        if elapsed_hours in SNAPSHOT_HOURS:
            snapshots.append({
                'elapsed_mins': elapsed_hours * 60,
                'pips_now':     round(pips_close, 2),
                'mae_pips':     round(mae_pips, 2),
                'mfe_pips':     round(mfe_pips, 2),
            })

    # Tag snapshots — did PC1 eventually arrive?
    future_pc1 = 1 if outcome == 'pc1' else 0
    for snap in snapshots:
        snap['future_pc1'] = future_pc1

    return {
        'outcome':      outcome,
        'outcome_mins': outcome_mins,
        'max_pips':     round(mfe_pips, 2),
        'min_pips':     round(mae_pips, 2),
        'snapshots':    snapshots,
    }


# =============================================================================
# DB SETUP
# =============================================================================
def init_db(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trade_simulations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT,
            direction    TEXT,
            entry_time   TEXT,
            entry_price  REAL,
            outcome      TEXT,
            outcome_mins INTEGER,
            max_pips     REAL,
            min_pips     REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS trade_snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id     INTEGER,
            elapsed_mins INTEGER,
            pips_now     REAL,
            mae_pips     REAL,
            mfe_pips     REAL,
            future_pc1   INTEGER
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_snap_trade ON trade_snapshots(trade_id)")
    conn.commit()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 62)
    print("Exit Q-Table Builder  (H1 CSV — 16 years)")
    print("=" * 62)

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed old DB: {DB_PATH}\n")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur  = conn.cursor()

    total_trades    = 0
    total_snapshots = 0

    for symbol, cfg in SYMBOLS.items():
        csv_path  = os.path.join(IMPORT_DIR, cfg['csv'])
        pc1_pips  = cfg['pc1_pips']
        sl_pips   = cfg['sl_pips']
        pip_value = cfg['pip_value']

        if not os.path.exists(csv_path):
            print(f"[{symbol}] CSV not found: {csv_path} — skipping")
            continue

        print(f"[{symbol}] Loading {cfg['csv']} ...")
        bars = load_csv(csv_path)

        if not bars:
            print(f"[{symbol}] No bars loaded — skipping")
            continue

        print(f"[{symbol}] {len(bars):,} bars  |  {bars[0]['time'].date()} -> {bars[-1]['time'].date()}")

        print(f"[{symbol}] Scanning for entry signals ...")
        entries = find_entries(bars)
        print(f"[{symbol}] Found {len(entries):,} entry signals")

        sym_trades  = 0
        sym_pc1     = 0
        sym_sl      = 0
        sym_timeout = 0

        for idx, (bar_idx, direction) in enumerate(entries):
            if idx % 2000 == 0:
                pct = 100 * idx / len(entries)
                print(f"  [{symbol}] {idx:,}/{len(entries):,}  ({pct:.0f}%) ...", end='\r')

            result = simulate_trade(bars, bar_idx, direction, pc1_pips, sl_pips, pip_value)

            if not result['snapshots']:
                continue  # Resolved before first snapshot — skip

            cur.execute("""
                INSERT INTO trade_simulations
                (symbol, direction, entry_time, entry_price, outcome, outcome_mins, max_pips, min_pips)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                direction,
                bars[bar_idx]['time'].strftime('%Y-%m-%d %H:%M'),
                bars[bar_idx]['close'],
                result['outcome'],
                result['outcome_mins'],
                result['max_pips'],
                result['min_pips'],
            ))
            trade_id = cur.lastrowid

            for snap in result['snapshots']:
                cur.execute("""
                    INSERT INTO trade_snapshots
                    (trade_id, elapsed_mins, pips_now, mae_pips, mfe_pips, future_pc1)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    snap['elapsed_mins'],
                    snap['pips_now'],
                    snap['mae_pips'],
                    snap['mfe_pips'],
                    snap['future_pc1'],
                ))

            sym_trades      += 1
            total_snapshots += len(result['snapshots'])

            if result['outcome'] == 'pc1':    sym_pc1     += 1
            elif result['outcome'] == 'sl':   sym_sl      += 1
            else:                             sym_timeout += 1

        conn.commit()
        total_trades += sym_trades

        print(f"\n[{symbol}] Complete: {sym_trades:,} trades stored")
        print(f"  PC1:     {sym_pc1:,}  ({100*sym_pc1/max(sym_trades,1):.1f}%)")
        print(f"  SL:      {sym_sl:,}  ({100*sym_sl/max(sym_trades,1):.1f}%)")
        print(f"  Timeout: {sym_timeout:,}  ({100*sym_timeout/max(sym_trades,1):.1f}%)\n")

    conn.close()

    print("=" * 62)
    print("COMPLETE")
    print(f"  Total trades:    {total_trades:,}")
    print(f"  Total snapshots: {total_snapshots:,}")
    print(f"  DB:              {DB_PATH}")
    print("\nNext step: run exit_qtable_train.py")
    print("=" * 62)


if __name__ == '__main__':
    main()
