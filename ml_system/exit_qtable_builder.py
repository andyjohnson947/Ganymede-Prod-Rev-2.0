"""
Exit Q-Table Builder — Step 1 of 2
====================================
Pulls 6 years of M15 data from MT5, simulates mean-reversion entries,
tracks pip position at 30-min intervals for each trade, and writes
results to SQLite for Q-table training.

Run this ONCE to generate training data.
Then run exit_qtable_train.py to build the Q-table.

Output: ml_system/data/exit_qtable_training.db

Does NOT touch any live bot files.
"""

import sys
import os
import sqlite3
import numpy as np
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')

# Add bot root to path so we can import nothing (standalone script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
DB_PATH    = os.path.join(DATA_DIR, 'exit_qtable_training.db')

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[ERROR] MetaTrader5 package not found")
    sys.exit(1)

# =============================================================================
# CONFIG
# =============================================================================
SYMBOLS = {
    'EURUSD': {'pc1_pips': 10, 'sl_pips': 35, 'pip_value': 0.0001},
    'GBPUSD': {'pc1_pips': 12, 'sl_pips': 38, 'pip_value': 0.0001},
}
FROM_DATE       = datetime(2020, 1, 1)
TO_DATE         = datetime(2026, 3, 1)   # up to just before current month
SNAPSHOT_MINS   = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 420, 480]
MAX_TRADE_MINS  = 480   # 8 hours max (covers the 8h37m outlier seen in live data)
M15_BARS_PER_SNAPSHOT = 2  # 30 min / 15 min per bar

# Entry signal parameters (simplified mean-reversion proxy)
SMA_PERIOD  = 20    # 20-bar SMA on M15
ATR_PERIOD  = 14    # ATR(14) for volatility context
ENTRY_ATR_MULT = 0.4  # Price must be at least 0.4x ATR away from SMA to qualify


# =============================================================================
# DB SETUP
# =============================================================================
def init_db(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trade_simulations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT,
            direction   TEXT,
            entry_time  TEXT,
            entry_price REAL,
            outcome     TEXT,       -- 'pc1', 'sl', 'timeout'
            outcome_mins INTEGER,   -- minutes until outcome resolved
            max_pips    REAL,       -- best pip excursion during trade
            min_pips    REAL        -- worst pip excursion (MAE)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS trade_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        INTEGER,
            elapsed_mins    INTEGER,
            pips_now        REAL,
            mae_pips        REAL,   -- max adverse excursion up to this point
            mfe_pips        REAL,   -- max favourable excursion up to this point
            future_pc1      INTEGER -- 1 = PC1 was eventually reached from this point
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_snap_trade ON trade_snapshots(trade_id)")
    conn.commit()


# =============================================================================
# INDICATORS
# =============================================================================
def calc_sma(closes, period):
    """Simple moving average — returns array same length as input (NaN for first period-1)."""
    out = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        out[i] = np.mean(closes[i - period + 1:i + 1])
    return out


def calc_atr(highs, lows, closes, period):
    """Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(abs(highs[1:] - closes[:-1]),
                    abs(lows[1:] - closes[:-1])))
    atr = np.full(len(closes), np.nan)
    if len(tr) >= period:
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(closes)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr


# =============================================================================
# ENTRY SIGNAL DETECTION
# =============================================================================
def find_entries(bars, pip_value):
    """
    Simplified mean-reversion entry signal:
    BUY  — previous close below SMA - ATR*mult, current bar reversal (close > prev close)
    SELL — previous close above SMA + ATR*mult, current bar reversal (close < prev close)

    Returns list of (bar_index, direction) tuples.
    """
    closes = np.array([b['close'] for b in bars])
    highs  = np.array([b['high']  for b in bars])
    lows   = np.array([b['low']   for b in bars])

    sma = calc_sma(closes, SMA_PERIOD)
    atr = calc_atr(highs, lows, closes, ATR_PERIOD)

    entries = []
    warmup = max(SMA_PERIOD, ATR_PERIOD) + 2

    for i in range(warmup, len(bars) - 1):
        if np.isnan(sma[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue

        prev_c = closes[i - 1]
        curr_c = closes[i]
        s      = sma[i]
        a      = atr[i]

        # BUY: price stretched below SMA, reversal candle
        if prev_c < s - ENTRY_ATR_MULT * a and curr_c > prev_c:
            entries.append((i, 'buy'))

        # SELL: price stretched above SMA, reversal candle
        elif prev_c > s + ENTRY_ATR_MULT * a and curr_c < prev_c:
            entries.append((i, 'sell'))

    return entries


# =============================================================================
# TRADE SIMULATION
# =============================================================================
def simulate_trade(bars, entry_idx, direction, pc1_pips, sl_pips, pip_value):
    """
    Forward-simulate a trade from entry_idx.
    Returns dict with outcome, snapshots, and trade stats.
    """
    entry_price = bars[entry_idx]['close']
    sign        = 1 if direction == 'buy' else -1

    mae_pips = 0.0   # max adverse excursion (always <= 0)
    mfe_pips = 0.0   # max favourable excursion (always >= 0)

    snapshots       = []  # (elapsed_mins, pips_now, mae, mfe)
    outcome         = 'timeout'
    outcome_mins    = MAX_TRADE_MINS
    bars_per_min    = 1 / 15  # 1 M15 bar = 15 minutes

    next_snapshot_idx = 0  # index into SNAPSHOT_MINS list
    max_bars = min(entry_idx + MAX_TRADE_MINS // 15 + 1, len(bars))

    for j in range(entry_idx + 1, max_bars):
        bar       = bars[j]
        elapsed   = (j - entry_idx) * 15  # minutes

        # Current pip position (use close of each bar for snapshots)
        pips_close = sign * (bar['close'] - entry_price) / pip_value

        # High/low in direction terms
        if direction == 'buy':
            pips_high = (bar['high'] - entry_price) / pip_value
            pips_low  = (bar['low']  - entry_price) / pip_value
        else:
            pips_high = (entry_price - bar['low'])  / pip_value
            pips_low  = (entry_price - bar['high']) / pip_value

        # Update excursions
        if pips_high > mfe_pips:
            mfe_pips = pips_high
        if pips_low < mae_pips:
            mae_pips = pips_low

        # Check PC1 hit (high touched target)
        if pips_high >= pc1_pips:
            outcome      = 'pc1'
            outcome_mins = elapsed
            break

        # Check SL hit (low breached stop)
        if pips_low <= -sl_pips:
            outcome      = 'sl'
            outcome_mins = elapsed
            break

        # Record snapshot at each 30-min mark
        if next_snapshot_idx < len(SNAPSHOT_MINS):
            if elapsed >= SNAPSHOT_MINS[next_snapshot_idx]:
                snapshots.append({
                    'elapsed_mins': SNAPSHOT_MINS[next_snapshot_idx],
                    'pips_now':     round(pips_close, 2),
                    'mae_pips':     round(mae_pips, 2),
                    'mfe_pips':     round(mfe_pips, 2),
                })
                next_snapshot_idx += 1

    # After outcome is known, tag each snapshot with future_pc1
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
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("Exit Q-Table Builder — generating training data")
    print("=" * 60)

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    # Remove old DB if exists (fresh run)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed old DB: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    c = conn.cursor()

    total_trades    = 0
    total_snapshots = 0

    for symbol, cfg in SYMBOLS.items():
        pc1_pips  = cfg['pc1_pips']
        sl_pips   = cfg['sl_pips']
        pip_value = cfg['pip_value']

        print(f"\n[{symbol}] Pulling M15 bars {FROM_DATE.date()} -> {TO_DATE.date()} ...")
        bars_raw = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, FROM_DATE, TO_DATE)

        if bars_raw is None or len(bars_raw) == 0:
            print(f"[{symbol}] No bars returned — skipping")
            continue

        print(f"[{symbol}] {len(bars_raw):,} bars received")

        # Convert to list of dicts for easier access
        bars = [{'time': b[0], 'open': b[1], 'high': b[2], 'low': b[3], 'close': b[4]}
                for b in bars_raw]

        # Detect entry signals
        print(f"[{symbol}] Scanning for entry signals ...")
        entries = find_entries(bars, pip_value)
        print(f"[{symbol}] Found {len(entries):,} entry signals")

        sym_trades    = 0
        sym_pc1       = 0
        sym_sl        = 0
        sym_timeout   = 0

        for idx, (bar_idx, direction) in enumerate(entries):
            # Progress every 1000 entries
            if idx % 1000 == 0:
                print(f"  [{symbol}] Processing entry {idx:,}/{len(entries):,} ...", end='\r')

            entry_time = datetime.utcfromtimestamp(bars[bar_idx]['time'])

            result = simulate_trade(bars, bar_idx, direction, pc1_pips, sl_pips, pip_value)

            if not result['snapshots']:
                continue  # Trade resolved before first 30-min snapshot — skip

            # Insert trade
            c.execute("""
                INSERT INTO trade_simulations
                (symbol, direction, entry_time, entry_price, outcome, outcome_mins, max_pips, min_pips)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                direction,
                entry_time.strftime('%Y-%m-%d %H:%M'),
                bars[bar_idx]['close'],
                result['outcome'],
                result['outcome_mins'],
                result['max_pips'],
                result['min_pips'],
            ))
            trade_id = c.lastrowid

            # Insert snapshots
            for snap in result['snapshots']:
                c.execute("""
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

            sym_trades += 1
            total_snapshots += len(result['snapshots'])
            if result['outcome'] == 'pc1':     sym_pc1 += 1
            elif result['outcome'] == 'sl':    sym_sl += 1
            else:                              sym_timeout += 1

        conn.commit()
        total_trades += sym_trades

        print(f"\n[{symbol}] Done: {sym_trades:,} trades stored")
        print(f"  PC1: {sym_pc1:,} ({100*sym_pc1/max(sym_trades,1):.1f}%)")
        print(f"  SL:  {sym_sl:,} ({100*sym_sl/max(sym_trades,1):.1f}%)")
        print(f"  Timeout: {sym_timeout:,} ({100*sym_timeout/max(sym_trades,1):.1f}%)")

    conn.close()
    mt5.shutdown()

    print("\n" + "=" * 60)
    print(f"COMPLETE")
    print(f"  Total trades:    {total_trades:,}")
    print(f"  Total snapshots: {total_snapshots:,}")
    print(f"  DB saved to:     {DB_PATH}")
    print(f"\nNext step: run exit_qtable_train.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
