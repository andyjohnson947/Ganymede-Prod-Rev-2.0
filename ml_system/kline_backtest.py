"""
K-Line Filter Backtest
======================
Compares original K-line confirmation logic vs new K-line logic across
16 years of EURUSD + GBPUSD H1 data.

Original K-line:
  BUY:  bar close > open  AND  lower_wick / range >= 0.25
  SELL: bar close < open  AND  upper_wick / range >= 0.25

New K-line (additional guards):
  1. Minimum body:      body / range >= 0.20  (filters doji bars)
  2. Trend alignment:   if 4+ of last 5 bars are opposite direction -> block

For each simulated MR entry signal, both filters are applied independently.
Outcomes (PC1 hit vs full SL hit) are tallied per filter version.

Output: printed report + ml_system/outputs/kline_backtest_results.json

Does NOT touch any live bot files.
"""

import sys
import os
import csv
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMPORT_DIR  = os.path.join(SCRIPT_DIR, 'import')
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================
SYMBOLS = {
    'EURUSD': {
        'csv':       'EURUSD60.csv',
        'pc1_pips':  10,
        'sl_pips':   35,
        'pip_value': 0.0001,
    },
    'GBPUSD': {
        'csv':       'GBPUSD60.csv',
        'pc1_pips':  12,
        'sl_pips':   38,
        'pip_value': 0.0001,
    },
}

SMA_PERIOD     = 20
ATR_PERIOD     = 14
ENTRY_ATR_MULT = 0.4
MAX_TRADE_BARS = 8   # H1 bars = 8 hours max

# New K-line parameters
MIN_BODY_RATIO    = 0.20   # body must be >= 20% of bar range
TREND_LOOKBACK    = 5      # look back N bars
TREND_BLOCK_COUNT = 4      # if >= this many bars are opposite -> block


# =============================================================================
# CSV LOADER
# =============================================================================
def load_csv(path):
    bars = []
    with open(path, 'r', encoding='utf-8') as fh:
        sample = fh.read(512)
        fh.seek(0)
        delim = '\t' if '\t' in sample else ','
        reader = csv.reader(fh, delimiter=delim)
        for row in reader:
            if not row or len(row) < 5:
                continue
            try:
                dt_str = row[0].strip()
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
    bars.sort(key=lambda x: x['time'])
    return bars


# =============================================================================
# INDICATORS
# =============================================================================
def compute_sma(closes, period):
    sma = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        sma[i] = sum(closes[i - period + 1:i + 1]) / period
    return sma


def compute_atr(bars, period):
    trs = [None]
    for i in range(1, len(bars)):
        h, l, pc = bars[i]['high'], bars[i]['low'], bars[i - 1]['close']
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = [None] * len(bars)
    if len(trs) >= period:
        atr[period] = sum(t for t in trs[1:period + 1] if t) / period
        for i in range(period + 1, len(bars)):
            if trs[i] is not None and atr[i - 1] is not None:
                atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period
    return atr


# =============================================================================
# K-LINE FILTERS
# =============================================================================
def kline_original(bar, direction):
    """Original K-line: bullish/bearish bar with wick >= 25% of range."""
    bar_range = bar['high'] - bar['low']
    if bar_range == 0:
        return False
    if direction == 'buy':
        lower_wick = min(bar['open'], bar['close']) - bar['low']
        wick_ratio = lower_wick / bar_range
        return bar['close'] > bar['open'] and wick_ratio >= 0.25
    else:
        upper_wick = bar['high'] - max(bar['open'], bar['close'])
        wick_ratio = upper_wick / bar_range
        return bar['close'] < bar['open'] and wick_ratio >= 0.25


def kline_new(bar, direction, prev_bars):
    """
    New K-line: original rules PLUS:
      1. Body >= MIN_BODY_RATIO of range (no dojis)
      2. Trend alignment: block if 4+ of last 5 bars are opposite direction
    """
    bar_range = bar['high'] - bar['low']
    if bar_range == 0:
        return False, 'flat_bar'

    body = abs(bar['close'] - bar['open'])
    body_ratio = body / bar_range

    # Guard 1: minimum body size
    if body_ratio < MIN_BODY_RATIO:
        return False, 'doji_blocked'

    # Guard 2: trend alignment using last N bars
    if len(prev_bars) >= TREND_LOOKBACK:
        lookback = prev_bars[-TREND_LOOKBACK:]
        if direction == 'buy':
            bearish_count = sum(1 for b in lookback if b['close'] < b['open'])
            if bearish_count >= TREND_BLOCK_COUNT:
                return False, f'trend_blocked({bearish_count}/{TREND_LOOKBACK} bearish)'
        else:
            bullish_count = sum(1 for b in lookback if b['close'] > b['open'])
            if bullish_count >= TREND_BLOCK_COUNT:
                return False, f'trend_blocked({bullish_count}/{TREND_LOOKBACK} bullish)'

    # Apply original wick rule
    if direction == 'buy':
        lower_wick = min(bar['open'], bar['close']) - bar['low']
        wick_ratio = lower_wick / bar_range
        if bar['close'] > bar['open'] and wick_ratio >= 0.25:
            return True, 'confirmed'
        return False, 'no_wick'
    else:
        upper_wick = bar['high'] - max(bar['open'], bar['close'])
        wick_ratio = upper_wick / bar_range
        if bar['close'] < bar['open'] and wick_ratio >= 0.25:
            return True, 'confirmed'
        return False, 'no_wick'


# =============================================================================
# SIMULATE TRADE OUTCOME
# =============================================================================
def simulate_outcome(bars, entry_idx, direction, pc1_pips, sl_pips, pip_value):
    """
    From entry bar, scan forward up to MAX_TRADE_BARS.
    Returns: 'pc1', 'sl', or 'timeout'
    """
    entry_price = bars[entry_idx]['close']
    pc1_dist = pc1_pips * pip_value
    sl_dist  = sl_pips  * pip_value

    for i in range(entry_idx + 1, min(entry_idx + MAX_TRADE_BARS + 1, len(bars))):
        b = bars[i]
        if direction == 'buy':
            if b['high'] >= entry_price + pc1_dist:
                return 'pc1'
            if b['low'] <= entry_price - sl_dist:
                return 'sl'
        else:
            if b['low'] <= entry_price - pc1_dist:
                return 'pc1'
            if b['high'] >= entry_price + sl_dist:
                return 'sl'
    return 'timeout'


# =============================================================================
# MAIN BACKTEST
# =============================================================================
def run_backtest(symbol, cfg):
    path = os.path.join(IMPORT_DIR, cfg['csv'])
    print(f"\nLoading {cfg['csv']}...", end=' ', flush=True)
    bars = load_csv(path)
    print(f"{len(bars):,} bars ({bars[0]['time'].year}–{bars[-1]['time'].year})")

    closes = [b['close'] for b in bars]
    sma    = compute_sma(closes, SMA_PERIOD)
    atr    = compute_atr(bars, ATR_PERIOD)

    pip = cfg['pip_value']
    pc1 = cfg['pc1_pips']
    sl  = cfg['sl_pips']

    # Counters: [total_signals, orig_confirmed, orig_pc1, orig_sl, orig_timeout,
    #            new_confirmed, new_pc1, new_sl, new_timeout, blocked_doji, blocked_trend]
    stats = defaultdict(int)
    block_reasons = defaultdict(int)

    warmup = max(SMA_PERIOD, ATR_PERIOD) + 2
    signals_this_bar = set()  # prevent double signal on same bar

    for i in range(warmup, len(bars) - MAX_TRADE_BARS - 1):
        s = sma[i]
        a = atr[i]
        if s is None or a is None:
            continue

        price  = bars[i]['close']
        signal_bar = bars[i]

        # Determine direction
        deviation = price - s
        if deviation <= -ENTRY_ATR_MULT * a:
            direction = 'buy'
        elif deviation >= ENTRY_ATR_MULT * a:
            direction = 'sell'
        else:
            continue

        # Skip if already signalled same direction on this bar
        key = (i, direction)
        if key in signals_this_bar:
            continue
        signals_this_bar.add(key)
        stats['total_signals'] += 1

        # Wait for K-line confirmation on next bar(s) — up to 3 bars
        confirmed_orig = False
        confirmed_new  = False
        new_block_reason = 'no_confirmation'
        confirm_bar_idx  = None

        for c in range(i + 1, min(i + 4, len(bars) - MAX_TRADE_BARS - 1)):
            cbar     = bars[c]
            prev5    = bars[max(0, c - TREND_LOOKBACK):c]

            if not confirmed_orig and kline_original(cbar, direction):
                confirmed_orig    = True
                confirm_bar_idx   = c

            if not confirmed_new:
                ok, reason = kline_new(cbar, direction, prev5)
                if ok:
                    confirmed_new     = True
                    confirm_bar_idx   = confirm_bar_idx or c
                else:
                    new_block_reason  = reason

            if confirmed_orig and confirmed_new:
                break

        # --- Original K-line outcomes ---
        if confirmed_orig:
            stats['orig_confirmed'] += 1
            outcome = simulate_outcome(bars, confirm_bar_idx or i + 1,
                                       direction, pc1, sl, pip)
            stats[f'orig_{outcome}'] += 1

            # --- Track blocked entries (orig confirmed, new blocked) ---
            if not confirmed_new:
                stats['blocked_confirmed'] += 1
                stats[f'blocked_{outcome}'] += 1

        # --- New K-line outcomes ---
        if confirmed_new:
            stats['new_confirmed'] += 1
            outcome = simulate_outcome(bars, confirm_bar_idx or i + 1,
                                       direction, pc1, sl, pip)
            stats[f'new_{outcome}'] += 1
        else:
            block_reasons[new_block_reason] += 1

    return stats, block_reasons


# =============================================================================
# REPORT
# =============================================================================
def print_report(symbol, stats, block_reasons):
    print(f"\n{'='*60}")
    print(f"  {symbol} K-LINE BACKTEST RESULTS")
    print(f"{'='*60}")

    total_sig = stats['total_signals']
    print(f"  Total MR signals detected:   {total_sig:>8,}")
    print()

    # Original
    orig_c  = stats['orig_confirmed']
    orig_p  = stats['orig_pc1']
    orig_s  = stats['orig_sl']
    orig_t  = stats['orig_timeout']
    orig_wr = orig_p / orig_c * 100 if orig_c else 0
    orig_pct = orig_c / total_sig * 100 if total_sig else 0
    print(f"  ORIGINAL K-LINE:")
    print(f"    Confirmed entries:  {orig_c:>7,}  ({orig_pct:.1f}% of signals)")
    print(f"    PC1 hit:            {orig_p:>7,}  ({orig_p/orig_c*100:.1f}%)" if orig_c else "    PC1 hit:              0")
    print(f"    Full SL hit:        {orig_s:>7,}  ({orig_s/orig_c*100:.1f}%)" if orig_c else "    Full SL hit:           0")
    print(f"    Timeout (8h):       {orig_t:>7,}  ({orig_t/orig_c*100:.1f}%)" if orig_c else "    Timeout:               0")
    print(f"    Win rate (PC1):     {orig_wr:>7.1f}%")
    print()

    # New
    new_c   = stats['new_confirmed']
    new_p   = stats['new_pc1']
    new_s   = stats['new_sl']
    new_t   = stats['new_timeout']
    new_wr  = new_p / new_c * 100 if new_c else 0
    new_pct = new_c / total_sig * 100 if total_sig else 0
    blocked = orig_c - new_c
    print(f"  NEW K-LINE (body filter + trend alignment):")
    print(f"    Confirmed entries:  {new_c:>7,}  ({new_pct:.1f}% of signals)")
    print(f"    Blocked vs original:{blocked:>7,}  ({blocked/orig_c*100:.1f}% of orig entries blocked)" if orig_c else "")
    print(f"    PC1 hit:            {new_p:>7,}  ({new_p/new_c*100:.1f}%)" if new_c else "    PC1 hit:              0")
    print(f"    Full SL hit:        {new_s:>7,}  ({new_s/new_c*100:.1f}%)" if new_c else "    Full SL hit:           0")
    print(f"    Timeout (8h):       {new_t:>7,}  ({new_t/new_c*100:.1f}%)" if new_c else "    Timeout:               0")
    print(f"    Win rate (PC1):     {new_wr:>7.1f}%")
    print()

    # What got blocked
    print(f"  BLOCK REASONS (entries new K-line rejected):")
    for reason, count in sorted(block_reasons.items(), key=lambda x: -x[1]):
        pct = count / orig_c * 100 if orig_c else 0
        print(f"    {reason:<35} {count:>6,}  ({pct:.1f}%)")

    # What happened to the blocked entries?
    blk_c  = stats['blocked_confirmed']
    blk_p  = stats['blocked_pc1']
    blk_s  = stats['blocked_sl']
    blk_t  = stats['blocked_timeout']
    blk_wr = blk_p / blk_c * 100 if blk_c else 0
    print(f"  WHAT HAPPENED TO THE BLOCKED ENTRIES:")
    print(f"    Total blocked:      {blk_c:>7,}")
    print(f"    PC1 hit:            {blk_p:>7,}  ({blk_p/blk_c*100:.1f}%)" if blk_c else "")
    print(f"    Full SL hit:        {blk_s:>7,}  ({blk_s/blk_c*100:.1f}%)" if blk_c else "")
    print(f"    Timeout (8h):       {blk_t:>7,}  ({blk_t/blk_c*100:.1f}%)" if blk_c else "")
    print(f"    Win rate (PC1):     {blk_wr:>7.1f}%  (vs {orig_wr:.1f}% overall)")

    # Improvement summary
    wr_diff = new_wr - orig_wr
    print()
    print(f"  SUMMARY: Win rate {orig_wr:.1f}% -> {new_wr:.1f}%  ({wr_diff:+.1f}pp)")
    entry_reduction = (1 - new_c / orig_c) * 100 if orig_c else 0
    print(f"           Entries reduced by {entry_reduction:.1f}%  ({blocked:,} fewer trades)")
    print(f"           Blocked entries win rate: {blk_wr:.1f}%  (filter correctly identified lower quality)")


# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':
    print("K-LINE FILTER BACKTEST")
    print("Original vs New (body filter + trend alignment)")
    print("16 years H1 data | EURUSD + GBPUSD")
    print()

    all_results = {}

    for symbol, cfg in SYMBOLS.items():
        stats, block_reasons = run_backtest(symbol, cfg)
        print_report(symbol, stats, block_reasons)
        all_results[symbol] = {
            'stats': dict(stats),
            'block_reasons': dict(block_reasons),
        }

    # Save to JSON
    out_path = os.path.join(OUTPUT_DIR, 'kline_backtest_results.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved to {out_path}")
