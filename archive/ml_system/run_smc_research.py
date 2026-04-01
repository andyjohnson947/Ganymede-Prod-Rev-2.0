#!/usr/bin/env python3
"""
Run SMC Research Analysis

Fetches OHLC data from MT5 and runs BOS/CHOCH pattern analysis.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'trading_bot'))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARNING] MetaTrader5 not available")

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[ERROR] pandas/numpy required")
    sys.exit(1)

from smc_research import SMCResearchAnalyzer, run_research


def fetch_mt5_ohlc(symbol: str, timeframe: int, bars: int = 5000) -> pd.DataFrame:
    """
    Fetch OHLC data from MT5.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1)
        bars: Number of bars to fetch

    Returns:
        DataFrame with OHLC data
    """
    if not MT5_AVAILABLE:
        print("MT5 not available")
        return None

    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return None

    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

        if rates is None or len(rates) == 0:
            print(f"No data returned for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLC
        df = df[['open', 'high', 'low', 'close']]

        return df

    finally:
        mt5.shutdown()


def save_ohlc_cache(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save OHLC data to cache file."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    cache_file = output_dir / f"{symbol}_{timeframe}_ohlc.csv"
    df.to_csv(cache_file)
    print(f"Cached OHLC data to {cache_file}")


def main():
    """Run SMC research on multiple symbols and timeframes."""
    print("=" * 80)
    print("SMC RESEARCH - BOS/CHOCH PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Symbols to analyze
    symbols = ['EURUSD', 'GBPUSD']

    # Timeframes (using H1 for meaningful structure)
    if MT5_AVAILABLE:
        timeframes = {
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
        }
    else:
        timeframes = {'H1': None}

    # Analysis parameters
    bars = 5000  # About 7 months of H1 data
    swing_length = 10  # Bars for swing detection (smaller = more events)
    lookahead = 20  # Bars to look ahead for outcome

    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"ANALYZING {symbol}")
        print(f"{'='*40}")

        for tf_name, tf_value in timeframes.items():
            print(f"\n[{tf_name}] Fetching {bars} bars...")

            if MT5_AVAILABLE and tf_value is not None:
                df = fetch_mt5_ohlc(symbol, tf_value, bars)
                if df is not None:
                    save_ohlc_cache(df, symbol, tf_name)
            else:
                # Try to load from cache
                cache_file = Path(__file__).parent / "outputs" / f"{symbol}_{tf_name}_ohlc.csv"
                if cache_file.exists():
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    df.columns = [c.lower() for c in df.columns]
                    print(f"Loaded {len(df)} bars from cache")
                else:
                    print(f"No data available for {symbol} {tf_name}")
                    continue

            if df is not None and len(df) >= 200:
                analyzer = SMCResearchAnalyzer()
                results = analyzer.analyze_ohlc_data(
                    df,
                    symbol=symbol,
                    swing_length=swing_length,
                    lookahead_bars=lookahead
                )

                all_results[f"{symbol}_{tf_name}"] = results
                analyzer.print_report(results)

    # Summary across all symbols
    print("\n" + "=" * 80)
    print("CROSS-SYMBOL SUMMARY")
    print("=" * 80)

    for key, results in all_results.items():
        stats = results.get('statistics', {}).get('summary', {})
        print(f"\n{key}:")
        print(f"  BOS Continuation: {stats.get('bos_continuation_pct', 0):.1f}%")
        print(f"  CHOCH Continuation: {stats.get('choch_continuation_pct', 0):.1f}%")

    # ML Integration recommendations
    print("\n" + "=" * 80)
    print("ML INTEGRATION RECOMMENDATIONS")
    print("=" * 80)

    print("""
Based on the analysis, here are potential ML features to track:

1. RECENT BOS DIRECTION
   - If recent BOS matches trade direction: +score
   - If recent BOS against trade direction: -score

2. RECENT CHOCH DIRECTION
   - CHOCH is a reversal signal
   - If CHOCH matches new trade direction: +score (reversal confirmed)
   - If CHOCH against new trade direction: -score (fighting the reversal)

3. SEQUENCE PATTERNS
   - Double BOS (same dir): Strong continuation signal
   - BOS then opposite CHOCH: Reversal warning
   - CHOCH then BOS (same dir): Reversal confirmed and continuing

4. SUGGESTED CONFLUENCE WEIGHTS (if data supports):
   - BOS aligned: +2 points
   - CHOCH aligned: +3 points
   - Double confirmation: +2 bonus
   - Against structure: -2 to -3 points

Review the percentage data above to validate these recommendations.
""")


if __name__ == '__main__':
    main()
