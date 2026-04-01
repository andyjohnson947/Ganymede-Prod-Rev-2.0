#!/usr/bin/env python3
"""
Historical Data Fetcher

Bulk downloads 4+ years of OHLCV data from MT5 using paginated calls
(1000-bar limit per API call). Saves to CSV for offline replay.

Usage:
    python data_fetcher.py                  # Fetch all symbols, all timeframes
    python data_fetcher.py --symbol EURUSD  # Fetch one symbol
    python data_fetcher.py --update         # Incremental update only
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import time

# Path setup (matches continuous_logger.py pattern)
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / 'trading_bot'))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from ml_system.market_state.config import (
    SYMBOLS, TIMEFRAMES, RAW_DIR,
    MT5_BARS_PER_CALL, YEARS_OF_DATA,
)


class HistoricalDataFetcher:
    """
    Bulk download historical data from MT5 and cache to CSV.

    MT5 has a 1000-bar limit per API call. For 4 years of H1 data:
      4 × 365 × 24 / 5 (weekday only) ≈ 35,000 bars = ~35 paginated calls.
    """

    def __init__(self):
        """Initialize (MT5 connection managed externally or auto-connected)."""
        self._connected = False

    def connect(self) -> bool:
        """Initialize MT5 connection."""
        if not MT5_AVAILABLE:
            print("[ERROR] MetaTrader5 package not available")
            return False

        if not mt5.initialize():
            print(f"[ERROR] Failed to initialize MT5: {mt5.last_error()}")
            return False

        self._connected = True
        account = mt5.account_info()
        print(f"[OK] MT5 connected | #{account.login} | {account.server}")
        return True

    def disconnect(self):
        """Shutdown MT5."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            print("[OK] MT5 disconnected")

    def fetch_full_history(
        self,
        symbol: str,
        timeframe: str,
        years: int = YEARS_OF_DATA
    ) -> Optional[pd.DataFrame]:
        """
        Fetch full history using MT5 copy_rates_range (single efficient call).

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe string ('H1', 'D1', 'W1')
            years: Years of data to fetch

        Returns:
            Complete DataFrame with all bars, or None.
        """
        if not self._connected:
            print("[ERROR] Not connected to MT5")
            return None

        tf_map = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }

        tf = tf_map.get(timeframe)
        if tf is None:
            print(f"[ERROR] Invalid timeframe: {timeframe}")
            return None

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365 * years)

        print(f"  Fetching {symbol} {timeframe} from {start_date.strftime('%Y-%m-%d')}...")

        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"  [ERROR] No data returned for {symbol} {timeframe}")
            return None

        # Convert to DataFrame
        result = pd.DataFrame(rates)
        result['time'] = pd.to_datetime(result['time'], unit='s')

        # Rename columns for consistency
        col_rename = {
            'tick_volume': 'volume', 'real_volume': 'real_volume', 'spread': 'spread',
        }
        result = result.rename(columns=col_rename)

        # Keep only the columns we need
        keep_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        if 'real_volume' in result.columns:
            keep_cols.append('real_volume')
        result = result[[c for c in keep_cols if c in result.columns]]

        # Deduplicate + sort
        result = result.drop_duplicates(subset='time', keep='last')
        result = result.sort_values('time').reset_index(drop=True)

        print(f"  [OK] {symbol} {timeframe}: {len(result):,} bars "
              f"({result['time'].iloc[0].strftime('%Y-%m-%d')} to "
              f"{result['time'].iloc[-1].strftime('%Y-%m-%d')})")

        return result

    def save_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        Save DataFrame to CSV in raw/ directory.

        Args:
            df: DataFrame to save
            symbol: Symbol name
            timeframe: Timeframe string

        Returns:
            Path to saved CSV file.
        """
        filepath = RAW_DIR / f"{symbol}_{timeframe}.csv"
        df.to_csv(filepath, index=False)
        print(f"  [SAVED] {filepath} ({len(df)} rows)")
        return str(filepath)

    def load_from_csv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load previously saved CSV.

        Returns:
            DataFrame with time as DatetimeIndex, or None if not found.
        """
        filepath = RAW_DIR / f"{symbol}_{timeframe}.csv"
        if not filepath.exists():
            return None

        df = pd.read_csv(filepath, parse_dates=['time'])
        print(f"  [LOADED] {filepath} ({len(df)} rows)")
        return df

    def load_for_replay(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load CSV and format for replay engine (DatetimeIndex + standard columns).

        Returns:
            DataFrame with time as index, columns: open, high, low, close, volume.
        """
        df = self.load_from_csv(symbol, timeframe)
        if df is None:
            return None

        # Set time as index
        df = df.set_index('time')
        df.index.name = 'time'

        return df

    def update_incremental(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load existing CSV, fetch new bars since last timestamp, append and resave.

        Returns:
            Complete DataFrame with new bars appended.
        """
        existing = self.load_from_csv(symbol, timeframe)

        if existing is not None and len(existing) > 0:
            last_time = existing['time'].iloc[-1]
            if isinstance(last_time, str):
                last_time = pd.Timestamp(last_time)

            print(f"  Updating {symbol} {timeframe} from {last_time}...")

            # Fetch bars after last known bar
            tf_map = {
                'H1': mt5.TIMEFRAME_H1, 'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1,
            }
            tf = tf_map.get(timeframe)
            if tf is None:
                return existing

            start = last_time.to_pydatetime() + timedelta(hours=1)
            rates = mt5.copy_rates_from(symbol, tf, start, MT5_BARS_PER_CALL)

            if rates is not None and len(rates) > 0:
                new_bars = pd.DataFrame(rates)
                new_bars['time'] = pd.to_datetime(new_bars['time'], unit='s')
                new_bars = new_bars.rename(columns={'tick_volume': 'volume'})

                keep_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                if 'real_volume' in new_bars.columns:
                    keep_cols.append('real_volume')
                new_bars = new_bars[[c for c in keep_cols if c in new_bars.columns]]

                result = pd.concat([existing, new_bars], ignore_index=True)
                result = result.drop_duplicates(subset='time', keep='last')
                result = result.sort_values('time').reset_index(drop=True)

                self.save_to_csv(result, symbol, timeframe)
                print(f"  [OK] Added {len(new_bars)} new bars to {symbol} {timeframe}")
                return result
            else:
                print(f"  [OK] {symbol} {timeframe} already up to date")
                return existing
        else:
            # No existing data, do a full fetch
            df = self.fetch_full_history(symbol, timeframe)
            if df is not None:
                self.save_to_csv(df, symbol, timeframe)
            return df

    def fetch_all_symbols(
        self,
        years: int = YEARS_OF_DATA,
        update_only: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch H1, D1, W1 for all symbols.

        Args:
            years: Years of data
            update_only: If True, only fetch new bars

        Returns:
            {symbol: {timeframe: DataFrame}}
        """
        result = {}

        for symbol in SYMBOLS:
            result[symbol] = {}
            print(f"\n{'='*60}")
            print(f"  {symbol}")
            print(f"{'='*60}")

            for tf_name in TIMEFRAMES.values():
                if update_only:
                    df = self.update_incremental(symbol, tf_name)
                else:
                    # Check if CSV already exists
                    existing = self.load_from_csv(symbol, tf_name)
                    if existing is not None and len(existing) > 10000 and tf_name == 'H1':
                        print(f"  [SKIP] {symbol} {tf_name} already has {len(existing)} bars")
                        df = existing
                    elif existing is not None and len(existing) > 500 and tf_name == 'D1':
                        print(f"  [SKIP] {symbol} {tf_name} already has {len(existing)} bars")
                        df = existing
                    else:
                        df = self.fetch_full_history(symbol, tf_name, years)
                        if df is not None:
                            self.save_to_csv(df, symbol, tf_name)

                if df is not None:
                    result[symbol][tf_name] = df

        return result


def main():
    """CLI entry point for data fetching."""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch historical data from MT5')
    parser.add_argument('--symbol', type=str, help='Fetch specific symbol only')
    parser.add_argument('--update', action='store_true', help='Incremental update only')
    parser.add_argument('--years', type=int, default=YEARS_OF_DATA, help='Years of data')
    args = parser.parse_args()

    fetcher = HistoricalDataFetcher()

    if not fetcher.connect():
        print("[FATAL] Could not connect to MT5")
        sys.exit(1)

    try:
        if args.symbol:
            symbols_to_fetch = [args.symbol.upper()]
        else:
            symbols_to_fetch = SYMBOLS

        total_bars = 0
        for symbol in symbols_to_fetch:
            print(f"\n{'='*60}")
            print(f"  {symbol}")
            print(f"{'='*60}")

            for tf_name in TIMEFRAMES.values():
                if args.update:
                    df = fetcher.update_incremental(symbol, tf_name)
                else:
                    df = fetcher.fetch_full_history(symbol, tf_name, args.years)
                    if df is not None:
                        fetcher.save_to_csv(df, symbol, tf_name)

                if df is not None:
                    total_bars += len(df)

        print(f"\n{'='*60}")
        print(f"  COMPLETE: {total_bars:,} total bars fetched/loaded")
        print(f"{'='*60}")

    finally:
        fetcher.disconnect()


if __name__ == '__main__':
    main()
