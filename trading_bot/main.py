#!/usr/bin/env python3
"""
Confluence Trading Bot - Main Entry Point
Based on EA analysis of 428 trades with 64.3% win rate

Usage:
    python main.py --login 12345 --password "yourpass" --server "Broker-Server"
    python main.py --gui  # Launch with GUI
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# CACHE CLEARING - Force Python to reload modules on every start
# This ensures code changes are always picked up without needing to restart Python interpreter
import importlib
if hasattr(importlib, 'invalidate_caches'):
    importlib.invalidate_caches()

# Clear __pycache__ for trading_bot modules to force fresh imports
def clear_pycache():
    """Clear Python cache files to ensure fresh module imports"""
    current_dir = Path(__file__).parent
    for pycache_dir in current_dir.rglob('__pycache__'):
        for cache_file in pycache_dir.glob('*.pyc'):
            try:
                cache_file.unlink()
            except Exception:
                pass

# Clear cache on startup
clear_pycache()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Disable Python bytecode generation (optional - prevents .pyc creation)
sys.dont_write_bytecode = True

# Force unbuffered stdout so trade signals print in real-time (not buffered until exit)
import os
os.environ['PYTHONUNBUFFERED'] = '1'

from core.mt5_manager import MT5Manager
from strategies.confluence_strategy import ConfluenceStrategy
from utils.logger import logger
from config.strategy_config import SYMBOLS

# ML System Integration
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root to path
from ml_system.ml_system_startup import start_ml_system, stop_ml_system
from ml_system.continuous_logger import ContinuousMLLogger
from ml_system.auto_tuner import get_live_tuner
import threading

# Global variables for continuous logger
_logger_instance = None  # Shared logger instance for manual logging if needed
_logger_thread = None
_logger_running = False
mt5_api_lock = threading.Lock()  # Global lock for thread-safe MT5 API access


def start_continuous_logger_with_backfill(login, password, server):
    """
    Initialize continuous logger and run backfill, then start background monitoring

    Threading fix: Uses shared lock (mt5_api_lock) to serialize MT5 API access.
    Backfill runs once in main thread, then logger monitors continuously in background.
    """
    global _logger_instance, _logger_thread, _logger_running

    try:
        # Threading fix: Reuse existing MT5 connection from main thread AND use shared lock
        # This prevents MT5 API conflicts when connecting in multiple threads
        _logger_instance = ContinuousMLLogger(use_existing_connection=True, api_lock=mt5_api_lock)

        if not _logger_instance.connect_mt5(login, password, server):
            logger.warning("Failed to connect continuous logger to MT5")
            return False

        # One-time: Import existing JSONL data into SQLite mirror
        if _logger_instance.trade_db:
            try:
                _logger_instance.trade_db.migrate_from_jsonl(str(_logger_instance.continuous_log))
            except Exception as e:
                logger.warning(f"[WARN] SQLite migration failed (non-critical): {e}")

        # Start continuous monitoring in background thread
        _logger_running = True

        def logger_worker():
            global _logger_running
            try:
                # CRITICAL: Wait 30 seconds before first check to let main thread fully initialize
                # This prevents deadlock during startup when both threads try to access MT5 simultaneously
                threading.Event().wait(30)
                while _logger_running:
                    try:
                        # Logger methods use internal _with_lock(), no outer lock needed
                        _logger_instance.check_for_new_trades()
                        _logger_instance.update_closed_trades()
                    except Exception as e:
                        if _logger_running:
                            logger.error(f"Logger check failed: {e}")
                    threading.Event().wait(60)  # Check every 60 seconds
            except Exception as e:
                logger.error(f"Logger thread crashed: {e}")

        _logger_thread = threading.Thread(target=logger_worker, daemon=True)
        _logger_thread.start()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize continuous logger: {e}")
        return False


def stop_continuous_logger():
    """Stop continuous ML logger background thread"""
    global _logger_running
    _logger_running = False
    logger.info("Stopping continuous logger...")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Confluence Trading Bot - Based on EA Reverse Engineering'
    )

    parser.add_argument(
        '--login',
        type=int,
        help='MT5 account login number'
    )

    parser.add_argument(
        '--password',
        type=str,
        help='MT5 account password'
    )

    parser.add_argument(
        '--server',
        type=str,
        help='MT5 server name'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Trading symbols (default: from config)'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch with GUI interface'
    )

    parser.add_argument(
        '--paper-trade',
        action='store_true',
        help='Paper trading mode (simulation only)'
    )

    parser.add_argument(
        '--test-mode',
        action='store_true',
        help=' TEST MODE: Trade all day, bypass time filters (for testing only)'
    )

    parser.add_argument(
        '--disable-ml',
        action='store_true',
        help='Disable ML system (continuous logger + auto-retraining) - trading only'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable verbose debug output (shows detailed signal detection, confluence scoring, etc.)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Print banner
    print()
    print("=" * 60)
    print("  GANYMEDE ROBOTIC TRADING SYSTEM")
    print("=" * 60)

    # Check if GUI mode
    if args.gui:
        launch_gui(args)
        return

    # Validate credentials
    if not all([args.login, args.password, args.server]):
        print("[ERROR] Error: MT5 credentials required")
        print("   Use: --login LOGIN --password PASSWORD --server SERVER")
        print("   Or use: --gui for graphical interface")
        sys.exit(1)

    # Get symbols
    symbols = args.symbols if args.symbols else SYMBOLS
    if not symbols:
        print("[ERROR] Error: No symbols specified")
        print("   Use: --symbols EURUSD GBPUSD")
        sys.exit(1)

    # Connect to MT5
    logger.info(f"Connecting to MT5 - Login: {args.login}, Server: {args.server}")

    mt5_manager = MT5Manager(
        login=args.login,
        password=args.password,
        server=args.server,
        api_lock=mt5_api_lock  # Pass lock for thread-safe API access
    )

    if not mt5_manager.connect():
        logger.error("Failed to connect to MT5")
        sys.exit(1)

    # Start ML System (optional - can be disabled with --disable-ml)
    ml_enabled = not args.disable_ml
    ml_logger_instance = None

    if ml_enabled:
        try:
            start_ml_system()

            try:
                if start_continuous_logger_with_backfill(args.login, args.password, args.server):
                    ml_logger_instance = _logger_instance
                else:
                    logger.warning("[WARN] ML logger failed to start")
            except Exception as e:
                logger.warning(f"[WARN] ML logger startup error: {e}")

            try:
                live_tuner = get_live_tuner()
                live_tuner.force_tune()
            except Exception as e:
                logger.warning(f"[WARN] Auto-Tuner startup error: {e}")

            print("[OK] ML System ready (logger + auto-tuner + retraining)")

        except Exception as e:
            logger.warning(f"[WARN] ML System startup failed: {e}")
            ml_enabled = False
    else:
        print("[INFO] ML System disabled")

    # Verify MT5 connection is still valid
    test_account = mt5_manager.get_account_info()
    if not test_account:
        logger.error("MT5 connection lost - reconnecting...")
        mt5_manager.disconnect()
        if not mt5_manager.connect():
            logger.error("Failed to reconnect to MT5")
            sys.exit(1)

    try:
        strategy = ConfluenceStrategy(mt5_manager, test_mode=args.test_mode, ml_logger=ml_logger_instance, debug=args.debug)

        # Market snapshot at startup
        try:
            import MetaTrader5 as mt5
            import pandas as pd
            from indicators.adx import calculate_adx

            print()
            print("=" * 60)
            print("  MARKET SNAPSHOT")
            print("=" * 60)

            for sym in symbols:
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 50)
                if rates is not None and len(rates) >= 20:
                    df = pd.DataFrame(rates)
                    df_adx = calculate_adx(df.copy(), period=14)
                    latest = df_adx.iloc[-1]
                    adx = latest['adx']
                    plus_di = latest['plus_di']
                    minus_di = latest['minus_di']

                    direction = "+DI dominant" if plus_di > minus_di else "-DI dominant"

                    tick = mt5.symbol_info_tick(sym)
                    price = tick.bid if tick else latest['close']

                    print(f"  {sym}: {price:.5f} | ADX={adx:.0f} {direction}")
                else:
                    print(f"  {sym}: No data available")

            print("=" * 60)
            print()
        except Exception as e:
            print(f"  [WARN] Market snapshot unavailable: {e}")
            print()

        # Show test mode warning if enabled
        if args.test_mode:
            print("\n" + "=" * 80)
            print("[WARN]  TEST MODE ENABLED - TRADING ALL DAY (NO TIME FILTERS)")
            print("=" * 80)
            print()

        strategy.start(symbols)

    except KeyboardInterrupt:
        print("\n\n[WARN]  Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if ml_enabled:
            logger.info("Stopping ML System...")
            try:
                stop_ml_system()
                stop_continuous_logger()
            except Exception as e:
                logger.warning(f"Error stopping ML system: {e}")
        mt5_manager.disconnect()
        logger.info("Bot stopped")


def launch_gui(args):
    """
    Launch GUI interface

    Args:
        args: Command line arguments
    """
    try:
        # Import GUI here to avoid dependency if not using GUI
        from gui.trading_gui import TradingGUI
        import tkinter as tk

        root = tk.Tk()
        app = TradingGUI(root)
        root.mainloop()

    except ImportError as e:
        print(f"[ERROR] GUI dependencies not available: {e}")
        print("   Install required packages:")
        print("   pip install tkinter matplotlib")
        sys.exit(1)


if __name__ == "__main__":
    main()
