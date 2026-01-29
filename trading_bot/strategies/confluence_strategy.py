"""
Main Confluence Strategy
Orchestrates signal detection, position management, and recovery
"""

import sys
from pathlib import Path

# Add project root to path for ml_system imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import logging

from core.mt5_manager import MT5Manager
from strategies.signal_detector import SignalDetector
from strategies.recovery_manager import RecoveryManager
from strategies.time_filters import TimeFilter
from strategies.breakout_strategy import BreakoutStrategy
from strategies.partial_close_manager import PartialCloseManager
from utils.risk_calculator import RiskCalculator
from utils.config_reloader import reload_config, print_current_config
from utils.timezone_manager import get_current_time
from portfolio.portfolio_manager import PortfolioManager
from ml_system.ml_integration_manager import MLIntegrationManager
from config.strategy_config import (
    SYMBOLS,
    TIMEFRAME,
    HTF_TIMEFRAMES,
    DATA_REFRESH_INTERVAL,
    MAX_OPEN_POSITIONS,
    MAX_POSITIONS_PER_SYMBOL,
    PROFIT_TARGET_PERCENT,
    MAX_POSITION_HOURS,
    PARTIAL_CLOSE_ENABLED,
    PARTIAL_CLOSE_RECOVERY,
    BREAKOUT_ENABLED,
    BREAKOUT_LOT_SIZE_MULTIPLIER,
    ENABLE_CASCADE_PROTECTION,  # Cascade stop protection
    TREND_BLOCK_MINUTES,        # Trade block duration after cascade
    MIN_CONFLUENCE_SCORE,       # Startup diagnostics
    DCA_ENABLED,                # Startup diagnostics
    HEDGE_ENABLED,              # Startup diagnostics
    ENABLE_TIME_FILTERS,        # Startup diagnostics
)

# Module-level logger
logger = logging.getLogger('ConfluenceStrategy')


class ConfluenceStrategy:
    """Main trading strategy implementation"""

    def __init__(self, mt5_manager: MT5Manager, test_mode: bool = False, ml_logger=None, debug: bool = False):
        """
        Initialize strategy

        Args:
            mt5_manager: MT5Manager instance (already connected)
            test_mode: If True, bypass all time filters for testing
            ml_logger: ContinuousMLLogger instance for trailing stop event logging
            debug: If True, enable verbose debug output
        """
        self.debug = debug
        if self.debug:
            print("[DEBUG] ConfluenceStrategy.__init__() starting...", flush=True)
        self.mt5 = mt5_manager
        self.test_mode = test_mode
        self.ml_logger = ml_logger  # ML logger for real-time event tracking

        # Initialize ML Integration Manager for enhanced data collection
        if self.debug:
            print("[DEBUG] Creating MLIntegrationManager...", flush=True)
        self.ml_manager = MLIntegrationManager(enable_adaptive_weighting=True)

        if self.debug:
            print("[DEBUG] Creating SignalDetector...", flush=True)
        self.signal_detector = SignalDetector(ml_logger=self.ml_logger, debug=self.debug)
        if self.debug:
            print("[DEBUG] Creating RecoveryManager...", flush=True)
        self.recovery_manager = RecoveryManager(mt5_manager=self.mt5, ml_logger=self.ml_logger)
        if self.debug:
            print("[DEBUG] Creating RiskCalculator...", flush=True)
        self.risk_calculator = RiskCalculator()
        if self.debug:
            print("[DEBUG] Creating PortfolioManager...", flush=True)
        self.portfolio_manager = PortfolioManager()

        # New strategy modules
        self.time_filter = TimeFilter()
        self.breakout_strategy = BreakoutStrategy() if BREAKOUT_ENABLED else None
        self.partial_close_manager = PartialCloseManager() if PARTIAL_CLOSE_ENABLED else None

        self.running = False
        self.last_data_refresh = {}
        self.market_data_cache = {}

        # Market state tracking for trend-based trade blocking
        self.market_trending_block = {}

        # Cascade protection - blocks new trades after cascade close
        self.cascade_blocks = {}  # {symbol: block_until_time}

        # Crash recovery tracking
        self.recovery_stacks_reconstructed = False

        # Periodic reconciliation tracking
        self.last_reconcile_time = None

        # Statistics
        self.stats = {
            'signals_detected': 0,
            'trades_opened': 0,
            'trades_closed': 0,
            'grid_levels_added': 0,
            'hedges_activated': 0,
            'dca_levels_added': 0,
        }
        if self.debug:
            print("[DEBUG] ConfluenceStrategy.__init__() completed successfully", flush=True)

    def start(self, symbols: List[str]):
        """
        Start the trading strategy

        Args:
            symbols: List of symbols to trade
        """
        import sys
        if self.debug:
            print("[DEBUG] Entered strategy.start() method", flush=True)
            sys.stdout.flush()

        print("=" * 80)
        print(" CONFLUENCE STRATEGY STARTING")
        print("=" * 80)
        print()

        # Get account info (with retry for MT5 threading issues)
        if self.debug:
            print("[DEBUG] Getting account info...", flush=True)
        account_info = None
        for retry in range(3):
            if self.debug:
                print(f"[DEBUG] Attempt {retry+1}/3 to get account info...", flush=True)
            account_info = self.mt5.get_account_info()
            if account_info:
                if self.debug:
                    print(f"[DEBUG] Got account info successfully!", flush=True)
                break
            print(f"[WARN] MT5 API busy, retrying ({retry+1}/3)...")
            time.sleep(0.5)  # Let background thread release MT5 API

        if not account_info:
            print("[ERROR] Failed to get account info after retries")
            return

        if self.debug:
            print(f"[DEBUG] Printing account details...", flush=True)
        print(f"Account Balance: ${account_info['balance']:.2f}", flush=True)
        print(f"Symbols: {', '.join(symbols)}", flush=True)
        print(f"Timeframe: {TIMEFRAME}", flush=True)
        print(f"HTF: {', '.join(HTF_TIMEFRAMES)}", flush=True)
        print(flush=True)

        # Set initial balance for drawdown tracking
        if self.debug:
            print("[DEBUG] Setting initial balance...", flush=True)
        self.risk_calculator.set_initial_balance(account_info['balance'])

        # CRASH RECOVERY: Load saved state and reconcile with MT5 reality
        print("[SYNC] Initializing crash recovery system...", flush=True)
        state_loaded = self.recovery_manager.load_state()

        # ALWAYS reconcile with MT5, even if state loaded (critical for crash recovery)
        print("[SYNC] Reconciling tracked positions with MT5...", flush=True)
        added, removed, validated = self.recovery_manager.reconcile_on_startup(self.mt5)

        if state_loaded:
            print(f"[OK] State loaded and reconciled:")
            print(f"   [OK] Validated: {validated} positions still open")
            if removed > 0:
                print(f"   [DEL]  Removed: {removed} closed positions")
            if added > 0:
                print(f"   [+] Added: {added} new positions from MT5")
        else:
            if added > 0:
                print(f"[OK] No saved state - discovered {added} MT5 positions")
            else:
                print("[OK] No saved state - starting fresh")

        print()

        # LOAD BLOCKING STATE: Restore cascade/trending blocks from previous session
        print("[LOAD] Loading blocking state...")
        blocks_loaded = self.load_blocking_state()
        if not blocks_loaded:
            print("[INFO] No previous blocks - starting with clean state")

        # MARKET STATE EVALUATION: Check market conditions on startup
        print("[INFO] Evaluating market state for all symbols...")
        for symbol in symbols:
            # Fetch initial H1 data for market state evaluation
            h1_data = self.mt5.get_historical_data(symbol, TIMEFRAME, bars=500)
            if h1_data is not None:
                # Evaluate market state (ADX analysis)
                market_state = self.recovery_manager.check_market_state_for_hedge_close(
                    symbol=symbol,
                    current_data=h1_data
                )

                # Initialize market_trending_block with current market state
                should_block = market_state.get('should_block_new_trades', False)
                self.market_trending_block[symbol] = should_block

                if should_block:
                    print(f"   [WARN]  {symbol}: Trading BLOCKED - {market_state.get('reason', 'Unknown')}")
                else:
                    print(f"   [OK] {symbol}: Trading ALLOWED - {market_state.get('reason', 'Unknown')}")
            else:
                print(f"   [WARN]  {symbol}: Could not fetch market data")
                self.market_trending_block[symbol] = False  # Allow trading if data unavailable

        # Save blocking state after startup evaluation
        self.save_blocking_state()

        # STARTUP DIAGNOSTICS: Show comprehensive trading status
        print()
        print("=" * 80)
        print(" TRADING STATUS DIAGNOSTIC")
        print("=" * 80)
        print()

        # Check position limits
        all_positions = self.mt5.get_positions()
        print(f"[INFO] Positions: {len(all_positions)}/{MAX_OPEN_POSITIONS} (max allowed)")
        for symbol in symbols:
            symbol_positions = [p for p in all_positions if p['symbol'] == symbol]
            print(f"   {symbol}: {len(symbol_positions)}/{MAX_POSITIONS_PER_SYMBOL}")

        print()

        # Show blocking status for each symbol
        print("[STATUS] Symbol Trading Status:")
        for symbol in symbols:
            status_parts = []

            # Check cascade block
            if symbol in self.cascade_blocks and self.cascade_blocks[symbol]:
                time_left = (self.cascade_blocks[symbol] - get_current_time()).total_seconds() / 60
                if time_left > 0:
                    status_parts.append(f"CASCADE BLOCK ({time_left:.0f}min left)")

            # Check market trending block
            if self.market_trending_block.get(symbol, False):
                status_parts.append("TRENDING BLOCK (ADX > 40)")

            # Check position limit
            symbol_positions = [p for p in all_positions if p['symbol'] == symbol]
            if len(symbol_positions) >= MAX_POSITIONS_PER_SYMBOL:
                status_parts.append("POSITION LIMIT REACHED")

            # Display status
            if status_parts:
                print(f"   [BLOCKED] {symbol}: BLOCKED - {', '.join(status_parts)}")
            else:
                print(f"   [OK] {symbol}: READY TO TRADE")

        print()

        # Show trading configuration
        print("[CONFIG]  Configuration:")
        print(f"   Confluence Score: {MIN_CONFLUENCE_SCORE}+ required")
        print(f"   DCA: {'Enabled' if DCA_ENABLED else 'Disabled'}")
        print(f"   Hedge: {'Enabled' if HEDGE_ENABLED else 'Disabled'}")
        print(f"   Cascade Protection: {'Enabled' if ENABLE_CASCADE_PROTECTION else 'Disabled'}")
        print(f"   Time Filters: {'Disabled (trade 24/7)' if not ENABLE_TIME_FILTERS else 'Enabled'}")

        print()

        # ML System status
        print("[ML] ML Integration:")
        print(f"   Enhanced Data Collection: Active")
        print(f"   Adaptive Confluence: {'Active' if self.ml_manager.confluence_analyzer else 'Pending data'}")
        print(f"   Logging: Trade entries, recovery decisions, market conditions")
        print(f"   Output: ml_system/outputs/")
        print(f"   Note: You maintain full control - ML observes and recommends")

        # Show ML insights from recent data
        try:
            from ml_system.ml_insights_reporter import MLInsightsReporter
            reporter = MLInsightsReporter()
            insights_report = reporter.format_startup_report()
            print(insights_report)
        except Exception as e:
            print(f"\n[WARN]  ML insights unavailable: {e}")
            print("   (Will be available after first trades are logged)\n")
            print("=" * 80)
            print()

        self.running = True
        loop_iteration = 0

        print("[START] STARTING MAIN LOOP", flush=True)
        print(f"Scanning for confluence signals every 60 seconds...", flush=True)
        print()

        try:
            while self.running:
                # Main trading loop
                if self.debug:
                    print(f"\n[{get_current_time().strftime('%Y-%m-%d %H:%M:%S')}] [LOOP #{loop_iteration + 1}] Starting iteration...", flush=True)
                self._trading_loop(symbols)
                if self.debug:
                    print(f"[LOOP #{loop_iteration + 1}] Iteration complete. Sleeping 60s...", flush=True)

                # Periodic state backup (every 10 minutes)
                loop_iteration += 1
                if loop_iteration % 10 == 0:
                    self.recovery_manager.save_state()

                # Sleep before next iteration
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            print("\n[WARN] Strategy stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Strategy error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop the strategy"""
        self.running = False

        # Save final state before shutdown
        print("\n Saving final state...")
        if self.recovery_manager.save_state():
            print("[OK] State saved successfully")
        else:
            print("[WARN]  Failed to save state")

        print()
        print("=" * 80)
        print(" STRATEGY STATISTICS")
        print("=" * 80)
        for key, value in self.stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print()

    def _trading_loop(self, symbols: List[str]):
        """
        Main trading loop iteration

        Args:
            symbols: Symbols to trade
        """
        for symbol in symbols:
            try:
                # 1. Check if we should refresh market data
                if self.debug:
                    print(f"[DEBUG] Refreshing market data for {symbol}...", flush=True)
                self._refresh_market_data(symbol)
                if self.debug:
                    print(f"[DEBUG] Market data refreshed for {symbol}", flush=True)

                # 2. Manage existing positions
                if self.debug:
                    print(f"[DEBUG] Managing positions for {symbol}...", flush=True)
                self._manage_positions(symbol)
                if self.debug:
                    print(f"[DEBUG] Positions managed for {symbol}", flush=True)

                # 3. Look for new signals
                if self._can_open_new_position(symbol):
                    if self.debug:
                        print(f"[DEBUG] Checking for signals on {symbol}...", flush=True)
                    self._check_for_signals(symbol)
                    if self.debug:
                        print(f"[DEBUG] Signal check complete for {symbol}", flush=True)

            except Exception as e:
                print(f"[ERROR] Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue

    def _refresh_market_data(self, symbol: str):
        """Refresh market data for symbol if needed"""
        now = get_current_time()
        last_refresh = self.last_data_refresh.get(symbol)

        # Check if refresh needed
        if last_refresh:
            minutes_since = (now - last_refresh).total_seconds() / 60
            if minutes_since < DATA_REFRESH_INTERVAL:
                return  # Data still fresh

        # Fetch H1 data
        h1_data = self.mt5.get_historical_data(symbol, TIMEFRAME, bars=500)
        if h1_data is None:
            return

        # Calculate VWAP on H1 data
        h1_data = self.signal_detector.vwap.calculate(h1_data)

        # Calculate ATR for breakout detection
        if 'atr' not in h1_data.columns:
            # Simple ATR calculation (14 period)
            high_low = h1_data['high'] - h1_data['low']
            high_close = abs(h1_data['high'] - h1_data['close'].shift())
            low_close = abs(h1_data['low'] - h1_data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            h1_data['atr'] = true_range.rolling(window=14).mean()

        # Fetch M15 data for faster trend detection (4x faster than H1)
        m15_data = self.mt5.get_historical_data(symbol, 'M15', bars=100)
        if m15_data is not None:
            # Calculate M15 ATR for candle size analysis
            if 'atr' not in m15_data.columns:
                high_low = m15_data['high'] - m15_data['low']
                high_close = abs(m15_data['high'] - m15_data['close'].shift())
                low_close = abs(m15_data['low'] - m15_data['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                m15_data['atr'] = true_range.rolling(window=14).mean()

        # Fetch HTF data
        d1_data = self.mt5.get_historical_data(symbol, 'D1', bars=100)
        w1_data = self.mt5.get_historical_data(symbol, 'W1', bars=50)

        if d1_data is None or w1_data is None:
            return

        # Cache the data
        self.market_data_cache[symbol] = {
            'h1': h1_data,
            'm15': m15_data if m15_data is not None else None,
            'd1': d1_data,
            'w1': w1_data,
            'last_update': now
        }

        self.last_data_refresh[symbol] = now

        # RE-EVALUATE MARKET STATE: Check if market conditions have changed
        # This updates market_trending_block every DATA_REFRESH_INTERVAL (60 min)
        market_state = self.recovery_manager.check_market_state_for_hedge_close(
            symbol=symbol,
            current_data=h1_data
        )

        # Update market trending block based on current conditions
        was_blocked = self.market_trending_block.get(symbol, False)
        should_block = market_state.get('should_block_new_trades', False)
        self.market_trending_block[symbol] = should_block

        # Log state changes
        if was_blocked and not should_block:
            print(f"\n[OK] {symbol}: Market state IMPROVED - Trading RESUMED")
            print(f"   {market_state.get('reason', 'Unknown')}")
            self.save_blocking_state()  # Save state change
        elif not was_blocked and should_block:
            print(f"\n[WARN]  {symbol}: Market state DEGRADED - Trading BLOCKED")
            print(f"   {market_state.get('reason', 'Unknown')}")
            self.save_blocking_state()  # Save state change

    def _manage_positions(self, symbol: str):
        """Manage existing positions for symbol"""
        positions = self.mt5.get_positions(symbol)

        # CRASH RECOVERY: Reconstruct recovery stacks on first run (do this once across all symbols)
        if not self.recovery_stacks_reconstructed:
            # Get ALL positions (not just this symbol)
            all_positions = self.mt5.get_positions()

            # First, track all original positions
            for pos in all_positions:
                pos_ticket = pos['ticket']
                pos_comment = pos.get('comment', '')

                # Check if it's a recovery order
                is_recovery = any([
                    'Grid' in pos_comment,
                    'Hedge' in pos_comment,
                    'DCA' in pos_comment,
                ])

                # Track original positions only
                if not is_recovery and pos_ticket not in self.recovery_manager.tracked_positions:
                    self.recovery_manager.track_position(
                        ticket=pos_ticket,
                        symbol=pos['symbol'],
                        entry_price=pos['price_open'],
                        position_type='buy' if pos['type'] == 0 else 'sell',
                        volume=pos['volume']
                    )

            # Now reconstruct recovery stacks
            if len(all_positions) > 0:
                self.recovery_manager.reconstruct_recovery_stacks(all_positions)

            # Mark as done (only do this once)
            self.recovery_stacks_reconstructed = True

        # CONTINUOUS ORPHAN DETECTION: Check for newly orphaned recovery orders
        # This handles the case where a master position closes during trading and leaves grids/hedges/DCA behind
        # We need to detect and adopt these orphans so they get recovery protection
        # Use silent=True to avoid spamming logs on every iteration
        all_positions = self.mt5.get_positions()
        if len(all_positions) > 0:
            self.recovery_manager.reconstruct_recovery_stacks(all_positions, silent=True)

        # PERIODIC RECONCILIATION: Check tracked vs MT5 positions every 30 minutes
        # Detects and auto-corrects state drift between bot and MT5
        current_time = get_current_time()
        if self.last_reconcile_time is None or (current_time - self.last_reconcile_time).total_seconds() >= 1800:
            if len(all_positions) > 0:
                reconcile_stats = self.recovery_manager.reconcile_with_mt5(all_positions, silent=True)
                if reconcile_stats['discrepancies_found'] > 0:
                    logger.warning(f"[RECONCILE] Found {reconcile_stats['discrepancies_found']} discrepancies, auto-corrected {reconcile_stats['auto_corrected']}")
            # CRITICAL: Update timer OUTSIDE the position check to prevent running every iteration
            self.last_reconcile_time = current_time

        # CASCADE PROTECTION: Check total unrealized loss across ALL open positions
        # Prevents trending markets from wiping out account before per-stack stops trigger
        if len(all_positions) > 0:
            total_unrealized = sum(pos['profit'] for pos in all_positions)

            # Emergency cascade threshold: -$100 total unrealized loss
            if total_unrealized <= -100:
                print(f"\n[ALERT] CASCADE PROTECTION TRIGGERED [ALERT]")
                print(f"   Total unrealized loss: ${total_unrealized:.2f}")
                print(f"   Threshold: -$100.00")
                print(f"   Open positions: {len(all_positions)}")
                print(f"   ACTION: Closing ALL positions immediately")
                print(f"   REASON: Multiple positions in drawdown - trending market detected")
                print()

                logger.critical(f"CASCADE PROTECTION: Total unrealized ${total_unrealized:.2f} exceeds -$100 threshold")
                logger.critical(f"   Closing {len(all_positions)} positions to prevent further losses")

                # Close all positions
                closed_count = 0
                for pos in all_positions:
                    if self.mt5.close_position(pos['ticket']):
                        closed_count += 1
                        self.recovery_manager.untrack_position(pos['ticket'])
                        self.stats['trades_closed'] += 1

                print(f"[CASCADE] Closed {closed_count}/{len(all_positions)} positions")
                print(f"[CASCADE] Blocking new trades for 60 minutes")
                print()

                # Block trading for 60 minutes to avoid re-entering during strong trend
                from datetime import timedelta
                self.trade_block_until = get_current_time() + timedelta(minutes=60)

                # Save recovery state
                self.recovery_manager.save_state()

                # Exit early - no position management needed
                return

        # Check for trading window closures - close negative positions if window is ending
        close_actions = self.portfolio_manager.check_window_closures()
        for action in close_actions:
            if action.symbol == symbol and action.close_negatives_only:
                # Close all negative positions for this symbol
                for pos in positions:
                    if pos['profit'] < 0:
                        ticket = pos['ticket']
                        print(f"[CLOSE] Closing negative position {ticket} - {action.reason}")
                        if self.mt5.close_position(ticket):
                            self.recovery_manager.untrack_position(ticket)
                            self.stats['trades_closed'] += 1

        for position in positions:
            ticket = position['ticket']
            comment = position.get('comment', '')

            # [WARN] CRITICAL FIX: Don't track recovery orders as new positions
            # Recovery orders have comments like "Grid L1 - 1001", "Hedge - 1001", "DCA L1 - 1001"
            # Only the ORIGINAL trade should spawn recovery, not recovery orders themselves
            is_recovery_order = any([
                'Grid' in comment,
                'Hedge' in comment,
                'DCA' in comment,
            ])

            # Check if position is being tracked
            if ticket not in self.recovery_manager.tracked_positions:
                # Only track original trades, NOT recovery orders
                if not is_recovery_order:
                    # Start tracking
                    self.recovery_manager.track_position(
                        ticket=ticket,
                        symbol=position['symbol'],
                        entry_price=position['price_open'],
                        position_type=position['type'],
                        volume=position['volume']
                    )

            # Get symbol info (needed for various checks)
            current_price = position['price_current']
            symbol_info = self.mt5.get_symbol_info(symbol)
            pip_value = symbol_info.get('point', 0.0001) if symbol_info else 0.0001

            # PC1/PC2/TRAILING STOP: ONLY for profitable ORIGINAL positions
            # NOT for recovery orders (grid/DCA/hedge) or positions in active recovery
            # This is the new exit strategy for positive trades moving toward TP

            # Check if this is a recovery order (grid/DCA/hedge)
            is_recovery_order = any([
                'Grid' in comment,
                'Hedge' in comment,
                'DCA' in comment,
            ])

            # Check if position is in active recovery (has underwater recovery stack)
            has_active_recovery = False
            if ticket in self.recovery_manager.tracked_positions:
                tracked_pos = self.recovery_manager.tracked_positions[ticket]
                has_active_recovery = tracked_pos.get('recovery_active', False)

            # ONLY apply to positive original positions without active recovery
            if position['profit'] > 0 and not is_recovery_order and not has_active_recovery:
                # Get instrument-specific PC1/PC2 levels from instruments_config
                from trading_bot.portfolio.instruments_config import INSTRUMENTS
                instrument_config = INSTRUMENTS.get(symbol, {})
                tp_settings = instrument_config.get('take_profit', {})

                pc1_pips = tp_settings.get('partial_1_pips', 10)
                pc2_pips = tp_settings.get('partial_2_pips', 20)
                pc1_percent = tp_settings.get('partial_1_percent', 0.25)
                pc2_percent = tp_settings.get('partial_2_percent', 0.25)

                # Calculate current profit in pips
                entry_price = position['price_open']
                current_price = position['price_current']
                pos_type = 'buy' if position['type'] == 0 else 'sell'

                if pos_type == 'buy':
                    profit_pips = (current_price - entry_price) / pip_value
                else:
                    profit_pips = (entry_price - current_price) / pip_value

                # Get tracked position state
                if ticket not in self.recovery_manager.tracked_positions:
                    # Start tracking if not already
                    self.recovery_manager.track_position(
                        ticket=ticket,
                        symbol=symbol,
                        entry_price=entry_price,
                        position_type=pos_type,
                        volume=position['volume']
                    )

                tracked_pos = self.recovery_manager.tracked_positions[ticket]
                pc1_closed = tracked_pos.get('partial_1_closed', False)
                pc2_closed = tracked_pos.get('partial_2_closed', False)

                # PC1 CHECK: Close 25% at 10 pips (EURUSD) or 12 pips (GBPUSD)
                if not pc1_closed and profit_pips >= pc1_pips:
                    close_volume = round(position['volume'] * pc1_percent, 2)

                    if close_volume > 0 and close_volume < position['volume']:
                        short_ticket = str(ticket)[-5:]
                        pc1_comment = f"PC1-25%@{profit_pips:.0f}pips-{short_ticket}"

                        if self.mt5.close_partial_position(ticket, close_volume, comment=pc1_comment):
                            print(f"[PC1] {ticket} - Closed 25% @ +{profit_pips:.1f} pips = ${close_volume * profit_pips * 10:.2f}")
                            tracked_pos['partial_1_closed'] = True

                            # DISABLE VWAP EXITS after PC1
                            print(f"[PC1] VWAP exits disabled for {ticket}")

                # PC2 CHECK: Close another 25% at 20 pips (EURUSD) or 25 pips (GBPUSD)
                elif pc1_closed and not pc2_closed and profit_pips >= pc2_pips:
                    # FIXED: Use INITIAL volume for PC2, not current volume (which is reduced after PC1)
                    # PC2 should close 25% of ORIGINAL position, leaving 50% running total
                    initial_volume = tracked_pos.get('initial_volume', position['volume'])
                    close_volume = round(initial_volume * pc2_percent, 2)

                    if close_volume > 0 and close_volume < position['volume']:
                        short_ticket = str(ticket)[-5:]
                        pc2_comment = f"PC2-25%@{profit_pips:.0f}pips-{short_ticket}"

                        if self.mt5.close_partial_position(ticket, close_volume, comment=pc2_comment):
                            print(f"[PC2] {ticket} - Closed 25% (50% total) @ +{profit_pips:.1f} pips = ${close_volume * profit_pips * 10:.2f}")
                            tracked_pos['partial_2_closed'] = True

                            # ACTIVATE TRAILING STOP
                            if tp_settings.get('trailing_stop_enabled') and not tracked_pos.get('trailing_stop_active'):
                                self.recovery_manager.activate_trailing_stop(ticket, current_price, tp_settings)
                                print(f"[PC2] Trailing stop activated for {ticket}")

                                # Set PC2 trigger time for 60-min limit
                                tracked_pos['pc2_trigger_time'] = get_current_time()

                                # ML LOGGING: Log PC2 trigger
                                if self.ml_logger:
                                    trailing_distance = tracked_pos.get('trailing_stop_distance_pips', 0)
                                    trailing_stop_price = tracked_pos.get('trailing_stop_price', 0)
                                    self.ml_logger.log_pc2_trigger(
                                        ticket=ticket,
                                        symbol=symbol,
                                        current_price=current_price,
                                        entry_price=entry_price,
                                        trailing_distance_pips=trailing_distance,
                                        trailing_stop_price=trailing_stop_price
                                    )

                            # MOVE HARDWARE SL TO BREAKEVEN
                            if self.mt5.modify_position(ticket, sl=entry_price):
                                print(f"[PC2] Hardware SL -> breakeven @ {entry_price:.5f}")

                                # ML LOGGING: Log SL to BE
                                if self.ml_logger:
                                    self.ml_logger.log_sl_to_breakeven(
                                        ticket=ticket,
                                        symbol=symbol,
                                        breakeven_price=entry_price
                                    )

            # TRAILING STOP SYSTEM: ONLY for positive original positions with trailing active
            # Recovery system manages underwater positions separately with grid/DCA/hedge
            if ticket in self.recovery_manager.tracked_positions:
                tracked_pos = self.recovery_manager.tracked_positions[ticket]

                # ONLY process trailing if: position is profitable AND no active recovery
                has_active_recovery = tracked_pos.get('recovery_active', False)
                is_trailing_active = tracked_pos.get('trailing_stop_active', False)
                is_positive = position['profit'] > 0

                # Skip trailing stop logic if position is in recovery mode (underwater)
                if has_active_recovery or not is_positive or not is_trailing_active:
                    # Position is either:
                    # - In active recovery (grid/DCA/hedge managing it) OR
                    # - Negative (shouldn't have trailing) OR
                    # - Trailing not activated yet
                    # Let recovery system handle it, skip trailing stop logic
                    pass
                else:
                    # Position is positive, standalone (no recovery), and trailing is active
                    # Apply PC2 time limit and trailing stop checks

                    # Check PC2 time limit (60 min) - close remaining 50% if time elapsed
                    pc2_time = tracked_pos.get('pc2_trigger_time')
                    if pc2_time:
                        # Normalize pc2_time to timezone-aware if needed
                        current_time = get_current_time()
                        if pc2_time.tzinfo is None:
                            import pytz
                            uk_tz = pytz.timezone("Europe/London")
                            pc2_time = uk_tz.localize(pc2_time)
                        else:
                            import pytz
                            uk_tz = pytz.timezone("Europe/London")
                            pc2_time = pc2_time.astimezone(uk_tz)

                        time_since_pc2 = current_time - pc2_time
                        if time_since_pc2 >= timedelta(minutes=60):
                            print(f"[PC2 TIME LIMIT] 60 min elapsed since PC2 for {ticket} - closing position")
                            if self.mt5.close_position(ticket):
                                # ML LOGGING: Log time-based exit
                                if self.ml_logger:
                                    entry_price = tracked_pos.get('entry_price', current_price)
                                    peak_price = tracked_pos.get('highest_profit_price', current_price)
                                    self.ml_logger.log_trailing_event(
                                        event_type='pc2_time_limit',
                                        ticket=ticket,
                                        symbol=symbol,
                                        time_since_pc2_minutes=float(time_since_pc2.total_seconds() / 60),
                                        exit_price=float(current_price),
                                        entry_price=float(entry_price),
                                        peak_price=float(peak_price)
                                    )
                                self.recovery_manager.untrack_position(ticket)
                                self.stats['trades_closed'] += 1
                            continue

                    # Update trailing stop (moves stop with price as profit increases)
                    # Store old stop for comparison
                    old_stop = tracked_pos.get('trailing_stop_price', 0)

                    # Update trailing stop
                    self.recovery_manager.update_trailing_stop(ticket, current_price)

                    # Check if stop moved and log the update
                    new_stop = tracked_pos.get('trailing_stop_price', 0)
                    if new_stop != old_stop and self.ml_logger:
                        pip_value = symbol_info.get('point', 0.0001) if symbol_info else 0.0001

                        # FIXED: Removed * 10 from pips calculation
                        pips_moved = abs(new_stop - old_stop) / pip_value
                        self.ml_logger.log_trailing_update(
                            ticket=ticket,
                            symbol=symbol,
                            old_stop=old_stop,
                            new_stop=new_stop,
                            current_price=current_price,
                            pips_moved=pips_moved
                        )

                    # Update hardware SL to match software trailing stop (crash protection)
                    trailing_stop_price = tracked_pos.get('trailing_stop_price')
                    if trailing_stop_price:
                        self.mt5.modify_position(ticket, sl=trailing_stop_price)

                    # Check if trailing stop hit
                    if self.recovery_manager.check_trailing_stop(ticket, current_price):
                        print(f"[TRAIL] Trailing stop hit for {ticket} - closing position")

                        # ML LOGGING: Log trailing stop hit with peak price and capture ratio
                        if self.ml_logger:
                            entry_price = tracked_pos.get('entry_price', current_price)
                            peak_price = tracked_pos.get('highest_profit_price', current_price)
                            self.ml_logger.log_trailing_hit(
                                ticket=ticket,
                                symbol=symbol,
                                trailing_stop_price=trailing_stop_price,
                                current_price=current_price,
                                entry_price=entry_price,
                                peak_price=peak_price
                            )

                        if self.mt5.close_position(ticket):
                            self.recovery_manager.untrack_position(ticket)
                            self.stats['trades_closed'] += 1
                        continue

            # RECOVERY & EXIT CONDITIONS: For UNDERWATER positions with active recovery
            # This handles positions that went negative and need grid/DCA/hedge management
            # Positive positions without recovery are handled by PC1/PC2/trailing above
            if ticket in self.recovery_manager.tracked_positions:
                # Get market data for trend detection in recovery triggers
                # Use M15 for fast trend detection (4x faster than H1)
                # Use H1 for ADX backup confirmation
                h1_data = None
                m15_data = None
                if symbol in self.market_data_cache:
                    h1_data = self.market_data_cache[symbol]['h1']
                    m15_data = self.market_data_cache[symbol].get('m15')

                # Check recovery triggers (grid, DCA, hedge for underwater positions)
                # Pass M15 for fast consecutive candle detection + H1 for ADX backup
                recovery_actions = self.recovery_manager.check_all_recovery_triggers(
                    ticket, current_price, pip_value, h1_data, m15_data
                )

                # Execute recovery actions
                for action in recovery_actions:
                    self._execute_recovery_action(action)

                # Check exit conditions (only for tracked original positions)
                # Priority order: 0) Stack drawdown (risk protection), 0.25) Per-stack stop loss, 0.5) Hedge drawdown, 1) Profit target, 2) Time limit, 3) VWAP reversion

                # Get account info and positions for checks
                account_info = self.mt5.get_account_info()
                all_positions = self.mt5.get_positions()

                # 0. Check stack drawdown (HIGHEST PRIORITY - risk protection)
                if self.recovery_manager.check_stack_drawdown(
                    ticket=ticket,
                    mt5_positions=all_positions,
                    pip_value=pip_value
                ):
                    self._close_recovery_stack(ticket, reason="Stack Drawdown Exceeded (4x expected profit loss)")
                    continue

                # 0.25. Check per-stack stop loss + cascade detection
                # Get current ADX for stop-out logging
                current_adx = None
                if symbol in self.market_data_cache:
                    h1_data = self.market_data_cache[symbol]['h1']
                    if 'adx' in h1_data.columns and len(h1_data) > 0:
                        current_adx = h1_data.iloc[-1]['adx']

                # Check stack stop loss (passes ADX for logging)
                stack_stop = self.recovery_manager.check_stack_stop_loss(
                    ticket=ticket,
                    mt5_positions=all_positions,
                    current_adx=current_adx
                )

                if stack_stop:
                    print(f"\n[STOP] Per-stack stop loss triggered")
                    print(f"   Stack type: {stack_stop['stack_type']}")
                    print(f"   Loss: ${stack_stop['loss_amount']:.2f}")
                    print(f"   Limit: ${stack_stop['stop_loss_limit']:.2f}")

                    # Close this stack
                    reason = f"Per-Stack Stop Loss ({stack_stop['stack_type']}: ${stack_stop['loss_amount']:.2f} exceeds ${stack_stop['stop_loss_limit']:.2f})"
                    self._close_recovery_stack(ticket, reason=reason)

                    # Check if cascade detected (2nd stop in 30min window)
                    if ENABLE_CASCADE_PROTECTION and self.recovery_manager.stop_out_tracker:
                        cascade_info = self.recovery_manager.stop_out_tracker.check_cascade()

                        if cascade_info:
                            # CASCADE DETECTED - close all underwater stacks
                            print(f"\n{'='*70}")
                            print(f"[CASCADE] MULTIPLE STOP-OUTS DETECTED")
                            print(f"{'='*70}")
                            print(f"   Stops in 30min: {cascade_info['stop_count']}")
                            if cascade_info['avg_adx']:
                                print(f"   Avg ADX: {cascade_info['avg_adx']:.1f}")
                            print(f"   Trend confirmed: {cascade_info['trend_confirmed']}")
                            print(f"   Symbols affected: {', '.join(cascade_info['symbols'])}")

                            # Get all underwater stacks
                            underwater_tickets = self.recovery_manager.get_underwater_stacks(all_positions)

                            if underwater_tickets:
                                print(f"\n   Closing {len(underwater_tickets)} underwater stack(s):")
                                for uw_ticket in underwater_tickets:
                                    if uw_ticket == ticket:
                                        continue  # Already closed above
                                    uw_profit = self.recovery_manager.calculate_net_profit(uw_ticket, all_positions)
                                    uw_symbol = self.recovery_manager.tracked_positions[uw_ticket]['symbol']
                                    print(f"     #{uw_ticket} ({uw_symbol}): ${uw_profit:.2f}")
                                    cascade_reason = f"Cascade Protection ({cascade_info['stop_count']} stops in 30min, trend confirmed, cutting losses)"
                                    self._close_recovery_stack(uw_ticket, reason=cascade_reason)

                                # Block new trades for affected symbols if trend confirmed
                                if cascade_info['trend_confirmed']:
                                    block_until = get_current_time() + timedelta(minutes=TREND_BLOCK_MINUTES)
                                    for affected_symbol in cascade_info['symbols']:
                                        self.cascade_blocks[affected_symbol] = block_until
                                        print(f"\n   [BLOCK] {affected_symbol} trades blocked for {TREND_BLOCK_MINUTES} minutes")
                                        if cascade_info['avg_adx']:
                                            print(f"          Market trending (ADX: {cascade_info['avg_adx']:.1f})")
                                    # Save cascade blocks to state file
                                    self.save_blocking_state()
                            else:
                                print(f"   No other underwater stacks found")

                            print(f"{'='*70}\n")

                    continue

                # 0.5. Check hedge drawdown (NEW - monitor hedge positions specifically)
                hedge_action = self.recovery_manager.check_hedge_drawdown(
                    ticket=ticket,
                    mt5_positions=all_positions
                )
                if hedge_action:
                    # Close underwater hedges
                    hedges_to_close = hedge_action.get('hedges_to_close', [])
                    for hedge in hedges_to_close:
                        hedge_ticket = hedge['ticket']
                        print(f"[STOP] Closing underwater hedge {hedge_ticket} (Loss: ${hedge['loss']:.2f})")
                        if self.mt5.close_position(hedge_ticket):
                            self.recovery_manager.remove_closed_hedge(ticket, hedge_ticket)
                            self.stats['trades_closed'] += 1

                    # Check market state after closing hedges
                    if symbol in self.market_data_cache:
                        h1_data = self.market_data_cache[symbol]['h1']
                        market_state = self.recovery_manager.check_market_state_for_hedge_close(
                            symbol=symbol,
                            current_data=h1_data
                        )

                        # Store market state for use in signal detection
                        if not hasattr(self, 'market_trending_block'):
                            self.market_trending_block = {}
                        self.market_trending_block[symbol] = market_state.get('should_block_new_trades', False)

                # 0.6. Check hedge DCA trigger (NEW - add DCA to underwater hedges)
                # IMPORTANT: This is SEPARATE from initial trade DCA (check_all_recovery_triggers)
                # Hedge DCA goes in ORIGINAL direction to help initial trade recovery
                hedge_dca_action = self.recovery_manager.check_hedge_dca_trigger(
                    ticket=ticket,
                    mt5_positions=all_positions,
                    pip_value=pip_value,
                    m15_data=m15_data  # M15 safeguard: requires 3 consecutive candles
                )
                if hedge_dca_action:
                    # Execute hedge DCA (separate from initial DCA)
                    self._execute_recovery_action(hedge_dca_action)

                # 0.7. Check hedge partial close (NEW - close hedge portions as original recovers)
                # IMPORTANT: This is SEPARATE from profit target close
                # Reduces hedge losses during recovery
                hedge_partial_action = self.recovery_manager.check_hedge_partial_close(
                    ticket=ticket,
                    mt5_positions=all_positions,
                    pip_value=pip_value
                )
                if hedge_partial_action:
                    # Execute partial hedge close
                    self._execute_hedge_partial_close(hedge_partial_action)

                # 1. Check profit target (from config)
                if account_info and self.recovery_manager.check_profit_target(
                    ticket=ticket,
                    mt5_positions=all_positions,
                    account_balance=account_info['balance'],
                    profit_percent=PROFIT_TARGET_PERCENT
                ):
                    profit = self.recovery_manager.calculate_net_profit(ticket, all_positions)
                    target = account_info['balance'] * (PROFIT_TARGET_PERCENT / 100.0)
                    self._close_recovery_stack(ticket, reason=f"Profit Target Reached (${profit:.2f} >= ${target:.2f} target)")
                    continue

                # 2. Check time limit (from config)
                if self.recovery_manager.check_time_limit(ticket, hours_limit=MAX_POSITION_HOURS):
                    self._close_recovery_stack(ticket, reason=f"Time Limit Exceeded (open > {MAX_POSITION_HOURS} hours)")
                    continue

            # 3. Check exit signal (VWAP reversion) - WITH FILTERING
            if symbol in self.market_data_cache:
                h1_data = self.market_data_cache[symbol]['h1']
                should_exit = self.signal_detector.check_exit_signal(position, h1_data)

                if should_exit:
                    # VWAP EXIT FILTERING: Only allow if conditions met
                    allow_vwap_exit = True

                    # Get instrument VWAP exit settings
                    from trading_bot.portfolio.instruments_config import INSTRUMENTS
                    instrument_config = INSTRUMENTS.get(symbol, {})
                    tp_settings = instrument_config.get('take_profit', {})
                    vwap_exit_enabled = tp_settings.get('vwap_exit_enabled', True)
                    vwap_exit_max_pips = tp_settings.get('vwap_exit_max_pips', 10)

                    # Check if this is a tracked position
                    if ticket in self.recovery_manager.tracked_positions:
                        tracked_pos = self.recovery_manager.tracked_positions[ticket]
                        pc1_closed = tracked_pos.get('partial_1_closed', False)

                        # Check if recovery is active (DCA or Hedge present)
                        has_dca = len(tracked_pos.get('dca_levels', [])) > 0
                        has_hedge = len(tracked_pos.get('hedges', [])) > 0
                        recovery_active = has_dca or has_hedge

                        # Calculate current profit in pips
                        entry_price = position['price_open']
                        current_price = position['price_current']
                        symbol_info = self.mt5.get_symbol_info(symbol)
                        pip_value = symbol_info.get('point', 0.0001) if symbol_info else 0.0001

                        pos_type = 'buy' if position['type'] == 0 else 'sell'
                        if pos_type == 'buy':
                            profit_pips = (current_price - entry_price) / pip_value
                        else:
                            profit_pips = (entry_price - current_price) / pip_value

                        # DISABLE VWAP exit if:
                        # 1. PC1 already triggered (new exit strategy active) OR
                        # 2. Profit >= max_pips (position should wait for PC1/PC2) OR
                        # 3. VWAP exit disabled in config OR
                        # 4. Recovery is active (DCA/Hedge working - let recovery finish)
                        if pc1_closed:
                            allow_vwap_exit = False
                            print(f"[VWAP] Exit blocked for {ticket} - PC1 triggered (new exit strategy active)")
                        elif profit_pips >= vwap_exit_max_pips:
                            allow_vwap_exit = False
                            print(f"[VWAP] Exit blocked for {ticket} - Profit {profit_pips:.1f} pips >= {vwap_exit_max_pips} (waiting for PC1)")
                        elif not vwap_exit_enabled:
                            allow_vwap_exit = False
                        elif recovery_active:
                            allow_vwap_exit = False
                            dca_count = len(tracked_pos.get('dca_levels', []))
                            hedge_count = len(tracked_pos.get('hedges', []))
                            print(f"[VWAP] Exit blocked for {ticket} - Recovery active (DCA: {dca_count}, Hedge: {hedge_count}) - Let recovery work!")

                    # Execute VWAP exit only if allowed
                    if allow_vwap_exit:
                        print(f"[VWAP] Exit signal detected for {ticket} - VWAP reversion")

                        # Check if this is a tracked position with recovery stack
                        if ticket in self.recovery_manager.tracked_positions:
                            print(f"[VWAP] Closing entire recovery stack for {ticket}")
                            entry_price = position['price_open']
                            current_price = position['price_current']
                            vwap_reason = f"VWAP Exit Signal (price reverted to VWAP: entry {entry_price:.5f} -> VWAP {current_price:.5f})"
                            self._close_recovery_stack(ticket, reason=vwap_reason)
                        else:
                            # Standalone position (no recovery) - close normally
                            if self.mt5.close_position(ticket):
                                self.stats['trades_closed'] += 1

        # Check for orphaned hedges (hedges whose original positions are gone)
        self._check_orphaned_hedges()

    def _check_orphaned_hedges(self):
        """Check ALL positions for orphaned hedges and close if losing > $75"""
        all_positions = self.mt5.get_positions()
        if not all_positions:
            return

        for position in all_positions:
            comment = position.get('comment', '')

            # Check if this is a hedge position
            if 'Hedge -' in comment:
                ticket = position.get('ticket')
                profit = position.get('profit', 0)
                volume = position.get('volume', 0)
                symbol = position.get('symbol', 'UNKNOWN')

                # Check if hedge is losing > $75
                if profit < -75.0:
                    loss = abs(profit)
                    print(f"[STOP] ORPHANED HEDGE DETECTED: {symbol} #{ticket}")
                    print(f"   Loss: ${loss:.2f} (exceeds $75 threshold)")
                    print(f"   Volume: {volume:.2f} lots")
                    print(f"   Comment: {comment}")
                    print(f"   Closing immediately for risk protection...")

                    if self.mt5.close_position(ticket):
                        print(f"   [OK] Orphaned hedge {ticket} closed successfully")
                        self.stats['trades_closed'] += 1
                    else:
                        print(f"   [ERROR] Failed to close orphaned hedge {ticket}")

    def _check_for_signals(self, symbol: str):
        """Check for new entry signals"""
        if symbol not in self.market_data_cache:
            return

        # Track blocking reasons for visibility
        blocking_reasons = []

        # Check if market is blocked due to trending (ADX > 40)
        if hasattr(self, 'market_trending_block') and self.market_trending_block.get(symbol, False):
            blocking_reasons.append("ADX > 40 (strong trend)")

        # Check if symbol is tradeable based on portfolio trading windows (bypass in test mode)
        if not self.test_mode and not self.portfolio_manager.is_symbol_tradeable(symbol):
            blocking_reasons.append("Outside trading window")
            # Show blocking summary
            print(f"   [BLOCKED] {symbol}: {', '.join(blocking_reasons)}", flush=True)
            return  # Not in trading window for this symbol

        cache = self.market_data_cache[symbol]
        h1_data = cache['h1']
        m15_data = cache.get('m15')  # M15 for fast trend detection
        d1_data = cache['d1']
        w1_data = cache['w1']

        current_time = get_current_time()
        signal = None

        # Check which strategy can trade based on time filters (bypass in test mode)
        if self.test_mode:
            can_trade_mr = True
            can_trade_bo = True
        else:
            can_trade_mr = self.time_filter.can_trade_mean_reversion(current_time)
            can_trade_bo = self.time_filter.can_trade_breakout(current_time)

        # Add time filter blocking reasons
        if not can_trade_mr:
            blocking_reasons.append("Mean reversion: outside allowed hours/days")
        if not can_trade_bo:
            blocking_reasons.append("Breakout: outside allowed hours/days")

        # Show blocking summary if any blocks
        if blocking_reasons:
            print(f"   [BLOCKED] {symbol}: {', '.join(blocking_reasons)}", flush=True)

        # Early return if ALL strategies are blocked (no point checking for signals)
        if not can_trade_mr and not can_trade_bo:
            return  # Nothing to check

        if self.debug:
            print(f"   [INFO] {symbol}: Checking signals (MR: {can_trade_mr}, BO: {can_trade_bo})...", flush=True)

        # Try mean reversion signal first (if allowed)
        if can_trade_mr:
            if self.debug:
                print(f"   [CHECK] {symbol}: Calling signal_detector.detect_signal()...", flush=True)
            signal = self.signal_detector.detect_signal(
                current_data=h1_data,
                daily_data=d1_data,
                weekly_data=w1_data,
                symbol=symbol,
                m15_data=m15_data  # Pass M15 for fast trend detection
            )
            if self.debug:
                print(f"   [CHECK] {symbol}: detect_signal() returned: {signal is not None} (signal object: {'YES' if signal else 'None'})", flush=True)
            if signal:
                signal['strategy_type'] = 'mean_reversion'
                if self.debug:
                    print(f"   [OK] {symbol}: Mean Reversion signal found (Score: {signal.get('confluence_score', 0)})", flush=True)
            else:
                if self.debug:
                    print(f"   [SKIP] {symbol}: detect_signal() returned None (either score < {MIN_CONFLUENCE_SCORE} or rejected by filters)", flush=True)
        else:
            if self.debug:
                print(f"   [PAUSE]  {symbol}: Mean reversion trading not allowed at this time", flush=True)

        # Try breakout signal (if mean reversion found nothing and breakout is allowed)
        if signal is None and can_trade_bo and self.breakout_strategy:
            # FIXED: Check if h1_data is empty before accessing iloc
            if len(h1_data) == 0:
                print(f"[WARN] Empty H1 data for {symbol}, skipping breakout check")
                return None

            # Get current price and volume
            latest_bar = h1_data.iloc[-1]
            current_price = latest_bar['close']

            # Get volume - handle both 'volume' and 'tick_volume' columns
            if 'volume' in latest_bar:
                current_volume = latest_bar['volume']
            elif 'tick_volume' in latest_bar:
                current_volume = latest_bar['tick_volume']
            else:
                current_volume = 0  # Default if no volume data
                print(f"[WARN] Warning: No volume data for {symbol}, using 0")

            # Calculate ATR
            atr = h1_data['atr'].iloc[-1] if 'atr' in h1_data.columns else 0

            # Calculate RSI and MACD for comprehensive breakout detection
            from indicators.technical import add_indicators_to_dataframe
            h1_with_indicators = add_indicators_to_dataframe(h1_data)
            rsi = h1_with_indicators['rsi'].iloc[-1] if 'rsi' in h1_with_indicators.columns else 50.0
            macd_histogram = h1_with_indicators['macd_histogram'].iloc[-1] if 'macd_histogram' in h1_with_indicators.columns else 0.0

            # Calculate volume profile for LVN breakouts
            volume_profile_data = self.signal_detector.volume_profile.get_signals(
                h1_data, current_price, lookback=200
            )
            volume_profile = volume_profile_data.get('profile', {})

            # Get weekly levels for weekly breakout detection
            htf_levels = self.signal_detector.htf_levels.get_all_levels(d1_data, w1_data)
            weekly_data = htf_levels.get('weekly', {})
            # Map to format expected by breakout strategy
            weekly_levels = {
                'high': weekly_data.get('prev_week_high', current_price + 0.01),
                'low': weekly_data.get('prev_week_low', current_price - 0.01)
            }

            # Prepare indicators dict for comprehensive breakout checker
            indicators = {
                'rsi': rsi,
                'macd_histogram': macd_histogram,
                'atr': atr,
                'volume': current_volume
            }

            # Check for comprehensive breakout signals (range + LVN + weekly)
            breakout_signal = self.breakout_strategy.check_breakout_signal(
                data=h1_data,
                current_price=current_price,
                volume_profile=volume_profile,
                weekly_levels=weekly_levels,
                indicators=indicators,
                current_time=current_time
            )

            if breakout_signal:
                # Convert breakout signal to standard signal format
                signal = {
                    'symbol': symbol,
                    'direction': breakout_signal['direction'],  # Now lowercase 'buy'/'sell'
                    'price': current_price,
                    'strategy_type': 'breakout',
                    'confluence_score': breakout_signal.get('score', 3),
                    'factors': breakout_signal.get('factors', []),
                    'breakout_details': breakout_signal  # Store full breakout details
                }
                if self.debug:
                    print(f"   [OK] {symbol}: Breakout signal found (Score: {signal.get('confluence_score', 0)})", flush=True)
            else:
                if self.debug:
                    print(f"   [SKIP] {symbol}: No breakout signal detected", flush=True)

        if signal is None:
            if self.debug:
                print(f"   [FAIL] {symbol}: No valid signals (need confluence >= {MIN_CONFLUENCE_SCORE})", flush=True)
            return

        # Signal detected!
        self.stats['signals_detected'] += 1

        # CRITICAL: Calculate ADX for conditional SL logic
        # If ADX > 30 (trending), apply hard SL at -50 pips
        from indicators.adx import calculate_adx
        h1_with_adx = calculate_adx(h1_data.copy(), period=14)
        current_adx = h1_with_adx['adx'].iloc[-1] if 'adx' in h1_with_adx.columns and len(h1_with_adx) > 0 else 0.0
        signal['adx'] = current_adx  # Add ADX to signal for use in _execute_signal

        print()
        print(f"Signal: {signal.get('strategy_type', 'unknown').upper()}")
        print(self.signal_detector.get_signal_summary(signal))

        # Add breakout-specific logging
        if signal.get('strategy_type') == 'breakout' and signal.get('breakout_details'):
            details = signal['breakout_details']
            breakout_type = details.get('type', 'unknown')
            print(f"   Breakout Strategy: {breakout_type.upper()}")

            if breakout_type == 'range_breakout':
                print(f"   Range: {details.get('range_low', 0):.5f} - {details.get('range_high', 0):.5f}")
                print(f"   Range Size: {details.get('range_size_pips', 0):.1f} pips")
                print(f"   Logic: Price breaks {details.get('breakout_type', 'unknown').upper()} -> {signal['direction'].upper()} (ride momentum)")
            elif breakout_type == 'lvn_breakout':
                print(f"   LVN Level: {details.get('lvn_level', 0):.5f}")
                print(f"   RSI: {details.get('rsi', 0):.1f}")
                print(f"   Logic: Price breaks through Low Volume Node -> {signal['direction'].upper()} (fast move)")
            elif breakout_type == 'weekly_breakout':
                print(f"   Weekly High: {details.get('weekly_high', 0):.5f}")
                print(f"   Weekly Low: {details.get('weekly_low', 0):.5f}")
                print(f"   RSI: {details.get('rsi', 0):.1f}")
                print(f"   Logic: Price breaks weekly level -> {signal['direction'].upper()} (continuation)")

        print()

        # Execute trade
        self._execute_signal(signal)

    def _execute_signal(self, signal: Dict):
        """
        Execute a trading signal

        Args:
            signal: Signal dict from signal detector
        """
        symbol = signal['symbol']
        direction = signal['direction']
        price = signal['price']

        # Validate direction is lowercase 'buy' or 'sell'
        if direction not in ['buy', 'sell']:
            print(f"ERROR: Invalid direction '{direction}'. Must be 'buy' or 'sell'")
            print(f"   Signal type: {signal.get('strategy_type', 'unknown')}")
            print(f"   Converting to lowercase...")
            direction = direction.lower()
            if direction not in ['buy', 'sell']:
                print(f"ERROR: Direction '{direction}' still invalid after lowercase conversion. Aborting trade.")
                return

        # Log order details
        print(f"Order Details:")
        print(f"   Direction: {direction.upper()}")
        print(f"   Strategy: {signal.get('strategy_type', 'unknown').upper()}")
        print(f"   Symbol: {symbol}")
        print(f"   Price: {price:.5f}")

        # Check if symbol is blocked due to cascade (multiple stop-outs = trend detected)
        if symbol in self.cascade_blocks:
            block_until = self.cascade_blocks[symbol]
            if get_current_time() < block_until:
                time_remaining = (block_until - get_current_time()).total_seconds() / 60
                print(f" New trade BLOCKED for {symbol} - CASCADE PROTECTION")
                print(f"   Multiple stop-outs detected (trend confirmed)")
                print(f"   Block expires in: {time_remaining:.0f} minutes")
                return
            else:
                # Block expired, remove it
                del self.cascade_blocks[symbol]
                print(f" Cascade block for {symbol} expired - resuming normal trading")

        # Check if new trades are blocked due to trending market
        # This is the "belt and braces" approach - prevent new trades when market is trending
        # Recovery trades are still allowed (they are executed via _execute_recovery_action, not here)
        if hasattr(self, 'market_trending_block') and self.market_trending_block.get(symbol, False):
            print(f" New trade BLOCKED for {symbol} - Market is trending")
            print(f"   This protection prevents opening new positions during strong trends")
            print(f"   Recovery trades are still allowed to manage existing positions")
            return

        # Get account and symbol info
        account_info = self.mt5.get_account_info()
        symbol_info = self.mt5.get_symbol_info(symbol)

        if not account_info or not symbol_info:
            print("[ERROR] Failed to get account/symbol info")
            return

        # Calculate position size
        volume = self.risk_calculator.calculate_position_size(
            account_balance=account_info['balance'],
            symbol_info=symbol_info
        )

        # Apply breakout multiplier if this is a breakout signal
        if signal.get('strategy_type') == 'breakout':
            volume = volume * BREAKOUT_LOT_SIZE_MULTIPLIER
            # Round to broker's volume step
            volume_step = symbol_info.get('volume_step', 0.01)
            volume = round(volume / volume_step) * volume_step
            # Ensure minimum volume
            volume = max(symbol_info.get('volume_min', 0.01), volume)
            print(f" Breakout signal: Reducing lot size to {volume} (50% of base)")

        # Get current positions for validation
        positions = self.mt5.get_positions()

        # Validate trade
        can_trade, reason = self.risk_calculator.validate_trade(
            account_info=account_info,
            symbol_info=symbol_info,
            volume=volume,
            current_positions=positions
        )

        if not can_trade:
            print(f"[ERROR] Trade validation failed: {reason}")
            return

        # Import INITIAL_TRADE_COUNT
        from config.strategy_config import INITIAL_TRADE_COUNT

        # ADDED: Check if opening INITIAL_TRADE_COUNT positions would exceed max lots
        total_volume_to_add = volume * INITIAL_TRADE_COUNT
        all_positions = self.mt5.get_positions()
        if not self.recovery_manager.check_max_lots_limit(total_volume_to_add, all_positions):
            print(f"[ERROR] Cannot open {INITIAL_TRADE_COUNT} trades - would exceed MAX_TOTAL_LOTS limit")
            return

        # ADX-CONDITIONAL HARD STOP LOGIC (CONFIGURABLE)
        # When ENABLE_ADX_HARD_STOPS = True:
        #   If ADX > threshold (trending market), apply hard SL to cut losses fast
        #   If ADX <= threshold (ranging/moderate), allow recovery system to work
        # When ENABLE_ADX_HARD_STOPS = False:
        #   Normal recovery system behavior (current system)
        from config.strategy_config import (
            ENABLE_ADX_HARD_STOPS,
            ADX_HARD_STOP_THRESHOLD,
            ADX_HARD_STOP_PIPS
        )

        current_adx = signal.get('adx', 0.0)
        hard_sl = None

        if ENABLE_ADX_HARD_STOPS and current_adx > ADX_HARD_STOP_THRESHOLD:
            # Calculate hard stop loss at configured pip distance
            symbol_info = self.mt5.get_symbol_info(symbol)
            point = symbol_info.get('point', 0.0001)
            pip_distance = ADX_HARD_STOP_PIPS * point

            # Get current price for SL calculation
            tick = self.mt5.get_symbol_tick(symbol)
            if tick:
                current_price = tick.get('bid') if direction == 'sell' else tick.get('ask')
                if direction == 'buy':
                    hard_sl = current_price - pip_distance  # SL below entry for BUY
                else:
                    hard_sl = current_price + pip_distance  # SL above entry for SELL

                print(f" [ADX HARD STOP] Trending market detected (ADX: {current_adx:.1f} > {ADX_HARD_STOP_THRESHOLD})")
                print(f"   Applying HARD SL at -{ADX_HARD_STOP_PIPS} pips: {hard_sl:.5f}")
                print(f"   Recovery (DCA/Hedge/Grid) will be BLOCKED for this position")

        # Place order(s) - open multiple trades if INITIAL_TRADE_COUNT > 1
        # Include strategy type in comment for ML analysis
        # VWAP = Mean reversion to VWAP/levels, BREAKOUT = Momentum through levels
        strategy_label = "VWAP" if signal.get('strategy_type') == 'mean_reversion' else "BREAKOUT"
        comment = f"{strategy_label}:C{signal['confluence_score']}"
        trades_opened = 0

        for i in range(INITIAL_TRADE_COUNT):
            trade_comment = comment if INITIAL_TRADE_COUNT == 1 else f"{comment} #{i+1}"

            ticket = self.mt5.place_order(
                symbol=symbol,
                order_type=direction,
                volume=volume,
                sl=hard_sl,  # ADX > 30: -50 pips hard SL, ADX <= 30: None (recovery allowed)
                tp=None,  # Using VWAP reversion instead
                comment=trade_comment
            )

            if ticket:
                trades_opened += 1
                self.stats['trades_opened'] += 1
                print(f"[OK] Trade #{i+1} opened: Ticket {ticket}")

                # Get actual fill price from MT5 (CRITICAL: Don't use signal price)
                actual_entry_price = price  # Default to signal price as fallback
                all_positions = self.mt5.get_positions()
                for pos in all_positions:
                    if pos['ticket'] == ticket:
                        actual_entry_price = pos['price_open']
                        slippage_pips = abs(actual_entry_price - price) * 10000
                        print(f"[ENTRY] Signal: {price:.5f}, Fill: {actual_entry_price:.5f}, Slippage: {slippage_pips:.1f} pips")
                        break

                # Start tracking for recovery with ACTUAL fill price
                # Pass ADX to determine if recovery is allowed
                self.recovery_manager.track_position(
                    ticket=ticket,
                    symbol=symbol,
                    entry_price=actual_entry_price,
                    position_type=direction,
                    volume=volume,
                    is_grid_child=False,
                    is_recovery_order=False,
                    open_adx=current_adx  # Store ADX at entry time
                )

                # ML LOGGING: Log trade entry with enhanced data
                try:
                    # Get current spread
                    tick = self.mt5.get_symbol_tick(symbol)
                    spread_pips = 0
                    if tick:
                        spread = tick.get('ask', 0) - tick.get('bid', 0)
                        point = symbol_info.get('point', 0.0001)
                        spread_pips = spread / point

                    # Get ATR from market data
                    atr_pips = 0
                    if symbol in self.market_data_cache:
                        h1_data = self.market_data_cache[symbol]['h1']
                        if 'atr' in h1_data.columns and len(h1_data) > 0:
                            atr = h1_data.iloc[-1]['atr']
                            atr_pips = atr / symbol_info.get('point', 0.0001)

                    # Get confluence factors from signal
                    confluence_factors = signal.get('factors', [])

                    # Prepare trade data for ML logging
                    trade_data = {
                        'ticket': ticket,
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': actual_entry_price,
                        'expected_price': price,
                        'volume': volume,
                        'spread_at_entry_pips': spread_pips,
                        'confluence_score': signal.get('confluence_score', 0),
                        'confluence_factors': confluence_factors,
                        'adx': current_adx,
                        'atr_pips': atr_pips,
                        'hour': get_current_time().hour,
                        'strategy_type': signal.get('strategy_type', 'unknown')
                    }

                    # Log to ML manager
                    self.ml_manager.log_trade_entry(trade_data)

                    if self.debug:
                        print(f"[DEBUG] ML: Logged trade entry for {ticket}", flush=True)

                except Exception as e:
                    print(f"[WARN] ML logging failed for {ticket}: {e}")
            else:
                print(f"[ERROR] Failed to open trade #{i+1}")

        if trades_opened > 0:
            print(f" Total trades opened: {trades_opened}/{INITIAL_TRADE_COUNT}")

    def _close_recovery_stack(self, original_ticket: int, reason: str = "Unknown"):
        """
        Close entire recovery stack (original + grid + hedge + DCA)

        Args:
            original_ticket: Original position ticket
            reason: WHY the stack is being closed (MANDATORY for debugging)
        """
        # Get all tickets in the stack
        stack_tickets = self.recovery_manager.get_all_stack_tickets(original_ticket)

        # Calculate final P&L before closing
        all_positions = self.mt5.get_positions()
        final_pnl = self.recovery_manager.calculate_net_profit(original_ticket, all_positions)

        print(f"\n{'='*70}")
        print(f"[STACK] CLOSING RECOVERY STACK #{original_ticket}")
        print(f"{'='*70}")
        print(f"   Reason: {reason}")
        print(f"   Final P&L: ${final_pnl:.2f}" if final_pnl is not None else "   Final P&L: Unknown")
        print(f"   Positions to close: {len(stack_tickets)}")
        print(f"{'='*70}")

        closed_count = 0
        for ticket in stack_tickets:
            if self.mt5.close_position(ticket):
                closed_count += 1
                self.stats['trades_closed'] += 1
                print(f"   [OK] Closed #{ticket}")
            else:
                print(f"   [ERROR] Failed to close #{ticket}")

        # Untrack the original position
        self.recovery_manager.untrack_position(original_ticket)

        print(f"[STACK] Stack closed: {closed_count}/{len(stack_tickets)} positions")
        print(f"{'='*70}\n")

    def _execute_recovery_action(self, action: Dict):
        """
        Execute a recovery action (grid/hedge/dca/hedge_dca)

        Args:
            action: Recovery action dict
        """
        action_type = action['action']
        symbol = action['symbol']
        order_type = action['type']
        volume = action['volume']
        comment = action['comment']
        original_ticket = action.get('original_ticket')

        # ML LOGGING: Log recovery decision before execution
        if action_type in ['dca', 'hedge'] and original_ticket:
            try:
                # Get current position data
                all_positions = self.mt5.get_positions()
                original_pos = None
                for pos in all_positions:
                    if pos['ticket'] == original_ticket:
                        original_pos = pos
                        break

                if original_pos and original_ticket in self.recovery_manager.tracked_positions:
                    tracked_pos = self.recovery_manager.tracked_positions[original_ticket]
                    symbol_info = self.mt5.get_symbol_info(symbol)
                    pip_value = symbol_info.get('point', 0.0001) if symbol_info else 0.0001

                    # Calculate pips underwater
                    entry_price = tracked_pos.get('entry_price', 0)
                    current_price = original_pos.get('price_current', 0)
                    pos_type = tracked_pos.get('type')

                    if pos_type == 'buy':
                        pips_underwater = (current_price - entry_price) / pip_value
                    else:
                        pips_underwater = (entry_price - current_price) / pip_value

                    # Get ADX at trigger
                    adx_at_trigger = 0
                    if symbol in self.market_data_cache:
                        h1_data = self.market_data_cache[symbol]['h1']
                        if 'adx' in h1_data.columns and len(h1_data) > 0:
                            adx_at_trigger = h1_data.iloc[-1]['adx']

                    # Get ADX at entry
                    adx_at_entry = tracked_pos.get('open_adx', 0)

                    # Prepare recovery decision data
                    recovery_data = {
                        'ticket': original_ticket,
                        'type': action_type.upper(),
                        'pips_underwater': pips_underwater,
                        'unrealized_pnl': original_pos.get('profit', 0),
                        'adx_at_entry': adx_at_entry,
                        'adx_at_trigger': adx_at_trigger,
                        'was_blocked': False,  # It's executing, so not blocked
                        'recovery_placed': True
                    }

                    # Log to ML manager
                    self.ml_manager.log_recovery_decision(recovery_data)

                    if self.debug:
                        print(f"[DEBUG] ML: Logged {action_type} recovery for {original_ticket}", flush=True)

            except Exception as e:
                print(f"[WARN] ML recovery logging failed: {e}")

        # Get calculated price for grid trades (if provided)
        # Grid trades should execute at specific price levels, not market price
        calculated_price = action.get('price', None)

        # Place order
        # IMPORTANT: Grid trades are PYRAMID trades - open new positions in same direction
        # This creates overlapping positions to lock in profit while maintaining exposure
        if action_type in ['grid', 'dca', 'hedge', 'hedge_dca']:
            # MARKET ORDER: Executes immediately for Grid/DCA/Hedge/Hedge DCA
            ticket = self.mt5.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                comment=comment
            )
        else:
            ticket = None

        if ticket:
            # Get actual fill price from MT5 (CRITICAL: Don't use calculated price)
            actual_entry_price = action.get('price', 0)  # Default to calculated price as fallback
            all_positions = self.mt5.get_positions()
            for pos in all_positions:
                if pos['ticket'] == ticket:
                    actual_entry_price = pos['price_open']
                    break

            # Grid trades get tracked with LIMITED recovery (DCA/Hedge YES, more Grids NO)
            # Hedge/DCA/Hedge_DCA trades link to original parent
            if action_type == 'grid':
                # Track grid as position with recovery, but flag to prevent grid spawning
                self.recovery_manager.track_position(
                    ticket=ticket,
                    symbol=symbol,
                    entry_price=actual_entry_price,
                    position_type=order_type,
                    volume=volume,
                    is_grid_child=True  # Prevents more grids (stops cascade)
                )
                print(f"   Grid trade {ticket} tracked with recovery (DCA/Hedge: YES, more Grids: NO)")
                self.stats['grid_levels_added'] += 1
            elif action_type == 'hedge_dca':
                # Hedge DCA links to HEDGE parent (not original)
                # Store in hedge's dca_levels array
                if original_ticket:
                    hedge_ticket = action.get('hedge_ticket')
                    # Find hedge in tracked position
                    if original_ticket in self.recovery_manager.tracked_positions:
                        position = self.recovery_manager.tracked_positions[original_ticket]
                        hedge_tickets = position.get('hedge_tickets', [])

                        # Find the specific hedge this DCA belongs to
                        for hedge_info in hedge_tickets:
                            if hedge_info.get('ticket') == hedge_ticket:
                                # Clear pending flag and store ticket
                                hedge_dca_levels = hedge_info.get('dca_levels', [])

                                # CRITICAL FIX: Always ensure ticket is stored
                                stored = False
                                if hedge_dca_levels and hedge_dca_levels[-1].get('pending'):
                                    # Normal case: clear pending flag
                                    hedge_dca_levels[-1]['pending'] = False
                                    hedge_dca_levels[-1]['ticket'] = ticket
                                    stored = True
                                    print(f"   Hedge DCA {ticket} linked to hedge {hedge_ticket} (cleared pending)")
                                else:
                                    # FAILSAFE: No pending entry found - add new entry
                                    # This prevents the cascade bug where DCA triggers every minute
                                    logger.warning(f"[HEDGE DCA] No pending entry for hedge {hedge_ticket} - adding manually")
                                    hedge_dca_levels.append({
                                        'level': len(hedge_dca_levels) + 1,
                                        'volume': volume,
                                        'trigger_pips': 0,  # Unknown at this point
                                        'time': get_current_time(),
                                        'pending': False,
                                        'ticket': ticket
                                    })
                                    stored = True
                                    print(f"   Hedge DCA {ticket} linked to hedge {hedge_ticket} (failsafe add)")

                                # Save updated list
                                hedge_info['dca_levels'] = hedge_dca_levels

                                # Force save state to disk
                                if stored:
                                    self.recovery_manager._save_state()
                                    print(f"   [OK] State saved - prevents duplicate hedge DCA triggers")

                                    # ML LOGGING: Log hedge DCA event
                                    if self.ml_logger:
                                        # Get data for logging
                                        all_positions = self.mt5.get_positions()
                                        original_pos = None
                                        hedge_pos = None
                                        for pos in all_positions:
                                            if pos.get('ticket') == original_ticket:
                                                original_pos = pos
                                            elif pos.get('ticket') == hedge_ticket:
                                                hedge_pos = pos

                                        if original_pos and hedge_pos:
                                            symbol_info = self.mt5.get_symbol_info(symbol)
                                            pip_value = symbol_info.get('point', 0.0001) if symbol_info else 0.0001

                                            # Calculate pips
                                            original_entry = position.get('entry_price', 0)
                                            original_current = original_pos.get('price_current', 0)
                                            original_type = position.get('type')

                                            if original_type == 'buy':
                                                original_pips = (original_current - original_entry) / pip_value
                                            else:
                                                original_pips = (original_entry - original_current) / pip_value

                                            hedge_entry = hedge_pos.get('price_open', 0)
                                            hedge_current = hedge_pos.get('price_current', 0)
                                            hedge_type = hedge_info.get('type')

                                            if hedge_type == 'buy':
                                                hedge_pips = (hedge_entry - hedge_current) / pip_value
                                            else:
                                                hedge_pips = (hedge_current - hedge_entry) / pip_value

                                            self.ml_logger.log_hedge_dca(
                                                original_ticket=original_ticket,
                                                hedge_ticket=hedge_ticket,
                                                symbol=symbol,
                                                hedge_dca_ticket=ticket,
                                                level=hedge_dca_levels[-1].get('level', 0),
                                                volume=volume,
                                                hedge_pips_underwater=hedge_pips,
                                                original_pips=original_pips,
                                                hedge_type=hedge_type,
                                                dca_type=order_type,
                                                m15_confirmed=True  # Always true if we got here (M15 check passed)
                                            )
                                break
                self.stats['dca_levels_added'] += 1
            else:
                # Hedge and DCA link to original parent
                if original_ticket:
                    self.recovery_manager.store_recovery_ticket(
                        original_ticket=original_ticket,
                        recovery_ticket=ticket,
                        action_type=action_type
                    )

                # Update statistics
                if action_type == 'hedge':
                    self.stats['hedges_activated'] += 1
                elif action_type == 'dca':
                    self.stats['dca_levels_added'] += 1

    def _execute_hedge_partial_close(self, action: Dict):
        """
        Execute partial close of hedge position + its DCAs.

        Args:
            action: Partial close action dict with:
                - hedge_ticket: Hedge position to partially close
                - hedge_info: Full hedge info dict (contains dca_levels)
                - close_percent: Percentage to close (0.5, 0.75, 1.0)
                - reason: Why closing (e.g., "Original recovered 50%")
        """
        hedge_ticket = action['hedge_ticket']
        hedge_info = action['hedge_info']
        close_percent = action['close_percent']
        reason = action['reason']
        symbol = action['symbol']

        print(f"\n{'='*70}")
        print(f"[PARTIAL] PARTIAL HEDGE CLOSE #{hedge_ticket}")
        print(f"{'='*70}")
        print(f"   Reason: {reason}")
        print(f"   Close: {close_percent*100:.0f}%")

        # Get all positions for this hedge (hedge + hedge DCAs)
        all_positions = self.mt5.get_positions()
        positions_to_close = []

        # Find the main hedge position
        hedge_pos = None
        for pos in all_positions:
            if pos.get('ticket') == hedge_ticket:
                hedge_pos = pos
                positions_to_close.append({
                    'ticket': hedge_ticket,
                    'volume': pos.get('volume', 0),
                    'type': 'hedge'
                })
                break

        # Find all hedge DCA positions (if any)
        hedge_dca_levels = hedge_info.get('dca_levels', [])
        for dca_info in hedge_dca_levels:
            dca_ticket = dca_info.get('ticket')
            if dca_ticket:
                for pos in all_positions:
                    if pos.get('ticket') == dca_ticket:
                        positions_to_close.append({
                            'ticket': dca_ticket,
                            'volume': pos.get('volume', 0),
                            'type': 'hedge_dca'
                        })
                        break

        print(f"   Positions to close: {len(positions_to_close)} (1 hedge + {len(positions_to_close)-1} hedge DCAs)")

        # Close positions
        if close_percent == 1.0:
            # Close 100% - close entire positions
            closed_count = 0
            for pos_info in positions_to_close:
                ticket = pos_info['ticket']
                if self.mt5.close_position(ticket):
                    closed_count += 1
                    self.stats['trades_closed'] += 1
                    print(f"   [OK] Closed {pos_info['type']} #{ticket} (100%)")
                else:
                    print(f"   [ERROR] Failed to close #{ticket}")

            # Clear hedge DCAs from tracking if 100% closed
            if closed_count > 0:
                hedge_info['dca_levels'] = []

        else:
            # Partial close - close percentage of volume
            closed_count = 0
            for pos_info in positions_to_close:
                ticket = pos_info['ticket']
                current_volume = pos_info['volume']
                close_volume = current_volume * close_percent

                # Round to broker step size
                from utils.helpers import round_volume_to_step
                close_volume = round_volume_to_step(close_volume)

                if close_volume > 0:
                    if self.mt5.close_position_partial(ticket, close_volume):
                        closed_count += 1
                        self.stats['trades_closed'] += 1
                        print(f"   [OK] Closed {pos_info['type']} #{ticket} ({close_percent*100:.0f}% = {close_volume:.2f} lots)")
                    else:
                        print(f"   [ERROR] Failed to partially close #{ticket}")

        # ML LOGGING: Log hedge partial close event
        if self.ml_logger and closed_count > 0:
            # Get final data for logging
            original_ticket = action['original_ticket']
            if original_ticket in self.recovery_manager.tracked_positions:
                position = self.recovery_manager.tracked_positions[original_ticket]
                all_positions_now = self.mt5.get_positions()

                # Find original position
                original_pos = None
                for pos in all_positions_now:
                    if pos.get('ticket') == original_ticket:
                        original_pos = pos
                        break

                if original_pos:
                    symbol_info = self.mt5.get_symbol_info(symbol)
                    pip_value = symbol_info.get('point', 0.0001) if symbol_info else 0.0001

                    # Calculate original pips
                    original_entry = position.get('entry_price', 0)
                    original_current = original_pos.get('price_current', 0)
                    original_type = position.get('type')

                    if original_type == 'buy':
                        original_pips = (original_current - original_entry) / pip_value
                    else:
                        original_pips = (original_entry - original_current) / pip_value

                    # Get hedge profit (if still exists, else 0)
                    hedge_profit = 0
                    if hedge_pos:
                        hedge_profit = hedge_pos.get('profit', 0)

                    hedge_dca_count = len(hedge_dca_levels)

                    self.ml_logger.log_hedge_partial_close(
                        original_ticket=original_ticket,
                        hedge_ticket=hedge_ticket,
                        symbol=symbol,
                        close_percent=close_percent,
                        reason=reason,
                        original_pips=original_pips,
                        hedge_profit=hedge_profit,
                        positions_closed=closed_count,
                        hedge_dca_count=hedge_dca_count
                    )

        print(f"{'='*70}")
        print(f"   Closed: {closed_count}/{len(positions_to_close)} positions")
        print(f"{'='*70}\n")

    def _can_open_new_position(self, symbol: str) -> bool:
        """Check if we can open a new position"""
        # Check cascade protection trade block
        if hasattr(self, 'trade_block_until'):
            if get_current_time() < self.trade_block_until:
                remaining = (self.trade_block_until - get_current_time()).total_seconds() / 60
                logger.debug(f"Trade blocked by cascade protection ({remaining:.1f} minutes remaining)")
                return False

        # Check total positions
        all_positions = self.mt5.get_positions()
        if len(all_positions) >= MAX_OPEN_POSITIONS:
            return False

        # Check positions per symbol
        symbol_positions = self.mt5.get_positions(symbol)
        if len(symbol_positions) >= MAX_POSITIONS_PER_SYMBOL:
            return False

        return True

    def get_status(self) -> Dict:
        """Get current strategy status"""
        account_info = self.mt5.get_account_info()
        positions = self.mt5.get_positions()

        risk_metrics = self.risk_calculator.get_risk_metrics(
            account_info=account_info or {},
            positions=positions
        )

        recovery_status = self.recovery_manager.get_all_positions_status()

        return {
            'running': self.running,
            'account': account_info,
            'risk_metrics': risk_metrics,
            'positions': positions,
            'recovery_status': recovery_status,
            'statistics': self.stats,
            'cached_symbols': list(self.market_data_cache.keys()),
        }

    def reload_config(self):
        """
        Reload configuration from strategy_config.py without restarting bot
        Fixes Python caching issue where config changes require full restart
        """
        print()
        print("=" * 60)
        print("[SYNC] RELOADING CONFIGURATION")
        print("=" * 60)
        success = reload_config()
        if success:
            print_current_config()
            print("[OK] Config reloaded successfully!")
            print("   Changes will take effect on next trading cycle")
        else:
            print("[ERROR] Config reload failed")
        print("=" * 60)
        print()

    def save_blocking_state(self, state_file: str = "data/recovery_state.json"):
        """
        Save blocking state (cascade_blocks, market_trending_block) to recovery state file.
        This is merged with position tracking state from RecoveryManager.

        Args:
            state_file: Path to state file
        """
        from pathlib import Path
        import json

        try:
            state_path = Path(state_file)

            # Load existing state (positions from RecoveryManager)
            existing_state = {}
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    existing_state = json.load(f)

            # Add blocking state
            existing_state['cascade_blocks'] = {}
            existing_state['market_trending_block'] = {}
            existing_state['last_block_update'] = datetime.now().isoformat()

            # Convert cascade blocks (datetime to ISO string)
            for symbol, block_until in self.cascade_blocks.items():
                if block_until:
                    existing_state['cascade_blocks'][symbol] = block_until.isoformat()
                else:
                    existing_state['cascade_blocks'][symbol] = None

            # Save market trending blocks (boolean)
            for symbol, blocked in self.market_trending_block.items():
                existing_state['market_trending_block'][symbol] = blocked

            # Write atomically
            temp_path = state_path.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(existing_state, f, indent=2, ensure_ascii=False)

            if state_path.exists():
                state_path.unlink()
            temp_path.rename(state_path)

        except Exception as e:
            print(f"[WARN] Failed to save blocking state: {e}")

    def load_blocking_state(self, state_file: str = "data/recovery_state.json") -> bool:
        """
        Load blocking state (cascade_blocks, market_trending_block) from recovery state file.

        Args:
            state_file: Path to state file

        Returns:
            bool: True if state was loaded, False otherwise
        """
        from pathlib import Path
        import json

        try:
            state_path = Path(state_file)

            if not state_path.exists():
                print("[INFO] No saved blocking state found")
                return False

            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # Load cascade blocks (ISO string to datetime)
            if 'cascade_blocks' in state:
                self.cascade_blocks = {}
                for symbol, block_until_str in state['cascade_blocks'].items():
                    if block_until_str:
                        self.cascade_blocks[symbol] = datetime.fromisoformat(block_until_str)
                    else:
                        self.cascade_blocks[symbol] = None
                print(f"[OK] Loaded cascade blocks: {len(self.cascade_blocks)} symbols")

            # Load market trending blocks (boolean)
            if 'market_trending_block' in state:
                self.market_trending_block = state['market_trending_block']
                print(f"[OK] Loaded market trending blocks: {len(self.market_trending_block)} symbols")

            # Check if blocks have expired
            if 'last_block_update' in state:
                last_update = datetime.fromisoformat(state['last_block_update'])
                age = datetime.now() - last_update
                print(f"[INFO] Blocking state age: {age.total_seconds() / 60:.1f} minutes")

                # Auto-expire old cascade blocks (older than 2 hours)
                if age.total_seconds() > 7200:
                    print("[INFO] Blocking state is stale (>2h) - will re-evaluate on startup")
                    self.cascade_blocks = {}
                    self.market_trending_block = {}

            return True

        except Exception as e:
            print(f"[WARN] Failed to load blocking state: {e}")
            return False
