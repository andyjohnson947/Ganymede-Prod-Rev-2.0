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
from portfolio.instruments_config import INSTRUMENTS
from ml_system.ml_integration_manager import MLIntegrationManager
from config.strategy_config import (
    SYMBOLS,
    TIMEFRAME,
    HTF_TIMEFRAMES,
    DATA_REFRESH_INTERVAL,
    MAX_OPEN_POSITIONS,
    MAX_POSITIONS_PER_SYMBOL,
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
    ENABLE_CONFIRMATION_REENTRY,  # Confirmation re-entry add-on
    REENTRY_EXPIRY_HOURS,          # Pending re-entry expiry window
    ENABLE_TIME_EXIT,              # 2-hour no-progress exit
    TIME_EXIT_MINUTES,             # Minutes before closing stale no-PC1 positions
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

        # Market State Alignment Scorer (Phase 5)
        self.alignment_scorers = {}
        try:
            from ml_system.market_state.alignment_scorer import AlignmentScorer
            for sym in ['EURUSD', 'GBPUSD']:
                scorer = AlignmentScorer(sym)
                if scorer.load_models():
                    self.alignment_scorers[sym] = scorer
                    print(f"[ALIGNMENT] Loaded alignment model for {sym}")
            if not self.alignment_scorers:
                print("[ALIGNMENT] No alignment models found — alignment check disabled")
        except Exception as e:
            print(f"[ALIGNMENT] Failed to load alignment scorer: {e}")
            self.alignment_scorers = {}

        self.running = False
        self.last_data_refresh = {}
        self.market_data_cache = {}

        # market_trending_block removed — K+Q handle entry filtering

        # Cascade protection - blocks new trades after cascade close
        self.cascade_blocks = {}  # {symbol: block_until_time}

        # Confirmation re-entry: pending re-entries after BE stop-outs (add-on)
        # {original_ticket: {symbol, direction, entry_price, trigger_price, sl_distance, volume, expiry}}
        self.pending_reentries = {}

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

        # Entry confirmation: pending signals waiting for next-bar rejection
        # {symbol: {'signal': signal_dict, 'signal_bar_time': datetime, 'level_price': float}}
        self.pending_signals = {}
        # Max bars to wait for confirmation (expire after this)
        self.CONFIRMATION_MAX_BARS = 4  # Wait up to 4 H1 bars for rejection candle
        # Cooldown after expired confirmation — prevents re-detecting same signal
        # {symbol: datetime_of_expiry}
        self.expired_signals = {}
        self.SIGNAL_COOLDOWN_BARS = 4  # 4 H1 bars = 4 hours cooldown
        self._last_block_reason = {}  # {symbol: str} — print block only on state change
        self.SIGNAL_STATE_FILE = "data/signal_state.json"  # Persistence file

        # SQLite trade database for persistent logging
        self.trade_db = None
        try:
            from ml_system.trade_database import TradeDatabase
            self.trade_db = TradeDatabase()
        except Exception as e:
            print(f"[DB WARN] Failed to initialize trade database: {e}")

        if self.debug:
            print("[DEBUG] ConfluenceStrategy.__init__() completed successfully", flush=True)

    def _db_log_exit(self, ticket: int, exit_price: float, pnl: float, pnl_pips: float, exit_reason: str):
        """Helper to log trade exit to DB."""
        if self.trade_db:
            try:
                self.trade_db.update_trade_exit(
                    ticket=ticket,
                    exit_time=datetime.utcnow().isoformat(),
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pips=pnl_pips,
                    exit_reason=exit_reason,
                )
            except Exception as e:
                print(f"[DB WARN] Exit logging failed for {ticket}: {e}")

    def _q_learn_on_exit(self, ticket: int, symbol: str, profit: float):
        """Online Q-table learning: update Q-table when a trade closes.

        Called at every exit point. Reads q_state from tracked_pos (stored at entry),
        calculates reward from profit, and updates the Q-table in place.
        Also logs MAE (Maximum Adverse Excursion) for SL optimization.
        """
        tracked_pos = self.recovery_manager.tracked_positions.get(ticket)
        if not tracked_pos:
            return

        # Log MAE at exit for SL optimization analysis
        mae_pips = tracked_pos.get('mae_pips', 0.0)
        result = "WIN" if profit > 0 else "LOSS"
        print(f"[MAE] #{ticket} {symbol} {result} ${profit:.2f} | worst drawdown: {mae_pips:.1f} pips")

        q_state = tracked_pos.get('q_state')
        if not q_state:
            return
        if not hasattr(self, 'signal_detector') or not self.signal_detector:
            return

        reward = 1.0 if profit > 0 else -1.0
        self.signal_detector.update_q_table_online(symbol, q_state, reward)

    def _db_log_recovery(self, original_ticket: int, recovery_type: str, recovery_ticket: int,
                         level: int, entry_price: float, volume: float, pips_underwater: float):
        """Helper to log recovery event to DB."""
        if self.trade_db:
            try:
                self.trade_db.log_recovery(
                    original_ticket=original_ticket,
                    recovery_type=recovery_type,
                    recovery_ticket=recovery_ticket,
                    level=level,
                    entry_time=datetime.utcnow().isoformat(),
                    entry_price=entry_price,
                    volume=volume,
                    pips_underwater=pips_underwater,
                )
            except Exception as e:
                print(f"[DB WARN] Recovery logging failed: {e}")

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

        # LOAD SIGNAL STATE: Restore pending signals & cooldowns from previous session
        self.load_signal_state()

        # K+Q handle entry filtering — no ADX/M15/alignment blocking needed

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

        # Save Q-tables with online learning updates
        if hasattr(self, 'signal_detector') and self.signal_detector:
            self.signal_detector.save_q_tables()

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

                # 3. Check pending signal confirmations (entry confirmation system)
                if symbol in self.pending_signals:
                    self._check_pending_confirmation(symbol)

                # 4. Look for new signals (only if no pending signal and no cooldown)
                if self._can_open_new_position(symbol) and symbol not in self.pending_signals:
                    # Check cooldown from expired confirmations
                    if symbol in self.expired_signals:
                        elapsed = (datetime.utcnow() - self.expired_signals[symbol]).total_seconds()
                        cooldown_seconds = self.SIGNAL_COOLDOWN_BARS * 3600  # H1 bars
                        if elapsed < cooldown_seconds:
                            hours_left = (cooldown_seconds - elapsed) / 3600
                            # Print once per hour to avoid log spam
                            if int(elapsed) % 3600 < 60:
                                print(f"   [COOLDOWN] {symbol}: {hours_left:.1f}h remaining after expired confirmation")
                            continue
                        else:
                            del self.expired_signals[symbol]
                            self.save_signal_state()  # Persist cooldown clear

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

        # K+Q handle entry filtering — no market state blocking needed

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
                        position_type=pos['type'],  # Already 'buy'/'sell' string from get_positions()
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

            # Emergency cascade threshold: -$240 total unrealized loss (4 positions × $60 each)
            if total_unrealized <= -240:
                print(f"\n[ALERT] CASCADE PROTECTION TRIGGERED [ALERT]")
                print(f"   Total unrealized loss: ${total_unrealized:.2f}")
                print(f"   Threshold: -$240.00")
                print(f"   Open positions: {len(all_positions)}")
                print(f"   ACTION: Closing UNDERWATER positions only (protecting profitable positions with active PC)")
                print()

                logger.critical(f"CASCADE PROTECTION: Total unrealized ${total_unrealized:.2f} exceeds -$240 threshold")

                # Close only underwater positions — let profitable ones run their PC/BE/TP cycle
                closed_count = 0
                for pos in all_positions:
                    if pos['profit'] >= 0:
                        print(f"   [KEPT] #{pos['ticket']} {pos['symbol']} +${pos['profit']:.2f} (profitable, PC active)")
                        continue
                    if self.mt5.close_position(pos['ticket']):
                        closed_count += 1
                        self._q_learn_on_exit(pos['ticket'], pos['symbol'], pos.get('profit', 0))
                        self.recovery_manager.untrack_position(pos['ticket'])
                        self.stats['trades_closed'] += 1
                        self._db_log_exit(pos['ticket'], pos.get('price_current', 0), pos.get('profit', 0), 0, 'cascade_close')

                print(f"[CASCADE] Closed {closed_count}/{len(all_positions)} positions")
                print(f"[CASCADE] Blocking new trades for 60 minutes")
                print()

                # Block trading for 60 minutes to avoid re-entering during strong trend
                self.trade_block_until = get_current_time() + timedelta(minutes=60)

                # Cancel any pending RE-ENTRY STOP orders for this symbol (trend invalidates them)
                self._cancel_reentry_orders(symbol)

                # Save recovery state
                self.recovery_manager.save_state()

                # Exit early - no position management needed
                return

        # Window close REMOVED — hardware SL handles losses, PC/trail handles profits

        # CONFIRMATION RE-ENTRY: Detect BE stop-outs (PC1 hit, PC2 not hit, position gone)
        if ENABLE_CONFIRMATION_REENTRY:
            current_tickets = {p['ticket'] for p in positions}
            for ticket, tracked_pos in list(self.recovery_manager.tracked_positions.items()):
                if tracked_pos.get('symbol') != symbol:
                    continue
                if not tracked_pos.get('partial_1_closed', False):
                    continue  # PC1 never hit — not a BE stop-out
                if tracked_pos.get('partial_2_closed', False):
                    continue  # PC2 hit — full run, no re-entry needed
                if tracked_pos.get('reentry_used', False):
                    continue  # Already registered re-entry for this position
                if ticket in current_tickets:
                    continue  # Position still open
                # Position had PC1, no PC2, and just disappeared — BE stop-out detected
                self._register_pending_reentry(ticket, tracked_pos)
                tracked_pos['reentry_used'] = True
                self.recovery_manager.save_state()  # Persist immediately so restart won't re-register

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
            # For 5-digit brokers: point=0.00001, pip=0.0001 = point*10
            point = symbol_info.get('point', 0.00001) if symbol_info else 0.00001
            digits = symbol_info.get('digits', 5) if symbol_info else 5
            pip_value = point * 10 if digits in (3, 5) else point  # Convert point to pip

            # MAE tracking: record worst drawdown (pips) per position every loop
            if ticket in self.recovery_manager.tracked_positions:
                tracked_pos = self.recovery_manager.tracked_positions[ticket]
                entry_price = position['price_open']
                pos_type = position['type']
                if pos_type == 'buy':
                    live_pips = (current_price - entry_price) / pip_value
                else:
                    live_pips = (entry_price - current_price) / pip_value
                prev_mae = tracked_pos.get('mae_pips', 0.0)
                if live_pips < prev_mae:
                    tracked_pos['mae_pips'] = live_pips

            # 2-HOUR NO-PROGRESS EXIT: close any position that has never hit PC1
            # after TIME_EXIT_MINUTES. Data shows zero recoveries past this point.
            # Only applies to original VWAP/signal positions, not recovery orders.
            if ENABLE_TIME_EXIT and not is_recovery_order:
                tracked_pos_te = self.recovery_manager.tracked_positions.get(ticket)
                if tracked_pos_te and not tracked_pos_te.get('partial_1_closed', False):
                    open_time = tracked_pos_te.get('open_time')
                    if open_time:
                        age_mins = (get_current_time() - open_time).total_seconds() / 60
                        if age_mins >= TIME_EXIT_MINUTES:
                            entry_p = position['price_open']
                            cur_p   = position['price_current']
                            pos_dir = position['type']
                            pips_now = (cur_p - entry_p) / pip_value if pos_dir == 'buy' else (entry_p - cur_p) / pip_value
                            # Only exit if in drawdown — positive positions are still making
                            # progress toward PC1 and should be left to run
                            if pips_now >= 0:
                                continue
                            print(f"\n[TIME EXIT] #{ticket} {symbol} — {age_mins:.0f}min open, "
                                  f"no PC1, {pips_now:+.1f}p — closing")
                            if self.mt5.close_position(ticket, comment=f"TIME-EXIT-{TIME_EXIT_MINUTES}min"):
                                self.stats['trades_closed'] += 1
                                self._q_learn_on_exit(ticket, symbol, position['profit'])
                                self.recovery_manager.untrack_position(ticket)
                                self._db_log_exit(ticket, cur_p, position['profit'], 0, 'time_exit')
                                print(f"[TIME EXIT] Closed #{ticket} @ {cur_p:.5f} | P&L: ${position['profit']:.2f}")
                            continue  # Skip remaining checks for this position

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
                # Get instrument-specific PC1/PC2 levels (INSTRUMENTS imported at top of file)
                instrument_config = INSTRUMENTS.get(symbol, {})
                tp_settings = instrument_config.get('take_profit', {})

                pc1_pips = tp_settings.get('partial_1_pips', 10)
                pc2_pips = tp_settings.get('partial_2_pips', 20)
                pc1_percent = tp_settings.get('partial_1_percent', 0.50)
                pc2_percent = tp_settings.get('partial_2_percent', 0.25)

                # Calculate current profit in pips
                entry_price = position['price_open']
                current_price = position['price_current']
                pos_type = position['type']  # Already 'buy'/'sell' string from get_positions()

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

                # PC CHECK LOGGING
                print(f"[PC CHECK] #{ticket} {symbol}: +{profit_pips:.1f}p | PC1={pc1_pips}p closed={pc1_closed} | PC2={pc2_pips}p closed={pc2_closed} | pip_val={pip_value}")

                # PC1 CHECK: Close 50% at 1R pips + SL→BE
                if not pc1_closed and profit_pips >= pc1_pips:
                    close_volume = round(position['volume'] * pc1_percent, 2)

                    if close_volume > 0 and close_volume < position['volume']:
                        short_ticket = str(ticket)[-5:]
                        pc1_comment = f"PC1-50%@{profit_pips:.0f}pips-{short_ticket}"

                        if self.mt5.close_partial_position(ticket, close_volume, comment=pc1_comment):
                            print(f"[PC1] {ticket} - Closed 50% @ +{profit_pips:.1f} pips = ${close_volume * profit_pips * 10:.2f}")
                            tracked_pos['partial_1_closed'] = True

                            # PERSIST STATE immediately so re-entry detection survives bot restart
                            self.recovery_manager.save_state()

                            # MOVE HARDWARE SL TO BREAKEVEN after PC1
                            if self.mt5.modify_position(ticket, sl=entry_price):
                                print(f"[PC1] Hardware SL -> breakeven @ {entry_price:.5f}")

                            # DISABLE VWAP EXITS after PC1
                            print(f"[PC1] VWAP exits disabled for {ticket}")

                # PC2 CHECK: Close 25% of original at 2R pips, trail remaining 25%
                elif pc1_closed and not pc2_closed and profit_pips >= pc2_pips:
                    # Use INITIAL volume for PC2 (current volume is reduced after PC1)
                    # PC2 closes 25% of original, leaving 25% running with trail
                    initial_volume = tracked_pos.get('initial_volume', position['volume'])
                    close_volume = round(initial_volume * pc2_percent, 2)

                    if close_volume > 0 and close_volume < position['volume']:
                        short_ticket = str(ticket)[-5:]
                        pc2_comment = f"PC2-25%@{profit_pips:.0f}pips-{short_ticket}"

                        if self.mt5.close_partial_position(ticket, close_volume, comment=pc2_comment):
                            print(f"[PC2] {ticket} - Closed 25% (75% total) @ +{profit_pips:.1f} pips = ${close_volume * profit_pips * 10:.2f}")
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
                                self._q_learn_on_exit(ticket, symbol, position.get('profit', 0))
                                self.recovery_manager.untrack_position(ticket)
                                self.stats['trades_closed'] += 1
                                self._db_log_exit(ticket, float(current_price), position.get('profit', 0), 0, 'pc2_time_limit')
                            continue

                    # Update trailing stop (moves stop with price as profit increases)
                    # Store old stop for comparison
                    old_stop = tracked_pos.get('trailing_stop_price', 0)

                    # Update trailing stop
                    self.recovery_manager.update_trailing_stop(ticket, current_price)

                    # Check if stop moved and log the update
                    new_stop = tracked_pos.get('trailing_stop_price', 0)
                    if new_stop != old_stop and self.ml_logger:
                        # For 5-digit brokers: point=0.00001, pip=0.0001 = point*10
                        _point = symbol_info.get('point', 0.00001) if symbol_info else 0.00001
                        _digits = symbol_info.get('digits', 5) if symbol_info else 5
                        pip_value = _point * 10 if _digits in (3, 5) else _point

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
                    # Only modify on MT5 when trailing stop actually moved (avoids "No changes" spam)
                    trailing_stop_price = tracked_pos.get('trailing_stop_price')
                    if trailing_stop_price and new_stop != old_stop:
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
                            self._q_learn_on_exit(ticket, symbol, position.get('profit', 0))
                            self.recovery_manager.untrack_position(ticket)
                            self.stats['trades_closed'] += 1
                            self._db_log_exit(ticket, current_price, position.get('profit', 0), 0, 'trailing_stop')
                        continue

            # RECOVERY & EXIT CONDITIONS: For UNDERWATER positions with active recovery
            # SIMPLIFIED EXIT MODEL: Only PC1/PC2/Trail + hardware SL
            # All recovery exits REMOVED:
            #   - Stack drawdown (4x) — REMOVED
            #   - Per-stack stop loss + cascade detection — REMOVED
            #   - Hedge drawdown — REMOVED
            #   - Hedge DCA trigger — REMOVED
            #   - Hedge partial close — REMOVED
            #   - Profit target — REMOVED
            #   - Time limit — REMOVED
            #   - VWAP exit — REMOVED
            #   - Orphaned hedge check — REMOVED
            # Losing trades are handled by hardware SL (-$30) set at entry on MT5
            # Cascade protection (-$120 total) remains as emergency account safety net above

        # CONFIRMATION RE-ENTRY: Check if any pending re-entries should fire
        if ENABLE_CONFIRMATION_REENTRY and self.pending_reentries:
            self._check_pending_reentries(symbol)

    # _check_orphaned_hedges REMOVED — no hedging in simplified model

    # =========================================================================
    # CONFIRMATION RE-ENTRY METHODS (add-on — disabled by default)
    # =========================================================================

    def _register_pending_reentry(self, original_ticket: int, tracked_pos: dict):
        """Register a pending confirmation re-entry after a BE stop-out.

        Called when a position that hit PC1 (but not PC2) disappears from MT5.
        Places an actual MT5 BUY_STOP/SELL_STOP order SL/3 pips beyond BE in the
        original direction — visible in MT5 terminal and requires no bot monitoring.
        The pending order auto-executes when price resumes, then tracked for PC1/PC2/trail.
        """
        from config.strategy_config import MAX_LOSS_PER_POSITION, MAX_POSITIONS_PER_SYMBOL

        symbol = tracked_pos.get('symbol', '')
        direction = tracked_pos.get('type', '')
        entry_price = tracked_pos.get('entry_price', 0.0)
        volume = tracked_pos.get('initial_volume', tracked_pos.get('volume', 0.16))

        if not symbol or not direction or not entry_price:
            print(f"[RE-ENTRY] Skipping #{original_ticket} — missing symbol/direction/entry")
            return

        # Don't place re-entry if already at max positions for this symbol
        existing = self.mt5.get_positions(symbol)
        if len(existing) >= MAX_POSITIONS_PER_SYMBOL:
            print(f"[RE-ENTRY] Skipping #{original_ticket} — {symbol} at max positions ({len(existing)}/{MAX_POSITIONS_PER_SYMBOL})")
            return

        # Don't place re-entry if cascade block is active
        if symbol in self.cascade_blocks and get_current_time() < self.cascade_blocks[symbol]:
            print(f"[RE-ENTRY] Skipping #{original_ticket} — {symbol} cascade blocked")
            return

        # Calculate SL distance (same formula as _execute_signal)
        symbol_info = self.mt5.get_symbol_info(symbol)
        point = symbol_info.get('point', 0.00001) if symbol_info else 0.00001
        digits = symbol_info.get('digits', 5) if symbol_info else 5
        pip_val = point * 10 if digits in (3, 5) else point
        pip_value_dollar = 10.0 * volume
        sl_pips = MAX_LOSS_PER_POSITION / pip_value_dollar if pip_value_dollar > 0 else 37.5
        sl_distance = sl_pips * pip_val

        # Trigger = BE price ± SL/3 pips in original direction
        # SELL: price must drop SL/3 below BE → SELL_STOP below entry price
        # BUY:  price must rise SL/3 above BE → BUY_STOP above entry price
        trigger_pips = sl_pips / 3.0
        trigger_distance = trigger_pips * pip_val

        if direction == 'buy':
            trigger_price = round(entry_price + trigger_distance, digits)
            pending_sl = round(trigger_price - sl_distance, digits)  # SL below trigger for BUY
        else:
            trigger_price = round(entry_price - trigger_distance, digits)
            pending_sl = round(trigger_price + sl_distance, digits)  # SL above trigger for SELL

        expiry = get_current_time() + timedelta(hours=REENTRY_EXPIRY_HOURS)

        # Place actual MT5 STOP pending order (visible in terminal, no bot monitoring needed)
        order_ticket = self.mt5.place_order(
            symbol=symbol,
            order_type=direction,
            volume=volume,
            price=trigger_price,
            sl=pending_sl,
            comment=f"RE-ENTRY:{original_ticket}",
            order_mode='stop',
            expiry=expiry,
        )

        if order_ticket:
            print(f"[RE-ENTRY] ✅ STOP order placed | #{original_ticket} -> pending #{order_ticket}"
                  f" | {symbol} {direction.upper()}_STOP @ {trigger_price:.5f}"
                  f" ({trigger_pips:.1f}p from BE {entry_price:.5f})"
                  f" | SL: {pending_sl:.5f} | Expires: {expiry.strftime('%H:%M UTC')}")
        else:
            print(f"[RE-ENTRY] ❌ Failed to place STOP order for #{original_ticket} {symbol} {direction.upper()}"
                  f" @ {trigger_price:.5f}")

    def _cancel_reentry_orders(self, symbol: str):
        """Cancel all pending RE-ENTRY STOP orders for a symbol (e.g. on cascade protection)."""
        import MetaTrader5 as mt5_lib
        orders = mt5_lib.orders_get(symbol=symbol)
        if not orders:
            return
        cancelled = 0
        for order in orders:
            if str(order.comment).startswith('RE-ENTRY:'):
                req = {
                    "action": mt5_lib.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                }
                res = mt5_lib.order_send(req)
                if res and res.retcode == mt5_lib.TRADE_RETCODE_DONE:
                    cancelled += 1
                    print(f"[RE-ENTRY] Cancelled pending STOP #{order.ticket} ({symbol}) — cascade block")
                else:
                    err = res.comment if res else mt5_lib.last_error()
                    print(f"[RE-ENTRY] Failed to cancel #{order.ticket}: {err}")
        if cancelled:
            print(f"[RE-ENTRY] Cancelled {cancelled} pending STOP order(s) for {symbol}")

    def _check_pending_reentries(self, symbol: str):
        """No-op: re-entry pending orders are now placed directly as MT5 STOP orders.
        MT5 monitors the trigger price and executes automatically — no bot loop needed.
        Legacy in-memory entries are cleared here on any remaining state."""
        # Clear any leftover in-memory entries (migration cleanup)
        to_remove = [t for t, r in self.pending_reentries.items() if r.get('symbol') == symbol]
        for t in to_remove:
            self.pending_reentries.pop(t, None)

    def _execute_reentry(self, orig_ticket: int, reentry: dict):
        """No-op: replaced by MT5 STOP pending orders placed in _register_pending_reentry."""
        pass

    def _check_pending_confirmation(self, symbol: str):
        """
        Check if a pending MR signal gets confirmed by the current bar.

        Confirmation = the bar since the signal shows rejection from the level:
          - For BUY: bar has a lower wick (low dipped toward level) but closed above open (bullish)
          - For SELL: bar has an upper wick (high pushed toward level) but closed below open (bearish)

        If no confirmation after CONFIRMATION_MAX_BARS, expire the signal.
        """
        pending = self.pending_signals.get(symbol)
        if not pending:
            return

        signal = pending['signal']
        direction = signal['direction']

        # Get current H1 bar
        cache = self.market_data_cache.get(symbol)
        if not cache:
            return
        h1_data = cache.get('h1')
        if h1_data is None or len(h1_data) < 2:
            return

        current_bar = h1_data.iloc[-1]
        current_bar_time = h1_data.index[-1]

        # Only process on NEW H1 bar (skip if same bar as last check)
        if current_bar_time == pending.get('last_checked_bar'):
            return  # Same bar, nothing new to check

        pending['last_checked_bar'] = current_bar_time
        pending['bars_waited'] += 1

        bar_open = current_bar['open']
        bar_close = current_bar['close']
        bar_high = current_bar['high']
        bar_low = current_bar['low']
        bar_range = bar_high - bar_low

        if bar_range == 0:
            return  # Flat bar, wait

        # Check confirmation
        confirmed = False

        if direction == 'buy':
            # BUY confirmation: bar closed bullish (close > open)
            # AND has meaningful lower wick (tested support and bounced)
            lower_wick = min(bar_open, bar_close) - bar_low
            wick_ratio = lower_wick / bar_range
            is_bullish = bar_close > bar_open
            confirmed = is_bullish and wick_ratio >= 0.25

        elif direction == 'sell':
            # SELL confirmation: bar closed bearish (close < open)
            # AND has meaningful upper wick (tested resistance and rejected)
            upper_wick = bar_high - max(bar_open, bar_close)
            wick_ratio = upper_wick / bar_range
            is_bearish = bar_close < bar_open
            confirmed = is_bearish and wick_ratio >= 0.25

        if confirmed:
            # Update price to current (enter at confirmation bar close, not signal bar)
            signal['price'] = bar_close
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            print(f"[CONFIRM] [{ts}] {symbol} {direction.upper()} CONFIRMED — rejection candle detected")
            print(f"   O={bar_open:.5f} H={bar_high:.5f} L={bar_low:.5f} C={bar_close:.5f}")
            del self.pending_signals[symbol]
            self.save_signal_state()  # Persist state change
            self._execute_signal(signal)
            return

        # Check expiry
        if pending['bars_waited'] >= self.CONFIRMATION_MAX_BARS:
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            print(f"[CONFIRM] [{ts}] {symbol} {direction.upper()} EXPIRED — no rejection in {pending['bars_waited']} bars")
            self.expired_signals[symbol] = datetime.utcnow()  # Start cooldown
            print(f"   [COOLDOWN] {symbol}: {self.SIGNAL_COOLDOWN_BARS}h cooldown started")
            del self.pending_signals[symbol]
            self.save_signal_state()  # Persist state change
            return

        ts = datetime.utcnow().strftime("%H:%M UTC")
        print(f"[CONFIRM] [{ts}] {symbol} {direction.upper()} waiting... bar {pending['bars_waited']}/{self.CONFIRMATION_MAX_BARS}")

    def _check_for_signals(self, symbol: str):
        """Check for new entry signals"""
        if symbol not in self.market_data_cache:
            return

        # Track blocking reasons for visibility
        blocking_reasons = []

        # K+Q handle entry filtering — no ADX blocking

        # Check if symbol is tradeable based on portfolio trading windows (bypass in test mode)
        if not self.test_mode and ENABLE_TIME_FILTERS and not self.portfolio_manager.is_symbol_tradeable(symbol):
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

        # Show blocking summary only on state change (avoid log spam)
        block_key = ', '.join(blocking_reasons)
        if blocking_reasons and block_key != self._last_block_reason.get(symbol):
            print(f"   [BLOCKED] {symbol}: {block_key}", flush=True)
            self._last_block_reason[symbol] = block_key
        elif not blocking_reasons and symbol in self._last_block_reason:
            print(f"   [UNBLOCKED] {symbol}: Trading resumed", flush=True)
            del self._last_block_reason[symbol]

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

        # ── SMC Structural Breakout ───────────────────────────────────
        # Always update the state machine (tracks compression/liquidity even outside BO hours)
        if self.breakout_strategy:
            try:
                self.breakout_strategy.update_state(
                    symbol=symbol,
                    h1_data=h1_data,
                    d1_data=d1_data,
                    current_bar_index=len(h1_data) - 1,
                )
            except Exception as e:
                if self.debug:
                    print(f"   [WARN] {symbol}: Breakout state machine error: {e}", flush=True)

        # Try structural breakout entry (if MR found nothing and BO window is active)
        if signal is None and can_trade_bo and self.breakout_strategy:
            bo_signal = self.breakout_strategy.get_structural_entry(symbol)
            if bo_signal:
                signal = {
                    'symbol': symbol,
                    'direction': bo_signal['direction'],
                    'price': bo_signal['entry_price'],
                    'strategy_type': 'breakout',
                    'confluence_score': bo_signal.get('score', 5),
                    'factors': bo_signal.get('factors', []),
                    'breakout_details': bo_signal,
                }
                if self.debug:
                    print(f"   [OK] {symbol}: Structural breakout signal — "
                          f"{signal['direction'].upper()} (stage 5)", flush=True)
            else:
                if self.debug:
                    state_info = self.breakout_strategy.get_state_summary(symbol)
                    print(f"   [SKIP] {symbol}: No structural breakout — "
                          f"stage {state_info.get('stage', 0)} ({state_info.get('stage_name', 'IDLE')})", flush=True)

        if signal is None:
            if self.debug:
                print(f"   [FAIL] {symbol}: No valid signals (need confluence >= {MIN_CONFLUENCE_SCORE})", flush=True)
            return

        # Signal detected!
        self.stats['signals_detected'] += 1

        # DB: Log accepted signal
        if self.trade_db:
            try:
                self.trade_db.log_signal(
                    symbol=symbol,
                    timestamp=datetime.utcnow().isoformat(),
                    direction=signal.get('direction', 'unknown'),
                    confluence_score=signal.get('confluence_score', 0),
                    factors=signal.get('factors', []),
                    accepted=True,
                    q_trade=signal.get('q_trade'),
                    q_skip=signal.get('q_skip'),
                )
            except Exception as e:
                print(f"[DB WARN] Signal logging failed: {e}")

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
            print(f"   Structural Breakout: {details.get('type', 'unknown').upper()}")
            print(f"   Direction: {signal['direction'].upper()}")
            print(f"   Range: {details.get('range_low', 0):.5f} - {details.get('range_high', 0):.5f} "
                  f"({details.get('range_size_pips', 0):.0f} pips)")
            print(f"   Sweep: {details.get('sweep_direction', '?')} to {details.get('sweep_extreme', 0):.5f}")
            print(f"   BOS level: {details.get('bos_level', 0):.5f}")
            print(f"   Retest: {details.get('retest_type', '?')} at {details.get('entry_price', 0):.5f}")
            print(f"   Sequence: {details.get('total_bars', 0)} bars")
            if details.get('factors'):
                for factor in details['factors']:
                    print(f"     • {factor}")

        print()

        # ── Market State Alignment Check ────────────────────────────
        # Score how well current environment aligns with proposed trade
        scorer = self.alignment_scorers.get(symbol)
        if scorer:
            try:
                strat_type = 'MR' if signal.get('strategy_type') == 'mean_reversion' else 'BO'
                alignment = scorer.score_from_live_data(
                    h1_data=h1_data,
                    d1_data=d1_data,
                    w1_data=w1_data,
                    proposed_strategy=strat_type,
                    proposed_direction=signal['direction'],
                )
                signal['alignment'] = alignment
                score = alignment.get('alignment_score', 0)
                cluster_id = alignment.get('cluster_id', -1)
                cluster_strat = alignment.get('cluster_strategy', '?')

                print(f"[ALIGNMENT] Score: {score:.3f} | Cluster: C{cluster_id} ({cluster_strat}) | "
                      f"Threshold: {alignment.get('threshold', 0.55)}")

                if not alignment.get('should_trade', False):
                    # DISABLED: K+Q are sufficient. Log only, don't block.
                    print(f"[ALIGNMENT] Info: score {score:.3f} < threshold (not blocking)")
                    # return  # Skip trade — disabled, K+Q handle entry filtering
                else:
                    print(f"[ALIGNMENT] PASSED: Environment supports {strat_type} {signal['direction']}")
            except Exception as e:
                print(f"[ALIGNMENT] Error scoring alignment: {e}")
                # Continue with trade on scoring error (fail-open)

        # Entry confirmation: queue MR signals, execute BO signals directly
        if signal.get('strategy_type') == 'breakout':
            # Breakout has its own confirmation (retest stage) — execute now
            self._execute_signal(signal)
            if self.breakout_strategy:
                self.breakout_strategy.on_trade_executed(signal['symbol'])
        else:
            # MR signals: queue as pending, wait for next-bar confirmation
            bar_time = h1_data.index[-1] if hasattr(h1_data.index[-1], 'hour') else datetime.utcnow()
            self.pending_signals[symbol] = {
                'signal': signal,
                'signal_bar_time': bar_time,
                'last_checked_bar': bar_time,  # Track bar changes (not 60s scans)
                'bars_waited': 0,
            }
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            print(f"[CONFIRM] [{ts}] {symbol} {signal['direction'].upper()} queued — waiting for next-bar rejection")
            self.save_signal_state()  # Persist across restarts

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

        # K+Q handle entry filtering — no market trending block

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

        # HARDWARE STOP LOSS: Always set -$30 SL on MT5 at entry (survives crashes)
        from config.strategy_config import MAX_LOSS_PER_POSITION

        current_adx = signal.get('adx', 0.0)
        hard_sl = None

        # Calculate $30 hard SL in pips
        # pip_value_per_lot = $10 for EURUSD/GBPUSD (standard lots)
        # SL_pips = max_loss / (pip_value_per_lot * volume)
        _point = symbol_info.get('point', 0.00001) if symbol_info else 0.00001
        _digits = symbol_info.get('digits', 5) if symbol_info else 5
        pip_val = _point * 10 if _digits in (3, 5) else _point
        pip_value_dollar = 10.0 * volume  # $10/pip/lot for EUR/GBP pairs
        sl_pips = MAX_LOSS_PER_POSITION / pip_value_dollar if pip_value_dollar > 0 else 20
        sl_distance = sl_pips * pip_val

        tick = self.mt5.get_symbol_tick(symbol)
        if tick:
            current_price = tick.get('bid') if direction == 'sell' else tick.get('ask')
            if direction == 'buy':
                hard_sl = current_price - sl_distance
            else:
                hard_sl = current_price + sl_distance
            print(f"[SL] Hardware SL set: -{sl_pips:.0f} pips = -${MAX_LOSS_PER_POSITION:.0f} @ {hard_sl:.5f}")

        # Place order(s) - open multiple trades if INITIAL_TRADE_COUNT > 1
        # Include strategy type in comment for ML analysis
        # VWAP = Mean reversion to VWAP/levels, BREAKOUT = Momentum through levels
        strategy_label = "VWAP" if signal.get('strategy_type') == 'mean_reversion' else "BREAKOUT"
        comment = f"{strategy_label}:C{signal['confluence_score']}"
        trades_opened = 0

        for i in range(INITIAL_TRADE_COUNT):
            trade_comment = comment if INITIAL_TRADE_COUNT == 1 else f"{comment} #{i+1}"

            # Small delay between batch orders to avoid MT5 trade context busy (10018)
            if i > 0:
                time.sleep(0.5)

            ticket = self.mt5.place_order(
                symbol=symbol,
                order_type=direction,
                volume=volume,
                sl=hard_sl,  # Always -$30 hardware SL (survives crashes)
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

                # DB: Log trade entry with full context
                if self.trade_db:
                    try:
                        strategy_type = signal.get('strategy_type', 'mr')
                        self.trade_db.log_trade(
                            ticket=ticket, symbol=symbol, direction=direction,
                            strategy=strategy_type,
                            entry_time=datetime.utcnow().isoformat(),
                            entry_price=actual_entry_price, volume=volume,
                            confluence_score=signal.get('confluence_score', 0),
                            q_trade=signal.get('q_trade'),
                            q_skip=signal.get('q_skip'),
                        )
                        self.trade_db.log_factors(ticket, signal.get('factors', []))
                        # Log market conditions
                        trend = signal.get('trend_filter', {})
                        vwap_sigs = signal.get('vwap_signals', {})
                        self.trade_db.log_market_conditions(
                            ticket=ticket,
                            hour_utc=datetime.utcnow().hour,
                            day_of_week=datetime.utcnow().weekday(),
                            adx=trend.get('adx', 0),
                            plus_di=trend.get('plus_di', 0),
                            minus_di=trend.get('minus_di', 0),
                            vwap_value=vwap_sigs.get('vwap', 0),
                            vwap_distance_pct=vwap_sigs.get('distance_pct', 0),
                            vwap_direction=vwap_sigs.get('direction', ''),
                        )
                    except Exception as e:
                        print(f"[DB WARN] Trade entry logging failed: {e}")

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

                # Store Q-state for online learning (used when trade closes)
                tracked_pos = self.recovery_manager.tracked_positions.get(ticket)
                if tracked_pos:
                    tracked_pos['q_state'] = signal.get('q_state')
                    tracked_pos['q_visits'] = signal.get('q_visits', 0)

                # ML LOGGING: Log trade entry with enhanced data
                try:
                    # Get current spread
                    tick = self.mt5.get_symbol_tick(symbol)
                    spread_pips = 0
                    # For 5-digit brokers: point=0.00001, pip=0.0001 = point*10
                    _point = symbol_info.get('point', 0.00001) if symbol_info else 0.00001
                    _digits = symbol_info.get('digits', 5) if symbol_info else 5
                    _pip_val = _point * 10 if _digits in (3, 5) else _point
                    if tick:
                        spread = tick.get('ask', 0) - tick.get('bid', 0)
                        spread_pips = spread / _pip_val

                    # Get ATR from market data
                    atr_pips = 0
                    if symbol in self.market_data_cache:
                        h1_data = self.market_data_cache[symbol]['h1']
                        if 'atr' in h1_data.columns and len(h1_data) > 0:
                            atr = h1_data.iloc[-1]['atr']
                            atr_pips = atr / _pip_val

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
        # Q-learn on the original position (recovery children don't have q_state)
        self._q_learn_on_exit(original_ticket,
                              self.recovery_manager.tracked_positions.get(original_ticket, {}).get('symbol', ''),
                              final_pnl or 0)
        for ticket in stack_tickets:
            if self.mt5.close_position(ticket):
                closed_count += 1
                self.stats['trades_closed'] += 1
                self._db_log_exit(ticket, 0, 0, 0, f'recovery_stack_close: {reason}')
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
                    # For 5-digit brokers: point=0.00001, pip=0.0001 = point*10
                    _point = symbol_info.get('point', 0.00001) if symbol_info else 0.00001
                    _digits = symbol_info.get('digits', 5) if symbol_info else 5
                    pip_value = _point * 10 if _digits in (3, 5) else _point

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
                                            # For 5-digit brokers: point=0.00001, pip=0.0001 = point*10
                                            _point = symbol_info.get('point', 0.00001) if symbol_info else 0.00001
                                            _digits = symbol_info.get('digits', 5) if symbol_info else 5
                                            pip_value = _point * 10 if _digits in (3, 5) else _point

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

    # _execute_hedge_partial_close REMOVED — no hedging in simplified model

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
        Save blocking state (cascade_blocks) to recovery state file.
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
            existing_state['last_block_update'] = datetime.now().isoformat()

            # Convert cascade blocks (datetime to ISO string)
            for symbol, block_until in self.cascade_blocks.items():
                if block_until:
                    existing_state['cascade_blocks'][symbol] = block_until.isoformat()
                else:
                    existing_state['cascade_blocks'][symbol] = None

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
        Load blocking state (cascade_blocks) from recovery state file.

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

            # Check if blocks have expired
            if 'last_block_update' in state:
                last_update = datetime.fromisoformat(state['last_block_update'])
                age = datetime.now() - last_update

                # Auto-expire old cascade blocks (older than 2 hours)
                if age.total_seconds() > 7200:
                    print("[INFO] Cascade blocks stale (>2h) - cleared")
                    self.cascade_blocks = {}

            return True

        except Exception as e:
            print(f"[WARN] Failed to load blocking state: {e}")
            return False

    def save_signal_state(self):
        """
        Persist pending_signals and expired_signals (cooldowns) to disk.
        Survives bot restarts/crashes. Uses atomic write (temp + rename).
        """
        from pathlib import Path
        import json

        try:
            state_path = Path(self.SIGNAL_STATE_FILE)
            state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'saved_at': datetime.utcnow().isoformat(),
                'pending_signals': {},
                'expired_signals': {},
            }

            def _sanitize(obj):
                """Convert numpy/pandas types to JSON-safe Python types."""
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                if hasattr(obj, 'item'):  # numpy int64, float64 etc
                    return obj.item()
                if isinstance(obj, dict):
                    return {str(k): _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                return obj

            # Serialize pending signals
            for symbol, pending in self.pending_signals.items():
                signal_copy = _sanitize(pending['signal'])

                # Convert bar timestamps
                bar_time = pending.get('signal_bar_time')
                last_bar = pending.get('last_checked_bar')

                state['pending_signals'][symbol] = {
                    'signal': signal_copy,
                    'signal_bar_time': bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time),
                    'last_checked_bar': last_bar.isoformat() if hasattr(last_bar, 'isoformat') else str(last_bar),
                    'bars_waited': pending.get('bars_waited', 0),
                }

            # Serialize expired signals (cooldowns)
            for symbol, expiry_time in self.expired_signals.items():
                state['expired_signals'][symbol] = expiry_time.isoformat() if hasattr(expiry_time, 'isoformat') else str(expiry_time)

            # Atomic write
            temp_path = state_path.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)

            if state_path.exists():
                state_path.unlink()
            temp_path.rename(state_path)

        except Exception as e:
            print(f"[WARN] Failed to save signal state: {e}")

    def load_signal_state(self):
        """
        Restore pending_signals and expired_signals from disk on startup.
        Validates that signals/cooldowns haven't expired while bot was down.
        """
        from pathlib import Path
        import json

        try:
            state_path = Path(self.SIGNAL_STATE_FILE)

            if not state_path.exists():
                print("[LOAD] No saved signal state found")
                return

            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            now = datetime.utcnow()
            restored_pending = 0
            restored_cooldowns = 0

            # Restore pending signals
            for symbol, data in state.get('pending_signals', {}).items():
                bars_waited = data.get('bars_waited', 0)

                # Parse signal_bar_time to check age
                try:
                    signal_bar_time = datetime.fromisoformat(data['signal_bar_time'])
                except (ValueError, KeyError):
                    signal_bar_time = now

                age_seconds = (now - signal_bar_time).total_seconds()
                max_age = self.CONFIRMATION_MAX_BARS * 3600  # e.g. 4 bars * 1h

                # Discard if already expired or too old
                if bars_waited >= self.CONFIRMATION_MAX_BARS:
                    print(f"[LOAD] Discarding {symbol} pending signal — already expired ({bars_waited}/{self.CONFIRMATION_MAX_BARS} bars)")
                    continue
                if age_seconds > max_age:
                    hours_old = age_seconds / 3600
                    print(f"[LOAD] Discarding {symbol} pending signal — too old ({hours_old:.1f}h > {self.CONFIRMATION_MAX_BARS}h)")
                    continue

                # Reconstruct the signal
                signal = data['signal']
                # Restore timestamp if present
                if 'timestamp' in signal and isinstance(signal['timestamp'], str):
                    try:
                        signal['timestamp'] = datetime.fromisoformat(signal['timestamp'])
                    except ValueError:
                        signal['timestamp'] = now

                # Parse bar timestamps
                try:
                    bar_time = datetime.fromisoformat(data['signal_bar_time'])
                except (ValueError, KeyError):
                    bar_time = now
                try:
                    last_bar = datetime.fromisoformat(data['last_checked_bar'])
                except (ValueError, KeyError):
                    last_bar = bar_time

                self.pending_signals[symbol] = {
                    'signal': signal,
                    'signal_bar_time': bar_time,
                    'last_checked_bar': last_bar,
                    'bars_waited': bars_waited,
                }
                direction = signal.get('direction', '?').upper()
                print(f"[LOAD] Restored pending signal: {symbol} {direction} bar {bars_waited}/{self.CONFIRMATION_MAX_BARS}")
                restored_pending += 1

            # Restore expired signals (cooldowns)
            for symbol, expiry_str in state.get('expired_signals', {}).items():
                try:
                    expiry_time = datetime.fromisoformat(expiry_str)
                except ValueError:
                    continue

                elapsed = (now - expiry_time).total_seconds()
                cooldown_seconds = self.SIGNAL_COOLDOWN_BARS * 3600

                if elapsed >= cooldown_seconds:
                    print(f"[LOAD] Discarding {symbol} cooldown — already elapsed ({elapsed/3600:.1f}h)")
                    continue

                self.expired_signals[symbol] = expiry_time
                hours_left = (cooldown_seconds - elapsed) / 3600
                print(f"[LOAD] Restored cooldown: {symbol} ({hours_left:.1f}h remaining)")
                restored_cooldowns += 1

            # Clean up state file after loading
            state_path.unlink()

            if restored_pending == 0 and restored_cooldowns == 0:
                print("[LOAD] Signal state file found but all entries expired — clean slate")
            else:
                print(f"[LOAD] Restored {restored_pending} pending signal(s), {restored_cooldowns} cooldown(s)")

        except Exception as e:
            print(f"[WARN] Failed to load signal state: {e}")
