"""
Strategy Configuration - Discovered from EA Analysis
All values extracted from 428 trades analyzed
"""

# =============================================================================
# TRADING PARAMETERS
# =============================================================================

# Symbols to trade (EURUSD and GBPUSD based on analysis)
SYMBOLS = ['EURUSD', 'GBPUSD']

# Primary timeframe for trading
TIMEFRAME = 'H1'

# Higher timeframes for institutional levels
HTF_TIMEFRAMES = ['D1', 'W1']

# =============================================================================
# CONFLUENCE PARAMETERS (Discovered from Analysis)
# =============================================================================

# Minimum confluence score required to enter trade
MIN_CONFLUENCE_SCORE = 2  # K+Q handle quality. Score 2 = swing point + any factor. Backtest: 76.7% WR, $80k P/L

# Optimal confluence score (83.3% win rate)
OPTIMAL_CONFLUENCE_SCORE = 8

# Confluence factor weights (higher = more important)
CONFLUENCE_WEIGHTS = {
    # Primary signals from analysis
    'vwap_band_1': 1,        # Used in 28.0% of trades
    'vwap_band_2': 1,        # Used in 39.5% of trades
    'poc': 1,                # Used in 38.1% of trades
    'swing_low': 1,          # Used in 17.1% of trades
    'swing_high': 1,         # Used in 17.1% of trades
    'above_vah': 1,          # Used in 14.0% of trades
    'below_val': 1,          # Used in 18.7% of trades
    'lvn': 1,                # Used in 16.1% of trades

    # HTF factors (higher weight)
    'prev_day_vah': 2,       # 364 occurrences
    'prev_day_val': 2,       # High importance
    'prev_day_poc': 2,       # 325 occurrences
    'daily_hvn': 2,          # 310 occurrences
    'daily_poc': 2,          # Institutional level
    'weekly_hvn': 3,         # 328 occurrences (highest weight)
    'weekly_poc': 3,         # Strong institutional level
    'prev_week_swing_low': 2,  # 325 occurrences
    'prev_week_swing_high': 2, # Strong resistance
    'prev_week_vwap': 2,     # Weekly pivot

    # NOTE: Fair Value Gaps (FVGs) are NOT included in production yet
    # FVGs are tracked by ML system for data collection & validation
    # Weights will be added here once ML proves effectiveness (15+ trades)
    # Recommended weights (when enabled):
    #   'daily_bullish_fvg': 2
    #   'daily_bearish_fvg': 2
    #   'weekly_bullish_fvg': 3
    #   'weekly_bearish_fvg': 3
}

# Price tolerance for level detection (swing high/low, POC, HTF levels)
# Sweep result: 0.0035 (~35 pips) optimal — 77.4% WR, $83k P/L, PF 2.35
# Wider tolerance = better swing direction detection = fewer wrong-direction MR trades
LEVEL_TOLERANCE_PCT = 0.0035

# Swing high/low detection tolerance (tighter than HTF levels)
# 0.0015 = ~17 pips on EURUSD — must be near the actual swing point
# HTF levels keep 0.0035 (institutional zones are wider)
SWING_TOLERANCE_PCT = 0.0015

# =============================================================================
# VWAP PARAMETERS
# =============================================================================

# VWAP calculation period (bars)
VWAP_PERIOD = 200  # Approximately 8 days on H1

# Standard deviation multipliers for bands
VWAP_BAND_MULTIPLIERS = [1, 2, 3]  # ±1σ, ±2σ, ±3σ

# =============================================================================
# VOLUME PROFILE PARAMETERS
# =============================================================================

# Number of bins for volume profile
VP_BINS = 100

# Number of HVN/LVN levels to track
HVN_LEVELS = 5
LVN_LEVELS = 5

# Swing high/low detection
SWING_LOOKBACK = 10  # Bars to look back for swing points

# Swing high/low detection tolerance (tighter than HTF levels)
# 0.0015 = ~17 pips on EURUSD — must be near the actual swing point
# HTF levels keep LEVEL_TOLERANCE_PCT (0.0035 = ~40 pips) for institutional zones
SWING_TOLERANCE_PCT = 0.0015

# =============================================================================
# TREND FILTER PARAMETERS (ADX + Candle Direction)
# =============================================================================
# DISABLED: K (confirmation candles) + Q (Q-table) are the entry gatekeepers.
# Backtest shows $51k profit with K+Q alone, no ADX/M15/alignment needed.
TREND_FILTER_ENABLED = False

# ADX parameters
ADX_PERIOD = 14  # Standard ADX period
ADX_THRESHOLD = 20  # Above this = trending market
ADX_STRONG_THRESHOLD = 35  # Above this = strong trend (never trade)

# Candle direction lookback
CANDLE_LOOKBACK = 5  # Number of recent candles to analyze
CANDLE_ALIGNMENT_PCT = 70  # % of candles in same direction = "aligned"

# Trading rules
ALLOW_WEAK_TRENDS = True  # Trade when ADX 20-25 (weak trend)
SKIP_STRONG_TRENDS = True  # Never trade when ADX > 40

# Candle momentum (used by ML logger for data collection)
CANDLE_MOMENTUM_ENABLED = True
CANDLE_MOMENTUM_LOOKBACK = 5
CANDLE_MOMENTUM_MIN_SHRINK_RATIO = 0.5


def get_adx_settings(symbol: str = None) -> dict:
    """Get ADX threshold settings for strategy eligibility checks."""
    return {
        'mr_max_adx': ADX_STRONG_THRESHOLD,   # MR blocked above this (35)
        'bo_adx_min': ADX_THRESHOLD,           # BO needs at least this (20)
        'bo_adx_max': 45,                      # BO blocked above this
    }

# =============================================================================
# GRID TRADING PARAMETERS (AGGRESSIVE RECOVERY SETTINGS)
# =============================================================================

# [WARN] NOTE: AGGRESSIVE RECOVERY MODE - Higher lot sizes and more levels
# The fix prevents recovery orders from spawning more recovery (300 trade bug fixed)

# =============================================================================
# GRID PARAMETERS (PYRAMID OVERLAPPING STRATEGY)
# =============================================================================
# Grid creates OVERLAPPING positions in same direction when in profit
# Example: BUY @ 1.1000 -> +10 pips -> Open Grid 1: BUY @ 1.1010
#                       -> +20 pips -> Open Grid 2: BUY @ 1.1020
# Result: 3 independent BUY positions running (parent + 2 grids)
#
# NOTE: Per-instrument grid settings (grid_spacing_pips, max_grid_levels) are
#       defined in instruments_config.py and OVERRIDE these globals.
#       Current: All instruments use 10 pips spacing, max 2 grid levels.

# DISABLED: Focus on DCA/hedge recovery only after $500 loss analysis
GRID_ENABLED = False  # System-wide enable/disable

# KILL SWITCH: Disable negative grid (grid on losing trades)
DISABLE_NEGATIVE_GRID = True  # Only allow grid on profitable positions

# Grid lot size (applies to all instruments)
GRID_LOT_SIZE = 0.04

# FALLBACK DEFAULTS (only used if instrument not in instruments_config.py)
GRID_SPACING_PIPS = 10  # Matches current instrument settings
MAX_GRID_LEVELS = 2     # Matches current instrument settings

# =============================================================================
# HEDGING PARAMETERS (AGGRESSIVE RECOVERY SETTINGS)
# =============================================================================
# NOTE: Per-instrument hedge_trigger_pips defined in instruments_config.py
#       (EURUSD: 45 pips, GBPUSD: 55 pips, USDJPY: 50 pips)

HEDGE_ENABLED = False  # Disabled: live data shows hedge turns small SL hits into big losses
HEDGE_RATIO = 1.5  # 1.5x hedge ratio (conservative)
MAX_HEDGES_PER_POSITION = 1  # Strict limit: ONE hedge per position
MAX_HEDGE_VOLUME = 0.40  # Safety cap: max 0.20 lots per hedge

# FALLBACK DEFAULT (only used if instrument not in instruments_config.py)
HEDGE_TRIGGER_PIPS = 50  # Matches USDJPY current setting

# Recovery Stack Drawdown Protection
# ALIGNED: Now uses same -$30/-$20 limits as Per-Stack Stop Loss (see below)
# Both check_stack_drawdown() and check_stack_stop_loss() use identical logic
# STACK_DRAWDOWN_MULTIPLIER is no longer used - kept for backwards compatibility only
STACK_DRAWDOWN_MULTIPLIER = 15.0  # DEPRECATED (now uses DCA_HEDGE_MAX_LOSS / DCA_ONLY_MAX_LOSS)

# =============================================================================
# DCA/MARTINGALE PARAMETERS (AGGRESSIVE RECOVERY SETTINGS)
# =============================================================================
# NOTE: Per-instrument DCA settings (dca_trigger_pips, max_dca_levels) are
#       defined in instruments_config.py and OVERRIDE these globals.
#       Current: EURUSD=30 pips, GBPUSD=40 pips, USDJPY=35 pips trigger
#                All instruments: max 4 DCA levels (ML-recommended increase)

DCA_ENABLED = False  # Disabled: live data shows DCA amplifies losses on MR reversals
DCA_MULTIPLIER = 2.4  # REDUCED: Changed from 2.0 to 1.2 after $500 loss (prevents volume cascade)

# FALLBACK DEFAULTS (only used if instrument not in instruments_config.py)
DCA_TRIGGER_PIPS = 35  # Matches USDJPY current setting
DCA_MAX_LEVELS = 12     # REDUCED: Changed from 4 to 2 after $500 loss (limits cascade depth)

# =============================================================================
# PER-STACK STOP LOSS MANAGEMENT
# =============================================================================
# Limits maximum loss per recovery stack (original + DCA + hedge as one unit)
# Prevents catastrophic drawdown situations where both DCA and hedges are underwater

# Enable per-stack stop loss checks
ENABLE_STACK_STOPS = True

# Maximum loss for DCA-only stacks (no hedge deployed)
# TIGHTENED: Reduced from -$25 to -$20 after $500 loss analysis
# Prevents excessive drawdown during trending markets
DCA_ONLY_MAX_LOSS = -75.0

# Maximum loss for DCA+Hedge stacks (hedge deployed)
# TIGHTENED: Reduced from -$50 to -$30 after $500 loss analysis
# Exits faster when trend protection filters block recovery additions
DCA_HEDGE_MAX_LOSS = -125.0

# =============================================================================
# CASCADE STOP PROTECTION (TREND DETECTION VIA STOP-OUTS)
# =============================================================================
# Detects when multiple stops trigger in short time (range→trend transition)
# Closes all underwater stacks to prevent cascade of losses

# Enable cascade protection
ENABLE_CASCADE_PROTECTION = True

# Time window to detect multiple stop-outs (minutes)
# If 2+ stops occur within this window, cascade is triggered
STOP_OUT_WINDOW_MINUTES = 30

# Number of stop-outs in window that triggers cascade
# 2 = after 2nd stop, close ALL underwater stacks
CASCADE_THRESHOLD = 2

# How long to block new trades after cascade (minutes)
# Gives trend time to develop before re-entering
TREND_BLOCK_MINUTES = 60

# Minimum ADX to confirm trend during cascade
# If avg ADX < this at cascade, don't block trades (might be noise)
CASCADE_ADX_THRESHOLD = 25

# =============================================================================
# ADX HARD STOP PROTECTION (TOGGLE FOR ML-RECOMMENDED APPROACH)
# =============================================================================
# ML Analysis shows 65.9% better performance with ADX hard stops vs recovery
# When enabled: ADX > 30 triggers -50 pip hard SL and blocks ALL recovery
# When disabled: Current recovery system (DCA/Hedge) operates normally
#
# See analysis: ml_system/outputs/adx_vs_recovery_comparison.json
# Result: ADX stops made $13.01 more profit (65.9% improvement) over 10 days

# Master toggle for ADX-conditional hard stops
ENABLE_ADX_HARD_STOPS = False  # Disabled: K+Q handle entry quality, normal recovery for all trades

# ADX threshold for hard stop activation
# Above this = trending market, apply hard stop and block recovery
ADX_HARD_STOP_THRESHOLD = 30

# Hard stop distance in pips (applied when ADX > threshold)
# This is a FIXED stop, not trailing
ADX_HARD_STOP_PIPS = 50  # -50 pips = ~$2.50 loss on 0.04 lot

# When ADX hard stops are enabled, also block recovery during spread hours
# Spread hours: 0, 9, 13, 20, 21 GMT (identified by ML as high-risk)
BLOCK_RECOVERY_SPREAD_HOURS = False  # Disabled: K+Q handle entry, recovery runs freely

# Spread hours to avoid recovery (when BLOCK_RECOVERY_SPREAD_HOURS = True)
SPREAD_HOURS = [0, 9, 13, 20, 21]  # GMT hours

# =============================================================================
# HARD STOP LOSS PER POSITION
# =============================================================================
# Every position gets a hardware SL on MT5 at entry (survives crashes/reboots)
# If trade never reaches PC1, MT5 auto-closes at this loss
MAX_LOSS_PER_POSITION = 60.0  # $60 max loss per position (~37 pips at 0.16 lots)

# =============================================================================
# CONFIRMATION RE-ENTRY (ADD-ON - DISABLED BY DEFAULT)
# =============================================================================
# After PC1 is hit and price pulls back to stop out at BE, place a pending
# re-entry SL/3 pips back in the original direction (confirmation the move resumes)
# One re-entry per individual position only. Disabled has zero impact on existing logic.
ENABLE_CONFIRMATION_REENTRY = True   # Enabled
REENTRY_EXPIRY_HOURS = 4             # Cancel pending re-entry after this many hours

# 2-hour no-progress exit: if a position has been open 120+ minutes and never
# hit PC1 (still fully in drawdown), close it. Data shows zero cases where a
# trade recovered to PC1 after 2 hours without any upward progress.
ENABLE_TIME_EXIT = True              # Kill trades with no PC1 progress after timeout
TIME_EXIT_MINUTES = 360              # 6 hours — only fires if still negative after 6h

# =============================================================================
# RISK MANAGEMENT (AGGRESSIVE SETTINGS)
# =============================================================================

# Base lot size for initial positions
BASE_LOT_SIZE = 0.16  # Per-trade lot size ($1000 account)

# Number of initial trades to open per signal
# Opens multiple separate positions instead of one large position
# Total exposure per signal: 0.16 x 4 = 0.64 lots
INITIAL_TRADE_COUNT = 4  # 4 batch trades per signal

# Risk per trade (if using dynamic position sizing)
RISK_PERCENT = 1.0

# Use fixed lot size (True) or calculate based on risk % (False)
USE_FIXED_LOT_SIZE = True

# Maximum total exposure across all positions
MAX_TOTAL_LOTS = 15.0  # AGGRESSIVE: Increased from 5.04 to accommodate larger recovery stacks

# Maximum drawdown before stopping
MAX_DRAWDOWN_PERCENT = 50.0  # Updated from 10.0 to 25.0 per user request

# Stop loss (if used)
STOP_LOSS_PIPS = None  # EA appears to not use hard stops

# Take profit (if used)
TAKE_PROFIT_PIPS = None  # EA uses VWAP reversion

# =============================================================================
# TIMING PARAMETERS
# =============================================================================

# Trading sessions (discovered from session analysis)
TRADE_SESSIONS = {
    'tokyo': {'start': '00:00', 'end': '09:00', 'enabled': True},
    'london': {'start': '08:00', 'end': '17:00', 'enabled': True},
    'new_york': {'start': '13:00', 'end': '22:00', 'enabled': True},
    'sydney': {'start': '22:00', 'end': '07:00', 'enabled': True},
}

# Days to trade
TRADE_DAYS = [0, 1, 2, 3, 4]  # Monday-Friday

# =============================================================================
# TIME FILTERS - STRATEGY-SPECIFIC TRADING WINDOWS
# =============================================================================

# ============================================================================
# STRATEGY ON/OFF SWITCHES - Quick enable/disable for each strategy
# ============================================================================

# Enable Mean Reversion strategy
# Set to False to disable ALL mean reversion trading (maintains BO only)
MEAN_REVERSION_ENABLED = True

# Enable Breakout strategy
# Set to False to disable ALL breakout trading (maintains MR only)
# DISABLED: Focus on mean reversion with M15 trend protection after $500 loss
BREAKOUT_ENABLED = True

# Enable time filtering (False = trade all hours for enabled strategies)
# If False, enabled strategies will trade 24/7 regardless of configured windows
# DISABLED: Backtest shows K(confirmation)+Q(q-table) are sufficient.
#   With hours ON:  EURUSD $8,822 / GBPUSD $20,510 (cuts profitable hours)
#   With hours OFF: EURUSD $12,456 / GBPUSD $31,899 (same WR, +$15k more)
ENABLE_TIME_FILTERS = False

# ============================================================================

# =============================================================================
# BROKER TIMEZONE CONFIGURATION
# =============================================================================

# MT5 brokers use different server timezones. Set your broker's GMT offset here.
# This is CRITICAL for time filters to work correctly!
#
# Common broker timezones:
#   0  = GMT/UTC (rare, used by some brokers)
#   +2 = GMT+2 (EET - most European brokers in winter)
#   +3 = GMT+3 (EET summer / some brokers use this year-round)
#   -4 = GMT-4 (EDT - some US brokers in summer)
#   -5 = GMT-5 (EST - some US brokers in winter)
#
# HOW TO FIND YOUR BROKER'S OFFSET:
# 1. Check current GMT time: https://time.is/GMT
# 2. Check your MT5 terminal time (bottom right corner)
# 3. Calculate: MT5 time - GMT time = your offset
#    Example: MT5 shows 14:00, GMT is 12:00 -> offset is +2
#
# IMPORTANT: All trading hours in this config are in GMT/UTC.
# The bot will automatically convert broker time to GMT using this offset.
BROKER_GMT_OFFSET = +2  # SET THIS TO YOUR BROKER'S OFFSET!

# =============================================================================

# MEAN REVERSION TRADING HOURS (GMT/UTC) — LOCKED (do not auto-tune)
# Data-driven optimal hours from Feb 10 analysis (UTC)
MEAN_REVERSION_HOURS = [0, 1, 2, 4, 6, 7, 8, 14, 16]

# MEAN REVERSION TRADING DAYS (0=Monday, 6=Sunday)
# Best days: Tuesday (73%), Wednesday (70%), Thursday (69%)
# Monday included per user request
MEAN_REVERSION_DAYS = [0, 1, 2, 3]  # Mon, Tue, Wed, Thu

# MEAN REVERSION SESSIONS
# Tokyo: 74% win rate, London early: 68% win rate
# Avoid New York: 53% win rate
MEAN_REVERSION_SESSIONS = ['tokyo', 'london']

# BREAKOUT TRADING HOURS (UTC) — LOCKED (do not auto-tune)
# Data-driven optimal hours from Feb 10 analysis (UTC)
BREAKOUT_HOURS = [0, 1, 5, 6, 16, 20, 21, 22, 23]

# BREAKOUT TRADING DAYS
# Best days: Tuesday (62% win, high volatility), Friday (trend exhaustion)
# Monday (week open breakouts)
# OPTIMIZED: Added Wed (2) and Thu (3) - captures 21 trades, $235.77 additional profit
BREAKOUT_DAYS = [0, 1, 2, 3, 4]  # Mon-Fri (full week coverage)

# BREAKOUT SESSIONS
# London/NY overlap = highest volatility for breakouts
# Tokyo included for 03:00 breakout hour (70% win rate in analysis)
BREAKOUT_SESSIONS = ['tokyo', 'london', 'new_york']

# =============================================================================
# BREAKOUT STRATEGY PARAMETERS
# =============================================================================

# NOTE: To enable/disable breakout strategy, see BREAKOUT_ENABLED at top of file
#       (STRATEGY ON/OFF SWITCHES section, line ~190)

# Breakout detection parameters (relaxed for testing to generate more signals)
BREAKOUT_LOOKBACK = 20  # Bars to identify range high/low
BREAKOUT_VOLUME_MULTIPLIER = 1.2  # Volume must be 1.2x average (was 1.5)
BREAKOUT_ATR_MULTIPLIER = 0.8  # ATR must be 0.8x median (was 1.2 - now allows lower volatility)

# Breakout entry conditions
BREAKOUT_MIN_RANGE_PIPS = 15  # Minimum range size to consider for breakout (was 20)
BREAKOUT_CLOSE_BEYOND_LEVEL = True  # Candle must close beyond level (not just wick)

# Breakout momentum filters (relaxed thresholds)
BREAKOUT_RSI_BUY_THRESHOLD = 55  # RSI > 55 for bullish breakouts (was 60)
BREAKOUT_RSI_SELL_THRESHOLD = 45  # RSI < 45 for bearish breakouts (was 40)

# Breakout position sizing
BREAKOUT_LOT_SIZE_MULTIPLIER = 1.0  # Same lot size as MR (4 x 0.16 = 0.64)

# Breakout profit targets
BREAKOUT_TARGET_METHOD = 'range_projection'  # 'range_projection', 'atr_multiple', 'lvn'
BREAKOUT_TARGET_MULTIPLIER = 1.0  # 1x range for 'range_projection'
BREAKOUT_ATR_TARGET_MULTIPLE = 2.0  # 2x ATR for 'atr_multiple'

# Breakout stop loss (tight stops - breakouts should not reverse)
BREAKOUT_STOP_PERCENT = 0.2  # 20% of range back from breakout level

# =============================================================================
# SMC STRUCTURAL BREAKOUT PARAMETERS
# =============================================================================
# The structural breakout logic uses a 6-stage state machine:
#   IDLE → COMPRESSION → LIQUIDITY_BUILT → SWEEP → BOS → RETEST_ENTRY → EXPANSION
# Entry only occurs at stage 5 (RETEST_ENTRY) after full structural confirmation.

# Stage 1: Compression / range detection
SMC_BO_COMPRESSION_ATR_PERCENTILE = 0.35   # ATR must be below this percentile (compressed)
SMC_BO_COMPRESSION_MAX_RANGE_PIPS = 30     # Max range size to qualify as compression
SMC_BO_COMPRESSION_MIN_BARS = 15           # Min bars contained within range
SMC_BO_COMPRESSION_LOOKBACK = 20           # Bars to evaluate for range boundaries
SMC_BO_COMPRESSION_ADX_MAX = 25            # ADX must be below this (ranging market)

# Stage 2: Liquidity build (EQH/EQL)
SMC_BO_EQL_EQH_TOLERANCE_PIPS = 3         # Tolerance for equal level clustering
SMC_BO_EQL_EQH_MIN_TOUCHES = 2            # Min touches to form liquidity cluster
SMC_BO_LIQUIDITY_TIMEOUT_BARS = 30         # Max bars to wait for liquidity build

# Stage 3: Sweep / false break
SMC_BO_SWEEP_MIN_PIPS = 3                 # Min distance beyond liquidity (confirms sweep)
SMC_BO_SWEEP_MAX_PIPS = 15                # Max distance (beyond = real breakout, not sweep)
SMC_BO_SWEEP_REVERSAL_BARS = 3            # Max bars for price to reverse after sweep
SMC_BO_SWEEP_TIMEOUT_BARS = 20            # Max bars to wait for sweep

# Stage 4: BOS confirmation
SMC_BO_BOS_SWING_LENGTH = 10              # Swing length for smartmoneyconcepts library
SMC_BO_BOS_REQUIRE_CANDLE_CLOSE = True    # Require candle CLOSE beyond structure level
SMC_BO_BOS_TIMEOUT_BARS = 10              # Max bars after sweep to see BOS

# Stage 5: Retest / FVG entry
SMC_BO_RETEST_TOLERANCE_PCT = 0.003       # 0.3% tolerance for retest of BOS level
SMC_BO_RETEST_TIMEOUT_BARS = 12           # Max bars after BOS to see retest
SMC_BO_FVG_ENTRY_ENABLED = True           # Allow entry at FVG fill (not just level retest)
SMC_BO_ENTRY_ADX_MIN = 20                 # Min ADX at entry (confirm emerging trend)
SMC_BO_ENTRY_ADX_MAX = 40                 # Max ADX at entry (not overextended)

# General sequence parameters
SMC_BO_MAX_SEQUENCE_BARS = 80             # Max total bars for full sequence (~3.3 days)
SMC_BO_INVALIDATION_ATR_MULTIPLE = 2.0    # Price move that invalidates sequence
SMC_BO_INVALIDATION_ADX_MAX = 45          # ADX above this during compression = invalid

# Exit management for structural breakout trades
SMC_BO_TARGET_RANGE_MULTIPLE = 1.5        # TP = 1.5x range projected from BOS level
SMC_BO_STOP_ATR_MULTIPLE = 1.0            # SL = 1x ATR behind retest level
SMC_BO_TRAIL_AFTER_1R = True              # Enable trailing stop after 1R target hit
SMC_BO_TRAIL_DISTANCE_PIPS = 20           # Trail distance once trailing is active

# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

# Maximum open positions
MAX_OPEN_POSITIONS = 8  # 4 per signal × 2 symbols

# Maximum positions per symbol
MAX_POSITIONS_PER_SYMBOL = 4  # 4 batch entries per signal

# =============================================================================
# EXIT MANAGEMENT (Net Profit Target + Time Limit + Partial Close)
# =============================================================================

# Net profit target for recovery stacks
# Close entire stack (original + grid + hedge + DCA) when combined P&L reaches this
PROFIT_TARGET_PERCENT = 0.5  # AGGRESSIVE: 0.5% target (easier to hit with larger lots)

# Time-based exit for stuck positions
# Auto-close recovery stack after this many hours if still open
MAX_POSITION_HOURS = 12  # AGGRESSIVE: 12 hours max (was 4) - gives recovery time to work

# =============================================================================
# PARTIAL CLOSE (SCALE OUT) SETTINGS
# =============================================================================
# SOURCE OF TRUTH: instruments_config.py per-instrument 'take_profit' dict
# Each instrument defines: partial_1_pips, partial_1_percent, partial_2_pips,
# partial_2_percent, full_tp_pips, trailing_stop_*, vwap_exit_*
#
# Current structure (all instruments): 50% / 25% / 25%
#   PC1: close 50% (0.32 of 0.64) at 1R pips + SL→BE
#   PC2: close 25% (0.16) at 2R pips + activate trailing stop
#   Remaining: 25% (0.16) runs with ATR trailing stop

# Enable partial close functionality
PARTIAL_CLOSE_ENABLED = True

# LEGACY: Used only by partial_close_manager.py (not the main PC flow)
# Main PC flow reads from instruments_config.py directly
PARTIAL_CLOSE_LEVELS = [
    {'percent_to_tp': 50, 'close_percent': 50},
    {'percent_to_tp': 75, 'close_percent': 50},
]

# Minimum profit required to enable partial close (in pips)
PARTIAL_CLOSE_MIN_PROFIT_PIPS = 10

# Apply partial close to recovery stacks (grid/hedge/DCA)
PARTIAL_CLOSE_RECOVERY = False  # Only apply to original positions

# LEGACY: Used only by partial_close_manager.py
TRAIL_STOP_AFTER_PARTIAL = True
TRAIL_STOP_DISTANCE_PIPS = 15

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

# Historical data bars to load
HISTORY_BARS = {
    'H1': 10000,  # ~416 days
    'D1': 500,    # ~2 years
    'W1': 104,    # ~2 years
}

# Data cache refresh interval (minutes)
DATA_REFRESH_INTERVAL = 60

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'
LOG_TRADES = True
LOG_SIGNALS = True

# =============================================================================
# BACKTESTING
# =============================================================================

BACKTEST_MODE = False
BACKTEST_START_DATE = '2024-01-01'
BACKTEST_END_DATE = '2024-12-31'
BACKTEST_INITIAL_BALANCE = 10000

# =============================================================================
# MT5 CONNECTION
# =============================================================================

MT5_TIMEOUT = 60000  # 60 seconds
MT5_MAGIC_NUMBER = 987654  # Unique identifier for bot trades
