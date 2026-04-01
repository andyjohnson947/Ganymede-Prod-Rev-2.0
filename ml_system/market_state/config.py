"""
Market State Engine Configuration
All paths and parameters for the replay engine
"""

from pathlib import Path

# =============================================================================
# SYMBOLS & TIMEFRAMES
# =============================================================================

SYMBOLS = ['EURUSD', 'GBPUSD', 'AUDUSD']
TIMEFRAMES = {'H1': 'H1', 'D1': 'D1', 'W1': 'W1'}

# =============================================================================
# DATA PATHS
# =============================================================================

# Base directory for market state engine
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
SNAPSHOT_DIR = DATA_DIR / 'snapshots'
BUCKETED_DIR = DATA_DIR / 'bucketed'
MODEL_DIR = DATA_DIR / 'models'

# Ensure directories exist
for d in [RAW_DIR, SNAPSHOT_DIR, BUCKETED_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# REPLAY PARAMETERS
# =============================================================================

# Minimum warmup bars before first snapshot (VWAP needs 200 H1 bars)
MIN_WARMUP_BARS_H1 = 200
MIN_WARMUP_BARS_D1 = 20
MIN_WARMUP_BARS_W1 = 10

# How many years of data to fetch
YEARS_OF_DATA = 4

# MT5 pagination (max bars per API call)
MT5_BARS_PER_CALL = 1000

# =============================================================================
# FORWARD OUTCOME PARAMETERS
# =============================================================================

# Horizons to measure forward outcomes (in H1 bars = hours)
FORWARD_HORIZONS = [1, 2, 4, 6, 8, 12, 24]

# R-values for strategy outcome measurement
MR_1R_PIPS = 15    # Mean reversion 1R target
MR_2R_PIPS = 30    # Mean reversion 2R target
BO_1R_PIPS = 20    # Breakout 1R target
BO_2R_PIPS = 40    # Breakout 2R target

# Pip factor for EURUSD/GBPUSD (4-digit pairs)
PIP_FACTOR = 10000.0

# =============================================================================
# CLUSTERING PARAMETERS
# =============================================================================

# Starting number of clusters (will be tuned via elbow method)
DEFAULT_N_CLUSTERS = 10

# Range to search for optimal k
CLUSTER_K_RANGE = range(5, 25)

# Minimum samples per cluster for statistical significance
MIN_CLUSTER_SAMPLES = 500

# Features to EXCLUDE from clustering input (still computed for analysis).
# Data analysis over 49k snapshots showed these have zero discriminating power:
#   session_block  — max 34% in any cluster, near-random (5 sessions = 20% expected)
#   day_of_week    — max 29%, near-random (5 days = 20% expected)
#   fvg_proximity  — 38-67% NONE; aligned FVG actually hurts BO WR by -3.3%
#   volatility_regime — NORMAL in 19/20 clusters; only 1 EXTREME cluster emerged at k=10
CLUSTER_EXCLUDE_FEATURES = {
    'session_block',
    'day_of_week',
    'fvg_proximity',
    'volatility_regime',
}

# =============================================================================
# ALIGNMENT SCORING PARAMETERS
# =============================================================================

# Sub-score weights (must sum to 1.0)
ALIGNMENT_WEIGHTS = {
    'regime': 0.40,      # Cluster dominance match
    'structure': 0.25,   # HTF confluence strength
    'momentum': 0.20,    # Momentum alignment
    'location': 0.15,    # VWAP/swing/VP position
}

# Minimum alignment score to trade
ALIGNMENT_THRESHOLD = 0.55

# Minimum confidence (pip gap between MR and BO expectancy) to declare
# single-strategy dominance. Below this, cluster returns 'BOTH' meaning
# both strategies are viable and priority is decided by win rate.
# Derived from validation: GBPUSD confidence ranges 0.03-0.65 for
# problematic clusters; setting 3.0 converts them all to BOTH.
MIN_DOMINANCE_CONFIDENCE = 3.0

# High confidence threshold for exclusive strategy recommendation.
# Above this, the cluster strongly prefers one strategy and the other
# is actively skipped (not just deprioritized).
HIGH_DOMINANCE_CONFIDENCE = 5.0

# =============================================================================
# LEVEL TOLERANCE (reuse from strategy config)
# =============================================================================

LEVEL_TOLERANCE_PCT = 0.001  # 0.1% = ~12 pips — price must BE at the level

# =============================================================================
# BROKER SETTINGS
# =============================================================================

BROKER_GMT_OFFSET = 2  # ICMarkets uses EET (GMT+2 winter, GMT+3 summer)
