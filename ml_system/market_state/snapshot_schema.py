#!/usr/bin/env python3
"""
Market State Snapshot Schema

Defines the complete market state recorded at each H1 bar during replay.
Approximately 95 fields covering every indicator the bot currently uses,
plus forward outcome measurements.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import numpy as np


def _serialize(obj):
    """JSON serializer that handles numpy types and NaN."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None:
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass
class MarketStateSnapshot:
    """
    Complete market state at a single H1 bar.

    Groups:
    - Identity (3 fields)
    - OHLCV (5 fields)
    - Temporal (6 fields)
    - ADX / Trend (8 fields)
    - Candle Direction (4 fields)
    - Candle Momentum (4 fields)
    - VWAP (8 fields)
    - Volume Profile (11 fields)
    - HTF Levels (12 fields)
    - Technical (4 fields)
    - Volatility (6 fields)
    - Liquidity Levels (6 fields)
    - FVG (2 fields)
    - SMC Structure (6 fields)
    - Forward Outcomes (20 fields)
    ≈ 105 total fields
    """

    # ── Identity ──────────────────────────────────────────────────
    symbol: str = ''
    timestamp: str = ''       # ISO 8601
    bar_index: int = 0

    # ── OHLCV ─────────────────────────────────────────────────────
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    # ── Temporal ──────────────────────────────────────────────────
    hour_utc: int = 0
    day_of_week: int = 0      # 0=Mon, 4=Fri
    day_of_month: int = 0
    week_of_year: int = 0
    month: int = 0
    session: str = ''          # 'asian_early', 'london_open', 'overlap', etc.

    # ── ADX / Trend ───────────────────────────────────────────────
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    di_separation: float = 0.0
    atr: float = 0.0
    adx_strength: str = ''     # weak/developing/moderate/strong/very_strong
    adx_direction: str = ''    # bullish/bearish
    adx_is_ranging: bool = False

    # ── Candle Direction ──────────────────────────────────────────
    candle_alignment: str = ''          # strong_bullish/.../mixed
    candle_bullish_pct: float = 0.0
    candle_bearish_pct: float = 0.0
    candle_avg_body: float = 0.0

    # ── Candle Momentum ───────────────────────────────────────────
    momentum_buy_shrinking: int = 0     # Shrinking pairs (buy direction)
    momentum_sell_shrinking: int = 0    # Shrinking pairs (sell direction)
    momentum_buy_is_fading: bool = False
    momentum_sell_is_fading: bool = False

    # ── VWAP ──────────────────────────────────────────────────────
    vwap_value: float = 0.0
    vwap_distance_pct: float = 0.0
    vwap_direction: str = ''            # 'above' / 'below'
    vwap_in_band_1: bool = False
    vwap_in_band_2: bool = False
    vwap_in_band_3: bool = False
    vwap_std: float = 0.0
    vwap_stretch: float = 0.0          # How far from VWAP in ATR units

    # ── Volume Profile ────────────────────────────────────────────
    vp_poc: float = 0.0
    vp_vah: float = 0.0
    vp_val: float = 0.0
    vp_at_poc: bool = False
    vp_above_vah: bool = False
    vp_below_val: bool = False
    vp_at_lvn: bool = False
    vp_at_swing_high: bool = False
    vp_at_swing_low: bool = False
    vp_swing_high_price: float = 0.0
    vp_swing_low_price: float = 0.0

    # ── HTF Levels ────────────────────────────────────────────────
    htf_confluence_score: int = 0
    htf_factor_count: int = 0
    htf_factors: List[str] = field(default_factory=list)
    prev_day_high: float = 0.0
    prev_day_low: float = 0.0
    prev_day_poc: float = 0.0
    prev_day_vah: float = 0.0
    prev_day_val: float = 0.0
    prev_week_high: float = 0.0
    prev_week_low: float = 0.0
    weekly_poc: float = 0.0
    prev_day_close: float = 0.0

    # ── Technical Indicators ──────────────────────────────────────
    rsi_14: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # ── Volatility ────────────────────────────────────────────────
    atr_14: float = 0.0
    atr_percentile_50: float = 0.5     # Where current ATR sits vs last 50 bars
    recent_range_pips: float = 0.0     # 10-bar high-low range
    momentum_20_pct: float = 0.0       # 20-bar rate of change
    distance_from_high_pct: float = 0.0
    distance_from_low_pct: float = 0.0

    # ── Liquidity Levels ──────────────────────────────────────────
    at_session_high: bool = False
    at_session_low: bool = False
    near_eqh: bool = False
    near_eql: bool = False
    at_bullish_ob: bool = False
    at_bearish_ob: bool = False

    # ── FVG ───────────────────────────────────────────────────────
    near_bullish_fvg: bool = False
    near_bearish_fvg: bool = False

    # ── SMC Structure ─────────────────────────────────────────────
    smc_available: bool = False
    smc_htf_bias: str = ''              # 'established_bullish', 'established_bearish', etc.
    smc_etf_bias: str = ''              # Entry TF bias
    smc_alignment: str = ''             # 'aligned_bullish', 'conflicting', etc.
    smc_last_bos_type: str = ''         # 'bullish' / 'bearish' / 'none'
    smc_last_bos_bars_ago: int = 999

    # ── Forward Outcomes (filled by forward_outcomes.py) ──────────
    fwd_1h_return_pips: Optional[float] = None
    fwd_2h_return_pips: Optional[float] = None
    fwd_4h_return_pips: Optional[float] = None
    fwd_6h_return_pips: Optional[float] = None
    fwd_8h_return_pips: Optional[float] = None
    fwd_12h_return_pips: Optional[float] = None
    fwd_24h_return_pips: Optional[float] = None
    fwd_24h_mfe_up_pips: Optional[float] = None
    fwd_24h_mfe_down_pips: Optional[float] = None
    fwd_24h_mae_up_pips: Optional[float] = None
    fwd_24h_mae_down_pips: Optional[float] = None
    fwd_24h_net_direction: Optional[str] = None
    fwd_mr_buy_1r: Optional[bool] = None
    fwd_mr_buy_2r: Optional[bool] = None
    fwd_mr_sell_1r: Optional[bool] = None
    fwd_mr_sell_2r: Optional[bool] = None
    fwd_bo_buy_1r: Optional[bool] = None
    fwd_bo_buy_2r: Optional[bool] = None
    fwd_bo_sell_1r: Optional[bool] = None
    fwd_bo_sell_2r: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string (one line, no newlines)."""
        return json.dumps(self.to_dict(), default=_serialize, separators=(',', ':'))

    @classmethod
    def from_dict(cls, d: dict) -> 'MarketStateSnapshot':
        """Create from dict (e.g., loaded from JSONL)."""
        # Only pass fields that exist on the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
