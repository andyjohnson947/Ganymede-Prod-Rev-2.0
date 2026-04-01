"""
State Encoder for Q-Table

Discretizes market conditions into a compact state tuple for Q-table lookup.
Reuses bucket definitions from feature_buckets.py where possible.

State space: 9 dimensions, ~97,200 discrete states.
Dimensions: hour_group, day, adx_regime, volatility, vwap_position,
            confluence_tier, factor_group, direction, range_position
"""

from typing import Dict, List, Tuple, Optional


# Confluence factor names that map to snapshot boolean fields
SNAPSHOT_TO_FACTOR = {
    'vwap_in_band_1': 'vwap_band_1',
    'vwap_in_band_2': 'vwap_band_2',
    'vp_at_poc': 'poc',
    'vp_at_swing_high': 'swing_high',
    'vp_at_swing_low': 'swing_low',
    'vp_above_vah': 'above_vah',
    'vp_below_val': 'below_val',
    'vp_at_lvn': 'lvn',
}

# Factor weights matching strategy_config.py
FACTOR_WEIGHTS = {
    'vwap_band_1': 1, 'vwap_band_2': 1, 'poc': 1, 'swing_high': 1,
    'swing_low': 1, 'above_vah': 1, 'below_val': 1, 'lvn': 1,
    'prev_day_vah': 2, 'prev_day_val': 2, 'prev_day_poc': 2,
    'daily_hvn': 2, 'daily_poc': 2,
    'weekly_hvn': 3, 'weekly_poc': 3,
    'prev_week_swing_low': 2, 'prev_week_swing_high': 2, 'prev_week_vwap': 2,
}

# Structural factors (swing levels, POC, session levels, VA extremes)
STRUCTURAL_FACTORS = {
    'swing_high', 'swing_low', 'poc', 'session_high', 'session_low',
    'above_vah', 'below_val', 'lvn',
}

# HTF factors (daily/weekly institutional levels)
HTF_FACTORS = {
    'prev_day_vah', 'prev_day_val', 'prev_day_poc', 'daily_hvn', 'daily_poc',
    'prev_day_high', 'prev_day_low', 'daily_swing_high', 'daily_swing_low',
    'weekly_hvn', 'weekly_poc', 'prev_week_swing_low', 'prev_week_swing_high',
    'prev_week_vwap', 'prev_week_high', 'prev_week_low',
}


class StateEncoder:
    """Encodes market conditions into a discrete state tuple for Q-table lookup."""

    def encode(self, market_state: Dict) -> Tuple:
        """
        Encode market conditions into a state tuple.

        Args:
            market_state: Dict with keys like hour_utc, day_of_week, adx,
                         atr_percentile_50, vwap_distance_pct, active_factors,
                         confluence_score, direction. Can also be a raw snapshot dict.

        Returns:
            Tuple of 8 discrete values representing the state.
        """
        hour = market_state.get('hour_utc', 0)
        day = market_state.get('day_of_week', 0)
        adx = market_state.get('adx', 0) or 0
        atr_pct = market_state.get('atr_percentile_50', 0.5) or 0.5
        vwap_dist = market_state.get('vwap_distance_pct', 0) or 0
        direction = market_state.get('direction', 'buy')

        # Get active factors — either from pre-built list or from snapshot booleans
        active_factors = market_state.get('active_factors', None)
        if active_factors is None:
            active_factors = self._extract_factors_from_snapshot(market_state)

        factor_count = len(active_factors) if active_factors else 0

        # Range position: where price sits in the 10-bar range (0=low, 100=high)
        range_pos = market_state.get('range_position_pct', None)
        if range_pos is None:
            # Derive from snapshot fields (training path)
            dist_high = market_state.get('distance_from_high_pct', 0) or 0
            dist_low = market_state.get('distance_from_low_pct', 0) or 0
            # dist_high is negative (price below high), dist_low is positive (price above low)
            total_range = dist_low - dist_high
            if total_range > 0:
                range_pos = (dist_low / total_range) * 100
            else:
                range_pos = 50.0  # No range = mid

        return (
            self._get_hour_group(hour),
            self._get_day(day),
            self._get_adx_regime(adx),
            self._get_volatility(atr_pct),
            self._get_vwap_position(vwap_dist),
            self._get_confluence_tier(factor_count),
            self._get_factor_group(active_factors or []),
            self._get_direction(direction),
            self._get_range_position(range_pos),
        )

    def _extract_factors_from_snapshot(self, snap: Dict) -> List[str]:
        """Extract active confluence factors from a raw snapshot dict."""
        factors = []

        # Volume profile + VWAP factors (boolean fields)
        for snap_field, factor_name in SNAPSHOT_TO_FACTOR.items():
            if snap.get(snap_field):
                factors.append(factor_name)

        # HTF factors from the htf_factors list field
        htf_list = snap.get('htf_factors', [])
        if htf_list:
            for f in htf_list:
                # Normalize HTF factor names to match config keys
                normalized = f.lower().replace(' ', '_').replace("'s", '')
                factors.append(normalized)

        return factors

    def calculate_confluence_score(self, factors: List[str]) -> int:
        """Calculate confluence score from factor list using config weights."""
        score = 0
        for f in factors:
            score += FACTOR_WEIGHTS.get(f, 0)
        return score

    # ── Dimension encoders ─────────────────────────────────────────

    def _get_hour_group(self, hour: int) -> str:
        if hour <= 4:
            return 'early_asian'
        elif hour <= 8:
            return 'london_open'
        elif hour <= 12:
            return 'london'
        elif hour <= 16:
            return 'overlap'
        elif hour <= 20:
            return 'ny'
        else:
            return 'late'

    def _get_day(self, day: int) -> int:
        return min(day, 4)  # Clamp to 0-4 (Mon-Fri)

    def _get_adx_regime(self, adx: float) -> str:
        if adx < 20:
            return 'weak'
        elif adx < 30:
            return 'emerging'
        else:
            return 'strong'

    def _get_volatility(self, atr_percentile: float) -> str:
        if atr_percentile < 0.25:
            return 'compressed'
        elif atr_percentile < 0.75:
            return 'normal'
        else:
            return 'expanded'

    def _get_vwap_position(self, vwap_distance_pct: float) -> str:
        if vwap_distance_pct < -0.05:
            return 'below'
        elif vwap_distance_pct <= 0.05:
            return 'at_vwap'
        else:
            return 'above'

    def _get_confluence_tier(self, factor_count: int) -> str:
        if factor_count <= 3:
            return 'low'
        elif factor_count <= 6:
            return 'medium'
        else:
            return 'high'

    def _get_factor_group(self, factors: List[str]) -> str:
        # Normalize to underscore format (handles both 'Swing High' and 'swing_high')
        factor_set = set(f.lower().replace(' ', '_').replace("'s", '') for f in factors)

        has_structural = bool(factor_set & STRUCTURAL_FACTORS)
        has_htf = bool(factor_set & HTF_FACTORS)
        has_vwap = any('vwap' in f for f in factor_set)

        if has_structural and has_htf:
            return 'mixed'
        elif has_structural:
            return 'structural'
        elif has_htf:
            return 'htf_heavy'
        else:
            return 'vwap_only'

    def _get_direction(self, direction) -> str:
        """Normalize direction to 'buy' or 'sell'."""
        if direction and str(direction).lower() in ('sell', 'short'):
            return 'sell'
        return 'buy'

    def _get_range_position(self, range_position_pct: float) -> str:
        """Bucket price position within 10-bar range.

        Args:
            range_position_pct: 0 = at 10-bar low, 100 = at 10-bar high.
                               Negative or >100 if price broke out of range.
        """
        if range_position_pct >= 85:
            return 'at_high'
        elif range_position_pct >= 65:
            return 'upper'
        elif range_position_pct >= 35:
            return 'mid_range'
        elif range_position_pct >= 15:
            return 'lower'
        else:
            return 'at_low'
