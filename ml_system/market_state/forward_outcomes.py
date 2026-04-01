#!/usr/bin/env python3
"""
Forward Outcome Measurement

For each snapshot at bar N, measures what happened from N+1 to N+24:
- Directional returns at multiple horizons
- MFE (Max Favorable Excursion) / MAE (Max Adverse Excursion)
- Whether MR or BO targets (1R, 2R) would have been hit
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from ml_system.market_state.config import (
    FORWARD_HORIZONS, PIP_FACTOR,
    MR_1R_PIPS, MR_2R_PIPS, BO_1R_PIPS, BO_2R_PIPS,
)


class ForwardOutcomeMeasurer:
    """
    Measures what happens AFTER each bar for strategy evaluation.

    For bar at index i, using bars i+1 through i+max_horizon:
    - Returns at each horizon (1h, 2h, 4h, 6h, 8h, 12h, 24h)
    - MFE up/down (best unrealized P/L in each direction)
    - MAE (worst adverse excursion)
    - Whether MR and BO targets would have been hit in each direction
    """

    def __init__(
        self,
        horizons: List[int] = None,
        mr_1r: float = MR_1R_PIPS,
        mr_2r: float = MR_2R_PIPS,
        bo_1r: float = BO_1R_PIPS,
        bo_2r: float = BO_2R_PIPS,
        pip_factor: float = PIP_FACTOR,
    ):
        self.horizons = horizons or FORWARD_HORIZONS
        self.max_horizon = max(self.horizons)
        self.mr_1r = mr_1r
        self.mr_2r = mr_2r
        self.bo_1r = bo_1r
        self.bo_2r = bo_2r
        self.pip_factor = pip_factor

    def measure_outcomes(
        self,
        h1_data: pd.DataFrame,
        bar_index: int,
    ) -> Dict:
        """
        Measure forward outcomes from bar at bar_index.

        Args:
            h1_data: Full H1 DataFrame (DatetimeIndex or integer index)
            bar_index: Integer position of the bar to measure from

        Returns:
            Dict with all forward outcome fields, or partial dict if near end of data.
        """
        result = {}
        entry_close = h1_data.iloc[bar_index]['close']

        # Check if we have enough future data
        max_available = len(h1_data) - bar_index - 1

        if max_available < 1:
            # No future data available (last bar in dataset)
            return self._empty_outcomes()

        # Determine how far forward we can look
        look_ahead = min(self.max_horizon, max_available)

        # Get future bars (bar_index+1 to bar_index+look_ahead inclusive)
        future_slice = h1_data.iloc[bar_index + 1: bar_index + 1 + look_ahead]

        # ── Returns at each horizon ───────────────────────────────
        for h in self.horizons:
            key = f'fwd_{h}h_return_pips'
            if h <= max_available:
                future_close = h1_data.iloc[bar_index + h]['close']
                result[key] = round((future_close - entry_close) * self.pip_factor, 2)
            else:
                result[key] = None

        # ── MFE / MAE over 24h window ────────────────────────────
        future_highs = future_slice['high'].values
        future_lows = future_slice['low'].values

        if len(future_highs) > 0:
            max_high = float(future_highs.max())
            min_low = float(future_lows.min())

            # MFE up: how far price went UP from entry
            result['fwd_24h_mfe_up_pips'] = round(
                (max_high - entry_close) * self.pip_factor, 2
            )

            # MFE down: how far price went DOWN from entry
            result['fwd_24h_mfe_down_pips'] = round(
                (entry_close - min_low) * self.pip_factor, 2
            )

            # MAE for a hypothetical BUY: worst drawdown = entry - min_low
            result['fwd_24h_mae_up_pips'] = round(
                (entry_close - min_low) * self.pip_factor, 2
            )

            # MAE for a hypothetical SELL: worst drawdown = max_high - entry
            result['fwd_24h_mae_down_pips'] = round(
                (max_high - entry_close) * self.pip_factor, 2
            )
        else:
            result['fwd_24h_mfe_up_pips'] = None
            result['fwd_24h_mfe_down_pips'] = None
            result['fwd_24h_mae_up_pips'] = None
            result['fwd_24h_mae_down_pips'] = None

        # ── Net direction ─────────────────────────────────────────
        if self.max_horizon <= max_available:
            final_close = h1_data.iloc[bar_index + self.max_horizon]['close']
            net_pips = (final_close - entry_close) * self.pip_factor
            if net_pips > 1:
                result['fwd_24h_net_direction'] = 'up'
            elif net_pips < -1:
                result['fwd_24h_net_direction'] = 'down'
            else:
                result['fwd_24h_net_direction'] = 'flat'
        else:
            result['fwd_24h_net_direction'] = None

        # ── R-target hits ─────────────────────────────────────────
        # Check whether price moved enough in each direction to hit R targets
        mfe_up = result.get('fwd_24h_mfe_up_pips')
        mfe_down = result.get('fwd_24h_mfe_down_pips')

        # MR BUY targets (price needs to go UP)
        if mfe_up is not None:
            result['fwd_mr_buy_1r'] = mfe_up >= self.mr_1r
            result['fwd_mr_buy_2r'] = mfe_up >= self.mr_2r
            result['fwd_bo_buy_1r'] = mfe_up >= self.bo_1r
            result['fwd_bo_buy_2r'] = mfe_up >= self.bo_2r
        else:
            result['fwd_mr_buy_1r'] = None
            result['fwd_mr_buy_2r'] = None
            result['fwd_bo_buy_1r'] = None
            result['fwd_bo_buy_2r'] = None

        # MR SELL / BO SELL targets (price needs to go DOWN)
        if mfe_down is not None:
            result['fwd_mr_sell_1r'] = mfe_down >= self.mr_1r
            result['fwd_mr_sell_2r'] = mfe_down >= self.mr_2r
            result['fwd_bo_sell_1r'] = mfe_down >= self.bo_1r
            result['fwd_bo_sell_2r'] = mfe_down >= self.bo_2r
        else:
            result['fwd_mr_sell_1r'] = None
            result['fwd_mr_sell_2r'] = None
            result['fwd_bo_sell_1r'] = None
            result['fwd_bo_sell_2r'] = None

        return result

    def _empty_outcomes(self) -> Dict:
        """Return dict with all outcome fields set to None."""
        result = {}
        for h in self.horizons:
            result[f'fwd_{h}h_return_pips'] = None

        result['fwd_24h_mfe_up_pips'] = None
        result['fwd_24h_mfe_down_pips'] = None
        result['fwd_24h_mae_up_pips'] = None
        result['fwd_24h_mae_down_pips'] = None
        result['fwd_24h_net_direction'] = None

        for prefix in ['fwd_mr_buy', 'fwd_mr_sell', 'fwd_bo_buy', 'fwd_bo_sell']:
            result[f'{prefix}_1r'] = None
            result[f'{prefix}_2r'] = None

        return result

    def measure_batch(
        self,
        h1_data: pd.DataFrame,
        bar_indices: List[int],
    ) -> List[Dict]:
        """
        Measure outcomes for multiple bars efficiently.

        Args:
            h1_data: Full H1 DataFrame
            bar_indices: List of integer bar positions to measure

        Returns:
            List of outcome dicts (same order as bar_indices)
        """
        return [self.measure_outcomes(h1_data, idx) for idx in bar_indices]
