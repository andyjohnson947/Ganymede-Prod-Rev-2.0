"""
Breakout Strategy Module — SMC Structural Breakout

Implements a 6-stage structural breakout detection based on Smart Money Concepts:
  1. Compression / range formation
  2. Liquidity build (EQH/EQL near range)
  3. Sweep / false break (smart money takes liquidity)
  4. Break of Structure (BOS) — real directional break
  5. Retest / FVG fill — ENTRY ZONE
  6. Expansion — ride the trend leg

Entry only occurs at stage 5 after full structural confirmation.
All legacy breakout methods (range, LVN, weekly) are removed.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

from config import strategy_config as cfg
from utils.logger import logger
from strategies.breakout_state_machine import BreakoutStateMachine


class BreakoutStrategy:
    """
    SMC Structural Breakout strategy.

    Uses a per-symbol state machine to track the 6-stage sequence.
    The state machine updates every bar (including outside trading windows)
    so setups can develop passively. Entry signals are only generated
    during the breakout trading window.
    """

    def __init__(self):
        self.state_machine = BreakoutStateMachine()
        logger.info("[BO] Structural breakout strategy initialized (state machine)")

    def is_breakout_window(self, current_time: datetime) -> bool:
        """Check if current time is within breakout trading window."""
        if not cfg.ENABLE_TIME_FILTERS:
            return True

        hour = current_time.hour
        day = current_time.weekday()

        return hour in cfg.BREAKOUT_HOURS and day in cfg.BREAKOUT_DAYS

    def update_state(
        self,
        symbol: str,
        h1_data: pd.DataFrame,
        d1_data: Optional[pd.DataFrame] = None,
        current_bar_index: Optional[int] = None,
    ) -> None:
        """
        Update the state machine for a symbol. Called every bar.

        This should be called even outside the breakout trading window
        so compression/liquidity builds can be tracked passively.

        Args:
            symbol: Currency pair
            h1_data: H1 OHLCV data (must include 'atr' column if available)
            d1_data: D1 data for FVG detection (optional)
            current_bar_index: Bar index (defaults to len-1)
        """
        if current_bar_index is None:
            current_bar_index = len(h1_data) - 1

        self.state_machine.update(
            symbol=symbol,
            h1_data=h1_data,
            d1_data=d1_data,
            current_bar_index=current_bar_index,
        )

    def get_structural_entry(self, symbol: str) -> Optional[Dict]:
        """
        Get entry signal if the state machine has reached stage 5 (RETEST_ENTRY).

        Returns:
            Signal dict with direction, entry_price, target, stop, factors, etc.
            None if no entry is ready.
        """
        signal = self.state_machine.get_entry_signal(symbol)
        if signal is None:
            return None

        state = self.state_machine.states[symbol]

        # Calculate target and stop loss
        signal['target'] = self._calculate_target(state, signal)
        signal['stop'] = self._calculate_stop(state, signal)

        # Add scoring info
        signal['score'] = 5  # Base confluence score for structural breakout
        signal['confidence'] = 'high'

        logger.info(f"[BO] {symbol}: Structural entry signal — "
                    f"{signal['direction'].upper()} at {signal['entry_price']:.5f}, "
                    f"TP={signal['target']:.5f}, SL={signal['stop']:.5f}")

        return signal

    def on_trade_executed(self, symbol: str) -> None:
        """Called after a breakout trade is successfully executed."""
        self.state_machine.advance_to_expansion(symbol)

    def on_trade_closed(self, symbol: str) -> None:
        """Called after a breakout trade is closed. Resets the state machine."""
        self.state_machine.reset(symbol)

    def get_state_summary(self, symbol: str) -> Dict:
        """Get a summary of the current state machine for a symbol."""
        return self.state_machine.get_state_summary(symbol)

    # ── Target & Stop Calculation ─────────────────────────────────────

    def _calculate_target(self, state, signal: Dict) -> float:
        """
        Calculate take-profit level.
        TP = BOS level ± (range_size × target_multiple)
        """
        range_price = state.range_size_pips * 0.0001
        projection = range_price * cfg.SMC_BO_TARGET_RANGE_MULTIPLE

        if state.direction == 'buy':
            return state.bos_level + projection
        else:
            return state.bos_level - projection

    def _calculate_stop(self, state, signal: Dict) -> float:
        """
        Calculate stop-loss level.
        SL = Retest level ± (ATR × stop_multiple)
        """
        atr_stop = state.compression_atr * cfg.SMC_BO_STOP_ATR_MULTIPLE

        if state.direction == 'buy':
            # SL below the retest level
            return state.retest_level - atr_stop
        else:
            # SL above the retest level
            return state.retest_level + atr_stop

    def calculate_position_size(
        self,
        base_lot_size: float,
        stop_distance_pips: float,
        account_balance: float,
        risk_percent: float = 1.0,
    ) -> float:
        """
        Calculate position size for breakout trade.
        Uses smaller lots due to breakout trade characteristics.
        """
        breakout_lots = base_lot_size * cfg.BREAKOUT_LOT_SIZE_MULTIPLIER

        if not cfg.USE_FIXED_LOT_SIZE:
            risk_amount = account_balance * (risk_percent / 100)
            pip_value = 10  # Standard for 1 lot
            calculated_lots = risk_amount / (stop_distance_pips * pip_value)
            breakout_lots = min(breakout_lots, calculated_lots)

        return round(breakout_lots, 2)


def validate_breakout_time_filters() -> Dict[str, bool]:
    """Validate that breakout and mean reversion windows don't overlap."""
    mr_hours = set(cfg.MEAN_REVERSION_HOURS)
    bo_hours = set(cfg.BREAKOUT_HOURS)
    overlap_hours = mr_hours.intersection(bo_hours)

    return {
        'valid': len(overlap_hours) == 0,
        'overlap_hours': list(overlap_hours),
        'mean_reversion_hours': list(mr_hours),
        'breakout_hours': list(bo_hours),
        'mean_reversion_days': cfg.MEAN_REVERSION_DAYS,
        'breakout_days': cfg.BREAKOUT_DAYS,
    }


if __name__ == '__main__':
    validation = validate_breakout_time_filters()
    print("Breakout Time Filter Validation:")
    print(f"  Valid: {validation['valid']}")
    print(f"  MR Hours: {sorted(validation['mean_reversion_hours'])}")
    print(f"  BO Hours: {sorted(validation['breakout_hours'])}")
    if validation['overlap_hours']:
        print(f"  [WARN] Overlap: {validation['overlap_hours']}")
