"""
Signal Detection with Confluence Scoring
Minimum confluence score: 4 (83.3% win rate at optimal score)
"""

import os
import logging
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime

from utils.timezone_manager import get_current_time
from utils.trading_calendar import get_trading_calendar
from indicators.vwap import VWAP
from indicators.volume_profile import VolumeProfile
from indicators.htf_levels import HTFLevels
from indicators.adx import calculate_adx, should_trade_based_on_trend
from config.strategy_config import (
    MIN_CONFLUENCE_SCORE,
    CONFLUENCE_WEIGHTS,
    LEVEL_TOLERANCE_PCT,
    TREND_FILTER_ENABLED,
    ADX_PERIOD,
    ADX_THRESHOLD,
    CANDLE_LOOKBACK,
    ALLOW_WEAK_TRENDS
)


class SignalDetector:
    """Detect entry signals based on confluence of multiple factors"""

    def __init__(self, ml_logger=None, debug: bool = False):
        """Initialize signal detector with indicators

        Args:
            ml_logger: ContinuousMLLogger instance for tracking M15 blocks (optional)
            debug: Enable verbose debug output
        """
        self.debug = debug
        self.vwap = VWAP()
        self.volume_profile = VolumeProfile()
        self.htf_levels = HTFLevels()
        self.ml_logger = ml_logger
        self._last_direction_block = {}  # {symbol: block_key} — dedup direction block logs

        # Persistent file logger for ALL signal pipeline decisions
        self.signal_logger = logging.getLogger('SignalPipeline')
        self.signal_logger.setLevel(logging.INFO)
        if not self.signal_logger.handlers:  # Avoid duplicate handlers on re-init
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'logs', 'signal_decisions.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
            self.signal_logger.addHandler(handler)
        # Keep backward compat alias
        self.direction_logger = self.signal_logger

        # ADDED: Load ML-optimized weights if available (overrides default CONFLUENCE_WEIGHTS)
        self.confluence_weights = CONFLUENCE_WEIGHTS.copy()  # Start with defaults
        try:
            from ml_system.weights_optimizer import load_optimized_weights
            optimized = load_optimized_weights()
            if optimized:
                # Merge optimized weights (override matching factors, keep others)
                self.confluence_weights.update(optimized)
                print(f"[ML FEEDBACK] Loaded {len(optimized)} optimized confluence weights")
            else:
                print("[ML FEEDBACK] No optimized weights found, using defaults")
        except Exception as e:
            print(f"[ML FEEDBACK] Failed to load optimized weights: {e}")
            # Use defaults on error

        # Q-TABLE: Load pre-trained Q-tables for signal filtering
        self.q_tables = {}
        self.state_encoder = None
        try:
            from ml_system.qtable.q_table import QTable
            from ml_system.qtable.state_encoder import StateEncoder
            self.state_encoder = StateEncoder()
            for sym in ['EURUSD', 'GBPUSD']:
                qt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       '..', 'ml_system', 'qtable', f'q_table_{sym}.json')
                qt_path = os.path.normpath(qt_path)
                if os.path.exists(qt_path):
                    qt = QTable()
                    qt.load(qt_path)
                    qt_version = getattr(qt, 'version', 1)
                    if qt_version < 2:
                        print(f"[Q-TABLE WARN] {sym}: v{qt_version} Q-table (no direction) — "
                              f"retrain needed. Gate disabled.")
                        continue
                    self.q_tables[sym] = qt
                    stats = qt.get_stats()
                    print(f"[Q-TABLE] Loaded {sym} v{qt_version}: {stats['total_states']} states, "
                          f"{stats['trade_pct']:.1f}% trade-favored, "
                          f"{stats['skip_favored']} skip-favored")
            if not self.q_tables:
                print("[Q-TABLE] No pre-trained Q-tables found — all signals pass through")
        except Exception as e:
            print(f"[Q-TABLE] Failed to load Q-tables: {e}")
            self.q_tables = {}
            self.state_encoder = None

    def detect_fair_value_gaps(self, data: pd.DataFrame, price: float, tolerance_pct: float = 0.003) -> Dict:
        """
        Detect Fair Value Gaps (FVGs) near current price.

        FVG = Price imbalance where candle 1's high < candle 3's low (bullish)
                                 or candle 1's low > candle 3's high (bearish)

        Args:
            data: DataFrame with OHLC data
            price: Current price to check
            tolerance_pct: Price tolerance (0.3% = 30 pips)

        Returns:
            Dictionary with near_bullish_fvg and near_bearish_fvg flags
        """
        if len(data) < 10:
            return {'near_bullish_fvg': False, 'near_bearish_fvg': False}

        # Look back through recent candles (last 20)
        lookback = min(20, len(data) - 3)
        near_bullish = False
        near_bearish = False

        for i in range(len(data) - 3, len(data) - 3 - lookback, -1):
            if i < 0:
                break

            candle1 = data.iloc[i]
            candle3 = data.iloc[i + 2]

            # Bullish FVG: Candle 1 high < Candle 3 low (gap up, support zone)
            if candle1['high'] < candle3['low']:
                gap_low = candle1['high']
                gap_high = candle3['low']

                # Check if current price is near this FVG
                if (gap_low * (1 - tolerance_pct) <= price <= gap_high * (1 + tolerance_pct)):
                    near_bullish = True
                    break  # Found nearest FVG

            # Bearish FVG: Candle 1 low > Candle 3 high (gap down, resistance zone)
            elif candle1['low'] > candle3['high']:
                gap_high = candle1['low']
                gap_low = candle3['high']

                # Check if current price is near this FVG
                if (gap_low * (1 - tolerance_pct) <= price <= gap_high * (1 + tolerance_pct)):
                    near_bearish = True
                    break  # Found nearest FVG

        return {'near_bullish_fvg': near_bullish, 'near_bearish_fvg': near_bearish}

    def detect_signal(
        self,
        current_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        weekly_data: pd.DataFrame,
        symbol: str,
        m15_data: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Detect trading signal based on confluence

        Args:
            current_data: H1 timeframe data with VWAP calculated
            daily_data: D1 timeframe data
            weekly_data: W1 timeframe data
            symbol: Trading symbol
            m15_data: M15 timeframe data for fast trend detection (optional)

        Returns:
            Dict with signal info or None if no signal
        """
        if len(current_data) < 200:
            if self.debug:
                print(f"   [WARN]  {symbol}: Insufficient data ({len(current_data)} bars, need 200)", flush=True)
            return None

        # Get current price
        latest = current_data.iloc[-1]
        price = latest['close']

        # Initialize signal
        signal = {
            'symbol': symbol,
            'timestamp': get_current_time(),
            'price': price,
            'direction': None,
            'confluence_score': 0,
            'factors': [],
            'should_trade': False,
            'vwap_signals': {},
            'vp_signals': {},
            'htf_signals': {}
        }

        # Trading calendar disabled — K+Q handle entry filtering
        # Forex markets set their own open/close times

        # Calculate indicators if not already done
        if 'vwap' not in current_data.columns:
            current_data = self.vwap.calculate(current_data)

        # 1. Check VWAP signals
        vwap_signals = self.vwap.get_signals(current_data)
        signal['vwap_signals'] = vwap_signals

        # Check VWAP bands
        if vwap_signals['in_band_1']:
            signal['confluence_score'] += self.confluence_weights.get('vwap_band_1', 1)
            signal['factors'].append('VWAP Band 1')
            # MEAN REVERSION LOGIC: Buy when price is BELOW VWAP (expecting reversion UP)
            #                       Sell when price is ABOVE VWAP (expecting reversion DOWN)
            signal['direction'] = 'buy' if vwap_signals['direction'] == 'below' else 'sell'

        elif vwap_signals['in_band_2']:
            signal['confluence_score'] += self.confluence_weights.get('vwap_band_2', 1)
            signal['factors'].append('VWAP Band 2')
            # MEAN REVERSION LOGIC: Buy when price is BELOW VWAP (expecting reversion UP)
            #                       Sell when price is ABOVE VWAP (expecting reversion DOWN)
            signal['direction'] = 'buy' if vwap_signals['direction'] == 'below' else 'sell'

        # 2. Check Volume Profile signals
        vp_signals = self.volume_profile.get_signals(current_data, price, lookback=200)
        signal['vp_signals'] = vp_signals

        if vp_signals['at_poc']:
            signal['confluence_score'] += self.confluence_weights.get('poc', 1)
            signal['factors'].append('POC')

        if vp_signals['above_vah']:
            signal['confluence_score'] += self.confluence_weights.get('above_vah', 1)
            signal['factors'].append('Above VAH')

        if vp_signals['below_val']:
            signal['confluence_score'] += self.confluence_weights.get('below_val', 1)
            signal['factors'].append('Below VAL')

        if vp_signals['at_lvn']:
            signal['confluence_score'] += self.confluence_weights.get('lvn', 1)
            signal['factors'].append('Low Volume Node')

        if vp_signals['at_swing_high']:
            signal['confluence_score'] += self.confluence_weights.get('swing_high', 1)
            signal['factors'].append('Swing High')
            signal['structural_direction'] = 'sell'  # metadata only — VWAP sets final direction

        if vp_signals['at_swing_low']:
            signal['confluence_score'] += self.confluence_weights.get('swing_low', 1)
            signal['factors'].append('Swing Low')
            signal['structural_direction'] = 'buy'  # metadata only — VWAP sets final direction

        # 3. Check HTF levels (CRITICAL - highest weights)
        htf_levels = self.htf_levels.get_all_levels(daily_data, weekly_data)
        htf_confluence = self.htf_levels.check_confluence(price, htf_levels, LEVEL_TOLERANCE_PCT)

        signal['htf_signals'] = htf_confluence
        signal['confluence_score'] += htf_confluence['score']
        signal['factors'].extend(htf_confluence['factors'])

        # Derive direction from HTF structural levels (resistance → SELL, support → BUY)
        # Only override if no structural_direction already set by swing high/low
        RESISTANCE_FACTORS = {'Prev Day VAH', 'Prev Day High', 'Daily Swing High',
                              'Prev Week High', 'Prev Week Swing High'}
        SUPPORT_FACTORS = {'Prev Day VAL', 'Prev Day Low', 'Daily Swing Low',
                           'Prev Week Low', 'Prev Week Swing Low'}
        htf_factor_set = set(htf_confluence['factors'])
        htf_resistance = htf_factor_set & RESISTANCE_FACTORS
        htf_support = htf_factor_set & SUPPORT_FACTORS

        if not signal.get('structural_direction'):
            if htf_resistance and not htf_support:
                signal['structural_direction'] = 'sell'  # metadata only
            elif htf_support and not htf_resistance:
                signal['structural_direction'] = 'buy'  # metadata only
        # HTF with mixed resistance+support = no structural direction (MIXED)

        # NOTE: Fair Value Gaps (FVGs) are NOT used in production bot
        # FVGs are tracked by ML system ONLY for data collection & analysis
        # Once ML proves FVGs are effective (15+ trades), they can be enabled here
        # detect_fair_value_gaps() method remains available for future use

        # 4. Determine if we should trade based on confluence
        signal['should_trade'] = signal['confluence_score'] >= MIN_CONFLUENCE_SCORE

        # GATE 1: Log confluence score decision
        if signal['should_trade']:
            self.signal_logger.info(
                f"CONFLUENCE {symbol} @ {price:.5f}: Score={signal['confluence_score']}/{MIN_CONFLUENCE_SCORE} "
                f"| Factors: {', '.join(signal['factors'])} | struct_dir={signal.get('structural_direction', 'none')} | PASS"
            )
        elif signal['confluence_score'] > 0:
            self.signal_logger.info(
                f"CONFLUENCE {symbol} @ {price:.5f}: Score={signal['confluence_score']}/{MIN_CONFLUENCE_SCORE} "
                f"| Factors: {', '.join(signal['factors'])} | FAIL (need {MIN_CONFLUENCE_SCORE - signal['confluence_score']} more)"
            )

        # DEBUG: Always show confluence score for visibility (even if 0)
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S")

            print(f"   🎲 {symbol}: Calculated confluence score: {signal['confluence_score']}/{MIN_CONFLUENCE_SCORE}", flush=True)

            if signal['should_trade']:
                print(f"\n   [{timestamp}] [SIGNAL] SIGNAL DETECTED - {symbol} {signal['direction'].upper()} @ {price:.5f}")
                print(f"      Confluence Score: {signal['confluence_score']} (min: {MIN_CONFLUENCE_SCORE})")
                print(f"      Factors ({len(signal['factors'])}): {', '.join(signal['factors'])}")
            elif signal['confluence_score'] > 0:
                # Show near-miss signals (score > 0 but below threshold)
                print(f"      [{timestamp}] {symbol} @ {price:.5f} - Score: {signal['confluence_score']}/{MIN_CONFLUENCE_SCORE} (NEED {MIN_CONFLUENCE_SCORE - signal['confluence_score']} MORE)")
                print(f"      Factors: {', '.join(signal['factors']) if signal['factors'] else 'None'}")
            else:
                # Show zero-score checks so user knows bot is alive
                print(f"      [{timestamp}] {symbol} @ {price:.5f} - Score: 0/{MIN_CONFLUENCE_SCORE} - Waiting for confluence...")

        # 5. Apply trend filter (if enabled)
        if signal['should_trade'] and TREND_FILTER_ENABLED:
            # Calculate ADX
            data_with_adx = calculate_adx(current_data.copy(), period=ADX_PERIOD)
            latest_adx = data_with_adx.iloc[-1]

            adx_value = latest_adx['adx']
            plus_di = latest_adx['plus_di']
            minus_di = latest_adx['minus_di']

            # Check if we should trade based on trend analysis
            should_trade, trend_reason = should_trade_based_on_trend(
                adx_value=adx_value,
                plus_di=plus_di,
                minus_di=minus_di,
                candle_data=current_data,
                candle_lookback=CANDLE_LOOKBACK,
                adx_threshold=ADX_THRESHOLD,
                allow_weak_trends=ALLOW_WEAK_TRENDS
            )

            signal['trend_filter'] = {
                'adx': adx_value,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'passed': should_trade,
                'reason': trend_reason
            }

            if not should_trade:
                signal['should_trade'] = False
                signal['reject_reason'] = trend_reason
                self.signal_logger.info(
                    f"TREND_FILTER {symbol} {signal['direction']}: ADX={adx_value:.1f} +DI={plus_di:.1f} -DI={minus_di:.1f} | BLOCKED ({trend_reason})"
                )
                return None  # Reject signal due to trend filter

        # 5b. M15 FAST TREND DETECTION
        # DISABLED: K+Q are sufficient entry filters. Kept for data collection only.
        if signal['should_trade'] and TREND_FILTER_ENABLED and m15_data is not None and len(m15_data) >= 4:
            last_4_candles = m15_data.tail(4)

            # Count CONSECUTIVE candles in one direction
            consecutive_bullish = 0
            consecutive_bearish = 0

            # Count from most recent backwards
            for idx in range(len(last_4_candles) - 1, -1, -1):
                candle = last_4_candles.iloc[idx]
                is_bullish = candle['close'] > candle['open']
                is_bearish = candle['close'] < candle['open']

                if is_bullish:
                    consecutive_bullish += 1
                    consecutive_bearish = 0  # Reset opposite counter
                elif is_bearish:
                    consecutive_bearish += 1
                    consecutive_bullish = 0  # Reset opposite counter
                else:
                    break  # Doji or pattern break

            # MEAN REVERSION LOGIC:
            # If trying to BUY (expecting price to go UP) but last 3 M15 candles are bearish → BLOCK
            # If trying to SELL (expecting price to go DOWN) but last 3 M15 candles are bullish → BLOCK
            m15_trend_blocks_entry = False
            m15_trend_reason = ""

            if signal['direction'] == 'buy' and consecutive_bearish >= 3:
                m15_trend_blocks_entry = True
                m15_trend_reason = f"M15 Trend Against Entry: {consecutive_bearish} consecutive bearish candles (trying to BUY into downtrend)"
            elif signal['direction'] == 'sell' and consecutive_bullish >= 3:
                m15_trend_blocks_entry = True
                m15_trend_reason = f"M15 Trend Against Entry: {consecutive_bullish} consecutive bullish candles (trying to SELL into uptrend)"

            signal['m15_trend_filter'] = {
                'consecutive_bullish': consecutive_bullish,
                'consecutive_bearish': consecutive_bearish,
                'passed': not m15_trend_blocks_entry,
                'reason': m15_trend_reason if m15_trend_blocks_entry else "M15 clear or no strong trend against entry"
            }

            # GATE 2: Log M15 trend filter decision
            if m15_trend_blocks_entry:
                self.signal_logger.info(
                    f"M15_FILTER {symbol} {signal['direction']}: bull={consecutive_bullish} bear={consecutive_bearish} | BLOCKED ({m15_trend_reason})"
                )
                signal['should_trade'] = False
                signal['reject_reason'] = m15_trend_reason
                print(f"[WARN]  [ENTRY BLOCKED] {symbol} - {m15_trend_reason}")
            else:
                self.signal_logger.info(
                    f"M15_FILTER {symbol} {signal['direction']}: bull={consecutive_bullish} bear={consecutive_bearish} | PASS"
                )

                # LOG TO ML: Track M15 entry blocks for fine-tuning
                if self.ml_logger:
                    try:
                        self.ml_logger.log_m15_entry_block(
                            symbol=symbol,
                            direction=signal['direction'],
                            price=price,
                            consecutive_bullish=consecutive_bullish,
                            consecutive_bearish=consecutive_bearish,
                            confluence_score=signal['confluence_score'],
                            signal_factors=signal['factors']
                        )
                    except Exception as e:
                        print(f"[WARN] ML logging failed for M15 entry block: {e}")

                return None  # Reject entry due to M15 trend

        # 6. Set direction from VWAP (MR: price below VWAP = BUY, above = SELL)
        #    Then check agreement with structural factors — BLOCK on conflict
        if signal['should_trade']:
            # VWAP ALWAYS determines MR direction
            vwap_dir = 'buy' if vwap_signals['direction'] == 'below' else 'sell'
            signal['direction'] = vwap_dir

            struct_dir = signal.get('structural_direction')

            if struct_dir and struct_dir != vwap_dir:
                # CONFLICT: Structure says opposite of VWAP — skip this trade
                signal['should_trade'] = False
                signal['reject_reason'] = (
                    f"Direction conflict: VWAP={vwap_dir.upper()} vs Structure={struct_dir.upper()}"
                )
                self.direction_logger.info(
                    f"BLOCKED {symbol}: VWAP={vwap_dir.upper()} vs Structure={struct_dir.upper()} "
                    f"| Score={signal['confluence_score']} | Factors: {', '.join(signal['factors'])}"
                )
                # Only log once per symbol until condition changes (avoid spam)
                block_key = f"{vwap_dir}_{struct_dir}"
                if self._last_direction_block.get(symbol) != block_key:
                    print(f"[DIRECTION BLOCKED] {symbol}: VWAP={vwap_dir.upper()} vs Structure={struct_dir.upper()} — skipping")
                    print(f"   Factors: {', '.join(signal['factors'])}")
                    self._last_direction_block[symbol] = block_key
                return None
            elif struct_dir:
                self._last_direction_block.pop(symbol, None)
                self.direction_logger.info(
                    f"ALIGNED {symbol}: VWAP + Structure agree -> {vwap_dir.upper()} "
                    f"| Score={signal['confluence_score']} | Factors: {', '.join(signal['factors'])}"
                )
                print(f"[DIRECTION] {symbol}: VWAP + Structure agree -> {vwap_dir.upper()}")
            else:
                self._last_direction_block.pop(symbol, None)
                self.direction_logger.info(
                    f"VWAP-ONLY {symbol}: {vwap_dir.upper()} (no structure) "
                    f"| Score={signal['confluence_score']} | Factors: {', '.join(signal['factors'])}"
                )
                print(f"[DIRECTION] {symbol}: VWAP -> {vwap_dir.upper()} (no structural direction)")

        # 7. Add detailed signal metadata for debugging
        if signal['should_trade']:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[OK] [{ts}] [SIGNAL PASSED ALL FILTERS] {symbol} {signal['direction']} @ {price:.5f}")
            print(f"   Proceeding to trade execution")

            signal['vwap_value'] = vwap_signals.get('vwap', 0)
            signal['vwap_distance_pct'] = vwap_signals.get('distance_pct', 0)
            signal['price_vs_vwap'] = vwap_signals['direction']  # 'above' or 'below'

        # 8. Q-TABLE GATE: Check if this state/pattern historically loses
        if signal['should_trade'] and self.state_encoder and symbol in self.q_tables:
            try:
                # Get ADX values (may already be calculated from trend filter)
                adx_val = signal.get('trend_filter', {}).get('adx', 0)
                vwap_dist = vwap_signals.get('distance_pct', 0) or 0

                # Compute real ATR percentile from H1 data
                atr_pct = 0.5  # fallback
                try:
                    if 'atr' in current_data.columns and len(current_data) >= 50:
                        atr_series = current_data['atr'].dropna()
                        if len(atr_series) >= 50:
                            current_atr = float(atr_series.iloc[-1])
                            window = atr_series.tail(50)
                            atr_pct = float((window < current_atr).sum() / len(window))
                except Exception:
                    pass

                # Try to get current hour/day
                current_time = signal.get('timestamp', datetime.now())
                if isinstance(current_time, str):
                    try:
                        current_time = datetime.fromisoformat(current_time)
                    except (ValueError, TypeError):
                        current_time = datetime.now()
                hour = current_time.hour if hasattr(current_time, 'hour') else 0
                day = current_time.weekday() if hasattr(current_time, 'weekday') else 0

                # Calculate range position within 10-bar range
                lookback_bars = current_data.tail(10)
                range_high = lookback_bars['high'].max()
                range_low = lookback_bars['low'].min()
                range_span = range_high - range_low
                range_position_pct = (price - range_low) / range_span * 100 if range_span > 0 else 50.0

                market_state = {
                    'hour_utc': hour,
                    'day_of_week': day,
                    'adx': adx_val,
                    'atr_percentile_50': atr_pct,
                    'vwap_distance_pct': vwap_dist,
                    'active_factors': signal['factors'],
                    'confluence_score': signal['confluence_score'],
                    'direction': signal['direction'],
                    'range_position_pct': range_position_pct,
                }
                state = self.state_encoder.encode(market_state)
                q_vals = self.q_tables[symbol].get_q_values(state)
                visits = self.q_tables[symbol].get_visit_count(state)

                signal['q_trade'] = q_vals['TRADE']
                signal['q_skip'] = q_vals['NO_TRADE']
                signal['q_state'] = str(state)
                signal['q_visits'] = visits

                if q_vals['NO_TRADE'] > q_vals['TRADE'] and visits >= 5:
                    signal['should_trade'] = False
                    signal['reject_reason'] = (
                        f"Q-table: skip (trade={q_vals['TRADE']:.3f}, "
                        f"skip={q_vals['NO_TRADE']:.3f}, visits={visits})"
                    )
                    self.signal_logger.info(
                        f"Q_TABLE {symbol} {signal['direction']}: TRADE={q_vals['TRADE']:.3f} SKIP={q_vals['NO_TRADE']:.3f} visits={visits} range={range_position_pct:.0f}% | BLOCKED"
                    )
                    print(f"[Q-TABLE BLOCKED] {symbol} {signal['direction']} @ {price:.5f} (range {range_position_pct:.0f}%)")
                    print(f"   Q(TRADE)={q_vals['TRADE']:.3f}, Q(SKIP)={q_vals['NO_TRADE']:.3f}, "
                          f"state visits={visits}")
                    print(f"   Factors: {', '.join(signal['factors'])}")
                    print(f"   State: {state}")
                elif signal['should_trade']:
                    self.signal_logger.info(
                        f"Q_TABLE {symbol} {signal['direction']}: TRADE={q_vals['TRADE']:.3f} SKIP={q_vals['NO_TRADE']:.3f} visits={visits} range={range_position_pct:.0f}% | PASS"
                    )
                    print(f"[Q-TABLE OK] {symbol} {signal['direction']} @ range {range_position_pct:.0f}% — "
                          f"Q(TRADE)={q_vals['TRADE']:.3f} > Q(SKIP)={q_vals['NO_TRADE']:.3f} "
                          f"(visits={visits})")
            except Exception as e:
                print(f"[Q-TABLE WARN] Error checking Q-table: {e}")
                # On error, let signal through (don't block)

        # GATE 5: Log final result
        if signal['should_trade']:
            self.signal_logger.info(
                f"SIGNAL_RESULT {symbol} {signal['direction'].upper()} @ {price:.5f}: TRADE "
                f"| Score={signal['confluence_score']} | Factors: {', '.join(signal['factors'])} "
                f"| struct_dir={signal.get('structural_direction', 'none')}"
            )
        else:
            self.signal_logger.info(
                f"SIGNAL_RESULT {symbol} @ {price:.5f}: REJECTED | Reason={signal.get('reject_reason', 'confluence too low')}"
            )

        return signal if signal['should_trade'] else None

    def update_q_table_online(self, symbol: str, q_state_str: str, reward: float):
        """Online Q-learning: update Q-table from live trade outcome.

        Called after each trade closes. Updates Q-value for the state that
        was active at entry, then periodically saves to disk for persistence.

        Args:
            symbol: Trading pair (EURUSD, GBPUSD)
            q_state_str: String repr of state tuple from signal time
            reward: +1.0 for profitable trade, -1.0 for losing trade
        """
        if not q_state_str or symbol not in self.q_tables:
            return
        try:
            import ast
            state = ast.literal_eval(q_state_str)
            self.q_tables[symbol].update(state, 'TRADE', reward)

            visits = self.q_tables[symbol].get_visit_count(state)
            q_vals = self.q_tables[symbol].get_q_values(state)
            print(f"[Q-LEARN] {symbol} updated: reward={reward:+.1f} | "
                  f"Q(TRADE)={q_vals['TRADE']:.3f} | visits={visits} | state={state}")

            # Periodic save (every 5 updates) for crash protection
            self._q_update_count = getattr(self, '_q_update_count', 0) + 1
            if self._q_update_count % 5 == 0:
                qt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       '..', 'ml_system', 'qtable', f'q_table_{symbol}.json')
                self.q_tables[symbol].save(os.path.normpath(qt_path))
                print(f"[Q-TABLE] Saved {symbol} after {self._q_update_count} online updates")
        except Exception as e:
            print(f"[Q-TABLE] Online update error: {e}")

    def save_q_tables(self):
        """Save all Q-tables to disk. Called on bot shutdown."""
        for sym, qt in self.q_tables.items():
            try:
                qt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       '..', 'ml_system', 'qtable', f'q_table_{sym}.json')
                qt.save(os.path.normpath(qt_path))
                stats = qt.get_stats()
                print(f"[Q-TABLE] Saved {sym} on shutdown: {stats['total_states']} states, "
                      f"{stats.get('avg_visits', 0):.0f} avg visits")
            except Exception as e:
                print(f"[Q-TABLE] Error saving {sym}: {e}")

    def check_exit_signal(
        self,
        position: Dict,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if position should be closed (VWAP reversion)

        Args:
            position: Position dict with entry info
            current_data: Current H1 data with VWAP

        Returns:
            bool: True if should exit
        """
        if 'vwap' not in current_data.columns:
            current_data = self.vwap.calculate(current_data)

        latest = current_data.iloc[-1]
        current_price = latest['close']
        vwap = latest['vwap']

        if pd.isna(vwap):
            return False

        entry_price = position['price_open']
        position_type = position['type']

        # Check VWAP reversion
        if position_type == 'buy':
            # For buy positions, exit when price reaches VWAP from below
            if entry_price < vwap and current_price >= vwap:
                return True
        else:
            # For sell positions, exit when price reaches VWAP from above
            if entry_price > vwap and current_price <= vwap:
                return True

        return False

    def analyze_signal_strength(self, signal: Dict) -> str:
        """
        Analyze signal strength based on confluence score

        Args:
            signal: Signal dict from detect_signal()

        Returns:
            str: 'weak', 'medium', 'strong', 'very_strong'
        """
        score = signal['confluence_score']

        if score >= 10:
            return 'very_strong'
        elif score >= 7:
            return 'strong'
        elif score >= 5:
            return 'medium'
        elif score >= MIN_CONFLUENCE_SCORE:
            return 'weak'
        else:
            return 'no_signal'

    def get_signal_summary(self, signal: Optional[Dict]) -> str:
        """
        Get human-readable signal summary

        Args:
            signal: Signal dict or None

        Returns:
            str: Formatted signal summary
        """
        if signal is None:
            return "No signal detected"

        strength = self.analyze_signal_strength(signal)

        summary = []
        summary.append(f"Signal: {signal['direction'].upper()}")
        summary.append(f"   Symbol: {signal['symbol']}")
        summary.append(f"   Price: {signal['price']:.5f}")

        # Add VWAP details for mean reversion signals
        if signal.get('vwap_value'):
            summary.append(f"   VWAP: {signal['vwap_value']:.5f}")
            summary.append(f"   Price vs VWAP: {signal['price_vs_vwap'].upper()} ({signal['vwap_distance_pct']:.2f}%)")

            # Explain mean reversion logic
            if signal['price_vs_vwap'] == 'below':
                summary.append(f"   Logic: Price BELOW VWAP -> BUY (expect reversion UP)")
            else:
                summary.append(f"   Logic: Price ABOVE VWAP -> SELL (expect reversion DOWN)")

        summary.append(f"   Confluence Score: {signal['confluence_score']} ({strength})")
        summary.append(f"   Factors ({len(signal['factors'])}):")

        for factor in signal['factors']:
            summary.append(f"     * {factor}")

        return '\n'.join(summary)

    def filter_signals_by_session(
        self,
        signals: List[Dict],
        current_time: datetime
    ) -> List[Dict]:
        """
        Filter signals by trading session

        Args:
            signals: List of detected signals
            current_time: Current datetime

        Returns:
            List of filtered signals
        """
        from config.strategy_config import TRADE_SESSIONS

        hour = current_time.hour
        day_of_week = current_time.weekday()

        # Check if in trading hours
        in_session = False

        for session_name, session_config in TRADE_SESSIONS.items():
            if not session_config['enabled']:
                continue

            start_hour = int(session_config['start'].split(':')[0])
            end_hour = int(session_config['end'].split(':')[0])

            # Handle sessions that cross midnight
            if start_hour > end_hour:
                if hour >= start_hour or hour < end_hour:
                    in_session = True
                    break
            else:
                if start_hour <= hour < end_hour:
                    in_session = True
                    break

        if not in_session:
            return []

        # Check day of week
        from config.strategy_config import TRADE_DAYS
        if day_of_week not in TRADE_DAYS:
            return []

        return signals

    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Rank signals by confluence score

        Args:
            signals: List of signals

        Returns:
            List of signals sorted by score (highest first)
        """
        return sorted(signals, key=lambda x: x['confluence_score'], reverse=True)
