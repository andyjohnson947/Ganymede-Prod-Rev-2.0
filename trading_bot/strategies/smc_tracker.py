#!/usr/bin/env python3
"""
SMC Tracker - READ ONLY Market Structure Tracking

Tracks BOS/CHOCH events across multiple timeframes (HTF and Entry TF)
and logs them alongside trade signals for future ML analysis.

NOT USED IN TRADE DECISIONS - Data collection only.

Tracks:
- Current market structure state (BOS/CHOCH direction)
- Recent event sequences
- HTF vs ETF alignment
- All data logged for correlation with trade outcomes
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'ml_system'))

# Try to import SMC library
try:
    import io
    _old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    from smartmoneyconcepts import smc
    sys.stdout = _old_stdout
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
except Exception:
    sys.stdout = _old_stdout if '_old_stdout' in dir() else sys.stdout
    SMC_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class SMCTracker:
    """
    Tracks Smart Money Concepts (BOS/CHOCH) across multiple timeframes.

    READ ONLY - Does not influence trade decisions.
    Collects data for future ML correlation analysis.
    """

    # Timeframe mappings
    TIMEFRAMES = {
        'M5': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
        'M15': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
        'M30': mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
        'H1': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
        'H4': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
        'D1': mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
    }

    def __init__(self, data_dir: str = None):
        """Initialize SMC tracker."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / 'ml_system' / 'outputs'
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Log file for SMC events with trades
        self.smc_log_file = self.data_dir / 'smc_trade_correlation.jsonl'

        # Cache for recent events per symbol/timeframe
        # Structure: {symbol: {timeframe: {'bos': deque, 'choch': deque, 'last_update': datetime}}}
        self.event_cache: Dict[str, Dict[str, Dict]] = {}

        # Configuration
        self.swing_length = 10  # Bars for swing detection
        self.lookback_bars = 200  # Bars to fetch for analysis
        self.cache_duration = timedelta(minutes=5)  # Cache refresh interval

        # Track logged events to avoid duplicates
        self._logged_events = set()

        # Available check
        self.available = SMC_AVAILABLE and PANDAS_AVAILABLE and MT5_AVAILABLE

        if not self.available:
            missing = []
            if not SMC_AVAILABLE:
                missing.append('smartmoneyconcepts')
            if not PANDAS_AVAILABLE:
                missing.append('pandas')
            if not MT5_AVAILABLE:
                missing.append('MetaTrader5')
            print(f"[SMC TRACKER] Not available. Missing: {', '.join(missing)}")

    def get_smc_state(
        self,
        symbol: str,
        entry_tf: str = 'M15',
        htf: str = 'H1'
    ) -> Dict:
        """
        Get current SMC state for a symbol across timeframes.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            entry_tf: Entry timeframe (e.g., 'M15')
            htf: Higher timeframe (e.g., 'H1')

        Returns:
            Dict with SMC state:
            {
                'available': bool,
                'entry_tf': {...},
                'htf': {...},
                'alignment': str,
                'sequence': str,
                'recommendation': str  # For logging only, not trading
            }
        """
        if not self.available:
            return self._empty_state(symbol, entry_tf, htf)

        try:
            # Get state for both timeframes
            etf_state = self._get_tf_state(symbol, entry_tf)
            htf_state = self._get_tf_state(symbol, htf)

            # Determine alignment
            alignment = self._determine_alignment(etf_state, htf_state)

            # Get recent sequence
            sequence = self._get_sequence(symbol, entry_tf)

            # Generate analysis (for logging, not trading)
            analysis = self._analyze_state(etf_state, htf_state, alignment, sequence)

            return {
                'available': True,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'entry_tf': etf_state,
                'htf': htf_state,
                'alignment': alignment,
                'sequence': sequence,
                'analysis': analysis
            }

        except Exception as e:
            print(f"[SMC TRACKER] Error getting state for {symbol}: {e}")
            return self._empty_state(symbol, entry_tf, htf)

    def _get_tf_state(self, symbol: str, timeframe: str) -> Dict:
        """Get SMC state for a specific timeframe."""
        # Check cache
        cache_key = f"{symbol}_{timeframe}"
        if symbol in self.event_cache and timeframe in self.event_cache[symbol]:
            cached = self.event_cache[symbol][timeframe]
            if datetime.now() - cached['last_update'] < self.cache_duration:
                return cached['state']

        # Fetch fresh data
        ohlc = self._fetch_ohlc(symbol, timeframe)
        if ohlc is None or len(ohlc) < 50:
            return self._empty_tf_state(timeframe)

        # Analyze with SMC library
        try:
            swing_hl = smc.swing_highs_lows(ohlc, swing_length=self.swing_length)
            bos_choch = smc.bos_choch(ohlc, swing_hl)

            # Get latest events
            latest_bos = self._get_latest_event(bos_choch, 'BOS')
            latest_choch = self._get_latest_event(bos_choch, 'CHOCH')

            # Get recent event sequence (last 5 events)
            recent_events = self._extract_recent_events(bos_choch, limit=5)

            state = {
                'timeframe': timeframe,
                'latest_bos': latest_bos,
                'latest_choch': latest_choch,
                'recent_events': recent_events,
                'current_bias': self._determine_bias(latest_bos, latest_choch),
                'bars_since_bos': latest_bos.get('bars_ago', 999) if latest_bos else 999,
                'bars_since_choch': latest_choch.get('bars_ago', 999) if latest_choch else 999,
            }

            # Update cache
            if symbol not in self.event_cache:
                self.event_cache[symbol] = {}
            self.event_cache[symbol][timeframe] = {
                'state': state,
                'last_update': datetime.now()
            }

            return state

        except Exception as e:
            print(f"[SMC TRACKER] Error analyzing {symbol} {timeframe}: {e}")
            return self._empty_tf_state(timeframe)

    def _fetch_ohlc(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch OHLC data from MT5."""
        if not MT5_AVAILABLE:
            return None

        tf_value = self.TIMEFRAMES.get(timeframe)
        if tf_value is None:
            return None

        try:
            if not mt5.initialize():
                return None

            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, self.lookback_bars)

            if rates is None or len(rates) == 0:
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.columns = [c.lower() for c in df.columns]

            return df[['open', 'high', 'low', 'close']]

        except Exception as e:
            print(f"[SMC TRACKER] Error fetching OHLC: {e}")
            return None

    def _get_latest_event(self, bos_choch: pd.DataFrame, event_type: str) -> Optional[Dict]:
        """Get the most recent event of a specific type."""
        col = event_type.upper()
        if col not in bos_choch.columns:
            return None

        # Find non-NaN values
        events = bos_choch[bos_choch[col].notna()]

        if len(events) == 0:
            return None

        # Get the last one
        last_idx = events.index[-1]
        last_row = events.iloc[-1]

        direction = 'bullish' if last_row[col] == 1 else 'bearish'
        bars_ago = len(bos_choch) - 1 - last_idx

        return {
            'type': event_type,
            'direction': direction,
            'level': float(last_row['Level']) if pd.notna(last_row['Level']) else None,
            'bars_ago': int(bars_ago),
            'index': int(last_idx)
        }

    def _extract_recent_events(self, bos_choch: pd.DataFrame, limit: int = 5) -> List[Dict]:
        """Extract recent BOS/CHOCH events in chronological order."""
        events = []

        for idx in range(len(bos_choch)):
            row = bos_choch.iloc[idx]

            if pd.notna(row['BOS']):
                events.append({
                    'type': 'BOS',
                    'direction': 'bullish' if row['BOS'] == 1 else 'bearish',
                    'index': idx,
                    'bars_ago': len(bos_choch) - 1 - idx
                })

            if pd.notna(row['CHOCH']):
                events.append({
                    'type': 'CHOCH',
                    'direction': 'bullish' if row['CHOCH'] == 1 else 'bearish',
                    'index': idx,
                    'bars_ago': len(bos_choch) - 1 - idx
                })

        # Return last N events
        return events[-limit:] if events else []

    def _determine_bias(self, latest_bos: Optional[Dict], latest_choch: Optional[Dict]) -> str:
        """Determine current market bias based on latest events."""
        if latest_choch and latest_choch['bars_ago'] < 20:
            # Recent CHOCH - reversal signal
            return f"reversal_{latest_choch['direction']}"

        if latest_bos and latest_bos['bars_ago'] < 20:
            # Recent BOS - continuation
            return f"continuation_{latest_bos['direction']}"

        # No recent events
        if latest_bos:
            return f"established_{latest_bos['direction']}"

        return 'neutral'

    def _determine_alignment(self, etf_state: Dict, htf_state: Dict) -> str:
        """Determine alignment between entry TF and HTF."""
        etf_bias = etf_state.get('current_bias', 'neutral')
        htf_bias = htf_state.get('current_bias', 'neutral')

        # Extract direction from bias
        etf_dir = 'bullish' if 'bullish' in etf_bias else ('bearish' if 'bearish' in etf_bias else 'neutral')
        htf_dir = 'bullish' if 'bullish' in htf_bias else ('bearish' if 'bearish' in htf_bias else 'neutral')

        if etf_dir == htf_dir and etf_dir != 'neutral':
            return f"aligned_{etf_dir}"
        elif etf_dir != htf_dir and etf_dir != 'neutral' and htf_dir != 'neutral':
            return f"conflicting_etf_{etf_dir}_htf_{htf_dir}"
        else:
            return 'neutral'

    def _get_sequence(self, symbol: str, timeframe: str) -> str:
        """Get recent event sequence as a string."""
        if symbol in self.event_cache and timeframe in self.event_cache[symbol]:
            state = self.event_cache[symbol][timeframe].get('state', {})
            events = state.get('recent_events', [])

            if events:
                # Format: BOS_bull->CHOCH_bear->BOS_bear
                seq_parts = []
                for e in events[-3:]:  # Last 3 events
                    direction = 'bull' if e['direction'] == 'bullish' else 'bear'
                    seq_parts.append(f"{e['type']}_{direction}")
                return '->'.join(seq_parts)

        return 'none'

    def _analyze_state(
        self,
        etf_state: Dict,
        htf_state: Dict,
        alignment: str,
        sequence: str
    ) -> Dict:
        """
        Analyze current state and generate insights.
        FOR LOGGING ONLY - Not used in trade decisions.
        """
        analysis = {
            'buy_score': 0,
            'sell_score': 0,
            'reasons': []
        }

        # HTF CHOCH (reversal signal)
        htf_choch = htf_state.get('latest_choch')
        if htf_choch and htf_choch['bars_ago'] < 20:
            if htf_choch['direction'] == 'bullish':
                # Bullish CHOCH on HTF - but research shows this often reverses DOWN
                analysis['sell_score'] += 2
                analysis['reasons'].append('HTF bullish CHOCH (often reverses down)')
            else:
                # Bearish CHOCH on HTF - often reverses UP
                analysis['buy_score'] += 2
                analysis['reasons'].append('HTF bearish CHOCH (often reverses up)')

        # ETF CHOCH
        etf_choch = etf_state.get('latest_choch')
        if etf_choch and etf_choch['bars_ago'] < 10:
            if etf_choch['direction'] == 'bullish':
                analysis['sell_score'] += 1
                analysis['reasons'].append('ETF bullish CHOCH (counter signal)')
            else:
                analysis['buy_score'] += 1
                analysis['reasons'].append('ETF bearish CHOCH (counter signal)')

        # HTF BOS (continuation - neutral effect ~50/50)
        htf_bos = htf_state.get('latest_bos')
        if htf_bos and htf_bos['bars_ago'] < 20:
            if htf_bos['direction'] == 'bullish':
                analysis['buy_score'] += 0.5
                analysis['reasons'].append('HTF bullish BOS (weak continuation)')
            else:
                analysis['sell_score'] += 0.5
                analysis['reasons'].append('HTF bearish BOS (weak continuation)')

        # Alignment bonus
        if 'aligned' in alignment:
            if 'bullish' in alignment:
                analysis['buy_score'] += 1
            else:
                analysis['sell_score'] += 1
            analysis['reasons'].append(f'TF alignment: {alignment}')

        # Sequence patterns
        if 'BOS' in sequence and 'CHOCH' in sequence:
            # Mixed signals
            analysis['reasons'].append(f'Mixed sequence: {sequence}')

        # Determine overall recommendation (for logging only)
        if analysis['buy_score'] > analysis['sell_score'] + 1:
            analysis['recommendation'] = 'bullish_bias'
        elif analysis['sell_score'] > analysis['buy_score'] + 1:
            analysis['recommendation'] = 'bearish_bias'
        else:
            analysis['recommendation'] = 'neutral'

        return analysis

    def _empty_state(self, symbol: str, entry_tf: str, htf: str) -> Dict:
        """Return empty state when SMC not available."""
        return {
            'available': False,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'entry_tf': self._empty_tf_state(entry_tf),
            'htf': self._empty_tf_state(htf),
            'alignment': 'unknown',
            'sequence': 'none',
            'analysis': {'buy_score': 0, 'sell_score': 0, 'reasons': ['SMC not available'], 'recommendation': 'neutral'}
        }

    def _empty_tf_state(self, timeframe: str) -> Dict:
        """Return empty timeframe state."""
        return {
            'timeframe': timeframe,
            'latest_bos': None,
            'latest_choch': None,
            'recent_events': [],
            'current_bias': 'neutral',
            'bars_since_bos': 999,
            'bars_since_choch': 999
        }

    def log_events_if_new(self, symbol: str, state: Dict):
        """
        Detect and log new BOS/CHOCH events since last check.

        Called every 60s from the main loop for continuous monitoring.
        Only logs events from the last 2 bars (fresh events).
        Writes to smc_live_events.jsonl for real-time event tracking.
        """
        events_file = self.data_dir / 'smc_live_events.jsonl'

        for tf_key in ['htf', 'entry_tf']:
            tf_state = state.get(tf_key, {})
            tf_name = tf_state.get('timeframe', tf_key)

            for event_type in ['latest_bos', 'latest_choch']:
                event = tf_state.get(event_type)
                if not event or event.get('bars_ago', 999) > 2:
                    continue  # Only log events from last 2 bars (fresh)

                # Check if we already logged this event
                cache_key = f"{symbol}_{tf_name}_{event_type}_{event.get('index')}"
                if cache_key in self._logged_events:
                    continue

                self._logged_events.add(cache_key)

                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'timeframe': tf_name,
                    'event_type': event['type'],
                    'direction': event['direction'],
                    'bars_ago': event['bars_ago'],
                    'level': event.get('level'),
                    'htf_bias': state.get('htf', {}).get('current_bias', 'unknown'),
                    'etf_bias': state.get('entry_tf', {}).get('current_bias', 'unknown'),
                    'alignment': state.get('alignment', 'unknown'),
                }

                # Print to bot console
                event_label = event['type']
                dir_label = event['direction'][:4].upper()
                print(f"[SMC] {symbol} {tf_name} {event_label} {dir_label} | HTF:{log_entry['htf_bias']} ETF:{log_entry['etf_bias']} | {log_entry['alignment']}")

                # Append to live events log
                try:
                    with open(events_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(log_entry, default=str) + '\n')
                except Exception as e:
                    print(f"[SMC TRACKER] Error writing live event: {e}")

    def log_with_trade(
        self,
        symbol: str,
        direction: str,
        trade_data: Dict,
        entry_tf: str = 'M15',
        htf: str = 'H1'
    ) -> Dict:
        """
        Log SMC state alongside a trade for future correlation analysis.

        Args:
            symbol: Trading symbol
            direction: 'buy' or 'sell'
            trade_data: Additional trade data (confluence factors, etc.)
            entry_tf: Entry timeframe
            htf: Higher timeframe

        Returns:
            Combined SMC + trade data
        """
        # Get current SMC state
        smc_state = self.get_smc_state(symbol, entry_tf, htf)

        # Combine with trade data
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'trade_data': trade_data,
            'smc_state': smc_state,
            # Pre-calculate some metrics for easier analysis later
            'smc_metrics': {
                'htf_bias': smc_state.get('htf', {}).get('current_bias', 'unknown'),
                'etf_bias': smc_state.get('entry_tf', {}).get('current_bias', 'unknown'),
                'alignment': smc_state.get('alignment', 'unknown'),
                'sequence': smc_state.get('sequence', 'none'),
                'trade_aligns_with_htf_bos': self._check_alignment(direction, smc_state.get('htf', {}).get('latest_bos')),
                'trade_aligns_with_htf_choch': self._check_alignment(direction, smc_state.get('htf', {}).get('latest_choch')),
                'trade_aligns_with_etf_bos': self._check_alignment(direction, smc_state.get('entry_tf', {}).get('latest_bos')),
                'trade_aligns_with_etf_choch': self._check_alignment(direction, smc_state.get('entry_tf', {}).get('latest_choch')),
            }
        }

        # Save to log file
        try:
            with open(self.smc_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
        except Exception as e:
            print(f"[SMC TRACKER] Error logging: {e}")

        return log_entry

    def _check_alignment(self, direction: str, event: Optional[Dict]) -> str:
        """Check if trade direction aligns with an event."""
        if not event:
            return 'no_event'

        event_dir = event.get('direction', '')
        trade_is_buy = direction.lower() == 'buy'
        event_is_bullish = event_dir == 'bullish'

        if trade_is_buy == event_is_bullish:
            return 'aligned'
        else:
            return 'opposing'

    def get_correlation_report(self, days: int = 30) -> Dict:
        """
        Generate correlation report between SMC events and trade outcomes.
        Requires trades to be closed with outcome data.
        """
        if not self.smc_log_file.exists():
            return {'error': 'No SMC log data available'}

        # Load continuous trade log for outcomes
        trade_log = self.data_dir / 'continuous_trade_log.jsonl'
        if not trade_log.exists():
            return {'error': 'No trade outcome data available'}

        # This would correlate SMC data with actual trade outcomes
        # For now, return a placeholder
        return {
            'message': 'Correlation analysis requires closed trade data',
            'smc_log_entries': sum(1 for _ in open(self.smc_log_file, 'r')),
            'note': 'Run after trades close to see correlations'
        }


# Global instance
_tracker_instance: Optional[SMCTracker] = None


def get_smc_tracker() -> SMCTracker:
    """Get or create the global SMC tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SMCTracker()
    return _tracker_instance


def get_smc_state(symbol: str, entry_tf: str = 'M15', htf: str = 'H1') -> Dict:
    """Convenience function to get SMC state."""
    return get_smc_tracker().get_smc_state(symbol, entry_tf, htf)


def log_smc_with_trade(symbol: str, direction: str, trade_data: Dict, entry_tf: str = 'M15', htf: str = 'H1') -> Dict:
    """Convenience function to log SMC with trade."""
    return get_smc_tracker().log_with_trade(symbol, direction, trade_data, entry_tf, htf)


if __name__ == '__main__':
    print("SMC Tracker - Market Structure Tracking")
    print("=" * 50)

    tracker = SMCTracker()

    if tracker.available:
        print("\nSMC Tracker is available")
        print("\nTesting EURUSD...")

        state = tracker.get_smc_state('EURUSD', entry_tf='M15', htf='H1')

        print(f"\nSMC State for EURUSD:")
        print(f"  Available: {state['available']}")
        print(f"  Alignment: {state['alignment']}")
        print(f"  Sequence: {state['sequence']}")

        if state['available']:
            print(f"\n  Entry TF (M15):")
            print(f"    Bias: {state['entry_tf']['current_bias']}")
            print(f"    Latest BOS: {state['entry_tf']['latest_bos']}")
            print(f"    Latest CHOCH: {state['entry_tf']['latest_choch']}")

            print(f"\n  HTF (H1):")
            print(f"    Bias: {state['htf']['current_bias']}")
            print(f"    Latest BOS: {state['htf']['latest_bos']}")
            print(f"    Latest CHOCH: {state['htf']['latest_choch']}")

            print(f"\n  Analysis:")
            analysis = state['analysis']
            print(f"    Buy Score: {analysis['buy_score']}")
            print(f"    Sell Score: {analysis['sell_score']}")
            print(f"    Recommendation: {analysis['recommendation']}")
            for reason in analysis['reasons']:
                print(f"      - {reason}")
    else:
        print("\nSMC Tracker not available")
        print("Install required packages:")
        print("  pip install smartmoneyconcepts pandas MetaTrader5")
