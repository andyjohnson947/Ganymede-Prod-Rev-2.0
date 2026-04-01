#!/usr/bin/env python3
"""
SMC Research Module - READ ONLY / NON-PRODUCTION

Analyzes Smart Money Concepts (BOS/CHOCH) patterns to determine:
- What happens after BOS? (continuation vs reversal)
- What happens after CHOCH? (reversal confirmation)
- Sequence analysis: BOS->BOS, BOS->CHOCH, CHOCH->BOS, etc.
- Percentage likelihood of continuation vs reversal

This data can inform future ML integration if results are promising.

NOT FOR PRODUCTION USE - Research/backtesting only.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys

# Try to import smartmoneyconcepts
try:
    # Suppress unicode print from library
    import io
    import sys
    _old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    from smartmoneyconcepts import smc
    sys.stdout = _old_stdout
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    print("[SMC RESEARCH] smartmoneyconcepts not installed. Run: pip install smartmoneyconcepts")
except Exception:
    # Restore stdout if something went wrong
    sys.stdout = _old_stdout if '_old_stdout' in dir() else sys.stdout
    SMC_AVAILABLE = False
    print("[SMC RESEARCH] Error importing smartmoneyconcepts")

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[SMC RESEARCH] pandas/numpy not installed")


class SMCResearchAnalyzer:
    """
    Research analyzer for Smart Money Concepts patterns.

    Tracks:
    - BOS (Break of Structure) events
    - CHOCH (Change of Character) events
    - Sequences of events and subsequent price behavior
    - Continuation vs Reversal percentages
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent / "outputs"
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Output files
        self.research_file = self.data_dir / "smc_research_results.json"
        self.events_file = self.data_dir / "smc_events_log.jsonl"

        # Research results cache
        self.results: Dict = {}

    def analyze_ohlc_data(
        self,
        ohlc_df: 'pd.DataFrame',
        symbol: str,
        swing_length: int = 10,
        lookahead_bars: int = 20,
        close_break: bool = True
    ) -> Dict:
        """
        Analyze OHLC data for BOS/CHOCH patterns and subsequent price behavior.

        Args:
            ohlc_df: DataFrame with columns [open, high, low, close] (lowercase)
            symbol: Trading symbol
            swing_length: Lookback for swing detection
            lookahead_bars: How many bars to look ahead for price behavior
            close_break: Whether to use close price for structure breaks

        Returns:
            Dict with analysis results
        """
        if not SMC_AVAILABLE or not PANDAS_AVAILABLE:
            return {'error': 'Required packages not installed'}

        # Ensure lowercase columns
        ohlc_df.columns = [c.lower() for c in ohlc_df.columns]

        # Get swing highs/lows
        swing_hl = smc.swing_highs_lows(ohlc_df, swing_length=swing_length)

        # Get BOS/CHOCH
        bos_choch = smc.bos_choch(ohlc_df, swing_hl, close_break=close_break)

        # Merge results (use .values to avoid index alignment issues)
        analysis_df = ohlc_df.copy()
        analysis_df['swing_high'] = swing_hl['HighLow'].values == 1
        analysis_df['swing_low'] = swing_hl['HighLow'].values == -1
        analysis_df['bos'] = bos_choch['BOS'].values
        analysis_df['choch'] = bos_choch['CHOCH'].values
        analysis_df['level'] = bos_choch['Level'].values

        # Analyze events and their outcomes
        events = self._extract_events(analysis_df, symbol, lookahead_bars)
        sequences = self._analyze_sequences(events)
        statistics = self._calculate_statistics(events, sequences)

        results = {
            'symbol': symbol,
            'analyzed_at': datetime.now().isoformat(),
            'total_bars': len(ohlc_df),
            'swing_length': swing_length,
            'lookahead_bars': lookahead_bars,
            'events': events,
            'sequences': sequences,
            'statistics': statistics
        }

        # Save results
        self._save_results(results)

        return results

    def _extract_events(
        self,
        df: 'pd.DataFrame',
        symbol: str,
        lookahead: int
    ) -> List[Dict]:
        """Extract BOS/CHOCH events with their outcomes."""
        events = []

        for i in range(len(df) - lookahead):
            row = df.iloc[i]

            event_type = None
            direction = None

            # Check for BOS (must not be NaN)
            if pd.notna(row['bos']):
                if row['bos'] == 1:
                    event_type = 'BOS'
                    direction = 'bullish'
                elif row['bos'] == -1:
                    event_type = 'BOS'
                    direction = 'bearish'

            # Check for CHOCH (must not be NaN) - takes priority over BOS
            if pd.notna(row['choch']):
                if row['choch'] == 1:
                    event_type = 'CHOCH'
                    direction = 'bullish'
                elif row['choch'] == -1:
                    event_type = 'CHOCH'
                    direction = 'bearish'

            if event_type:
                # Calculate price behavior after event
                current_close = row['close']
                future_slice = df.iloc[i+1:i+lookahead+1]

                if len(future_slice) > 0:
                    future_high = future_slice['high'].max()
                    future_low = future_slice['low'].min()
                    future_close = future_slice['close'].iloc[-1]

                    # Calculate moves in pips (assume 4 decimal places for FX)
                    pip_multiplier = 10000 if 'JPY' not in symbol else 100

                    up_move_pips = (future_high - current_close) * pip_multiplier
                    down_move_pips = (current_close - future_low) * pip_multiplier
                    net_move_pips = (future_close - current_close) * pip_multiplier

                    # Determine outcome
                    if direction == 'bullish':
                        # For bullish signal, did price continue up or reverse down?
                        continued = net_move_pips > 0
                        reversed_dir = net_move_pips < 0
                        continuation_strength = up_move_pips
                        reversal_strength = down_move_pips
                    else:
                        # For bearish signal, did price continue down or reverse up?
                        continued = net_move_pips < 0
                        reversed_dir = net_move_pips > 0
                        continuation_strength = down_move_pips
                        reversal_strength = up_move_pips

                    event = {
                        'index': i,
                        'timestamp': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                        'symbol': symbol,
                        'event_type': event_type,
                        'direction': direction,
                        'price_at_event': current_close,
                        'level': row['level'] if pd.notna(row['level']) else None,
                        'future_high': future_high,
                        'future_low': future_low,
                        'future_close': future_close,
                        'up_move_pips': round(up_move_pips, 1),
                        'down_move_pips': round(down_move_pips, 1),
                        'net_move_pips': round(net_move_pips, 1),
                        'continued': continued,
                        'reversed': reversed_dir,
                        'continuation_strength_pips': round(continuation_strength, 1),
                        'reversal_strength_pips': round(reversal_strength, 1)
                    }
                    events.append(event)

        return events

    def _analyze_sequences(self, events: List[Dict]) -> Dict:
        """Analyze sequences of events (BOS->BOS, BOS->CHOCH, etc.)."""
        sequences = {
            'BOS_then_BOS': [],
            'BOS_then_CHOCH': [],
            'CHOCH_then_BOS': [],
            'CHOCH_then_CHOCH': [],
            'double_BOS': [],  # 2 BOS in same direction
            'BOS_CHOCH_same_dir': [],  # BOS then CHOCH same direction
            'BOS_CHOCH_opposite': [],  # BOS then CHOCH opposite direction
        }

        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]

            # Check if events are close enough (within 50 bars)
            if next_event['index'] - current['index'] > 50:
                continue

            seq_key = f"{current['event_type']}_then_{next_event['event_type']}"

            sequence_data = {
                'first_event': current,
                'second_event': next_event,
                'bars_between': next_event['index'] - current['index'],
                'same_direction': current['direction'] == next_event['direction'],
                'outcome_after_second': next_event['continued']
            }

            if seq_key in sequences:
                sequences[seq_key].append(sequence_data)

            # Special sequences
            if current['event_type'] == 'BOS' and next_event['event_type'] == 'BOS':
                if current['direction'] == next_event['direction']:
                    sequences['double_BOS'].append(sequence_data)

            if current['event_type'] == 'BOS' and next_event['event_type'] == 'CHOCH':
                if current['direction'] == next_event['direction']:
                    sequences['BOS_CHOCH_same_dir'].append(sequence_data)
                else:
                    sequences['BOS_CHOCH_opposite'].append(sequence_data)

        return sequences

    def _calculate_statistics(self, events: List[Dict], sequences: Dict) -> Dict:
        """Calculate continuation/reversal percentages."""
        stats = {
            'single_events': {},
            'sequences': {},
            'summary': {}
        }

        # Single event stats
        for event_type in ['BOS', 'CHOCH']:
            for direction in ['bullish', 'bearish']:
                key = f"{event_type}_{direction}"
                matching = [e for e in events
                           if e['event_type'] == event_type and e['direction'] == direction]

                if matching:
                    continued_count = sum(1 for e in matching if e['continued'])
                    reversed_count = sum(1 for e in matching if e['reversed'])
                    total = len(matching)

                    avg_continuation = sum(e['continuation_strength_pips'] for e in matching) / total
                    avg_reversal = sum(e['reversal_strength_pips'] for e in matching) / total
                    avg_net = sum(e['net_move_pips'] for e in matching) / total

                    stats['single_events'][key] = {
                        'total': total,
                        'continued': continued_count,
                        'reversed': reversed_count,
                        'continuation_pct': round((continued_count / total) * 100, 1),
                        'reversal_pct': round((reversed_count / total) * 100, 1),
                        'avg_continuation_pips': round(avg_continuation, 1),
                        'avg_reversal_pips': round(avg_reversal, 1),
                        'avg_net_pips': round(avg_net, 1)
                    }

        # Sequence stats
        for seq_name, seq_list in sequences.items():
            if seq_list:
                continued_count = sum(1 for s in seq_list if s['outcome_after_second'])
                total = len(seq_list)
                same_dir_count = sum(1 for s in seq_list if s['same_direction'])

                stats['sequences'][seq_name] = {
                    'total': total,
                    'continuation_after_sequence_pct': round((continued_count / total) * 100, 1),
                    'same_direction_pct': round((same_dir_count / total) * 100, 1),
                    'avg_bars_between': round(sum(s['bars_between'] for s in seq_list) / total, 1)
                }

        # Summary
        total_bos = sum(1 for e in events if e['event_type'] == 'BOS')
        total_choch = sum(1 for e in events if e['event_type'] == 'CHOCH')

        bos_continuation = sum(1 for e in events if e['event_type'] == 'BOS' and e['continued'])
        choch_continuation = sum(1 for e in events if e['event_type'] == 'CHOCH' and e['continued'])

        stats['summary'] = {
            'total_events': len(events),
            'total_bos': total_bos,
            'total_choch': total_choch,
            'bos_continuation_pct': round((bos_continuation / total_bos * 100), 1) if total_bos > 0 else 0,
            'choch_continuation_pct': round((choch_continuation / total_choch * 100), 1) if total_choch > 0 else 0,
            'insight': self._generate_insight(stats)
        }

        return stats

    def _generate_insight(self, stats: Dict) -> str:
        """Generate human-readable insight from statistics."""
        insights = []

        single = stats.get('single_events', {})

        # BOS insights
        bos_bull = single.get('BOS_bullish', {})
        bos_bear = single.get('BOS_bearish', {})

        if bos_bull.get('total', 0) >= 10:
            pct = bos_bull['continuation_pct']
            if pct >= 60:
                insights.append(f"Bullish BOS has {pct}% continuation rate - RELIABLE for continuation")
            elif pct <= 40:
                insights.append(f"Bullish BOS has only {pct}% continuation - often REVERSES")

        if bos_bear.get('total', 0) >= 10:
            pct = bos_bear['continuation_pct']
            if pct >= 60:
                insights.append(f"Bearish BOS has {pct}% continuation rate - RELIABLE for continuation")
            elif pct <= 40:
                insights.append(f"Bearish BOS has only {pct}% continuation - often REVERSES")

        # CHOCH insights
        choch_bull = single.get('CHOCH_bullish', {})
        choch_bear = single.get('CHOCH_bearish', {})

        if choch_bull.get('total', 0) >= 10:
            pct = choch_bull['continuation_pct']
            insights.append(f"Bullish CHOCH confirms reversal {pct}% of the time")

        if choch_bear.get('total', 0) >= 10:
            pct = choch_bear['continuation_pct']
            insights.append(f"Bearish CHOCH confirms reversal {pct}% of the time")

        # Sequence insights
        seqs = stats.get('sequences', {})

        if seqs.get('double_BOS', {}).get('total', 0) >= 5:
            pct = seqs['double_BOS']['continuation_after_sequence_pct']
            insights.append(f"Double BOS (same dir) leads to continuation {pct}% of the time")

        if seqs.get('BOS_CHOCH_opposite', {}).get('total', 0) >= 5:
            pct = seqs['BOS_CHOCH_opposite']['continuation_after_sequence_pct']
            insights.append(f"BOS followed by opposite CHOCH = reversal confirmed {100-pct}% of the time")

        return " | ".join(insights) if insights else "Insufficient data for insights"

    def _save_results(self, results: Dict):
        """Save research results to file."""
        try:
            # Save full results
            with open(self.research_file, 'w', encoding='utf-8') as f:
                # Remove events list from main file (too large), save separately
                results_summary = {k: v for k, v in results.items() if k != 'events'}
                results_summary['events_count'] = len(results.get('events', []))
                json.dump(results_summary, f, indent=2, default=str)

            # Log events to JSONL
            events = results.get('events', [])
            with open(self.events_file, 'a', encoding='utf-8') as f:
                for event in events:
                    f.write(json.dumps(event, default=str) + '\n')

        except Exception as e:
            print(f"[SMC RESEARCH] Error saving results: {e}")

    def print_report(self, results: Dict = None):
        """Print formatted research report."""
        if results is None:
            if self.research_file.exists():
                with open(self.research_file, 'r') as f:
                    results = json.load(f)
            else:
                print("No results to display. Run analyze_ohlc_data() first.")
                return

        print()
        print("=" * 80)
        print("SMC RESEARCH REPORT - BOS/CHOCH ANALYSIS")
        print("=" * 80)
        print(f"Symbol: {results.get('symbol', 'N/A')}")
        print(f"Bars Analyzed: {results.get('total_bars', 0)}")
        print(f"Lookahead Bars: {results.get('lookahead_bars', 20)}")
        print()

        stats = results.get('statistics', {})

        # Single Event Stats
        print("-" * 80)
        print("SINGLE EVENT STATISTICS")
        print("-" * 80)
        print(f"{'Event':<20} {'Total':>8} {'Cont%':>8} {'Rev%':>8} {'AvgNet':>10}")
        print("-" * 80)

        for event_key, event_stats in stats.get('single_events', {}).items():
            print(f"{event_key:<20} {event_stats['total']:>8} "
                  f"{event_stats['continuation_pct']:>7.1f}% "
                  f"{event_stats['reversal_pct']:>7.1f}% "
                  f"{event_stats['avg_net_pips']:>9.1f}p")

        # Sequence Stats
        print()
        print("-" * 80)
        print("SEQUENCE STATISTICS")
        print("-" * 80)
        print(f"{'Sequence':<25} {'Total':>8} {'Cont%':>10} {'SameDir%':>10}")
        print("-" * 80)

        for seq_name, seq_stats in stats.get('sequences', {}).items():
            if seq_stats['total'] > 0:
                print(f"{seq_name:<25} {seq_stats['total']:>8} "
                      f"{seq_stats['continuation_after_sequence_pct']:>9.1f}% "
                      f"{seq_stats['same_direction_pct']:>9.1f}%")

        # Summary
        print()
        print("-" * 80)
        print("SUMMARY")
        print("-" * 80)
        summary = stats.get('summary', {})
        print(f"Total BOS Events: {summary.get('total_bos', 0)}")
        print(f"Total CHOCH Events: {summary.get('total_choch', 0)}")
        print(f"BOS Overall Continuation: {summary.get('bos_continuation_pct', 0):.1f}%")
        print(f"CHOCH Overall Continuation: {summary.get('choch_continuation_pct', 0):.1f}%")
        print()
        print("INSIGHTS:")
        print(summary.get('insight', 'No insights available'))
        print()
        print("=" * 80)


def fetch_ohlc_data(symbol: str, timeframe: str = '1h', bars: int = 5000) -> Optional['pd.DataFrame']:
    """
    Fetch OHLC data for analysis.

    This is a placeholder - in production you'd connect to MT5 or your data source.
    For now, tries to read from saved price data if available.
    """
    if not PANDAS_AVAILABLE:
        return None

    # Try to find cached price data
    data_paths = [
        Path(__file__).parent / "outputs" / f"{symbol}_{timeframe}_ohlc.csv",
        Path(__file__).parent.parent / "trading_bot" / "data" / f"{symbol}_{timeframe}.csv",
    ]

    for path in data_paths:
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=['time'] if 'time' in pd.read_csv(path, nrows=1).columns else True)
                if 'time' in df.columns:
                    df.set_index('time', inplace=True)
                df.columns = [c.lower() for c in df.columns]
                return df
            except Exception as e:
                print(f"Error reading {path}: {e}")

    print(f"[SMC RESEARCH] No cached data found for {symbol}. Please provide OHLC DataFrame.")
    return None


def run_research(symbol: str = 'EURUSD', ohlc_df: 'pd.DataFrame' = None) -> Dict:
    """
    Run SMC research analysis.

    Args:
        symbol: Trading symbol
        ohlc_df: Optional DataFrame with OHLC data. If None, attempts to fetch.

    Returns:
        Research results dictionary
    """
    analyzer = SMCResearchAnalyzer()

    if ohlc_df is None:
        ohlc_df = fetch_ohlc_data(symbol)

    if ohlc_df is None or len(ohlc_df) < 100:
        return {'error': 'Insufficient OHLC data for analysis'}

    results = analyzer.analyze_ohlc_data(ohlc_df, symbol)
    analyzer.print_report(results)

    return results


def get_smc_confluence_score(
    bos_direction: Optional[str],
    choch_direction: Optional[str],
    trade_direction: str,
    research_results: Dict = None
) -> Dict:
    """
    Get a confluence score based on SMC events for potential future ML integration.

    Args:
        bos_direction: 'bullish', 'bearish', or None
        choch_direction: 'bullish', 'bearish', or None
        trade_direction: 'buy' or 'sell'
        research_results: Optional research results for dynamic scoring

    Returns:
        Dict with score and reasoning
    """
    score = 0
    reasons = []

    trade_is_bullish = trade_direction.lower() == 'buy'

    # BOS alignment
    if bos_direction:
        if (bos_direction == 'bullish' and trade_is_bullish) or \
           (bos_direction == 'bearish' and not trade_is_bullish):
            score += 2
            reasons.append(f"BOS aligned with trade direction (+2)")
        else:
            score -= 1
            reasons.append(f"BOS against trade direction (-1)")

    # CHOCH alignment (stronger signal)
    if choch_direction:
        if (choch_direction == 'bullish' and trade_is_bullish) or \
           (choch_direction == 'bearish' and not trade_is_bullish):
            score += 3
            reasons.append(f"CHOCH confirms trade direction (+3)")
        else:
            score -= 2
            reasons.append(f"CHOCH against trade direction (-2)")

    # Double confirmation
    if bos_direction and choch_direction:
        if bos_direction == choch_direction:
            if (bos_direction == 'bullish' and trade_is_bullish) or \
               (bos_direction == 'bearish' and not trade_is_bullish):
                score += 2
                reasons.append(f"BOS + CHOCH double confirmation (+2)")

    return {
        'smc_score': score,
        'max_score': 7,
        'score_pct': round((score / 7) * 100, 1) if score > 0 else 0,
        'bos': bos_direction,
        'choch': choch_direction,
        'reasons': reasons,
        'recommendation': 'STRONG' if score >= 5 else 'MODERATE' if score >= 2 else 'WEAK' if score >= 0 else 'AVOID'
    }


if __name__ == '__main__':
    print("SMC Research Module")
    print("=" * 50)

    if not SMC_AVAILABLE:
        print("\nTo install required package:")
        print("  pip install smartmoneyconcepts")
        print("\nThen provide OHLC data to analyze:")
        print("  from smc_research import run_research")
        print("  results = run_research('EURUSD', your_ohlc_dataframe)")
    else:
        print("\nPackage installed. Ready for analysis.")
        print("\nUsage:")
        print("  from smc_research import run_research, SMCResearchAnalyzer")
        print("  ")
        print("  # With DataFrame:")
        print("  results = run_research('EURUSD', ohlc_df)")
        print("  ")
        print("  # Or manually:")
        print("  analyzer = SMCResearchAnalyzer()")
        print("  results = analyzer.analyze_ohlc_data(ohlc_df, 'EURUSD')")
        print("  analyzer.print_report(results)")

        # Try to run with sample data if available
        print("\n" + "-" * 50)
        print("Attempting to find cached OHLC data...")

        for symbol in ['EURUSD', 'GBPUSD']:
            df = fetch_ohlc_data(symbol)
            if df is not None and len(df) >= 100:
                print(f"\nFound data for {symbol} ({len(df)} bars). Running analysis...")
                results = run_research(symbol, df)
                break
        else:
            print("No cached OHLC data found. Please provide data for analysis.")
