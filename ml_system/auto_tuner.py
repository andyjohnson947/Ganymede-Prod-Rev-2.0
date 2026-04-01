#!/usr/bin/env python3
"""
Auto-Tuning System for Trading Strategy Configuration

Reads ML analysis outputs and automatically updates strategy_config.py
to optimize performance based on collected trade data.

Features:
1. Analyzes confluence factor performance from training data
2. Updates HTF and Entry confluence weights
3. Adjusts thresholds based on win rate analysis
4. Updates trading hours based on spread hour analysis
5. Logs all changes with before/after values
6. Safety constraints to prevent extreme adjustments

Usage:
    python auto_tuner.py              # Run tuning once
    python auto_tuner.py --dry-run    # Show changes without applying
    python auto_tuner.py --schedule   # Run on schedule (every 8 hours)
"""

import json
import os
import re
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging - file gets full detail, console only warnings+
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

_file_handler = logging.FileHandler(log_dir / 'auto_tuner.log')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('AutoTuner')
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root logger
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)


class AutoTuner:
    """
    Auto-tuning system that reads ML outputs and updates strategy_config.py
    """

    # Safety constraints - prevent extreme changes
    CONSTRAINTS = {
        'min_trades_for_factor': 10,     # Minimum trades to consider a factor
        'min_trades_for_threshold': 50,  # Minimum trades to adjust thresholds
        'max_weight_change': 2,          # Maximum weight change per tuning cycle
        'min_win_rate_for_increase': 60, # Minimum win rate to increase weight
        'max_win_rate_for_decrease': 45, # Maximum win rate to decrease weight
        'min_htf_score': 2,              # Never go below this for HTF
        'max_htf_score': 8,              # Never go above this for HTF
        'min_entry_score': 2,            # Never go below this for entry
        'max_entry_score': 6,            # Never go above this for entry
        'min_combined_score': 5,         # Never go below this for combined
        'max_combined_score': 15,        # Never go above this for combined
    }

    def __init__(self, dry_run: bool = False):
        """
        Initialize auto-tuner

        Args:
            dry_run: If True, show changes without applying them
        """
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        self.ml_outputs_dir = self.project_root / 'ml_system' / 'outputs'
        self.config_path = self.project_root / 'trading_bot' / 'config' / 'strategy_config.py'
        self.training_data_path = self.project_root / 'ml_system' / 'data' / 'training_data.csv'

        # Track changes
        self.changes: List[Dict] = []
        self.tuning_report: Dict = {}

    def run(self) -> Dict:
        """
        Main entry point - run full auto-tuning cycle

        Returns:
            Dict with tuning results and applied changes
        """
        logger.info("Auto-tuning started...")

        try:
            # 1. Load all ML analysis data
            ml_data = self._load_ml_outputs()

            # 2. Load current config
            current_config = self._load_current_config()

            # 3. Analyze and calculate optimal parameters
            recommendations = self._calculate_recommendations(ml_data, current_config)

            # 4. Apply changes (or show dry run)
            if recommendations:
                self._apply_changes(recommendations, current_config)

            # 5. Generate report
            self.tuning_report = self._generate_report(recommendations)

            # 6. Save tuning history
            self._save_tuning_history()

            if self.changes:
                logger.info(f"Auto-tuning: {len(self.changes)} changes applied")

            return self.tuning_report

        except Exception as e:
            logger.error(f"Auto-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def _load_ml_outputs(self) -> Dict:
        """Load all relevant ML analysis outputs"""
        ml_data = {}

        # Load confluence quality analysis
        confluence_path = self.ml_outputs_dir / 'confluence_quality_analysis.json'
        if confluence_path.exists():
            with open(confluence_path, 'r', encoding='utf-8') as f:
                ml_data['confluence'] = json.load(f)
            logger.info(f"Loaded confluence analysis")

        # Load spread hours analysis
        spread_path = self.ml_outputs_dir / 'spread_hours_analysis.json'
        if spread_path.exists():
            with open(spread_path, 'r', encoding='utf-8') as f:
                ml_data['spread_hours'] = json.load(f)
            logger.info(f"Loaded spread hours analysis")

        # Load time performance
        time_path = self.ml_outputs_dir / 'time_performance.json'
        if time_path.exists():
            with open(time_path, 'r', encoding='utf-8') as f:
                ml_data['time_performance'] = json.load(f)
            logger.info(f"Loaded time performance analysis")

        # Load adaptive confluence weights
        adaptive_path = self.ml_outputs_dir / 'adaptive_confluence_weights.json'
        if adaptive_path.exists():
            with open(adaptive_path, 'r', encoding='utf-8') as f:
                ml_data['adaptive_weights'] = json.load(f)
            logger.info(f"Loaded adaptive weights analysis")

        # Load factor effectiveness analysis (actual win rates from closed trades)
        try:
            # Import from same directory
            import sys
            ml_system_path = Path(__file__).parent
            if str(ml_system_path) not in sys.path:
                sys.path.insert(0, str(ml_system_path))
            from factor_effectiveness import get_factor_analyzer
            analyzer = get_factor_analyzer()
            factor_analysis = analyzer.analyze_factors(days=30)
            if factor_analysis:
                ml_data['factor_effectiveness'] = factor_analysis
                logger.info(f"Loaded factor effectiveness analysis ({factor_analysis.get('trades_analyzed', 0)} trades)")
        except Exception as e:
            logger.warning(f"Could not load factor effectiveness: {e}")

        # Load training data for factor analysis
        if self.training_data_path.exists():
            ml_data['factor_stats'] = self._analyze_training_data()
            logger.info(f"Analyzed training data for factor performance")

        # Analyze factor timing patterns directly from continuous trade log
        timing_patterns = self._analyze_factor_timing_patterns()
        if timing_patterns:
            ml_data['timing_patterns'] = timing_patterns
            logger.info(f"Analyzed factor timing patterns from trade log")

        # NEW: Load closed trades for SL/BE/PC/Strategy analysis
        closed_trades = self._load_closed_trades()
        if closed_trades:
            logger.info(f"Loaded {len(closed_trades)} closed trades for optimization analysis")

            # SL Effectiveness Analysis
            sl_analysis = self.analyze_stop_loss_effectiveness(closed_trades)
            if sl_analysis:
                ml_data['stop_loss_analysis'] = sl_analysis
                logger.info(f"Analyzed stop loss effectiveness ({sl_analysis.get('trades_analyzed', 0)} trades)")

            # PC Effectiveness Analysis
            pc_analysis = self.analyze_pc_effectiveness(closed_trades)
            if pc_analysis:
                ml_data['pc_analysis'] = pc_analysis
                logger.info(f"Analyzed PC1/PC2 effectiveness")

            # BE Effectiveness Analysis
            be_analysis = self.analyze_be_effectiveness(closed_trades)
            if be_analysis:
                ml_data['be_analysis'] = be_analysis
                logger.info(f"Analyzed break even effectiveness")

            # Strategy Selection Analysis
            strategy_analysis = self.analyze_strategy_selection(closed_trades)
            if strategy_analysis:
                ml_data['strategy_analysis'] = strategy_analysis
                logger.info(f"Analyzed MR vs BO strategy performance")

        return ml_data

    def _analyze_factor_timing_patterns(self) -> Optional[Dict]:
        """
        Analyze factor timing patterns (day-of-week, hour-of-day)
        from SQLite (fallback: continuous_trade_log.jsonl)
        """
        # Try to load closed trades from SQLite first, then JSONL fallback
        closed_trades = None
        try:
            from ml_system.trade_db import get_trade_db
            db = get_trade_db()
            if db:
                closed_trades = db.get_closed_trades()
        except Exception:
            pass

        if closed_trades is None:
            # SQLite unavailable — no JSONL fallback (SQLite is source of truth)
            return None

        # Build factor-day and factor-hour matrices
        factor_day_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0}))
        factor_hour_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0}))

        trades_analyzed = 0
        min_trades = self.CONSTRAINTS['min_trades_for_factor']

        try:
            for trade in closed_trades:
                try:
                    outcome = trade.get('outcome', {})

                    entry_time = trade.get('entry_time', '')
                    if not entry_time:
                        continue

                    # Parse timestamp
                    if isinstance(entry_time, str):
                        dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    else:
                        continue

                    day_name = dt.strftime('%A')
                    hour = dt.hour
                    win = 1 if outcome.get('profit', 0) > 0 else 0
                    factors = trade.get('confluence_factors', [])

                    trades_analyzed += 1

                    for factor in factors:
                        factor_day_stats[factor][day_name]['total'] += 1
                        factor_day_stats[factor][day_name]['wins'] += win
                        factor_hour_stats[factor][hour]['total'] += 1
                        factor_hour_stats[factor][hour]['wins'] += win

                except (KeyError, ValueError):
                    continue

            if trades_analyzed < min_trades:
                return None

            # Find significant day patterns
            day_patterns = []
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

            for factor, day_stats in factor_day_stats.items():
                total_trades = sum(d['total'] for d in day_stats.values())
                if total_trades < min_trades:
                    continue

                day_rates = []
                for day in day_order:
                    if day in day_stats and day_stats[day]['total'] >= 2:
                        day_wr = day_stats[day]['wins'] / day_stats[day]['total']
                        day_rates.append((day, day_wr, day_stats[day]['total']))

                if len(day_rates) >= 2:
                    best_day = max(day_rates, key=lambda x: x[1])
                    worst_day = min(day_rates, key=lambda x: x[1])
                    edge = best_day[1] - worst_day[1]

                    if edge > 0.20:  # >20% difference
                        day_patterns.append({
                            'factor': factor,
                            'best_day': best_day[0],
                            'best_day_wr': best_day[1],
                            'best_day_count': best_day[2],
                            'worst_day': worst_day[0],
                            'worst_day_wr': worst_day[1],
                            'worst_day_count': worst_day[2],
                            'edge': edge
                        })

            # Find significant hour patterns
            hour_patterns = []

            for factor, hour_stats in factor_hour_stats.items():
                total_trades = sum(d['total'] for d in hour_stats.values())
                if total_trades < min_trades:
                    continue

                hour_rates = []
                for hour, stats in hour_stats.items():
                    if stats['total'] >= 2:
                        hour_wr = stats['wins'] / stats['total']
                        hour_rates.append((hour, hour_wr, stats['total']))

                if len(hour_rates) >= 2:
                    best_hour = max(hour_rates, key=lambda x: x[1])
                    worst_hour = min(hour_rates, key=lambda x: x[1])
                    edge = best_hour[1] - worst_hour[1]

                    if edge > 0.25:  # >25% difference
                        hour_patterns.append({
                            'factor': factor,
                            'best_hour': best_hour[0],
                            'best_hour_wr': best_hour[1],
                            'best_hour_count': best_hour[2],
                            'worst_hour': worst_hour[0],
                            'worst_hour_wr': worst_hour[1],
                            'worst_hour_count': worst_hour[2],
                            'edge': edge
                        })

            # Sort by edge
            day_patterns.sort(key=lambda x: x['edge'], reverse=True)
            hour_patterns.sort(key=lambda x: x['edge'], reverse=True)

            return {
                'trades_analyzed': trades_analyzed,
                'day_patterns': day_patterns[:10],
                'hour_patterns': hour_patterns[:5]
            }

        except Exception as e:
            logger.warning(f"Failed to analyze timing patterns: {e}")
            return None

    def _analyze_training_data(self) -> Dict:
        """Analyze training data to get factor performance statistics"""
        import pandas as pd

        try:
            df = pd.read_csv(self.training_data_path)
            logger.info(f"Loaded {len(df)} trades from training data")
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
            return {}

        # Determine outcome column - training_data.csv uses 'target' (1=win, 0=loss)
        # Other formats may use 'profit', 'is_win', or 'outcome'
        outcome_col = None
        for col in ['target', 'is_win', 'outcome', 'profit']:
            if col in df.columns:
                outcome_col = col
                break

        if outcome_col is None:
            logger.warning("No outcome column found in training data")
            return {}

        logger.info(f"Using '{outcome_col}' as outcome column")

        # Get all factor columns (binary columns that are confluence factors)
        # These are the columns we want to analyze for weight tuning
        factor_columns = [
            # Entry factors
            'vwap_band_1', 'vwap_band_2', 'vwap_band_3',
            'at_poc', 'at_lvn', 'above_vah', 'below_val',
            'at_swing_high', 'at_swing_low',
            # HTF factors
            'at_prev_day_vah', 'at_prev_day_val', 'at_prev_day_poc',
            'at_prev_day_high', 'at_prev_day_low',
            'at_weekly_hvn', 'at_daily_hvn',
            'at_prev_week_high', 'at_prev_week_low',
            # Liquidity factors (NEW)
            'at_session_high', 'at_session_low',
            'at_equal_highs', 'at_equal_lows',
            'at_bullish_ob', 'at_bearish_ob',
        ]

        # Calculate stats for each factor
        factor_stats = {}

        for factor in factor_columns:
            if factor not in df.columns:
                continue

            # Get trades where this factor was present (value == 1)
            factor_trades = df[df[factor] == 1]

            if len(factor_trades) < self.CONSTRAINTS['min_trades_for_factor']:
                continue

            # Calculate performance metrics based on outcome column
            if outcome_col == 'target' or outcome_col == 'is_win':
                # Binary: 1 = win, 0 = loss
                win_rate = factor_trades[outcome_col].mean() * 100 if len(factor_trades) > 0 else 0
                wins = int(factor_trades[outcome_col].sum())
            elif outcome_col == 'profit':
                # Numeric: positive = win, negative = loss
                win_rate = (factor_trades[outcome_col] > 0).mean() * 100 if len(factor_trades) > 0 else 0
                wins = int((factor_trades[outcome_col] > 0).sum())
            else:
                win_rate = 0
                wins = 0

            # Calculate importance score (100% based on win rate since we don't have profit amounts)
            importance_score = win_rate

            factor_stats[factor] = {
                'trades': len(factor_trades),
                'wins': wins,
                'win_rate': round(win_rate, 1),
                'avg_profit': 0.0,  # Not available in training_data.csv
                'total_profit': 0.0,  # Not available in training_data.csv
                'importance_score': round(importance_score, 1),
                'recommended_weight': self._importance_to_weight(importance_score)
            }

        # Sort by importance
        factor_stats = dict(sorted(
            factor_stats.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        ))

        return factor_stats

    def _importance_to_weight(self, importance_score: float) -> int:
        """Convert importance score to weight (1-5)"""
        if importance_score >= 85:
            return 5
        elif importance_score >= 70:
            return 4
        elif importance_score >= 55:
            return 3
        elif importance_score >= 40:
            return 2
        else:
            return 1

    # =========================================================================
    # ML OPTIMIZATION ANALYSIS METHODS (SL/BE/PC/Strategy)
    # =========================================================================

    def analyze_stop_loss_effectiveness(self, trades: List[Dict] = None) -> Optional[Dict]:
        """
        Analyze which stop loss distances work best by symbol.

        Reads from continuous_trade_log.jsonl and analyzes:
        - Win rate by initial SL distance (bucketed)
        - Average profit by SL type (hard_adx vs normal vs recovery_disabled)
        - Optimal SL distance per symbol

        Returns:
            Dict with SL effectiveness analysis and recommendations
        """
        if trades is None:
            trades = self._load_closed_trades()

        if not trades or len(trades) < 30:
            logger.warning(f"Not enough trades for SL analysis (have {len(trades)}, need 30+)")
            return None

        # Categorize trades by SL distance and type
        by_sl_distance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profits': []})
        by_sl_type = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profits': []})
        by_symbol = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'profits': []}))

        analyzed = 0
        for trade in trades:
            outcome = trade.get('outcome', {})
            sl_data = outcome.get('stop_loss', {})

            if not sl_data or sl_data.get('initial_pips') is None:
                continue

            profit = outcome.get('profit', 0)
            is_win = profit > 0
            sl_pips = sl_data.get('initial_pips', 0)
            sl_type = sl_data.get('sl_type', 'unknown')
            symbol = trade.get('symbol', 'UNKNOWN')

            # Bucket SL distance (10-pip buckets)
            sl_bucket = f"{int(sl_pips // 10) * 10}-{int(sl_pips // 10) * 10 + 10}"

            # Aggregate
            by_sl_distance[sl_bucket]['wins' if is_win else 'losses'] += 1
            by_sl_distance[sl_bucket]['profits'].append(profit)

            by_sl_type[sl_type]['wins' if is_win else 'losses'] += 1
            by_sl_type[sl_type]['profits'].append(profit)

            by_symbol[symbol][sl_bucket]['wins' if is_win else 'losses'] += 1
            by_symbol[symbol][sl_bucket]['profits'].append(profit)

            analyzed += 1

        if analyzed < 20:
            logger.warning(f"Only {analyzed} trades have SL data - need more for analysis")
            return None

        # Calculate metrics
        def calc_metrics(data):
            total = data['wins'] + data['losses']
            if total == 0:
                return None
            return {
                'total': total,
                'wins': data['wins'],
                'win_rate': data['wins'] / total * 100,
                'avg_profit': sum(data['profits']) / len(data['profits']) if data['profits'] else 0,
                'total_profit': sum(data['profits'])
            }

        results = {
            'trades_analyzed': analyzed,
            'by_sl_distance': {},
            'by_sl_type': {},
            'by_symbol': {},
            'recommendations': []
        }

        # Process SL distance buckets
        for bucket, data in sorted(by_sl_distance.items()):
            metrics = calc_metrics(data)
            if metrics and metrics['total'] >= 5:
                results['by_sl_distance'][bucket] = metrics

        # Process SL types
        for sl_type, data in by_sl_type.items():
            metrics = calc_metrics(data)
            if metrics and metrics['total'] >= 5:
                results['by_sl_type'][sl_type] = metrics

        # Process by symbol
        for symbol, buckets in by_symbol.items():
            results['by_symbol'][symbol] = {}
            for bucket, data in sorted(buckets.items()):
                metrics = calc_metrics(data)
                if metrics and metrics['total'] >= 3:
                    results['by_symbol'][symbol][bucket] = metrics

        # Generate recommendations
        if results['by_sl_distance']:
            best_bucket = max(results['by_sl_distance'].items(),
                            key=lambda x: x[1]['win_rate'] if x[1]['total'] >= 5 else 0)
            results['recommendations'].append({
                'type': 'sl_distance',
                'current': 'varies',
                'optimal': best_bucket[0],
                'reason': f"Best win rate {best_bucket[1]['win_rate']:.0f}% ({best_bucket[1]['total']} trades)"
            })

        logger.info(f"SL analysis complete: {analyzed} trades analyzed")
        return results

    def analyze_pc_effectiveness(self, trades: List[Dict] = None) -> Optional[Dict]:
        """
        Analyze PC1/PC2 partial close effectiveness.

        Analyzes:
        - Win rate when PC1 triggered vs not
        - Profit improvement when PC2 triggered
        - Optimal PC pip levels by symbol
        - Time to PC1/PC2 trigger

        Returns:
            Dict with PC effectiveness analysis and recommendations
        """
        if trades is None:
            trades = self._load_closed_trades()

        if not trades or len(trades) < 30:
            return None

        # Categorize trades
        pc1_triggered = []
        pc1_not_triggered = []
        pc2_triggered = []
        pc1_timings = []
        pc2_timings = []

        for trade in trades:
            outcome = trade.get('outcome', {})
            pc_data = outcome.get('pc_data', {})
            profit = outcome.get('profit', 0)

            has_pc1 = pc_data.get('pc1_triggered', False)
            has_pc2 = pc_data.get('pc2_triggered', False)

            if has_pc1:
                pc1_triggered.append(profit)
                if pc_data.get('pc1_time'):
                    # Calculate time from entry to PC1
                    try:
                        entry_time = trade.get('entry_time')
                        pc1_time = pc_data.get('pc1_time')
                        # Would calculate timing here if we have proper datetime objects
                    except:
                        pass
            else:
                pc1_not_triggered.append(profit)

            if has_pc2:
                pc2_triggered.append(profit)

        results = {
            'trades_analyzed': len(trades),
            'pc1_stats': {
                'triggered_count': len(pc1_triggered),
                'not_triggered_count': len(pc1_not_triggered),
                'triggered_avg_profit': sum(pc1_triggered) / len(pc1_triggered) if pc1_triggered else 0,
                'not_triggered_avg_profit': sum(pc1_not_triggered) / len(pc1_not_triggered) if pc1_not_triggered else 0,
            },
            'pc2_stats': {
                'triggered_count': len(pc2_triggered),
                'triggered_avg_profit': sum(pc2_triggered) / len(pc2_triggered) if pc2_triggered else 0,
            },
            'recommendations': []
        }

        # Generate recommendations
        if len(pc1_triggered) >= 10 and len(pc1_not_triggered) >= 10:
            pc1_benefit = results['pc1_stats']['triggered_avg_profit'] - results['pc1_stats']['not_triggered_avg_profit']
            if pc1_benefit > 0:
                results['recommendations'].append({
                    'type': 'pc1',
                    'insight': f"PC1 triggers improve avg profit by ${pc1_benefit:.2f}",
                    'action': 'Keep PC1 active'
                })
            else:
                results['recommendations'].append({
                    'type': 'pc1',
                    'insight': f"PC1 triggers reduce avg profit by ${abs(pc1_benefit):.2f}",
                    'action': 'Consider adjusting PC1 target pips'
                })

        logger.info(f"PC analysis complete: {len(pc1_triggered)} with PC1, {len(pc2_triggered)} with PC2")
        return results

    def analyze_be_effectiveness(self, trades: List[Dict] = None) -> Optional[Dict]:
        """
        Analyze Break Even activation effectiveness.

        Analyzes:
        - Win rate when BE activated vs not
        - "BE trap" pattern detection (stopped at BE, trend continues)
        - Optimal BE trigger threshold

        Returns:
            Dict with BE effectiveness analysis and recommendations
        """
        if trades is None:
            trades = self._load_closed_trades()

        if not trades or len(trades) < 30:
            return None

        be_activated = []
        be_not_activated = []
        be_traps = 0  # Stopped at BE when could have been profitable

        for trade in trades:
            outcome = trade.get('outcome', {})
            be_data = outcome.get('break_even', {})
            sl_data = outcome.get('stop_loss', {})
            profit = outcome.get('profit', 0)
            exit_price = outcome.get('exit_price', 0)
            entry_price = trade.get('entry_price', 0)

            has_be = be_data.get('activated', False)

            if has_be:
                be_activated.append(profit)
                # Detect BE trap: exited very close to entry (within 2 pips) with small loss
                if abs(exit_price - entry_price) < 0.0002 and profit <= 0:
                    be_traps += 1
            else:
                be_not_activated.append(profit)

        results = {
            'trades_analyzed': len(trades),
            'be_stats': {
                'activated_count': len(be_activated),
                'not_activated_count': len(be_not_activated),
                'activated_avg_profit': sum(be_activated) / len(be_activated) if be_activated else 0,
                'not_activated_avg_profit': sum(be_not_activated) / len(be_not_activated) if be_not_activated else 0,
                'be_traps': be_traps,
                'be_trap_rate': be_traps / len(be_activated) * 100 if be_activated else 0
            },
            'recommendations': []
        }

        if be_traps > 5 and results['be_stats']['be_trap_rate'] > 20:
            results['recommendations'].append({
                'type': 'be_trap',
                'insight': f"{be_traps} BE traps detected ({results['be_stats']['be_trap_rate']:.0f}% of BE trades)",
                'action': 'Consider increasing BE trigger threshold or adding buffer'
            })

        logger.info(f"BE analysis complete: {len(be_activated)} with BE, {be_traps} traps")
        return results

    def analyze_strategy_selection(self, trades: List[Dict] = None) -> Optional[Dict]:
        """
        Analyze MR vs BO strategy performance by conditions.

        Analyzes:
        - Win rate per strategy overall
        - Performance by hour, ADX level, symbol
        - Conditions where each strategy excels

        Returns:
            Dict with strategy comparison and recommendations
        """
        if trades is None:
            trades = self._load_closed_trades()

        if not trades or len(trades) < 30:
            return None

        mr_trades = []
        bo_trades = []
        by_hour = defaultdict(lambda: {'mr': [], 'bo': []})
        by_adx = defaultdict(lambda: {'mr': [], 'bo': []})
        by_symbol = defaultdict(lambda: {'mr': [], 'bo': []})

        for trade in trades:
            strategy = trade.get('strategy_type', 'unknown')
            profit = trade.get('outcome', {}).get('profit', 0)
            hour = trade.get('market_context', {}).get('hour', 0)
            symbol = trade.get('symbol', 'UNKNOWN')
            adx = trade.get('trend_filter', {}).get('adx', 0)

            # ADX buckets
            if adx < 20:
                adx_bucket = 'ranging'
            elif adx < 30:
                adx_bucket = 'weak_trend'
            else:
                adx_bucket = 'strong_trend'

            if strategy in ['vwap', 'mean_reversion']:
                mr_trades.append(profit)
                by_hour[hour]['mr'].append(profit)
                by_adx[adx_bucket]['mr'].append(profit)
                by_symbol[symbol]['mr'].append(profit)
            elif strategy == 'breakout':
                bo_trades.append(profit)
                by_hour[hour]['bo'].append(profit)
                by_adx[adx_bucket]['bo'].append(profit)
                by_symbol[symbol]['bo'].append(profit)

        def calc_stats(profits):
            if not profits:
                return {'count': 0, 'win_rate': 0, 'avg_profit': 0}
            wins = sum(1 for p in profits if p > 0)
            return {
                'count': len(profits),
                'win_rate': wins / len(profits) * 100,
                'avg_profit': sum(profits) / len(profits)
            }

        results = {
            'trades_analyzed': len(trades),
            'overall': {
                'mr': calc_stats(mr_trades),
                'bo': calc_stats(bo_trades)
            },
            'by_adx': {
                bucket: {
                    'mr': calc_stats(data['mr']),
                    'bo': calc_stats(data['bo'])
                } for bucket, data in by_adx.items()
            },
            'by_symbol': {
                symbol: {
                    'mr': calc_stats(data['mr']),
                    'bo': calc_stats(data['bo'])
                } for symbol, data in by_symbol.items()
            },
            'recommendations': []
        }

        # Generate recommendations
        mr_wr = results['overall']['mr']['win_rate']
        bo_wr = results['overall']['bo']['win_rate']
        mr_count = results['overall']['mr']['count']
        bo_count = results['overall']['bo']['count']

        if mr_count >= 20 and bo_count >= 10:
            if mr_wr > bo_wr + 15:
                results['recommendations'].append({
                    'type': 'strategy_preference',
                    'insight': f"MR outperforms BO by {mr_wr - bo_wr:.0f}% win rate",
                    'action': 'Favor MR strategy, reduce BO frequency'
                })
            elif bo_wr > mr_wr + 15:
                results['recommendations'].append({
                    'type': 'strategy_preference',
                    'insight': f"BO outperforms MR by {bo_wr - mr_wr:.0f}% win rate",
                    'action': 'Favor BO strategy in current market'
                })

        # ADX-based recommendations
        for bucket, data in results['by_adx'].items():
            mr_stats = data['mr']
            bo_stats = data['bo']
            if mr_stats['count'] >= 10 and bo_stats['count'] >= 5:
                if mr_stats['win_rate'] > bo_stats['win_rate'] + 20:
                    results['recommendations'].append({
                        'type': 'adx_strategy',
                        'condition': bucket,
                        'insight': f"In {bucket} markets: MR wins {mr_stats['win_rate']:.0f}% vs BO {bo_stats['win_rate']:.0f}%",
                        'action': f"Disable BO in {bucket} conditions"
                    })

        logger.info(f"Strategy analysis complete: {mr_count} MR, {bo_count} BO trades")
        return results

    def _load_closed_trades(self) -> List[Dict]:
        """Load closed trades from SQLite (fallback: JSONL)"""
        # Try SQLite first
        try:
            from ml_system.trade_db import get_trade_db
            db = get_trade_db()
            if db:
                return db.get_closed_trades()
        except Exception as e:
            logger.warning(f"SQLite read failed, falling back to JSONL: {e}")

        # Fallback: parse JSONL
        trade_log_path = self.ml_outputs_dir / 'continuous_trade_log.jsonl'

        if not trade_log_path.exists():
            return []

        trades = []
        try:
            with open(trade_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        trade = json.loads(line)
                        if trade.get('outcome', {}).get('status') == 'closed':
                            trades.append(trade)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to load trade log: {e}")

        return trades

    def _load_current_config(self) -> Dict:
        """Load current strategy config values"""
        config = {}

        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            return config

        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract HTF weights
        htf_match = re.search(r'HTF_CONFLUENCE_WEIGHTS\s*=\s*\{([^}]+)\}', content, re.DOTALL)
        if htf_match:
            config['htf_weights'] = self._parse_weights_dict(htf_match.group(1))

        # Extract Entry weights
        entry_match = re.search(r'ENTRY_CONFLUENCE_WEIGHTS\s*=\s*\{([^}]+)\}', content, re.DOTALL)
        if entry_match:
            config['entry_weights'] = self._parse_weights_dict(entry_match.group(1))

        # Extract thresholds
        for param in ['MIN_HTF_CONFLUENCE_SCORE', 'MIN_ENTRY_CONFLUENCE_SCORE',
                      'MIN_CONFLUENCE_SCORE', 'OPTIMAL_CONFLUENCE_SCORE']:
            match = re.search(rf'{param}\s*=\s*(\d+)', content)
            if match:
                config[param.lower()] = int(match.group(1))

        # Extract trading hours
        mr_hours_match = re.search(r'MEAN_REVERSION_HOURS\s*=\s*\[([^\]]+)\]', content)
        if mr_hours_match:
            config['mean_reversion_hours'] = [int(x.strip()) for x in mr_hours_match.group(1).split(',')]

        # Extract spread hours
        spread_match = re.search(r'SPREAD_HOURS\s*=\s*\[([^\]]+)\]', content)
        if spread_match:
            config['spread_hours'] = [int(x.strip()) for x in spread_match.group(1).split(',')]

        # Extract DI direction filter parameters
        di_enabled_match = re.search(r'ENABLE_DI_DIRECTION_FILTER\s*=\s*(True|False)', content)
        if di_enabled_match:
            config['enable_di_direction_filter'] = di_enabled_match.group(1) == 'True'

        di_sep_match = re.search(r'DI_DIRECTION_MIN_SEPARATION\s*=\s*([\d.]+)', content)
        if di_sep_match:
            config['di_direction_min_separation'] = float(di_sep_match.group(1))

        # Extract SMC direction filter parameters
        smc_enabled_match = re.search(r'ENABLE_SMC_DIRECTION_FILTER\s*=\s*(True|False)', content)
        if smc_enabled_match:
            config['enable_smc_direction_filter'] = smc_enabled_match.group(1) == 'True'

        smc_htf_match = re.search(r'SMC_FILTER_REQUIRE_HTF\s*=\s*(True|False)', content)
        if smc_htf_match:
            config['smc_filter_require_htf'] = smc_htf_match.group(1) == 'True'

        # Extract per-symbol override dicts
        for dict_name in ['SYMBOL_HTF_OVERRIDES', 'SYMBOL_ENTRY_OVERRIDES',
                          'SYMBOL_DI_SETTINGS', 'SYMBOL_SMC_SETTINGS',
                          'SYMBOL_ADX_SETTINGS']:
            pattern = rf'{dict_name}\s*=\s*(\{{[^}}]*\{{[^}}]*\}}[^}}]*\}})'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    # Safe eval for simple nested dicts
                    import ast
                    config[dict_name.lower()] = ast.literal_eval(match.group(1))
                except (ValueError, SyntaxError):
                    pass

        logger.info(f"Loaded current config with {len(config)} parameters")
        return config

    def _parse_weights_dict(self, dict_str: str) -> Dict[str, int]:
        """Parse a Python dict string into a dictionary"""
        weights = {}
        # Match patterns like 'factor_name': 2, or "factor_name": 2
        pattern = r"['\"]([^'\"]+)['\"]\s*:\s*(\d+)"
        for match in re.finditer(pattern, dict_str):
            weights[match.group(1)] = int(match.group(2))
        return weights

    def _calculate_recommendations(self, ml_data: Dict, current_config: Dict) -> Dict:
        """
        Calculate recommended parameter changes based on ML analysis

        Returns:
            Dict with recommended changes
        """
        recommendations = {
            'htf_weights': {},
            'entry_weights': {},
            'thresholds': {},
            'hours': {},
            'di_filter': {},
            'smc_filter': {},
            'per_symbol': {},  # Per-symbol weight overrides, DI/SMC settings
        }

        # 1. Analyze factor performance and recommend weight changes
        if 'factor_stats' in ml_data:
            self._recommend_weight_changes(
                ml_data['factor_stats'],
                current_config.get('htf_weights', {}),
                current_config.get('entry_weights', {}),
                recommendations
            )

        # 2. Analyze confluence quality and recommend threshold changes
        if 'confluence' in ml_data:
            self._recommend_threshold_changes(
                ml_data['confluence'],
                current_config,
                recommendations
            )

        # 3. Analyze spread hours and recommend hour changes
        if 'spread_hours' in ml_data:
            self._recommend_hour_changes(
                ml_data['spread_hours'],
                current_config,
                recommendations
            )

        # 4. Analyze DI direction filter effectiveness (global)
        self._recommend_di_filter_changes(current_config, recommendations)

        # 5. Analyze SMC direction filter effectiveness (global)
        self._recommend_smc_filter_changes(current_config, recommendations)

        # 6. Per-symbol analysis using factor_effectiveness by_symbol data
        if 'factor_effectiveness' in ml_data:
            self._calculate_per_symbol_recommendations(
                ml_data['factor_effectiveness'],
                current_config,
                recommendations
            )

        # 7. Per-symbol DI/SMC filter analysis
        self._recommend_per_symbol_di_smc(current_config, recommendations)

        # 8. Per-symbol ADX threshold analysis
        self._recommend_adx_settings(current_config, recommendations)

        # 9. PC1/PC2 partial close effectiveness analysis
        self._recommend_pc_settings(current_config, recommendations)

        # 10. Factor timing patterns (day-of-week, hour-of-day analysis)
        if 'timing_patterns' in ml_data:
            self._recommend_timing_based_adjustments(
                ml_data['timing_patterns'],
                current_config,
                recommendations
            )

        return recommendations

    def _recommend_weight_changes(
        self,
        factor_stats: Dict,
        current_htf: Dict,
        current_entry: Dict,
        recommendations: Dict
    ):
        """Recommend weight changes based on factor performance"""

        # Mapping from training_data.csv column names to strategy_config.py factor names
        # training_data uses prefixes like 'at_' for binary presence indicators
        training_to_config = {
            # HTF factors
            'at_prev_day_vah': 'prev_day_vah',
            'at_prev_day_val': 'prev_day_val',
            'at_prev_day_poc': 'prev_day_poc',
            'at_prev_day_high': 'prev_day_high',
            'at_prev_day_low': 'prev_day_low',
            'at_daily_hvn': 'daily_hvn',
            'at_weekly_hvn': 'weekly_hvn',
            'at_prev_week_high': 'prev_week_high',
            'at_prev_week_low': 'prev_week_low',
            # Entry factors
            'vwap_band_1': 'vwap_band_1',
            'vwap_band_2': 'vwap_band_2',
            'vwap_band_3': 'vwap_band_3',
            'at_poc': 'poc',
            'at_lvn': 'lvn',
            'above_vah': 'above_vah',
            'below_val': 'below_val',
            'at_swing_high': 'swing_high',
            'at_swing_low': 'swing_low',
            # Liquidity factors (NEW)
            'at_session_high': 'session_high',
            'at_session_low': 'session_low',
            'at_equal_highs': 'equal_highs',
            'at_equal_lows': 'equal_lows',
            'at_bullish_ob': 'bullish_ob',
            'at_bearish_ob': 'bearish_ob',
        }

        # HTF factors (config names)
        htf_factors = [
            'prev_day_vah', 'prev_day_val', 'prev_day_poc',
            'prev_day_high', 'prev_day_low', 'daily_hvn',
            'daily_swing_high', 'daily_swing_low',
            'weekly_hvn', 'weekly_poc',
            'prev_week_high', 'prev_week_low',
            'prev_week_swing_high', 'prev_week_swing_low', 'prev_week_vwap'
        ]

        # Entry factors (config names)
        entry_factors = [
            'vwap_band_1', 'vwap_band_2', 'vwap_band_3',
            'poc', 'lvn', 'above_vah', 'below_val',
            'swing_high', 'swing_low',
            # Liquidity factors (NEW)
            'session_high', 'session_low',
            'equal_highs', 'equal_lows',
            'bullish_ob', 'bearish_ob',
        ]

        # Build reverse mapping (config name -> training data name)
        config_to_training = {v: k for k, v in training_to_config.items()}

        # Process HTF weights
        for config_factor in htf_factors:
            # Find the stats - try config name first, then mapped training name
            training_factor = config_to_training.get(config_factor, config_factor)
            stats = factor_stats.get(training_factor) or factor_stats.get(config_factor)

            if stats is None:
                continue

            current_weight = current_htf.get(config_factor, 1)
            recommended = stats['recommended_weight']

            # Apply constraints
            change = self._constrain_change(
                current_weight, recommended,
                stats['win_rate'], stats['trades']
            )

            if change != 0:
                new_weight = max(1, min(5, current_weight + change))
                recommendations['htf_weights'][config_factor] = {
                    'current': current_weight,
                    'recommended': new_weight,
                    'change': change,
                    'reason': f"Win rate: {stats['win_rate']}%, Trades: {stats['trades']}"
                }

        # Process Entry weights
        for config_factor in entry_factors:
            # Special case: vwap_band_3 should stay at 0 per user request
            if config_factor == 'vwap_band_3':
                continue

            # Find the stats - try config name first, then mapped training name
            training_factor = config_to_training.get(config_factor, config_factor)
            stats = factor_stats.get(training_factor) or factor_stats.get(config_factor)

            if stats is None:
                continue

            current_weight = current_entry.get(config_factor, 1)
            recommended = stats['recommended_weight']

            change = self._constrain_change(
                current_weight, recommended,
                stats['win_rate'], stats['trades']
            )

            if change != 0:
                new_weight = max(0, min(3, current_weight + change))
                recommendations['entry_weights'][config_factor] = {
                    'current': current_weight,
                    'recommended': new_weight,
                    'change': change,
                    'reason': f"Win rate: {stats['win_rate']}%, Trades: {stats['trades']}"
                }

    def _constrain_change(
        self,
        current: int,
        recommended: int,
        win_rate: float,
        trades: int
    ) -> int:
        """Apply safety constraints to weight changes"""

        # Not enough data
        if trades < self.CONSTRAINTS['min_trades_for_factor']:
            return 0

        # Calculate desired change
        desired_change = recommended - current

        # Limit change magnitude
        if abs(desired_change) > self.CONSTRAINTS['max_weight_change']:
            desired_change = self.CONSTRAINTS['max_weight_change'] if desired_change > 0 else -self.CONSTRAINTS['max_weight_change']

        # Only increase if win rate is good enough
        if desired_change > 0 and win_rate < self.CONSTRAINTS['min_win_rate_for_increase']:
            return 0

        # Only decrease if win rate is bad enough
        if desired_change < 0 and win_rate > self.CONSTRAINTS['max_win_rate_for_decrease']:
            return 0

        return desired_change

    def _recommend_threshold_changes(
        self,
        confluence_data: Dict,
        current_config: Dict,
        recommendations: Dict
    ):
        """Recommend threshold changes based on confluence quality analysis"""

        # Get threshold analysis from confluence data
        threshold_analysis = confluence_data.get('optimal_threshold', {}).get('threshold_analysis', {})

        if not threshold_analysis:
            return

        # Find optimal thresholds
        best_score = 0
        best_threshold = current_config.get('min_confluence_score', 7)

        for threshold_str, stats in threshold_analysis.items():
            threshold = int(threshold_str)
            score = stats.get('score', 0)
            trades = stats.get('total_trades', 0)

            # Need minimum trades
            if trades < self.CONSTRAINTS['min_trades_for_threshold']:
                continue

            if score > best_score:
                best_score = score
                best_threshold = threshold

        # Recommend threshold change if different
        current_combined = current_config.get('min_confluence_score', 7)
        if best_threshold != current_combined:
            # Constrain to safe range
            new_threshold = max(
                self.CONSTRAINTS['min_combined_score'],
                min(self.CONSTRAINTS['max_combined_score'], best_threshold)
            )

            if new_threshold != current_combined:
                recommendations['thresholds']['min_confluence_score'] = {
                    'current': current_combined,
                    'recommended': new_threshold,
                    'reason': f"Best score {best_score:.1f} at threshold {best_threshold}"
                }

    def _recommend_hour_changes(
        self,
        spread_data: Dict,
        current_config: Dict,
        recommendations: Dict
    ):
        """Recommend trading hour changes based on spread analysis"""

        spread_hours = spread_data.get('spread_hours_analysis', {}).get('spread_hour_numbers', [])
        best_hours = spread_data.get('trading_hours_analysis', {}).get('recommended_hours', [])

        if spread_hours:
            current_spread = current_config.get('spread_hours', [])
            if set(spread_hours) != set(current_spread):
                recommendations['hours']['spread_hours'] = {
                    'current': current_spread,
                    'recommended': spread_hours,
                    'reason': 'Updated from spread hours analysis'
                }

        if best_hours:
            current_mr_hours = current_config.get('mean_reversion_hours', [])
            # Only recommend if significantly different
            overlap = set(best_hours) & set(current_mr_hours)
            if len(overlap) / max(len(best_hours), len(current_mr_hours), 1) < 0.7:
                recommendations['hours']['mean_reversion_hours'] = {
                    'current': current_mr_hours,
                    'recommended': best_hours,
                    'reason': 'Updated from profitable hours analysis'
                }

    def _recommend_di_filter_changes(self, current_config: Dict, recommendations: Dict):
        """
        Analyze trade log to determine if DI directional filter is helping.
        Adjusts DI_DIRECTION_MIN_SEPARATION based on win rates of DI-aligned vs DI-conflicting trades.
        """
        # Load closed trades from SQLite (fallback: JSONL)
        closed_trades = self._load_closed_trades()
        if not closed_trades:
            return

        try:
            aligned_wins = 0
            aligned_total = 0
            conflicting_wins = 0
            conflicting_total = 0

            for trade in closed_trades:
                try:
                    outcome = trade.get('outcome', {})

                    direction = trade.get('direction', '').upper()
                    trend = trade.get('trend_filter', {})
                    plus_di = trend.get('plus_di', 0)
                    minus_di = trend.get('minus_di', 0)
                    profit = outcome.get('profit', 0)

                    if plus_di == 0 and minus_di == 0:
                        continue

                    # Determine if trade was DI-aligned
                    di_aligned = (
                        (direction == 'BUY' and plus_di > minus_di) or
                        (direction == 'SELL' and minus_di > plus_di)
                    )
                    is_win = profit > 0

                    if di_aligned:
                        aligned_total += 1
                        if is_win:
                            aligned_wins += 1
                    else:
                        conflicting_total += 1
                        if is_win:
                            conflicting_wins += 1
                except (KeyError, ValueError):
                    continue

            min_trades = self.CONSTRAINTS['min_trades_for_factor']

            if aligned_total >= min_trades and conflicting_total >= min_trades:
                aligned_wr = (aligned_wins / aligned_total) * 100
                conflicting_wr = (conflicting_wins / conflicting_total) * 100

                logger.info(f"DI Analysis: Aligned={aligned_wr:.1f}% ({aligned_total} trades), "
                           f"Conflicting={conflicting_wr:.1f}% ({conflicting_total} trades)")

                current_sep = current_config.get('di_direction_min_separation', 5.0)
                current_enabled = current_config.get('enable_di_direction_filter', True)

                # If conflicting trades have much lower win rate, tighten the filter
                wr_gap = aligned_wr - conflicting_wr

                if wr_gap >= 15:
                    # Strong evidence: DI-aligned trades win much more often
                    # Lower the separation threshold to filter more aggressively
                    recommended_sep = max(3.0, current_sep - 1.0)
                    if not current_enabled:
                        recommendations['di_filter']['enable_di_direction_filter'] = {
                            'current': False,
                            'recommended': True,
                            'reason': f"DI-aligned WR:{aligned_wr:.0f}% vs conflicting WR:{conflicting_wr:.0f}% (gap:{wr_gap:.0f}%)"
                        }
                    if recommended_sep != current_sep:
                        recommendations['di_filter']['di_direction_min_separation'] = {
                            'current': current_sep,
                            'recommended': recommended_sep,
                            'reason': f"Strong DI signal (WR gap:{wr_gap:.0f}%). Tightening filter."
                        }
                elif wr_gap >= 5:
                    # Moderate evidence: keep current settings, enable if disabled
                    if not current_enabled:
                        recommendations['di_filter']['enable_di_direction_filter'] = {
                            'current': False,
                            'recommended': True,
                            'reason': f"Moderate DI signal (aligned:{aligned_wr:.0f}% vs conflicting:{conflicting_wr:.0f}%)"
                        }
                elif wr_gap < 0:
                    # DI filter not helping — widen threshold or disable
                    if current_sep < 15.0:
                        recommended_sep = min(15.0, current_sep + 2.0)
                        recommendations['di_filter']['di_direction_min_separation'] = {
                            'current': current_sep,
                            'recommended': recommended_sep,
                            'reason': f"DI filter not effective (aligned:{aligned_wr:.0f}% vs conflicting:{conflicting_wr:.0f}%). Loosening."
                        }
                    elif current_enabled and conflicting_total >= min_trades * 2:
                        recommendations['di_filter']['enable_di_direction_filter'] = {
                            'current': True,
                            'recommended': False,
                            'reason': f"DI filter hurting (conflicting WR:{conflicting_wr:.0f}% > aligned:{aligned_wr:.0f}%). Disabling."
                        }

        except Exception as e:
            logger.warning(f"DI filter analysis failed: {e}")

    def _recommend_smc_filter_changes(self, current_config: Dict, recommendations: Dict):
        """
        Analyze SMC trade correlation log to determine if SMC filter is effective.
        Compares win rates of SMC-aligned vs SMC-conflicting trades.
        """
        smc_log_path = self.ml_outputs_dir / 'smc_trade_correlation.jsonl'

        if not smc_log_path.exists():
            return

        try:
            # Build ticket -> SMC alignment map from correlation log
            ticket_smc = {}
            with open(smc_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        alignment = entry.get('smc_state', {}).get('alignment', 'unknown')
                        # Map timestamp to find matching trades
                        ticket_smc[entry.get('timestamp', '')] = {
                            'alignment': alignment,
                            'direction': entry.get('direction', ''),
                            'symbol': entry.get('symbol', ''),
                        }
                    except (json.JSONDecodeError, KeyError):
                        continue

            if len(ticket_smc) < self.CONSTRAINTS['min_trades_for_factor']:
                logger.info(f"SMC filter: Not enough correlated data ({len(ticket_smc)} entries, need {self.CONSTRAINTS['min_trades_for_factor']})")
                return

            # Count aligned vs conflicting outcomes from correlation entries
            aligned_wins = 0
            aligned_total = 0
            conflicting_wins = 0
            conflicting_total = 0

            # Read trade outcomes from SQLite (fallback: JSONL)
            closed_trades = self._load_closed_trades()

            for trade in closed_trades:
                try:
                    outcome = trade.get('outcome', {})

                    direction = trade.get('direction', '').lower()
                    profit = outcome.get('profit', 0)

                    # Check SMC state if available on this trade
                    smc_state = trade.get('smc_state', {})
                    if not smc_state or not smc_state.get('available'):
                        continue

                    alignment = smc_state.get('alignment', 'neutral')
                    is_win = profit > 0

                    # Determine if trade aligned with SMC
                    smc_aligned = (
                        (direction == 'buy' and 'bullish' in alignment) or
                        (direction == 'sell' and 'bearish' in alignment)
                    )
                    smc_conflicting = (
                        (direction == 'buy' and 'bearish' in alignment) or
                        (direction == 'sell' and 'bullish' in alignment)
                    )

                    if smc_aligned:
                        aligned_total += 1
                        if is_win:
                            aligned_wins += 1
                    elif smc_conflicting:
                        conflicting_total += 1
                        if is_win:
                            conflicting_wins += 1
                except (KeyError, ValueError):
                    continue

            min_trades = self.CONSTRAINTS['min_trades_for_factor']
            current_enabled = current_config.get('enable_smc_direction_filter', True)

            if aligned_total >= min_trades and conflicting_total >= min_trades:
                aligned_wr = (aligned_wins / aligned_total) * 100
                conflicting_wr = (conflicting_wins / conflicting_total) * 100
                wr_gap = aligned_wr - conflicting_wr

                logger.info(f"SMC Analysis: Aligned={aligned_wr:.1f}% ({aligned_total} trades), "
                           f"Conflicting={conflicting_wr:.1f}% ({conflicting_total} trades)")

                if wr_gap >= 10 and not current_enabled:
                    recommendations['smc_filter']['enable_smc_direction_filter'] = {
                        'current': False,
                        'recommended': True,
                        'reason': f"SMC-aligned WR:{aligned_wr:.0f}% vs conflicting WR:{conflicting_wr:.0f}% (gap:{wr_gap:.0f}%)"
                    }
                elif wr_gap < -5 and current_enabled and conflicting_total >= min_trades * 2:
                    recommendations['smc_filter']['enable_smc_direction_filter'] = {
                        'current': True,
                        'recommended': False,
                        'reason': f"SMC filter not effective (conflicting WR:{conflicting_wr:.0f}% > aligned:{aligned_wr:.0f}%). Disabling."
                    }
            else:
                logger.info(f"SMC filter: Not enough closed trades with SMC data "
                           f"(aligned:{aligned_total}, conflicting:{conflicting_total})")

        except Exception as e:
            logger.warning(f"SMC filter analysis failed: {e}")

    def _calculate_per_symbol_recommendations(
        self,
        factor_data: Dict,
        current_config: Dict,
        recommendations: Dict
    ):
        """
        Calculate per-symbol weight overrides using factor_effectiveness by_symbol data.
        Compares per-symbol win rates against global weights and generates overrides.
        """
        by_symbol = factor_data.get('by_symbol', {})
        symbol_rankings = factor_data.get('symbol_rankings', {})
        global_stats = factor_data.get('global', {})
        min_trades = self.CONSTRAINTS['min_trades_for_factor']

        # Current per-symbol overrides
        current_htf_overrides = current_config.get('symbol_htf_overrides', {})
        current_entry_overrides = current_config.get('symbol_entry_overrides', {})
        current_htf_weights = current_config.get('htf_weights', {})
        current_entry_weights = current_config.get('entry_weights', {})

        # Factor name mapping (factor_effectiveness uses full names)
        factor_to_htf = {
            'prev_day_vah': 'prev_day_vah', 'prev_day_val': 'prev_day_val',
            'prev_day_poc': 'prev_day_poc', 'prev_day_high': 'prev_day_high',
            'prev_day_low': 'prev_day_low', 'daily_hvn': 'daily_hvn',
            'weekly_hvn': 'weekly_hvn', 'weekly_poc': 'weekly_poc',
            'prev_week_high': 'prev_week_high', 'prev_week_low': 'prev_week_low',
            'prev_week_swing_high': 'prev_week_swing_high',
            'prev_week_swing_low': 'prev_week_swing_low',
        }
        factor_to_entry = {
            'VWAP Band 1': 'vwap_band_1', 'VWAP Band 2': 'vwap_band_2',
            'POC': 'poc', 'Low Volume Node': 'lvn',
            'Above VAH': 'above_vah', 'Below VAL': 'below_val',
            'Swing High': 'swing_high', 'Swing Low': 'swing_low',
            'Session High': 'session_high', 'Session Low': 'session_low',
            'Equal Highs (EQH)': 'equal_highs', 'Equal Lows (EQL)': 'equal_lows',
        }

        for symbol, sym_stats in by_symbol.items():
            htf_overrides = {}
            entry_overrides = {}

            for factor_name, stats in sym_stats.items():
                total = stats.get('total', 0)
                if total < min_trades:
                    continue

                win_rate = stats.get('win_rate', 50)

                # Check if this is an HTF factor
                config_name = factor_to_htf.get(factor_name)
                if config_name:
                    global_weight = current_htf_weights.get(config_name, 2)
                    recommended = self._importance_to_weight(win_rate)
                    change = self._constrain_change(global_weight, recommended, win_rate, total)
                    if change != 0:
                        htf_overrides[config_name] = max(0, min(5, global_weight + change))

                # Check if this is an entry factor
                config_name = factor_to_entry.get(factor_name)
                if config_name:
                    global_weight = current_entry_weights.get(config_name, 2)
                    recommended = self._importance_to_weight(win_rate)
                    change = self._constrain_change(global_weight, recommended, win_rate, total)
                    if change != 0:
                        entry_overrides[config_name] = max(0, min(3, global_weight + change))

            if htf_overrides or entry_overrides:
                recommendations['per_symbol'][symbol] = {
                    'htf_overrides': htf_overrides,
                    'entry_overrides': entry_overrides,
                }
                logger.info(f"Per-symbol {symbol}: HTF overrides={htf_overrides}, Entry overrides={entry_overrides}")

    def _recommend_per_symbol_di_smc(self, current_config: Dict, recommendations: Dict):
        """
        Analyze DI and SMC filter effectiveness per symbol.
        Generates per-symbol DI_SETTINGS and SMC_SETTINGS recommendations.
        """
        # Load closed trades from SQLite (fallback: JSONL)
        closed_trades = self._load_closed_trades()
        if not closed_trades:
            return

        try:
            # Per-symbol DI stats
            di_stats = defaultdict(lambda: {'aligned_wins': 0, 'aligned_total': 0,
                                             'conflicting_wins': 0, 'conflicting_total': 0})
            # Per-symbol SMC stats
            smc_stats = defaultdict(lambda: {'aligned_wins': 0, 'aligned_total': 0,
                                              'conflicting_wins': 0, 'conflicting_total': 0})

            for trade in closed_trades:
                try:
                    outcome = trade.get('outcome', {})

                    sym = trade.get('symbol', '')
                    direction = trade.get('direction', '').upper()
                    profit = outcome.get('profit', 0)
                    is_win = profit > 0

                    # DI analysis
                    trend = trade.get('trend_filter', {})
                    plus_di = trend.get('plus_di', 0)
                    minus_di = trend.get('minus_di', 0)
                    if plus_di > 0 or minus_di > 0:
                        di_aligned = (
                            (direction == 'BUY' and plus_di > minus_di) or
                            (direction == 'SELL' and minus_di > plus_di)
                        )
                        s = di_stats[sym]
                        if di_aligned:
                            s['aligned_total'] += 1
                            if is_win:
                                s['aligned_wins'] += 1
                        else:
                            s['conflicting_total'] += 1
                            if is_win:
                                s['conflicting_wins'] += 1

                    # SMC analysis
                    smc_state = trade.get('smc_state', {})
                    if smc_state and smc_state.get('available'):
                        alignment = smc_state.get('alignment', 'neutral')
                        dir_lower = direction.lower()
                        smc_aligned = (
                            (dir_lower == 'buy' and 'bullish' in alignment) or
                            (dir_lower == 'sell' and 'bearish' in alignment)
                        )
                        smc_conflicting = (
                            (dir_lower == 'buy' and 'bearish' in alignment) or
                            (dir_lower == 'sell' and 'bullish' in alignment)
                        )
                        ss = smc_stats[sym]
                        if smc_aligned:
                            ss['aligned_total'] += 1
                            if is_win:
                                ss['aligned_wins'] += 1
                        elif smc_conflicting:
                            ss['conflicting_total'] += 1
                            if is_win:
                                ss['conflicting_wins'] += 1
                except (KeyError, ValueError):
                    continue

            min_trades = self.CONSTRAINTS['min_trades_for_factor']
            current_di_settings = current_config.get('symbol_di_settings', {})
            current_smc_settings = current_config.get('symbol_smc_settings', {})
            per_sym = recommendations.setdefault('per_symbol', {})

            for sym, stats in di_stats.items():
                if stats['aligned_total'] >= min_trades and stats['conflicting_total'] >= min_trades:
                    aligned_wr = (stats['aligned_wins'] / stats['aligned_total']) * 100
                    conflicting_wr = (stats['conflicting_wins'] / stats['conflicting_total']) * 100
                    wr_gap = aligned_wr - conflicting_wr

                    sym_di = current_di_settings.get(sym, current_di_settings.get('DEFAULT', {}))
                    current_sep = sym_di.get('min_separation', 3.0)

                    new_di = {}
                    if wr_gap >= 15:
                        new_di = {'enabled': True, 'min_separation': max(2.0, current_sep - 1.0)}
                    elif wr_gap < 0:
                        new_di = {'enabled': True, 'min_separation': min(15.0, current_sep + 2.0)}

                    if new_di:
                        entry = per_sym.setdefault(sym, {})
                        entry['di_settings'] = new_di
                        logger.info(f"Per-symbol DI {sym}: gap={wr_gap:.0f}% -> {new_di}")

            for sym, stats in smc_stats.items():
                if stats['aligned_total'] >= min_trades and stats['conflicting_total'] >= min_trades:
                    aligned_wr = (stats['aligned_wins'] / stats['aligned_total']) * 100
                    conflicting_wr = (stats['conflicting_wins'] / stats['conflicting_total']) * 100
                    wr_gap = aligned_wr - conflicting_wr

                    new_smc = {}
                    if wr_gap >= 10:
                        new_smc = {'enabled': True, 'require_htf': True}
                    elif wr_gap < -5:
                        new_smc = {'enabled': False, 'require_htf': True}

                    if new_smc:
                        entry = per_sym.setdefault(sym, {})
                        entry['smc_settings'] = new_smc
                        logger.info(f"Per-symbol SMC {sym}: gap={wr_gap:.0f}% -> {new_smc}")

        except Exception as e:
            logger.warning(f"Per-symbol DI/SMC analysis failed: {e}")

    def _recommend_adx_settings(self, current_config: Dict, recommendations: Dict):
        """
        Analyze ADX sweet spots per symbol and recommend SYMBOL_ADX_SETTINGS changes.
        Buckets trades by symbol + 5-point ADX range + strategy type (MR/BO).
        Safety: min 10 trades per bucket, max ±5 change per tuning run, absolute bounds enforced.
        """
        closed_trades = self._load_closed_trades()
        if not closed_trades:
            return

        try:
            # Bucket trades by symbol + ADX range + strategy type
            # ADX ranges: 0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30-35, 35-40, 40-45, 45+
            from collections import defaultdict
            buckets = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0.0})

            for trade in closed_trades:
                try:
                    outcome = trade.get('outcome', {})
                    sym = trade.get('symbol', '')
                    profit = outcome.get('profit', 0)
                    is_win = profit > 0

                    # Get ADX at entry from trend_filter
                    trend = trade.get('trend_filter', {})
                    adx = trend.get('adx', 0)
                    if adx <= 0:
                        continue

                    # Determine strategy type
                    strategy = trade.get('strategy', 'mean_reversion')
                    strat_type = 'BO' if 'breakout' in strategy.lower() else 'MR'

                    # 5-point bucket
                    bucket = int(adx // 5) * 5
                    key = (sym, strat_type, bucket)

                    buckets[key]['total'] += 1
                    buckets[key]['pnl'] += profit
                    if is_win:
                        buckets[key]['wins'] += 1
                except (KeyError, ValueError, TypeError):
                    continue

            if not buckets:
                return

            MIN_TRADES_PER_BUCKET = 10
            MAX_CHANGE_PER_RUN = 5
            # Absolute bounds
            ABS_MR_THRESHOLD_MIN = 15
            ABS_MR_THRESHOLD_MAX = 35
            ABS_MR_MAX_ADX_MIN = 25
            ABS_MR_MAX_ADX_MAX = 50
            ABS_BO_MIN_MIN = 15
            ABS_BO_MIN_MAX = 35
            ABS_BO_MAX_MIN = 30
            ABS_BO_MAX_MAX = 55

            current_adx_settings = current_config.get('symbol_adx_settings', {})
            per_sym = recommendations.setdefault('per_symbol', {})

            for sym in ['EURUSD', 'GBPUSD']:
                sym_current = current_adx_settings.get(sym, current_adx_settings.get('DEFAULT', {}))
                cur_mr_threshold = sym_current.get('mr_adx_threshold', 25)
                cur_mr_max = sym_current.get('mr_max_adx', 40)
                cur_bo_min = sym_current.get('bo_adx_min', 25)
                cur_bo_max = sym_current.get('bo_adx_max', 40)

                new_settings = {}

                # --- MR analysis ---
                mr_buckets = []
                for (s, st, b), stats in buckets.items():
                    if s == sym and st == 'MR' and stats['total'] >= MIN_TRADES_PER_BUCKET:
                        wr = (stats['wins'] / stats['total']) * 100
                        mr_buckets.append((b, stats['total'], wr, stats['pnl']))

                if mr_buckets:
                    mr_buckets.sort(key=lambda x: x[0])  # Sort by ADX range

                    # Find mr_max_adx: highest bucket where WR > 50% AND P/L > 0
                    best_max = cur_mr_max
                    for bucket, total, wr, pnl in reversed(mr_buckets):
                        if wr > 50 and pnl > 0:
                            # Upper edge of this bucket
                            best_max = bucket + 5
                            break

                    # Find mr_adx_threshold: where WR drops below 55%
                    best_threshold = cur_mr_threshold
                    for bucket, total, wr, pnl in mr_buckets:
                        if wr >= 55:
                            best_threshold = bucket + 5
                            break

                    # Apply max ±5 change constraint
                    rec_mr_max = max(cur_mr_max - MAX_CHANGE_PER_RUN,
                                     min(cur_mr_max + MAX_CHANGE_PER_RUN, best_max))
                    rec_mr_threshold = max(cur_mr_threshold - MAX_CHANGE_PER_RUN,
                                           min(cur_mr_threshold + MAX_CHANGE_PER_RUN, best_threshold))

                    # Enforce absolute bounds
                    rec_mr_max = max(ABS_MR_MAX_ADX_MIN, min(ABS_MR_MAX_ADX_MAX, rec_mr_max))
                    rec_mr_threshold = max(ABS_MR_THRESHOLD_MIN, min(ABS_MR_THRESHOLD_MAX, rec_mr_threshold))

                    # Ensure threshold < max_adx
                    if rec_mr_threshold >= rec_mr_max:
                        rec_mr_threshold = rec_mr_max - 5

                    if rec_mr_max != cur_mr_max or rec_mr_threshold != cur_mr_threshold:
                        new_settings['mr_adx_threshold'] = rec_mr_threshold
                        new_settings['mr_max_adx'] = rec_mr_max
                        logger.info(f"ADX MR {sym}: threshold {cur_mr_threshold}->{rec_mr_threshold}, "
                                    f"max {cur_mr_max}->{rec_mr_max} "
                                    f"(buckets: {[(b, t, f'{w:.0f}%', f'${p:.1f}') for b, t, w, p in mr_buckets]})")

                # --- BO analysis ---
                bo_buckets = []
                for (s, st, b), stats in buckets.items():
                    if s == sym and st == 'BO' and stats['total'] >= MIN_TRADES_PER_BUCKET:
                        wr = (stats['wins'] / stats['total']) * 100
                        bo_buckets.append((b, stats['total'], wr, stats['pnl']))

                if bo_buckets:
                    bo_buckets.sort(key=lambda x: x[0])

                    # Find profitable BO range
                    profitable = [(b, t, w, p) for b, t, w, p in bo_buckets if w > 50 and p > 0]
                    if profitable:
                        best_bo_min = profitable[0][0]
                        best_bo_max = profitable[-1][0] + 5

                        rec_bo_min = max(cur_bo_min - MAX_CHANGE_PER_RUN,
                                         min(cur_bo_min + MAX_CHANGE_PER_RUN, best_bo_min))
                        rec_bo_max = max(cur_bo_max - MAX_CHANGE_PER_RUN,
                                         min(cur_bo_max + MAX_CHANGE_PER_RUN, best_bo_max))

                        rec_bo_min = max(ABS_BO_MIN_MIN, min(ABS_BO_MIN_MAX, rec_bo_min))
                        rec_bo_max = max(ABS_BO_MAX_MIN, min(ABS_BO_MAX_MAX, rec_bo_max))

                        if rec_bo_min >= rec_bo_max:
                            rec_bo_min = rec_bo_max - 5

                        if rec_bo_min != cur_bo_min or rec_bo_max != cur_bo_max:
                            new_settings['bo_adx_min'] = rec_bo_min
                            new_settings['bo_adx_max'] = rec_bo_max
                            logger.info(f"ADX BO {sym}: min {cur_bo_min}->{rec_bo_min}, "
                                        f"max {cur_bo_max}->{rec_bo_max}")

                # Only emit recommendation if something changed
                if new_settings:
                    # Merge with current values for unchanged fields
                    full_settings = {
                        'mr_adx_threshold': new_settings.get('mr_adx_threshold', cur_mr_threshold),
                        'mr_max_adx': new_settings.get('mr_max_adx', cur_mr_max),
                        'bo_adx_min': new_settings.get('bo_adx_min', cur_bo_min),
                        'bo_adx_max': new_settings.get('bo_adx_max', cur_bo_max),
                    }
                    entry = per_sym.setdefault(sym, {})
                    entry['adx_settings'] = full_settings
                    logger.info(f"Per-symbol ADX {sym}: {full_settings}")

        except Exception as e:
            logger.warning(f"Per-symbol ADX analysis failed: {e}")

    def _recommend_pc_settings(self, current_config: Dict, recommendations: Dict):
        """
        Analyze PC1/PC2 partial close effectiveness per symbol.
        Tracks: PC1 conversion to PC2, profit after PC1 vs no-PC trades,
        optimal close percentages.
        Writes recommendations to tuning history for review.
        """
        closed_trades = self._load_closed_trades()
        if not closed_trades:
            return

        try:
            from collections import defaultdict

            # Per-symbol PC stats
            pc_stats = defaultdict(lambda: {
                'total': 0, 'had_pc1': 0, 'had_pc2': 0,
                'pc1_win': 0, 'pc1_loss': 0, 'pc1_pnl': 0.0,
                'no_pc_win': 0, 'no_pc_loss': 0, 'no_pc_pnl': 0.0,
                'pc2_pnl': 0.0
            })

            for trade in closed_trades:
                try:
                    sym = trade.get('symbol', '')
                    outcome = trade.get('outcome', {})
                    profit = outcome.get('profit', 0)
                    recovery = outcome.get('recovery', {})
                    pc_count = recovery.get('partial_close_count', 0)
                    had_pc = trade.get('had_partial_close', False) or pc_count >= 1

                    stats = pc_stats[sym]
                    stats['total'] += 1

                    if had_pc or pc_count >= 1:
                        stats['had_pc1'] += 1
                        if pc_count >= 2:
                            stats['had_pc2'] += 1
                            stats['pc2_pnl'] += profit
                        if profit > 0:
                            stats['pc1_win'] += 1
                        else:
                            stats['pc1_loss'] += 1
                        stats['pc1_pnl'] += profit
                    else:
                        if profit > 0:
                            stats['no_pc_win'] += 1
                        else:
                            stats['no_pc_loss'] += 1
                        stats['no_pc_pnl'] += profit

                except (KeyError, ValueError, TypeError):
                    continue

            # Generate recommendations
            pc_recs = {}
            for sym, stats in pc_stats.items():
                if stats['had_pc1'] < 5:
                    continue  # Not enough PC1 data

                conversion_rate = stats['had_pc2'] / stats['had_pc1'] * 100 if stats['had_pc1'] else 0
                pc1_wr = stats['pc1_win'] / stats['had_pc1'] * 100 if stats['had_pc1'] else 0
                no_pc_total = stats['no_pc_win'] + stats['no_pc_loss']
                no_pc_wr = stats['no_pc_win'] / no_pc_total * 100 if no_pc_total else 0

                rec = {
                    'total_trades': stats['total'],
                    'pc1_count': stats['had_pc1'],
                    'pc2_count': stats['had_pc2'],
                    'pc1_to_pc2_conversion': round(conversion_rate, 1),
                    'pc1_win_rate': round(pc1_wr, 1),
                    'pc1_net_pnl': round(stats['pc1_pnl'], 2),
                    'no_pc_win_rate': round(no_pc_wr, 1),
                    'no_pc_net_pnl': round(stats['no_pc_pnl'], 2),
                    'pc2_net_pnl': round(stats['pc2_pnl'], 2),
                }

                # Recommendation logic
                if conversion_rate < 30:
                    rec['suggestion'] = 'PC1 close % should be HIGH (50%+) — low PC2 conversion'
                elif conversion_rate > 60:
                    rec['suggestion'] = 'PC1 close % can be LOWER (25-35%) — good PC2 conversion'
                else:
                    rec['suggestion'] = 'PC1 close % at 50% is balanced'

                pc_recs[sym] = rec
                logger.info(f"PC Analysis {sym}: PC1={stats['had_pc1']}, "
                            f"PC2={stats['had_pc2']} ({conversion_rate:.0f}% conv), "
                            f"PC1 WR={pc1_wr:.0f}%, No-PC WR={no_pc_wr:.0f}%")

            if pc_recs:
                recommendations['pc_analysis'] = pc_recs

        except Exception as e:
            logger.warning(f"PC settings analysis failed: {e}")

    def _recommend_timing_based_adjustments(
        self,
        timing_data: Dict,
        current_config: Dict,
        recommendations: Dict
    ):
        """
        Analyze factor timing patterns (day-of-week, hour-of-day) and apply
        weight adjustments based on when factors perform best/worst.

        Only activates after 500+ trades for statistical significance.
        """
        MIN_TRADES_FOR_TIMING = 500

        trades_analyzed = timing_data.get('trades_analyzed', 0)
        day_patterns = timing_data.get('day_patterns', [])
        hour_patterns = timing_data.get('hour_patterns', [])

        if not day_patterns and not hour_patterns:
            return

        # Check if we have enough data to act
        if trades_analyzed < MIN_TRADES_FOR_TIMING:
            logger.info(f"Timing patterns: {trades_analyzed}/{MIN_TRADES_FOR_TIMING} trades - collecting data (no action yet)")
            return

        logger.info(f"Timing patterns: {trades_analyzed} trades - sufficient data, applying adjustments")

        # Get current weights
        current_htf = current_config.get('htf_weights', {})
        current_entry = current_config.get('entry_weights', {})

        # Map factor display names to config names
        factor_to_config = {
            'VWAP Band 1': 'vwap_band_1',
            'VWAP Band 2': 'vwap_band_2',
            'Below VAL': 'below_val',
            'Above VAH': 'above_vah',
            'Swing High': 'swing_high',
            'Swing Low': 'swing_low',
            'POC': 'poc',
            'Low Volume Node': 'lvn',
            'Prev Day VAH': 'prev_day_vah',
            'Prev Day VAL': 'prev_day_val',
            'Prev Day POC': 'prev_day_poc',
            'Prev Day High': 'prev_day_high',
            'Prev Day Low': 'prev_day_low',
            'Daily HVN': 'daily_hvn',
            'Weekly HVN': 'weekly_hvn',
            'Weekly POC': 'weekly_poc',
            'Prev Week High': 'prev_week_high',
            'Prev Week Low': 'prev_week_low',
        }

        # HTF factors (for reference)
        htf_factors = {'prev_day_vah', 'prev_day_val', 'prev_day_poc', 'prev_day_high',
                       'prev_day_low', 'daily_hvn', 'weekly_hvn', 'weekly_poc',
                       'prev_week_high', 'prev_week_low'}

        # Entry factors
        entry_factors = {'vwap_band_1', 'vwap_band_2', 'below_val', 'above_vah',
                         'swing_high', 'swing_low', 'poc', 'lvn'}

        # Process day patterns - adjust weights for factors with significant day-of-week edge
        for pattern in day_patterns:
            factor_display = pattern.get('factor', '')
            edge = pattern.get('edge', 0)
            worst_day_wr = pattern.get('worst_day_wr', 0)
            best_day = pattern.get('best_day', '')
            worst_day = pattern.get('worst_day', '')

            # Only act on significant edges (>25%)
            if edge < 0.25:
                continue

            config_name = factor_to_config.get(factor_display)
            if not config_name:
                continue

            # If worst day win rate is very poor (<40%), reduce weight
            # This penalizes factors that are inconsistent across days
            if worst_day_wr < 0.40:
                if config_name in htf_factors:
                    current_weight = current_htf.get(config_name, 2)
                    new_weight = max(1, current_weight - 1)
                    if new_weight != current_weight:
                        recommendations['htf_weights'][config_name] = {
                            'current': current_weight,
                            'recommended': new_weight,
                            'change': -1,
                            'reason': f"Timing: {worst_day_wr*100:.0f}% WR on {worst_day} (edge: {edge*100:.0f}%)"
                        }
                        logger.info(f"Timing adjustment: {config_name} HTF weight {current_weight} -> {new_weight} (poor {worst_day})")

                elif config_name in entry_factors:
                    current_weight = current_entry.get(config_name, 2)
                    new_weight = max(0, current_weight - 1)
                    if new_weight != current_weight:
                        recommendations['entry_weights'][config_name] = {
                            'current': current_weight,
                            'recommended': new_weight,
                            'change': -1,
                            'reason': f"Timing: {worst_day_wr*100:.0f}% WR on {worst_day} (edge: {edge*100:.0f}%)"
                        }
                        logger.info(f"Timing adjustment: {config_name} Entry weight {current_weight} -> {new_weight} (poor {worst_day})")

        # Process hour patterns similarly
        for pattern in hour_patterns:
            factor_display = pattern.get('factor', '')
            edge = pattern.get('edge', 0)
            worst_hour_wr = pattern.get('worst_hour_wr', 0)
            best_hour = pattern.get('best_hour', 0)
            worst_hour = pattern.get('worst_hour', 0)

            if edge < 0.30:
                continue

            config_name = factor_to_config.get(factor_display)
            if not config_name:
                continue

            # If worst hour win rate is very poor (<35%), reduce weight
            if worst_hour_wr < 0.35:
                if config_name in htf_factors and config_name not in recommendations.get('htf_weights', {}):
                    current_weight = current_htf.get(config_name, 2)
                    new_weight = max(1, current_weight - 1)
                    if new_weight != current_weight:
                        recommendations['htf_weights'][config_name] = {
                            'current': current_weight,
                            'recommended': new_weight,
                            'change': -1,
                            'reason': f"Timing: {worst_hour_wr*100:.0f}% WR at {worst_hour:02d}:00 (edge: {edge*100:.0f}%)"
                        }
                        logger.info(f"Timing adjustment: {config_name} HTF weight {current_weight} -> {new_weight} (poor hour {worst_hour})")

                elif config_name in entry_factors and config_name not in recommendations.get('entry_weights', {}):
                    current_weight = current_entry.get(config_name, 2)
                    new_weight = max(0, current_weight - 1)
                    if new_weight != current_weight:
                        recommendations['entry_weights'][config_name] = {
                            'current': current_weight,
                            'recommended': new_weight,
                            'change': -1,
                            'reason': f"Timing: {worst_hour_wr*100:.0f}% WR at {worst_hour:02d}:00 (edge: {edge*100:.0f}%)"
                        }
                        logger.info(f"Timing adjustment: {config_name} Entry weight {current_weight} -> {new_weight} (poor hour {worst_hour})")

    def _apply_changes(self, recommendations: Dict, current_config: Dict):
        """Apply recommended changes to strategy_config.py"""

        if not self.config_path.exists():
            logger.error("Config file not found, cannot apply changes")
            return

        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # NOTE: Global HTF/Entry weight changes are DISABLED.
        # All weight tuning is per-symbol only (via _PER_SYMBOL_* dicts below).
        # This prevents one symbol's data from affecting another symbol's settings.

        # SKIP global HTF weight changes — use per-symbol path instead
        if recommendations.get('htf_weights'):
            logger.info(f"Skipping {len(recommendations['htf_weights'])} global HTF weight changes (per-symbol only)")

        # SKIP global Entry weight changes — use per-symbol path instead
        if recommendations.get('entry_weights'):
            logger.info(f"Skipping {len(recommendations['entry_weights'])} global Entry weight changes (per-symbol only)")

        # Apply threshold changes
        for param, change_info in recommendations.get('thresholds', {}).items():
            new_value = change_info['recommended']
            param_upper = param.upper()
            pattern = rf"({param_upper}\s*=\s*)\d+"
            replacement = rf"\g<1>{new_value}"
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                self.changes.append({
                    'type': 'threshold',
                    'parameter': param,
                    'old_value': change_info['current'],
                    'new_value': new_value,
                    'reason': change_info['reason']
                })
                logger.info(f"Threshold change: {param} {change_info['current']} -> {new_value}")

        # SKIP hour changes — hours are LOCKED to data-driven optimal values in strategy_config.py
        if recommendations.get('hours'):
            logger.info(f"Skipping {len(recommendations['hours'])} hour changes (hours locked in config)")

        # SKIP global DI filter changes — use per-symbol path instead
        if recommendations.get('di_filter'):
            logger.info(f"Skipping {len(recommendations['di_filter'])} global DI filter changes (per-symbol only)")

        # SKIP global SMC filter changes — use per-symbol path instead
        if recommendations.get('smc_filter'):
            logger.info(f"Skipping {len(recommendations['smc_filter'])} global SMC filter changes (per-symbol only)")

        # Apply per-symbol override changes
        per_symbol = recommendations.get('per_symbol', {})
        for sym, sym_rec in per_symbol.items():
            # Update SYMBOL_HTF_OVERRIDES
            htf_ov = sym_rec.get('htf_overrides', {})
            if htf_ov:
                # Replace the symbol's entry in SYMBOL_HTF_OVERRIDES
                pattern = rf"(SYMBOL_HTF_OVERRIDES\s*=\s*\{{[^}}]*?'{sym}':\s*)\{{[^}}]*?\}}"
                replacement = rf"\g<1>{htf_ov}"
                new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                if new_content != content:
                    content = new_content
                    self.changes.append({
                        'type': 'per_symbol_htf',
                        'parameter': f'{sym} HTF overrides',
                        'old_value': '{}',
                        'new_value': htf_ov,
                        'reason': f'Per-symbol factor analysis'
                    })
                    logger.info(f"Per-symbol HTF {sym}: {htf_ov}")

            # Update SYMBOL_ENTRY_OVERRIDES
            entry_ov = sym_rec.get('entry_overrides', {})
            if entry_ov:
                pattern = rf"(SYMBOL_ENTRY_OVERRIDES\s*=\s*\{{[^}}]*?'{sym}':\s*)\{{[^}}]*?\}}"
                replacement = rf"\g<1>{entry_ov}"
                new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                if new_content != content:
                    content = new_content
                    self.changes.append({
                        'type': 'per_symbol_entry',
                        'parameter': f'{sym} Entry overrides',
                        'old_value': '{}',
                        'new_value': entry_ov,
                        'reason': f'Per-symbol factor analysis'
                    })
                    logger.info(f"Per-symbol Entry {sym}: {entry_ov}")

            # Update SYMBOL_DI_SETTINGS
            di_settings = sym_rec.get('di_settings')
            if di_settings:
                pattern = rf"(SYMBOL_DI_SETTINGS\s*=\s*\{{[^}}]*?'{sym}':\s*)\{{[^}}]*?\}}"
                replacement = rf"\g<1>{di_settings}"
                new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                if new_content != content:
                    content = new_content
                    self.changes.append({
                        'type': 'per_symbol_di',
                        'parameter': f'{sym} DI settings',
                        'old_value': current_config.get('symbol_di_settings', {}).get(sym, {}),
                        'new_value': di_settings,
                        'reason': f'Per-symbol DI analysis'
                    })
                    logger.info(f"Per-symbol DI {sym}: {di_settings}")

            # Update SYMBOL_SMC_SETTINGS
            smc_settings = sym_rec.get('smc_settings')
            if smc_settings:
                pattern = rf"(SYMBOL_SMC_SETTINGS\s*=\s*\{{[^}}]*?'{sym}':\s*)\{{[^}}]*?\}}"
                replacement = rf"\g<1>{smc_settings}"
                new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                if new_content != content:
                    content = new_content
                    self.changes.append({
                        'type': 'per_symbol_smc',
                        'parameter': f'{sym} SMC settings',
                        'old_value': current_config.get('symbol_smc_settings', {}).get(sym, {}),
                        'new_value': smc_settings,
                        'reason': f'Per-symbol SMC analysis'
                    })
                    logger.info(f"Per-symbol SMC {sym}: {smc_settings}")

            # Update SYMBOL_ADX_SETTINGS
            adx_settings = sym_rec.get('adx_settings')
            if adx_settings:
                pattern = rf"(SYMBOL_ADX_SETTINGS\s*=\s*\{{[^}}]*?'{sym}':\s*)\{{[^}}]*?\}}"
                replacement = rf"\g<1>{adx_settings}"
                new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                if new_content != content:
                    content = new_content
                    self.changes.append({
                        'type': 'per_symbol_adx',
                        'parameter': f'{sym} ADX settings',
                        'old_value': current_config.get('symbol_adx_settings', {}).get(sym, {}),
                        'new_value': adx_settings,
                        'reason': f'Per-symbol ADX analysis'
                    })
                    logger.info(f"Per-symbol ADX {sym}: {adx_settings}")

        # Write changes if not dry run
        if content != original_content:
            if self.dry_run:
                logger.info("DRY RUN - Changes NOT applied")
                logger.info(f"Would apply {len(self.changes)} changes")
            else:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Applied {len(self.changes)} changes to strategy_config.py")
        else:
            logger.info("No changes to apply")

    def _generate_report(self, recommendations: Dict) -> Dict:
        """Generate tuning report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'mode': 'dry_run' if self.dry_run else 'live',
            'changes_count': len(self.changes),
            'changes': self.changes,
            'recommendations': recommendations,
            'constraints': self.CONSTRAINTS
        }

    def _save_tuning_history(self):
        """Save tuning history to file"""
        history_path = self.ml_outputs_dir / 'auto_tuning_history.jsonl'

        record = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'dry_run' if self.dry_run else 'live',
            'changes_count': len(self.changes),
            'changes': self.changes
        }

        with open(history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

        logger.info(f"Saved tuning history to {history_path}")

    def print_report(self):
        """Print human-readable tuning report"""
        print()
        print("=" * 80)
        print("AUTO-TUNING REPORT")
        print("=" * 80)
        print(f"Timestamp: {self.tuning_report.get('timestamp', 'N/A')}")
        print(f"Mode: {self.tuning_report.get('mode', 'N/A').upper()}")
        print(f"Changes: {self.tuning_report.get('changes_count', 0)}")
        print()

        if self.changes:
            print("-" * 80)
            print("APPLIED CHANGES")
            print("-" * 80)

            for change in self.changes:
                change_type = change.get('type', '')
                if change_type in ['htf_weight', 'entry_weight']:
                    print(f"  [{change_type.upper()}] {change.get('factor')}: "
                          f"{change.get('old_value')} -> {change.get('new_value')}")
                    print(f"      Reason: {change.get('reason')}")
                elif change_type == 'threshold':
                    print(f"  [THRESHOLD] {change.get('parameter')}: "
                          f"{change.get('old_value')} -> {change.get('new_value')}")
                    print(f"      Reason: {change.get('reason')}")
                elif change_type == 'hours':
                    print(f"  [HOURS] {change.get('parameter')}")
                    print(f"      Old: {change.get('old_value')}")
                    print(f"      New: {change.get('new_value')}")
                print()
        else:
            print("No changes recommended - current configuration is optimal")

        print("=" * 80)


def run_scheduled():
    """Run auto-tuning on a schedule"""
    import schedule
    import time

    def scheduled_tune():
        logger.info(f"Scheduled auto-tuning triggered at {datetime.now()}")
        tuner = AutoTuner(dry_run=False)
        result = tuner.run()
        tuner.print_report()
        return result

    # Schedule tuning every 8 hours
    schedule.every(8).hours.do(scheduled_tune)

    # Run initial tuning
    logger.info("Running initial auto-tuning...")
    scheduled_tune()

    # Keep running
    logger.info("Scheduler active. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Auto-tune trading strategy configuration')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show changes without applying them')
    parser.add_argument('--schedule', action='store_true',
                        help='Run on schedule (every 8 hours)')

    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        tuner = AutoTuner(dry_run=args.dry_run)
        result = tuner.run()
        tuner.print_report()

        # Save report
        report_path = tuner.ml_outputs_dir / 'auto_tuning_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull report saved to: {report_path}")


class LiveAutoTuner:
    """
    Lightweight auto-tuner that runs after each closed trade.
    Designed to be called from the continuous logger when trades close.
    """

    def __init__(self):
        """Initialize live auto-tuner"""
        self.project_root = Path(__file__).parent.parent
        self.ml_outputs_dir = self.project_root / 'ml_system' / 'outputs'
        self.config_path = self.project_root / 'trading_bot' / 'config' / 'strategy_config.py'
        self.training_data_path = self.project_root / 'ml_system' / 'data' / 'training_data.csv'

        # Track tuning state
        self.trades_since_last_tune = 0
        self.min_trades_between_tunes = 5  # Tune every 5 closed trades
        self.last_tune_time = None

        # Load current config on init
        self._current_config = None

        logger.info("[AUTO-TUNER] LiveAutoTuner initialized")

    def on_trade_closed(self, trade_data: Dict = None) -> bool:
        """
        Called when a trade closes. Decides whether to trigger tuning.

        Args:
            trade_data: Optional trade data from the closed trade

        Returns:
            True if tuning was performed, False otherwise
        """
        self.trades_since_last_tune += 1

        # Check if we should tune
        if self.trades_since_last_tune < self.min_trades_between_tunes:
            logger.debug(f"[AUTO-TUNER] Trade closed ({self.trades_since_last_tune}/{self.min_trades_between_tunes})")
            return False

        # Time to tune
        logger.info(f"[AUTO-TUNER] Triggering auto-tune after {self.trades_since_last_tune} trades")
        self.trades_since_last_tune = 0

        return self.run_tuning()

    def run_tuning(self) -> bool:
        """
        Run the actual tuning logic.

        Returns:
            True if tuning was successful
        """
        try:
            tuner = AutoTuner(dry_run=False)
            result = tuner.run()

            if result.get('error'):
                logger.error(f"[AUTO-TUNER] Tuning failed: {result['error']}")
                return False

            changes_count = result.get('changes_count', 0)
            if changes_count > 0:
                logger.info(f"[AUTO-TUNER] Applied {changes_count} configuration changes")
                tuner.print_report()
            else:
                logger.info("[AUTO-TUNER] No changes needed - config is optimal")

            self.last_tune_time = datetime.now()
            return True

        except Exception as e:
            logger.error(f"[AUTO-TUNER] Error during tuning: {e}")
            import traceback
            traceback.print_exc()
            return False

    def force_tune(self) -> bool:
        """Force immediate tuning regardless of trade count"""
        logger.info("[AUTO-TUNER] Force tuning requested")
        self.trades_since_last_tune = 0
        return self.run_tuning()


# Global instance for use by continuous logger
_live_tuner_instance: Optional[LiveAutoTuner] = None


def get_live_tuner() -> LiveAutoTuner:
    """Get or create the global LiveAutoTuner instance"""
    global _live_tuner_instance
    if _live_tuner_instance is None:
        _live_tuner_instance = LiveAutoTuner()
    return _live_tuner_instance


def on_trade_closed(trade_data: Dict = None) -> bool:
    """
    Convenience function to call when a trade closes.
    Called by continuous_logger.update_closed_trades()
    """
    tuner = get_live_tuner()
    return tuner.on_trade_closed(trade_data)


if __name__ == '__main__':
    main()
