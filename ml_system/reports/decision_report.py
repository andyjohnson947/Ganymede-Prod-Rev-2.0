#!/usr/bin/env python3
"""
ML Decision Report Generator
Provides actionable recommendations based on ML analysis
Focus: What to change, not what happened
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import logging

# Add project root to path dynamically
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_system.features.extractor import FeatureExtractor
from ml_system.analysis.cascade_analyzer import CascadeAnalyzer
from ml_system.ml_optimizer import MLParameterOptimizer

# Don't use basicConfig - it adds global handlers causing duplicate logs
logger = logging.getLogger('DecisionReport')
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root / console


class DecisionReportGenerator:
    """Generate actionable decision reports"""

    def __init__(self):
        # Use absolute path based on project root to avoid duplication
        project_root = Path(__file__).parent.parent.parent
        self.report_dir = project_root / 'ml_system' / 'reports' / 'daily'
        os.makedirs(self.report_dir, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        self.cascade_analyzer = CascadeAnalyzer()
        self.ml_optimizer = MLParameterOptimizer()

        # Minimum samples to avoid overfitting
        self.MIN_DAILY = 5
        self.MIN_WEEKLY = 15
        self.MIN_MONTHLY = 30

        # Report width (for box drawing)
        self.WIDTH = 80

    # =========================================================================
    # BOX DRAWING HELPERS - Clean terminal-style formatting
    # =========================================================================

    def _box_top(self, title: str = "") -> str:
        """Draw box top: ┌─ TITLE ────────────┐"""
        if title:
            title_part = f"─ {title} "
            padding = "─" * (self.WIDTH - len(title_part) - 2)
            return f"┌{title_part}{padding}┐"
        return "┌" + "─" * (self.WIDTH - 2) + "┐"

    def _box_row(self, content: str) -> str:
        """Draw box row: │  content           │"""
        # Truncate if too long
        max_content = self.WIDTH - 4
        if len(content) > max_content:
            content = content[:max_content-2] + ".."
        padding = " " * (self.WIDTH - len(content) - 4)
        return f"│  {content}{padding}│"

    def _box_bottom(self) -> str:
        """Draw box bottom: └────────────────────┘"""
        return "└" + "─" * (self.WIDTH - 2) + "┘"

    def _box_divider(self) -> str:
        """Draw inner divider: │  ───────────────  │"""
        return f"│  {'─' * (self.WIDTH - 6)}  │"

    def _header_box(self, title: str, subtitle: str = "") -> list:
        """Draw double-line header box"""
        lines = []
        lines.append("╔" + "═" * (self.WIDTH - 2) + "╗")
        # Center title
        title_padded = title.center(self.WIDTH - 4)
        lines.append(f"║  {title_padded}║")
        if subtitle:
            sub_padded = subtitle.center(self.WIDTH - 4)
            lines.append(f"║  {sub_padded}║")
        lines.append("╚" + "═" * (self.WIDTH - 2) + "╝")
        return lines

    def _table_row(self, values: list, widths: list, align: list = None) -> str:
        """Format a table row with specified column widths

        Args:
            values: List of values
            widths: List of column widths
            align: List of alignments ('l', 'r', 'c') - default left
        """
        if align is None:
            align = ['l'] * len(values)

        parts = []
        for i, (val, width) in enumerate(zip(values, widths)):
            val_str = str(val)[:width]  # Truncate if needed
            a = align[i] if i < len(align) else 'l'
            if a == 'r':
                parts.append(val_str.rjust(width))
            elif a == 'c':
                parts.append(val_str.center(width))
            else:
                parts.append(val_str.ljust(width))

        content = "  ".join(parts)
        return self._box_row(content)

    def _format_pct(self, value: float, include_sign: bool = False) -> str:
        """Format percentage with optional sign"""
        if include_sign and value > 0:
            return f"+{value:.0f}%"
        return f"{value:.0f}%"

    def _format_money(self, value: float, include_sign: bool = True) -> str:
        """Format money with sign"""
        if include_sign:
            return f"${value:+.2f}" if value != 0 else "$0.00"
        return f"${value:.2f}"

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis if needed"""
        if len(text) <= max_len:
            return text
        return text[:max_len-2] + ".."

    def load_bot_config(self):
        """Load current bot configuration"""
        try:
            import sys
            sys.path.insert(0, 'trading_bot')
            from config import strategy_config

            config = {
                'confluence_threshold': getattr(strategy_config, 'MIN_CONFLUENCE_SCORE', 8),
                'swing_low_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('swing_low', 1),
                'swing_high_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('swing_high', 1),
                'vwap_band_1_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('vwap_band_1', 1),
                'poc_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('poc', 1),
                'prev_day_vah_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('prev_day_vah', 2),
                'weekly_hvn_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('weekly_hvn', 3),
                'hedge_enabled': getattr(strategy_config, 'HEDGE_ENABLED', False),
                'hedge_trigger_pips': getattr(strategy_config, 'HEDGE_TRIGGER_PIPS', 8),
                'dca_enabled': getattr(strategy_config, 'DCA_ENABLED', True),
                'dca_max_levels': getattr(strategy_config, 'DCA_MAX_LEVELS', 6),
                'dca_trigger_pips': getattr(strategy_config, 'DCA_TRIGGER_PIPS', 20),
                'grid_enabled': getattr(strategy_config, 'GRID_ENABLED', True),
                'disable_negative_grid': getattr(strategy_config, 'DISABLE_NEGATIVE_GRID', True),
                'adx_threshold': getattr(strategy_config, 'ADX_THRESHOLD', 25),
                'risk_percent': getattr(strategy_config, 'RISK_PERCENT', 1.0),
            }
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def get_trades(self, days=365):
        """Get all trades from SQLite (fallback: JSONL)"""
        # Try SQLite first
        try:
            from ml_system.trade_db import get_trade_db
            db = get_trade_db()
            if db:
                return db.get_trades_since(days)
        except Exception as e:
            logger.warning(f"SQLite read failed, falling back to JSONL: {e}")

        # Fallback: parse JSONL
        trades = []
        # Use absolute path based on project root
        project_root = Path(__file__).parent.parent.parent
        log_file = project_root / 'ml_system' / 'outputs' / 'continuous_trade_log.jsonl'

        if not log_file.exists():
            logger.warning(f"Trade log not found at: {log_file}")
            return trades

        # Only log on first call (debug level to reduce spam)
        logger.debug(f"Loading trades from: {log_file} (days={days})")

        cutoff = datetime.now() - timedelta(days=days)

        with open(str(log_file), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                    if entry_time >= cutoff:
                        trades.append(trade)
                except:
                    continue

        return trades

    def analyze_feature_importance(self, trades):
        """Compare ML feature importance vs current config weights"""
        recommendations = []

        try:
            # Load feature importance
            project_root = Path(__file__).parent.parent.parent
            importance_file = project_root / 'ml_system' / 'models' / 'feature_importance_baseline.csv'
            if not importance_file.exists():
                logger.warning(f"Feature importance file not found at: {importance_file}")
                return recommendations

            feature_imp = pd.read_csv(str(importance_file))
            config = self.load_bot_config()

            # Map features to config settings
            feature_mapping = {
                'at_swing_low': ('swing_low_weight', config.get('swing_low_weight', 1)),
                'at_swing_high': ('swing_high_weight', config.get('swing_high_weight', 1)),
                'at_poc': ('poc_weight', config.get('poc_weight', 1)),
                'at_prev_day_vah': ('prev_day_vah_weight', config.get('prev_day_vah_weight', 2)),
            }

            for _, row in feature_imp.head(20).iterrows():
                feature = row['feature']
                importance = row['importance']

                if feature in feature_mapping:
                    setting_name, current_weight = feature_mapping[feature]

                    # High importance but low weight = opportunity
                    if importance > 0.06 and current_weight < 3:
                        priority = 'HIGH' if importance > 0.07 else 'MEDIUM'
                        optimal_weight = 3 if importance > 0.07 else 2

                        recommendations.append({
                            'priority': priority,
                            'setting': setting_name,
                            'feature': feature,
                            'current': current_weight,
                            'optimal': optimal_weight,
                            'importance': importance,
                            'reason': f"ML importance {importance:.1%}, current weight {current_weight}",
                            'impact': '+5% win rate' if importance > 0.07 else '+3% win rate'
                        })

        except Exception as e:
            logger.error(f"Error analyzing features: {e}")

        return recommendations

    def analyze_time_patterns(self, trades):
        """Analyze daily/weekly patterns"""
        patterns = {
            'best_hours': [],
            'worst_hours': [],
            'best_days': [],
            'worst_days': [],
        }

        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_WEEKLY:
            return patterns

        df = pd.DataFrame([{
            'hour': datetime.fromisoformat(t['entry_time']).hour,
            'day': datetime.fromisoformat(t['entry_time']).strftime('%A'),
            'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
        } for t in closed])

        # Hour analysis
        hour_stats = df.groupby('hour')['win'].agg(['mean', 'count'])
        hour_stats = hour_stats[hour_stats['count'] >= 2]  # At least 2 trades

        if not hour_stats.empty:
            best_hours = hour_stats.nlargest(3, 'mean')
            worst_hours = hour_stats.nsmallest(3, 'mean')

            patterns['best_hours'] = [
                (int(h), float(row['mean']), int(row['count']))
                for h, row in best_hours.iterrows()
            ]
            patterns['worst_hours'] = [
                (int(h), float(row['mean']), int(row['count']))
                for h, row in worst_hours.iterrows()
            ]

        # Day analysis
        day_stats = df.groupby('day')['win'].agg(['mean', 'count'])
        day_stats = day_stats[day_stats['count'] >= 2]

        if not day_stats.empty:
            patterns['best_days'] = [
                (str(d), float(row['mean']), int(row['count']))
                for d, row in day_stats.nlargest(2, 'mean').iterrows()
            ]
            patterns['worst_days'] = [
                (str(d), float(row['mean']), int(row['count']))
                for d, row in day_stats.nsmallest(2, 'mean').iterrows()
            ]

        return patterns

    def analyze_market_regime(self, trades):
        """Analyze ADX/market regime patterns"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_WEEKLY:
            return None

        df = pd.DataFrame([{
            'adx': t.get('trend_filter', {}).get('adx', 0),
            'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
        } for t in closed])

        ranging = df[df['adx'] < 20]
        trending = df[(df['adx'] >= 20) & (df['adx'] < 40)]
        strong_trending = df[df['adx'] >= 40]

        regime = {
            'ranging': {'count': len(ranging), 'winrate': ranging['win'].mean() if len(ranging) > 0 else 0},
            'trending': {'count': len(trending), 'winrate': trending['win'].mean() if len(trending) > 0 else 0},
            'strong': {'count': len(strong_trending), 'winrate': strong_trending['win'].mean() if len(strong_trending) > 0 else 0},
            'current_dominant': 'ranging' if len(ranging) > len(df) * 0.6 else 'trending'
        }

        return regime

    def analyze_confluence_factors(self, trades):
        """Analyze individual confluence factor performance"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return {}

        # Build dataframe with all confluence factors
        factor_data = []
        for t in closed:
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0

            # Extract factors from correct locations in trade record
            vwap_data = t.get('vwap', {})
            vp_data = t.get('volume_profile', {})
            htf_data = t.get('htf_levels', {})
            fvg_data = t.get('fair_value_gaps', {})

            # Parse HTF factors from factors_matched list
            htf_factors_matched = htf_data.get('factors_matched', [])

            factor_data.append({
                # Volume Profile factors
                'swing_low': vp_data.get('at_swing_low', False),
                'swing_high': vp_data.get('at_swing_high', False),
                'poc': vp_data.get('at_poc', False),
                'above_vah': vp_data.get('above_vah', False),
                'below_val': vp_data.get('below_val', False),
                'lvn': vp_data.get('at_lvn', False),
                'hvn': vp_data.get('at_hvn', False),

                # VWAP factors
                'vwap_band_1': vwap_data.get('in_band_1', False),
                'vwap_band_2': vwap_data.get('in_band_2', False),

                # HTF factors (from factors_matched list)
                'prev_day_vah': 'Prev Day VAH' in htf_factors_matched,
                'prev_day_val': 'Prev Day VAL' in htf_factors_matched,
                'prev_day_poc': 'Prev Day POC' in htf_factors_matched,
                'prev_day_high': 'Prev Day High' in htf_factors_matched,
                'prev_day_low': 'Prev Day Low' in htf_factors_matched,
                'daily_hvn': 'Daily HVN' in htf_factors_matched,
                'daily_poc': 'Daily POC' in htf_factors_matched,
                'weekly_hvn': 'Weekly HVN' in htf_factors_matched,
                'weekly_poc': 'Weekly POC' in htf_factors_matched,
                'prev_week_high': 'Prev Week High' in htf_factors_matched,
                'prev_week_low': 'Prev Week Low' in htf_factors_matched,
                'prev_week_swing_high': 'Prev Week Swing High' in htf_factors_matched,
                'prev_week_swing_low': 'Prev Week Swing Low' in htf_factors_matched,
                'prev_week_vwap': 'Prev Week VWAP' in htf_factors_matched,

                # FVG factors
                'daily_bullish_fvg': fvg_data.get('daily_bullish_fvg', False),
                'daily_bearish_fvg': fvg_data.get('daily_bearish_fvg', False),
                'weekly_bullish_fvg': fvg_data.get('weekly_bullish_fvg', False),
                'weekly_bearish_fvg': fvg_data.get('weekly_bearish_fvg', False),

                'win': win
            })

        df = pd.DataFrame(factor_data)

        # Analyze each factor
        factor_analysis = {}
        for factor in df.columns:
            if factor == 'win':
                continue

            present = df[df[factor] == True]
            absent = df[df[factor] == False]

            if len(present) > 0:
                factor_analysis[factor] = {
                    'count': len(present),
                    'winrate': present['win'].mean(),
                    'winrate_without': absent['win'].mean() if len(absent) > 0 else 0,
                    'edge': present['win'].mean() - (absent['win'].mean() if len(absent) > 0 else 0)
                }

        # Sort by edge (positive contribution)
        sorted_factors = sorted(
            factor_analysis.items(),
            key=lambda x: x[1]['edge'],
            reverse=True
        )

        return dict(sorted_factors)

    def analyze_confluence_combinations(self, trades):
        """Find best confluence combinations (2-3 factors)"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_WEEKLY:
            return []

        # Build combinations
        combinations = defaultdict(lambda: {'wins': 0, 'total': 0})

        for t in closed:
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0

            # Extract factors from correct locations
            vwap_data = t.get('vwap', {})
            vp_data = t.get('volume_profile', {})
            htf_data = t.get('htf_levels', {})
            htf_factors_matched = htf_data.get('factors_matched', [])

            # Get active factors
            active = []

            # Volume Profile
            if vp_data.get('at_swing_low'): active.append('swing_low')
            if vp_data.get('at_swing_high'): active.append('swing_high')
            if vp_data.get('at_poc'): active.append('poc')
            if vp_data.get('above_vah'): active.append('above_vah')
            if vp_data.get('below_val'): active.append('below_val')
            if vp_data.get('at_lvn'): active.append('lvn')
            if vp_data.get('at_hvn'): active.append('hvn')

            # VWAP
            if vwap_data.get('in_band_1'): active.append('vwap_band_1')
            if vwap_data.get('in_band_2'): active.append('vwap_band_2')

            # HTF (from factors_matched list)
            if 'Prev Day VAH' in htf_factors_matched: active.append('prev_day_vah')
            if 'Prev Day VAL' in htf_factors_matched: active.append('prev_day_val')
            if 'Prev Day POC' in htf_factors_matched: active.append('prev_day_poc')
            if 'Daily HVN' in htf_factors_matched: active.append('daily_hvn')
            if 'Weekly HVN' in htf_factors_matched: active.append('weekly_hvn')
            if 'Weekly POC' in htf_factors_matched: active.append('weekly_poc')

            # Analyze 2-factor combinations
            for i in range(len(active)):
                for j in range(i+1, len(active)):
                    combo = tuple(sorted([active[i], active[j]]))
                    combinations[combo]['total'] += 1
                    combinations[combo]['wins'] += win

        # Filter combinations with at least 3 occurrences
        valid_combos = []
        for combo, stats in combinations.items():
            if stats['total'] >= 3:
                winrate = stats['wins'] / stats['total']
                valid_combos.append({
                    'factors': ' + '.join(combo),
                    'count': stats['total'],
                    'winrate': winrate
                })

        # Sort by winrate
        valid_combos.sort(key=lambda x: x['winrate'], reverse=True)

        return valid_combos[:5]  # Top 5

    def analyze_initial_vs_recovery(self, trades):
        """Separate analysis for initial trades vs recovery trades"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return None

        # Separate trades
        initial_only = []
        with_recovery = []

        for t in closed:
            recovery = t.get('outcome', {}).get('recovery', {})
            had_dca = recovery.get('dca_count', 0) > 0
            had_hedge = recovery.get('hedge_count', 0) > 0
            had_grid = recovery.get('grid_count', 0) > 0

            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            profit = t.get('outcome', {}).get('profit', 0)

            if had_dca or had_hedge or had_grid:
                with_recovery.append({
                    'win': win,
                    'profit': profit,
                    'had_dca': had_dca,
                    'had_hedge': had_hedge,
                    'had_grid': had_grid
                })
            else:
                initial_only.append({
                    'win': win,
                    'profit': profit
                })

        analysis = {
            'initial': {
                'count': len(initial_only),
                'winrate': sum(t['win'] for t in initial_only) / len(initial_only) if initial_only else 0,
                'avg_profit': sum(t['profit'] for t in initial_only) / len(initial_only) if initial_only else 0
            },
            'recovery': {
                'count': len(with_recovery),
                'winrate': sum(t['win'] for t in with_recovery) / len(with_recovery) if with_recovery else 0,
                'avg_profit': sum(t['profit'] for t in with_recovery) / len(with_recovery) if with_recovery else 0
            }
        }

        return analysis

    def analyze_dca_success_factors(self, trades):
        """What market conditions make DCA successful"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        dca_trades = []
        for t in closed:
            dca_count = t.get('outcome', {}).get('recovery', {}).get('dca_count', 0)
            if dca_count > 0:
                dca_trades.append({
                    'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
                    'profit': t.get('outcome', {}).get('profit', 0),
                    'dca_count': dca_count,
                    'adx': t.get('trend_filter', {}).get('adx', 0),
                    'di_spread': t.get('trend_filter', {}).get('di_spread', 0),
                    'confluence': t.get('confluence_score') or 0
                })

        if len(dca_trades) < 3:
            return None

        df = pd.DataFrame(dca_trades)

        # Find patterns
        winners = df[df['win'] == 1]
        losers = df[df['win'] == 0]

        analysis = {
            'total': len(dca_trades),
            'winrate': df['win'].mean(),
            'avg_profit': df['profit'].mean(),
            'avg_dca_count': df['dca_count'].mean(),
            'winner_patterns': {
                'avg_adx': winners['adx'].mean() if len(winners) > 0 else 0,
                'avg_confluence': winners['confluence'].mean() if len(winners) > 0 else 0
            },
            'loser_patterns': {
                'avg_adx': losers['adx'].mean() if len(losers) > 0 else 0,
                'avg_confluence': losers['confluence'].mean() if len(losers) > 0 else 0
            }
        }

        return analysis

    def analyze_hedge_success_factors(self, trades):
        """What market conditions make hedges successful"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        hedge_trades = []
        for t in closed:
            hedge_count = t.get('outcome', {}).get('recovery', {}).get('hedge_count', 0)
            if hedge_count > 0:
                hedge_trades.append({
                    'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
                    'profit': t.get('outcome', {}).get('profit', 0),
                    'hedge_count': hedge_count,
                    'adx': t.get('trend_filter', {}).get('adx', 0),
                    'trend_strength': t.get('trend_filter', {}).get('trend_strength', 0)
                })

        if len(hedge_trades) < 3:
            return None

        df = pd.DataFrame(hedge_trades)

        analysis = {
            'total': len(hedge_trades),
            'winrate': df['win'].mean(),
            'avg_profit': df['profit'].mean()
        }

        return analysis

    def analyze_grid_performance(self, trades):
        """Analyze positive grid performance"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        grid_trades = []
        for t in closed:
            grid_count = t.get('outcome', {}).get('recovery', {}).get('grid_count', 0)
            if grid_count > 0:
                grid_trades.append({
                    'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
                    'profit': t.get('outcome', {}).get('profit', 0),
                    'grid_count': grid_count
                })

        if len(grid_trades) < 3:
            return None

        df = pd.DataFrame(grid_trades)

        analysis = {
            'total': len(grid_trades),
            'winrate': df['win'].mean(),
            'avg_profit': df['profit'].mean(),
            'avg_grid_count': df['grid_count'].mean()
        }

        return analysis

    def analyze_vwap_vs_breakout(self, trades):
        """Compare Mean Reversion vs Breakout (momentum) performance"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return None

        # Separate by strategy type
        mr_trades = []
        breakout_trades = []
        legacy_trades = []

        for t in closed:
            strategy_type = t.get('strategy_type', 'confluence')
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            profit = t.get('outcome', {}).get('profit', 0) or 0
            confluence = t.get('confluence_score') or 0

            if strategy_type in ('vwap', 'mean_reversion'):
                mr_trades.append({'win': win, 'profit': profit, 'confluence': confluence})
            elif strategy_type == 'breakout':
                breakout_trades.append({'win': win, 'profit': profit, 'confluence': confluence})
            else:
                legacy_trades.append({'win': win, 'profit': profit, 'confluence': confluence})

        analysis = {
            'mean_reversion': {
                'count': len(mr_trades),
                'winrate': sum(t['win'] for t in mr_trades) / len(mr_trades) if mr_trades else 0,
                'avg_profit': sum(t['profit'] for t in mr_trades) / len(mr_trades) if mr_trades else 0,
                'avg_confluence': sum(t['confluence'] for t in mr_trades) / len(mr_trades) if mr_trades else 0
            },
            'breakout': {
                'count': len(breakout_trades),
                'winrate': sum(t['win'] for t in breakout_trades) / len(breakout_trades) if breakout_trades else 0,
                'avg_profit': sum(t['profit'] for t in breakout_trades) / len(breakout_trades) if breakout_trades else 0,
                'avg_confluence': sum(t['confluence'] for t in breakout_trades) / len(breakout_trades) if breakout_trades else 0
            },
            'legacy': {
                'count': len(legacy_trades),
                'winrate': sum(t['win'] for t in legacy_trades) / len(legacy_trades) if legacy_trades else 0,
                'avg_profit': sum(t['profit'] for t in legacy_trades) / len(legacy_trades) if legacy_trades else 0
            }
        }

        return analysis

    def analyze_per_symbol_performance(self, trades):
        """Analyze performance broken down by trading symbol"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return {}

        # Group by symbol
        symbol_data = defaultdict(list)
        for t in closed:
            symbol = t.get('symbol', 'UNKNOWN')
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            profit = t.get('outcome', {}).get('profit', 0) or 0
            factors = t.get('confluence_factors', [])

            symbol_data[symbol].append({
                'win': win,
                'profit': profit,
                'factors': factors
            })

        # Calculate per-symbol stats
        results = {}
        for symbol, trades_list in symbol_data.items():
            wins = sum(t['win'] for t in trades_list)
            total = len(trades_list)
            total_profit = sum(t['profit'] for t in trades_list)

            # Find best and worst factors for this symbol
            factor_wins = defaultdict(lambda: {'wins': 0, 'total': 0})
            for t in trades_list:
                for factor in t['factors']:
                    factor_wins[factor]['total'] += 1
                    factor_wins[factor]['wins'] += t['win']

            # Get factor win rates (minimum 2 occurrences)
            factor_rates = {
                f: stats['wins'] / stats['total']
                for f, stats in factor_wins.items()
                if stats['total'] >= 2
            }

            best_factor = max(factor_rates.items(), key=lambda x: x[1]) if factor_rates else (None, 0)
            worst_factor = min(factor_rates.items(), key=lambda x: x[1]) if factor_rates else (None, 0)

            results[symbol] = {
                'count': total,
                'wins': wins,
                'winrate': wins / total if total > 0 else 0,
                'total_profit': total_profit,
                'avg_profit': total_profit / total if total > 0 else 0,
                'best_factor': best_factor[0],
                'best_factor_rate': best_factor[1],
                'worst_factor': worst_factor[0],
                'worst_factor_rate': worst_factor[1]
            }

        return results

    def analyze_di_alignment(self, trades):
        """Analyze DI filter alignment with trade direction and outcomes"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return None

        # Categorize trades by DI alignment
        di_aligned_buy = []  # +DI > -DI and BUY
        di_aligned_sell = []  # -DI > +DI and SELL
        di_neutral = []  # Small DI gap

        for t in closed:
            trend_filter = t.get('trend_filter', {})
            plus_di = trend_filter.get('plus_di', 0)
            minus_di = trend_filter.get('minus_di', 0)
            direction = t.get('direction', '').upper()
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            profit = t.get('outcome', {}).get('profit', 0) or 0

            di_gap = abs(plus_di - minus_di)

            if di_gap < 3:  # Neutral zone
                di_neutral.append({'win': win, 'profit': profit})
            elif plus_di > minus_di and direction == 'BUY':
                di_aligned_buy.append({'win': win, 'profit': profit})
            elif minus_di > plus_di and direction == 'SELL':
                di_aligned_sell.append({'win': win, 'profit': profit})
            elif plus_di > minus_di and direction == 'SELL':
                # Counter-DI sell
                di_neutral.append({'win': win, 'profit': profit, 'counter': True})
            elif minus_di > plus_di and direction == 'BUY':
                # Counter-DI buy
                di_neutral.append({'win': win, 'profit': profit, 'counter': True})

        def calc_stats(trade_list):
            if not trade_list:
                return {'count': 0, 'winrate': 0, 'avg_profit': 0}
            return {
                'count': len(trade_list),
                'winrate': sum(t['win'] for t in trade_list) / len(trade_list),
                'avg_profit': sum(t['profit'] for t in trade_list) / len(trade_list)
            }

        return {
            'bullish_buy': calc_stats(di_aligned_buy),
            'bearish_sell': calc_stats(di_aligned_sell),
            'neutral': calc_stats(di_neutral),
            'total_aligned': len(di_aligned_buy) + len(di_aligned_sell),
            'total_neutral': len(di_neutral)
        }

    def analyze_smc_correlation(self, trades):
        """
        Analyze SMC (BOS/CHOCH) correlation with trade outcomes.
        Reads from smc_trade_correlation.jsonl which contains SMC state at trade entry.
        """
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return None

        # Load SMC correlation data from dedicated log
        project_root = Path(__file__).parent.parent.parent
        smc_log = project_root / 'ml_system' / 'outputs' / 'smc_trade_correlation.jsonl'

        if not smc_log.exists():
            return None

        # Build lookup of SMC state by timestamp (approximate match within 5 minutes)
        smc_data_by_time = {}
        try:
            with open(str(smc_log), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        ts = entry.get('timestamp', '')
                        symbol = entry.get('symbol', '')
                        if ts and symbol:
                            key = f"{symbol}_{ts[:16]}"  # Match by symbol + minute
                            smc_data_by_time[key] = entry.get('smc_state', {})
                    except:
                        continue
        except Exception as e:
            logger.warning(f"Failed to load SMC correlation log: {e}")
            return None

        # Match trades to SMC data
        with_bos = []
        with_choch = []
        no_smc = []

        for t in closed:
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            profit = t.get('outcome', {}).get('profit', 0) or 0
            direction = t.get('direction', '').upper()

            # Try to find matching SMC data
            entry_time = t.get('entry_time', '')
            symbol = t.get('symbol', '')

            smc_state = None
            if entry_time and symbol:
                # Try exact minute match
                key = f"{symbol}_{entry_time[:16]}"
                smc_state = smc_data_by_time.get(key)

            trade_data = {'win': win, 'profit': profit, 'direction': direction}

            if smc_state and smc_state.get('available'):
                # Check entry timeframe for BOS/CHOCH
                entry_tf = smc_state.get('entry_tf', {})
                latest_bos = entry_tf.get('latest_bos', {})
                latest_choch = entry_tf.get('latest_choch', {})

                # BOS within recent bars (< 50 bars ago) is considered active
                has_recent_bos = latest_bos and latest_bos.get('bars_ago', 999) < 50
                has_recent_choch = latest_choch and latest_choch.get('bars_ago', 999) < 50

                # Check alignment with trade direction
                bos_dir = latest_bos.get('direction', '') if latest_bos else ''
                choch_dir = latest_choch.get('direction', '') if latest_choch else ''

                # Trade aligns with BOS direction?
                bos_aligned = (direction == 'BUY' and bos_dir == 'bullish') or \
                              (direction == 'SELL' and bos_dir == 'bearish')

                # Trade aligns with CHOCH direction?
                choch_aligned = (direction == 'BUY' and choch_dir == 'bullish') or \
                                (direction == 'SELL' and choch_dir == 'bearish')

                if has_recent_bos and bos_aligned:
                    with_bos.append(trade_data)
                elif has_recent_choch and choch_aligned:
                    with_choch.append(trade_data)
                else:
                    no_smc.append(trade_data)
            else:
                no_smc.append(trade_data)

        def calc_stats(trade_list):
            if not trade_list:
                return {'count': 0, 'winrate': 0, 'avg_profit': 0}
            return {
                'count': len(trade_list),
                'winrate': sum(t['win'] for t in trade_list) / len(trade_list),
                'avg_profit': sum(t['profit'] for t in trade_list) / len(trade_list)
            }

        return {
            'with_bos': calc_stats(with_bos),
            'with_choch': calc_stats(with_choch),
            'no_smc': calc_stats(no_smc)
        }

    def analyze_execution_quality(self, trades):
        """Analyze trade execution quality metrics"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return None

        slippages = []
        spreads = []
        exact_fills = 0

        for t in closed:
            exec_data = t.get('execution_quality', {})
            if exec_data:
                slip = exec_data.get('slippage_pips', 0)
                spread = exec_data.get('spread_at_entry_pips', 0)

                if slip is not None:
                    slippages.append(slip)
                    if slip == 0:
                        exact_fills += 1
                if spread is not None:
                    spreads.append(spread)

        if not slippages:
            return None

        return {
            'avg_slippage': sum(slippages) / len(slippages),
            'max_slippage': max(slippages),
            'exact_fill_pct': exact_fills / len(slippages) * 100 if slippages else 0,
            'within_1_pip_pct': sum(1 for s in slippages if s <= 1) / len(slippages) * 100,
            'avg_spread': sum(spreads) / len(spreads) if spreads else 0,
            'sample_size': len(slippages)
        }

    def analyze_factors_by_day(self, trades):
        """Analyze which confluence factors work best on which days"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_WEEKLY:
            return None

        # Build factor-day matrix
        factor_day_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0}))

        for t in closed:
            entry_time = t.get('entry_time')
            if not entry_time:
                continue

            try:
                if isinstance(entry_time, str):
                    dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                else:
                    dt = entry_time
                day_name = dt.strftime('%A')
            except:
                continue

            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            factors = t.get('confluence_factors', [])

            for factor in factors:
                factor_day_stats[factor][day_name]['total'] += 1
                factor_day_stats[factor][day_name]['wins'] += win

        # Find significant patterns (factors with >= 3 trades on a day)
        patterns = []
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        for factor, day_stats in factor_day_stats.items():
            # Calculate overall factor win rate
            total_wins = sum(d['wins'] for d in day_stats.values())
            total_trades = sum(d['total'] for d in day_stats.values())
            if total_trades < 5:
                continue
            overall_wr = total_wins / total_trades

            # Find best and worst days for this factor
            day_rates = []
            for day in day_order:
                if day in day_stats and day_stats[day]['total'] >= 2:
                    day_wr = day_stats[day]['wins'] / day_stats[day]['total']
                    day_rates.append((day, day_wr, day_stats[day]['total']))

            if len(day_rates) >= 2:
                best_day = max(day_rates, key=lambda x: x[1])
                worst_day = min(day_rates, key=lambda x: x[1])

                # Only report if there's a significant difference (>20%)
                if best_day[1] - worst_day[1] > 0.2:
                    patterns.append({
                        'factor': factor,
                        'overall_wr': overall_wr,
                        'best_day': best_day[0],
                        'best_day_wr': best_day[1],
                        'best_day_count': best_day[2],
                        'worst_day': worst_day[0],
                        'worst_day_wr': worst_day[1],
                        'worst_day_count': worst_day[2],
                        'edge': best_day[1] - worst_day[1]
                    })

        # Sort by edge (biggest day-to-day difference)
        patterns.sort(key=lambda x: x['edge'], reverse=True)

        return patterns[:10] if patterns else None

    def analyze_factors_by_hour(self, trades):
        """Analyze which confluence factors work best at which hours"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_WEEKLY:
            return None

        # Build factor-hour matrix
        factor_hour_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0}))

        for t in closed:
            entry_time = t.get('entry_time')
            if not entry_time:
                continue

            try:
                if isinstance(entry_time, str):
                    dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                else:
                    dt = entry_time
                hour = dt.hour
            except:
                continue

            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            factors = t.get('confluence_factors', [])

            for factor in factors:
                factor_hour_stats[factor][hour]['total'] += 1
                factor_hour_stats[factor][hour]['wins'] += win

        # Find significant patterns
        patterns = []

        for factor, hour_stats in factor_hour_stats.items():
            total_trades = sum(d['total'] for d in hour_stats.values())
            if total_trades < 5:
                continue

            # Find best and worst hours for this factor
            hour_rates = []
            for hour, stats in hour_stats.items():
                if stats['total'] >= 2:
                    hour_wr = stats['wins'] / stats['total']
                    hour_rates.append((hour, hour_wr, stats['total']))

            if len(hour_rates) >= 2:
                best_hour = max(hour_rates, key=lambda x: x[1])
                worst_hour = min(hour_rates, key=lambda x: x[1])

                # Only report if there's a significant difference (>25%)
                if best_hour[1] - worst_hour[1] > 0.25:
                    patterns.append({
                        'factor': factor,
                        'best_hour': best_hour[0],
                        'best_hour_wr': best_hour[1],
                        'best_hour_count': best_hour[2],
                        'worst_hour': worst_hour[0],
                        'worst_hour_wr': worst_hour[1],
                        'worst_hour_count': worst_hour[2],
                        'edge': best_hour[1] - worst_hour[1]
                    })

        patterns.sort(key=lambda x: x['edge'], reverse=True)

        return patterns[:5] if patterns else None

    def generate_recommendations(self, trades):
        """Generate prioritized recommendations"""
        recommendations = []
        config = self.load_bot_config()
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return [{
                'priority': 'INFO',
                'message': f'Need {self.MIN_DAILY - len(closed)} more closed trades for recommendations',
                'action': 'Continue collecting data'
            }]

        # Feature importance analysis
        feature_recs = self.analyze_feature_importance(trades)
        recommendations.extend(feature_recs)

        # Market regime analysis
        regime = self.analyze_market_regime(trades)
        if regime and regime['ranging']['count'] > len(closed) * 0.7:
            # Strongly ranging market
            if config.get('adx_threshold', 25) > 20:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'setting': 'ADX filter',
                    'current': f"threshold {config.get('adx_threshold', 25)}",
                    'optimal': 'Add ranging filter (ADX < 20)',
                    'reason': f"{regime['ranging']['count']/len(closed)*100:.0f}% trades in ranging markets",
                    'impact': '+8% win rate, -30% trade frequency'
                })

        # Confluence threshold analysis
        df = pd.DataFrame([{
            'confluence': t.get('confluence_score') or 0,
            'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
        } for t in closed])

        avg_conf_winners = df[df['win'] == 1]['confluence'].mean()
        current_threshold = config.get('confluence_threshold', 8)

        if avg_conf_winners < current_threshold - 1:
            recommendations.append({
                'priority': 'LOW',
                'setting': 'confluence_threshold',
                'current': current_threshold,
                'optimal': int(avg_conf_winners),
                'reason': f"Winners average {avg_conf_winners:.1f}, threshold is {current_threshold}",
                'impact': 'Same win rate, +25% trade frequency'
            })

        return recommendations

    def generate_report(self):
        """Generate decision-focused report with clean formatting"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        date_str = datetime.now().strftime('%Y-%m-%d')

        # Load data
        trades_7d = self.get_trades(days=7)
        trades_30d = self.get_trades(days=30)
        trades_all = self.get_trades(days=365)
        config = self.load_bot_config()

        closed_7d = [t for t in trades_7d if t.get('outcome', {}).get('status') == 'closed']
        closed_30d = [t for t in trades_30d if t.get('outcome', {}).get('status') == 'closed']
        closed_all = [t for t in trades_all if t.get('outcome', {}).get('status') == 'closed']

        # Calculate summary stats
        winrate_7d = sum(1 for t in closed_7d if t.get('outcome', {}).get('profit', 0) > 0) / len(closed_7d) if closed_7d else 0
        avg_profit_7d = sum(t.get('outcome', {}).get('profit', 0) for t in closed_7d) / len(closed_7d) if closed_7d else 0

        # Build report
        report = []

        # ═══════════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════════
        report.extend(self._header_box(
            "ML DECISION REPORT",
            f"{len(closed_7d)} trades | {winrate_7d*100:.0f}% win | {self._format_money(avg_profit_7d)} avg | {timestamp}"
        ))
        report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # ACTION ITEMS
        # ───────────────────────────────────────────────────────────────────────
        report.append(self._box_top("ACTION ITEMS"))

        recommendations = self.generate_recommendations(trades_7d)

        if len(closed_7d) < self.MIN_DAILY:
            report.append(self._box_row(f"Collecting data... {len(closed_7d)}/{self.MIN_DAILY} trades needed"))
        else:
            high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
            medium_priority = [r for r in recommendations if r.get('priority') == 'MEDIUM']
            low_priority = [r for r in recommendations if r.get('priority') == 'LOW']

            if high_priority:
                for rec in high_priority:
                    report.append(self._box_row(f"! {rec.get('setting')}: {rec.get('current')} -> {rec.get('optimal')}"))
            if medium_priority:
                for rec in medium_priority:
                    report.append(self._box_row(f"* {rec.get('setting')}: {rec.get('current')} -> {rec.get('optimal')}"))
            if low_priority:
                for rec in low_priority:
                    report.append(self._box_row(f"~ {rec.get('setting')}: {rec.get('current')} -> {rec.get('optimal')}"))

            if not (high_priority or medium_priority or low_priority):
                report.append(self._box_row("All settings optimal - no changes needed"))

        report.append(self._box_bottom())
        report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # CONFIG VS OPTIMAL
        # ───────────────────────────────────────────────────────────────────────
        report.append(self._box_top("CONFIG vs OPTIMAL"))

        # Get ML recommendations
        ml_rec = self.ml_optimizer.get_all_recommendations(trades_30d) if len(trades_30d) >= 10 else {}
        conf_rec = ml_rec.get('confluence_threshold', {})
        conf_optimal = conf_rec.get('optimal', '-')
        weights = ml_rec.get('weights', {})

        # Header row
        report.append(self._table_row(
            ['Setting', 'Current', 'Optimal', 'Status'],
            [24, 12, 12, 10],
            ['l', 'c', 'c', 'c']
        ))
        report.append(self._box_divider())

        # Config rows
        config_data = [
            ('Confluence Threshold', str(config.get('confluence_threshold', 8)), str(conf_optimal), 'REVIEW' if conf_optimal != config.get('confluence_threshold', 8) else 'OK'),
            ('DCA Enabled', 'Yes' if config.get('dca_enabled') else 'No', 'Yes', 'OK'),
            ('Grid Enabled', 'Yes' if config.get('grid_enabled') else 'No', 'Yes', 'OK'),
            ('Hedge Enabled', 'Yes' if config.get('hedge_enabled') else 'No', 'Sparingly', 'REVIEW' if config.get('hedge_enabled') else 'OK'),
        ]

        for row in config_data:
            status_icon = '.' if row[3] == 'OK' else '!'
            report.append(self._table_row(
                [row[0], row[1], row[2], f"{status_icon} {row[3]}"],
                [24, 12, 12, 10],
                ['l', 'c', 'c', 'c']
            ))

        report.append(self._box_bottom())
        report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # STRATEGY PERFORMANCE
        # ───────────────────────────────────────────────────────────────────────
        vb_analysis = self.analyze_vwap_vs_breakout(trades_7d)
        if vb_analysis and (vb_analysis['mean_reversion']['count'] > 0 or vb_analysis['breakout']['count'] > 0):
            report.append(self._box_top("STRATEGY PERFORMANCE"))

            report.append(self._table_row(
                ['Strategy', 'Trades', 'Win%', 'Avg P/L', 'Note'],
                [18, 8, 8, 10, 22],
                ['l', 'r', 'r', 'r', 'l']
            ))
            report.append(self._box_divider())

            mr = vb_analysis['mean_reversion']
            bo = vb_analysis['breakout']

            if mr['count'] > 0:
                mr_note = ""
                if bo['count'] > 0 and mr['winrate'] > bo['winrate'] + 0.1:
                    mr_note = f"+{(mr['winrate']-bo['winrate'])*100:.0f}% vs BO"
                report.append(self._table_row(
                    ['MEAN REVERSION', str(mr['count']), f"{mr['winrate']*100:.0f}%", self._format_money(mr['avg_profit']), mr_note],
                    [18, 8, 8, 10, 22],
                    ['l', 'r', 'r', 'r', 'l']
                ))

            if bo['count'] > 0:
                bo_note = "All losses" if bo['winrate'] == 0 else ""
                report.append(self._table_row(
                    ['BREAKOUT', str(bo['count']), f"{bo['winrate']*100:.0f}%", self._format_money(bo['avg_profit']), bo_note],
                    [18, 8, 8, 10, 22],
                    ['l', 'r', 'r', 'r', 'l']
                ))

            report.append(self._box_bottom())
            report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # SYMBOL BREAKDOWN
        # ───────────────────────────────────────────────────────────────────────
        symbol_perf = self.analyze_per_symbol_performance(trades_7d)
        if symbol_perf:
            report.append(self._box_top("SYMBOL BREAKDOWN"))

            report.append(self._table_row(
                ['Symbol', 'Trades', 'Win%', 'P/L', 'Best Factor', 'Worst Factor'],
                [8, 7, 7, 9, 18, 18],
                ['l', 'r', 'r', 'r', 'l', 'l']
            ))
            report.append(self._box_divider())

            for symbol, stats in sorted(symbol_perf.items()):
                best = f"{self._truncate(stats['best_factor'], 12)} ({stats['best_factor_rate']*100:.0f}%)" if stats['best_factor'] else "-"
                worst = f"{self._truncate(stats['worst_factor'], 12)} ({stats['worst_factor_rate']*100:.0f}%)" if stats['worst_factor'] else "-"
                report.append(self._table_row(
                    [symbol, str(stats['count']), f"{stats['winrate']*100:.0f}%", self._format_money(stats['avg_profit']), best, worst],
                    [8, 7, 7, 9, 18, 18],
                    ['l', 'r', 'r', 'r', 'l', 'l']
                ))

            report.append(self._box_bottom())
            report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # TOP FACTORS
        # ───────────────────────────────────────────────────────────────────────
        if len(closed_7d) >= self.MIN_DAILY:
            factors = self.analyze_confluence_factors(trades_7d)
            if factors:
                report.append(self._box_top("TOP CONFLUENCE FACTORS"))

                report.append(self._table_row(
                    ['Factor', 'Trades', 'Win%', 'Edge'],
                    [26, 8, 8, 10],
                    ['l', 'r', 'r', 'r']
                ))
                report.append(self._box_divider())

                for factor_name, stats in list(factors.items())[:8]:
                    edge_str = f"+{stats['edge']*100:.0f}%" if stats['edge'] > 0 else f"{stats['edge']*100:.0f}%"
                    report.append(self._table_row(
                        [self._truncate(factor_name, 26), str(stats['count']), f"{stats['winrate']*100:.0f}%", edge_str],
                        [26, 8, 8, 10],
                        ['l', 'r', 'r', 'r']
                    ))

                # Best/Worst insight
                report.append(self._box_divider())
                best_factor = list(factors.items())[0]
                worst_factor = list(factors.items())[-1]
                report.append(self._box_row(f"BEST: {best_factor[0]} ({best_factor[1]['winrate']*100:.0f}% win, +{best_factor[1]['edge']*100:.0f}% edge)"))
                if worst_factor[1]['edge'] < 0:
                    report.append(self._box_row(f"WORST: {worst_factor[0]} ({worst_factor[1]['winrate']*100:.0f}% win, {worst_factor[1]['edge']*100:.0f}% edge)"))

                report.append(self._box_bottom())
                report.append("")

                # Best combos (compact)
                combos = self.analyze_confluence_combinations(trades_7d)
                if combos:
                    report.append(self._box_top("BEST FACTOR COMBOS"))
                    for i, combo in enumerate(combos[:3], 1):
                        report.append(self._box_row(f"{i}. {combo['factors']} = {combo['winrate']*100:.0f}% ({combo['count']} trades)"))
                    report.append(self._box_bottom())
                    report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # TIMING PATTERNS
        # ───────────────────────────────────────────────────────────────────────
        patterns = self.analyze_time_patterns(trades_7d)
        regime = self.analyze_market_regime(trades_7d)

        if len(closed_7d) >= self.MIN_DAILY:
            report.append(self._box_top("TIMING & MARKET"))

            if patterns['best_hours']:
                best_h, best_wr, best_cnt = patterns['best_hours'][0]
                report.append(self._box_row(f"Best Hour:  {best_h:02d}:00 UTC ({best_wr*100:.0f}% win, {best_cnt} trades)"))
            if patterns['worst_hours']:
                worst_h, worst_wr, worst_cnt = patterns['worst_hours'][0]
                report.append(self._box_row(f"Avoid:      {worst_h:02d}:00 UTC ({worst_wr*100:.0f}% win, {worst_cnt} trades)"))
            if patterns['best_days']:
                best_day, best_wr, best_cnt = patterns['best_days'][0]
                report.append(self._box_row(f"Best Day:   {best_day} ({best_wr*100:.0f}% win, {best_cnt} trades)"))
            if patterns['worst_days']:
                worst_day, worst_wr, worst_cnt = patterns['worst_days'][0]
                report.append(self._box_row(f"Worst Day:  {worst_day} ({worst_wr*100:.0f}% win, {worst_cnt} trades)"))

            if regime:
                report.append(self._box_divider())
                report.append(self._box_row(f"Ranging (ADX<20):   {regime['ranging']['count']} trades, {regime['ranging']['winrate']*100:.0f}% win"))
                report.append(self._box_row(f"Trending (ADX>20):  {regime['trending']['count']} trades, {regime['trending']['winrate']*100:.0f}% win"))
                report.append(self._box_row(f"Current:            {regime['current_dominant'].upper()}"))

            report.append(self._box_bottom())
            report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # RECOVERY SUMMARY (compact)
        # ───────────────────────────────────────────────────────────────────────
        init_rec = self.analyze_initial_vs_recovery(trades_7d)
        dca_analysis = self.analyze_dca_success_factors(trades_7d)
        grid_analysis = self.analyze_grid_performance(trades_7d)
        hedge_analysis = self.analyze_hedge_success_factors(trades_7d)

        has_recovery_data = init_rec or dca_analysis or grid_analysis or hedge_analysis

        if has_recovery_data:
            report.append(self._box_top("RECOVERY MECHANISMS"))

            if init_rec:
                init_wr = init_rec['initial']['winrate'] * 100
                rec_wr = init_rec['recovery']['winrate'] * 100
                diff = rec_wr - init_wr
                sign = "+" if diff > 0 else ""
                report.append(self._box_row(f"Initial: {init_rec['initial']['count']} trades @ {init_wr:.0f}% | Recovery: {init_rec['recovery']['count']} @ {rec_wr:.0f}% ({sign}{diff:.0f}%)"))

            if dca_analysis:
                report.append(self._box_row(f"DCA: {dca_analysis['total']} trades @ {dca_analysis['winrate']*100:.0f}% win, avg {dca_analysis['avg_dca_count']:.1f} levels"))
            if grid_analysis:
                report.append(self._box_row(f"Grid: {grid_analysis['total']} trades @ {grid_analysis['winrate']*100:.0f}% win"))
            if hedge_analysis:
                report.append(self._box_row(f"Hedge: {hedge_analysis['total']} trades @ {hedge_analysis['winrate']*100:.0f}% win"))

            report.append(self._box_bottom())
            report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # EXECUTION QUALITY (compact)
        # ───────────────────────────────────────────────────────────────────────
        exec_analysis = self.analyze_execution_quality(trades_7d)
        if exec_analysis and exec_analysis['sample_size'] > 0:
            report.append(self._box_top("EXECUTION QUALITY"))
            report.append(self._box_row(f"Slippage: avg {exec_analysis['avg_slippage']:.2f} pips, max {exec_analysis['max_slippage']:.2f} pips"))
            report.append(self._box_row(f"Exact fills: {exec_analysis['exact_fill_pct']:.0f}% | Spread: {exec_analysis['avg_spread']:.2f} pips avg"))
            report.append(self._box_bottom())
            report.append("")

        # ───────────────────────────────────────────────────────────────────────
        # CASCADE PROTECTION (compact)
        # ───────────────────────────────────────────────────────────────────────
        try:
            events = self.cascade_analyzer.parse_stop_out_log()
            if events:
                analysis = self.cascade_analyzer.analyze_stop_out_patterns(events)
                report.append(self._box_top("CASCADE PROTECTION"))
                report.append(self._box_row(f"Stop-outs: {analysis['total_stops']} | Avg loss: {self._format_money(analysis['avg_loss'])} | Max: {self._format_money(analysis['max_loss'])}"))
                if analysis.get('cascades_detected', 0) > 0:
                    report.append(self._box_row(f"Cascade events: {analysis['cascades_detected']}"))
                report.append(self._box_bottom())
                report.append("")
        except Exception:
            pass  # Skip cascade section if error

        # ───────────────────────────────────────────────────────────────────────
        # FOOTER
        # ───────────────────────────────────────────────────────────────────────
        report.append(self._box_top("SUMMARY"))
        report.append(self._box_row(f"7 Days:  {len(closed_7d)} trades | {winrate_7d*100:.1f}% win | {self._format_money(avg_profit_7d)} avg"))
        if len(closed_30d) > 0:
            wr_30d = sum(1 for t in closed_30d if t.get('outcome', {}).get('profit', 0) > 0) / len(closed_30d)
            avg_30d = sum(t.get('outcome', {}).get('profit', 0) for t in closed_30d) / len(closed_30d)
            report.append(self._box_row(f"30 Days: {len(closed_30d)} trades | {wr_30d*100:.1f}% win | {self._format_money(avg_30d)} avg"))
        report.append(self._box_row(f"All Time: {len(closed_all)} trades"))
        report.append(self._box_bottom())

        report_text = '\n'.join(report)

        # Save report
        report_file = self.report_dir / f'decision_report_{date_str}.txt'
        with open(str(report_file), 'w', encoding='utf-8', errors='ignore') as f:
            f.write(report_text)

        logger.info(f"[OK] Decision report saved to: {report_file}")

        return report_text, str(report_file)

    def generate_report_legacy(self):
        """LEGACY: Old verbose report format - kept for reference"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date_str = datetime.now().strftime('%Y-%m-%d')

        # Load data
        trades_7d = self.get_trades(days=7)
        trades_30d = self.get_trades(days=30)
        trades_all = self.get_trades(days=365)
        config = self.load_bot_config()

        closed_7d = [t for t in trades_7d if t.get('outcome', {}).get('status') == 'closed']
        closed_30d = [t for t in trades_30d if t.get('outcome', {}).get('status') == 'closed']
        closed_all = [t for t in trades_all if t.get('outcome', {}).get('status') == 'closed']

        # Build report
        report = []
        report.append("=" * 80)
        report.append("ML DECISION REPORT - Daily Recommendations")
        report.append("=" * 80)
        report.append(f"Generated: {timestamp}")
        report.append(f"Analysis Period: Last 7 days ({len(closed_7d)} closed trades)")
        report.append("")

        # Section 1: Action Items
        report.append("SECTION 1: IMMEDIATE ACTION ITEMS")
        report.append("=" * 80)

        recommendations = self.generate_recommendations(trades_7d)

        if len(closed_7d) < self.MIN_DAILY:
            report.append(f"[INFO] Need {self.MIN_DAILY - len(closed_7d)} more closed trades for reliable recommendations")
            report.append("")
            report.append("STATUS: Collecting data...")
            report.append(f"Progress: {len(closed_7d)}/{self.MIN_DAILY} trades")
        else:
            high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
            medium_priority = [r for r in recommendations if r.get('priority') == 'MEDIUM']
            low_priority = [r for r in recommendations if r.get('priority') == 'LOW']

            if high_priority:
                report.append("[!] HIGH PRIORITY CHANGES:")
                report.append("")
                for i, rec in enumerate(high_priority, 1):
                    report.append(f"{i}. Change {rec.get('setting', 'setting')}")
                    report.append(f"   Current: {rec.get('current', 'N/A')}")
                    report.append(f"   Optimal: {rec.get('optimal', 'N/A')}")
                    report.append(f"   Why: {rec.get('reason', 'N/A')}")
                    report.append(f"   Expected: {rec.get('impact', 'N/A')}")
                    report.append("")

            if medium_priority:
                report.append("[*] MEDIUM PRIORITY CHANGES:")
                report.append("")
                for i, rec in enumerate(medium_priority, 1):
                    report.append(f"{i}. {rec.get('setting', 'setting')}")
                    report.append(f"   Current: {rec.get('current', 'N/A')}")
                    report.append(f"   Optimal: {rec.get('optimal', 'N/A')}")
                    report.append(f"   Expected: {rec.get('impact', 'N/A')}")
                    report.append("")

            if low_priority:
                report.append("[~] LOW PRIORITY (Optional):")
                report.append("")
                for rec in low_priority:
                    report.append(f"  - {rec.get('setting', 'setting')}: {rec.get('current', 'N/A')} -> {rec.get('optimal', 'N/A')}")
                report.append("")

            # Info-level items (not enough data, etc.)
            info_items = [r for r in recommendations if r.get('priority') == 'INFO']

            if not (high_priority or medium_priority or low_priority):
                if info_items:
                    # Show info items instead of "NO CHANGES"
                    report.append("[i] STATUS:")
                    for info in info_items:
                        report.append(f"  - {info.get('message', 'Collecting data')}")
                        if info.get('action'):
                            report.append(f"    -> {info['action']}")
                    report.append("")
                else:
                    report.append("[OK] ALL SETTINGS OPTIMAL")
                    report.append("No changes recommended based on recent performance.")
                    report.append("")

        return '\n'.join(report), None

    def send_email(self, report_text, email_config):
        """Send report via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"ML Decision Report - {datetime.now().strftime('%Y-%m-%d')}"

            # Add report as body
            msg.attach(MIMEText(report_text, 'plain'))

            # Connect to SMTP server
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['from_email'], email_config['password'])
                server.send_message(msg)

            logger.info(f"[OK] Email sent to {email_config['to_email']}")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to send email: {e}")
            return False


if __name__ == '__main__':
    generator = DecisionReportGenerator()
    report_text, report_file = generator.generate_report()
    print(report_text)
