"""
ML-Driven Recovery Trigger Optimizer
Analyzes historical recovery triggers and suggests optimal thresholds
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class RecoveryTriggerOptimizer:
    """
    Analyzes recovery_triggers.jsonl logs and recommends optimal trigger thresholds.
    Helps find sweet spot between too early (wasting capital) and too late (deep losses).
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize optimizer

        Args:
            output_dir: Directory containing recovery_triggers.jsonl
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / 'ml_system' / 'output'

        self.output_dir = Path(output_dir)
        self.triggers_file = self.output_dir / 'recovery_triggers.jsonl'
        self.recommendations_file = self.output_dir / 'recovery_thresholds_optimized.json'

    def load_triggers(self, min_triggers: int = 20) -> Optional[pd.DataFrame]:
        """
        Load recovery triggers from log file.

        Args:
            min_triggers: Minimum triggers required for analysis

        Returns:
            DataFrame or None if insufficient data
        """
        if not self.triggers_file.exists():
            print(f"[OPTIMIZER] No triggers file found: {self.triggers_file}")
            return None

        triggers = []
        try:
            with open(self.triggers_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        trigger = json.loads(line)
                        triggers.append(trigger)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[OPTIMIZER] Failed to read triggers file: {e}")
            return None

        if len(triggers) < min_triggers:
            print(f"[OPTIMIZER] Insufficient triggers: {len(triggers)}/{min_triggers}")
            return None

        return pd.DataFrame(triggers)

    def analyze_recovery_success(self, triggers_df: pd.DataFrame) -> Dict:
        """
        Analyze which recovery types and thresholds correlate with success.
        Note: This is a simplified analysis based on trigger patterns.
        Full analysis requires linking triggers to trade outcomes.

        Args:
            triggers_df: DataFrame of recovery triggers

        Returns:
            Dict with analysis results
        """
        analysis = {}

        # Analyze by recovery type
        for recovery_type in triggers_df['recovery_type'].unique():
            type_df = triggers_df[triggers_df['recovery_type'] == recovery_type]

            # Calculate statistics
            stats = {
                'count': len(type_df),
                'avg_pips_underwater': type_df['pips_underwater'].mean(),
                'avg_time_since_entry': type_df['time_since_entry_minutes'].mean(),
                'avg_volume': type_df['volume'].mean(),
                'avg_trigger_threshold': type_df['trigger_threshold'].mean()
            }

            # Analyze ADX distribution (when available)
            adx_available = type_df['current_adx'].notna().sum()
            if adx_available > 0:
                stats['avg_adx'] = type_df['current_adx'].mean()
                stats['adx_available_pct'] = (adx_available / len(type_df)) * 100

            # Group by symbols if available
            if 'symbol' in type_df.columns:
                symbol_counts = type_df['symbol'].value_counts().to_dict()
                stats['by_symbol'] = symbol_counts

            analysis[recovery_type] = stats

        return analysis

    def generate_recommendations(self, analysis: Dict) -> Dict:
        """
        Generate recommended thresholds based on analysis.

        Args:
            analysis: Analysis results from analyze_recovery_success

        Returns:
            Dict with recommended thresholds
        """
        recommendations = {
            'last_updated': datetime.now().isoformat(),
            'based_on_triggers': sum(a['count'] for a in analysis.values()),
            'recommendations': {},
            'note': 'These are data-driven suggestions. Test carefully before applying.'
        }

        # DCA recommendations
        if 'dca' in analysis:
            dca_stats = analysis['dca']
            current_threshold = dca_stats.get('avg_trigger_threshold', 30)

            # Recommend slightly higher threshold if lots of DCA triggers
            # (suggests entering too early)
            if dca_stats['count'] > 50:
                recommended = current_threshold * 1.1  # 10% higher
                recommendations['recommendations']['dca_trigger_pips'] = {
                    'current_avg': round(current_threshold, 1),
                    'recommended': round(recommended, 1),
                    'reason': f"High DCA frequency ({dca_stats['count']} triggers) suggests entering too early"
                }

        # Hedge recommendations
        if 'hedge' in analysis:
            hedge_stats = analysis['hedge']
            current_threshold = hedge_stats.get('avg_trigger_threshold', 45)

            # Recommend adjustments based on ADX patterns
            if 'avg_adx' in hedge_stats and hedge_stats['avg_adx'] > 30:
                recommendations['recommendations']['hedge_trigger_pips'] = {
                    'current_avg': round(current_threshold, 1),
                    'recommended': round(current_threshold * 1.2, 1),
                    'reason': f"High average ADX ({hedge_stats['avg_adx']:.1f}) suggests trending markets need wider hedges"
                }

        # Grid recommendations (grid triggers on profit)
        if 'grid' in analysis:
            grid_stats = analysis['grid']
            current_threshold = grid_stats.get('avg_trigger_threshold', 15)

            recommendations['recommendations']['grid_spacing_pips'] = {
                'current_avg': round(current_threshold, 1),
                'note': f"Grid triggered {grid_stats['count']} times at avg {current_threshold:.1f} pips profit"
            }

        return recommendations

    def optimize_and_save(self, min_triggers: int = 20) -> Optional[Dict]:
        """
        Analyze triggers and save recommendations to JSON file.

        Args:
            min_triggers: Minimum triggers required

        Returns:
            Dict with recommendations or None if failed
        """
        # Load triggers
        triggers_df = self.load_triggers(min_triggers)
        if triggers_df is None:
            return None

        # Analyze
        analysis = self.analyze_recovery_success(triggers_df)

        # Generate recommendations
        recommendations = self.generate_recommendations(analysis)

        # Save to file (UTF-8 encoding)
        try:
            with open(self.recommendations_file, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, indent=2, ensure_ascii=False)

            print(f"[OPTIMIZER] Saved recovery threshold recommendations: {self.recommendations_file}")
            print(f"[OPTIMIZER] Based on {recommendations['based_on_triggers']} triggers")

            return recommendations

        except Exception as e:
            print(f"[OPTIMIZER] Failed to save recommendations: {e}")
            return None

    def get_current_recommendations(self) -> Optional[Dict]:
        """
        Read current recommendations from file.

        Returns:
            Dict of recommendations or None if file doesn't exist
        """
        if not self.recommendations_file.exists():
            return None

        try:
            with open(self.recommendations_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[OPTIMIZER] Failed to read recommendations: {e}")
            return None


if __name__ == '__main__':
    # Test the optimizer
    optimizer = RecoveryTriggerOptimizer()
    result = optimizer.optimize_and_save()

    if result:
        print("\n=== RECOVERY THRESHOLD RECOMMENDATIONS ===")
        for recovery_type, rec in result['recommendations'].items():
            print(f"\n{recovery_type}:")
            for key, value in rec.items():
                print(f"  {key}: {value}")
    else:
        print("[INFO] Not enough data yet for recommendations (need 20+ triggers)")
