#!/usr/bin/env python3
"""
ML Readiness Assessment

Determines if ML has enough data to exit shadow mode and take control of the bot.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class MLReadinessAssessment:
    """Assess if ML system has sufficient data for autonomous operation"""

    def __init__(self, ml_outputs_dir: str = None):
        if ml_outputs_dir is None:
            project_root = Path(__file__).parent.parent
            ml_outputs_dir = project_root / "ml_system" / "outputs"
        self.outputs_dir = Path(ml_outputs_dir)

        # Load all available data
        self.recovery_patterns = self._load_json("recovery_pattern_analysis.json")
        self.time_performance = self._load_json("time_performance.json")
        self.signal_quality = self._load_json("signal_quality.json")
        self.analysis_summary = self._load_json("analysis_summary.json")

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file"""
        filepath = self.outputs_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def assess_data_volume(self) -> Dict:
        """Assess quantity of data available"""

        # Check recovery pattern data
        dca_by_conf = self.recovery_patterns.get('dca_patterns', {}).get('by_confluence_score', {})

        total_trades = 0
        min_trades_per_conf = float('inf')
        max_trades_per_conf = 0
        conf_with_data = 0

        for score, data in dca_by_conf.items():
            count = data.get('count', 0)
            if count > 0:
                total_trades += count
                conf_with_data += 1
                min_trades_per_conf = min(min_trades_per_conf, count)
                max_trades_per_conf = max(max_trades_per_conf, count)

        # Check time performance data
        by_hour = self.time_performance.get('by_hour', {})
        hours_with_data = len([h for h, d in by_hour.items() if d.get('trades', 0) > 0])
        total_hour_trades = sum(d.get('trades', 0) for d in by_hour.values())

        # Check for continuous log
        continuous_log = self.outputs_dir / "continuous_trade_log.jsonl"
        has_continuous_log = continuous_log.exists()
        continuous_log_size = continuous_log.stat().st_size if has_continuous_log else 0

        return {
            'total_trades': total_trades,
            'confluence_levels_with_data': conf_with_data,
            'min_trades_per_confluence': min_trades_per_conf if conf_with_data > 0 else 0,
            'max_trades_per_confluence': max_trades_per_conf,
            'hours_with_data': hours_with_data,
            'total_hour_trades': total_hour_trades,
            'has_continuous_log': has_continuous_log,
            'continuous_log_size_mb': continuous_log_size / (1024 * 1024) if has_continuous_log else 0,
            'data_timestamp': self.recovery_patterns.get('timestamp', 'unknown')
        }

    def assess_data_quality(self) -> Dict:
        """Assess quality and recency of data"""

        # Check data freshness
        timestamp_str = self.recovery_patterns.get('timestamp', '')
        data_age_days = 0
        is_stale = False

        if timestamp_str:
            try:
                data_timestamp = datetime.fromisoformat(timestamp_str)
                data_age_days = (datetime.now() - data_timestamp).days
                is_stale = data_age_days > 7  # Data older than 1 week is stale
            except:
                is_stale = True

        # Check statistical significance
        dca_by_conf = self.recovery_patterns.get('dca_patterns', {}).get('by_confluence_score', {})

        statistically_significant = []
        insufficient_data = []

        for score, data in dca_by_conf.items():
            count = data.get('count', 0)
            if count >= 30:  # Standard minimum for statistical significance
                statistically_significant.append(int(score))
            elif count > 0:
                insufficient_data.append({'confluence': int(score), 'trades': count, 'needed': 30 - count})

        return {
            'data_age_days': data_age_days,
            'is_stale': is_stale,
            'statistically_significant_confluences': statistically_significant,
            'insufficient_data_confluences': insufficient_data,
            'total_significant': len(statistically_significant),
            'total_insufficient': len(insufficient_data)
        }

    def assess_ml_readiness(self) -> Dict:
        """Determine if ML is ready to take control"""

        volume = self.assess_data_volume()
        quality = self.assess_data_quality()

        # Readiness criteria
        criteria = {
            'min_total_trades': 100,
            'min_trades_per_confluence': 30,
            'min_confluence_levels': 5,
            'max_data_age_days': 7,
            'requires_continuous_log': True
        }

        # Check each criterion
        checks = {
            'total_trades': {
                'pass': volume['total_trades'] >= criteria['min_total_trades'],
                'value': volume['total_trades'],
                'required': criteria['min_total_trades'],
                'message': f"{volume['total_trades']} trades (need {criteria['min_total_trades']})"
            },
            'trades_per_confluence': {
                'pass': volume['min_trades_per_confluence'] >= criteria['min_trades_per_confluence'],
                'value': volume['min_trades_per_confluence'],
                'required': criteria['min_trades_per_confluence'],
                'message': f"Min {volume['min_trades_per_confluence']} trades per confluence (need {criteria['min_trades_per_confluence']})"
            },
            'confluence_coverage': {
                'pass': volume['confluence_levels_with_data'] >= criteria['min_confluence_levels'],
                'value': volume['confluence_levels_with_data'],
                'required': criteria['min_confluence_levels'],
                'message': f"{volume['confluence_levels_with_data']} confluence levels (need {criteria['min_confluence_levels']})"
            },
            'data_freshness': {
                'pass': not quality['is_stale'],
                'value': quality['data_age_days'],
                'required': criteria['max_data_age_days'],
                'message': f"Data is {quality['data_age_days']} days old (need < {criteria['max_data_age_days']} days)"
            },
            'continuous_logging': {
                'pass': volume['has_continuous_log'],
                'value': volume['has_continuous_log'],
                'required': criteria['requires_continuous_log'],
                'message': "Continuous trade log exists" if volume['has_continuous_log'] else "No continuous trade log found"
            }
        }

        # Calculate readiness score
        passed = sum(1 for c in checks.values() if c['pass'])
        total = len(checks)
        readiness_score = (passed / total) * 100

        # Determine recommendation
        if readiness_score == 100:
            recommendation = "READY"
            status = "ML has sufficient data to exit shadow mode"
        elif readiness_score >= 60:
            recommendation = "PARTIAL"
            status = "ML has some data but needs more for full autonomy"
        else:
            recommendation = "NOT_READY"
            status = "ML needs significantly more data before taking control"

        return {
            'readiness_score': readiness_score,
            'recommendation': recommendation,
            'status': status,
            'checks': checks,
            'passed': passed,
            'total': total,
            'volume_assessment': volume,
            'quality_assessment': quality
        }

    def get_action_plan(self, assessment: Dict) -> List[str]:
        """Generate action plan based on assessment"""

        actions = []
        checks = assessment['checks']
        volume = assessment['volume_assessment']
        quality = assessment['quality_assessment']

        # Data collection actions
        if not checks['continuous_logging']['pass']:
            actions.append("START continuous ML logger (ml_system/continuous_logger.py)")
            actions.append("  This will collect detailed trade data going forward")

        if checks['data_freshness']['pass'] == False:
            actions.append(f"UPDATE data - current data is {quality['data_age_days']} days old")
            actions.append("  Run ML analysis on recent closed trades")

        if not checks['total_trades']['pass']:
            trades_needed = checks['total_trades']['required'] - checks['total_trades']['value']
            actions.append(f"COLLECT {trades_needed} more trades (currently have {checks['total_trades']['value']})")
            actions.append("  Continue running bot to gather more data")

        if not checks['trades_per_confluence']['pass']:
            actions.append(f"BALANCE data across confluence levels")
            actions.append(f"  Some levels have < 30 trades (minimum for significance)")
            if quality['insufficient_data_confluences']:
                for item in quality['insufficient_data_confluences'][:3]:
                    actions.append(f"    Confluence {item['confluence']}: {item['trades']} trades (need {item['needed']} more)")

        # What can be done NOW
        if assessment['recommendation'] == 'PARTIAL':
            actions.append("")
            actions.append("WHAT YOU CAN DO NOW:")
            actions.append("  [OK] Use ML-recommended strategies (ADX stops, spread hour blocking)")
            actions.append("  [OK] Monitor ML insights from existing data")
            actions.append("  [OK] Use ML for pattern detection and alerts")
            actions.append("  [X] Full autonomous decision-making (need more data)")

        return actions

    def print_assessment(self, assessment: Dict):
        """Print human-readable assessment"""
        print("=" * 100)
        print("ML READINESS ASSESSMENT - Can ML Exit Shadow Mode?")
        print("=" * 100)
        print()

        # Overall verdict
        rec = assessment['recommendation']
        score = assessment['readiness_score']

        if rec == 'READY':
            emoji = "🟢"
        elif rec == 'PARTIAL':
            emoji = "🟡"
        else:
            emoji = "🔴"

        print(f"{emoji} READINESS SCORE: {score:.1f}%")
        print(f"   Recommendation: {rec}")
        print(f"   Status: {assessment['status']}")
        print()

        # Detailed checks
        print("READINESS CRITERIA:")
        print("-" * 100)
        for name, check in assessment['checks'].items():
            status = "[OK] PASS" if check['pass'] else "[X] FAIL"
            print(f"  {status:8} {name:25} {check['message']}")
        print()

        # Data summary
        volume = assessment['volume_assessment']
        quality = assessment['quality_assessment']

        print("DATA SUMMARY:")
        print("-" * 100)
        print(f"  Total trades analyzed: {volume['total_trades']}")
        print(f"  Confluence levels with data: {volume['confluence_levels_with_data']}")
        print(f"  Hours with data: {volume['hours_with_data']}/24")
        print(f"  Data age: {quality['data_age_days']} days")
        print(f"  Statistically significant levels: {quality['total_significant']}")
        print(f"  Insufficient data levels: {quality['total_insufficient']}")
        print(f"  Continuous log: {'YES' if volume['has_continuous_log'] else 'NO'}")
        print()

        # Action plan
        actions = self.get_action_plan(assessment)
        print("ACTION PLAN:")
        print("-" * 100)
        for action in actions:
            print(f"  {action}")
        print()

        print("=" * 100)

    def save_assessment(self, assessment: Dict):
        """Save assessment to file"""
        output_path = self.outputs_dir / "ml_readiness_assessment.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(assessment, f, indent=2, ensure_ascii=False)
        return output_path


def main():
    assessor = MLReadinessAssessment()
    assessment = assessor.assess_ml_readiness()
    assessor.print_assessment(assessment)
    output_path = assessor.save_assessment(assessment)
    print(f"Assessment saved to: {output_path}")


if __name__ == '__main__':
    main()
