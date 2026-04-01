#!/usr/bin/env python3
"""
Automatic Strategy Tuning Scheduler

Runs auto-tuning every 8 hours to optimize strategy parameters
based on accumulated trade data and ML analysis.

Features:
1. Analyzes confluence factor performance
2. Updates HTF and Entry weights in strategy_config.py
3. Adjusts thresholds based on win rate analysis
4. Updates trading hours based on spread analysis
5. Logs all changes with full audit trail

Usage:
    python auto_tune_scheduler.py              # Run scheduler
    python auto_tune_scheduler.py --interval 4 # Custom interval (hours)
    python auto_tune_scheduler.py --once       # Run once and exit
"""

import sys
import time
import logging
from datetime import datetime
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_system.auto_tuner import AutoTuner

# Configure logging
log_dir = project_root / 'ml_system' / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'auto_tune_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutoTuneScheduler')


def check_sufficient_data() -> bool:
    """Check if there's sufficient new data for tuning"""
    try:
        import pandas as pd
        training_path = project_root / 'ml_system' / 'data' / 'training_data.csv'

        if not training_path.exists():
            logger.warning("Training data not found")
            return False

        df = pd.read_csv(training_path)
        trade_count = len(df)

        # Need at least 50 trades for reliable tuning
        if trade_count < 50:
            logger.warning(f"Insufficient data for tuning (need 50+, have {trade_count})")
            return False

        logger.info(f"Found {trade_count} trades - sufficient for tuning")
        return True

    except Exception as e:
        logger.error(f"Error checking data: {e}")
        return False


def run_auto_tuning() -> bool:
    """Execute auto-tuning"""
    logger.info("=" * 80)
    logger.info("SCHEDULED AUTO-TUNING STARTED")
    logger.info("=" * 80)

    try:
        tuner = AutoTuner(dry_run=False)
        result = tuner.run()
        tuner.print_report()

        if result.get('error'):
            logger.error(f"Auto-tuning failed: {result['error']}")
            return False

        changes_count = result.get('changes_count', 0)
        logger.info(f"Auto-tuning completed: {changes_count} changes applied")

        return True

    except Exception as e:
        logger.error(f"Auto-tuning error: {e}")
        import traceback
        traceback.print_exc()
        return False


def scheduled_tune():
    """Scheduled tuning job"""
    logger.info(f"Scheduled tuning triggered at {datetime.now()}")

    # Check data availability
    if not check_sufficient_data():
        logger.info("Skipping tuning - insufficient data")
        return

    # Run tuning
    success = run_auto_tuning()

    if success:
        logger.info("[OK] Scheduled tuning completed successfully")
    else:
        logger.error("[ERROR] Scheduled tuning failed")


def main():
    """Run automatic tuning scheduler"""
    import argparse

    parser = argparse.ArgumentParser(description='Auto-tune strategy scheduler')
    parser.add_argument('--interval', type=int, default=8,
                        help='Tuning interval in hours (default: 8)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("AUTO-TUNE SCHEDULER STARTED")
    logger.info("=" * 80)
    logger.info(f"Interval: Every {args.interval} hours")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 80)

    # Run initial tuning
    logger.info("Running initial auto-tuning...")
    scheduled_tune()

    if args.once:
        logger.info("Single run completed. Exiting.")
        return

    # Schedule periodic tuning
    try:
        import schedule

        schedule.every(args.interval).hours.do(scheduled_tune)

        logger.info("Scheduler active. Press Ctrl+C to stop.")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    except ImportError:
        logger.error("'schedule' module not installed. Install with: pip install schedule")
        logger.info("Running in simple loop mode instead...")

        # Simple loop fallback
        interval_seconds = args.interval * 3600
        while True:
            time.sleep(interval_seconds)
            scheduled_tune()

    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")


if __name__ == '__main__':
    main()
