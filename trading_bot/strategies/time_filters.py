"""
Time Filter Module
Manages trading time windows for mean reversion and breakout strategies

TIMEZONE: All trading hours are defined in TRUE GMT (Europe/London winter time).
The bot passes datetime.now(UK_TIMEZONE) to these filters — this is GMT in winter,
BST (GMT+1) in summer. ICMarkets MT5 server runs on EET (GMT+2 winter / GMT+3 summer),
but the bot does NOT use broker time for filtering. Config hours are in GMT.
"""

from datetime import datetime, time, timedelta
from typing import Dict, Optional
from config import strategy_config as cfg


class TimeFilter:
    """
    Time-based trading window filter
    Determines which strategy (if any) can trade at current time

    All times are in TRUE GMT (Europe/London winter). The bot passes
    datetime.now(UK_TIMEZONE) which is GMT in winter, BST in summer.
    NOT MT5 server time (which is EET/GMT+2 for ICMarkets).
    """

    def __init__(self, broker_gmt_offset: Optional[int] = None):
        """
        Initialize time filter

        Args:
            broker_gmt_offset: Legacy parameter, no longer used.
                              Bot now uses MT5 server time directly.
        """
        self.enable_filters = cfg.ENABLE_TIME_FILTERS
        self.broker_offset = broker_gmt_offset if broker_gmt_offset is not None else cfg.BROKER_GMT_OFFSET

        # Mean reversion windows (MT5 server time)
        self.mr_hours = set(cfg.MEAN_REVERSION_HOURS)
        self.mr_days = set(cfg.MEAN_REVERSION_DAYS)
        self.mr_sessions = set(cfg.MEAN_REVERSION_SESSIONS)

        # Breakout windows (MT5 server time)
        self.bo_hours = set(cfg.BREAKOUT_HOURS)
        self.bo_days = set(cfg.BREAKOUT_DAYS)
        self.bo_sessions = set(cfg.BREAKOUT_SESSIONS)

    def broker_time_to_gmt(self, broker_time: datetime) -> datetime:
        """
        Convert broker time to GMT/UTC

        Args:
            broker_time: Datetime in broker timezone

        Returns:
            Datetime in GMT/UTC
        """
        return broker_time - timedelta(hours=self.broker_offset)

    def gmt_to_broker_time(self, gmt_time: datetime) -> datetime:
        """
        Convert GMT/UTC to broker time

        Args:
            gmt_time: Datetime in GMT/UTC

        Returns:
            Datetime in broker timezone
        """
        return gmt_time + timedelta(hours=self.broker_offset)

    def get_session(self, current_time: datetime) -> str:
        """
        Determine which trading session is active

        Args:
            current_time: Current datetime (UTC)

        Returns:
            Session name: 'tokyo', 'london', 'new_york', 'sydney'
        """
        hour = current_time.hour

        for session_name, session_info in cfg.TRADE_SESSIONS.items():
            start_hour = int(session_info['start'].split(':')[0])
            end_hour = int(session_info['end'].split(':')[0])

            # Handle session crossing midnight
            if start_hour > end_hour:
                if hour >= start_hour or hour < end_hour:
                    return session_name
            else:
                if start_hour <= hour < end_hour:
                    return session_name

        return 'unknown'

    def _get_symbol_overrides(self, symbol: Optional[str] = None):
        """Get per-symbol hour/day/session overrides if configured"""
        if symbol and hasattr(cfg, 'SYMBOL_TRADING_HOURS'):
            return cfg.SYMBOL_TRADING_HOURS.get(symbol)
        return None

    def can_trade_mean_reversion(self, mt5_server_time: datetime, symbol: Optional[str] = None) -> bool:
        """
        MR trades any time — Q-table and confluences decide, not the clock.
        """
        if not cfg.MEAN_REVERSION_ENABLED:
            return False
        return True

    def can_trade_breakout(self, mt5_server_time: datetime, symbol: Optional[str] = None) -> bool:
        """
        Check if breakout strategy can trade now

        Args:
            mt5_server_time: Current time
            symbol: Optional symbol for per-symbol hour overrides
        """
        if not self.enable_filters:
            return True

        if not cfg.BREAKOUT_ENABLED:
            return False

        hour = mt5_server_time.hour
        day = mt5_server_time.weekday()
        session = self.get_session(mt5_server_time)

        # Use per-symbol overrides if configured, else defaults
        overrides = self._get_symbol_overrides(symbol)
        if overrides:
            bo_hours = set(overrides.get('bo_hours', cfg.BREAKOUT_HOURS))
            bo_days = set(overrides.get('bo_days', cfg.BREAKOUT_DAYS))
            bo_sessions = set(overrides.get('bo_sessions', cfg.BREAKOUT_SESSIONS))
        else:
            bo_hours = self.bo_hours
            bo_days = self.bo_days
            bo_sessions = self.bo_sessions

        return hour in bo_hours and day in bo_days and session in bo_sessions

    def get_active_strategy(self, mt5_server_time: datetime) -> Optional[str]:
        """
        Determine which strategy should be active now

        Args:
            mt5_server_time: Current MT5 server time (UTC+2/EET)

        Returns:
            'mean_reversion', 'breakout', or None
        """
        # Check mean reversion first (higher priority if both active)
        if self.can_trade_mean_reversion(mt5_server_time):
            return 'mean_reversion'

        # Then check breakout
        if self.can_trade_breakout(mt5_server_time):
            return 'breakout'

        return None

    def get_time_status(self, mt5_server_time: datetime) -> Dict:
        """
        Get comprehensive time filter status

        Args:
            mt5_server_time: Current MT5 server time (UTC+2/EET)

        Returns:
            Dict with detailed status information
        """
        # Use MT5 server time directly (no conversion)
        hour = mt5_server_time.hour
        day = mt5_server_time.weekday()
        day_name = mt5_server_time.strftime('%A')
        session = self.get_session(mt5_server_time)

        can_mr = self.can_trade_mean_reversion(mt5_server_time)
        can_bo = self.can_trade_breakout(mt5_server_time)
        active_strategy = self.get_active_strategy(mt5_server_time)

        return {
            'mt5_server_time': mt5_server_time.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'day': day,
            'day_name': day_name,
            'session': session,
            'filters_enabled': self.enable_filters,
            'can_trade_mean_reversion': can_mr,
            'can_trade_breakout': can_bo,
            'active_strategy': active_strategy,
            'mean_reversion_next_hour': self._next_hour_in(hour, self.mr_hours),
            'breakout_next_hour': self._next_hour_in(hour, self.bo_hours)
        }

    def _next_hour_in(self, current_hour: int, hour_set: set) -> Optional[int]:
        """
        Find next available hour in given set

        Args:
            current_hour: Current hour (0-23)
            hour_set: Set of valid hours

        Returns:
            Next valid hour, or None
        """
        sorted_hours = sorted(hour_set)

        for h in sorted_hours:
            if h > current_hour:
                return h

        # Return first hour of next day
        return sorted_hours[0] if sorted_hours else None

    def print_schedule(self):
        """
        Print the complete trading schedule
        """
        print("=" * 80)
        print("TRADING SCHEDULE")
        print("=" * 80)

        print("\n MEAN REVERSION STRATEGY")
        print(f"   Hours: {sorted(self.mr_hours)}")
        print(f"   Days: {[self._day_name(d) for d in sorted(self.mr_days)]}")
        print(f"   Sessions: {sorted(self.mr_sessions)}")

        print("\n BREAKOUT STRATEGY")
        print(f"   Enabled: {cfg.BREAKOUT_ENABLED}")
        if cfg.BREAKOUT_ENABLED:
            print(f"   Hours: {sorted(self.bo_hours)}")
            print(f"   Days: {[self._day_name(d) for d in sorted(self.bo_days)]}")
            print(f"   Sessions: {sorted(self.bo_sessions)}")

        print("\n HOURLY BREAKDOWN (UTC)")
        for hour in range(24):
            strategies = []
            if hour in self.mr_hours:
                strategies.append("MR")
            if hour in self.bo_hours and cfg.BREAKOUT_ENABLED:
                strategies.append("BO")

            if strategies:
                session = self._get_session_for_hour(hour)
                print(f"   {hour:02d}:00 - {', '.join(strategies)} ({session})")

    def _day_name(self, day_num: int) -> str:
        """Convert day number to name"""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return days[day_num] if 0 <= day_num < 7 else 'Unknown'

    def _get_session_for_hour(self, hour: int) -> str:
        """Get session name for given hour"""
        if 0 <= hour < 9:
            return 'Tokyo'
        elif 8 <= hour < 17:
            return 'London'
        elif 13 <= hour < 22:
            return 'NY'
        else:
            return 'Sydney'


def test_time_filters():
    """Test time filter functionality with timezone conversion"""

    # Test with different broker offsets
    print("=" * 80)
    print("TIMEZONE CONVERSION TEST")
    print("=" * 80)

    test_offsets = [0, 2, 3, -5]  # GMT, GMT+2, GMT+3, EST

    for offset in test_offsets:
        print(f"\n BROKER TIMEZONE: GMT{offset:+d}")
        print("-" * 80)

        filter = TimeFilter(broker_gmt_offset=offset)

        # Test time: 07:00 in broker time (should be MR window if GMT is 05:00)
        broker_time = datetime(2025, 12, 23, 7, 0)  # Tuesday 07:00 broker time
        gmt_time = filter.broker_time_to_gmt(broker_time)

        print(f"Broker Time: {broker_time.strftime('%H:%M')}")
        print(f"GMT Time:    {gmt_time.strftime('%H:%M')}")

        can_mr = filter.can_trade_mean_reversion(broker_time)
        can_bo = filter.can_trade_breakout(broker_time)
        active = filter.get_active_strategy(broker_time)

        print(f"Can trade MR: {can_mr}")
        print(f"Can trade BO: {can_bo}")
        print(f"Active Strategy: {active}")

    # Detailed test with configured offset
    print("\n" + "=" * 80)
    print("DETAILED TEST - Using configured offset")
    print("=" * 80)

    filter = TimeFilter()  # Uses cfg.BROKER_GMT_OFFSET

    print(f"\nBroker GMT Offset: {filter.broker_offset:+d} hours")
    print("All trading hours below are in GMT/UTC\n")
    filter.print_schedule()

    # Test specific broker times (assuming they come from MT5)
    test_times = [
        datetime(2025, 12, 23, 5, 0),   # Broker time
        datetime(2025, 12, 23, 12, 0),  # Broker time
        datetime(2025, 12, 23, 14, 0),  # Broker time
        datetime(2025, 12, 23, 3, 0),   # Broker time
        datetime(2025, 12, 23, 20, 0),  # Broker time
    ]

    print("\n" + "=" * 80)
    print("TEST SCENARIOS (Times from MT5/Broker)")
    print("=" * 80)

    for broker_time in test_times:
        status = filter.get_time_status(broker_time)
        print(f"\nBroker: {status['broker_time']} -> GMT: {status['gmt_time']}")
        print(f"  Day: {status['day_name']}, Session: {status['session']}")
        print(f"  Active Strategy: {status['active_strategy']}")
        print(f"  Can MR: {status['can_trade_mean_reversion']}, Can BO: {status['can_trade_breakout']}")


if __name__ == '__main__':
    test_time_filters()
