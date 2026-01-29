#!/usr/bin/env python3
"""
Trade Relationship Visualizer

Shows how all trades tie together: original positions, recovery cascades, outcomes
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta


class TradeRelationshipAnalyzer:
    """Analyze and visualize trade relationships and cascades"""

    def __init__(self, ml_outputs_dir: str = None):
        if ml_outputs_dir is None:
            project_root = Path(__file__).parent.parent
            ml_outputs_dir = project_root / "ml_system" / "outputs"
        self.outputs_dir = Path(ml_outputs_dir)

        # Try to load continuous trade log
        self.continuous_log = self.outputs_dir / "continuous_trade_log.jsonl"
        self.trades = self._load_trades()

    def _load_trades(self) -> List[Dict]:
        """Load trades from continuous log"""
        if not self.continuous_log.exists():
            return []

        trades = []
        with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    trade = json.loads(line)
                    trades.append(trade)
                except:
                    continue

        # Sort by entry time
        trades.sort(key=lambda x: x.get('entry_time', ''))
        return trades

    def get_last_week_trades(self) -> List[Dict]:
        """Get trades from last 7 days"""
        if not self.trades:
            return []

        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_trades = []

        for trade in self.trades:
            try:
                entry_time = datetime.fromisoformat(trade['entry_time'])
                if entry_time >= seven_days_ago:
                    recent_trades.append(trade)
            except:
                continue

        return recent_trades

    def analyze_trade_with_recovery(self, trade: Dict) -> Dict:
        """Extract all recovery info from a trade"""
        outcome = trade.get('outcome', {})
        recovery = outcome.get('recovery', {})
        partial_closes = outcome.get('partial_closes', {})
        exit_strategy = outcome.get('exit_strategy', {})

        return {
            'ticket': trade.get('ticket'),
            'symbol': trade.get('symbol'),
            'entry_time': trade.get('entry_time'),
            'entry_price': trade.get('entry_price'),
            'direction': trade.get('direction'),
            'volume': trade.get('volume'),
            'confluence': trade.get('confluence_score'),
            'strategy_type': trade.get('strategy_type'),

            # Hour and session
            'hour': datetime.fromisoformat(trade.get('entry_time')).hour if trade.get('entry_time') else None,
            'session': self._get_session(datetime.fromisoformat(trade.get('entry_time')).hour) if trade.get('entry_time') else None,

            # Recovery info
            'dca_count': recovery.get('dca_count', 0),
            'hedge_count': recovery.get('hedge_count', 0),
            'grid_count': recovery.get('grid_count', 0),
            'total_recovery_volume': recovery.get('total_recovery_volume', 0),
            'recovery_cost': recovery.get('recovery_cost', 0),
            'had_recovery': outcome.get('had_recovery', False),

            # Partial closes
            'partial_close_count': partial_closes.get('count', 0),
            'partial_profit': partial_closes.get('total_profit_from_partials', 0),

            # Exit info
            'exit_time': outcome.get('exit_time'),
            'exit_price': outcome.get('exit_price'),
            'profit': outcome.get('profit', 0),
            'net_profit': outcome.get('net_profit', 0),
            'hold_hours': outcome.get('hold_hours', 0),
            'status': outcome.get('status', 'open'),

            # Exit strategy
            'exit_method': exit_strategy.get('exit_method', 'unknown'),
            'pc1_triggered': exit_strategy.get('pc1_triggered', False),
            'pc2_triggered': exit_strategy.get('pc2_triggered', False),
            'trailing_triggered': exit_strategy.get('trailing_triggered', False),
            'final_pips': exit_strategy.get('final_pips', 0),
            'peak_pips': exit_strategy.get('peak_pips', 0),
        }

    def _get_session(self, hour: int) -> str:
        """Determine trading session"""
        if 0 <= hour < 8:
            return 'Tokyo'
        elif 8 <= hour < 13:
            return 'London'
        elif 13 <= hour < 21:
            return 'NY'
        else:
            return 'Sydney'

    def _is_spread_hour(self, hour: int) -> bool:
        """Check if hour is a spread hour"""
        return hour in [0, 9, 13, 20, 21]

    def generate_comprehensive_table(self) -> List[Dict]:
        """Generate comprehensive table of all trades"""
        last_week = self.get_last_week_trades()

        table_data = []
        for trade in last_week:
            analyzed = self.analyze_trade_with_recovery(trade)

            # Add classifications
            analyzed['is_spread_hour'] = self._is_spread_hour(analyzed['hour']) if analyzed['hour'] is not None else False
            analyzed['recovery_cascade'] = 'YES' if analyzed['had_recovery'] else 'NO'
            analyzed['cascade_type'] = self._get_cascade_type(analyzed)
            analyzed['outcome_category'] = self._categorize_outcome(analyzed)

            table_data.append(analyzed)

        return table_data

    def _get_cascade_type(self, trade: Dict) -> str:
        """Determine type of recovery cascade"""
        dca = trade['dca_count']
        hedge = trade['hedge_count']
        grid = trade['grid_count']

        if dca == 0 and hedge == 0 and grid == 0:
            return 'None'

        parts = []
        if dca > 0:
            parts.append(f'DCA×{dca}')
        if hedge > 0:
            parts.append(f'Hedge×{hedge}')
        if grid > 0:
            parts.append(f'Grid×{grid}')

        return '+'.join(parts)

    def _categorize_outcome(self, trade: Dict) -> str:
        """Categorize trade outcome"""
        profit = trade['net_profit']
        had_recovery = trade['had_recovery']

        if trade['status'] == 'open':
            return 'OPEN'

        if profit > 0:
            if had_recovery:
                return 'WIN_WITH_RECOVERY'
            else:
                return 'CLEAN_WIN'
        else:
            if had_recovery:
                return 'LOSS_WITH_RECOVERY'
            else:
                return 'CLEAN_LOSS'

    def print_table(self, table_data: List[Dict]):
        """Print comprehensive ASCII table"""
        print("=" * 180)
        print("COMPLETE TRADE ANALYSIS - LAST 7 DAYS")
        print("=" * 180)
        print()

        if not table_data:
            print("No trades found in last 7 days (or no continuous_trade_log.jsonl exists)")
            print()
            print("To generate this data, the continuous ML logger needs to run.")
            print("Location: ml_system/continuous_logger.py")
            return

        # Summary stats first
        closed_trades = [t for t in table_data if t['status'] == 'closed']
        with_recovery = [t for t in closed_trades if t['had_recovery']]
        without_recovery = [t for t in closed_trades if not t['had_recovery']]
        spread_hour_trades = [t for t in closed_trades if t['is_spread_hour']]

        print("SUMMARY STATISTICS:")
        print(f"  Total trades: {len(table_data)}")
        print(f"  Closed: {len(closed_trades)} | Open: {len(table_data) - len(closed_trades)}")
        print(f"  With recovery: {len(with_recovery)} ({len(with_recovery)/len(closed_trades)*100:.1f}%)" if closed_trades else "  With recovery: 0")
        print(f"  Without recovery: {len(without_recovery)} ({len(without_recovery)/len(closed_trades)*100:.1f}%)" if closed_trades else "  Without recovery: 0")
        print(f"  During spread hours: {len(spread_hour_trades)} ({len(spread_hour_trades)/len(closed_trades)*100:.1f}%)" if closed_trades else "  During spread hours: 0")
        print()

        if with_recovery:
            avg_with = sum(t['net_profit'] for t in with_recovery) / len(with_recovery)
            print(f"  Avg profit WITH recovery: ${avg_with:.2f}")
        if without_recovery:
            avg_without = sum(t['net_profit'] for t in without_recovery) / len(without_recovery)
            print(f"  Avg profit WITHOUT recovery: ${avg_without:.2f}")
        print()

        # Detailed table
        print("DETAILED TRADE BREAKDOWN:")
        print("-" * 180)

        # Header
        print(f"{'Ticket':>10} {'Symbol':<8} {'Entry Time':>16} {'Hour':>4} {'Spread?':>7} {'Conf':>4} "
              f"{'Direction':>5} {'Vol':>5} {'Cascade':>15} {'Exit':>8} {'Profit':>8} {'Net$':>8} {'Outcome':>18}")
        print("-" * 180)

        # Rows
        for trade in table_data:
            ticket = str(trade['ticket'])[-8:] if trade['ticket'] else 'N/A'
            symbol = trade['symbol'][:8] if trade['symbol'] else 'N/A'

            entry_time_str = trade['entry_time'][:16] if trade['entry_time'] else 'N/A'
            hour = str(trade['hour']) if trade['hour'] is not None else 'N/A'
            spread = 'YES' if trade['is_spread_hour'] else 'NO'
            conf = str(trade['confluence']) if trade['confluence'] else 'N/A'
            direction = (trade['direction'] or 'N/A')[:5]
            volume = f"{trade['volume']:.2f}" if trade['volume'] else 'N/A'
            cascade = trade['cascade_type'][:15]
            exit_method = (trade['exit_method'] or 'open')[:8]
            profit = f"${trade['profit']:.2f}" if trade['status'] == 'closed' else 'OPEN'
            net_profit = f"${trade['net_profit']:.2f}" if trade['status'] == 'closed' else 'OPEN'
            outcome = trade['outcome_category']

            print(f"{ticket:>10} {symbol:<8} {entry_time_str:>16} {hour:>4} {spread:>7} {conf:>4} "
                  f"{direction:>5} {volume:>5} {cascade:>15} {exit_method:>8} {profit:>8} {net_profit:>8} {outcome:>18}")

        print("-" * 180)
        print()

    def print_cascade_detail(self, table_data: List[Dict]):
        """Print detailed cascade analysis"""
        print("=" * 180)
        print("RECOVERY CASCADE DETAIL")
        print("=" * 180)
        print()

        trades_with_recovery = [t for t in table_data if t['had_recovery'] and t['status'] == 'closed']

        if not trades_with_recovery:
            print("No trades with recovery in last 7 days")
            return

        print(f"{'Ticket':>10} {'Hour':>4} {'Spread?':>7} {'Conf':>4} {'Cascade Type':>20} {'Recovery Cost':>13} {'Gross Profit':>12} {'Net Profit':>11} {'Impact':>10}")
        print("-" * 180)

        for trade in trades_with_recovery:
            ticket = str(trade['ticket'])[-8:] if trade['ticket'] else 'N/A'
            hour = str(trade['hour']) if trade['hour'] is not None else 'N/A'
            spread = 'YES' if trade['is_spread_hour'] else 'NO'
            conf = str(trade['confluence']) if trade['confluence'] else 'N/A'
            cascade = trade['cascade_type']
            recovery_cost = f"${trade['recovery_cost']:.2f}"
            gross = f"${trade['profit']:.2f}"
            net = f"${trade['net_profit']:.2f}"
            impact = f"${trade['profit'] - trade['net_profit']:.2f}"

            print(f"{ticket:>10} {hour:>4} {spread:>7} {conf:>4} {cascade:>20} {recovery_cost:>13} {gross:>12} {net:>11} {impact:>10}")

        print("-" * 180)
        print()

        # Summary of cascade impact
        total_recovery_cost = sum(t['recovery_cost'] for t in trades_with_recovery)
        total_gross = sum(t['profit'] for t in trades_with_recovery)
        total_net = sum(t['net_profit'] for t in trades_with_recovery)

        print(f"TOTAL RECOVERY COST: ${total_recovery_cost:.2f}")
        print(f"TOTAL GROSS PROFIT: ${total_gross:.2f}")
        print(f"TOTAL NET PROFIT: ${total_net:.2f}")
        print(f"RECOVERY IMPACT: ${total_gross - total_net:.2f} (what recovery cost you)")
        print()

    def print_spread_hour_analysis(self, table_data: List[Dict]):
        """Analyze trades during spread hours"""
        print("=" * 180)
        print("SPREAD HOUR ANALYSIS")
        print("=" * 180)
        print()

        spread_trades = [t for t in table_data if t['is_spread_hour'] and t['status'] == 'closed']
        normal_trades = [t for t in table_data if not t['is_spread_hour'] and t['status'] == 'closed']

        if not spread_trades:
            print("No closed trades during spread hours (0, 9, 13, 20, 21)")
            return

        print("DURING SPREAD HOURS (0, 9, 13, 20, 21):")
        print(f"  Total trades: {len(spread_trades)}")
        print(f"  With recovery: {len([t for t in spread_trades if t['had_recovery']])} ({len([t for t in spread_trades if t['had_recovery']])/len(spread_trades)*100:.1f}%)")
        print(f"  Avg net profit: ${sum(t['net_profit'] for t in spread_trades)/len(spread_trades):.2f}")
        print()

        print("DURING NORMAL HOURS:")
        print(f"  Total trades: {len(normal_trades)}")
        if normal_trades:
            print(f"  With recovery: {len([t for t in normal_trades if t['had_recovery']])} ({len([t for t in normal_trades if t['had_recovery']])/len(normal_trades)*100:.1f}%)")
            print(f"  Avg net profit: ${sum(t['net_profit'] for t in normal_trades)/len(normal_trades):.2f}")
        print()

        print("COMPARISON:")
        spread_avg = sum(t['net_profit'] for t in spread_trades)/len(spread_trades) if spread_trades else 0
        normal_avg = sum(t['net_profit'] for t in normal_trades)/len(normal_trades) if normal_trades else 0
        print(f"  Spread hours avg: ${spread_avg:.2f}")
        print(f"  Normal hours avg: ${normal_avg:.2f}")
        print(f"  Difference: ${spread_avg - normal_avg:.2f} (negative = spread hours worse)")
        print()

    def save_report(self, table_data: List[Dict]):
        """Save detailed report"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'period': 'last_7_days',
            'total_trades': len(table_data),
            'trades': table_data
        }

        output_path = self.outputs_dir / "trade_relationship_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output_path


def main():
    analyzer = TradeRelationshipAnalyzer()
    table_data = analyzer.generate_comprehensive_table()

    analyzer.print_table(table_data)
    analyzer.print_cascade_detail(table_data)
    analyzer.print_spread_hour_analysis(table_data)

    output_path = analyzer.save_report(table_data)
    print(f"Full report saved to: {output_path}")


if __name__ == '__main__':
    main()
