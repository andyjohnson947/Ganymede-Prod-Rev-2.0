# Ganymede Trading Bot - Claude Code Configuration

## Project Overview
MT5 trading bot with Mean Reversion (MR) and Breakout (BO) strategies.
- Python path: `/c/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe`
- MT5 Broker: ICMarkets (UTC time)
- Symbols: EURUSD, GBPUSD

## Custom Commands

### /bot-status
Check current bot status, positions, and account.
```bash
"/c/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe" << 'PYEOF'
import MetaTrader5 as mt5
from datetime import datetime

mt5.initialize()

# Server time
tick = mt5.symbol_info_tick('EURUSD')
server_time = datetime.utcfromtimestamp(tick.time)
print(f"Server Time (UTC): {server_time.strftime('%Y-%m-%d %H:%M')} (Hour {server_time.hour})")

# Account
acc = mt5.account_info()
print(f"Balance: ${acc.balance:.2f} | Equity: ${acc.equity:.2f} | Profit: ${acc.profit:.2f}")

# Positions
positions = mt5.positions_get()
print(f"\nOpen Positions: {len(positions) if positions else 0}")
if positions:
    for p in positions:
        ptype = 'BUY' if p.type == 0 else 'SELL'
        pips = (p.price_current - p.price_open) / 0.0001 if p.type == 0 else (p.price_open - p.price_current) / 0.0001
        print(f"  {p.symbol} {ptype} #{p.ticket} @ {p.price_open:.5f} -> {p.price_current:.5f} ({pips:+.1f} pips) ${p.profit:.2f}")

mt5.shutdown()
PYEOF
```

### /signals
Check current signal conditions for both pairs.
```bash
"/c/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe" << 'PYEOF'
import MetaTrader5 as mt5
from datetime import datetime

mt5.initialize()

MR_HOURS = [0, 1, 2, 4, 6, 7, 8, 14, 16]
BO_HOURS = [0, 1, 5, 6, 16, 20, 21, 22, 23]

tick = mt5.symbol_info_tick('EURUSD')
server_time = datetime.utcfromtimestamp(tick.time)
hour = server_time.hour
day = server_time.weekday()

print(f"Server: {server_time.strftime('%H:%M')} UTC (Hour {hour}, Day {day})")
print(f"MR Active: {hour in MR_HOURS and day in [0,1,2,3]}")
print(f"BO Active: {hour in BO_HOURS and day in [0,1,2,3,4]}")

for symbol in ['EURUSD', 'GBPUSD']:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 20)
    closes = [r['close'] for r in rates]
    highs = [r['high'] for r in rates]
    lows = [r['low'] for r in rates]

    tick = mt5.symbol_info_tick(symbol)

    # Simple ADX approximation
    plus_dm = sum(max(highs[i] - highs[i-1], 0) for i in range(1, 14)) / 14
    minus_dm = sum(max(lows[i-1] - lows[i], 0) for i in range(1, 14)) / 14
    atr = sum(highs[i] - lows[i] for i in range(14)) / 14
    plus_di = 100 * plus_dm / atr if atr else 0
    minus_di = 100 * minus_dm / atr if atr else 0

    swing_high = max(highs[-10:])
    swing_low = min(lows[-10:])
    at_level = tick.bid >= swing_high * 0.999 or tick.bid <= swing_low * 1.001

    print(f"\n{symbol}: {tick.bid:.5f}")
    print(f"  +DI: {plus_di:.1f} | -DI: {minus_di:.1f} | Gap: {abs(plus_di-minus_di):.1f}")
    print(f"  Swing H: {swing_high:.5f} | L: {swing_low:.5f} | At level: {at_level}")

mt5.shutdown()
PYEOF
```

### /trades-today
Show today's completed trades.
```bash
"/c/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe" << 'PYEOF'
import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()

today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
deals = mt5.history_deals_get(today, datetime.now() + timedelta(days=1))

if deals:
    total_pnl = 0
    print(f"{'Time':>5} | {'Symbol':>6} | {'Type':>4} | {'P/L':>8} | Comment")
    print("-" * 55)
    for d in deals:
        if d.profit != 0 or d.entry == 0:  # Entry or exit with P/L
            ts = datetime.fromtimestamp(d.time)
            dtype = 'BUY' if d.type == 0 else 'SELL' if d.type == 1 else 'BAL'
            cmt = (d.comment[:20] if d.comment else "")
            print(f"{ts.strftime('%H:%M'):>5} | {d.symbol:>6} | {dtype:>4} | ${d.profit:>7.2f} | {cmt}")
            total_pnl += d.profit
    print("-" * 55)
    print(f"Total P/L: ${total_pnl:.2f}")
else:
    print("No deals today")

mt5.shutdown()
PYEOF
```

### /hours
Show current hour vs allowed trading hours.
```bash
"/c/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe" << 'PYEOF'
import MetaTrader5 as mt5
from datetime import datetime

mt5.initialize()

MR_HOURS = [0, 1, 2, 4, 6, 7, 8, 14, 16]
BO_HOURS = [0, 1, 5, 6, 16, 20, 21, 22, 23]
MR_DAYS = [0, 1, 2, 3]
BO_DAYS = [0, 1, 2, 3, 4]

tick = mt5.symbol_info_tick('EURUSD')
t = datetime.utcfromtimestamp(tick.time)

print(f"Current: {t.strftime('%Y-%m-%d %H:%M')} UTC (Hour {t.hour}, Day {t.weekday()})")
print()
print(f"MR Hours: {sorted(MR_HOURS)}")
print(f"   Current hour {t.hour} allowed: {'YES' if t.hour in MR_HOURS else 'NO'}")
print(f"   Current day {t.weekday()} allowed: {'YES' if t.weekday() in MR_DAYS else 'NO'}")
print()
print(f"BO Hours: {sorted(BO_HOURS)}")
print(f"   Current hour {t.hour} allowed: {'YES' if t.hour in BO_HOURS else 'NO'}")
print(f"   Current day {t.weekday()} allowed: {'YES' if t.weekday() in BO_DAYS else 'NO'}")

# Next trading window
for h in range(t.hour + 1, t.hour + 25):
    check_h = h % 24
    if check_h in MR_HOURS or check_h in BO_HOURS:
        strat = 'MR' if check_h in MR_HOURS else 'BO'
        hours_until = (h - t.hour) if h > t.hour else (24 - t.hour + h)
        print(f"\nNext window: Hour {check_h} ({strat}) in {hours_until}h")
        break

mt5.shutdown()
PYEOF
```

### /ml-report
Generate quick ML performance summary.
```bash
"/c/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe" << 'PYEOF'
import json
from collections import defaultdict

trades = []
with open('ml_system/outputs/continuous_trade_log.jsonl', 'r') as f:
    for line in f:
        try:
            trades.append(json.loads(line.strip()))
        except:
            pass

closed = [t for t in trades if isinstance(t.get('outcome'), dict) and t['outcome'].get('status') == 'closed']
wins = [t for t in closed if t['outcome'].get('profit', 0) > 0]
total_pnl = sum(t['outcome'].get('profit', 0) for t in closed)

print(f"Total Trades: {len(closed)}")
print(f"Win Rate: {len(wins)/len(closed)*100:.1f}%" if closed else "N/A")
print(f"Total P/L: ${total_pnl:.2f}")

# By symbol
by_sym = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0})
for t in closed:
    sym = t.get('symbol', 'UNK')
    by_sym[sym]['total'] += 1
    by_sym[sym]['pnl'] += t['outcome'].get('profit', 0)
    if t['outcome'].get('profit', 0) > 0:
        by_sym[sym]['wins'] += 1

print("\nBy Symbol:")
for sym, data in by_sym.items():
    wr = data['wins']/data['total']*100 if data['total'] else 0
    print(f"  {sym}: {data['total']} trades, {wr:.0f}% WR, ${data['pnl']:.2f}")
PYEOF
```

## Key Files
- `trading_bot/config/strategy_config.py` - Trading hours, thresholds
- `trading_bot/strategies/confluence_strategy.py` - Main strategy logic
- `trading_bot/core/mt5_manager.py` - MT5 connection
- `ml_system/outputs/continuous_trade_log.jsonl` - Trade history

## Recent Fixes (Feb 10, 2026)
1. Server time: Fixed to use UTC (was using VPS local time -5h)
2. Position type: Fixed string vs int comparison
3. Pip value: Fixed point * 10 calculation
4. BO exit flow: Added target + stop loss checks
5. Trading hours: Updated to UTC with data-driven optimal hours
