"""
MT5 Connection and Order Management
Handles all MetaTrader 5 operations
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import time

from config.strategy_config import (
    MT5_TIMEOUT,
    MT5_MAGIC_NUMBER,
    SYMBOLS,
    HISTORY_BARS
)


class MT5Manager:
    """Manages MT5 connection, data fetching, and order execution"""

    def __init__(self, login: int, password: str, server: str, api_lock=None):
        """
        Initialize MT5 Manager

        Args:
            login: MT5 account login
            password: MT5 account password
            server: MT5 server name
            api_lock: Optional threading.Lock for thread-safe MT5 API access
        """
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.magic_number = MT5_MAGIC_NUMBER
        self.api_lock = api_lock  # Thread-safe access lock

    def connect(self) -> bool:
        """
        Connect to MT5 terminal

        Returns:
            bool: True if connection successful
        """
        if not mt5.initialize():
            print(f"[ERROR] Failed to initialize MT5: {mt5.last_error()}")
            return False

        authorized = mt5.login(self.login, self.password, self.server)

        if not authorized:
            print(f"[ERROR] Failed to login to MT5: {mt5.last_error()}")
            mt5.shutdown()
            return False

        self.connected = True
        account_info = mt5.account_info()
        print(f"[OK] MT5 Connected | #{account_info.login} | ${account_info.balance:.2f} | {account_info.server}")

        return True

    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("[OK] Disconnected from MT5")

    def _with_lock(self, func):
        """Helper to conditionally use lock for thread-safe MT5 API access"""
        if self.api_lock:
            with self.api_lock:
                return func()
        return func()

    def get_account_info(self) -> Optional[Dict]:
        """
        Get current account information

        Returns:
            Dict with account info or None
        """
        if not self.connected:
            return None

        def _get_info():
            info = mt5.account_info()
            if info is None:
                return None

            return {
                'balance': info.balance,
                'equity': info.equity,
                'margin': info.margin,
                'free_margin': info.margin_free,
                'margin_level': info.margin_level if info.margin > 0 else 0,
                'profit': info.profit,
                'currency': info.currency
            }

        return self._with_lock(_get_info)

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 1000,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe string ('H1', 'D1', 'W1', etc.)
            bars: Number of bars to fetch
            start_date: Optional start date

        Returns:
            DataFrame with OHLCV data or None
        """
        if not self.connected:
            print("[ERROR] Not connected to MT5")
            return None

        # Convert timeframe string to MT5 constant
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }

        tf = tf_map.get(timeframe)
        if tf is None:
            print(f"[ERROR] Invalid timeframe: {timeframe}")
            return None

        # Fetch data
        if start_date:
            rates = mt5.copy_rates_from(symbol, tf, start_date, bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            print(f"[ERROR] Failed to fetch data for {symbol} {timeframe}: {mt5.last_error()}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Rename columns for consistency
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        }, inplace=True)

        return df

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions

        Args:
            symbol: Optional symbol to filter positions

        Returns:
            List of position dictionaries
        """
        if not self.connected:
            return []

        def _get_positions():
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                return []

            result = []
            for pos in positions:
                # Only include positions opened by this bot
                if pos.magic != self.magic_number:
                    continue

                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'time': datetime.fromtimestamp(pos.time),
                    'comment': pos.comment
                })

            return result

        return self._with_lock(_get_positions)

    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        order_mode: str = "market",
        expiry=None  # datetime: expiry time for pending orders (ORDER_TIME_SPECIFIED)
    ) -> Optional[int]:
        """
        Place a market or limit order

        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell'
            volume: Lot size
            price: Optional limit price (required for limit orders, optional for market orders)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            order_mode: 'market' for immediate execution, 'limit' for pending order at specific price

        Returns:
            Order ticket number or None if failed
        """
        if not self.connected:
            print("[ERROR] Not connected to MT5")
            return None

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[ERROR] Symbol {symbol} not found")
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"[ERROR] Failed to select symbol {symbol}")
                return None

        # Prepare request
        point = symbol_info.point

        # Determine order type and price based on mode
        if order_mode.lower() == 'limit':
            # LIMIT ORDER: Pending order that executes when price reaches specified level
            if price is None:
                print("[ERROR] Limit orders require a price to be specified")
                return None

            if order_type.lower() == 'buy':
                order_type_mt5 = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                order_type_mt5 = mt5.ORDER_TYPE_SELL_LIMIT

            action = mt5.TRADE_ACTION_PENDING

        elif order_mode.lower() == 'stop':
            # STOP ORDER: Pending order that executes when price breaks THROUGH specified level
            # BUY_STOP: placed ABOVE current price — triggers when price rises to that level
            # SELL_STOP: placed BELOW current price — triggers when price falls to that level
            if price is None:
                print("[ERROR] Stop orders require a price to be specified")
                return None

            if order_type.lower() == 'buy':
                order_type_mt5 = mt5.ORDER_TYPE_BUY_STOP
            else:
                order_type_mt5 = mt5.ORDER_TYPE_SELL_STOP

            action = mt5.TRADE_ACTION_PENDING

        else:
            # MARKET ORDER: Immediate execution at current market price
            if order_type.lower() == 'buy':
                order_type_mt5 = mt5.ORDER_TYPE_BUY

                # FIXED: Add None check before accessing tick attributes
                if price is None:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        print(f"[ERROR] Failed to get tick price for {symbol}")
                        return None
                    price = tick.ask
            else:
                order_type_mt5 = mt5.ORDER_TYPE_SELL

                # FIXED: Add None check before accessing tick attributes
                if price is None:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        print(f"[ERROR] Failed to get tick price for {symbol}")
                        return None
                    price = tick.bid

            action = mt5.TRADE_ACTION_DEAL

        # Determine supported filling mode
        filling_type = self._get_filling_mode(symbol_info)

        # Set time-in-force: use ORDER_TIME_SPECIFIED if expiry provided, else GTC
        if expiry is not None:
            type_time = mt5.ORDER_TIME_SPECIFIED
            expiration_ts = int(expiry.timestamp())
        else:
            type_time = mt5.ORDER_TIME_GTC
            expiration_ts = None

        request = {
            "action": action,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": type_time,
            "type_filling": filling_type,
        }

        if expiration_ts is not None:
            request["expiration"] = expiration_ts
        if sl:
            request["sl"] = sl
        if tp:
            request["tp"] = tp

        # ADDED: Retry logic with exponential backoff for transient errors
        # Retry up to 3 times for network/server issues
        max_retries = 3
        retry_delays = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s

        for attempt in range(max_retries):
            # Send order
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                if attempt < max_retries - 1:
                    print(f"[RETRY] Order send failed (attempt {attempt + 1}/{max_retries}): {error}")
                    print(f"   Retrying in {retry_delays[attempt]}s...")
                    import time
                    time.sleep(retry_delays[attempt])
                    continue
                else:
                    print(f"[ERROR] Order send failed after {max_retries} attempts: {error}")
                    return None

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Success!
                order_mode_str = order_mode.upper() if order_mode.lower() in ('limit', 'stop') else "MARKET"
                print(f"[OK] {order_mode_str} order placed: {order_type.upper()} {volume} {symbol} @ {price:.5f}")
                print(f"   Ticket: {result.order}")
                if attempt > 0:
                    print(f"   (Succeeded on attempt {attempt + 1})")
                return result.order

            # Check if error is retryable
            retryable_codes = [
                mt5.TRADE_RETCODE_REQUOTE,  # Requote
                mt5.TRADE_RETCODE_PRICE_OFF,  # Invalid price
                mt5.TRADE_RETCODE_TIMEOUT,  # Timeout
                mt5.TRADE_RETCODE_CONNECTION,  # Connection error
                mt5.TRADE_RETCODE_PRICE_CHANGED,  # Price changed
                mt5.TRADE_RETCODE_TOO_MANY_REQUESTS,  # Rate limited
                10018,  # TRADE_RETCODE_TRADE_CONTEXT_BUSY — broker processing previous order
            ]

            if result.retcode in retryable_codes and attempt < max_retries - 1:
                print(f"[RETRY] Order failed with retryable error (attempt {attempt + 1}/{max_retries}): {result.comment}")
                print(f"   Retrying in {retry_delays[attempt]}s...")
                import time
                time.sleep(retry_delays[attempt])
                continue
            else:
                # Non-retryable error or max retries reached
                print(f"[ERROR] Order failed: {result.comment}")
                if attempt > 0:
                    print(f"   (Failed after {attempt + 1} attempts)")
                return None

        return None

    def close_position(self, ticket: int, comment: str = "Close by bot") -> bool:
        """
        Close an open position

        Args:
            ticket: Position ticket number
            comment: Close reason comment (e.g. "4HDDTO", "VWAP_EXIT", "TRAIL_STOP")

        Returns:
            bool: True if closed successfully
        """
        if not self.connected:
            print("[ERROR] Not connected to MT5")
            return False

        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            print(f"[ERROR] Position {ticket} not found")
            return False

        position = position[0]

        # Prepare close request
        symbol = position.symbol
        volume = position.volume

        # Get symbol info to determine filling mode
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[ERROR] Symbol {symbol} not found")
            return False

        # Determine supported filling mode
        filling_type = self._get_filling_mode(symbol_info)

        # Opposite order type
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"[ERROR] close_position: symbol_info_tick returned None for {symbol}")
            return False
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)

        if result is None:
            print(f"[ERROR] Close order failed: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[ERROR] Close failed: {result.comment}")
            return False

        print(f"[OK] Position closed: {ticket}")
        return True

    def close_partial_position(self, ticket: int, partial_volume: float, comment: str = None) -> bool:
        """
        Close partial volume of an open position

        Args:
            ticket: Position ticket number
            partial_volume: Volume to close (must be less than position volume)
            comment: Optional custom comment for the partial close (default: auto-generated)

        Returns:
            bool: True if closed successfully
        """
        if not self.connected:
            print("[ERROR] Not connected to MT5")
            return False

        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            print(f"[ERROR] Position {ticket} not found")
            return False

        position = position[0]

        # Validate partial volume
        if partial_volume >= position.volume:
            print(f"[ERROR] Partial volume {partial_volume} must be less than position volume {position.volume}")
            return False

        if partial_volume <= 0:
            print(f"[ERROR] Partial volume must be greater than 0")
            return False

        # Prepare close request
        symbol = position.symbol

        # Get symbol info to determine filling mode and validate volume
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[ERROR] Symbol {symbol} not found")
            return False

        # Round volume to broker step
        volume_step = symbol_info.volume_step
        partial_volume = round(partial_volume / volume_step) * volume_step

        # Check minimum volume
        if partial_volume < symbol_info.volume_min:
            print(f"[ERROR] Partial volume {partial_volume} below minimum {symbol_info.volume_min}")
            return False

        # Determine supported filling mode
        filling_type = self._get_filling_mode(symbol_info)

        # Opposite order type
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"[ERROR] close_partial_position: symbol_info_tick returned None for {symbol}")
            return False
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        # Generate comment (max 31 chars for MT5, Windows encoding only)
        if comment is None:
            comment = f"PC {partial_volume}L"  # Simple default: "PC 0.02L"
        # Truncate to 31 characters if needed
        comment = comment[:31]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": partial_volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)

        if result is None:
            print(f"[ERROR] Partial close order failed: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[ERROR] Partial close failed: {result.comment}")
            return False

        print(f"[OK] Partial close successful: {ticket} - {partial_volume} lots")
        return True

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> bool:
        """
        Modify stop loss or take profit of an open position

        Args:
            ticket: Position ticket number
            sl: New stop loss price (None to keep current)
            tp: New take profit price (None to keep current)

        Returns:
            bool: True if modified successfully
        """
        if not self.connected:
            print("[ERROR] Not connected to MT5")
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            print(f"[ERROR] Position {ticket} not found")
            return False

        position = position[0]

        new_sl = sl if sl is not None else position.sl
        new_tp = tp if tp is not None else position.tp

        # Skip if nothing actually changed (avoids "No changes" spam from MT5)
        if abs(new_sl - position.sl) < 1e-6 and abs(new_tp - position.tp) < 1e-6:
            return True  # Already set, nothing to do

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
        }

        result = mt5.order_send(request)

        if result is None:
            print(f"[ERROR] Modify failed: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[ERROR] Modify failed: {result.comment}")
            return False

        print(f"[OK] Position modified: {ticket}")
        return True

    def _get_filling_mode(self, symbol_info) -> int:
        """
        Determine the best filling mode for the broker

        Args:
            symbol_info: MT5 symbol info object

        Returns:
            int: MT5 filling mode constant
        """
        # Get filling mode flags from symbol
        filling_mode = symbol_info.filling_mode

        # SYMBOL_FILLING_FOK = 1 (bit 0)
        # SYMBOL_FILLING_IOC = 2 (bit 1)
        # SYMBOL_FILLING_RETURN = 4 (bit 2)

        # Check RETURN first (most compatible, used by most brokers)
        if filling_mode & 4:  # Check bit 2
            print(f"Using ORDER_FILLING_RETURN for {symbol_info.name}")
            return mt5.ORDER_FILLING_RETURN

        # Check FOK
        if filling_mode & 1:  # Check bit 0
            print(f"Using ORDER_FILLING_FOK for {symbol_info.name}")
            return mt5.ORDER_FILLING_FOK

        # Check IOC
        if filling_mode & 2:  # Check bit 1
            print(f"Using ORDER_FILLING_IOC for {symbol_info.name}")
            return mt5.ORDER_FILLING_IOC

        # Absolute fallback - try RETURN
        print(f"[WARN] No filling mode detected for {symbol_info.name}, defaulting to RETURN")
        return mt5.ORDER_FILLING_RETURN

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol information

        Args:
            symbol: Trading symbol

        Returns:
            Dict with symbol info or None
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
        }

    def get_symbol_tick(self, symbol: str) -> Optional[Dict]:
        """
        Get current tick data for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Dict with tick data (bid, ask, last, time) or None
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'time': tick.time,
            'volume': tick.volume,
        }

    def get_server_time(self) -> datetime:
        """
        Get current MT5 server time from tick data.

        ICMarkets uses UTC (no offset). The tick.time is a UTC timestamp.
        We use utcfromtimestamp() to get UTC time matching Market Watch.

        Returns:
            datetime: Current MT5 server time (UTC)
        """
        # Try EURUSD first (most liquid, always active)
        for symbol in ['EURUSD', 'GBPUSD'] + SYMBOLS:
            tick = mt5.symbol_info_tick(symbol)
            if tick and tick.time:
                return datetime.utcfromtimestamp(tick.time)

        # Fallback to UTC if no tick data available
        print("[WARN] Could not get MT5 server time from tick data, using UTC")
        return datetime.utcnow()
