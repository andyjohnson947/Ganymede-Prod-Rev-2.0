"""
SQLite Trade Database

Structured database for all trades, signals, confluences, recoveries, and market conditions.
Thread-safe for use alongside the main trading loop.

DB path: ml_system/outputs/trading.db
"""

import sqlite3
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


class TradeDatabase:
    """Thread-safe SQLite database for comprehensive trade logging."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base, 'outputs', 'trading.db')

        # Ensure directory exists (skip for in-memory DB)
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        print(f"[DB] Trade database initialized: {db_path}")

    def _create_tables(self):
        """Create all tables if they don't exist."""
        with self._lock:
            cursor = self.conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    ticket INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strategy TEXT,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    confluence_score INTEGER,
                    q_trade_value REAL,
                    q_skip_value REAL,
                    exit_time TEXT,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pips REAL,
                    exit_reason TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS confluence_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER REFERENCES trades(ticket),
                    factor_name TEXT NOT NULL,
                    factor_weight INTEGER DEFAULT 1
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER REFERENCES trades(ticket),
                    hour_utc INTEGER,
                    day_of_week INTEGER,
                    session TEXT,
                    adx REAL,
                    plus_di REAL,
                    minus_di REAL,
                    atr_pips REAL,
                    vwap_value REAL,
                    vwap_distance_pct REAL,
                    vwap_direction TEXT,
                    spread_pips REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    direction TEXT,
                    confluence_score INTEGER,
                    factors TEXT,
                    accepted INTEGER,
                    reject_reason TEXT,
                    q_trade_value REAL,
                    q_skip_value REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recovery_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_ticket INTEGER REFERENCES trades(ticket),
                    recovery_type TEXT,
                    recovery_ticket INTEGER,
                    level INTEGER,
                    entry_time TEXT,
                    entry_price REAL,
                    volume REAL,
                    pips_underwater REAL,
                    pnl REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS partial_closes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER REFERENCES trades(ticket),
                    close_time TEXT,
                    volume_closed REAL,
                    close_price REAL,
                    pnl REAL,
                    close_type TEXT
                )
            """)

            # Q-table state persistence
            # Both entry Q-table (per symbol, per state) and exit Q-table stored here.
            # JSON files remain the startup load cache; this is the persistent audit trail.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS q_table_entries (
                    symbol     TEXT NOT NULL,
                    q_type     TEXT NOT NULL,   -- 'entry' or 'exit'
                    state_key  TEXT NOT NULL,
                    q_trade    REAL,            -- entry: Q(TRADE)
                    q_skip     REAL,            -- entry: Q(NO_TRADE)
                    pc1_prob   REAL,            -- exit: P(pc1 hit)
                    action     TEXT,            -- exit: HOLD / CLOSE
                    visits     INTEGER DEFAULT 0,
                    live_updates INTEGER DEFAULT 0,
                    last_updated TEXT,
                    PRIMARY KEY (symbol, q_type, state_key)
                )
            """)

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_factors_ticket ON confluence_factors(ticket)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_recovery_original ON recovery_events(original_ticket)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_qtable_symbol ON q_table_entries(symbol, q_type)")

            self.conn.commit()

    # ── Q-Table Persistence ────────────────────────────────────────

    def upsert_q_state(self, symbol: str, q_type: str, state_key: str,
                       q_trade: float = None, q_skip: float = None,
                       pc1_prob: float = None, action: str = None,
                       visits: int = None, live_updates: int = None):
        """Upsert a single Q-table state into the database.

        Called after every online learning update. Non-blocking: wrapped in
        try/except so a DB error never interrupts the trading loop.
        """
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT INTO q_table_entries
                           (symbol, q_type, state_key, q_trade, q_skip,
                            pc1_prob, action, visits, live_updates, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(symbol, q_type, state_key) DO UPDATE SET
                           q_trade      = COALESCE(excluded.q_trade,      q_trade),
                           q_skip       = COALESCE(excluded.q_skip,       q_skip),
                           pc1_prob     = COALESCE(excluded.pc1_prob,     pc1_prob),
                           action       = COALESCE(excluded.action,       action),
                           visits       = COALESCE(excluded.visits,       visits),
                           live_updates = COALESCE(excluded.live_updates, live_updates),
                           last_updated = excluded.last_updated""",
                    (symbol, q_type, state_key, q_trade, q_skip,
                     pc1_prob, action, visits, live_updates,
                     datetime.utcnow().isoformat())
                )
                self.conn.commit()
            except Exception as e:
                print(f"[DB WARN] upsert_q_state failed: {e}")

    def load_q_table(self, symbol: str, q_type: str) -> List[Dict]:
        """Load all states for a symbol/type from the database.

        Returns list of dicts with state_key + all Q values.
        Returns empty list if no states found.
        """
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT * FROM q_table_entries WHERE symbol=? AND q_type=?",
                    (symbol, q_type)
                )
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            except Exception as e:
                print(f"[DB WARN] load_q_table failed: {e}")
                return []

    def migrate_from_json(self, json_path: str, symbol: str, q_type: str) -> int:
        """One-time bulk migration of a Q-table JSON file into the database.

        Skips states already present (INSERT OR IGNORE).
        Returns number of rows inserted.
        """
        try:
            import json as _json
            from pathlib import Path
            data = _json.loads(Path(json_path).read_text(encoding='utf-8'))
            inserted = 0
            with self._lock:
                if q_type == 'entry':
                    q_values = data.get('q_values', {})
                    visit_counts = data.get('visit_counts', {})
                    now = datetime.utcnow().isoformat()
                    for state_key, qv in q_values.items():
                        try:
                            self.conn.execute(
                                """INSERT OR IGNORE INTO q_table_entries
                                   (symbol, q_type, state_key, q_trade, q_skip, visits, last_updated)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (symbol, q_type, state_key,
                                 qv.get('TRADE', 0.0), qv.get('NO_TRADE', 0.0),
                                 visit_counts.get(state_key, 0), now)
                            )
                            inserted += 1
                        except Exception:
                            pass
                elif q_type == 'exit':
                    states = data.get('states', {})
                    now = datetime.utcnow().isoformat()
                    for state_key, sv in states.items():
                        try:
                            self.conn.execute(
                                """INSERT OR IGNORE INTO q_table_entries
                                   (symbol, q_type, state_key, pc1_prob, action,
                                    visits, live_updates, last_updated)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                (symbol, q_type, state_key,
                                 sv.get('pc1_prob', sv.get('q_hold', 0.5)),
                                 sv.get('action', 'HOLD'),
                                 sv.get('n_total', 0),
                                 sv.get('live_updates', 0), now)
                            )
                            inserted += 1
                        except Exception:
                            pass
                self.conn.commit()
            return inserted
        except Exception as e:
            print(f"[DB WARN] migrate_from_json failed ({json_path}): {e}")
            return 0

    # ── Trade Logging ──────────────────────────────────────────────

    def log_trade(self, ticket: int, symbol: str, direction: str, strategy: str,
                  entry_time: str, entry_price: float, volume: float,
                  confluence_score: int = 0, q_trade: float = None,
                  q_skip: float = None):
        """Log a new trade entry."""
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT OR REPLACE INTO trades
                       (ticket, symbol, direction, strategy, entry_time, entry_price,
                        volume, confluence_score, q_trade_value, q_skip_value, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                    (ticket, symbol, direction, strategy, entry_time, entry_price,
                     volume, confluence_score, q_trade, q_skip)
                )
                self.conn.commit()
            except Exception as e:
                print(f"[DB ERROR] log_trade: {e}")

    def log_factors(self, ticket: int, factors: List[str], weights: Dict[str, int] = None):
        """Log confluence factors for a trade."""
        if weights is None:
            from ml_system.qtable.state_encoder import FACTOR_WEIGHTS
            weights = FACTOR_WEIGHTS

        with self._lock:
            try:
                for factor in factors:
                    weight = weights.get(factor.lower().replace(' ', '_'), 1)
                    self.conn.execute(
                        "INSERT INTO confluence_factors (ticket, factor_name, factor_weight) VALUES (?, ?, ?)",
                        (ticket, factor, weight)
                    )
                self.conn.commit()
            except Exception as e:
                print(f"[DB ERROR] log_factors: {e}")

    def log_market_conditions(self, ticket: int, hour_utc: int = 0, day_of_week: int = 0,
                              session: str = '', adx: float = 0, plus_di: float = 0,
                              minus_di: float = 0, atr_pips: float = 0, vwap_value: float = 0,
                              vwap_distance_pct: float = 0, vwap_direction: str = '',
                              spread_pips: float = 0):
        """Log market conditions at trade entry."""
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT INTO market_conditions
                       (ticket, hour_utc, day_of_week, session, adx, plus_di, minus_di,
                        atr_pips, vwap_value, vwap_distance_pct, vwap_direction, spread_pips)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (ticket, hour_utc, day_of_week, session, adx, plus_di, minus_di,
                     atr_pips, vwap_value, vwap_distance_pct, vwap_direction, spread_pips)
                )
                self.conn.commit()
            except Exception as e:
                print(f"[DB ERROR] log_market_conditions: {e}")

    def update_trade_exit(self, ticket: int, exit_time: str, exit_price: float,
                          pnl: float, pnl_pips: float, exit_reason: str):
        """Update a trade with exit information."""
        with self._lock:
            try:
                self.conn.execute(
                    """UPDATE trades SET exit_time=?, exit_price=?, pnl=?,
                       pnl_pips=?, exit_reason=?, status='closed' WHERE ticket=?""",
                    (exit_time, exit_price, pnl, pnl_pips, exit_reason, ticket)
                )
                self.conn.commit()
            except Exception as e:
                print(f"[DB ERROR] update_trade_exit: {e}")

    # ── Signal Logging ─────────────────────────────────────────────

    def log_signal(self, symbol: str, timestamp: str, direction: str,
                   confluence_score: int, factors: List[str], accepted: bool,
                   reject_reason: str = None, q_trade: float = None,
                   q_skip: float = None):
        """Log every signal (accepted or rejected) for analysis."""
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT INTO signals
                       (symbol, timestamp, direction, confluence_score, factors,
                        accepted, reject_reason, q_trade_value, q_skip_value)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, timestamp, direction, confluence_score,
                     json.dumps(factors), 1 if accepted else 0, reject_reason,
                     q_trade, q_skip)
                )
                self.conn.commit()
            except Exception as e:
                print(f"[DB ERROR] log_signal: {e}")

    # ── Recovery Logging ───────────────────────────────────────────

    def log_recovery(self, original_ticket: int, recovery_type: str,
                     recovery_ticket: int, level: int, entry_time: str,
                     entry_price: float, volume: float, pips_underwater: float):
        """Log a recovery event (DCA/hedge/grid)."""
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT INTO recovery_events
                       (original_ticket, recovery_type, recovery_ticket, level,
                        entry_time, entry_price, volume, pips_underwater)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (original_ticket, recovery_type, recovery_ticket, level,
                     entry_time, entry_price, volume, pips_underwater)
                )
                self.conn.commit()
            except Exception as e:
                print(f"[DB ERROR] log_recovery: {e}")

    # ── Partial Close Logging ──────────────────────────────────────

    def log_partial_close(self, ticket: int, close_time: str, volume_closed: float,
                          close_price: float, pnl: float, close_type: str):
        """Log a partial close event (PC1/PC2/trail)."""
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT INTO partial_closes
                       (ticket, close_time, volume_closed, close_price, pnl, close_type)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ticket, close_time, volume_closed, close_price, pnl, close_type)
                )
                self.conn.commit()
            except Exception as e:
                print(f"[DB ERROR] log_partial_close: {e}")

    # ── Query Methods ──────────────────────────────────────────────

    def get_recent_trades(self, symbol: str = None, days: int = 7) -> List[Dict]:
        """Get recent trades, optionally filtered by symbol."""
        with self._lock:
            cursor = self.conn.cursor()
            if symbol:
                cursor.execute(
                    """SELECT * FROM trades WHERE symbol=?
                       AND entry_time >= datetime('now', ?) ORDER BY entry_time DESC""",
                    (symbol, f'-{days} days')
                )
            else:
                cursor.execute(
                    """SELECT * FROM trades
                       WHERE entry_time >= datetime('now', ?) ORDER BY entry_time DESC""",
                    (f'-{days} days',)
                )
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_factor_performance(self, days: int = 30) -> List[Dict]:
        """Get performance stats per confluence factor."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT cf.factor_name,
                       COUNT(*) as trade_count,
                       SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as wins,
                       AVG(t.pnl) as avg_pnl,
                       SUM(t.pnl) as total_pnl
                FROM confluence_factors cf
                JOIN trades t ON cf.ticket = t.ticket
                WHERE t.status = 'closed'
                  AND t.entry_time >= datetime('now', ?)
                GROUP BY cf.factor_name
                ORDER BY avg_pnl DESC
            """, (f'-{days} days',))
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_signal_stats(self, days: int = 7) -> Dict:
        """Get signal acceptance/rejection stats."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_signals,
                    SUM(accepted) as accepted,
                    COUNT(*) - SUM(accepted) as rejected
                FROM signals
                WHERE timestamp >= datetime('now', ?)
            """, (f'-{days} days',))
            row = cursor.fetchone()
            if row:
                return {'total': row[0], 'accepted': row[1], 'rejected': row[2]}
            return {'total': 0, 'accepted': 0, 'rejected': 0}

    def close(self):
        """Close the database connection."""
        with self._lock:
            self.conn.close()
