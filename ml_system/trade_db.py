#!/usr/bin/env python3
"""
SQLite Trade Database
Mirrors continuous_trade_log.jsonl to SQLite for fast querying.
JSONL remains as fallback. All DB operations are wrapped in try/except
so failures never affect the bot.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set


# Module-level singleton for consumers that don't have a direct reference
_instance = None


def get_trade_db():
    """Get shared TradeDatabase instance. Returns None if DB not available."""
    global _instance
    if _instance is None:
        try:
            db_path = Path(__file__).parent / "outputs" / "trades.db"
            if db_path.exists():
                _instance = TradeDatabase(db_path)
        except Exception:
            pass
    return _instance


class TradeDatabase:
    """SQLite trade database with read and write capabilities"""

    def __init__(self, db_path):
        global _instance
        # If singleton already exists for this path, reuse its connection
        if _instance is not None and str(_instance.db_path) == str(Path(db_path)):
            self.db_path = _instance.db_path
            self.conn = _instance.conn
            self._is_reused = True
            return
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read/write
        self._create_tables()
        self._migrate_schema()
        self._is_reused = False
        # Register as singleton so get_trade_db() reuses this instance
        if _instance is None:
            _instance = self
            print(f"[DB] SQLite mirror ready: {self.db_path.name}")

    def _create_tables(self):
        """Create tables if they don't exist"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT,
                entry_time TEXT,
                entry_price REAL,
                direction TEXT,
                volume REAL,
                strategy_type TEXT,
                breakout_subtype TEXT,
                breakout_confidence TEXT,
                confluence_score INTEGER,
                confluence_factors TEXT,
                hour INTEGER,
                day_of_week TEXT,
                session TEXT,
                logged_at TEXT,
                status TEXT DEFAULT 'open',
                exit_time TEXT,
                exit_price REAL,
                profit REAL,
                hold_hours REAL,
                net_profit REAL,
                had_recovery INTEGER,
                had_grid INTEGER,
                had_partial_close INTEGER,
                dca_count INTEGER,
                hedge_count INTEGER,
                grid_count INTEGER,
                exit_method TEXT,
                vwap_data TEXT,
                volume_profile_data TEXT,
                htf_levels_data TEXT,
                execution_quality_data TEXT,
                outcome_data TEXT,
                updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_type);
            CREATE INDEX IF NOT EXISTS idx_trades_breakout ON trades(breakout_subtype);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_trades_profit ON trades(profit);
        """)
        self.conn.commit()

    def _migrate_schema(self):
        """Add columns that were missing in Phase 1 (safe - ALTER TABLE ADD is idempotent-ish)"""
        new_columns = [
            ("trend_filter_data TEXT", "trend_filter_data"),
            ("adx REAL", "adx"),
            ("atr_pips REAL", "atr_pips"),
            ("spread_at_entry_pips REAL", "spread_at_entry_pips"),
            ("fair_value_gaps_data TEXT", "fair_value_gaps_data"),
            ("volatility_data TEXT", "volatility_data"),
            ("entry_quality_data TEXT", "entry_quality_data"),
            ("trade_sequencing_data TEXT", "trade_sequencing_data"),
            ("market_microstructure_data TEXT", "market_microstructure_data"),
            ("position_sizing_data TEXT", "position_sizing_data"),
        ]
        for col_def, col_name in new_columns:
            try:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        self.conn.commit()

    def insert_trade(self, record: Dict[str, Any]):
        """Insert a new trade record (mirrors JSONL append)"""
        market_ctx = record.get('market_context', {})

        self.conn.execute("""
            INSERT OR IGNORE INTO trades (
                ticket, symbol, entry_time, entry_price, direction, volume,
                strategy_type, breakout_subtype, breakout_confidence,
                confluence_score, confluence_factors,
                hour, day_of_week, session, logged_at, status,
                vwap_data, volume_profile_data, htf_levels_data,
                execution_quality_data,
                trend_filter_data, adx, atr_pips, spread_at_entry_pips,
                fair_value_gaps_data, volatility_data, entry_quality_data,
                trade_sequencing_data, market_microstructure_data, position_sizing_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get('ticket'),
            record.get('symbol'),
            record.get('entry_time'),
            record.get('entry_price'),
            record.get('direction'),
            record.get('volume'),
            record.get('strategy_type'),
            record.get('breakout_subtype'),
            record.get('breakout_confidence'),
            record.get('confluence_score'),
            json.dumps(record.get('confluence_factors', [])),
            market_ctx.get('hour'),
            market_ctx.get('day_of_week'),
            market_ctx.get('session'),
            record.get('logged_at'),
            json.dumps(record.get('vwap', {})),
            json.dumps(record.get('volume_profile', {})),
            json.dumps(record.get('htf_levels', {})),
            json.dumps(record.get('execution_quality', {})),
            json.dumps(record.get('trend_filter', {})),
            record.get('trend_filter', {}).get('adx') or record.get('adx'),
            record.get('volatility', {}).get('atr_pips') or record.get('atr_pips'),
            record.get('execution_quality', {}).get('spread_at_entry_pips') or record.get('spread_at_entry_pips'),
            json.dumps(record.get('fair_value_gaps', {})),
            json.dumps(record.get('volatility', {})),
            json.dumps(record.get('entry_quality', {})),
            json.dumps(record.get('trade_sequencing', {})),
            json.dumps(record.get('market_microstructure', {})),
            json.dumps(record.get('position_sizing', {})),
        ))
        self.conn.commit()

    def update_outcome(self, ticket: int, outcome: Dict[str, Any]):
        """Update a trade with outcome data (mirrors JSONL rewrite)"""
        recovery = outcome.get('recovery', {})
        exit_strategy = outcome.get('exit_strategy', {})

        self.conn.execute("""
            UPDATE trades SET
                status = ?,
                exit_time = ?,
                exit_price = ?,
                profit = ?,
                hold_hours = ?,
                net_profit = ?,
                had_recovery = ?,
                had_grid = ?,
                had_partial_close = ?,
                dca_count = ?,
                hedge_count = ?,
                grid_count = ?,
                exit_method = ?,
                outcome_data = ?,
                updated_at = ?
            WHERE ticket = ?
        """, (
            outcome.get('status', 'closed'),
            outcome.get('exit_time'),
            outcome.get('exit_price'),
            outcome.get('profit'),
            outcome.get('hold_hours'),
            outcome.get('net_profit'),
            int(outcome.get('had_recovery', False)),
            int(outcome.get('had_grid', False)),
            int(outcome.get('had_partial_close', False)),
            recovery.get('dca_count', 0),
            recovery.get('hedge_count', 0),
            recovery.get('grid_count', 0),
            exit_strategy.get('exit_method'),
            json.dumps(outcome),
            datetime.now().isoformat(),
            ticket,
        ))
        self.conn.commit()

    def migrate_from_jsonl(self, jsonl_path: str):
        """One-time import of existing JSONL data. Idempotent (INSERT OR IGNORE)."""
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            print("[DB] No JSONL file to migrate")
            return

        count = 0
        skipped = 0

        with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    record = json.loads(line)
                except:
                    continue

                ticket = record.get('ticket')
                if not ticket:
                    continue

                # Check if already in DB
                cursor = self.conn.execute("SELECT 1 FROM trades WHERE ticket = ?", (ticket,))
                if cursor.fetchone():
                    skipped += 1
                    continue

                # Insert the trade
                market_ctx = record.get('market_context', {})
                outcome = record.get('outcome', {})
                recovery = outcome.get('recovery', {})
                exit_strategy = outcome.get('exit_strategy', {})

                self.conn.execute("""
                    INSERT OR IGNORE INTO trades (
                        ticket, symbol, entry_time, entry_price, direction, volume,
                        strategy_type, breakout_subtype, breakout_confidence,
                        confluence_score, confluence_factors,
                        hour, day_of_week, session, logged_at,
                        status, exit_time, exit_price, profit, hold_hours, net_profit,
                        had_recovery, had_grid, had_partial_close,
                        dca_count, hedge_count, grid_count, exit_method,
                        vwap_data, volume_profile_data, htf_levels_data,
                        execution_quality_data, outcome_data, updated_at,
                        trend_filter_data, adx, atr_pips, spread_at_entry_pips,
                        fair_value_gaps_data, volatility_data, entry_quality_data,
                        trade_sequencing_data, market_microstructure_data, position_sizing_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticket,
                    record.get('symbol'),
                    record.get('entry_time'),
                    record.get('entry_price'),
                    record.get('direction'),
                    record.get('volume'),
                    record.get('strategy_type'),
                    record.get('breakout_subtype'),
                    record.get('breakout_confidence'),
                    record.get('confluence_score'),
                    json.dumps(record.get('confluence_factors', [])),
                    market_ctx.get('hour'),
                    market_ctx.get('day_of_week'),
                    market_ctx.get('session'),
                    record.get('logged_at'),
                    outcome.get('status', 'open'),
                    outcome.get('exit_time'),
                    outcome.get('exit_price'),
                    outcome.get('profit'),
                    outcome.get('hold_hours'),
                    outcome.get('net_profit'),
                    int(outcome.get('had_recovery', False)) if outcome else None,
                    int(outcome.get('had_grid', False)) if outcome else None,
                    int(outcome.get('had_partial_close', False)) if outcome else None,
                    recovery.get('dca_count', 0) if recovery else None,
                    recovery.get('hedge_count', 0) if recovery else None,
                    recovery.get('grid_count', 0) if recovery else None,
                    exit_strategy.get('exit_method') if exit_strategy else None,
                    json.dumps(record.get('vwap', {})),
                    json.dumps(record.get('volume_profile', {})),
                    json.dumps(record.get('htf_levels', {})),
                    json.dumps(record.get('execution_quality', {})),
                    json.dumps(outcome) if outcome else None,
                    outcome.get('updated_at') if outcome else None,
                    json.dumps(record.get('trend_filter', {})),
                    record.get('trend_filter', {}).get('adx') or record.get('adx'),
                    record.get('volatility', {}).get('atr_pips') or record.get('atr_pips'),
                    record.get('execution_quality', {}).get('spread_at_entry_pips') or record.get('spread_at_entry_pips'),
                    json.dumps(record.get('fair_value_gaps', {})),
                    json.dumps(record.get('volatility', {})),
                    json.dumps(record.get('entry_quality', {})),
                    json.dumps(record.get('trade_sequencing', {})),
                    json.dumps(record.get('market_microstructure', {})),
                    json.dumps(record.get('position_sizing', {})),
                ))
                count += 1

        self.conn.commit()
        total = count + skipped
        if count > 0:
            print(f"[DB] Migrated {count} new trades ({total} total)")

        # Backfill new columns for existing records that have NULLs
        self._backfill_new_columns(jsonl_file)

    def _backfill_new_columns(self, jsonl_file: Path):
        """Update existing DB records with new columns from JSONL data"""
        # Check if backfill is needed (any of the data columns NULL)
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE trend_filter_data IS NULL OR trade_sequencing_data IS NULL"
        )
        null_count = cursor.fetchone()[0]
        if null_count == 0:
            return

        updated = 0
        with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    record = json.loads(line)
                except:
                    continue
                ticket = record.get('ticket')
                if not ticket:
                    continue

                trend_filter = record.get('trend_filter', {})
                volatility = record.get('volatility', {})
                exec_quality = record.get('execution_quality', {})

                self.conn.execute("""
                    UPDATE trades SET
                        trend_filter_data = COALESCE(trend_filter_data, ?),
                        adx = COALESCE(adx, ?),
                        atr_pips = COALESCE(atr_pips, ?),
                        spread_at_entry_pips = COALESCE(spread_at_entry_pips, ?),
                        fair_value_gaps_data = COALESCE(fair_value_gaps_data, ?),
                        volatility_data = COALESCE(volatility_data, ?),
                        entry_quality_data = COALESCE(entry_quality_data, ?),
                        trade_sequencing_data = COALESCE(trade_sequencing_data, ?),
                        market_microstructure_data = COALESCE(market_microstructure_data, ?),
                        position_sizing_data = COALESCE(position_sizing_data, ?)
                    WHERE ticket = ? AND (trend_filter_data IS NULL OR trade_sequencing_data IS NULL)
                """, (
                    json.dumps(trend_filter),
                    trend_filter.get('adx') or record.get('adx'),
                    volatility.get('atr_pips') or record.get('atr_pips'),
                    exec_quality.get('spread_at_entry_pips') or record.get('spread_at_entry_pips'),
                    json.dumps(record.get('fair_value_gaps', {})),
                    json.dumps(volatility),
                    json.dumps(record.get('entry_quality', {})),
                    json.dumps(record.get('trade_sequencing', {})),
                    json.dumps(record.get('market_microstructure', {})),
                    json.dumps(record.get('position_sizing', {})),
                    ticket,
                ))
                updated += 1

        self.conn.commit()
        if updated > 0:
            print(f"[DB] Backfilled columns for {updated} records")

    # =====================================================================
    # READ METHODS - Used by consumers (auto-tuner, reports, ML system)
    # =====================================================================

    def _row_to_trade_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a DB row back to the nested dict format consumers expect"""
        trade = {
            'ticket': row['ticket'],
            'symbol': row['symbol'],
            'entry_time': row['entry_time'],
            'entry_price': row['entry_price'],
            'direction': row['direction'],
            'volume': row['volume'],
            'strategy_type': row['strategy_type'],
            'breakout_subtype': row['breakout_subtype'],
            'breakout_confidence': row['breakout_confidence'],
            'confluence_score': row['confluence_score'],
            'confluence_factors': json.loads(row['confluence_factors']) if row['confluence_factors'] else [],
            'market_context': {
                'hour': row['hour'],
                'day_of_week': row['day_of_week'],
                'session': row['session'],
            },
            'logged_at': row['logged_at'],
            'vwap': json.loads(row['vwap_data']) if row['vwap_data'] else {},
            'volume_profile': json.loads(row['volume_profile_data']) if row['volume_profile_data'] else {},
            'htf_levels': json.loads(row['htf_levels_data']) if row['htf_levels_data'] else {},
            'execution_quality': json.loads(row['execution_quality_data']) if row['execution_quality_data'] else {},
            'trend_filter': json.loads(row['trend_filter_data']) if row['trend_filter_data'] else {},
            'adx': row['adx'],
            'atr_pips': row['atr_pips'],
            'spread_at_entry_pips': row['spread_at_entry_pips'],
            'fair_value_gaps': json.loads(row['fair_value_gaps_data']) if row['fair_value_gaps_data'] else {},
            'volatility': json.loads(row['volatility_data']) if row['volatility_data'] else {},
            'entry_quality': json.loads(row['entry_quality_data']) if row['entry_quality_data'] else {},
            'trade_sequencing': json.loads(row['trade_sequencing_data']) if row['trade_sequencing_data'] else {},
            'market_microstructure': json.loads(row['market_microstructure_data']) if row['market_microstructure_data'] else {},
            'position_sizing': json.loads(row['position_sizing_data']) if row['position_sizing_data'] else {},
        }

        # Add outcome if trade is closed
        if row['outcome_data']:
            try:
                trade['outcome'] = json.loads(row['outcome_data'])
            except:
                pass
        elif row['status'] == 'closed':
            # Build outcome from columns if JSON blob missing
            trade['outcome'] = {
                'status': 'closed',
                'exit_time': row['exit_time'],
                'exit_price': row['exit_price'],
                'profit': row['profit'],
                'hold_hours': row['hold_hours'],
                'net_profit': row['net_profit'],
                'had_recovery': bool(row['had_recovery']),
                'had_grid': bool(row['had_grid']),
                'had_partial_close': bool(row['had_partial_close']),
                'recovery': {
                    'dca_count': row['dca_count'] or 0,
                    'hedge_count': row['hedge_count'] or 0,
                    'grid_count': row['grid_count'] or 0,
                },
                'exit_strategy': {
                    'exit_method': row['exit_method'],
                },
            }

        return trade

    def get_all_tickets(self) -> Set[int]:
        """Get set of all ticket IDs in the database"""
        self.conn.row_factory = None
        cursor = self.conn.execute("SELECT ticket FROM trades")
        tickets = {row[0] for row in cursor}
        self.conn.row_factory = sqlite3.Row
        return tickets

    def get_closed_trades(self) -> List[Dict]:
        """Get all closed trades with full data (replaces _load_closed_trades)"""
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute(
            "SELECT * FROM trades WHERE status = 'closed' ORDER BY entry_time"
        )
        return [self._row_to_trade_dict(row) for row in cursor]

    def get_trades_since(self, days: int = 365) -> List[Dict]:
        """Get all trades from the last N days (replaces get_trades/get_recent_trades)"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute(
            "SELECT * FROM trades WHERE entry_time >= ? ORDER BY entry_time",
            (cutoff,)
        )
        return [self._row_to_trade_dict(row) for row in cursor]

    def get_trade_counts(self) -> Dict[str, int]:
        """Get trade counts (replaces counting loops)"""
        self.conn.row_factory = None
        total = self.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        closed = self.conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'").fetchone()[0]
        self.conn.row_factory = sqlite3.Row
        return {
            'total': total,
            'closed': closed,
            'open': total - closed,
        }

    def get_all_trades(self) -> List[Dict]:
        """Get all trades (replaces full-log reads)"""
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute("SELECT * FROM trades ORDER BY entry_time")
        return [self._row_to_trade_dict(row) for row in cursor]

    def close(self):
        """Clean shutdown"""
        if self.conn:
            self.conn.close()
