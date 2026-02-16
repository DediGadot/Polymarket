"""
SQLite storage for pipeline telemetry. WAL mode for concurrent read/write.

Single file at report/data/report.db. All writes go through ReportStore;
the server reads the same file for API queries.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "report.db"


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


class ReportStore:
    """Thread-safe SQLite store for pipeline telemetry."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or DB_PATH)
        self._local = threading.local()
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = _dict_factory
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn
        conn.executescript(_SCHEMA)
        self._migrate(conn)
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply incremental migrations to existing tables."""
        trade_cols = {row["name"] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
        if "simulated" not in trade_cols:
            conn.execute("ALTER TABLE trades ADD COLUMN simulated INTEGER NOT NULL DEFAULT 0")

        opp_cols = {row["name"] for row in conn.execute("PRAGMA table_info(opportunities)").fetchall()}
        if "event_title" not in opp_cols:
            conn.execute("ALTER TABLE opportunities ADD COLUMN event_title TEXT NOT NULL DEFAULT ''")

    def begin_transaction(self) -> None:
        """Begin a write transaction if one is not already active."""
        if not self._conn.in_transaction:
            self._conn.execute("BEGIN")

    def commit(self) -> None:
        """Commit the current transaction if active."""
        if self._conn.in_transaction:
            self._conn.commit()

    def rollback(self) -> None:
        """Rollback the current transaction if active."""
        if self._conn.in_transaction:
            self._conn.rollback()

    # ── Write methods ──

    def start_session(
        self,
        mode: str,
        config_json: str = "{}",
        cli_args_json: str = "{}",
        *,
        commit: bool = True,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO sessions (start_ts, mode, config_json, cli_args_json) VALUES (?, ?, ?, ?)",
            (time.time(), mode, config_json, cli_args_json),
        )
        if commit:
            self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def end_session(self, session_id: int, *, commit: bool = True) -> None:
        self._conn.execute(
            "UPDATE sessions SET end_ts = ? WHERE id = ?",
            (time.time(), session_id),
        )
        if commit:
            self._conn.commit()

    def insert_cycle(
        self,
        session_id: int,
        cycle_num: int,
        elapsed_sec: float,
        strategy_mode: str,
        gas_price_gwei: float,
        spike_count: int,
        has_momentum: bool,
        win_rate: float,
        markets_scanned: int,
        binary_count: int,
        negrisk_events: int,
        negrisk_markets: int,
        opps_found: int,
        opps_executed: int,
        funnel: dict[str, int] | None = None,
        scanner_counts: dict[str, int] | None = None,
        strategy_snapshot: dict[str, Any] | None = None,
        *,
        commit: bool = True,
    ) -> int:
        """Insert a cycle and its related funnel/scanner/strategy data atomically."""
        conn = self._conn
        now = time.time()
        cur = conn.execute(
            """INSERT OR REPLACE INTO cycles
               (session_id, cycle_num, timestamp, elapsed_sec,
                strategy_mode, gas_price_gwei, spike_count, has_momentum,
                win_rate, markets_scanned, binary_count, negrisk_events,
                negrisk_markets, opps_found, opps_executed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, cycle_num, now, elapsed_sec,
                strategy_mode, gas_price_gwei, spike_count, int(has_momentum),
                win_rate, markets_scanned, binary_count, negrisk_events,
                negrisk_markets, opps_found, opps_executed,
            ),
        )
        cycle_id = cur.lastrowid

        if funnel:
            conn.executemany(
                "INSERT OR REPLACE INTO funnel (cycle_id, stage, count) VALUES (?, ?, ?)",
                [(cycle_id, stage, count) for stage, count in funnel.items()],
            )

        if scanner_counts:
            conn.executemany(
                "INSERT OR REPLACE INTO scanner_counts (cycle_id, scanner, count) VALUES (?, ?, ?)",
                [(cycle_id, scanner, count) for scanner, count in scanner_counts.items()],
            )

        if strategy_snapshot:
            conn.execute(
                """INSERT INTO strategy_transitions
                   (cycle_id, mode, gas_price_gwei, active_spike_count,
                    has_crypto_momentum, recent_win_rate, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    cycle_id,
                    strategy_snapshot.get("mode", ""),
                    strategy_snapshot.get("gas_price_gwei", 0.0),
                    strategy_snapshot.get("active_spike_count", 0),
                    int(strategy_snapshot.get("has_crypto_momentum", False)),
                    strategy_snapshot.get("recent_win_rate", 0.0),
                    now,
                ),
            )

        if commit:
            conn.commit()
        return cycle_id  # type: ignore[return-value]

    def insert_opportunities(
        self,
        cycle_id: int,
        opps: list[dict[str, Any]],
        *,
        commit: bool = True,
    ) -> None:
        if not opps:
            return
        self._conn.executemany(
            """INSERT INTO opportunities
               (cycle_id, event_id, opp_type, net_profit, roi_pct,
                required_capital, n_legs, is_buy_arb, platform,
                total_score, profit_score, fill_score, efficiency_score,
                urgency_score, competition_score, persistence_score,
                book_depth_ratio, confidence, market_volume,
                time_to_resolution_hrs, legs_json, event_title, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    cycle_id,
                    o["event_id"], o["opp_type"], o["net_profit"], o["roi_pct"],
                    o["required_capital"], o["n_legs"], int(o["is_buy_arb"]),
                    o.get("platform", "polymarket"),
                    o["total_score"], o["profit_score"], o["fill_score"],
                    o["efficiency_score"], o["urgency_score"],
                    o["competition_score"], o["persistence_score"],
                    o["book_depth_ratio"], o["confidence"],
                    o["market_volume"], o["time_to_resolution_hrs"],
                    o.get("legs_json", "[]"), o.get("event_title", ""),
                    o["timestamp"],
                )
                for o in opps
            ],
        )
        if commit:
            self._conn.commit()

    def insert_safety_rejections(
        self,
        cycle_id: int,
        rejections: list[dict[str, Any]],
        *,
        commit: bool = True,
    ) -> None:
        if not rejections:
            return
        self._conn.executemany(
            """INSERT INTO safety_rejections
               (cycle_id, event_id, opp_type, check_name, reason,
                net_profit, roi_pct, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    cycle_id,
                    r["event_id"], r["opp_type"], r["check_name"], r["reason"],
                    r["net_profit"], r["roi_pct"], r["timestamp"],
                )
                for r in rejections
            ],
        )
        if commit:
            self._conn.commit()

    def insert_cross_platform_matches(
        self,
        cycle_id: int,
        matches: list[dict[str, Any]],
        *,
        commit: bool = True,
    ) -> None:
        if not matches:
            return
        self._conn.executemany(
            """INSERT INTO cross_platform_matches
               (cycle_id, pm_event_id, pm_title, platform, ext_event_ticker,
                confidence, match_method, pm_best_ask, ext_best_ask,
                price_diff, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    cycle_id,
                    m["pm_event_id"], m["pm_title"], m["platform"],
                    m["ext_event_ticker"], m["confidence"], m["match_method"],
                    m["pm_best_ask"], m["ext_best_ask"], m["price_diff"],
                    m["timestamp"],
                )
                for m in matches
            ],
        )
        if commit:
            self._conn.commit()

    def insert_trade(
        self,
        session_id: int,
        cycle_id: int,
        event_id: str,
        opp_type: str,
        n_legs: int,
        fully_filled: bool,
        fill_prices_json: str,
        fill_sizes_json: str,
        fees: float,
        gas_cost: float,
        net_pnl: float,
        execution_time_ms: float,
        total_score: float,
        simulated: bool = False,
        *,
        commit: bool = True,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO trades
               (session_id, cycle_id, event_id, opp_type, n_legs,
                fully_filled, fill_prices_json, fill_sizes_json,
                fees, gas_cost, net_pnl, execution_time_ms,
                total_score, simulated, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, cycle_id, event_id, opp_type, n_legs,
                int(fully_filled), fill_prices_json, fill_sizes_json,
                fees, gas_cost, net_pnl, execution_time_ms,
                total_score, int(simulated), time.time(),
            ),
        )
        if commit:
            self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def insert_trades(
        self,
        trades: list[dict[str, Any]],
        *,
        commit: bool = True,
    ) -> int:
        """Bulk insert trades. Returns number of inserted rows."""
        if not trades:
            return 0
        self._conn.executemany(
            """INSERT INTO trades
               (session_id, cycle_id, event_id, opp_type, n_legs,
                fully_filled, fill_prices_json, fill_sizes_json,
                fees, gas_cost, net_pnl, execution_time_ms,
                total_score, simulated, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    t["session_id"],
                    t["cycle_id"],
                    t["event_id"],
                    t["opp_type"],
                    t["n_legs"],
                    int(t["fully_filled"]),
                    t["fill_prices_json"],
                    t["fill_sizes_json"],
                    t["fees"],
                    t["gas_cost"],
                    t["net_pnl"],
                    t["execution_time_ms"],
                    t["total_score"],
                    int(t.get("simulated", False)),
                    t.get("timestamp", time.time()),
                )
                for t in trades
            ],
        )
        if commit:
            self._conn.commit()
        return len(trades)

    # ── Read methods (for API) ──

    def get_sessions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT s.*,
                      (SELECT COUNT(*) FROM cycles c WHERE c.session_id = s.id) AS cycle_count,
                      (SELECT COUNT(*) FROM trades t WHERE t.session_id = s.id) AS trade_count,
                      (SELECT COALESCE(SUM(t.net_pnl), 0) FROM trades t WHERE t.session_id = s.id) AS total_pnl
               FROM sessions s ORDER BY s.start_ts DESC"""
        ).fetchall()
        return rows

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        return self._conn.execute(
            """SELECT s.*,
                      (SELECT COUNT(*) FROM cycles c WHERE c.session_id = s.id) AS cycle_count,
                      (SELECT COUNT(*) FROM trades t WHERE t.session_id = s.id) AS trade_count,
                      (SELECT COALESCE(SUM(t.net_pnl), 0) FROM trades t WHERE t.session_id = s.id) AS total_pnl
               FROM sessions s WHERE s.id = ?""",
            (session_id,),
        ).fetchone()

    def get_cycles(
        self, session_id: int, limit: int = 500, offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self._conn.execute(
            "SELECT * FROM cycles WHERE session_id = ? ORDER BY cycle_num DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        ).fetchall()

    def get_latest_cycle(self, session_id: int) -> dict[str, Any] | None:
        return self._conn.execute(
            "SELECT * FROM cycles WHERE session_id = ? ORDER BY cycle_num DESC LIMIT 1",
            (session_id,),
        ).fetchone()

    def get_funnel(self, cycle_id: int) -> list[dict[str, Any]]:
        return self._conn.execute(
            "SELECT stage, count FROM funnel WHERE cycle_id = ? ORDER BY id",
            (cycle_id,),
        ).fetchall()

    def get_funnel_aggregated(self, session_id: int) -> list[dict[str, Any]]:
        """Aggregate funnel across all cycles in a session."""
        return self._conn.execute(
            """SELECT f.stage, SUM(f.count) AS count, AVG(f.count) AS avg_count
               FROM funnel f
               JOIN cycles c ON f.cycle_id = c.id
               WHERE c.session_id = ?
               GROUP BY f.stage
               ORDER BY SUM(f.count) DESC""",
            (session_id,),
        ).fetchall()

    def get_opportunities(
        self,
        session_id: int,
        opp_type: str | None = None,
        min_score: float = 0.0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = """SELECT o.* FROM opportunities o
                   JOIN cycles c ON o.cycle_id = c.id
                   WHERE c.session_id = ?"""
        params: list[Any] = [session_id]
        if opp_type:
            query += " AND o.opp_type = ?"
            params.append(opp_type)
        if min_score > 0:
            query += " AND o.total_score >= ?"
            params.append(min_score)
        query += " ORDER BY o.total_score DESC LIMIT ?"
        params.append(limit)
        return self._conn.execute(query, params).fetchall()

    def get_unique_actionable(
        self,
        session_id: int,
        *,
        limit: int = 50,
        min_score: float = 0.0,
        min_fill_score: float = 0.50,
        min_persistence: float = 0.50,
        max_candidates: int = 5000,
    ) -> list[dict[str, Any]]:
        """
        Return top-N unique actionable opportunities for a session.

        Actionable proxy (reporting-only):
        - buy-side opportunities
        - non-maker strategy rows
        - positive score/fill/persistence thresholds

        Uniqueness is keyed by a stable fingerprint over
        (opp_type, event_id, sorted[(token_id, side)]).
        """
        rows = self._conn.execute(
            """SELECT o.* FROM opportunities o
               JOIN cycles c ON o.cycle_id = c.id
               WHERE c.session_id = ?
                 AND o.is_buy_arb = 1
                 AND o.opp_type != 'maker_rebalance'
                 AND o.total_score >= ?
                 AND o.fill_score >= ?
                 AND o.persistence_score >= ?
               ORDER BY o.total_score DESC, o.net_profit DESC
               LIMIT ?""",
            (session_id, min_score, min_fill_score, min_persistence, max_candidates),
        ).fetchall()

        best_by_signature: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_dict = dict(row)
            signature_hash = self._opportunity_signature_hash(row_dict)
            existing = best_by_signature.get(signature_hash)
            if existing is None:
                row_dict["signature_hash"] = signature_hash
                row_dict["duplicate_count"] = 1
                best_by_signature[signature_hash] = row_dict
            else:
                existing["duplicate_count"] += 1

        unique_rows = list(best_by_signature.values())
        unique_rows.sort(key=lambda r: (r["total_score"], r["net_profit"]), reverse=True)
        return unique_rows[:limit]

    @staticmethod
    def _opportunity_signature_hash(row: dict[str, Any]) -> str:
        """Build a stable fingerprint hash for opportunity deduplication."""
        legs_raw = row.get("legs_json", "[]")
        try:
            legs = json.loads(legs_raw)
        except (TypeError, json.JSONDecodeError):
            legs = []

        norm_legs = sorted(
            (str(leg.get("token_id", "")), str(leg.get("side", "")))
            for leg in legs
            if isinstance(leg, dict)
        )
        payload = json.dumps(
            [str(row.get("opp_type", "")), str(row.get("event_id", "")), norm_legs],
            separators=(",", ":"),
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get_safety_rejections(
        self, session_id: int, limit: int = 200,
    ) -> list[dict[str, Any]]:
        return self._conn.execute(
            """SELECT sr.* FROM safety_rejections sr
               JOIN cycles c ON sr.cycle_id = c.id
               WHERE c.session_id = ?
               ORDER BY sr.timestamp DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()

    def get_strategy_timeline(self, session_id: int) -> list[dict[str, Any]]:
        return self._conn.execute(
            """SELECT st.* FROM strategy_transitions st
               JOIN cycles c ON st.cycle_id = c.id
               WHERE c.session_id = ?
               ORDER BY st.timestamp""",
            (session_id,),
        ).fetchall()

    def get_cross_platform_matches(self, session_id: int) -> list[dict[str, Any]]:
        return self._conn.execute(
            """SELECT cpm.* FROM cross_platform_matches cpm
               JOIN cycles c ON cpm.cycle_id = c.id
               WHERE c.session_id = ?
               ORDER BY cpm.price_diff DESC""",
            (session_id,),
        ).fetchall()

    def get_trades(self, session_id: int) -> list[dict[str, Any]]:
        return self._conn.execute(
            "SELECT * FROM trades WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

    def get_pnl_series(self, session_id: int) -> list[dict[str, Any]]:
        """Cumulative P&L time series."""
        rows = self._conn.execute(
            """SELECT timestamp, net_pnl,
                      SUM(net_pnl) OVER (ORDER BY timestamp) AS cumulative_pnl
               FROM trades WHERE session_id = ?
               ORDER BY timestamp""",
            (session_id,),
        ).fetchall()
        return rows

    def get_scanner_distribution(self, session_id: int) -> list[dict[str, Any]]:
        return self._conn.execute(
            """SELECT sc.scanner, SUM(sc.count) AS total_count
               FROM scanner_counts sc
               JOIN cycles c ON sc.cycle_id = c.id
               WHERE c.session_id = ?
               GROUP BY sc.scanner
               ORDER BY total_count DESC""",
            (session_id,),
        ).fetchall()

    def get_top_opportunities(
        self, session_id: int, n: int = 20,
    ) -> list[dict[str, Any]]:
        return self._conn.execute(
            """SELECT o.* FROM opportunities o
               JOIN cycles c ON o.cycle_id = c.id
               WHERE c.session_id = ?
               ORDER BY o.total_score DESC LIMIT ?""",
            (session_id, n),
        ).fetchall()

    def get_untapped(
        self, session_id: int, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Opportunities that scored well but were rejected by safety."""
        return self._conn.execute(
            """SELECT sr.event_id, sr.opp_type, sr.check_name, sr.reason,
                      sr.net_profit, sr.roi_pct,
                      o.total_score
               FROM safety_rejections sr
               JOIN cycles c ON sr.cycle_id = c.id
               LEFT JOIN opportunities o
                   ON o.cycle_id = sr.cycle_id AND o.event_id = sr.event_id
               WHERE c.session_id = ? AND (o.total_score IS NULL OR o.total_score > 0.3)
               ORDER BY sr.net_profit DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()

    def get_safety_pass_rates(self, session_id: int) -> list[dict[str, Any]]:
        """Pass/fail rates per safety check."""
        return self._conn.execute(
            """SELECT check_name, COUNT(*) AS fail_count
               FROM safety_rejections sr
               JOIN cycles c ON sr.cycle_id = c.id
               WHERE c.session_id = ?
               GROUP BY check_name
               ORDER BY fail_count DESC""",
            (session_id,),
        ).fetchall()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts REAL NOT NULL,
    end_ts REAL,
    mode TEXT NOT NULL,
    config_json TEXT DEFAULT '{}',
    cli_args_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    cycle_num INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    elapsed_sec REAL NOT NULL,
    strategy_mode TEXT NOT NULL,
    gas_price_gwei REAL NOT NULL,
    spike_count INTEGER NOT NULL DEFAULT 0,
    has_momentum INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    markets_scanned INTEGER NOT NULL DEFAULT 0,
    binary_count INTEGER NOT NULL DEFAULT 0,
    negrisk_events INTEGER NOT NULL DEFAULT 0,
    negrisk_markets INTEGER NOT NULL DEFAULT 0,
    opps_found INTEGER NOT NULL DEFAULT 0,
    opps_executed INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_id, cycle_num)
);

CREATE TABLE IF NOT EXISTS funnel (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL REFERENCES cycles(id),
    stage TEXT NOT NULL,
    count INTEGER NOT NULL,
    UNIQUE(cycle_id, stage)
);

CREATE TABLE IF NOT EXISTS scanner_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL REFERENCES cycles(id),
    scanner TEXT NOT NULL,
    count INTEGER NOT NULL,
    UNIQUE(cycle_id, scanner)
);

CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL REFERENCES cycles(id),
    event_id TEXT NOT NULL,
    opp_type TEXT NOT NULL,
    net_profit REAL NOT NULL,
    roi_pct REAL NOT NULL,
    required_capital REAL NOT NULL,
    n_legs INTEGER NOT NULL,
    is_buy_arb INTEGER NOT NULL DEFAULT 0,
    platform TEXT NOT NULL DEFAULT 'polymarket',
    total_score REAL NOT NULL,
    profit_score REAL NOT NULL,
    fill_score REAL NOT NULL,
    efficiency_score REAL NOT NULL,
    urgency_score REAL NOT NULL,
    competition_score REAL NOT NULL,
    persistence_score REAL NOT NULL DEFAULT 0.5,
    book_depth_ratio REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.5,
    market_volume REAL NOT NULL DEFAULT 0.0,
    time_to_resolution_hrs REAL NOT NULL DEFAULT 720.0,
    legs_json TEXT NOT NULL DEFAULT '[]',
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opp_cycle ON opportunities(cycle_id);
CREATE INDEX IF NOT EXISTS idx_opp_event ON opportunities(event_id);
CREATE INDEX IF NOT EXISTS idx_opp_type ON opportunities(opp_type);

CREATE TABLE IF NOT EXISTS safety_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL REFERENCES cycles(id),
    event_id TEXT NOT NULL,
    opp_type TEXT NOT NULL,
    check_name TEXT NOT NULL,
    reason TEXT NOT NULL,
    net_profit REAL NOT NULL DEFAULT 0.0,
    roi_pct REAL NOT NULL DEFAULT 0.0,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_safety_cycle ON safety_rejections(cycle_id);

CREATE TABLE IF NOT EXISTS strategy_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL REFERENCES cycles(id),
    mode TEXT NOT NULL,
    gas_price_gwei REAL NOT NULL,
    active_spike_count INTEGER NOT NULL DEFAULT 0,
    has_crypto_momentum INTEGER NOT NULL DEFAULT 0,
    recent_win_rate REAL NOT NULL DEFAULT 0.0,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cross_platform_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL REFERENCES cycles(id),
    pm_event_id TEXT NOT NULL,
    pm_title TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL,
    ext_event_ticker TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    match_method TEXT NOT NULL DEFAULT '',
    pm_best_ask REAL NOT NULL DEFAULT 0.0,
    ext_best_ask REAL NOT NULL DEFAULT 0.0,
    price_diff REAL NOT NULL DEFAULT 0.0,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    cycle_id INTEGER NOT NULL REFERENCES cycles(id),
    event_id TEXT NOT NULL,
    opp_type TEXT NOT NULL,
    n_legs INTEGER NOT NULL,
    fully_filled INTEGER NOT NULL DEFAULT 0,
    fill_prices_json TEXT NOT NULL DEFAULT '[]',
    fill_sizes_json TEXT NOT NULL DEFAULT '[]',
    fees REAL NOT NULL DEFAULT 0.0,
    gas_cost REAL NOT NULL DEFAULT 0.0,
    net_pnl REAL NOT NULL DEFAULT 0.0,
    execution_time_ms REAL NOT NULL DEFAULT 0.0,
    total_score REAL NOT NULL DEFAULT 0.0,
    simulated INTEGER NOT NULL DEFAULT 0,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id);
"""
