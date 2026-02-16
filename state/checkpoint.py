"""
Checkpoint manager. Persists tracker state to SQLite for crash recovery.

Supports auto-save every N cycles and graceful shutdown save via signal handlers.
Each save is atomic (single SQLite transaction).
"""

from __future__ import annotations

import json
import logging
import signal
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracker_state (
    tracker_name TEXT PRIMARY KEY,
    data_json TEXT NOT NULL,
    cycle_num INTEGER DEFAULT 0,
    updated_at REAL NOT NULL
);
"""

DEFAULT_DB_PATH = Path("state.db")


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that support to_dict/from_dict serialization."""

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> Any: ...


class CheckpointManager:
    """
    Persists tracker state to SQLite. Thread-safe for single-writer usage.

    Usage:
        mgr = CheckpointManager(db_path="state.db", auto_save_interval=10)
        mgr.save("arb_tracker", tracker)
        restored = mgr.load("arb_tracker", ArbTracker)
        mgr.tick()  # call each cycle; saves all registered trackers every N cycles
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        auto_save_interval: int = 10,
    ) -> None:
        self._db_path = str(db_path)
        self._auto_save_interval = auto_save_interval
        self._cycle_count = 0
        self._save_count = 0
        self._load_count = 0
        self._trackers: dict[str, Serializable] = {}
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def register(self, name: str, tracker: Serializable) -> None:
        """Register a tracker for auto-save. Does not save immediately."""
        with self._lock:
            self._trackers[name] = tracker

    def unregister(self, name: str) -> None:
        """Remove a tracker from auto-save."""
        with self._lock:
            self._trackers.pop(name, None)

    def save(self, name: str, tracker: Serializable, cycle_num: int = 0) -> None:
        """Persist a single tracker to SQLite. Atomic."""
        data = tracker.to_dict()
        data_json = json.dumps(data, default=str)
        now = time.time()

        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO tracker_state "
                "(tracker_name, data_json, cycle_num, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (name, data_json, cycle_num, now),
            )
            conn.commit()
            self._save_count += 1

        logger.debug(
            "Checkpoint saved: %s (%d bytes, cycle %d)", name, len(data_json), cycle_num
        )

    def save_all(
        self, trackers: dict[str, Serializable] | None = None, cycle_num: int = 0
    ) -> int:
        """
        Save trackers in a single atomic transaction.

        If *trackers* is None, saves all registered trackers.
        Returns count of trackers saved.
        """
        if trackers is None:
            with self._lock:
                snapshot = dict(self._trackers)
        else:
            snapshot = trackers

        rows: list[tuple[str, str, int, float]] = []
        now = time.time()
        for name, tracker in snapshot.items():
            try:
                data = tracker.to_dict()
                data_json = json.dumps(data, default=str)
                rows.append((name, data_json, cycle_num, now))
            except Exception as e:
                logger.error("Failed to serialize %s: %s", name, e)

        if not rows:
            return 0

        with self._lock:
            conn = self._get_conn()
            conn.executemany(
                "INSERT OR REPLACE INTO tracker_state "
                "(tracker_name, data_json, cycle_num, updated_at) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            self._save_count += len(rows)

        logger.debug("Checkpoint batch saved: %d trackers at cycle %d", len(rows), cycle_num)
        return len(rows)

    def load(self, name: str, tracker_cls: type) -> Any | None:
        """
        Load a tracker from SQLite. Returns None if not found.
        Returns a new instance via tracker_cls.from_dict() on success.
        Falls back gracefully on corrupt JSON.
        """
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT data_json, cycle_num, updated_at FROM tracker_state "
                "WHERE tracker_name = ?",
                (name,),
            ).fetchone()

        if row is None:
            logger.debug("Checkpoint not found: %s", name)
            return None

        data_json, cycle_num, updated_at = row
        try:
            data = json.loads(data_json)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Corrupt checkpoint for %s, ignoring: %s", name, e)
            return None

        try:
            tracker = tracker_cls.from_dict(data)
            age_sec = time.time() - updated_at
            self._load_count += 1
            logger.info(
                "Checkpoint restored: %s (cycle %d, %.1fs ago)", name, cycle_num, age_sec
            )
            return tracker
        except Exception as e:
            logger.warning("Failed to restore %s from checkpoint: %s", name, e)
            return None

    def load_metadata(self, name: str) -> dict[str, Any] | None:
        """Get metadata about a checkpoint without loading the full state."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT cycle_num, updated_at, LENGTH(data_json) "
                "FROM tracker_state WHERE tracker_name = ?",
                (name,),
            ).fetchone()

        if row is None:
            return None

        return {
            "cycle_num": row[0],
            "updated_at": row[1],
            "age_sec": time.time() - row[1],
            "data_bytes": row[2],
        }

    def tick(self) -> int:
        """
        Called once per scan cycle. Saves all registered trackers
        every auto_save_interval cycles. Returns count saved (0 if not a save cycle).
        """
        self._cycle_count += 1
        if self._cycle_count % self._auto_save_interval == 0:
            return self.save_all(cycle_num=self._cycle_count)
        return 0

    def install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers that save all trackers before exit."""
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        def _shutdown_handler(signum: int, frame: Any) -> None:
            logger.info("Signal %d received, saving checkpoints...", signum)
            try:
                saved = self.save_all(cycle_num=self._cycle_count)
                logger.info("Saved %d tracker(s) on shutdown", saved)
            except Exception as e:
                logger.error("Checkpoint save on shutdown failed: %s", e)

            # Re-raise via original handler
            original = original_sigterm if signum == signal.SIGTERM else original_sigint
            if callable(original):
                original(signum, frame)
            elif original == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)

        signal.signal(signal.SIGTERM, _shutdown_handler)
        signal.signal(signal.SIGINT, _shutdown_handler)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all saved checkpoints with metadata."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT tracker_name, cycle_num, updated_at, LENGTH(data_json) "
                "FROM tracker_state ORDER BY tracker_name"
            ).fetchall()
        now = time.time()
        return [
            {
                "name": r[0],
                "cycle_num": r[1],
                "updated_at": r[2],
                "age_sec": now - r[2],
                "data_bytes": r[3],
            }
            for r in rows
        ]

    def delete(self, name: str) -> bool:
        """Delete a checkpoint. Returns True if it existed."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "DELETE FROM tracker_state WHERE tracker_name = ?", (name,)
            )
            conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "db_path": self._db_path,
            "save_count": self._save_count,
            "load_count": self._load_count,
            "checkpoints": len(self.list_checkpoints()),
        }
