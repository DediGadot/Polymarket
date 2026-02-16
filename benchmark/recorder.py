"""
Cycle recorder for pipeline state capture. Records full pipeline state
per cycle as NDJSON for offline replay with different scorer weights.

Uses NullRecorder pattern (same as report/) for zero overhead when disabled.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scanner.models import Opportunity, OrderBook
from scanner.scorer import ScoringContext

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CycleRecord:
    """A single cycle's pipeline state for replay."""
    cycle: int
    timestamp: float
    books: dict[str, dict]       # {token_id: {"bids": [...], "asks": [...]}}
    opportunities: list[dict]
    scoring_contexts: list[dict]
    strategy_mode: str = ""
    config: dict = field(default_factory=dict)


class BaseRecorder(ABC):
    """Abstract base for cycle recording."""

    @abstractmethod
    def record_cycle(
        self,
        cycle: int,
        books: dict[str, OrderBook],
        opportunities: list[Opportunity],
        scoring_contexts: list[ScoringContext],
        strategy_mode: str = "",
        config: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def stats(self) -> dict[str, Any]: ...


class NullRecorder(BaseRecorder):
    """No-op recorder for when recording is disabled. Zero overhead."""

    def record_cycle(self, cycle, books, opportunities, scoring_contexts,
                     strategy_mode="", config=None):
        pass

    def close(self):
        pass

    @property
    def stats(self) -> dict[str, Any]:
        return {"enabled": False, "cycles_recorded": 0}


class CycleRecorder(BaseRecorder):
    """
    Records full pipeline state per cycle as NDJSON.

    Each line in the output file is a JSON object representing one cycle.
    Books are compressed to top N levels to manage storage.
    File rotation occurs when max_mb is exceeded.
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        max_mb: int = 500,
        max_book_levels: int = 5,
    ):
        self._output_dir = Path(output_dir)
        self._max_bytes = max_mb * 1024 * 1024
        self._max_levels = max_book_levels
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._current_file: str = ""
        self._file_handle: Any = None
        self._bytes_written: int = 0
        self._cycles_recorded: int = 0
        self._files_rotated: int = 0
        self._first_cycle_done: bool = False
        self._config_written: bool = False
        self._file_seq: int = 0

        self._open_new_file()

    def record_cycle(
        self,
        cycle: int,
        books: dict[str, OrderBook],
        opportunities: list[Opportunity],
        scoring_contexts: list[ScoringContext],
        strategy_mode: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Record a cycle's state as one NDJSON line."""
        lines: list[str] = []
        if config is not None and not self._config_written:
            config_record = {
                "type": "config",
                "schema_version": SCHEMA_VERSION,
                "timestamp": time.time(),
                "data": config,
            }
            lines.append(json.dumps(config_record, default=str) + "\n")
            self._config_written = True

        contexts = [self._serialize_ctx(c) for c in scoring_contexts]
        cycle_record: dict[str, Any] = {
            "type": "cycle",
            "schema_version": SCHEMA_VERSION,
            "cycle": cycle,
            "timestamp": time.time(),
            "data": {
                "books": self._compress_books(books),
                "opportunities": [self._serialize_opp(o) for o in opportunities],
                # Keep both keys during migration. replay.py consumes "contexts".
                "contexts": contexts,
                "scoring_contexts": contexts,
                "strategy_mode": strategy_mode,
            },
        }
        lines.append(json.dumps(cycle_record, default=str) + "\n")

        total_bytes = sum(len(line.encode("utf-8")) for line in lines)
        if self._bytes_written + total_bytes > self._max_bytes:
            self._rotate_file()

        if self._file_handle is not None:
            for line in lines:
                self._file_handle.write(line)
                self._bytes_written += len(line.encode("utf-8"))
            self._file_handle.flush()
            self._cycles_recorded += 1
            self._first_cycle_done = True

    def close(self) -> None:
        """Close the current output file."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": SCHEMA_VERSION,
            "cycles_recorded": self._cycles_recorded,
            "current_file": self._current_file,
            "bytes_written": self._bytes_written,
            "files_rotated": self._files_rotated,
        }

    # ── internal helpers ──

    def _compress_books(self, books: dict[str, OrderBook]) -> dict[str, dict]:
        """Compress books to top N levels only."""
        result: dict[str, dict] = {}
        for tid, book in books.items():
            result[tid] = {
                "bids": [
                    {"price": lvl.price, "size": lvl.size}
                    for lvl in book.bids[: self._max_levels]
                ],
                "asks": [
                    {"price": lvl.price, "size": lvl.size}
                    for lvl in book.asks[: self._max_levels]
                ],
            }
        return result

    def _serialize_opp(self, opp: Opportunity) -> dict:
        """Serialize an Opportunity to a JSON-safe dict."""
        return {
            "type": opp.type.value,
            "event_id": opp.event_id,
            "legs": [
                {
                    "token_id": leg.token_id,
                    "side": leg.side.value,
                    "price": leg.price,
                    "size": leg.size,
                    "platform": leg.platform,
                    "tick_size": leg.tick_size,
                }
                for leg in opp.legs
            ],
            "expected_profit_per_set": opp.expected_profit_per_set,
            "net_profit_per_set": opp.net_profit_per_set,
            "max_sets": opp.max_sets,
            "gross_profit": opp.gross_profit,
            "estimated_gas_cost": opp.estimated_gas_cost,
            "net_profit": opp.net_profit,
            "roi_pct": opp.roi_pct,
            "required_capital": opp.required_capital,
            "pair_fill_prob": opp.pair_fill_prob,
            "toxicity_score": opp.toxicity_score,
            "reason_code": opp.reason_code,
            "risk_flags": list(opp.risk_flags),
            "timestamp": opp.timestamp,
        }

    def _serialize_ctx(self, ctx: ScoringContext) -> dict:
        """Serialize a ScoringContext to a JSON-safe dict."""
        return {
            "market_volume": ctx.market_volume,
            "recent_trade_count": ctx.recent_trade_count,
            "time_to_resolution_hours": ctx.time_to_resolution_hours,
            "is_spike": ctx.is_spike,
            "book_depth_ratio": ctx.book_depth_ratio,
            "confidence": ctx.confidence,
            "realized_ev_score": ctx.realized_ev_score,
            "ofi_divergence": ctx.ofi_divergence,
        }

    def _open_new_file(self) -> None:
        """Create a new recording file with unique sequence suffix."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}_{self._file_seq:03d}.jsonl"
        self._file_seq += 1
        self._current_file = str(self._output_dir / filename)
        self._file_handle = open(self._current_file, "w")  # noqa: SIM115
        self._bytes_written = 0

    def _rotate_file(self) -> None:
        """Close current file and open a new one."""
        self.close()
        self._files_rotated += 1
        self._open_new_file()
        logger.info(
            "CycleRecorder: rotated to %s (rotation #%d)",
            self._current_file,
            self._files_rotated,
        )


def create_recorder(enabled: bool = False, **kwargs: Any) -> BaseRecorder:
    """Factory: create a CycleRecorder or NullRecorder based on enabled flag."""
    if enabled:
        return CycleRecorder(**kwargs)
    return NullRecorder()
