"""
P&L tracking with append-only JSON ledger.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict

from scanner.models import TradeResult, Side

logger = logging.getLogger(__name__)

LEDGER_FILE = "pnl_ledger.json"


@dataclass
class PnLEntry:
    timestamp: float
    opportunity_type: str
    event_id: str
    n_legs: int
    fully_filled: bool
    fill_prices: list[float]
    fill_sizes: list[float]
    fees: float
    gas_cost: float
    net_pnl: float
    execution_time_ms: float
    order_ids: list[str]


@dataclass
class PnLTracker:
    """Track aggregate P&L and persist individual trades to disk."""

    ledger_path: str = LEDGER_FILE

    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_volume: float = 0.0
    current_exposure: float = 0.0

    _session_start: float = field(default_factory=time.time)

    def record(self, result: TradeResult) -> None:
        """Record a completed trade. Updates aggregates and appends to ledger."""
        self.total_trades += 1
        self.total_pnl += result.net_pnl

        if result.net_pnl >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        volume = sum(
            fp * fs for fp, fs in zip(result.fill_prices, result.fill_sizes)
        )
        self.total_volume += volume

        # Update exposure tracking
        if result.fully_filled:
            # Exposure reflects deployed buy-side notional from filled legs.
            buy_notional = 0.0
            for leg, fill_price, fill_size in zip(
                result.opportunity.legs, result.fill_prices, result.fill_sizes,
            ):
                if leg.side == Side.BUY and fill_size > 0:
                    buy_notional += fill_price * fill_size
            self.current_exposure += buy_notional
        else:
            # Partial fills were unwound, exposure is minimal
            pass

        entry = PnLEntry(
            timestamp=result.timestamp,
            opportunity_type=result.opportunity.type.value,
            event_id=result.opportunity.event_id,
            n_legs=len(result.opportunity.legs),
            fully_filled=result.fully_filled,
            fill_prices=result.fill_prices,
            fill_sizes=result.fill_sizes,
            fees=result.fees,
            gas_cost=result.gas_cost,
            net_pnl=result.net_pnl,
            execution_time_ms=result.execution_time_ms,
            order_ids=result.order_ids,
        )
        self._append_ledger(entry)

        logger.info(
            "PnL update: trade_pnl=$%.2f total_pnl=$%.2f trades=%d win_rate=%.1f%%",
            result.net_pnl, self.total_pnl, self.total_trades, self.win_rate,
        )

    def _append_ledger(self, entry: PnLEntry) -> None:
        """Append a trade entry to the JSON ledger file (one JSON object per line)."""
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100.0

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def session_duration_sec(self) -> float:
        return time.time() - self._session_start

    def reduce_exposure(self, amount: float) -> None:
        """
        Reduce current exposure by the given amount.
        Used when positions are sold, unwound, or resolved.
        Exposure is floored at 0 (never goes negative).
        """
        self.current_exposure = max(0.0, self.current_exposure - amount)

    def summary(self) -> dict:
        """Return a summary dict of current P&L state."""
        return {
            "total_pnl": round(self.total_pnl, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": round(self.win_rate, 1),
            "avg_pnl": round(self.avg_pnl, 2),
            "total_volume": round(self.total_volume, 2),
            "current_exposure": round(self.current_exposure, 2),
            "session_duration_sec": round(self.session_duration_sec, 0),
        }
