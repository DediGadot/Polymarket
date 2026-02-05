"""
Scan-only summary tracker. Accumulates opportunities across cycles
and produces an aggregate summary on shutdown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from scanner.models import Opportunity


@dataclass
class ScanTracker:
    """Mirrors PnLTracker but for scan-only / dry-run mode."""

    total_cycles: int = 0
    total_markets_scanned: int = 0
    opportunities: list[Opportunity] = field(default_factory=list)
    unique_event_ids: set[str] = field(default_factory=set)
    _session_start: float = field(default_factory=time.time)

    def record_cycle(
        self, cycle: int, n_markets: int, opportunities: list[Opportunity]
    ) -> None:
        """Record one scan cycle's results."""
        self.total_cycles = cycle
        self.total_markets_scanned += n_markets
        self.opportunities.extend(opportunities)
        for opp in opportunities:
            self.unique_event_ids.add(opp.event_id)

    def summary(self) -> dict:
        """Return aggregate summary dict, consistent with PnLTracker.summary() pattern."""
        by_type: dict[str, int] = {}
        for opp in self.opportunities:
            key = opp.type.value
            by_type[key] = by_type.get(key, 0) + 1

        n = len(self.opportunities)
        best_roi = max((o.roi_pct for o in self.opportunities), default=0.0)
        best_profit = max((o.net_profit for o in self.opportunities), default=0.0)
        total_profit = sum(o.net_profit for o in self.opportunities)
        avg_roi = (sum(o.roi_pct for o in self.opportunities) / n) if n else 0.0
        avg_profit = (total_profit / n) if n else 0.0

        return {
            "total_cycles": self.total_cycles,
            "duration_sec": round(time.time() - self._session_start, 0),
            "markets_scanned": self.total_markets_scanned,
            "opportunities_found": n,
            "unique_events": len(self.unique_event_ids),
            "by_type": by_type,
            "best_roi_pct": round(best_roi, 2),
            "best_profit_usd": round(best_profit, 2),
            "total_theoretical_profit_usd": round(total_profit, 2),
            "avg_roi_pct": round(avg_roi, 2),
            "avg_profit_usd": round(avg_profit, 2),
        }
