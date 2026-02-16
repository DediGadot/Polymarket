"""
ArbTracker confidence model. Tracks opportunity persistence across scan cycles
to assign confidence scores. Persistent arbs (seen multiple consecutive cycles)
are more likely to be real. First-seen arbs with deep books get moderate
confidence. Sell-side arbs without inventory are heavily penalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scanner.models import Opportunity


@dataclass
class ArbTracker:
    """Tracks which events appear across scan cycles and scores confidence."""

    _history: dict[str, list[int]] = field(default_factory=dict)
    _failures: dict[str, int] = field(default_factory=dict)
    _max_history: int = 10

    def record(self, cycle_num: int, opportunities: list[Opportunity]) -> None:
        """Record which events appeared in this cycle."""
        for opp in opportunities:
            eid = opp.event_id
            if eid not in self._history:
                self._history[eid] = []
            cycles = self._history[eid]
            if not cycles or cycles[-1] != cycle_num:
                cycles.append(cycle_num)
        self._purge_stale(cycle_num)

    def record_failure(self, event_id: str) -> None:
        """Record a safety check failure for an event. Penalizes confidence."""
        self._failures[event_id] = self._failures.get(event_id, 0) + 1

    def confidence(
        self,
        event_id: str,
        depth_ratio: float = 1.0,
        has_inventory: bool = True,
    ) -> float:
        """
        Return confidence score for an event.

        - Unknown event: 0.0
        - Sell-side without inventory: 0.1
        - 2+ consecutive cycles: 1.0
        - First-seen with depth_ratio >= 2.0: 0.3
        - First-seen with depth_ratio < 2.0: 0.1

        Failure penalty: each recorded failure reduces base score by 20%,
        floored at 0.05.
        """
        cycles = self._history.get(event_id)
        if not cycles:
            return 0.0

        if not has_inventory:
            return 0.1

        if self._is_persistent(cycles):
            base = 1.0
        elif depth_ratio >= 2.0:
            base = 0.3
        else:
            base = 0.1

        failures = self._failures.get(event_id, 0)
        if failures > 0:
            base = max(base * (1 - failures * 0.2), 0.05)

        return base

    def _is_persistent(self, cycles: list[int]) -> bool:
        """Check if the last 2 entries are consecutive cycle numbers."""
        if len(cycles) < 2:
            return False
        return cycles[-1] == cycles[-2] + 1

    def _purge_stale(self, current_cycle: int) -> None:
        """Remove entries whose last sighting is >max_history cycles ago."""
        stale_keys = [
            eid
            for eid, cycles in self._history.items()
            if current_cycle - cycles[-1] > self._max_history
        ]
        for eid in stale_keys:
            del self._history[eid]

    def to_dict(self) -> dict:
        """Serialize tracker state to a JSON-safe dict."""
        return {
            "history": {k: list(v) for k, v in self._history.items()},
            "failures": dict(self._failures),
            "max_history": self._max_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArbTracker:
        """Restore tracker from a serialized dict."""
        tracker = cls()
        tracker._history = {k: list(v) for k, v in data.get("history", {}).items()}
        tracker._failures = dict(data.get("failures", {}))
        tracker._max_history = data.get("max_history", 10)
        return tracker
