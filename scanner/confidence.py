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
        - First-seen with depth_ratio >= 2.0: 0.7
        - First-seen with depth_ratio < 2.0: 0.3
        """
        cycles = self._history.get(event_id)
        if not cycles:
            return 0.0

        if not has_inventory:
            return 0.1

        if self._is_persistent(cycles):
            return 1.0

        if depth_ratio >= 2.0:
            return 0.7

        return 0.3

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
