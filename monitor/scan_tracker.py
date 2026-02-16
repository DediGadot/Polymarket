"""
Scan-only summary tracker. Accumulates opportunities across cycles
and produces an aggregate summary on shutdown.

Memory-bounded: opportunities list is capped to prevent unbounded growth
in long-running sessions. Default max 100 cycles worth of opportunities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from scanner.models import Opportunity, OpportunityType


@dataclass
class ScanTracker:
    """Mirrors PnLTracker but for scan-only / dry-run mode."""

    total_cycles: int = 0
    total_markets_scanned: int = 0
    opportunities: list[Opportunity] = field(default_factory=list)
    executable_opportunities: list[Opportunity] = field(default_factory=list)
    research_opportunities: list[Opportunity] = field(default_factory=list)
    actionable_now: list[Opportunity] = field(default_factory=list)
    maker_candidates: list[Opportunity] = field(default_factory=list)
    unique_event_ids: set[str] = field(default_factory=set)
    _session_start: float = field(default_factory=time.time)
    total_opportunities_found: int = 0
    total_roi_sum: float = 0.0
    total_profit_sum: float = 0.0
    best_roi_pct_seen: float = 0.0
    best_profit_usd_seen: float = 0.0
    by_type_totals: dict[str, int] = field(default_factory=dict)
    buy_arb_count_total: int = 0
    sell_arb_count_total: int = 0
    buy_arb_profit_total: float = 0.0
    sell_arb_profit_total: float = 0.0
    executable_opp_count_total: int = 0
    executable_opp_profit_total: float = 0.0
    research_opp_count_total: int = 0
    research_opp_profit_total: float = 0.0
    actionable_now_count_total: int = 0
    actionable_now_profit_total: float = 0.0
    maker_candidate_count_total: int = 0
    maker_candidate_profit_total: float = 0.0
    unique_opportunities_found_total: int = 0
    repeated_opportunities_found_total: int = 0
    unique_opportunity_profit_total: float = 0.0
    repeated_opportunity_profit_total: float = 0.0
    dedup_window_sec: float = 30.0
    _last_seen_fingerprint_at: dict[tuple, float] = field(default_factory=dict)
    # Memory cap: max opportunities to retain (prevents unbounded growth)
    max_opportunities: int = 100

    def _fingerprint(self, opp: Opportunity) -> tuple:
        """Stable fingerprint for short-window duplicate detection."""
        legs = tuple(
            sorted(
                (
                    leg.token_id,
                    leg.side.value,
                    round(leg.price, 4),
                    round(leg.size, 4),
                )
                for leg in opp.legs
            )
        )
        return (
            opp.type.value,
            opp.event_id,
            opp.reason_code,
            tuple(sorted(opp.risk_flags)),
            round(opp.net_profit, 2),
            round(opp.roi_pct, 2),
            legs,
        )

    def record_cycle(
        self,
        cycle: int,
        n_markets: int,
        opportunities: list[Opportunity],
        executable_opps: list[Opportunity] | None = None,
        research_opps: list[Opportunity] | None = None,
        actionable_now: list[Opportunity] | None = None,
        maker_candidates: list[Opportunity] | None = None,
    ) -> None:
        """Record one scan cycle's results."""
        now = time.time()
        self.total_cycles = cycle
        self.total_markets_scanned += n_markets
        self.total_opportunities_found += len(opportunities)
        self.opportunities.extend(opportunities)
        if executable_opps:
            self.executable_opportunities.extend(executable_opps)
        if research_opps:
            self.research_opportunities.extend(research_opps)
        if actionable_now:
            self.actionable_now.extend(actionable_now)
        if maker_candidates:
            self.maker_candidates.extend(maker_candidates)
        for opp in opportunities:
            fingerprint = self._fingerprint(opp)
            last_seen = self._last_seen_fingerprint_at.get(fingerprint)
            is_repeat = (
                last_seen is not None
                and self.dedup_window_sec > 0
                and (now - last_seen) <= self.dedup_window_sec
            )
            if is_repeat:
                self.repeated_opportunities_found_total += 1
                self.repeated_opportunity_profit_total += opp.net_profit
            else:
                self.unique_opportunities_found_total += 1
                self.unique_opportunity_profit_total += opp.net_profit
            self._last_seen_fingerprint_at[fingerprint] = now

            self.unique_event_ids.add(opp.event_id)
            self.by_type_totals[opp.type.value] = self.by_type_totals.get(opp.type.value, 0) + 1
            self.total_roi_sum += opp.roi_pct
            self.total_profit_sum += opp.net_profit
            if opp.roi_pct > self.best_roi_pct_seen:
                self.best_roi_pct_seen = opp.roi_pct
            if opp.net_profit > self.best_profit_usd_seen:
                self.best_profit_usd_seen = opp.net_profit
            if opp.is_buy_arb and opp.type != OpportunityType.MAKER_REBALANCE:
                self.buy_arb_count_total += 1
                self.buy_arb_profit_total += opp.net_profit
            if opp.is_sell_arb:
                self.sell_arb_count_total += 1
                self.sell_arb_profit_total += opp.net_profit

        cycle_executable = executable_opps if executable_opps is not None else []
        cycle_research = research_opps if research_opps is not None else []
        cycle_actionable = actionable_now if actionable_now is not None else []
        cycle_maker = maker_candidates if maker_candidates is not None else []
        if executable_opps is None and research_opps is None:
            research_types = {
                OpportunityType.CORRELATION_ARB,
                OpportunityType.RESOLUTION_SNIPE,
                OpportunityType.STALE_QUOTE_ARB,
                OpportunityType.NEGRISK_VALUE,
            }
            cycle_research = [o for o in opportunities if o.type in research_types]
            cycle_executable = [o for o in opportunities if o.type not in research_types]
        if actionable_now is None:
            cycle_actionable = [
                o for o in opportunities if o.is_buy_arb and o.type != OpportunityType.MAKER_REBALANCE
            ]
        if maker_candidates is None:
            cycle_maker = [o for o in opportunities if o.type == OpportunityType.MAKER_REBALANCE]

        self.executable_opp_count_total += len(cycle_executable)
        self.executable_opp_profit_total += sum(o.net_profit for o in cycle_executable)
        self.research_opp_count_total += len(cycle_research)
        self.research_opp_profit_total += sum(o.net_profit for o in cycle_research)
        self.actionable_now_count_total += len(cycle_actionable)
        self.actionable_now_profit_total += sum(o.net_profit for o in cycle_actionable)
        self.maker_candidate_count_total += len(cycle_maker)
        self.maker_candidate_profit_total += sum(o.net_profit for o in cycle_maker)

        # Trim to max_opportunities to prevent memory leak
        if len(self.opportunities) > self.max_opportunities:
            # Keep only the most recent opportunities
            self.opportunities = self.opportunities[-self.max_opportunities:]
        if len(self.executable_opportunities) > self.max_opportunities:
            self.executable_opportunities = self.executable_opportunities[-self.max_opportunities:]
        if len(self.research_opportunities) > self.max_opportunities:
            self.research_opportunities = self.research_opportunities[-self.max_opportunities:]
        if len(self.actionable_now) > self.max_opportunities:
            self.actionable_now = self.actionable_now[-self.max_opportunities:]
        if len(self.maker_candidates) > self.max_opportunities:
            self.maker_candidates = self.maker_candidates[-self.max_opportunities:]

        # Prune stale dedup entries to keep memory bounded.
        if self.dedup_window_sec > 0 and self._last_seen_fingerprint_at:
            cutoff = now - self.dedup_window_sec
            stale = [
                fp for fp, ts in self._last_seen_fingerprint_at.items()
                if ts < cutoff
            ]
            for fp in stale:
                del self._last_seen_fingerprint_at[fp]

    def summary(self) -> dict:
        """Return aggregate summary dict, consistent with PnLTracker.summary() pattern."""
        n = self.total_opportunities_found
        total_profit = self.total_profit_sum
        avg_roi = (self.total_roi_sum / n) if n else 0.0
        avg_profit = (total_profit / n) if n else 0.0

        return {
            "total_cycles": self.total_cycles,
            "duration_sec": round(time.time() - self._session_start, 0),
            "markets_scanned": self.total_markets_scanned,
            "opportunities_found": n,
            "unique_events": len(self.unique_event_ids),
            "by_type": dict(self.by_type_totals),
            "best_roi_pct": round(self.best_roi_pct_seen, 2),
            "best_profit_usd": round(self.best_profit_usd_seen, 2),
            "total_theoretical_profit_usd": round(total_profit, 2),
            "avg_roi_pct": round(avg_roi, 2),
            "avg_profit_usd": round(avg_profit, 2),
            "buy_arb_count": self.buy_arb_count_total,
            "sell_arb_count": self.sell_arb_count_total,
            "buy_arb_profit_usd": round(self.buy_arb_profit_total, 2),
            "sell_arb_profit_usd": round(self.sell_arb_profit_total, 2),
            "executable_opp_count": self.executable_opp_count_total,
            "executable_opp_profit_usd": round(self.executable_opp_profit_total, 2),
            "research_opp_count": self.research_opp_count_total,
            "research_opp_profit_usd": round(self.research_opp_profit_total, 2),
            "actionable_now_count": self.actionable_now_count_total,
            "actionable_now_profit_usd": round(self.actionable_now_profit_total, 2),
            "maker_candidate_count": self.maker_candidate_count_total,
            "maker_candidate_profit_usd": round(self.maker_candidate_profit_total, 2),
            "dedup_window_sec": round(self.dedup_window_sec, 1),
            "unique_opportunities_found": self.unique_opportunities_found_total,
            "repeated_opportunities_found": self.repeated_opportunities_found_total,
            "unique_opportunity_profit_usd": round(self.unique_opportunity_profit_total, 2),
            "repeated_opportunity_profit_usd": round(self.repeated_opportunity_profit_total, 2),
        }
