"""
Pipeline telemetry collector. Buffers one cycle of events, flushes to SQLite.

ReportCollector — active collector, injected into main loop.
NullCollector — no-op, zero overhead when --report not passed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from scanner.labels import resolve_opportunity_label
from report.store import ReportStore

logger = logging.getLogger(__name__)


# ── Event dataclasses (frozen, pure data) ──

@dataclass(frozen=True)
class FunnelStage:
    stage: str
    count: int


@dataclass(frozen=True)
class StrategySnapshot:
    cycle: int
    mode: str
    gas_price_gwei: float
    active_spike_count: int
    has_crypto_momentum: bool
    recent_win_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ScoredOppSnapshot:
    event_id: str
    opp_type: str
    net_profit: float
    roi_pct: float
    required_capital: float
    n_legs: int
    is_buy_arb: bool
    total_score: float
    profit_score: float
    fill_score: float
    efficiency_score: float
    urgency_score: float
    competition_score: float
    persistence_score: float
    book_depth_ratio: float
    confidence: float
    market_volume: float
    time_to_resolution_hours: float
    platform: str = "polymarket"
    legs_json: str = "[]"
    event_title: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SafetyRejection:
    event_id: str
    opp_type: str
    check_name: str
    reason: str
    net_profit: float = 0.0
    roi_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CrossPlatformSnapshot:
    pm_event_id: str
    pm_title: str
    platform: str
    ext_event_ticker: str
    confidence: float
    match_method: str
    pm_best_ask: float
    ext_best_ask: float
    price_diff: float
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TradeSnapshot:
    event_id: str
    opp_type: str
    n_legs: int
    fully_filled: bool
    fill_prices_json: str
    fill_sizes_json: str
    fees: float
    gas_cost: float
    net_pnl: float
    execution_time_ms: float
    total_score: float
    simulated: bool = False


class ReportCollector:
    """Buffers one cycle of telemetry events, flushes to SQLite on end_cycle()."""

    def __init__(self, store: ReportStore, sse_callback: Callable[[dict], None] | None = None) -> None:
        self._store = store
        self._sse_callback = sse_callback
        self._session_id: int | None = None
        self._cycle_id: int | None = None
        self._cycle_num: int = 0
        self._cycle_start: float = 0.0

        # Per-cycle buffers (reset on begin_cycle)
        self._funnel: dict[str, int] = {}
        self._strategy: StrategySnapshot | None = None
        self._scored_opps: list[ScoredOppSnapshot] = []
        self._safety_rejections: list[SafetyRejection] = []
        self._scanner_counts: dict[str, int] = {}
        self._cross_platform_matches: list[CrossPlatformSnapshot] = []
        self._pending_trades: list[TradeSnapshot] = []
        self._trades_this_cycle: int = 0

    def start_session(self, mode: str, config_json: str = "{}", cli_args_json: str = "{}") -> int:
        self._session_id = self._store.start_session(mode, config_json, cli_args_json)
        logger.debug("Report session started: id=%d", self._session_id)
        return self._session_id

    def end_session(self) -> None:
        if self._session_id is not None:
            self._store.end_session(self._session_id)
            logger.debug("Report session ended: id=%d", self._session_id)

    def begin_cycle(self, cycle: int) -> None:
        self._cycle_num = cycle
        self._cycle_start = time.time()
        self._funnel = {}
        self._strategy = None
        self._scored_opps = []
        self._safety_rejections = []
        self._scanner_counts = {}
        self._cross_platform_matches = []
        self._pending_trades = []
        self._trades_this_cycle = 0

    def record_funnel(self, stage: str, count: int) -> None:
        self._funnel[stage] = count

    def record_strategy(self, snapshot: StrategySnapshot) -> None:
        self._strategy = snapshot

    def record_scored_opps(
        self,
        scored_opps: list,
        contexts: list,
        event_questions: dict[str, str] | None = None,
        market_questions: dict[str, str] | None = None,
    ) -> None:
        """Snapshot scored opportunities with their full score breakdowns.

        Args:
            scored_opps: list of ScoredOpportunity from scorer.py
            contexts: list of ScoringContext matching each opportunity
            event_questions: optional {event_id: title} lookup for event labels
            market_questions: optional {token_id: question} lookup for market labels
        """
        titles = event_questions or {}
        market_map = market_questions or {}
        now = time.time()
        for scored, ctx in zip(scored_opps, contexts):
            opp = scored.opportunity
            # Determine platform from first leg
            platform = "polymarket"
            if opp.legs:
                leg_platform = opp.legs[0].platform
                if leg_platform:
                    platform = leg_platform

            legs_data = [
                {
                    "token_id": leg.token_id,
                    "side": leg.side.value,
                    "price": leg.price,
                    "size": leg.size,
                    "platform": leg.platform,
                }
                for leg in opp.legs
            ]

            self._scored_opps.append(ScoredOppSnapshot(
                event_id=opp.event_id,
                opp_type=opp.type.value,
                net_profit=opp.net_profit,
                roi_pct=opp.roi_pct,
                required_capital=opp.required_capital,
                n_legs=len(opp.legs),
                is_buy_arb=opp.is_buy_arb,
                total_score=scored.total_score,
                profit_score=scored.profit_score,
                fill_score=scored.fill_score,
                efficiency_score=scored.efficiency_score,
                urgency_score=scored.urgency_score,
                competition_score=scored.competition_score,
                persistence_score=scored.persistence_score,
                book_depth_ratio=ctx.book_depth_ratio,
                confidence=ctx.confidence,
                market_volume=ctx.market_volume,
                time_to_resolution_hours=ctx.time_to_resolution_hours,
                platform=platform,
                legs_json=json.dumps(legs_data, separators=(",", ":")),
                event_title=resolve_opportunity_label(
                    opp,
                    event_questions=titles,
                    market_questions=market_map,
                ),
                timestamp=now,
            ))

    def record_safety_rejection(
        self,
        opp: Any,
        check_name: str,
        reason: str,
    ) -> None:
        self._safety_rejections.append(SafetyRejection(
            event_id=opp.event_id,
            opp_type=opp.type.value,
            check_name=check_name,
            reason=reason,
            net_profit=opp.net_profit,
            roi_pct=opp.roi_pct,
        ))

    def record_scanner_counts(self, counts: dict[str, int]) -> None:
        self._scanner_counts = dict(counts)

    def record_cross_platform_match(self, snapshot: CrossPlatformSnapshot) -> None:
        self._cross_platform_matches.append(snapshot)

    def record_trade(self, result: Any, total_score: float) -> None:
        """Record a completed trade from TradeResult."""
        if self._session_id is None:
            return
        opp = result.opportunity
        self._pending_trades.append(TradeSnapshot(
            event_id=opp.event_id,
            opp_type=opp.type.value,
            n_legs=len(opp.legs),
            fully_filled=result.fully_filled,
            fill_prices_json=json.dumps(list(result.fill_prices)),
            fill_sizes_json=json.dumps(list(result.fill_sizes)),
            fees=result.fees,
            gas_cost=result.gas_cost,
            net_pnl=result.net_pnl,
            execution_time_ms=result.execution_time_ms,
            total_score=total_score,
            simulated=False,
        ))
        self._trades_this_cycle += 1

    def record_simulated_trade(self, scored_opp: Any) -> None:
        """Record a simulated execution observation (non-executed, for analysis only)."""
        if self._session_id is None:
            return
        opp = scored_opp.opportunity
        self._pending_trades.append(TradeSnapshot(
            event_id=opp.event_id,
            opp_type=opp.type.value,
            n_legs=len(opp.legs),
            fully_filled=False,
            fill_prices_json=json.dumps([leg.price for leg in opp.legs]),
            fill_sizes_json=json.dumps([leg.size for leg in opp.legs]),
            fees=0.0,
            gas_cost=0.0,
            net_pnl=0.0,
            execution_time_ms=0.0,
            total_score=scored_opp.total_score,
            simulated=True,
        ))

    def end_cycle(self) -> None:
        """Flush all buffered data for this cycle to SQLite, then push SSE."""
        if self._session_id is None:
            return

        elapsed = time.time() - self._cycle_start
        strategy_mode = self._strategy.mode if self._strategy else "unknown"
        gas_gwei = self._strategy.gas_price_gwei if self._strategy else 0.0
        spike_count = self._strategy.active_spike_count if self._strategy else 0
        has_momentum = self._strategy.has_crypto_momentum if self._strategy else False
        win_rate = self._strategy.recent_win_rate if self._strategy else 0.0

        # Derive counts from funnel/buffers
        markets_scanned = self._funnel.get("after_filter", self._funnel.get("raw_markets", 0))
        binary_count = self._funnel.get("binary_count", 0)
        negrisk_events = self._funnel.get("negrisk_events", 0)
        negrisk_markets = self._funnel.get("negrisk_markets", 0)
        opps_found = self._funnel.get("opps_found", len(self._scored_opps))
        opps_executed = self._trades_this_cycle

        strategy_snap = asdict(self._strategy) if self._strategy else None

        try:
            self._store.begin_transaction()
            cycle_id = self._store.insert_cycle(
                session_id=self._session_id,
                cycle_num=self._cycle_num,
                elapsed_sec=elapsed,
                strategy_mode=strategy_mode,
                gas_price_gwei=gas_gwei,
                spike_count=spike_count,
                has_momentum=has_momentum,
                win_rate=win_rate,
                markets_scanned=markets_scanned,
                binary_count=binary_count,
                negrisk_events=negrisk_events,
                negrisk_markets=negrisk_markets,
                opps_found=opps_found,
                opps_executed=opps_executed,
                funnel=self._funnel if self._funnel else None,
                scanner_counts=self._scanner_counts if self._scanner_counts else None,
                strategy_snapshot=strategy_snap,
                commit=False,
            )
            self._cycle_id = cycle_id

            # Flush scored opportunities
            if self._scored_opps:
                opp_dicts = [
                    {
                        "event_id": o.event_id,
                        "opp_type": o.opp_type,
                        "net_profit": o.net_profit,
                        "roi_pct": o.roi_pct,
                        "required_capital": o.required_capital,
                        "n_legs": o.n_legs,
                        "is_buy_arb": o.is_buy_arb,
                        "platform": o.platform,
                        "total_score": o.total_score,
                        "profit_score": o.profit_score,
                        "fill_score": o.fill_score,
                        "efficiency_score": o.efficiency_score,
                        "urgency_score": o.urgency_score,
                        "competition_score": o.competition_score,
                        "persistence_score": o.persistence_score,
                        "book_depth_ratio": o.book_depth_ratio,
                        "confidence": o.confidence,
                        "market_volume": o.market_volume,
                        "time_to_resolution_hrs": o.time_to_resolution_hours,
                        "legs_json": o.legs_json,
                        "event_title": o.event_title,
                        "timestamp": o.timestamp,
                    }
                    for o in self._scored_opps
                ]
                self._store.insert_opportunities(cycle_id, opp_dicts, commit=False)

            # Flush safety rejections
            if self._safety_rejections:
                rej_dicts = [
                    {
                        "event_id": r.event_id,
                        "opp_type": r.opp_type,
                        "check_name": r.check_name,
                        "reason": r.reason,
                        "net_profit": r.net_profit,
                        "roi_pct": r.roi_pct,
                        "timestamp": r.timestamp,
                    }
                    for r in self._safety_rejections
                ]
                self._store.insert_safety_rejections(cycle_id, rej_dicts, commit=False)

            # Flush cross-platform matches
            if self._cross_platform_matches:
                match_dicts = [
                    {
                        "pm_event_id": m.pm_event_id,
                        "pm_title": m.pm_title,
                        "platform": m.platform,
                        "ext_event_ticker": m.ext_event_ticker,
                        "confidence": m.confidence,
                        "match_method": m.match_method,
                        "pm_best_ask": m.pm_best_ask,
                        "ext_best_ask": m.ext_best_ask,
                        "price_diff": m.price_diff,
                        "timestamp": m.timestamp,
                    }
                    for m in self._cross_platform_matches
                ]
                self._store.insert_cross_platform_matches(cycle_id, match_dicts, commit=False)

            # Flush trades recorded during the cycle after cycle_id exists.
            if self._pending_trades:
                trade_rows = [
                    {
                        "session_id": self._session_id,
                        "cycle_id": cycle_id,
                        "event_id": trade.event_id,
                        "opp_type": trade.opp_type,
                        "n_legs": trade.n_legs,
                        "fully_filled": trade.fully_filled,
                        "fill_prices_json": trade.fill_prices_json,
                        "fill_sizes_json": trade.fill_sizes_json,
                        "fees": trade.fees,
                        "gas_cost": trade.gas_cost,
                        "net_pnl": trade.net_pnl,
                        "execution_time_ms": trade.execution_time_ms,
                        "total_score": trade.total_score,
                        "simulated": trade.simulated,
                        "timestamp": time.time(),
                    }
                    for trade in self._pending_trades
                ]
                self._store.insert_trades(trade_rows, commit=False)

            self._store.commit()
        except Exception:
            self._store.rollback()
            raise

        # Push SSE summary
        if self._sse_callback:
            summary = {
                "cycle": self._cycle_num,
                "elapsed_sec": round(elapsed, 2),
                "strategy": {
                    "mode": strategy_mode,
                    "gas_price_gwei": gas_gwei,
                    "active_spike_count": spike_count,
                    "has_crypto_momentum": has_momentum,
                    "recent_win_rate": win_rate,
                },
                "funnel": dict(self._funnel),
                "markets_scanned": markets_scanned,
                "opps_found": opps_found,
                "opps_executed": opps_executed,
                "gas_gwei": gas_gwei,
            }
            try:
                self._sse_callback(summary)
            except Exception:
                logger.debug("SSE callback failed", exc_info=True)


class NullCollector:
    """No-op collector. All methods are pass. Zero overhead."""

    def start_session(self, mode: str, config_json: str = "{}", cli_args_json: str = "{}") -> int:
        return 0

    def end_session(self) -> None:
        pass

    def begin_cycle(self, cycle: int) -> None:
        pass

    def record_funnel(self, stage: str, count: int) -> None:
        pass

    def record_strategy(self, snapshot: StrategySnapshot) -> None:
        pass

    def record_scored_opps(
        self,
        scored_opps: list,
        contexts: list,
        event_questions: dict[str, str] | None = None,
        market_questions: dict[str, str] | None = None,
    ) -> None:
        pass

    def record_safety_rejection(self, opp: Any, check_name: str, reason: str) -> None:
        pass

    def record_scanner_counts(self, counts: dict[str, int]) -> None:
        pass

    def record_cross_platform_match(self, snapshot: CrossPlatformSnapshot) -> None:
        pass

    def record_trade(self, result: Any, total_score: float) -> None:
        pass

    def record_simulated_trade(self, scored_opp: Any) -> None:
        pass

    def end_cycle(self) -> None:
        pass
