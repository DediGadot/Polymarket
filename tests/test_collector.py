"""Tests for report.collector — ReportCollector + NullCollector."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from report.collector import (
    ReportCollector,
    NullCollector,
    StrategySnapshot,
    SafetyRejection,
    CrossPlatformSnapshot,
    ScoredOppSnapshot,
)
from report.store import ReportStore


@pytest.fixture
def store(tmp_path: Path) -> ReportStore:
    return ReportStore(db_path=tmp_path / "test.db")


@pytest.fixture
def collector(store: ReportStore) -> ReportCollector:
    return ReportCollector(store)


def _make_scored_opp() -> MagicMock:
    """Build a mock ScoredOpportunity."""
    opp = MagicMock()
    opp.event_id = "evt_123"
    opp.type.value = "binary_rebalance"
    opp.net_profit = 1.50
    opp.roi_pct = 3.0
    opp.required_capital = 50.0
    opp.legs = []
    opp.is_buy_arb = True

    scored = MagicMock()
    scored.opportunity = opp
    scored.total_score = 0.75
    scored.profit_score = 0.6
    scored.fill_score = 0.8
    scored.efficiency_score = 0.5
    scored.urgency_score = 0.5
    scored.competition_score = 1.0
    scored.persistence_score = 0.5
    return scored


def _make_context() -> MagicMock:
    """Build a mock ScoringContext."""
    ctx = MagicMock()
    ctx.book_depth_ratio = 1.5
    ctx.confidence = 0.7
    ctx.market_volume = 10000.0
    ctx.time_to_resolution_hours = 48.0
    return ctx


class TestReportCollector:
    def test_session_lifecycle(self, collector: ReportCollector, store: ReportStore) -> None:
        sid = collector.start_session(mode="DRY-RUN")
        assert sid > 0

        sessions = store.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]["mode"] == "DRY-RUN"

        collector.end_session()
        session = store.get_session(sid)
        assert session["end_ts"] is not None

    def test_full_cycle_flow(self, collector: ReportCollector, store: ReportStore) -> None:
        sid = collector.start_session(mode="TEST")
        collector.begin_cycle(1)

        # Record funnel
        collector.record_funnel("raw_markets", 1000)
        collector.record_funnel("after_limit", 500)
        collector.record_funnel("after_prefilter", 400)

        # Record strategy
        collector.record_strategy(StrategySnapshot(
            cycle=1, mode="aggressive", gas_price_gwei=30.0,
            active_spike_count=0, has_crypto_momentum=False,
            recent_win_rate=0.5,
        ))

        # Record scanner counts
        collector.record_scanner_counts({"binary": 3, "negrisk": 2})

        # Record scored opps
        scored = _make_scored_opp()
        ctx = _make_context()
        collector.record_scored_opps([scored], [ctx])

        # End cycle
        collector.end_cycle()

        # Verify data persisted
        cycles = store.get_cycles(sid)
        assert len(cycles) == 1
        assert cycles[0]["strategy_mode"] == "aggressive"

        funnel = store.get_funnel(cycles[0]["id"])
        stages = {f["stage"]: f["count"] for f in funnel}
        assert stages["raw_markets"] == 1000

        opps = store.get_opportunities(sid)
        assert len(opps) == 1
        assert opps[0]["event_id"] == "evt_123"

    def test_record_safety_rejection(self, collector: ReportCollector, store: ReportStore) -> None:
        sid = collector.start_session(mode="TEST")
        collector.begin_cycle(1)
        collector.record_strategy(StrategySnapshot(
            cycle=1, mode="aggressive", gas_price_gwei=30.0,
            active_spike_count=0, has_crypto_momentum=False,
            recent_win_rate=0.5,
        ))

        opp = MagicMock()
        opp.event_id = "evt_456"
        opp.type.value = "binary_rebalance"
        opp.net_profit = 1.0
        opp.roi_pct = 2.0
        collector.record_safety_rejection(opp, "depth", "Insufficient depth")

        collector.end_cycle()

        rejections = store.get_safety_rejections(sid)
        assert len(rejections) == 1
        assert rejections[0]["check_name"] == "depth"

    def test_record_cross_platform_match(self, collector: ReportCollector, store: ReportStore) -> None:
        sid = collector.start_session(mode="TEST")
        collector.begin_cycle(1)
        collector.record_strategy(StrategySnapshot(
            cycle=1, mode="aggressive", gas_price_gwei=30.0,
            active_spike_count=0, has_crypto_momentum=False,
            recent_win_rate=0.5,
        ))

        collector.record_cross_platform_match(CrossPlatformSnapshot(
            pm_event_id="pm_evt1", pm_title="Test",
            platform="kalshi", ext_event_ticker="KALSHI-1",
            confidence=0.95, match_method="manual",
            pm_best_ask=0.45, ext_best_ask=0.42, price_diff=0.03,
        ))

        collector.end_cycle()

        matches = store.get_cross_platform_matches(sid)
        assert len(matches) == 1
        assert matches[0]["platform"] == "kalshi"

    def test_sse_callback_called(self, store: ReportStore) -> None:
        callback = MagicMock()
        collector = ReportCollector(store, sse_callback=callback)
        collector.start_session(mode="TEST")
        collector.begin_cycle(1)
        collector.record_funnel("raw_markets", 100)
        collector.record_funnel("opps_found", 2)
        collector.record_strategy(StrategySnapshot(
            cycle=1, mode="aggressive", gas_price_gwei=30.0,
            active_spike_count=0, has_crypto_momentum=False,
            recent_win_rate=0.5,
        ))
        collector.end_cycle()

        callback.assert_called_once()
        summary = callback.call_args[0][0]
        assert summary["cycle"] == 1
        assert summary["strategy"]["mode"] == "aggressive"
        assert summary["strategy"]["gas_price_gwei"] == 30.0
        assert summary["funnel"]["raw_markets"] == 100
        assert summary["funnel"]["opps_found"] == 2


    def test_record_simulated_trade(self, collector: ReportCollector, store: ReportStore) -> None:
        sid = collector.start_session(mode="SCAN-ONLY")
        collector.begin_cycle(1)
        collector.record_strategy(StrategySnapshot(
            cycle=1, mode="aggressive", gas_price_gwei=30.0,
            active_spike_count=0, has_crypto_momentum=False,
            recent_win_rate=0.5,
        ))

        scored = _make_scored_opp()
        # Add legs with price/size for fill_prices/fill_sizes serialization
        leg = MagicMock()
        leg.price = 0.45
        leg.size = 10.0
        leg.side.value = "BUY"
        leg.token_id = "tok_1"
        leg.platform = "polymarket"
        scored.opportunity.legs = [leg]
        scored.opportunity.estimated_gas_cost = 0.005

        collector.end_cycle()  # flush to get cycle_id

        collector.begin_cycle(2)
        collector.record_strategy(StrategySnapshot(
            cycle=2, mode="aggressive", gas_price_gwei=30.0,
            active_spike_count=0, has_crypto_momentum=False,
            recent_win_rate=0.5,
        ))
        collector.record_simulated_trade(scored)
        collector.end_cycle()

        trades = store.get_trades(sid)
        assert len(trades) == 1
        assert trades[0]["simulated"] == 1
        assert trades[0]["event_id"] == "evt_123"
        assert trades[0]["execution_time_ms"] == 0.0
        assert trades[0]["fees"] == 0.0
        assert trades[0]["gas_cost"] == 0.0
        assert trades[0]["net_pnl"] == 0.0
        assert trades[0]["fully_filled"] == 0
        cycle2 = next(c for c in store.get_cycles(sid) if c["cycle_num"] == 2)
        assert trades[0]["cycle_id"] == cycle2["id"]
        assert cycle2["opps_executed"] == 0

    def test_record_simulated_trade_no_session(self, collector: ReportCollector, store: ReportStore) -> None:
        """record_simulated_trade is a no-op when no session is active."""
        scored = _make_scored_opp()
        scored.opportunity.legs = []
        scored.opportunity.estimated_gas_cost = 0.0
        collector.record_simulated_trade(scored)
        # No crash, no data written


class TestNullCollector:
    def test_all_methods_are_noop(self) -> None:
        nc = NullCollector()
        assert nc.start_session(mode="TEST") == 0
        nc.begin_cycle(1)
        nc.record_funnel("raw_markets", 100)
        nc.record_strategy(StrategySnapshot(
            cycle=1, mode="aggressive", gas_price_gwei=30.0,
            active_spike_count=0, has_crypto_momentum=False,
            recent_win_rate=0.5,
        ))
        nc.record_scored_opps([], [])
        nc.record_safety_rejection(MagicMock(), "depth", "x")
        nc.record_scanner_counts({})
        nc.record_cross_platform_match(MagicMock())
        nc.record_trade(MagicMock(), 0.5)
        nc.record_simulated_trade(MagicMock())
        nc.end_cycle()
        nc.end_session()
        # No exceptions raised
