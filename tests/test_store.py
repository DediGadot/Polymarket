"""Tests for report.store — SQLite telemetry storage."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from report.store import ReportStore


@pytest.fixture
def store(tmp_path: Path) -> ReportStore:
    db_path = tmp_path / "test_report.db"
    return ReportStore(db_path=db_path)


class TestSessionLifecycle:
    def test_start_and_end_session(self, store: ReportStore) -> None:
        sid = store.start_session(mode="DRY-RUN", config_json='{"k":"v"}')
        assert sid > 0

        session = store.get_session(sid)
        assert session is not None
        assert session["mode"] == "DRY-RUN"
        assert session["end_ts"] is None

        store.end_session(sid)
        session = store.get_session(sid)
        assert session["end_ts"] is not None

    def test_get_sessions_returns_list(self, store: ReportStore) -> None:
        store.start_session(mode="A")
        store.start_session(mode="B")
        sessions = store.get_sessions()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0]["mode"] == "B"

    def test_get_nonexistent_session(self, store: ReportStore) -> None:
        assert store.get_session(999) is None


class TestCycleInsert:
    def test_insert_cycle_basic(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=2.5,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=3, opps_executed=1,
        )
        assert cid > 0

        cycles = store.get_cycles(sid)
        assert len(cycles) == 1
        assert cycles[0]["strategy_mode"] == "aggressive"
        assert cycles[0]["markets_scanned"] == 100

    def test_insert_cycle_with_funnel(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="conservative", gas_price_gwei=50.0,
            spike_count=0, has_momentum=False, win_rate=0.0,
            markets_scanned=50, binary_count=40, negrisk_events=2,
            negrisk_markets=10, opps_found=0, opps_executed=0,
            funnel={"raw_markets": 200, "after_limit": 100, "after_prefilter": 50},
        )
        funnel = store.get_funnel(cid)
        assert len(funnel) == 3
        stages = {f["stage"]: f["count"] for f in funnel}
        assert stages["raw_markets"] == 200
        assert stages["after_prefilter"] == 50

    def test_insert_cycle_with_scanner_counts(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=5, opps_executed=0,
            scanner_counts={"binary": 3, "negrisk": 2},
        )
        dist = store.get_scanner_distribution(sid)
        assert len(dist) == 2
        scanners = {d["scanner"]: d["total_count"] for d in dist}
        assert scanners["binary"] == 3

    def test_get_latest_cycle(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=5, opps_executed=0,
        )
        store.insert_cycle(
            session_id=sid, cycle_num=2, elapsed_sec=1.0,
            strategy_mode="conservative", gas_price_gwei=50.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=3, opps_executed=0,
        )
        latest = store.get_latest_cycle(sid)
        assert latest is not None
        assert latest["cycle_num"] == 2


class TestOpportunities:
    def test_insert_and_query_opportunities(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=2, opps_executed=0,
        )
        opps = [
            {
                "event_id": "evt1", "opp_type": "binary_rebalance",
                "net_profit": 1.50, "roi_pct": 3.0, "required_capital": 50.0,
                "n_legs": 2, "is_buy_arb": True, "platform": "polymarket",
                "total_score": 0.75, "profit_score": 0.6, "fill_score": 0.8,
                "efficiency_score": 0.5, "urgency_score": 0.5,
                "competition_score": 1.0, "persistence_score": 0.5,
                "book_depth_ratio": 1.5, "confidence": 0.7,
                "market_volume": 10000.0, "time_to_resolution_hrs": 48.0,
                "legs_json": "[]", "timestamp": time.time(),
            },
            {
                "event_id": "evt2", "opp_type": "negrisk_rebalance",
                "net_profit": 0.80, "roi_pct": 2.0, "required_capital": 40.0,
                "n_legs": 3, "is_buy_arb": True, "platform": "polymarket",
                "total_score": 0.55, "profit_score": 0.4, "fill_score": 0.6,
                "efficiency_score": 0.3, "urgency_score": 0.5,
                "competition_score": 0.8, "persistence_score": 0.5,
                "book_depth_ratio": 1.0, "confidence": 0.5,
                "market_volume": 5000.0, "time_to_resolution_hrs": 168.0,
                "legs_json": "[]", "timestamp": time.time(),
            },
        ]
        store.insert_opportunities(cid, opps)

        result = store.get_opportunities(sid)
        assert len(result) == 2
        assert result[0]["total_score"] >= result[1]["total_score"]

    def test_filter_opportunities_by_type(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=2, opps_executed=0,
        )
        opps = [
            {
                "event_id": "evt1", "opp_type": "binary_rebalance",
                "net_profit": 1.0, "roi_pct": 2.0, "required_capital": 50.0,
                "n_legs": 2, "is_buy_arb": True, "platform": "polymarket",
                "total_score": 0.5, "profit_score": 0.5, "fill_score": 0.5,
                "efficiency_score": 0.5, "urgency_score": 0.5,
                "competition_score": 0.5, "persistence_score": 0.5,
                "book_depth_ratio": 1.0, "confidence": 0.5,
                "market_volume": 1000.0, "time_to_resolution_hrs": 48.0,
                "legs_json": "[]", "timestamp": time.time(),
            },
        ]
        store.insert_opportunities(cid, opps)

        result = store.get_opportunities(sid, opp_type="binary_rebalance")
        assert len(result) == 1
        result = store.get_opportunities(sid, opp_type="negrisk_rebalance")
        assert len(result) == 0


class TestSafetyRejections:
    def test_insert_and_query(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=1, opps_executed=0,
        )
        store.insert_safety_rejections(cid, [
            {
                "event_id": "evt1", "opp_type": "binary_rebalance",
                "check_name": "depth", "reason": "Insufficient depth",
                "net_profit": 1.0, "roi_pct": 2.0, "timestamp": time.time(),
            },
        ])
        result = store.get_safety_rejections(sid)
        assert len(result) == 1
        assert result[0]["check_name"] == "depth"

    def test_safety_pass_rates(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=3, opps_executed=0,
        )
        store.insert_safety_rejections(cid, [
            {"event_id": "e1", "opp_type": "binary_rebalance", "check_name": "depth", "reason": "x", "net_profit": 1.0, "roi_pct": 2.0, "timestamp": time.time()},
            {"event_id": "e2", "opp_type": "binary_rebalance", "check_name": "depth", "reason": "x", "net_profit": 1.0, "roi_pct": 2.0, "timestamp": time.time()},
            {"event_id": "e3", "opp_type": "binary_rebalance", "check_name": "gas", "reason": "x", "net_profit": 1.0, "roi_pct": 2.0, "timestamp": time.time()},
        ])
        rates = store.get_safety_pass_rates(sid)
        assert len(rates) == 2
        assert rates[0]["check_name"] == "depth"
        assert rates[0]["fail_count"] == 2


class TestTrades:
    def test_insert_and_query_trade(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=1, opps_executed=1,
        )
        tid = store.insert_trade(
            session_id=sid, cycle_id=cid, event_id="evt1",
            opp_type="binary_rebalance", n_legs=2, fully_filled=True,
            fill_prices_json="[0.45, 0.53]", fill_sizes_json="[10.0, 10.0]",
            fees=0.05, gas_cost=0.01, net_pnl=0.14,
            execution_time_ms=250.0, total_score=0.75,
        )
        assert tid > 0

        trades = store.get_trades(sid)
        assert len(trades) == 1
        assert trades[0]["net_pnl"] == 0.14

    def test_insert_simulated_trade(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=1, opps_executed=0,
        )
        tid = store.insert_trade(
            session_id=sid, cycle_id=cid, event_id="evt_sim",
            opp_type="binary_rebalance", n_legs=2, fully_filled=True,
            fill_prices_json="[0.45, 0.53]", fill_sizes_json="[10.0, 10.0]",
            fees=0.0, gas_cost=0.01, net_pnl=0.14,
            execution_time_ms=0.0, total_score=0.75,
            simulated=True,
        )
        assert tid > 0

        trades = store.get_trades(sid)
        assert len(trades) == 1
        assert trades[0]["simulated"] == 1
        assert trades[0]["execution_time_ms"] == 0.0

    def test_insert_real_trade_not_simulated(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=1, opps_executed=1,
        )
        store.insert_trade(
            session_id=sid, cycle_id=cid, event_id="evt_real",
            opp_type="binary_rebalance", n_legs=2, fully_filled=True,
            fill_prices_json="[]", fill_sizes_json="[]",
            fees=0.05, gas_cost=0.01, net_pnl=0.14,
            execution_time_ms=250.0, total_score=0.75,
        )
        trades = store.get_trades(sid)
        assert len(trades) == 1
        assert trades[0]["simulated"] == 0

    def test_pnl_series(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=2, opps_executed=2,
        )
        store.insert_trade(
            session_id=sid, cycle_id=cid, event_id="e1",
            opp_type="binary_rebalance", n_legs=2, fully_filled=True,
            fill_prices_json="[]", fill_sizes_json="[]",
            fees=0.0, gas_cost=0.0, net_pnl=1.0,
            execution_time_ms=100.0, total_score=0.5,
        )
        store.insert_trade(
            session_id=sid, cycle_id=cid, event_id="e2",
            opp_type="binary_rebalance", n_legs=2, fully_filled=True,
            fill_prices_json="[]", fill_sizes_json="[]",
            fees=0.0, gas_cost=0.0, net_pnl=-0.50,
            execution_time_ms=100.0, total_score=0.4,
        )
        series = store.get_pnl_series(sid)
        assert len(series) == 2
        assert series[-1]["cumulative_pnl"] == 0.50


class TestCrossPlatformMatches:
    def test_insert_and_query(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=1, opps_executed=0,
        )
        store.insert_cross_platform_matches(cid, [
            {
                "pm_event_id": "pm_evt1", "pm_title": "Test Event",
                "platform": "kalshi", "ext_event_ticker": "KALSHI-EVT1",
                "confidence": 0.95, "match_method": "manual",
                "pm_best_ask": 0.45, "ext_best_ask": 0.42,
                "price_diff": 0.03, "timestamp": time.time(),
            },
        ])
        result = store.get_cross_platform_matches(sid)
        assert len(result) == 1
        assert result[0]["platform"] == "kalshi"


class TestUntapped:
    def test_untapped_opportunities(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        cid = store.insert_cycle(
            session_id=sid, cycle_num=1, elapsed_sec=1.0,
            strategy_mode="aggressive", gas_price_gwei=30.0,
            spike_count=0, has_momentum=False, win_rate=0.5,
            markets_scanned=100, binary_count=80, negrisk_events=5,
            negrisk_markets=20, opps_found=1, opps_executed=0,
        )
        # Insert an opp that scored well
        store.insert_opportunities(cid, [{
            "event_id": "evt1", "opp_type": "binary_rebalance",
            "net_profit": 2.0, "roi_pct": 4.0, "required_capital": 50.0,
            "n_legs": 2, "is_buy_arb": True, "platform": "polymarket",
            "total_score": 0.8, "profit_score": 0.7, "fill_score": 0.8,
            "efficiency_score": 0.6, "urgency_score": 0.5,
            "competition_score": 1.0, "persistence_score": 0.6,
            "book_depth_ratio": 1.5, "confidence": 0.7,
            "market_volume": 10000.0, "time_to_resolution_hrs": 48.0,
            "legs_json": "[]", "timestamp": time.time(),
        }])
        # But it was rejected by safety
        store.insert_safety_rejections(cid, [{
            "event_id": "evt1", "opp_type": "binary_rebalance",
            "check_name": "depth", "reason": "Insufficient depth",
            "net_profit": 2.0, "roi_pct": 4.0, "timestamp": time.time(),
        }])
        untapped = store.get_untapped(sid)
        assert len(untapped) == 1
        assert untapped[0]["check_name"] == "depth"


class TestClose:
    def test_close(self, store: ReportStore) -> None:
        store.start_session(mode="TEST")
        store.close()
        # After close, new operations open a fresh connection
        sessions = store.get_sessions()
        assert len(sessions) == 1
