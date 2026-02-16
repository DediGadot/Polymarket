"""Tests for report.server — FastAPI endpoints."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from report.store import ReportStore

# FastAPI test client requires httpx
try:
    from fastapi.testclient import TestClient
    from report.server import create_app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture
def store(tmp_path: Path) -> ReportStore:
    return ReportStore(db_path=tmp_path / "test.db")


@pytest.fixture
def client(store: ReportStore) -> "TestClient":
    app = create_app(store)
    return TestClient(app)


@pytest.fixture
def populated_store(store: ReportStore) -> tuple[ReportStore, int]:
    """Store with a session, cycle, opportunities, and trades."""
    sid = store.start_session(mode="TEST")
    cid = store.insert_cycle(
        session_id=sid, cycle_num=1, elapsed_sec=2.0,
        strategy_mode="aggressive", gas_price_gwei=30.0,
        spike_count=0, has_momentum=False, win_rate=0.65,
        markets_scanned=100, binary_count=80, negrisk_events=5,
        negrisk_markets=20, opps_found=3, opps_executed=1,
        funnel={"raw_markets": 200, "after_prefilter": 100, "opps_found": 3},
        scanner_counts={"binary": 2, "negrisk": 1},
    )
    store.insert_opportunities(cid, [{
        "event_id": "evt1", "opp_type": "binary_rebalance",
        "net_profit": 1.50, "roi_pct": 3.0, "required_capital": 50.0,
        "n_legs": 2, "is_buy_arb": True, "platform": "polymarket",
        "total_score": 0.75, "profit_score": 0.6, "fill_score": 0.8,
        "efficiency_score": 0.5, "urgency_score": 0.5,
        "competition_score": 1.0, "persistence_score": 0.5,
        "book_depth_ratio": 1.5, "confidence": 0.7,
        "market_volume": 10000.0, "time_to_resolution_hrs": 48.0,
        "legs_json": "[]", "timestamp": time.time(),
    }])
    store.insert_trade(
        session_id=sid, cycle_id=cid, event_id="evt1",
        opp_type="binary_rebalance", n_legs=2, fully_filled=True,
        fill_prices_json="[0.45, 0.53]", fill_sizes_json="[10.0, 10.0]",
        fees=0.05, gas_cost=0.01, net_pnl=0.14,
        execution_time_ms=250.0, total_score=0.75,
    )
    return store, sid


class TestStaticRoute:
    def test_index_serves_html(self, client: "TestClient") -> None:
        resp = client.get("/")
        # Will be 200 if index.html exists, 404 otherwise
        assert resp.status_code in (200, 404)


class TestSessionEndpoints:
    def test_list_sessions_empty(self, client: "TestClient") -> None:
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_sessions(self, client: "TestClient", store: ReportStore) -> None:
        store.start_session(mode="A")
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["mode"] == "A"

    def test_get_session(self, client: "TestClient", store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "TEST"

    def test_get_session_not_found(self, client: "TestClient") -> None:
        resp = client.get("/api/sessions/999")
        assert resp.status_code == 404


class TestCycleEndpoints:
    def test_get_cycles(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/cycles")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1


class TestFunnelEndpoint:
    def test_get_funnel_aggregated(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/funnel")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0


class TestOpportunityEndpoint:
    def test_get_opportunities(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/opportunities")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["event_id"] == "evt1"

    def test_filter_by_type(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/opportunities?type=negrisk_rebalance")
        assert resp.status_code == 200
        assert resp.json() == []


class TestSafetyEndpoint:
    def test_get_safety_empty(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/safety")
        assert resp.status_code == 200
        assert resp.json() == []


class TestTradeEndpoints:
    def test_get_trades(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_get_pnl(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/pnl")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert "cumulative_pnl" in data[0]


class TestScannerEndpoint:
    def test_get_scanners(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/scanners")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


class TestReadinessEndpoint:
    def test_get_readiness(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "recommendation" in data
        assert "checks" in data
        assert len(data["checks"]) == 6


class TestUntappedEndpoint:
    def test_get_untapped_empty(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/untapped")
        assert resp.status_code == 200


class TestStrategyEndpoint:
    def test_get_strategy(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/strategy")
        assert resp.status_code == 200


class TestCrossPlatformEndpoint:
    def test_get_cross_platform(self, client: "TestClient", populated_store: tuple) -> None:
        store, sid = populated_store
        resp = client.get(f"/api/sessions/{sid}/cross-platform")
        assert resp.status_code == 200
