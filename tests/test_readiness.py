"""Tests for report.readiness — Go-live composite scoring."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from report.store import ReportStore
from report.readiness import compute_readiness


@pytest.fixture
def store(tmp_path: Path) -> ReportStore:
    return ReportStore(db_path=tmp_path / "test.db")


def _setup_session_with_trades(
    store: ReportStore,
    pnls: list[float],
    exec_ms: float = 200.0,
    simulated: bool = False,
) -> int:
    """Helper: create a session with trades having the given PnL values."""
    sid = store.start_session(mode="TEST")
    cid = store.insert_cycle(
        session_id=sid, cycle_num=1, elapsed_sec=5.0,
        strategy_mode="aggressive", gas_price_gwei=30.0,
        spike_count=0, has_momentum=False, win_rate=0.5,
        markets_scanned=100, binary_count=80, negrisk_events=5,
        negrisk_markets=20, opps_found=len(pnls), opps_executed=len(pnls),
    )
    for i, pnl in enumerate(pnls):
        store.insert_trade(
            session_id=sid, cycle_id=cid, event_id=f"evt_{i}",
            opp_type="binary_rebalance", n_legs=2, fully_filled=True,
            fill_prices_json="[]", fill_sizes_json="[]",
            fees=0.01, gas_cost=0.005, net_pnl=pnl,
            execution_time_ms=exec_ms, total_score=0.5,
            simulated=simulated,
        )
    return sid


class TestReadiness:
    def test_empty_session(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        result = compute_readiness(store, sid)
        assert result["score"] >= 0
        assert result["recommendation"] in ("GO", "CAUTION", "STOP")
        assert len(result["checks"]) == 6

    def test_good_session_scores_high(self, store: ReportStore) -> None:
        # 80% win rate, fast execution, positive PnL
        pnls = [1.0, 0.5, 1.2, 0.8, -0.3, 1.5, 0.9, 1.1, 0.7, -0.2]  # 80% wins
        sid = _setup_session_with_trades(store, pnls, exec_ms=150.0)
        result = compute_readiness(store, sid)
        assert result["score"] >= 50
        assert result["recommendation"] in ("GO", "CAUTION")

    def test_bad_session_scores_low(self, store: ReportStore) -> None:
        # All losses
        pnls = [-1.0, -0.5, -1.2, -0.8, -0.3, -1.5, -0.9, -1.1, -0.7, -0.2]
        sid = _setup_session_with_trades(store, pnls)
        result = compute_readiness(store, sid)
        assert result["score"] < 50
        assert result["recommendation"] in ("CAUTION", "STOP")

    def test_checks_structure(self, store: ReportStore) -> None:
        pnls = [1.0, -0.5]
        sid = _setup_session_with_trades(store, pnls)
        result = compute_readiness(store, sid)

        for check in result["checks"]:
            assert "name" in check
            assert "score" in check
            assert "weight" in check
            assert "detail" in check
            assert "passed" in check
            assert 0.0 <= check["score"] <= 1.0

    def test_summary_text_present(self, store: ReportStore) -> None:
        sid = store.start_session(mode="TEST")
        result = compute_readiness(store, sid)
        assert "summary" in result
        assert len(result["summary"]) > 0

    def test_slow_execution_lowers_score(self, store: ReportStore) -> None:
        pnls = [1.0, 0.5, 1.2]
        sid_fast = _setup_session_with_trades(store, pnls, exec_ms=100.0)
        sid_slow = _setup_session_with_trades(store, pnls, exec_ms=1000.0)

        fast_result = compute_readiness(store, sid_fast)
        slow_result = compute_readiness(store, sid_slow)

        fast_speed = next(c for c in fast_result["checks"] if c["name"] == "execution_speed")
        slow_speed = next(c for c in slow_result["checks"] if c["name"] == "execution_speed")
        assert fast_speed["score"] > slow_speed["score"]

    def test_simulated_trades_are_excluded_from_performance_checks(self, store: ReportStore) -> None:
        """Simulated trades should not contribute to readiness performance metrics."""
        pnls = [1.0, 0.5, 1.2, 0.8, -0.3, 1.5, 0.9, 1.1, 0.7, -0.2]
        sid = _setup_session_with_trades(store, pnls, exec_ms=0.0, simulated=True)
        result = compute_readiness(store, sid)

        assert result["simulated_only"] is True
        assert result["real_trade_count"] == 0
        assert result["simulated_trade_count"] == len(pnls)

        wr = next(c for c in result["checks"] if c["name"] == "win_rate")
        assert "No real/paper trades" in wr["detail"]
        assert wr["score"] == 0.0

        pnl_check = next(c for c in result["checks"] if c["name"] == "pnl_trend")
        assert "No real/paper P&L data" in pnl_check["detail"]

    def test_win_rate_uses_real_trades_only_when_simulated_exist(self, store: ReportStore) -> None:
        """Win rate should be computed only from real/paper fills."""
        sid = _setup_session_with_trades(store, [1.0, -1.0], simulated=False)
        cycle_id = store.get_cycles(sid, limit=1)[0]["id"]
        # Add simulated trades that would otherwise bias win-rate to 75%.
        store.insert_trade(
            session_id=sid, cycle_id=cycle_id, event_id="evt_sim_1",
            opp_type="binary_rebalance", n_legs=2, fully_filled=False,
            fill_prices_json="[]", fill_sizes_json="[]",
            fees=0.0, gas_cost=0.0, net_pnl=1.0,
            execution_time_ms=0.0, total_score=0.5, simulated=True,
        )
        store.insert_trade(
            session_id=sid, cycle_id=cycle_id, event_id="evt_sim_2",
            opp_type="binary_rebalance", n_legs=2, fully_filled=False,
            fill_prices_json="[]", fill_sizes_json="[]",
            fees=0.0, gas_cost=0.0, net_pnl=1.0,
            execution_time_ms=0.0, total_score=0.5, simulated=True,
        )

        result = compute_readiness(store, sid)

        wr = next(c for c in result["checks"] if c["name"] == "win_rate")
        assert "50%" in wr["detail"]
        assert "2 simulated excluded" in wr["detail"]
        assert not wr["passed"]  # 50% < 60% threshold
