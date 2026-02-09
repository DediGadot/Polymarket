"""
Unit tests for monitor/pnl.py -- P&L tracking.
"""

import json
import os
import tempfile

from monitor.pnl import PnLTracker
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
    TradeResult,
)


def _make_opp():
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id="e1",
        legs=(
            LegOrder("y1", Side.BUY, 0.45, 100),
            LegOrder("n1", Side.BUY, 0.45, 100),
        ),
        expected_profit_per_set=0.10,
        net_profit_per_set=0.10,
        max_sets=100,
        gross_profit=10.0,
        estimated_gas_cost=0.01,
        net_profit=9.99,
        roi_pct=11.1,
        required_capital=90.0,
    )


def _make_result(pnl=1.0, fully_filled=True):
    return TradeResult(
        opportunity=_make_opp(),
        order_ids=["o1", "o2"],
        fill_prices=[0.45, 0.45],
        fill_sizes=[10.0, 10.0],
        fees=0.0,
        gas_cost=0.01,
        net_pnl=pnl,
        execution_time_ms=50.0,
        fully_filled=fully_filled,
    )


def _make_sell_result(pnl=1.0, fully_filled=True):
    sell_opp = Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id="e1",
        legs=(
            LegOrder("y1", Side.SELL, 0.55, 100),
            LegOrder("n1", Side.SELL, 0.55, 100),
        ),
        expected_profit_per_set=0.10,
        net_profit_per_set=0.10,
        max_sets=100,
        gross_profit=10.0,
        estimated_gas_cost=0.01,
        net_profit=9.99,
        roi_pct=11.1,
        required_capital=100.0,
    )
    return TradeResult(
        opportunity=sell_opp,
        order_ids=["o1", "o2"],
        fill_prices=[0.55, 0.55],
        fill_sizes=[10.0, 10.0],
        fees=0.0,
        gas_cost=0.01,
        net_pnl=pnl,
        execution_time_ms=50.0,
        fully_filled=fully_filled,
    )


class TestPnLTracker:
    def test_initial_state(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        assert tracker.total_pnl == 0.0
        assert tracker.total_trades == 0
        assert tracker.win_rate == 0.0
        assert tracker.avg_pnl == 0.0

    def test_record_winning_trade(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        tracker.record(_make_result(pnl=5.0))

        assert tracker.total_pnl == 5.0
        assert tracker.total_trades == 1
        assert tracker.winning_trades == 1
        assert tracker.losing_trades == 0
        assert tracker.win_rate == 100.0

    def test_record_losing_trade(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        tracker.record(_make_result(pnl=-2.0))

        assert tracker.total_pnl == -2.0
        assert tracker.winning_trades == 0
        assert tracker.losing_trades == 1
        assert tracker.win_rate == 0.0

    def test_multiple_trades(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        tracker.record(_make_result(pnl=5.0))
        tracker.record(_make_result(pnl=-2.0))
        tracker.record(_make_result(pnl=3.0))

        assert tracker.total_pnl == 6.0
        assert tracker.total_trades == 3
        assert tracker.winning_trades == 2
        assert tracker.losing_trades == 1
        assert abs(tracker.win_rate - 66.666) < 1

    def test_avg_pnl(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        tracker.record(_make_result(pnl=10.0))
        tracker.record(_make_result(pnl=-2.0))
        assert abs(tracker.avg_pnl - 4.0) < 1e-9

    def test_ledger_append(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            ledger_path = f.name

        try:
            tracker = PnLTracker(ledger_path=ledger_path)
            tracker.record(_make_result(pnl=5.0))
            tracker.record(_make_result(pnl=-1.0))

            with open(ledger_path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            entry1 = json.loads(lines[0])
            assert entry1["net_pnl"] == 5.0
            assert entry1["opportunity_type"] == "binary_rebalance"

            entry2 = json.loads(lines[1])
            assert entry2["net_pnl"] == -1.0
        finally:
            os.unlink(ledger_path)

    def test_summary(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        tracker.record(_make_result(pnl=10.0))
        summary = tracker.summary()

        assert summary["total_pnl"] == 10.0
        assert summary["total_trades"] == 1
        assert summary["win_rate_pct"] == 100.0
        assert "session_duration_sec" in summary

    def test_volume_tracking(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        # fill_prices=[0.45, 0.45], fill_sizes=[10.0, 10.0]
        # volume = 0.45*10 + 0.45*10 = 9.0
        tracker.record(_make_result(pnl=1.0))
        assert abs(tracker.total_volume - 9.0) < 1e-9

    def test_exposure_tracks_filled_buy_capital_only(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        tracker.record(_make_result(pnl=1.0, fully_filled=True))
        # BUY-only trade: exposure should reflect actual filled buy notional.
        assert tracker.current_exposure == 9.0

    def test_sell_only_trade_does_not_increase_exposure(self):
        tracker = PnLTracker(ledger_path="/dev/null")
        tracker.record(_make_sell_result(pnl=1.0, fully_filled=True))
        assert tracker.current_exposure == 0.0
