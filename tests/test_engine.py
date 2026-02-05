"""
Unit tests for executor/engine.py -- trade execution engine.
"""

from unittest.mock import patch, MagicMock, call
import pytest

from executor.engine import execute_opportunity, _paper_execute
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
    TradeResult,
)


def _make_binary_opp():
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id="e1",
        legs=(
            LegOrder("y1", Side.BUY, 0.45, 100),
            LegOrder("n1", Side.BUY, 0.45, 100),
        ),
        expected_profit_per_set=0.10,
        max_sets=100,
        gross_profit=10.0,
        estimated_gas_cost=0.01,
        net_profit=9.99,
        roi_pct=11.1,
        required_capital=90.0,
    )


def _make_negrisk_opp():
    return Opportunity(
        type=OpportunityType.NEGRISK_REBALANCE,
        event_id="e1",
        legs=(
            LegOrder("y1", Side.BUY, 0.30, 100),
            LegOrder("y2", Side.BUY, 0.30, 100),
            LegOrder("y3", Side.BUY, 0.30, 100),
        ),
        expected_profit_per_set=0.10,
        max_sets=100,
        gross_profit=10.0,
        estimated_gas_cost=0.01,
        net_profit=9.99,
        roi_pct=11.1,
        required_capital=90.0,
    )


class TestPaperExecute:
    def test_paper_binary(self):
        opp = _make_binary_opp()
        result = execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=True)

        assert result.fully_filled is True
        assert len(result.order_ids) == 2
        assert all(oid.startswith("paper_") for oid in result.order_ids)
        assert result.net_pnl > 0
        assert result.execution_time_ms >= 0

    def test_paper_negrisk(self):
        opp = _make_negrisk_opp()
        result = execute_opportunity(MagicMock(), opp, size=30.0, paper_trading=True)

        assert result.fully_filled is True
        assert len(result.order_ids) == 3
        assert result.fill_sizes == [30.0, 30.0, 30.0]


class TestExecuteBinary:
    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_full_fill(self, mock_create, mock_post):
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        opp = _make_binary_opp()
        result = execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False)

        assert result.fully_filled is True
        assert result.order_ids == ["o1", "o2"]
        assert mock_create.call_count == 2
        mock_post.assert_called_once()

    @patch("executor.engine._unwind_partial")
    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_partial_fill_triggers_unwind(self, mock_create, mock_post, mock_unwind):
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "open"},  # not filled
        ]

        opp = _make_binary_opp()
        result = execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False)

        assert result.fully_filled is False
        mock_unwind.assert_called_once()

    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_correct_order_params(self, mock_create, mock_post):
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        opp = _make_binary_opp()
        execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False)

        # Verify create_limit_order was called with correct params
        calls = mock_create.call_args_list
        assert len(calls) == 2
        # First leg: BUY YES at 0.45
        assert calls[0].kwargs["token_id"] == "y1"
        assert calls[0].kwargs["side"] == Side.BUY
        assert calls[0].kwargs["price"] == 0.45
        assert calls[0].kwargs["size"] == 50.0


class TestExecuteNegRisk:
    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_full_fill_3_legs(self, mock_create, mock_post):
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
            {"orderID": "o3", "status": "matched"},
        ]

        opp = _make_negrisk_opp()
        result = execute_opportunity(MagicMock(), opp, size=30.0, paper_trading=False)

        assert result.fully_filled is True
        assert len(result.order_ids) == 3
        assert mock_create.call_count == 3

    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_large_event_batched(self, mock_create, mock_post):
        """Events with >15 outcomes should be batched."""
        legs = tuple(
            LegOrder(f"y{i}", Side.BUY, 0.05, 100)
            for i in range(20)
        )
        opp = Opportunity(
            type=OpportunityType.NEGRISK_REBALANCE,
            event_id="e1",
            legs=legs,
            expected_profit_per_set=0.0,
            max_sets=100,
            gross_profit=0.0,
            estimated_gas_cost=0.01,
            net_profit=0.0,
            roi_pct=0.0,
            required_capital=1.0,
        )

        mock_create.return_value = MagicMock()
        # First batch: 15 orders, second batch: 5 orders
        mock_post.side_effect = [
            [{"orderID": f"o{i}", "status": "matched"} for i in range(15)],
            [{"orderID": f"o{i+15}", "status": "matched"} for i in range(5)],
        ]

        result = execute_opportunity(MagicMock(), opp, size=10.0, paper_trading=False)

        assert mock_post.call_count == 2  # two batches
        assert len(result.order_ids) == 20
        assert result.fully_filled is True
