"""
Unit tests for executor/engine.py -- trade execution engine.
"""

from unittest.mock import patch, MagicMock
import pytest

from executor.engine import execute_opportunity, _unwind_partial, UnwindFailed
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
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
        net_profit_per_set=0.10,
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
        net_profit_per_set=0.10,
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

    @patch("executor.engine._unwind_partial")
    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_partial_status_triggers_unwind_with_reported_fill_size(self, mock_create, mock_post, mock_unwind):
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "partial", "filled_size": "10"},
            {"orderID": "o2", "status": "matched"},
        ]

        opp = _make_binary_opp()
        result = execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False)

        assert result.fully_filled is False
        mock_unwind.assert_called_once()
        fill_sizes = mock_unwind.call_args.args[3]
        assert fill_sizes[0] == pytest.approx(10.0)
        assert fill_sizes[1] == pytest.approx(50.0)

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
            net_profit_per_set=0.0,
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


class TestExecutionSafety:
    @patch("executor.engine.time.sleep")
    @patch("executor.engine.cancel_order")
    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_gtc_polls_before_timeout_unwind(self, mock_create, mock_post, mock_cancel, mock_sleep):
        """GTC mode should poll order status before deciding to unwind."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "open"},
            {"orderID": "o2", "status": "open"},
        ]
        client = MagicMock()
        client.get_order.return_value = {"status": "open"}

        opp = _make_binary_opp()
        result = execute_opportunity(
            client,
            opp,
            size=50.0,
            paper_trading=False,
            use_fak=False,
            order_timeout_sec=0.2,
        )

        assert result.fully_filled is False
        assert client.get_order.called
        assert mock_sleep.called
        assert mock_cancel.call_count == 2

    @patch("executor.engine.post_order")
    @patch("executor.engine.create_market_order")
    def test_binary_unwind_uses_non_negrisk_flag(self, mock_create_market_order, mock_post_order):
        """Binary unwind must keep neg_risk=False on the recovery market order."""
        mock_create_market_order.return_value = MagicMock()
        mock_post_order.return_value = {"status": "filled"}

        _unwind_partial(
            MagicMock(),
            order_ids=["o1"],
            legs=(LegOrder("y1", Side.BUY, 0.45, 100),),
            fill_sizes=[10.0],
            neg_risk=False,
        )

        assert mock_create_market_order.call_args.kwargs["neg_risk"] is False


class TestExecuteSingleLeg:
    @patch("executor.engine.post_order")
    @patch("executor.engine.create_limit_order")
    def test_single_leg_filled(self, mock_create, mock_post):
        mock_create.return_value = MagicMock()
        mock_post.return_value = {"orderID": "o1", "status": "matched"}

        opp = Opportunity(
            type=OpportunityType.LATENCY_ARB,
            event_id="e1",
            legs=(LegOrder("y1", Side.BUY, 0.50, 200),),
            expected_profit_per_set=0.10,
            net_profit_per_set=0.10,
            max_sets=200,
            gross_profit=20.0,
            estimated_gas_cost=0.005,
            net_profit=19.995,
            roi_pct=20.0,
            required_capital=100.0,
        )

        result = execute_opportunity(MagicMock(), opp, size=100.0, paper_trading=False)
        assert result.fully_filled is True
        assert result.order_ids == ["o1"]
        assert result.net_pnl > 0

    @patch("executor.engine.post_order")
    @patch("executor.engine.create_limit_order")
    def test_single_leg_not_filled(self, mock_create, mock_post):
        mock_create.return_value = MagicMock()
        mock_post.return_value = {"orderID": "o1", "status": "open"}

        opp = Opportunity(
            type=OpportunityType.LATENCY_ARB,
            event_id="e1",
            legs=(LegOrder("y1", Side.BUY, 0.50, 200),),
            expected_profit_per_set=0.10,
            net_profit_per_set=0.10,
            max_sets=200,
            gross_profit=20.0,
            estimated_gas_cost=0.005,
            net_profit=19.995,
            roi_pct=20.0,
            required_capital=100.0,
        )

        result = execute_opportunity(MagicMock(), opp, size=100.0, paper_trading=False)
        assert result.fully_filled is False
        assert result.net_pnl == 0.0


class TestWaitForFill:
    def test_immediate_fill(self):
        from executor.engine import _wait_for_fill
        client = MagicMock()
        client.get_order.return_value = {"status": "matched"}
        assert _wait_for_fill(client, "o1", timeout_sec=1.0) is True

    def test_cancelled_order_returns_false(self):
        from executor.engine import _wait_for_fill
        client = MagicMock()
        client.get_order.return_value = {"status": "cancelled"}
        assert _wait_for_fill(client, "o1", timeout_sec=1.0) is False

    def test_expired_order_returns_false(self):
        from executor.engine import _wait_for_fill
        client = MagicMock()
        client.get_order.return_value = {"status": "expired"}
        assert _wait_for_fill(client, "o1", timeout_sec=1.0) is False

    def test_rejected_order_returns_false(self):
        from executor.engine import _wait_for_fill
        client = MagicMock()
        client.get_order.return_value = {"status": "rejected"}
        assert _wait_for_fill(client, "o1", timeout_sec=1.0) is False

    def test_zero_timeout_returns_false(self):
        from executor.engine import _wait_for_fill
        assert _wait_for_fill(MagicMock(), "o1", timeout_sec=0) is False

    def test_poll_error_returns_false(self):
        from executor.engine import _wait_for_fill
        client = MagicMock()
        client.get_order.side_effect = Exception("network error")
        assert _wait_for_fill(client, "o1", timeout_sec=0.1) is False


class TestOrderIsFilled:
    def test_matched_is_filled(self):
        from executor.engine import _order_is_filled
        assert _order_is_filled("matched") is True

    def test_filled_is_filled(self):
        from executor.engine import _order_is_filled
        assert _order_is_filled("filled") is True

    def test_partial_is_filled(self):
        from executor.engine import _order_is_filled
        assert _order_is_filled("partial") is False

    def test_open_is_not_filled(self):
        from executor.engine import _order_is_filled
        assert _order_is_filled("open") is False

    def test_case_insensitive(self):
        from executor.engine import _order_is_filled
        assert _order_is_filled("MATCHED") is True
        assert _order_is_filled("Filled") is True
        assert _order_is_filled("PARTIAL") is False


class TestFAKOrderType:
    """Verify we use FAK (Fill-and-Kill) not FOK (Fill-or-Kill)."""

    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_uses_fak_order_type(self, mock_create, mock_post):
        from py_clob_client.clob_types import OrderType
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        opp = _make_binary_opp()
        execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False, use_fak=True)

        # post_orders receives list of (signed_order, order_type) tuples
        call_args = mock_post.call_args[0][1]
        for _, order_type in call_args:
            assert order_type == OrderType.FAK, f"Expected FAK, got {order_type}"

    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_gtc_when_fak_disabled(self, mock_create, mock_post):
        from py_clob_client.clob_types import OrderType
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        opp = _make_binary_opp()
        execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False, use_fak=False)

        call_args = mock_post.call_args[0][1]
        for _, order_type in call_args:
            assert order_type == OrderType.GTC, f"Expected GTC, got {order_type}"


class TestPaperLatencyArb:
    def test_paper_single_leg(self):
        opp = Opportunity(
            type=OpportunityType.LATENCY_ARB,
            event_id="e1",
            legs=(LegOrder("y1", Side.BUY, 0.50, 200),),
            expected_profit_per_set=0.10,
            net_profit_per_set=0.10,
            max_sets=200,
            gross_profit=20.0,
            estimated_gas_cost=0.005,
            net_profit=19.995,
            roi_pct=20.0,
            required_capital=100.0,
        )
        result = execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=True)
        assert result.fully_filled is True
        assert len(result.order_ids) == 1
        assert result.order_ids[0].startswith("paper_")


class TestPaperSpikeLag:
    def test_paper_spike_lag(self):
        opp = Opportunity(
            type=OpportunityType.SPIKE_LAG,
            event_id="e1",
            legs=(
                LegOrder("y1", Side.BUY, 0.20, 100),
                LegOrder("y2", Side.BUY, 0.25, 100),
                LegOrder("y3", Side.BUY, 0.30, 100),
            ),
            expected_profit_per_set=0.05,
            net_profit_per_set=0.05,
            max_sets=100,
            gross_profit=5.0,
            estimated_gas_cost=0.015,
            net_profit=4.985,
            roi_pct=6.6,
            required_capital=75.0,
        )
        result = execute_opportunity(MagicMock(), opp, size=30.0, paper_trading=True)
        assert result.fully_filled is True
        assert len(result.order_ids) == 3


class TestUnwindFailed:
    @patch("executor.engine.post_order")
    @patch("executor.engine.create_market_order")
    @patch("executor.engine.cancel_order")
    def test_unwind_raises_on_market_order_failure(self, mock_cancel, mock_create_market, mock_post):
        """If a market-sell unwind fails, UnwindFailed should be raised."""
        mock_create_market.side_effect = Exception("network error")

        with pytest.raises(UnwindFailed, match="Failed to unwind"):
            _unwind_partial(
                MagicMock(),
                order_ids=["o1", "o2"],
                legs=(
                    LegOrder("y1", Side.BUY, 0.45, 100),
                    LegOrder("y2", Side.BUY, 0.45, 100),
                ),
                fill_sizes=[10.0, 0.0],  # o1 filled, o2 not
                neg_risk=False,
            )

        # o2 should have been cancelled (check order_id arg, not client arg)
        assert mock_cancel.call_count == 1
        assert mock_cancel.call_args[0][1] == "o2"

    @patch("executor.engine.post_order")
    @patch("executor.engine.create_market_order")
    @patch("executor.engine.cancel_order")
    def test_unwind_succeeds_no_exception(self, mock_cancel, mock_create_market, mock_post):
        """If all unwinds succeed, no exception should be raised."""
        mock_create_market.return_value = MagicMock()
        mock_post.return_value = {"status": "filled"}

        # Should not raise
        _unwind_partial(
            MagicMock(),
            order_ids=["o1", "o2"],
            legs=(
                LegOrder("y1", Side.BUY, 0.45, 100),
                LegOrder("y2", Side.BUY, 0.45, 100),
            ),
            fill_sizes=[10.0, 0.0],
            neg_risk=False,
        )
