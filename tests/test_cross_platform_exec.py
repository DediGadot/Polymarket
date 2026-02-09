"""
Unit tests for executor/cross_platform.py -- dual-platform execution.
"""

from unittest.mock import MagicMock, patch
import pytest

from executor.cross_platform import (
    execute_cross_platform,
    CrossPlatformUnwindFailed,
    _unwind_kalshi,
)
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)


def _make_cross_opp() -> Opportunity:
    return Opportunity(
        type=OpportunityType.CROSS_PLATFORM_ARB,
        event_id="e1",
        legs=(
            LegOrder("pm_no", Side.BUY, 0.40, 100, platform="polymarket"),
            LegOrder("K-TEST", Side.BUY, 0.40, 100, platform="kalshi"),
        ),
        expected_profit_per_set=0.20,
        net_profit_per_set=0.20,
        max_sets=100,
        gross_profit=20.0,
        estimated_gas_cost=0.005,
        net_profit=19.995,
        roi_pct=25.0,
        required_capital=80.0,
    )


class TestPaperExecution:
    def test_paper_cross_platform(self):
        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), MagicMock(),
            opp, size=50.0, paper_trading=True,
        )
        assert result.fully_filled is True
        assert len(result.order_ids) == 2
        assert all("paper_xp_" in oid for oid in result.order_ids)
        assert result.net_pnl > 0

    def test_paper_preserves_opportunity(self):
        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), MagicMock(),
            opp, size=50.0, paper_trading=True,
        )
        assert result.opportunity is opp


class TestLiveExecution:
    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_both_sides_fill(self, mock_create, mock_post):
        """Both Kalshi and PM fill -> fully_filled=True."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = {"orderID": "pm_o1", "status": "matched"}

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k_o1", "status": "executed"},
        }

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client,
            opp, size=50.0, paper_trading=False,
        )

        assert result.fully_filled is True
        assert len(result.order_ids) == 2
        assert "k_o1" in result.order_ids
        assert "pm_o1" in result.order_ids
        assert result.net_pnl > 0

    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_kalshi_fails_aborts_early(self, mock_create, mock_post):
        """If Kalshi fails, should not place PM order."""
        kalshi_client = MagicMock()
        kalshi_client.place_order.side_effect = Exception("Kalshi API down")

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client,
            opp, size=50.0, paper_trading=False,
        )

        assert result.fully_filled is False
        # PM order should not have been attempted
        mock_create.assert_not_called()
        mock_post.assert_not_called()

    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_pm_fails_unwinds_kalshi(self, mock_create, mock_post):
        """If PM fails after Kalshi fills, should unwind Kalshi."""
        mock_create.return_value = MagicMock()
        mock_post.side_effect = Exception("PM CLOB error")

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k_o1", "status": "executed"},
        }

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client,
            opp, size=50.0, paper_trading=False,
        )

        assert result.fully_filled is False
        # Should have attempted to unwind Kalshi
        assert kalshi_client.place_order.call_count >= 2  # 1 original + 1 unwind

    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_pm_open_status_triggers_unwind(self, mock_create, mock_post):
        """PM returning 'open' status (not filled) should trigger unwind."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = {"orderID": "pm_o1", "status": "open"}

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k_o1", "status": "executed"},
        }

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client,
            opp, size=50.0, paper_trading=False,
        )

        assert result.fully_filled is False

    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_fractional_size_normalized_to_whole_contracts(self, mock_create, mock_post):
        """Kalshi requires integer contracts, so PM leg must use the same integer size."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = {"orderID": "pm_o1", "status": "matched"}

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k_o1", "status": "executed"},
        }

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client,
            opp, size=10.7, paper_trading=False,
        )

        assert result.fully_filled is True
        kalshi_kwargs = kalshi_client.place_order.call_args_list[0].kwargs
        assert kalshi_kwargs["count"] == 10
        pm_kwargs = mock_create.call_args_list[0].kwargs
        assert pm_kwargs["size"] == pytest.approx(10.0)

    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_sub_one_contract_size_skips_execution(self, mock_create, mock_post):
        """If normalized contract size is zero, execution should short-circuit safely."""
        kalshi_client = MagicMock()
        opp = _make_cross_opp()

        result = execute_cross_platform(
            MagicMock(), kalshi_client,
            opp, size=0.7, paper_trading=False,
        )

        assert result.fully_filled is False
        kalshi_client.place_order.assert_not_called()
        mock_create.assert_not_called()
        mock_post.assert_not_called()


class TestUnwindKalshi:
    def test_successful_unwind(self):
        """Should place opposite order on Kalshi."""
        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {"order": {"order_id": "uw1"}}

        legs = [LegOrder("K-TEST", Side.BUY, 0.40, 100, platform="kalshi")]
        _unwind_kalshi(kalshi_client, legs, 50.0)

        kalshi_client.place_order.assert_called_once()
        call_kwargs = kalshi_client.place_order.call_args.kwargs
        assert call_kwargs["action"] == "sell"
        assert call_kwargs["type"] == "market"

    def test_unwind_failure_raises(self):
        """Failed unwind should raise CrossPlatformUnwindFailed."""
        kalshi_client = MagicMock()
        kalshi_client.place_order.side_effect = Exception("API error")

        legs = [LegOrder("K-TEST", Side.BUY, 0.40, 100, platform="kalshi")]
        with pytest.raises(CrossPlatformUnwindFailed, match="Failed to unwind"):
            _unwind_kalshi(kalshi_client, legs, 50.0)


class TestKalshiSideMapping:
    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_buy_leg_uses_yes_side_buy_action(self, mock_create, mock_post):
        """BUY leg should send side='yes', action='buy'."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = {"orderID": "pm1", "status": "matched"}

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k1", "status": "executed"},
        }

        opp = _make_cross_opp()
        execute_cross_platform(MagicMock(), kalshi_client, opp, size=10.0, paper_trading=False)

        # Kalshi leg is BUY 0.40
        call_kwargs = kalshi_client.place_order.call_args_list[0].kwargs
        assert call_kwargs["side"] == "yes"
        assert call_kwargs["action"] == "buy"
        assert call_kwargs["yes_price"] == 40  # dollars_to_cents(0.40)

    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_sell_leg_uses_yes_side_sell_action(self, mock_create, mock_post):
        """SELL leg should send side='yes', action='sell' (not side='no')."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = {"orderID": "pm1", "status": "matched"}

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k1", "status": "executed"},
        }

        sell_opp = Opportunity(
            type=OpportunityType.CROSS_PLATFORM_ARB,
            event_id="e1",
            legs=(
                LegOrder("pm_yes", Side.BUY, 0.30, 100, platform="polymarket"),
                LegOrder("K-TEST", Side.SELL, 0.70, 100, platform="kalshi"),
            ),
            expected_profit_per_set=0.30,
            net_profit_per_set=0.30,
            max_sets=100,
            gross_profit=30.0,
            estimated_gas_cost=0.005,
            net_profit=29.995,
            roi_pct=30.0,
            required_capital=100.0,
        )

        execute_cross_platform(MagicMock(), kalshi_client, sell_opp, size=10.0, paper_trading=False)

        call_kwargs = kalshi_client.place_order.call_args_list[0].kwargs
        assert call_kwargs["side"] == "yes"
        assert call_kwargs["action"] == "sell"
        assert call_kwargs["yes_price"] == 70


class TestRestingOrderPolling:
    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    @patch("executor.cross_platform._wait_for_kalshi_fill")
    def test_resting_then_filled_proceeds(self, mock_wait, mock_create, mock_post):
        """Resting order that fills during polling should proceed to PM leg."""
        mock_wait.return_value = True
        mock_create.return_value = MagicMock()
        mock_post.return_value = {"orderID": "pm1", "status": "matched"}

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k1", "status": "resting"},
        }

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client, opp, size=10.0, paper_trading=False,
        )

        assert result.fully_filled is True
        mock_wait.assert_called_once()

    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    @patch("executor.cross_platform._wait_for_kalshi_fill")
    def test_resting_timeout_cancels_and_aborts(self, mock_wait, mock_create, mock_post):
        """Resting order that doesn't fill should be cancelled; no PM leg."""
        mock_wait.return_value = False

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k1", "status": "resting"},
        }

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client, opp, size=10.0, paper_trading=False,
        )

        assert result.fully_filled is False
        kalshi_client.cancel_order.assert_called_with("k1")
        mock_create.assert_not_called()
        mock_post.assert_not_called()


class TestUnwindLossTracking:
    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    def test_pm_failure_tracks_negative_pnl(self, mock_create, mock_post):
        """When PM fails after Kalshi fills, net_pnl should be negative."""
        mock_create.return_value = MagicMock()
        mock_post.side_effect = Exception("PM error")

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k1", "status": "executed"},
        }

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client, opp, size=50.0, paper_trading=False,
        )

        assert result.fully_filled is False
        assert result.net_pnl < 0  # Unwind loss tracked
        # Unwind loss = _UNWIND_LOSS_PER_CONTRACT * size = 0.02 * 50 = 1.0
        assert result.net_pnl == pytest.approx(-1.0)

    def test_unwind_returns_loss_estimate(self):
        """_unwind_kalshi should return float loss, not None."""
        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {"order": {"order_id": "uw1"}}

        legs = [LegOrder("K-TEST", Side.BUY, 0.40, 100, platform="kalshi")]
        loss = _unwind_kalshi(kalshi_client, legs, 50.0)

        assert isinstance(loss, float)
        assert loss == pytest.approx(1.0)  # 0.02 * 50


class TestDeadlineExceeded:
    @patch("executor.cross_platform.post_order")
    @patch("executor.cross_platform.create_limit_order")
    @patch("executor.cross_platform.time")
    def test_deadline_exceeded_unwinds(self, mock_time, mock_create, mock_post):
        """If deadline exceeded after Kalshi fill, should unwind without PM leg."""
        # Start at t=0, deadline at t=5, time check at t=6
        mock_time.time.side_effect = [0.0, 6.0, 6.0, 6.1]
        mock_time.sleep = MagicMock()

        kalshi_client = MagicMock()
        kalshi_client.place_order.return_value = {
            "order": {"order_id": "k1", "status": "executed"},
        }

        opp = _make_cross_opp()
        result = execute_cross_platform(
            MagicMock(), kalshi_client, opp, size=10.0,
            paper_trading=False, deadline_sec=5.0,
        )

        assert result.fully_filled is False
        assert result.net_pnl < 0
        mock_create.assert_not_called()
        mock_post.assert_not_called()


class TestEngineDispatch:
    def test_cross_platform_dispatch(self):
        """execute_opportunity should dispatch CROSS_PLATFORM_ARB correctly."""
        from executor.engine import execute_opportunity

        opp = _make_cross_opp()
        result = execute_opportunity(
            MagicMock(), opp, size=50.0,
            paper_trading=True,
            kalshi_client=MagicMock(),
        )
        assert result.fully_filled is True
        assert len(result.order_ids) == 2

    def test_cross_platform_no_kalshi_client_raises(self):
        """Should raise if kalshi_client not provided for CROSS_PLATFORM_ARB."""
        from executor.engine import execute_opportunity

        opp = _make_cross_opp()
        with pytest.raises(ValueError, match="kalshi_client required"):
            execute_opportunity(
                MagicMock(), opp, size=50.0,
                paper_trading=False,
                kalshi_client=None,
            )
