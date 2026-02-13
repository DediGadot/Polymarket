"""
Unit tests for run.py helper behavior.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from config import Config
from scanner.models import Opportunity, OpportunityType, LegOrder, Side
import run


def _make_binary_opp(leg_size: float = 100.0) -> Opportunity:
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id="evt1",
        legs=(
            LegOrder("y1", Side.BUY, 0.45, leg_size),
            LegOrder("n1", Side.BUY, 0.45, leg_size),
        ),
        expected_profit_per_set=0.10,
        net_profit_per_set=0.10,
        max_sets=leg_size,
        gross_profit=10.0,
        estimated_gas_cost=0.01,
        net_profit=9.99,
        roi_pct=11.1,
        required_capital=90.0,
    )


def _make_cross_platform_opp(leg_size: float = 100.0) -> Opportunity:
    return Opportunity(
        type=OpportunityType.CROSS_PLATFORM_ARB,
        event_id="evt_xp",
        legs=(
            LegOrder("pm_yes", Side.BUY, 0.40, leg_size, platform="polymarket"),
            LegOrder("K-TEST", Side.SELL, 0.60, leg_size, platform="kalshi"),
        ),
        expected_profit_per_set=0.20,
        net_profit_per_set=0.20,
        max_sets=leg_size,
        gross_profit=20.0,
        estimated_gas_cost=0.01,
        net_profit=19.99,
        roi_pct=25.0,
        required_capital=80.0,
    )


class TestExecuteSingle:
    @patch("run.execute_opportunity")
    @patch("run.verify_depth")
    @patch("run.verify_edge_intact")
    @patch("run.verify_prices_fresh")
    @patch("run.verify_opportunity_ttl")
    @patch("run.get_orderbooks")
    @patch("run.compute_position_size")
    def test_depth_check_uses_sized_legs(
        self, mock_size, mock_get_books, mock_verify_ttl, mock_verify_fresh,
        mock_verify_edge, mock_verify_depth, mock_execute,
    ):
        mock_size.return_value = 10.0
        mock_execute.return_value = MagicMock(net_pnl=1.0)
        mock_get_books.return_value = {
            "y1": MagicMock(),
            "n1": MagicMock(),
        }

        cfg = Config(max_exposure_per_trade=500.0, max_total_exposure=5000.0)
        opp = _make_binary_opp(leg_size=100.0)
        pnl = MagicMock()
        breaker = MagicMock()

        run._execute_single(
            client=MagicMock(),
            cfg=cfg,
            opp=opp,
            pnl=pnl,
            breaker=breaker,
            gas_oracle=None,
        )

        checked_opp = mock_verify_depth.call_args.args[0]
        assert checked_opp.legs[0].size == pytest.approx(10.0)
        assert checked_opp.legs[1].size == pytest.approx(10.0)

    @patch("run.execute_opportunity")
    @patch("run.verify_gas_reasonable")
    @patch("run.verify_depth")
    @patch("run.verify_edge_intact")
    @patch("run.verify_prices_fresh")
    @patch("run.verify_opportunity_ttl")
    @patch("run.get_orderbooks")
    @patch("run.compute_position_size")
    def test_gas_check_receives_execution_size(
        self, mock_size, mock_get_books, mock_verify_ttl, mock_verify_fresh,
        mock_verify_edge, mock_verify_depth, mock_verify_gas, mock_execute,
    ):
        mock_size.return_value = 10.0
        mock_execute.return_value = MagicMock(net_pnl=1.0)
        mock_get_books.return_value = {"y1": MagicMock(), "n1": MagicMock()}

        cfg = Config(max_exposure_per_trade=500.0, max_total_exposure=5000.0)
        opp = _make_binary_opp(leg_size=100.0)
        pnl = MagicMock()
        breaker = MagicMock()
        gas_oracle = MagicMock()

        run._execute_single(
            client=MagicMock(),
            cfg=cfg,
            opp=opp,
            pnl=pnl,
            breaker=breaker,
            gas_oracle=gas_oracle,
        )

        assert mock_verify_gas.call_args.kwargs["size"] == pytest.approx(10.0)

    @patch("run.get_orderbooks")
    def test_cross_platform_requires_platform_clients(self, mock_get_books):
        mock_get_books.side_effect = AssertionError("should not fetch PM books without platform clients")
        cfg = Config(max_exposure_per_trade=500.0, max_total_exposure=5000.0)
        opp = _make_cross_platform_opp(leg_size=100.0)

        with pytest.raises(run.SafetyCheckFailed, match="platform_clients required"):
            run._execute_single(
                client=MagicMock(),
                cfg=cfg,
                opp=opp,
                pnl=MagicMock(),
                breaker=MagicMock(),
                platform_clients=None,
            )

    @patch("run.execute_opportunity")
    @patch("run.verify_cross_platform_books")
    @patch("run.verify_gas_reasonable")
    @patch("run.verify_depth")
    @patch("run.verify_edge_intact")
    @patch("run.verify_prices_fresh")
    @patch("run.verify_opportunity_ttl")
    @patch("run.get_orderbooks")
    @patch("run.compute_position_size")
    def test_cross_platform_fetches_books_by_platform(
        self, mock_size, mock_get_books, mock_verify_ttl, mock_verify_fresh,
        mock_verify_edge, mock_verify_depth, mock_verify_gas, mock_verify_cross_books, mock_execute,
    ):
        mock_size.return_value = 10.0
        mock_execute.return_value = MagicMock(net_pnl=1.0)
        pm_books = {"pm_yes": MagicMock()}
        kalshi_books = {"K-TEST": MagicMock()}
        mock_get_books.return_value = pm_books
        kalshi_client = MagicMock()
        kalshi_client.get_orderbooks.return_value = kalshi_books

        cfg = Config(max_exposure_per_trade=500.0, max_total_exposure=5000.0)
        opp = _make_cross_platform_opp(leg_size=100.0)

        run._execute_single(
            client=MagicMock(),
            cfg=cfg,
            opp=opp,
            pnl=MagicMock(),
            breaker=MagicMock(),
            platform_clients={"kalshi": kalshi_client},
        )

        assert mock_get_books.call_args.args[1] == ["pm_yes"]
        kalshi_client.get_orderbooks.assert_called_once_with(["K-TEST"])


class TestPolymarketOnlyMode:
    def test_disables_external_scanners_when_flag_off(self):
        cfg = Config(
            allow_non_polymarket_apis=False,
            latency_enabled=True,
            cross_platform_enabled=True,
        )

        run._enforce_polymarket_only_mode(cfg)

        assert cfg.latency_enabled is False
        assert cfg.cross_platform_enabled is False

    def test_keeps_external_scanners_when_flag_on(self):
        cfg = Config(
            allow_non_polymarket_apis=True,
            latency_enabled=True,
            cross_platform_enabled=True,
        )

        run._enforce_polymarket_only_mode(cfg)

        assert cfg.latency_enabled is True
        assert cfg.cross_platform_enabled is True


class TestShutdownCleanup:
    @patch("run.cancel_all")
    def test_live_mode_cancels_open_orders(self, mock_cancel_all):
        cfg = Config(paper_trading=False)
        args = Namespace(scan_only=False, dry_run=False)

        run._cancel_open_orders_on_shutdown(MagicMock(), args, cfg)

        mock_cancel_all.assert_called_once()

    @patch("run.cancel_all")
    def test_scan_only_skips_cancellation(self, mock_cancel_all):
        cfg = Config(paper_trading=False)
        args = Namespace(scan_only=True, dry_run=False)

        run._cancel_open_orders_on_shutdown(MagicMock(), args, cfg)

        mock_cancel_all.assert_not_called()


class TestFormatDuration:
    def test_seconds(self):
        assert run._format_duration(30.5) == "30.5s"

    def test_minutes(self):
        result = run._format_duration(125.0)
        assert result == "2m 5s"

    def test_hours(self):
        result = run._format_duration(3725.0)
        assert result == "1h 2m"

    def test_zero(self):
        assert run._format_duration(0.0) == "0.0s"


class TestModeLabel:
    def test_dry_run(self):
        args = Namespace(dry_run=True, scan_only=True, live=False)
        cfg = Config()
        label = run._mode_label(args, cfg)
        assert "DRY-RUN" in label

    def test_scan_only(self):
        args = Namespace(dry_run=False, scan_only=True, live=False)
        cfg = Config()
        label = run._mode_label(args, cfg)
        assert "SCAN-ONLY" in label

    def test_paper_trading(self):
        args = Namespace(dry_run=False, scan_only=False, live=False)
        cfg = Config(paper_trading=True)
        label = run._mode_label(args, cfg)
        assert "PAPER" in label

    def test_live_trading(self):
        args = Namespace(dry_run=False, scan_only=False, live=True)
        cfg = Config(paper_trading=False)
        label = run._mode_label(args, cfg)
        assert "LIVE" in label


class TestWithSizedLegs:
    def test_resizes_legs(self):
        opp = _make_binary_opp(leg_size=100.0)
        sized = run._with_sized_legs(opp, 25.0)
        assert sized.legs[0].size == 25.0
        assert sized.legs[1].size == 25.0
        # Original unchanged (frozen dataclass)
        assert opp.legs[0].size == 100.0

    def test_preserves_other_fields(self):
        opp = _make_binary_opp(leg_size=100.0)
        sized = run._with_sized_legs(opp, 50.0)
        assert sized.event_id == opp.event_id
        assert sized.type == opp.type
        assert sized.net_profit == opp.net_profit
        assert sized.legs[0].token_id == "y1"
        assert sized.legs[0].price == 0.45


class TestSleepRemaining:
    @patch("run.time.sleep")
    def test_sleeps_remaining_time(self, mock_sleep):
        import time as real_time
        start = real_time.time() - 0.5  # started 0.5s ago
        run._sleep_remaining(start, interval=1.0, shutdown=False)
        if mock_sleep.called:
            slept = mock_sleep.call_args[0][0]
            assert slept > 0
            assert slept <= 1.0

    @patch("run.time.sleep")
    def test_no_sleep_on_shutdown(self, mock_sleep):
        import time as real_time
        run._sleep_remaining(real_time.time(), interval=10.0, shutdown=True)
        mock_sleep.assert_not_called()
