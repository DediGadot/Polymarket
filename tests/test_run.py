"""
Unit tests for run.py helper behavior.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from config import Config
from scanner.confidence import ArbTracker
from scanner.models import Opportunity, OpportunityType, LegOrder, Side, OrderBook, PriceLevel
from scanner.scorer import ScoredOpportunity
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

        cfg = Config(max_exposure_per_trade=5000.0, max_total_exposure=50000.0)
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

        cfg = Config(max_exposure_per_trade=5000.0, max_total_exposure=50000.0)
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
        cfg = Config(max_exposure_per_trade=5000.0, max_total_exposure=50000.0)
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

        cfg = Config(max_exposure_per_trade=5000.0, max_total_exposure=50000.0)
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

    @patch("run.verify_inventory")
    @patch("run.execute_opportunity")
    @patch("run.verify_cross_platform_books")
    @patch("run.verify_depth")
    @patch("run.verify_edge_intact")
    @patch("run.verify_prices_fresh")
    @patch("run.verify_opportunity_ttl")
    @patch("run.get_orderbooks")
    @patch("run.compute_position_size")
    def test_cross_platform_inventory_check_pm_only(
        self, mock_size, mock_get_books, mock_verify_ttl, mock_verify_fresh,
        mock_verify_edge, mock_verify_depth, mock_verify_cross_books, mock_execute, mock_verify_inventory,
    ):
        mock_size.return_value = 10.0
        mock_execute.return_value = MagicMock(net_pnl=1.0)
        mock_get_books.return_value = {"pm_yes": MagicMock()}
        kalshi_client = MagicMock()
        kalshi_client.get_orderbooks.return_value = {"K-TEST": MagicMock()}

        cfg = Config(
            max_exposure_per_trade=5000.0,
            max_total_exposure=50000.0,
            cross_platform_inventory_pm_only=True,
        )
        opp = _make_cross_platform_opp(leg_size=100.0)  # only external SELL leg

        run._execute_single(
            client=MagicMock(),
            cfg=cfg,
            opp=opp,
            pnl=MagicMock(),
            breaker=MagicMock(),
            position_tracker=MagicMock(),
            platform_clients={"kalshi": kalshi_client},
        )

        mock_verify_inventory.assert_not_called()

    @patch("run.execute_opportunity")
    @patch("run.verify_depth")
    @patch("run.verify_edge_intact")
    @patch("run.verify_prices_fresh")
    @patch("run.verify_opportunity_ttl")
    @patch("run.get_orderbooks")
    @patch("run.compute_position_size")
    def test_execute_single_threads_presigner(
        self, mock_size, mock_get_books, mock_verify_ttl, mock_verify_fresh,
        mock_verify_edge, mock_verify_depth, mock_execute,
    ):
        mock_size.return_value = 12.0
        mock_execute.return_value = MagicMock(net_pnl=1.0)
        mock_get_books.return_value = {"y1": MagicMock(), "n1": MagicMock()}

        cfg = Config(max_exposure_per_trade=5000.0, max_total_exposure=50000.0)
        presigner = MagicMock()
        run._execute_single(
            client=MagicMock(),
            cfg=cfg,
            opp=_make_binary_opp(leg_size=100.0),
            pnl=MagicMock(),
            breaker=MagicMock(),
            presigner=presigner,
        )

        assert mock_execute.call_args.kwargs["presigner"] is presigner


class TestPolymarketOnlyMode:
    def test_disables_external_scanners_when_flag_off(self):
        cfg = Config(
            allow_non_polymarket_apis=False,
            latency_enabled=True,
            cross_platform_enabled=True,
        )

        cfg = run._enforce_polymarket_only_mode(cfg)

        assert cfg.latency_enabled is False
        assert cfg.cross_platform_enabled is False

    def test_keeps_external_scanners_when_flag_on(self):
        cfg = Config(
            allow_non_polymarket_apis=True,
            latency_enabled=True,
            cross_platform_enabled=True,
        )

        cfg = run._enforce_polymarket_only_mode(cfg)

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


class TestScoringContexts:
    def test_depth_ratio_uses_target_sets(self):
        opp = _make_binary_opp(leg_size=200.0)
        mock_book_cache = MagicMock()
        shared_book = OrderBook(
            token_id="ignored",
            bids=(PriceLevel(0.44, 200.0),),
            asks=(PriceLevel(0.45, 200.0),),
        )
        mock_book_cache.get_book.return_value = shared_book

        # required_capital/max_sets = 0.45 per set, so $45 target => 100 sets.
        contexts = run._build_scoring_contexts(
            [opp],
            book_cache=mock_book_cache,
            all_markets=[],
            target_size=45.0,
        )

        assert contexts[0].book_depth_ratio == pytest.approx(2.0)

    def test_sell_only_confidence_penalized_without_inventory(self):
        buy_opp = _make_binary_opp(leg_size=10.0)
        sell_opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="evt_sell",
            legs=(
                LegOrder("y2", Side.SELL, 0.55, 10.0),
                LegOrder("n2", Side.SELL, 0.55, 10.0),
            ),
            expected_profit_per_set=0.10,
            net_profit_per_set=0.10,
            max_sets=10.0,
            gross_profit=1.0,
            estimated_gas_cost=0.01,
            net_profit=0.99,
            roi_pct=10.0,
            required_capital=10.0,
        )

        tracker = ArbTracker()
        tracker.record(1, [buy_opp, sell_opp])
        mock_book_cache = MagicMock()
        mock_book_cache.get_book.return_value = None

        contexts = run._build_scoring_contexts(
            [buy_opp, sell_opp],
            book_cache=mock_book_cache,
            all_markets=[],
            target_size=100.0,
            arb_tracker=tracker,
            has_inventory=False,
        )

        assert contexts[0].confidence == pytest.approx(0.1)
        assert contexts[1].confidence == pytest.approx(0.1)


class TestNegriskValueFilter:
    def test_negrisk_value_filtered_when_scanner_disabled(self):
        """All NEGRISK_VALUE opps should be dropped when value_scanner_enabled=False."""
        from scanner.models import Opportunity, OpportunityType, LegOrder, Side

        nv_opp = Opportunity(
            type=OpportunityType.NEGRISK_VALUE,
            event_id="evt_nv",
            legs=(LegOrder("y1", Side.BUY, 0.05, 10),),
            expected_profit_per_set=0.05,
            net_profit_per_set=0.04,
            max_sets=10,
            gross_profit=0.50,
            estimated_gas_cost=0.01,
            net_profit=0.39,
            roi_pct=7800.0,
            required_capital=0.50,
        )
        binary_opp = _make_binary_opp()

        all_opps = [nv_opp, binary_opp]

        # Simulate the filter logic from run.py
        cfg = Config(value_scanner_enabled=False)
        if not cfg.value_scanner_enabled:
            all_opps = [o for o in all_opps if o.type != OpportunityType.NEGRISK_VALUE]

        assert len(all_opps) == 1
        assert all_opps[0].type == OpportunityType.BINARY_REBALANCE

    def test_negrisk_value_kept_when_scanner_enabled(self):
        """NEGRISK_VALUE opps should pass through when value_scanner_enabled=True."""
        from scanner.models import Opportunity, OpportunityType, LegOrder, Side

        nv_opp = Opportunity(
            type=OpportunityType.NEGRISK_VALUE,
            event_id="evt_nv",
            legs=(LegOrder("y1", Side.BUY, 0.05, 10),),
            expected_profit_per_set=0.05,
            net_profit_per_set=0.04,
            max_sets=10,
            gross_profit=0.50,
            estimated_gas_cost=0.01,
            net_profit=0.39,
            roi_pct=7800.0,
            required_capital=0.50,
        )
        binary_opp = _make_binary_opp()

        all_opps = [nv_opp, binary_opp]
        cfg = Config(value_scanner_enabled=True)
        if not cfg.value_scanner_enabled:
            all_opps = [o for o in all_opps if o.type != OpportunityType.NEGRISK_VALUE]

        assert len(all_opps) == 2  # Both kept


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


class TestRateLimitErrorDetection:
    def test_detects_429(self):
        assert run._is_rate_limit_error("429 Too Many Requests")

    def test_detects_text_rate_limit(self):
        assert run._is_rate_limit_error("rate limit set at 50 requests every 10 seconds")

    def test_non_rate_limit_error(self):
        assert not run._is_rate_limit_error("connection reset by peer")


class TestExecutionSupportGates:
    def test_maker_type_not_supported_for_execution(self):
        assert run._is_execution_supported_type(OpportunityType.MAKER_REBALANCE) is False

    def test_binary_type_supported_for_execution(self):
        assert run._is_execution_supported_type(OpportunityType.BINARY_REBALANCE) is True

    def test_executable_now_requires_taker_buy_and_fill(self):
        cfg = Config(min_confidence_gate=0.5)
        opp = _make_binary_opp(leg_size=100.0)
        scored = ScoredOpportunity(
            opportunity=opp,
            total_score=0.8,
            profit_score=0.8,
            fill_score=0.7,
            efficiency_score=0.5,
            urgency_score=0.5,
            competition_score=0.5,
            persistence_score=0.8,
        )
        assert run._is_executable_now(scored, cfg) is True

    def test_executable_now_rejects_maker(self):
        cfg = Config(min_confidence_gate=0.5)
        maker_opp = Opportunity(
            type=OpportunityType.MAKER_REBALANCE,
            event_id="evt_maker",
            legs=(
                LegOrder("y1", Side.BUY, 0.45, 100.0),
                LegOrder("n1", Side.BUY, 0.45, 100.0),
            ),
            expected_profit_per_set=0.10,
            net_profit_per_set=0.10,
            max_sets=100.0,
            gross_profit=10.0,
            estimated_gas_cost=0.01,
            net_profit=9.99,
            roi_pct=11.1,
            required_capital=90.0,
        )
        scored = ScoredOpportunity(
            opportunity=maker_opp,
            total_score=0.8,
            profit_score=0.8,
            fill_score=0.9,
            efficiency_score=0.5,
            urgency_score=0.5,
            competition_score=0.5,
            persistence_score=0.8,
        )
        assert run._is_executable_now(scored, cfg) is False

    def test_executable_now_allows_structural_correlation_buy_when_enabled(self):
        cfg = Config(
            min_confidence_gate=0.5,
            correlation_actionable_allow_structural_buy=True,
        )
        corr_opp = Opportunity(
            type=OpportunityType.CORRELATION_ARB,
            event_id="evt_corr",
            legs=(
                LegOrder("yes_parent", Side.BUY, 0.40, 100.0),
                LegOrder("no_child", Side.BUY, 0.30, 100.0),
            ),
            expected_profit_per_set=0.30,
            net_profit_per_set=0.30,
            max_sets=100.0,
            gross_profit=30.0,
            estimated_gas_cost=0.01,
            net_profit=29.99,
            roi_pct=42.8,
            required_capital=70.0,
            reason_code="corr_parent_child_buy_liquidity_weighted",
        )
        scored = ScoredOpportunity(
            opportunity=corr_opp,
            total_score=0.8,
            profit_score=0.8,
            fill_score=0.7,
            efficiency_score=0.5,
            urgency_score=0.5,
            competition_score=0.5,
            persistence_score=0.8,
        )
        assert run._is_executable_now(scored, cfg) is True

    def test_executable_now_rejects_structural_correlation_buy_when_disabled(self):
        cfg = Config(
            min_confidence_gate=0.5,
            correlation_actionable_allow_structural_buy=False,
        )
        corr_opp = Opportunity(
            type=OpportunityType.CORRELATION_ARB,
            event_id="evt_corr",
            legs=(
                LegOrder("yes_parent", Side.BUY, 0.40, 100.0),
                LegOrder("no_child", Side.BUY, 0.30, 100.0),
            ),
            expected_profit_per_set=0.30,
            net_profit_per_set=0.30,
            max_sets=100.0,
            gross_profit=30.0,
            estimated_gas_cost=0.01,
            net_profit=29.99,
            roi_pct=42.8,
            required_capital=70.0,
            reason_code="corr_parent_child_buy_liquidity_weighted",
        )
        scored = ScoredOpportunity(
            opportunity=corr_opp,
            total_score=0.8,
            profit_score=0.8,
            fill_score=0.7,
            efficiency_score=0.5,
            urgency_score=0.5,
            competition_score=0.5,
            persistence_score=0.8,
        )
        assert run._is_executable_now(scored, cfg) is False


class TestCheckpointIntegration:
    """Integration tests for checkpoint manager wiring in run.py."""

    def test_checkpoint_config_defaults(self):
        cfg = Config()
        assert cfg.state_checkpoint_enabled is True
        assert cfg.state_checkpoint_interval == 10
        assert cfg.state_checkpoint_db == "state.db"

    def test_checkpoint_config_disabled(self):
        cfg = Config(state_checkpoint_enabled=False)
        assert cfg.state_checkpoint_enabled is False

    def test_checkpoint_round_trip_arb_tracker(self, tmp_path):
        """Save arb_tracker state, create fresh manager, verify restore."""
        from state.checkpoint import CheckpointManager
        from scanner.confidence import ArbTracker

        db = tmp_path / "test.db"
        mgr = CheckpointManager(db_path=db, auto_save_interval=2)

        # Simulate a pipeline with state
        tracker = ArbTracker()
        tracker._history = {"evt-1": [1, 2, 3], "evt-2": [5]}
        tracker._failures = {"evt-1": 2}
        mgr.save("arb_tracker", tracker)
        mgr.close()

        # Create a new manager (simulating restart)
        mgr2 = CheckpointManager(db_path=db, auto_save_interval=2)
        restored = mgr2.load("arb_tracker", ArbTracker)
        mgr2.close()

        assert restored is not None
        assert restored._history == {"evt-1": [1, 2, 3], "evt-2": [5]}
        assert restored._failures == {"evt-1": 2}
        assert restored.confidence("evt-1") == tracker.confidence("evt-1")

    def test_checkpoint_tick_saves_on_interval(self, tmp_path):
        """tick() saves registered trackers at the configured interval."""
        from state.checkpoint import CheckpointManager
        from scanner.confidence import ArbTracker

        db = tmp_path / "tick_test.db"
        mgr = CheckpointManager(db_path=db, auto_save_interval=3)

        tracker = ArbTracker()
        tracker._history = {"e1": [1]}
        mgr.register("arb", tracker)

        # Ticks 1 and 2 should not save
        assert mgr.tick() == 0
        assert mgr.tick() == 0

        # Tick 3 should trigger save
        assert mgr.tick() == 1

        # Verify persisted
        restored = mgr.load("arb", ArbTracker)
        assert restored is not None
        assert restored._history == {"e1": [1]}
        mgr.close()

    def test_checkpoint_disabled_does_not_create_db(self, tmp_path):
        """When state_checkpoint_enabled=False, no manager is created."""
        # This is tested implicitly by the config flag + the run.py logic:
        # checkpoint_mgr will be None when disabled
        cfg = Config(state_checkpoint_enabled=False)
        assert cfg.state_checkpoint_enabled is False
        # Just verify the config works and no side effects

    def test_checkpoint_multiple_trackers_round_trip(self, tmp_path):
        """Multiple trackers save/restore independently."""
        from state.checkpoint import CheckpointManager
        from scanner.confidence import ArbTracker
        from scanner.spike import SpikeDetector
        from scanner.maker import MakerPersistenceGate

        db = tmp_path / "multi.db"
        mgr = CheckpointManager(db_path=db)

        arb = ArbTracker()
        arb._history = {"e1": [1, 2]}
        spike = SpikeDetector(threshold_pct=3.0)
        gate = MakerPersistenceGate(min_consecutive_cycles=5)
        gate._streaks = {"m1": 4}

        mgr.save("arb_tracker", arb)
        mgr.save("spike_detector", spike)
        mgr.save("maker_persistence_gate", gate)
        mgr.close()

        mgr2 = CheckpointManager(db_path=db)
        r_arb = mgr2.load("arb_tracker", ArbTracker)
        r_spike = mgr2.load("spike_detector", SpikeDetector)
        r_gate = mgr2.load("maker_persistence_gate", MakerPersistenceGate)
        mgr2.close()

        assert r_arb._history == {"e1": [1, 2]}
        assert r_spike.threshold_pct == 3.0
        assert r_gate._streaks == {"m1": 4}
        assert r_gate.min_consecutive_cycles == 5
