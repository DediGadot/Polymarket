"""
Integration test: full scan -> detect -> size -> execute pipeline with mocks.
Tests the complete flow without hitting real APIs.
"""

from unittest.mock import patch, MagicMock
import pytest

from config import Config
from scanner.binary import scan_binary_markets
from scanner.negrisk import scan_negrisk_events
from scanner.models import (
    Market,
    Event,
    OrderBook,
    PriceLevel,
    OpportunityType,
)
from executor.sizing import compute_position_size
from executor.safety import CircuitBreaker, verify_prices_fresh, verify_depth
from executor.engine import execute_opportunity
from monitor.pnl import PnLTracker


def _cfg():
    return Config(
        private_key="test_key",
        polymarket_profile_address="0xtest",
        min_profit_usd=0.01,
        min_roi_pct=0.1,
        max_exposure_per_trade=500,
        max_total_exposure=5000,
        paper_trading=True,
    )


def _make_book(token_id, bid_price, bid_size, ask_price, ask_size):
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(bid_price, bid_size),),
        asks=(PriceLevel(ask_price, ask_size),),
    )


class TestFullPipelineBinaryArb:
    """End-to-end: discover binary arb -> size -> paper execute -> track PnL."""

    @patch("scanner.binary.get_orderbooks")
    @patch("executor.safety.get_orderbooks")
    def test_complete_flow(self, mock_safety_books, mock_scan_books):
        cfg = _cfg()
        client = MagicMock()

        # Setup: binary market with clear arb (YES=0.40, NO=0.40, total=0.80)
        markets = [Market(
            condition_id="c1", question="Binary arb test?",
            yes_token_id="y1", no_token_id="n1",
            neg_risk=False, event_id="e1",
            min_tick_size="0.01", active=True, volume=50000,
        )]

        books = {
            "y1": _make_book("y1", 0.39, 200, 0.40, 200),
            "n1": _make_book("n1", 0.39, 200, 0.40, 200),
        }
        mock_scan_books.return_value = books
        mock_safety_books.return_value = books

        # Step 1: Scan
        opps = scan_binary_markets(
            client, markets,
            cfg.min_profit_usd, cfg.min_roi_pct,
            cfg.gas_per_order, cfg.gas_price_gwei,
        )
        assert len(opps) >= 1
        opp = opps[0]
        assert opp.type == OpportunityType.BINARY_REBALANCE
        assert opp.expected_profit_per_set == pytest.approx(0.20, abs=0.01)

        # Step 2: Safety checks
        verify_prices_fresh(client, opp)
        verify_depth(client, opp)

        # Step 3: Size
        size = compute_position_size(
            opp,
            bankroll=cfg.max_total_exposure,
            max_exposure_per_trade=cfg.max_exposure_per_trade,
            max_total_exposure=cfg.max_total_exposure,
            current_exposure=0,
        )
        assert size > 0

        # Step 4: Execute (paper)
        result = execute_opportunity(client, opp, size, paper_trading=True)
        assert result.fully_filled is True
        assert result.net_pnl > 0

        # Step 5: Track PnL
        pnl = PnLTracker(ledger_path="/dev/null")
        pnl.record(result)
        assert pnl.total_pnl > 0
        assert pnl.total_trades == 1
        assert pnl.win_rate == 100.0


class TestFullPipelineNegRiskArb:
    """End-to-end: discover negrisk arb -> size -> paper execute -> track PnL."""

    @patch("scanner.negrisk.get_orderbooks")
    @patch("executor.safety.get_orderbooks")
    def test_complete_flow(self, mock_safety_books, mock_scan_books):
        cfg = _cfg()
        client = MagicMock()

        # Setup: 4-outcome event, each YES at 0.20 = total 0.80
        markets = [
            Market(f"c{i}", f"Outcome {i}?", f"y{i}", f"n{i}", True, "e1", "0.01", True)
            for i in range(4)
        ]
        event = Event(event_id="e1", title="4-way race", markets=tuple(markets), neg_risk=True)

        books = {f"y{i}": _make_book(f"y{i}", 0.19, 150, 0.20, 150) for i in range(4)}
        mock_scan_books.return_value = books
        mock_safety_books.return_value = books

        # Scan
        opps = scan_negrisk_events(
            client, [event],
            cfg.min_profit_usd, cfg.min_roi_pct,
            cfg.gas_per_order, cfg.gas_price_gwei,
        )
        assert len(opps) >= 1
        opp = opps[0]
        assert opp.type == OpportunityType.NEGRISK_REBALANCE
        assert opp.expected_profit_per_set == pytest.approx(0.20, abs=0.01)
        assert len(opp.legs) == 4

        # Safety
        verify_prices_fresh(client, opp)
        verify_depth(client, opp)

        # Size
        size = compute_position_size(
            opp,
            bankroll=cfg.max_total_exposure,
            max_exposure_per_trade=cfg.max_exposure_per_trade,
            max_total_exposure=cfg.max_total_exposure,
            current_exposure=0,
        )
        assert size > 0

        # Execute (paper)
        result = execute_opportunity(client, opp, size, paper_trading=True)
        assert result.fully_filled is True
        assert len(result.order_ids) == 4

        # PnL
        pnl = PnLTracker(ledger_path="/dev/null")
        pnl.record(result)
        assert pnl.total_pnl > 0


class TestCircuitBreakerIntegration:
    """Test that circuit breaker properly halts after repeated losses."""

    def test_breaker_halts_pipeline(self):
        breaker = CircuitBreaker(
            max_loss_per_hour=5.0,
            max_loss_per_day=10.0,
            max_consecutive_failures=3,
        )
        pnl = PnLTracker(ledger_path="/dev/null")

        from executor.safety import CircuitBreakerTripped

        with pytest.raises(CircuitBreakerTripped):
            for i in range(10):
                result = MagicMock()
                result.net_pnl = -2.0
                result.opportunity = MagicMock()
                result.opportunity.required_capital = 10.0
                result.opportunity.type.value = "binary_rebalance"
                result.opportunity.event_id = f"e{i}"
                result.opportunity.legs = ()
                result.fill_prices = [0.50]
                result.fill_sizes = [10.0]
                result.fees = 0.0
                result.gas_cost = 0.01
                result.execution_time_ms = 50.0
                result.fully_filled = False
                result.order_ids = [f"o{i}"]
                result.timestamp = 0.0

                breaker.record_trade(-2.0)


class TestNoArbScenario:
    """Test that correctly-priced markets produce no opportunities."""

    @patch("scanner.binary.get_orderbooks")
    @patch("scanner.negrisk.get_orderbooks")
    def test_no_opps_in_fair_markets(self, mock_nr_books, mock_bin_books):
        client = MagicMock()

        # Binary: YES=0.55, NO=0.45, total=1.00
        binary_markets = [Market(
            "c1", "Fair binary?", "y1", "n1", False, "e1", "0.01", True,
        )]
        mock_bin_books.return_value = {
            "y1": _make_book("y1", 0.54, 500, 0.55, 500),
            "n1": _make_book("n1", 0.44, 500, 0.45, 500),
        }

        bin_opps = scan_binary_markets(
            client, binary_markets, 0.50, 2.0, 150000, 30.0,
        )
        assert bin_opps == []

        # NegRisk: 3 outcomes at 0.35, 0.35, 0.30 = 1.00
        nr_markets = [
            Market(f"c{i}", f"Q{i}", f"y{i}", f"n{i}", True, "e2", "0.01", True)
            for i in range(3)
        ]
        event = Event("e2", "Fair 3-way", tuple(nr_markets), True)
        mock_nr_books.return_value = {
            "y0": _make_book("y0", 0.34, 200, 0.35, 200),
            "y1": _make_book("y1", 0.34, 200, 0.35, 200),
            "y2": _make_book("y2", 0.29, 200, 0.30, 200),
        }

        nr_opps = scan_negrisk_events(
            client, [event], 0.50, 2.0, 150000, 30.0,
        )
        assert nr_opps == []
