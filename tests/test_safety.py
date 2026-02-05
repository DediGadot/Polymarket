"""
Unit tests for executor/safety.py -- circuit breakers and pre-trade checks.
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from executor.safety import (
    CircuitBreaker,
    CircuitBreakerTripped,
    SafetyCheckFailed,
    verify_prices_fresh,
    verify_depth,
)
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
    OrderBook,
    PriceLevel,
)


def _make_opp(legs=None):
    if legs is None:
        legs = (
            LegOrder("y1", Side.BUY, 0.45, 100),
            LegOrder("n1", Side.BUY, 0.45, 100),
        )
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id="e1",
        legs=legs,
        expected_profit_per_set=0.10,
        max_sets=100,
        gross_profit=10.0,
        estimated_gas_cost=0.01,
        net_profit=9.99,
        roi_pct=11.1,
        required_capital=90.0,
    )


def _make_book(token_id, bid_price, bid_size, ask_price, ask_size):
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(bid_price, bid_size),),
        asks=(PriceLevel(ask_price, ask_size),),
    )


class TestCircuitBreaker:
    def test_no_trip_on_wins(self):
        cb = CircuitBreaker(
            max_loss_per_hour=50, max_loss_per_day=200, max_consecutive_failures=5,
        )
        for _ in range(20):
            cb.record_trade(1.0)  # all wins

    def test_trip_on_hourly_loss(self):
        cb = CircuitBreaker(
            max_loss_per_hour=10, max_loss_per_day=200, max_consecutive_failures=100,
        )
        with pytest.raises(CircuitBreakerTripped, match="Hourly loss"):
            for _ in range(20):
                cb.record_trade(-1.0)

    def test_trip_on_daily_loss(self):
        cb = CircuitBreaker(
            max_loss_per_hour=1000, max_loss_per_day=10, max_consecutive_failures=100,
        )
        with pytest.raises(CircuitBreakerTripped, match="Daily loss"):
            for _ in range(20):
                cb.record_trade(-1.0)

    def test_trip_on_consecutive_failures(self):
        cb = CircuitBreaker(
            max_loss_per_hour=1000, max_loss_per_day=1000, max_consecutive_failures=3,
        )
        with pytest.raises(CircuitBreakerTripped, match="Consecutive failures"):
            for _ in range(5):
                cb.record_trade(-0.01)

    def test_consecutive_reset_on_win(self):
        cb = CircuitBreaker(
            max_loss_per_hour=1000, max_loss_per_day=1000, max_consecutive_failures=3,
        )
        cb.record_trade(-0.01)
        cb.record_trade(-0.01)
        cb.record_trade(1.0)  # resets counter
        cb.record_trade(-0.01)
        cb.record_trade(-0.01)
        # Should not trip because we had a win in between

    def test_old_losses_pruned(self):
        cb = CircuitBreaker(
            max_loss_per_hour=10, max_loss_per_day=200, max_consecutive_failures=100,
        )
        # Inject old losses that should be pruned
        old_time = time.time() - 7200  # 2 hours ago
        cb._hourly_losses = [(old_time, 100.0)]
        # Should not trip because old losses are pruned
        cb.record_trade(-1.0)


class TestVerifyPricesFresh:
    @patch("executor.safety.get_orderbooks")
    def test_prices_within_tolerance(self, mock_books):
        opp = _make_opp()
        mock_books.return_value = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 200),
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        # Should not raise
        verify_prices_fresh(MagicMock(), opp)

    @patch("executor.safety.get_orderbooks")
    def test_ask_moved_up_raises(self, mock_books):
        opp = _make_opp()
        mock_books.return_value = {
            "y1": _make_book("y1", 0.50, 200, 0.52, 200),  # moved from 0.45 to 0.52
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        with pytest.raises(SafetyCheckFailed, match="Ask moved"):
            verify_prices_fresh(MagicMock(), opp)

    @patch("executor.safety.get_orderbooks")
    def test_missing_book_raises(self, mock_books):
        opp = _make_opp()
        mock_books.return_value = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 200),
            # n1 missing
        }
        with pytest.raises(SafetyCheckFailed, match="No orderbook"):
            verify_prices_fresh(MagicMock(), opp)

    @patch("executor.safety.get_orderbooks")
    def test_sell_leg_bid_moved_down_raises(self, mock_books):
        legs = (
            LegOrder("y1", Side.SELL, 0.55, 100),
            LegOrder("n1", Side.SELL, 0.55, 100),
        )
        opp = _make_opp(legs=legs)
        mock_books.return_value = {
            "y1": _make_book("y1", 0.50, 200, 0.56, 200),  # bid moved from 0.55 to 0.50
            "n1": _make_book("n1", 0.55, 200, 0.56, 200),
        }
        with pytest.raises(SafetyCheckFailed, match="Bid moved"):
            verify_prices_fresh(MagicMock(), opp)


class TestVerifyDepth:
    @patch("executor.safety.get_orderbooks")
    def test_sufficient_depth(self, mock_books):
        opp = _make_opp()
        mock_books.return_value = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 200),
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        verify_depth(MagicMock(), opp)  # should not raise

    @patch("executor.safety.get_orderbooks")
    def test_insufficient_depth_raises(self, mock_books):
        opp = _make_opp()
        mock_books.return_value = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 50),  # only 50 available, need 100
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        with pytest.raises(SafetyCheckFailed, match="Insufficient ask depth"):
            verify_depth(MagicMock(), opp)
