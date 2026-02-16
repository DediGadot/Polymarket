"""
Integration tests for presigner + execution engine wiring.
Verifies that OrderPresigner is correctly used by execute_opportunity()
and that fallback to create_limit_order() works when presigner is disabled.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock, call
import pytest

from executor.engine import execute_opportunity, _sign_order
from executor.presigner import OrderPresigner
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


def _make_latency_opp():
    return Opportunity(
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


class TestSignOrder:
    """Test the _sign_order helper directly."""

    @patch("executor.engine.create_limit_order")
    def test_no_presigner_uses_create_limit_order(self, mock_create):
        """When presigner is None, falls back to create_limit_order."""
        mock_create.return_value = MagicMock(name="signed_order")
        leg = LegOrder("tok1", Side.BUY, 0.45, 100)
        client = MagicMock()

        result = _sign_order(client, leg, size=50.0, neg_risk=False, presigner=None)

        mock_create.assert_called_once()
        assert result == mock_create.return_value

    def test_presigner_cache_hit_returns_cached(self):
        """When presigner has a cached order, return it without calling sign_fn."""
        cached_order = MagicMock(name="cached_signed_order")
        sign_fn = MagicMock(return_value=cached_order)
        presigner = OrderPresigner(sign_fn=sign_fn, max_age_sec=60.0)

        leg = LegOrder("tok1", Side.BUY, 0.45, 100, tick_size="0.01")

        # First call: cache miss -> signs
        result1 = _sign_order(MagicMock(), leg, size=50.0, neg_risk=False, presigner=presigner)
        assert sign_fn.call_count == 1

        # Second call: cache hit -> no additional sign
        result2 = _sign_order(MagicMock(), leg, size=50.0, neg_risk=False, presigner=presigner)
        assert sign_fn.call_count == 1  # still 1
        assert result2 == cached_order

    @patch("executor.engine.create_limit_order")
    def test_presigner_returns_none_falls_back(self, mock_create):
        """When presigner.get_or_sign() returns None, fall back to create_limit_order."""
        presigner = OrderPresigner(sign_fn=None)  # no sign_fn -> returns None
        mock_create.return_value = MagicMock(name="fallback_signed")
        leg = LegOrder("tok1", Side.BUY, 0.45, 100)
        client = MagicMock()

        result = _sign_order(client, leg, size=50.0, neg_risk=False, presigner=presigner)

        mock_create.assert_called_once()
        assert result == mock_create.return_value


class TestPresignerWithBinaryExecution:
    @patch("executor.engine.post_orders")
    def test_presigner_used_for_binary(self, mock_post):
        """Binary execution should use presigner when provided."""
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        signed_order = MagicMock(name="presigned")
        sign_fn = MagicMock(return_value=signed_order)
        presigner = OrderPresigner(sign_fn=sign_fn, max_age_sec=60.0)

        opp = _make_binary_opp()
        result = execute_opportunity(
            MagicMock(), opp, size=50.0,
            paper_trading=False, presigner=presigner,
        )

        assert result.fully_filled is True
        # sign_fn was called for both legs (cache misses on first call)
        assert sign_fn.call_count == 2

    @patch("executor.engine.post_orders")
    def test_presigner_cache_hit_no_redundant_sign(self, mock_post):
        """Second execution of same opp should use cached presigned orders."""
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        signed_order = MagicMock(name="presigned")
        sign_fn = MagicMock(return_value=signed_order)
        presigner = OrderPresigner(sign_fn=sign_fn, max_age_sec=60.0)

        opp = _make_binary_opp()

        # First execution: cache miss
        execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False, presigner=presigner)
        assert sign_fn.call_count == 2

        # Second execution: cache hit (same tokens, prices, sizes)
        execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False, presigner=presigner)
        assert sign_fn.call_count == 2  # still 2 (no new signs)


class TestPresignerWithNegRiskExecution:
    @patch("executor.engine.post_orders")
    def test_presigner_used_for_negrisk(self, mock_post):
        """NegRisk execution should use presigner when provided."""
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
            {"orderID": "o3", "status": "matched"},
        ]

        signed_order = MagicMock(name="presigned")
        sign_fn = MagicMock(return_value=signed_order)
        presigner = OrderPresigner(sign_fn=sign_fn, max_age_sec=60.0)

        opp = _make_negrisk_opp()
        result = execute_opportunity(
            MagicMock(), opp, size=30.0,
            paper_trading=False, presigner=presigner,
        )

        assert result.fully_filled is True
        assert sign_fn.call_count == 3


class TestPresignerWithSingleLeg:
    @patch("executor.engine.post_order")
    def test_presigner_used_for_single_leg(self, mock_post):
        """Single-leg execution should use presigner when provided."""
        mock_post.return_value = {"orderID": "o1", "status": "matched"}

        signed_order = MagicMock(name="presigned")
        sign_fn = MagicMock(return_value=signed_order)
        presigner = OrderPresigner(sign_fn=sign_fn, max_age_sec=60.0)

        opp = _make_latency_opp()
        result = execute_opportunity(
            MagicMock(), opp, size=100.0,
            paper_trading=False, presigner=presigner,
        )

        assert result.fully_filled is True
        assert sign_fn.call_count == 1


class TestBackwardCompatibility:
    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_no_presigner_backward_compat(self, mock_create, mock_post):
        """Without presigner, execution should work exactly as before."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        opp = _make_binary_opp()
        result = execute_opportunity(
            MagicMock(), opp, size=50.0,
            paper_trading=False,
            # No presigner argument
        )

        assert result.fully_filled is True
        assert mock_create.call_count == 2
        mock_post.assert_called_once()

    def test_paper_trading_ignores_presigner(self):
        """Paper trading should not use presigner even if provided."""
        sign_fn = MagicMock()
        presigner = OrderPresigner(sign_fn=sign_fn)

        opp = _make_binary_opp()
        result = execute_opportunity(
            MagicMock(), opp, size=50.0,
            paper_trading=True, presigner=presigner,
        )

        assert result.fully_filled is True
        sign_fn.assert_not_called()


class TestPresignerStats:
    @patch("executor.engine.post_orders")
    def test_stats_track_hits_and_misses(self, mock_post):
        """Presigner stats should reflect cache hits and misses."""
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        signed_order = MagicMock(name="presigned")
        sign_fn = MagicMock(return_value=signed_order)
        presigner = OrderPresigner(sign_fn=sign_fn, max_age_sec=60.0)

        opp = _make_binary_opp()

        # First execution: 2 misses
        execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False, presigner=presigner)
        stats = presigner.stats
        assert stats["misses"] == 2
        assert stats["hits"] == 0

        # Second execution: 2 hits
        execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=False, presigner=presigner)
        stats = presigner.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(0.5)
