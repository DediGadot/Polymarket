"""
Tests for critical untested paths in executor/engine.py (task #23).
"""

import pytest
from unittest.mock import MagicMock


class TestFilledSizeFromResponse:
    """Tests for _filled_size_from_response critical path."""

    def test_returns_zero_for_empty_response(self):
        """Empty response should return 0.0."""
        from executor.engine import _filled_size_from_response
        result = _filled_size_from_response({}, requested_size=10.0)
        assert result == 0.0

    def test_extracts_known_fields(self):
        """Should extract size from known field names."""
        from executor.engine import _filled_size_from_response

        for field in ["filled_size", "filledSize", "size_filled", "sizeFilled"]:
            resp = {field: 5.0}
            result = _filled_size_from_response(resp, requested_size=10.0)
            assert result == 5.0, f"Failed to extract {field}"

    def test_prefers_full_filled(self):
        """Prefer filled over partial."""
        from executor.engine import _filled_size_from_response

        resp = {
            "filled_size": 3.0,
            "partial_filled": 5.0,
        }
        result = _filled_size_from_response(resp, requested_size=10.0)
        # Should return filled_size (3.0) not partial (5.0)
        assert result == 3.0


class TestBuildScoringContexts:
    """Tests for _build_scoring_contexts in run.py."""

    def test_empty_opportunities_returns_empty_list(self):
        """Empty opportunities list should return empty contexts."""
        from run import _build_scoring_contexts

        mock_book_cache = MagicMock()
        mock_book_cache.get_book.return_value = None

        result = _build_scoring_contexts(
            [],
            mock_book_cache,
            all_markets=[],
            target_size=10.0
        )
        assert result == []

    def test_single_opportunity_creates_single_context(self):
        """Should create one context per opportunity."""
        from run import _build_scoring_contexts
        from scanner.models import Opportunity, OpportunityType, LegOrder, Side

        opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="evt1",
            legs=(LegOrder(token_id="t1", side=Side.BUY, price=0.5, size=10.0),),
            expected_profit_per_set=0.05,
            net_profit_per_set=0.04,
            max_sets=10.0,
            gross_profit=0.5,
            estimated_gas_cost=0.01,
            net_profit=0.4,
            roi_pct=2.0,
            required_capital=5.0,
        )

        mock_book_cache = MagicMock()
        mock_book_cache.get_book.return_value = MagicMock(
            best_ask=MagicMock(price=0.52),
            best_bid=MagicMock(price=0.48),
        )

        result = _build_scoring_contexts(
            [opp],
            mock_book_cache,
            all_markets=[],
            target_size=10.0
        )
        assert len(result) == 1
        # Default values should be set when book unavailable
        assert result[0].market_volume == 0.0
