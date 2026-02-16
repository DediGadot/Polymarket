"""
Integration tests for correlation scanner pipeline integration.
Verifies: detection → scoring → ranking → execution dispatch.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from scanner.correlation import CorrelationScanner
from scanner.models import (
    Event,
    LegOrder,
    Market,
    Opportunity,
    OpportunityType,
    OrderBook,
    PriceLevel,
    Side,
)
from scanner.scorer import (
    ScoringContext,
    rank_opportunities,
    score_opportunity,
)
from executor.engine import execute_opportunity


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_market(
    condition_id: str,
    yes_token_id: str,
    no_token_id: str,
    event_id: str,
    question: str = "Q?",
    active: bool = True,
    volume: float = 5000.0,
) -> Market:
    return Market(
        condition_id=condition_id,
        question=question,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        neg_risk=False,
        event_id=event_id,
        min_tick_size="0.01",
        active=active,
        volume=volume,
    )


def _make_event(event_id: str, title: str) -> Event:
    return Event(
        event_id=event_id,
        title=title,
        markets=(
            _make_market(
                condition_id=f"cond_{event_id}",
                yes_token_id=f"yes_{event_id}",
                no_token_id=f"no_{event_id}",
                event_id=event_id,
            ),
        ),
        neg_risk=False,
    )


def _make_book(
    token_id: str,
    best_ask: float = 0.50,
    best_bid: float = 0.45,
    ask_size: float = 200.0,
    bid_size: float = 200.0,
) -> OrderBook:
    asks = (PriceLevel(price=best_ask, size=ask_size),) if best_ask else ()
    bids = (PriceLevel(price=best_bid, size=bid_size),) if best_bid else ()
    return OrderBook(token_id=token_id, bids=bids, asks=asks)


# ── Integration tests ────────────────────────────────────────────────────

class TestCorrelationPipelineDetection:
    def test_detects_parent_child_violation(self):
        """
        Full pipeline: detect parent-child probability violation across 2 events.
        Parent should be >= child, but here parent(0.35) < child(0.65).
        """
        parent = _make_event("e_parent", "Will Trump win the presidency?")
        child = _make_event("e_child", "Will Trump win Ohio?")

        books = {
            "yes_e_parent": _make_book("yes_e_parent", best_ask=0.35, best_bid=0.30),
            "yes_e_child": _make_book("yes_e_child", best_ask=0.65, best_bid=0.60),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)

        assert len(opps) == 1
        opp = opps[0]
        assert opp.type == OpportunityType.CORRELATION_ARB
        assert opp.net_profit > 0
        assert len(opp.legs) == 2

    def test_detects_temporal_violation(self):
        """
        Temporal: earlier deadline should be <= later, but here earlier(0.60) > later(0.30).
        """
        earlier = _make_event("e_mar", "Bitcoin to $100K by March 2026")
        later = _make_event("e_jun", "Bitcoin to $100K by June 2026")

        books = {
            "yes_e_mar": _make_book("yes_e_mar", best_ask=0.60, best_bid=0.55),
            "yes_e_jun": _make_book("yes_e_jun", best_ask=0.30, best_bid=0.25),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        opps = scanner.scan([earlier, later], books, gas_cost_usd=0.001)

        assert len(opps) == 1
        opp = opps[0]
        assert opp.type == OpportunityType.CORRELATION_ARB
        # Edge: 0.60 - 0.30 = 0.30
        assert opp.expected_profit_per_set > 0.25


class TestCorrelationScoring:
    def test_urgency_is_steady_state(self):
        """CORRELATION_ARB should get urgency=0.50 (steady-state)."""
        opp = Opportunity(
            type=OpportunityType.CORRELATION_ARB,
            event_id="e1",
            legs=(
                LegOrder("tok1", Side.BUY, 0.35, 200),
                LegOrder("tok2", Side.SELL, 0.60, 200),
            ),
            expected_profit_per_set=0.25,
            net_profit_per_set=0.25,
            max_sets=200.0,
            gross_profit=50.0,
            estimated_gas_cost=0.002,
            net_profit=49.998,
            roi_pct=71.4,
            required_capital=70.0,
        )
        ctx = ScoringContext(book_depth_ratio=1.5, confidence=0.8)
        scored = score_opportunity(opp, ctx)

        assert scored.urgency_score == 0.50
        assert scored.total_score > 0

    def test_ranks_with_other_types(self):
        """CORRELATION_ARB should rank alongside other opp types correctly."""
        corr_opp = Opportunity(
            type=OpportunityType.CORRELATION_ARB,
            event_id="e1",
            legs=(
                LegOrder("tok1", Side.BUY, 0.35, 200),
                LegOrder("tok2", Side.SELL, 0.60, 200),
            ),
            expected_profit_per_set=0.25,
            net_profit_per_set=0.25,
            max_sets=200.0,
            gross_profit=50.0,
            estimated_gas_cost=0.002,
            net_profit=49.998,
            roi_pct=71.4,
            required_capital=70.0,
        )
        binary_opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e2",
            legs=(
                LegOrder("tok3", Side.BUY, 0.45, 100),
                LegOrder("tok4", Side.BUY, 0.45, 100),
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

        ranked = rank_opportunities(
            [binary_opp, corr_opp],
            [ScoringContext(), ScoringContext(book_depth_ratio=1.5, confidence=0.8)],
        )

        assert len(ranked) == 2
        # Correlation opp should rank higher (much higher profit)
        assert ranked[0].opportunity.type == OpportunityType.CORRELATION_ARB


class TestCorrelationExecution:
    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_correlation_arb_executes_as_binary(self, mock_create, mock_post):
        """CORRELATION_ARB should execute via the binary path (2-leg batch)."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        opp = Opportunity(
            type=OpportunityType.CORRELATION_ARB,
            event_id="e1",
            legs=(
                LegOrder("tok1", Side.BUY, 0.35, 200),
                LegOrder("tok2", Side.SELL, 0.60, 200),
            ),
            expected_profit_per_set=0.25,
            net_profit_per_set=0.25,
            max_sets=200.0,
            gross_profit=50.0,
            estimated_gas_cost=0.002,
            net_profit=49.998,
            roi_pct=71.4,
            required_capital=70.0,
        )

        result = execute_opportunity(MagicMock(), opp, size=100.0, paper_trading=False)

        assert result.fully_filled is True
        assert len(result.order_ids) == 2
        assert mock_create.call_count == 2
        mock_post.assert_called_once()

    def test_correlation_arb_paper_trading(self):
        """CORRELATION_ARB should work in paper trading mode."""
        opp = Opportunity(
            type=OpportunityType.CORRELATION_ARB,
            event_id="e1",
            legs=(
                LegOrder("tok1", Side.BUY, 0.35, 200),
                LegOrder("tok2", Side.SELL, 0.60, 200),
            ),
            expected_profit_per_set=0.25,
            net_profit_per_set=0.25,
            max_sets=200.0,
            gross_profit=50.0,
            estimated_gas_cost=0.002,
            net_profit=49.998,
            roi_pct=71.4,
            required_capital=70.0,
        )

        result = execute_opportunity(MagicMock(), opp, size=50.0, paper_trading=True)

        assert result.fully_filled is True
        assert all(oid.startswith("paper_") for oid in result.order_ids)


class TestCorrelationEndToEnd:
    @patch("executor.engine.post_orders")
    @patch("executor.engine.create_limit_order")
    def test_detect_score_execute_pipeline(self, mock_create, mock_post):
        """Full E2E: detect correlation arb → score → execute."""
        mock_create.return_value = MagicMock()
        mock_post.return_value = [
            {"orderID": "o1", "status": "matched"},
            {"orderID": "o2", "status": "matched"},
        ]

        # 1. Detect
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.35, best_bid=0.30),
            "yes_e2": _make_book("yes_e2", best_ask=0.65, best_bid=0.60),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        assert len(opps) >= 1

        # 2. Score
        ctxs = [ScoringContext(book_depth_ratio=1.5, confidence=0.8) for _ in opps]
        ranked = rank_opportunities(opps, ctxs)
        assert len(ranked) >= 1
        best = ranked[0]
        assert best.total_score > 0

        # 3. Execute
        result = execute_opportunity(
            MagicMock(), best.opportunity, size=50.0, paper_trading=False,
        )
        assert result.fully_filled is True
        assert result.order_ids == ("o1", "o2")
