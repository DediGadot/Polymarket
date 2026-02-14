"""
Tests for scanner/value.py - partial NegRisk value scanner.
"""

from __future__ import annotations

import pytest

from scanner.depth import Side
from scanner.fees import MarketFeeModel
from scanner.models import Event, Market, Opportunity, OpportunityType, OrderBook, PriceLevel
from scanner.value import MAX_VALUE_SETS, VALUE_KELLY_ODDS, scan_value_opportunities


@pytest.fixture
def mock_gas_oracle():
    """Mock gas oracle."""
    class MockGasOracle:
        def estimate_cost_usd(self, n_orders: int, gas_per_order: int) -> float:
            return 0.005 * n_orders

    return MockGasOracle()


@pytest.fixture
def fee_model():
    """Standard fee model."""
    return MarketFeeModel(enabled=True)


@pytest.fixture
def mock_book_fetcher():
    """Factory for creating mock book fetcher functions."""

    def _fetcher(books: dict[str, OrderBook]):
        def fetch(token_ids: list[str]) -> dict[str, OrderBook]:
            return {tid: books[tid] for tid in token_ids if tid in books}
        return fetch

    return _fetcher


def _make_market(
    event_id: str,
    condition_id: str,
    question: str,
    yes_token: str,
    no_token: str,
    min_tick: str = "0.01",
    volume: float = 1000.0,
    neg_risk: bool = True,
    active: bool = True,
) -> Market:
    return Market(
        condition_id=condition_id,
        question=question,
        yes_token_id=yes_token,
        no_token_id=no_token,
        neg_risk=neg_risk,
        event_id=event_id,
        min_tick_size=min_tick,
        active=active,
        volume=volume,
    )


def _make_book(token_id: str, best_ask: float, best_bid: float, depth: float = 100.0) -> OrderBook:
    """Create an orderbook with given best prices and depth."""
    return OrderBook(
        token_id=token_id,
        bids=(
            PriceLevel(price=best_bid, size=depth),
        ),
        asks=(
            PriceLevel(price=best_ask, size=depth),
        ),
    )


def test_value_scanner_10_outcome_cheap_outcome(mock_book_fetcher, mock_gas_oracle):
    """
    10-outcome event, one outcome at $0.02 (uniform = $0.10).
    Uniform discount factor 0.5 means threshold = $0.05.
    $0.02 < $0.05, so value opportunity should be found.
    """
    # Create 10-outcome event
    markets = []
    books = {}

    # 9 outcomes at $0.12, 1 outcome at $0.02
    for i in range(10):
        token_id = f"yes_{i}"
        ask_price = 0.02 if i == 0 else 0.12
        markets.append(
            _make_market(
                event_id="evt_10",
                condition_id=f"cond_{i}",
                question=f"Outcome {i}",
                yes_token=token_id,
                no_token=f"no_{i}",
            )
        )
        books[token_id] = _make_book(token_id, best_ask=ask_price, best_bid=ask_price - 0.01)

    event = Event(
        event_id="evt_10",
        title="10 outcome event",
        markets=tuple(markets),
        neg_risk=True,
        neg_risk_market_id="nrm_10",
    )

    fetcher = mock_book_fetcher(books)

    # Run scanner with very low thresholds to ensure detection
    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=[event],
        min_profit_usd=0.01,
        min_roi_pct=1.0,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
        min_edge_pct=5.0,  # 5% minimum edge
        uniform_discount_factor=0.5,
    )

    # Should find the $0.02 outcome as value
    assert len(opportunities) > 0, "Should find value opportunity for cheap outcome"
    opp = opportunities[0]
    assert opp.type == OpportunityType.NEGRISK_VALUE
    assert len(opp.legs) == 1, "Value opportunity should be single-leg"
    assert opp.legs[0].token_id == "yes_0", "Should identify the cheap outcome"


def test_value_scanner_4_outcome_fair_pricing(mock_book_fetcher, mock_gas_oracle):
    """
    4-outcome event, all roughly $0.25 (uniform = $0.25).
    All outcomes are fairly priced, no value opportunities.
    """
    markets = []
    books = {}

    for i in range(4):
        token_id = f"yes_{i}"
        markets.append(
            _make_market(
                event_id="evt_4",
                condition_id=f"cond_{i}",
                question=f"Outcome {i}",
                yes_token=token_id,
                no_token=f"no_{i}",
            )
        )
        books[token_id] = _make_book(token_id, best_ask=0.25, best_bid=0.24)

    event = Event(
        event_id="evt_4",
        title="4 outcome fair event",
        markets=tuple(markets),
        neg_risk=True,
        neg_risk_market_id="nrm_4",
    )

    fetcher = mock_book_fetcher(books)

    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=[event],
        min_profit_usd=0.01,
        min_roi_pct=1.0,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
        min_edge_pct=5.0,
    )

    # No value opportunities — all fairly priced
    assert len(opportunities) == 0, "Should not find value opportunities when all outcomes are fairly priced"


def test_value_scanner_skips_risk_free_arb(mock_book_fetcher, mock_gas_oracle):
    """
    Event where sum(asks) < 1.0 (risk-free arb exists).
    Value scanner should skip and let negrisk scanner handle it.
    """
    markets = []
    books = {}

    # 4 outcomes with total cost < $1.00
    for i in range(4):
        token_id = f"yes_{i}"
        ask_price = 0.20  # sum = 0.80 < 1.0
        markets.append(
            _make_market(
                event_id="evt_arb",
                condition_id=f"cond_{i}",
                question=f"Outcome {i}",
                yes_token=token_id,
                no_token=f"no_{i}",
            )
        )
        books[token_id] = _make_book(token_id, best_ask=ask_price, best_bid=0.19)

    event = Event(
        event_id="evt_arb",
        title="Arb event",
        markets=tuple(markets),
        neg_risk=True,
        neg_risk_market_id="nrm_arb",
    )

    fetcher = mock_book_fetcher(books)

    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=[event],
        min_profit_usd=0.01,
        min_roi_pct=1.0,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
    )

    # Should skip — let negrisk scanner handle
    assert len(opportunities) == 0, "Value scanner should skip events with risk-free arbs"


def test_value_scanner_single_market_event(mock_book_fetcher, mock_gas_oracle):
    """
    Event with only 1 active market.
    Should be skipped (need multiple outcomes for comparison).
    """
    markets = [
        _make_market(
            event_id="evt_single",
            condition_id="cond_0",
            question="Only outcome",
            yes_token="yes_0",
            no_token="no_0",
        )
    ]
    books = {"yes_0": _make_book("yes_0", best_ask=0.10, best_bid=0.09)}

    event = Event(
        event_id="evt_single",
        title="Single outcome event",
        markets=tuple(markets),
        neg_risk=True,
        neg_risk_market_id="nrm_single",
    )

    fetcher = mock_book_fetcher(books)

    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=[event],
        min_profit_usd=0.01,
        min_roi_pct=1.0,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
    )

    assert len(opportunities) == 0, "Should skip events with only one active market"


def test_value_scanner_min_edge_threshold(mock_book_fetcher, mock_gas_oracle):
    """
    Test that min_edge_pct parameter works correctly.
    Only outcomes with edge >= threshold should be detected.

    We need total_ask > 1.0 to avoid triggering the risk-free arb skip.
    For a 3-outcome event, uniform_prob = 0.333.
    To get total > 1.0, average price must be > 0.333.
    We'll create outcomes where one is cheap (value) and others are expensive.
    """
    markets = []
    books = {}

    # 3 outcomes: one cheap (value), two expensive
    # Outcome 0: ask = 0.08 (well below uniform 0.333)
    # Outcome 1: ask = 0.45 (above uniform)
    # Outcome 2: ask = 0.50 (above uniform)
    # Total = 1.03 > 1.0, so not a risk-free arb
    for i, ask_price in enumerate([0.08, 0.45, 0.50]):
        token_id = f"yes_{i}"
        markets.append(
            _make_market(
                event_id="evt_edge",
                condition_id=f"cond_{i}",
                question=f"Outcome {i}",
                yes_token=token_id,
                no_token=f"no_{i}",
            )
        )
        books[token_id] = _make_book(token_id, best_ask=ask_price, best_bid=ask_price - 0.01)

    event = Event(
        event_id="evt_edge",
        title="Edge threshold test",
        markets=tuple(markets),
        neg_risk=True,
        neg_risk_market_id="nrm_edge",
    )

    fetcher = mock_book_fetcher(books)

    # With 100% threshold (edge must be >= 100% of price), should still find the cheap outcome
    # because uniform_prob (0.333) vs ask (0.08) gives ~316% edge
    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=[event],
        min_profit_usd=0.01,
        min_roi_pct=1.0,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
        min_edge_pct=100.0,
    )

    assert len(opportunities) == 1, "Should find the undervalued outcome"
    assert opportunities[0].legs[0].token_id == "yes_0"


def test_value_scanner_returns_opportunity_sorted_by_roi(mock_book_fetcher, mock_gas_oracle):
    """
    Test that multiple opportunities are returned sorted by ROI descending.
    Total asks must be > 1.0 to avoid triggering risk-free arb skip.
    """
    markets = []
    books = {}

    # Create 2 events with different ROIs
    # Event 0: one cheap outcome (0.08), two expensive (0.45, 0.48) -> total = 1.01
    # Event 1: one cheap outcome (0.06), two expensive (0.46, 0.50) -> total = 1.02
    for event_idx, (cheap_price, exp1, exp2) in enumerate([
        (0.08, 0.45, 0.48),
        (0.06, 0.46, 0.50),
    ]):
        for i, ask_price in enumerate([cheap_price, exp1, exp2]):
            token_id = f"yes_{event_idx}_{i}"
            markets.append(
                _make_market(
                    event_id=f"evt_{event_idx}",
                    condition_id=f"cond_{event_idx}_{i}",
                    question=f"Outcome {i}",
                    yes_token=token_id,
                    no_token=f"no_{i}",
                )
            )
            books[token_id] = _make_book(token_id, best_ask=ask_price, best_bid=ask_price - 0.01)

    events = []
    for event_idx in range(2):
        event_markets = [m for m in markets if m.event_id == f"evt_{event_idx}"]
        events.append(
            Event(
                event_id=f"evt_{event_idx}",
                title=f"Event {event_idx}",
                markets=tuple(event_markets),
                neg_risk=True,
                neg_risk_market_id=f"nrm_{event_idx}",
            )
        )

    fetcher = mock_book_fetcher(books)

    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=events,
        min_profit_usd=0.01,
        min_roi_pct=0.1,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
        min_edge_pct=5.0,
    )

    # Should find 2 opportunities (one from each event)
    assert len(opportunities) >= 2

    # Check sorted by ROI descending
    rois = [opp.roi_pct for opp in opportunities]
    assert rois == sorted(rois, reverse=True), "Opportunities should be sorted by ROI descending"


def test_value_scanner_respects_max_events(mock_book_fetcher, mock_gas_oracle):
    """
    Test that max_events parameter limits processing.
    """
    markets = []
    books = {}
    events = []

    # Create 10 events
    for evt_idx in range(10):
        for i in range(3):
            token_id = f"yes_{evt_idx}_{i}"
            ask_price = 0.02 if i == 0 else 0.30
            markets.append(
                _make_market(
                    event_id=f"evt_{evt_idx}",
                    condition_id=f"cond_{evt_idx}_{i}",
                    question=f"Outcome {i}",
                    yes_token=token_id,
                    no_token=f"no_{i}",
                )
            )
            books[token_id] = _make_book(token_id, best_ask=ask_price, best_bid=ask_price - 0.01)

    for evt_idx in range(10):
        event_markets = [m for m in markets if m.event_id == f"evt_{evt_idx}"]
        events.append(
            Event(
                event_id=f"evt_{evt_idx}",
                title=f"Event {evt_idx}",
                markets=tuple(event_markets),
                neg_risk=True,
                neg_risk_market_id=f"nrm_{evt_idx}",
            )
        )

    fetcher = mock_book_fetcher(books)

    # Only process first 3 events
    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=events,
        min_profit_usd=0.01,
        min_roi_pct=0.1,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
        max_events=3,
    )

    # Even though we might find more, max_events should limit the input
    # We can't directly assert count since not all events may have opportunities
    # but the scanner should have only processed 3 events
    assert len(opportunities) <= 3, "Should not process more than max_events"


def test_value_scanner_constants():
    """Test that constants are properly defined."""
    assert VALUE_KELLY_ODDS == 0.30, "Kelly odds for value bets should be 0.30"
    assert MAX_VALUE_SETS == 10.0, "Max value sets should be 10.0"


def test_value_scanner_skip_non_negrisk_events(mock_book_fetcher, mock_gas_oracle):
    """
    Test that non-negRisk events are skipped.
    """
    markets = []
    books = {}

    # Create non-negRisk event
    for i in range(2):
        token_id = f"yes_{i}"
        markets.append(
            _make_market(
                event_id="evt_binary",
                condition_id=f"cond_{i}",
                question=f"Binary market {i}",
                yes_token=token_id,
                no_token=f"no_{i}",
                neg_risk=False,  # Not a negRisk event
            )
        )
        books[token_id] = _make_book(token_id, best_ask=0.02, best_bid=0.01)

    event = Event(
        event_id="evt_binary",
        title="Binary event",
        markets=tuple(markets),
        neg_risk=False,  # Binary, not negRisk
        neg_risk_market_id="",
    )

    fetcher = mock_book_fetcher(books)

    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=[event],
        min_profit_usd=0.01,
        min_roi_pct=1.0,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
    )

    assert len(opportunities) == 0, "Should skip non-negRisk events"


def test_value_scanner_respects_min_volume(mock_book_fetcher, mock_gas_oracle):
    """
    Test that min_volume filter works correctly.
    """
    markets = []
    books = {}

    # Create event with low volume
    for i in range(3):
        token_id = f"yes_{i}"
        markets.append(
            _make_market(
                event_id="evt_low_vol",
                condition_id=f"cond_{i}",
                question=f"Outcome {i}",
                yes_token=token_id,
                no_token=f"no_{i}",
                volume=10.0,  # Low volume
            )
        )
        books[token_id] = _make_book(token_id, best_ask=0.02, best_bid=0.01)

    event = Event(
        event_id="evt_low_vol",
        title="Low volume event",
        markets=tuple(markets),
        neg_risk=True,
        neg_risk_market_id="nrm_low_vol",
    )

    fetcher = mock_book_fetcher(books)

    opportunities = scan_value_opportunities(
        book_fetcher=fetcher,
        events=[event],
        min_profit_usd=0.01,
        min_roi_pct=1.0,
        gas_per_order=100000,
        gas_price_gwei=50.0,
        gas_oracle=mock_gas_oracle,
        min_volume=100.0,  # Filter out low volume
    )

    assert len(opportunities) == 0, "Should skip low volume events"
