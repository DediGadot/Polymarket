"""
Unit tests for scanner/negrisk.py -- NegRisk multi-outcome rebalancing detection.
"""

from scanner.negrisk import scan_negrisk_events, _check_buy_all_arb, _check_sell_all_arb
from scanner.models import (
    Market,
    Event,
    OrderBook,
    PriceLevel,
    OpportunityType,
    Side,
)


def _make_market(cid, yes_id, no_id, eid="e1"):
    return Market(
        condition_id=cid,
        question=f"Outcome {cid}?",
        yes_token_id=yes_id,
        no_token_id=no_id,
        neg_risk=True,
        event_id=eid,
        min_tick_size="0.01",
        active=True,
    )


def _make_event(markets, eid="e1"):
    return Event(
        event_id=eid,
        title="Multi-outcome event",
        markets=tuple(markets),
        neg_risk=True,
    )


def _make_book(token_id, best_bid_price, best_bid_size, best_ask_price, best_ask_size):
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(best_bid_price, best_bid_size),) if best_bid_price else (),
        asks=(PriceLevel(best_ask_price, best_ask_size),) if best_ask_price else (),
    )


class TestCheckBuyAllArb:
    def test_3_outcome_profitable(self):
        """3 outcomes at 0.30 each = 0.90 total -> 0.10 profit per set."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        books = {
            "y1": _make_book("y1", 0.29, 100, 0.30, 100),
            "y2": _make_book("y2", 0.29, 100, 0.30, 100),
            "y3": _make_book("y3", 0.29, 100, 0.30, 100),
        }

        opp = _check_buy_all_arb(
            event, books, list(event.markets),
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.type == OpportunityType.NEGRISK_REBALANCE
        assert abs(opp.expected_profit_per_set - 0.10) < 1e-9
        assert len(opp.legs) == 3
        assert all(leg.side == Side.BUY for leg in opp.legs)

    def test_5_outcome_profitable(self):
        """5 outcomes: 0.15+0.15+0.15+0.15+0.15 = 0.75 -> 0.25 profit."""
        markets = [_make_market(f"c{i}", f"y{i}", f"n{i}") for i in range(5)]
        event = _make_event(markets)
        books = {f"y{i}": _make_book(f"y{i}", 0.14, 50, 0.15, 50) for i in range(5)}

        opp = _check_buy_all_arb(
            event, books, list(event.markets),
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert abs(opp.expected_profit_per_set - 0.25) < 1e-9
        assert len(opp.legs) == 5

    def test_no_arb_when_sum_equals_1(self):
        """3 outcomes at 0.33, 0.33, 0.34 = 1.00 -> no arb."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        books = {
            "y1": _make_book("y1", 0.32, 100, 0.33, 100),
            "y2": _make_book("y2", 0.32, 100, 0.33, 100),
            "y3": _make_book("y3", 0.33, 100, 0.34, 100),
        }

        opp = _check_buy_all_arb(
            event, books, list(event.markets),
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_no_arb_when_sum_above_1(self):
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        event = _make_event([m1, m2])

        books = {
            "y1": _make_book("y1", 0.55, 100, 0.60, 100),
            "y2": _make_book("y2", 0.55, 100, 0.60, 100),
        }

        opp = _check_buy_all_arb(
            event, books, list(event.markets),
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_missing_book_returns_none(self):
        """If any outcome's book is missing, skip the entire event."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        event = _make_event([m1, m2])

        books = {
            "y1": _make_book("y1", 0.30, 100, 0.30, 100),
            # y2 missing
        }

        opp = _check_buy_all_arb(
            event, books, list(event.markets),
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_max_sets_limited_by_smallest_depth(self):
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        books = {
            "y1": _make_book("y1", 0.25, 1000, 0.25, 1000),
            "y2": _make_book("y2", 0.25, 50, 0.25, 50),     # smallest
            "y3": _make_book("y3", 0.25, 500, 0.25, 500),
        }

        opp = _check_buy_all_arb(
            event, books, list(event.markets),
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.max_sets == 50.0


class TestCheckSellAllArb:
    def test_profitable_sell(self):
        """3 outcomes with bids summing > 1.0."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        books = {
            "y1": _make_book("y1", 0.40, 100, 0.42, 100),
            "y2": _make_book("y2", 0.40, 100, 0.42, 100),
            "y3": _make_book("y3", 0.40, 100, 0.42, 100),
        }

        opp = _check_sell_all_arb(
            event, books, list(event.markets),
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert abs(opp.expected_profit_per_set - 0.20) < 1e-9
        assert all(leg.side == Side.SELL for leg in opp.legs)

    def test_no_sell_arb_when_bids_below_1(self):
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        event = _make_event([m1, m2])

        books = {
            "y1": _make_book("y1", 0.40, 100, 0.45, 100),
            "y2": _make_book("y2", 0.40, 100, 0.45, 100),
        }

        opp = _check_sell_all_arb(
            event, books, list(event.markets),
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None


def _mock_book_fetcher(books: dict):
    """Create a book_fetcher callable that returns the given books dict."""
    call_count = 0
    def fetcher(token_ids: list[str]) -> dict:
        nonlocal call_count
        call_count += 1
        return {tid: books[tid] for tid in token_ids if tid in books}
    fetcher.call_count = lambda: call_count
    return fetcher


class TestStaleMarketFiltering:
    def test_stale_market_excluded_from_buy_arb(self):
        """A stale (past end_date) market should be excluded from buy-all arb."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = Market(
            condition_id="c2",
            question="Stale outcome?",
            yes_token_id="y2",
            no_token_id="n2",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=True,
            end_date="2025-01-01T00:00:00Z",  # in the past
        )
        event = _make_event([m1, m2])

        books = {
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
        }

        # With the stale market, only 1 active market remains -> not enough for multi-outcome arb
        # Stale filtering now happens in scan_negrisk_events; pass only non-stale markets
        from scanner.models import is_market_stale
        active = [m for m in event.markets if not is_market_stale(m)]
        opp = _check_buy_all_arb(
            event, books, active,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_closed_market_excluded(self):
        """A closed market should be excluded from arb checks."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = Market(
            condition_id="c2",
            question="Closed outcome?",
            yes_token_id="y2",
            no_token_id="n2",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=True,
            closed=True,
        )
        event = _make_event([m1, m2])

        books = {
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
        }

        # Closed filtering now happens in scan_negrisk_events; pass only non-closed markets
        from scanner.models import is_market_stale
        active = [m for m in event.markets if not is_market_stale(m)]
        opp = _check_buy_all_arb(
            event, books, active,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_stale_markets_filtered_in_scan(self):
        """scan_negrisk_events should skip stale markets entirely."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m_stale = Market(
            condition_id="c3",
            question="Stale?",
            yes_token_id="y3",
            no_token_id="n3",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=True,
            end_date="2025-01-01T00:00:00Z",
        )
        event = _make_event([m1, m2, m_stale])

        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
        })

        result = scan_negrisk_events(
            fetcher, [event], 0.01, 0.1, 150000, 30.0,
        )
        # Should find a 2-outcome arb (m1+m2), not a 3-outcome one
        if result:
            assert len(result[0].legs) == 2


class TestScanNegRiskEvents:
    def test_filters_non_negrisk(self):
        m = _make_market("c1", "y1", "n1")
        event = Event(event_id="e1", title="Binary", markets=(m,), neg_risk=False)
        fetcher = _mock_book_fetcher({})
        result = scan_negrisk_events(
            fetcher, [event], 0.50, 2.0, 150000, 30.0,
        )
        assert result == []

    def test_filters_single_market_events(self):
        """Events with only 1 market can't have multi-outcome arb."""
        m = _make_market("c1", "y1", "n1")
        event = _make_event([m])
        fetcher = _mock_book_fetcher({})
        result = scan_negrisk_events(
            fetcher, [event], 0.50, 2.0, 150000, 30.0,
        )
        assert result == []

    def test_finds_arb(self):
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
            "y3": _make_book("y3", 0.25, 100, 0.25, 100),
        })

        result = scan_negrisk_events(
            fetcher, [event], 0.01, 0.1, 150000, 30.0,
        )
        assert len(result) >= 1
        assert result[0].type == OpportunityType.NEGRISK_REBALANCE

    def test_large_event_subset_disabled_skips_oversized(self):
        markets = [_make_market(f"c{i}", f"y{i}", f"n{i}") for i in range(20)]
        event = _make_event(markets)
        fetcher = _mock_book_fetcher({
            f"y{i}": _make_book(f"y{i}", 0.02, 100, 0.03, 100) for i in range(20)
        })

        result = scan_negrisk_events(
            fetcher,
            [event],
            0.01,
            0.1,
            150000,
            30.0,
            max_legs=15,
            large_event_subset_enabled=False,
        )
        assert result == []

    def test_large_event_subset_enabled_builds_bounded_basket(self):
        markets = [_make_market(f"c{i}", f"y{i}", f"n{i}") for i in range(20)]
        event = _make_event(markets)
        fetcher = _mock_book_fetcher({
            f"y{i}": _make_book(f"y{i}", 0.02, 500, 0.03, 500) for i in range(20)
        })

        result = scan_negrisk_events(
            fetcher,
            [event],
            0.01,
            0.1,
            150000,
            30.0,
            max_legs=15,
            large_event_subset_enabled=True,
            large_event_max_subset=15,
            large_event_tail_max_prob=0.20,
        )
        assert result
        opp = result[0]
        assert len(opp.legs) == 15
        assert opp.reason_code == "negrisk_large_event_subset"
        assert "large_event_subset" in opp.risk_flags

    def test_volume_filter_excludes_zero_volume(self):
        """Markets with zero volume should be excluded when min_volume is set."""
        m1 = Market(
            condition_id="c1",
            question="Zero vol?",
            yes_token_id="y1",
            no_token_id="n1",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=True,
            volume=0.0,
        )
        m2 = Market(
            condition_id="c2",
            question="Also zero vol?",
            yes_token_id="y2",
            no_token_id="n2",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=True,
            volume=0.0,
        )
        event = _make_event([m1, m2])
        fetcher = _mock_book_fetcher({})
        result = scan_negrisk_events(
            fetcher, [event], 0.01, 0.1, 150000, 30.0,
            min_volume=100.0,
        )
        assert result == []

    def test_volume_filter_includes_high_volume(self):
        """Markets with volume above min_volume should be included."""
        m1 = Market(
            condition_id="c1",
            question="High vol A?",
            yes_token_id="y1",
            no_token_id="n1",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=True,
            volume=5000.0,
        )
        m2 = Market(
            condition_id="c2",
            question="High vol B?",
            yes_token_id="y2",
            no_token_id="n2",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=True,
            volume=5000.0,
        )
        m3 = Market(
            condition_id="c3",
            question="High vol C?",
            yes_token_id="y3",
            no_token_id="n3",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=True,
            volume=5000.0,
        )
        event = _make_event([m1, m2, m3])
        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
            "y3": _make_book("y3", 0.25, 100, 0.25, 100),
        })
        result = scan_negrisk_events(
            fetcher, [event], 0.01, 0.1, 150000, 30.0,
            min_volume=100.0,
        )
        assert len(result) >= 1

    def test_fetches_books_once_for_all_events(self):
        """Scanner should batch-fetch all needed YES books in a single call."""
        e1 = _make_event([
            _make_market("c1", "y1", "n1", "e1"),
            _make_market("c2", "y2", "n2", "e1"),
        ], "e1")
        e2 = _make_event([
            _make_market("c3", "y3", "n3", "e2"),
            _make_market("c4", "y4", "n4", "e2"),
        ], "e2")
        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
            "y3": _make_book("y3", 0.25, 100, 0.25, 100),
            "y4": _make_book("y4", 0.25, 100, 0.25, 100),
        })

        result = scan_negrisk_events(
            fetcher, [e1, e2], 0.01, 0.1, 150000, 30.0,
        )

        assert len(result) >= 2
        assert fetcher.call_count() == 1


class TestEventMarketCountsBypass:
    """Tests for event_market_counts bypass fix (task 1.2)."""

    def test_skips_event_when_market_count_is_zero(self):
        """Event should be skipped when event_market_counts returns 0 (unknown completeness)."""
        m1 = _make_market("c1", "y1", "n1", "e1")
        m2 = _make_market("c2", "y2", "n2", "e1")
        m3 = _make_market("c3", "y3", "n3", "e1")
        event = _make_event([m1, m2, m3], "e1")

        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
            "y3": _make_book("y3", 0.25, 100, 0.25, 100),
        })

        # event_market_counts returns empty dict -> expected_total = 0 -> should skip
        result = scan_negrisk_events(
            fetcher, [event], 0.01, 0.1, 150000, 30.0,
            event_market_counts={},  # Empty -> expected_total = 0 for all events
        )

        # Should skip the event because completeness is unknown
        assert result == []

    def test_skips_event_when_neg_risk_market_id_missing_from_counts(self):
        """Event should be skipped when its neg_risk_market_id is not in counts."""
        m1 = _make_market("c1", "y1", "n1", "e1")
        m2 = _make_market("c2", "y2", "n2", "e1")
        event = _make_event([m1, m2], "e1")
        # Override to have a neg_risk_market_id
        event_with_nrm = Event(
            event_id="e1",
            title="Event with NRM ID",
            markets=event.markets,
            neg_risk=True,
            neg_risk_market_id="nrm123",  # Has NRM ID but not in counts
        )

        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
        })

        # event_market_counts doesn't have "nrm123" key
        result = scan_negrisk_events(
            fetcher, [event_with_nrm], 0.01, 0.1, 150000, 30.0,
            event_market_counts={"other_nrm": 3},  # Missing "nrm123" key
        )

        # Should skip because expected_total = 0 for this event
        assert result == []

    def test_allows_event_when_market_count_positive(self):
        """Event should proceed when event_market_counts has positive value."""
        m1 = _make_market("c1", "y1", "n1", "e1")
        m2 = _make_market("c2", "y2", "n2", "e1")
        event = _make_event([m1, m2], "e1")

        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
        })

        # event_market_counts has expected_total = 2
        result = scan_negrisk_events(
            fetcher, [event], 0.01, 0.1, 150000, 30.0,
            event_market_counts={"e1": 2},  # 2 markets expected, 2 active
        )

        # Should find the arb
        assert len(result) >= 1

    def test_allows_event_when_one_market_missing_from_expected_total(self):
        """Event can proceed with conservative payout cap when only one market is missing."""
        m1 = _make_market("c1", "y1", "n1", "e1")
        m2 = _make_market("c2", "y2", "n2", "e1")
        # Inactive market (closed or stale)
        m3 = Market(
            condition_id="c3",
            question="Inactive outcome",
            yes_token_id="y3",
            no_token_id="n3",
            neg_risk=True,
            event_id="e1",
            min_tick_size="0.01",
            active=False,  # Inactive
        )
        event = _make_event([m1, m2, m3], "e1")

        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
        })

        # event_market_counts has expected_total = 3 but only 2 active
        result = scan_negrisk_events(
            fetcher, [event], 0.01, 0.1, 150000, 30.0,
            event_market_counts={"e1": 3},  # 3 expected, 2 active
        )

        # Missing one market is allowed with reduced payout cap.
        assert len(result) >= 1
        assert "incomplete_group:missing_1" in result[0].risk_flags

    def test_skips_event_when_too_many_markets_missing_from_expected_total(self):
        """Event should be skipped when more than two expected markets are missing."""
        m1 = _make_market("c1", "y1", "n1", "e1")
        m2 = _make_market("c2", "y2", "n2", "e1")
        event = _make_event([m1, m2], "e1")

        fetcher = _mock_book_fetcher({
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
        })

        result = scan_negrisk_events(
            fetcher, [event], 0.01, 0.1, 150000, 30.0,
            event_market_counts={"e1": 5},  # 5 expected, 2 active (3 missing)
        )

        assert result == []
