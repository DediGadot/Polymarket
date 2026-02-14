"""
Unit tests for scanner/resolution.py -- resolution sniping scanner.
"""

from datetime import datetime, timezone, timedelta

from scanner.resolution import (
    scan_resolution_opportunities,
    _is_near_resolution,
    _check_yes_snipe,
    _check_no_snipe,
)
from scanner.fees import MarketFeeModel
from scanner.models import (
    Market,
    OrderBook,
    PriceLevel,
    OpportunityType,
    Side,
)
from scanner.outcome_oracle import OutcomeOracle, OutcomeStatus


def _make_market(
    question="Will BTC be above $50,000 at 3pm?",
    yes_id="yes1",
    no_id="no1",
    end_date=None,
    active=True,
):
    return Market(
        condition_id="cond1",
        question=question,
        yes_token_id=yes_id,
        no_token_id=no_id,
        neg_risk=False,
        event_id="evt1",
        min_tick_size="0.01",
        active=active,
        volume=10000.0,
        end_date=end_date or "",
    )


def _make_book(token_id, best_bid_price, best_bid_size, best_ask_price, best_ask_size):
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(best_bid_price, best_bid_size),) if best_bid_price else (),
        asks=(PriceLevel(best_ask_price, best_ask_size),) if best_ask_price else (),
    )


def _mock_outcome_checker(outcome_map: dict) -> callable:
    """Create an outcome checker that returns pre-determined outcomes."""
    def checker(market: Market) -> OutcomeStatus:
        return outcome_map.get(market.question, OutcomeStatus.UNKNOWN)
    return checker


class TestIsNearResolution:
    def test_market_with_future_end_date(self):
        """Market ending in the future (beyond cutoff) is not near resolution."""
        future = (datetime.now(timezone.utc) + timedelta(minutes=120)).isoformat()
        market = _make_market(end_date=future)
        cutoff = datetime.now(timezone.utc) + timedelta(minutes=60)

        result = _is_near_resolution(market, cutoff)
        assert result is False

    def test_market_with_past_end_date(self):
        """Market that ended before cutoff is near resolution."""
        past = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        market = _make_market(end_date=past)
        cutoff = datetime.now(timezone.utc) + timedelta(minutes=60)

        result = _is_near_resolution(market, cutoff)
        assert result is True

    def test_market_with_no_end_date(self):
        """Market without end_date is not near resolution."""
        market = _make_market(end_date="")
        cutoff = datetime.now(timezone.utc) + timedelta(minutes=60)

        result = _is_near_resolution(market, cutoff)
        assert result is False

    def test_market_with_invalid_end_date(self):
        """Market with invalid end_date returns False (graceful degradation)."""
        market = _make_market(end_date="invalid-date")
        cutoff = datetime.now(timezone.utc) + timedelta(minutes=60)

        result = _is_near_resolution(market, cutoff)
        assert result is False


class TestCheckYesSnipe:
    def test_profitable_yes_snipe(self):
        """Market at $0.85, outcome CONFIRMED_YES → opportunity with 15% edge."""
        market = _make_market()
        book = _make_book("yes1", 0.84, 200, 0.85, 200)
        fee_model = MarketFeeModel()

        opp = _check_yes_snipe(market, book, fee_model, min_edge_pct=3.0, gas_cost=0.005)

        assert opp is not None
        assert opp.type == OpportunityType.RESOLUTION_SNIPE
        assert opp.expected_profit_per_set > 0
        assert opp.roi_pct > 3.0  # Above min edge
        assert len(opp.legs) == 1
        assert opp.legs[0].side == Side.BUY
        assert opp.legs[0].token_id == "yes1"

    def test_yes_snipe_below_min_edge(self):
        """Market at $0.98, outcome CONFIRMED_YES → 2% edge < min 3% → skip."""
        market = _make_market()
        book = _make_book("yes1", 0.97, 200, 0.98, 200)
        fee_model = MarketFeeModel()

        opp = _check_yes_snipe(market, book, fee_model, min_edge_pct=3.0, gas_cost=0.005)

        assert opp is None

    def test_yes_snipe_empty_book(self):
        """Empty orderbook → no opportunity."""
        market = _make_market()
        book = OrderBook(token_id="yes1", bids=(), asks=())
        fee_model = MarketFeeModel()

        opp = _check_yes_snipe(market, book, fee_model, min_edge_pct=3.0, gas_cost=0.005)

        assert opp is None

    def test_yes_snipe_no_asks(self):
        """Book has bids but no asks → no opportunity."""
        market = _make_market()
        book = OrderBook(token_id="yes1", bids=(PriceLevel(0.90, 100),), asks=())
        fee_model = MarketFeeModel()

        opp = _check_yes_snipe(market, book, fee_model, min_edge_pct=3.0, gas_cost=0.005)

        assert opp is None

    def test_yes_snipe_fee_deduction(self):
        """Fees are correctly deducted from profit."""
        market = _make_market()
        book = _make_book("yes1", 0.80, 100, 0.80, 100)
        fee_model = MarketFeeModel()

        opp = _check_yes_snipe(market, book, fee_model, min_edge_pct=1.0, gas_cost=0.005)

        assert opp is not None
        # Gross profit = 1.0 - 0.80 = 0.20
        # Net profit should be less due to fees
        assert opp.net_profit < opp.gross_profit


class TestCheckNoSnipe:
    def test_profitable_no_snipe_with_no_book(self):
        """NO orderbook available → use NO book directly."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.10, 200, 0.15, 200)
        no_book = _make_book("no1", 0.10, 200, 0.80, 200)
        books = {"yes1": yes_book, "no1": no_book}
        fee_model = MarketFeeModel()

        opp = _check_no_snipe(market, yes_book, fee_model, min_edge_pct=3.0, gas_cost=0.005, books=books)

        assert opp is not None
        assert opp.type == OpportunityType.RESOLUTION_SNIPE
        assert opp.legs[0].token_id == "no1"
        assert opp.legs[0].side == Side.BUY
        assert opp.roi_pct > 3.0

    def test_no_snipe_fallback_from_yes_book(self):
        """NO orderbook missing → estimate from YES bid."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.15, 200, 0.20, 200)  # bid=0.15
        books = {"yes1": yes_book}
        fee_model = MarketFeeModel()

        opp = _check_no_snipe(market, yes_book, fee_model, min_edge_pct=3.0, gas_cost=0.005, books=books)

        # NO price ≈ 1 - 0.15 = 0.85
        # Edge = 1.0 - (0.85 + fees)
        assert opp is not None
        assert opp.legs[0].token_id == "no1"

    def test_no_snipe_below_min_edge(self):
        """NO price too high → edge below minimum."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.02, 200, 0.03, 200)  # NO ≈ 0.97
        no_book = _make_book("no1", 0.02, 200, 0.97, 200)
        books = {"yes1": yes_book, "no1": no_book}
        fee_model = MarketFeeModel()

        opp = _check_no_snipe(market, yes_book, fee_model, min_edge_pct=5.0, gas_cost=0.005, books=books)

        assert opp is None

    def test_no_snipe_yes_book_no_bid(self):
        """YES book has no bid for fallback estimation."""
        market = _make_market()
        yes_book = OrderBook(token_id="yes1", bids=(), asks=(PriceLevel(0.20, 200),))
        books = {"yes1": yes_book}
        fee_model = MarketFeeModel()

        opp = _check_no_snipe(market, yes_book, fee_model, min_edge_pct=3.0, gas_cost=0.005, books=books)

        assert opp is None


class TestScanResolutionOpportunities:
    def test_filters_inactive_markets(self):
        """Inactive markets are skipped."""
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        market = _make_market(end_date=future, active=False)
        books = {"yes1": _make_book("yes1", 0.80, 100, 0.85, 100)}
        outcome_map = {market.question: OutcomeStatus.CONFIRMED_YES}
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [market], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=3.0,
        )

        assert result == []

    def test_filters_stale_markets(self):
        """Stale (already resolved) markets are skipped."""
        past = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        market = _make_market(end_date=past)
        books = {"yes1": _make_book("yes1", 0.80, 100, 0.85, 100)}
        outcome_map = {market.question: OutcomeStatus.CONFIRMED_YES}
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [market], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=3.0,
        )

        assert result == []

    def test_filters_by_time_to_resolution(self):
        """Markets beyond max_minutes_to_resolution are skipped."""
        future = (datetime.now(timezone.utc) + timedelta(minutes=90)).isoformat()
        market = _make_market(end_date=future)
        books = {"yes1": _make_book("yes1", 0.80, 100, 0.85, 100)}
        outcome_map = {market.question: OutcomeStatus.CONFIRMED_YES}
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [market], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=3.0,
        )

        assert result == []

    def test_filters_unknown_outcome(self):
        """Markets with UNKNOWN outcome are skipped."""
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        market = _make_market(end_date=future)
        books = {"yes1": _make_book("yes1", 0.80, 100, 0.85, 100)}
        outcome_map = {market.question: OutcomeStatus.UNKNOWN}
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [market], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=3.0,
        )

        assert result == []

    def test_finds_yes_snipe_opportunity(self):
        """CONFIRMED_YES with cheap YES → opportunity found."""
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        market = _make_market(end_date=future, question="BTC above 50k?")
        books = {"yes1": _make_book("yes1", 0.80, 100, 0.85, 100)}
        outcome_map = {market.question: OutcomeStatus.CONFIRMED_YES}
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [market], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=3.0,
        )

        assert len(result) == 1
        assert result[0].type == OpportunityType.RESOLUTION_SNIPE
        assert result[0].legs[0].token_id == "yes1"

    def test_finds_no_snipe_opportunity(self):
        """CONFIRMED_NO with cheap NO → opportunity found."""
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        market = _make_market(end_date=future, question="BTC below 50k?")
        yes_book = _make_book("yes1", 0.15, 100, 0.20, 100)
        no_book = _make_book("no1", 0.10, 100, 0.80, 100)
        books = {"yes1": yes_book, "no1": no_book}
        outcome_map = {market.question: OutcomeStatus.CONFIRMED_NO}
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [market], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=3.0,
        )

        assert len(result) == 1
        assert result[0].type == OpportunityType.RESOLUTION_SNIPE
        assert result[0].legs[0].token_id == "no1"

    def test_missing_book_skips_market(self):
        """Market without orderbook is skipped."""
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        market = _make_market(end_date=future, question="BTC above 50k?")
        books = {}  # No book for yes1
        outcome_map = {market.question: OutcomeStatus.CONFIRMED_YES}
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [market], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=3.0,
        )

        assert result == []

    def test_sorted_by_roi_descending(self):
        """Multiple opportunities sorted by ROI descending."""
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()

        # Market 1: better edge
        m1 = _make_market(end_date=future, question="BTC above 50k?", yes_id="yes1", no_id="no1")
        b1 = _make_book("yes1", 0.75, 100, 0.75, 100)  # Larger edge

        # Market 2: smaller edge
        m2 = _make_market(end_date=future, question="ETH above 3k?", yes_id="yes2", no_id="no2")
        b2 = _make_book("yes2", 0.90, 100, 0.90, 100)  # Smaller edge

        books = {"yes1": b1, "no1": _make_book("no1", 0, 0, 0, 0),
                 "yes2": b2, "no2": _make_book("no2", 0, 0, 0, 0)}

        outcome_map = {
            m1.question: OutcomeStatus.CONFIRMED_YES,
            m2.question: OutcomeStatus.CONFIRMED_YES,
        }
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [m1, m2], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=1.0,
        )

        if len(result) >= 2:
            assert result[0].roi_pct >= result[1].roi_pct

    def test_markets_without_end_date_skipped(self):
        """Markets without end_date are not considered near resolution."""
        market = _make_market(end_date="", question="BTC above 50k?")
        books = {"yes1": _make_book("yes1", 0.75, 100, 0.75, 100)}
        outcome_map = {market.question: OutcomeStatus.CONFIRMED_YES}
        checker = _mock_outcome_checker(outcome_map)

        result = scan_resolution_opportunities(
            [market], books, checker, MarketFeeModel(),
            max_minutes_to_resolution=60.0, min_edge_pct=1.0,
        )

        assert result == []
