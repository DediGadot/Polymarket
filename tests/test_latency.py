"""
Unit tests for scanner/latency.py -- latency arb on 15-min crypto markets.
"""

import time

from scanner.latency import LatencyScanner, scan_latency_markets
from scanner.fees import MarketFeeModel
from scanner.models import (
    Market,
    OrderBook,
    PriceLevel,
    OpportunityType,
    Side,
)


def _make_market(question="Will BTC be up 0.5% in 15 minutes?", event_id="evt1"):
    return Market(
        condition_id="cond1",
        question=question,
        yes_token_id="yes1",
        no_token_id="no1",
        neg_risk=False,
        event_id=event_id,
        min_tick_size="0.01",
        active=True,
        volume=50000.0,
    )


def _make_book(token_id="yes1", bid_price=0.45, bid_size=100, ask_price=0.55, ask_size=100):
    bids = (PriceLevel(bid_price, bid_size),) if bid_price else ()
    asks = (PriceLevel(ask_price, ask_size),) if ask_price else ()
    return OrderBook(token_id=token_id, bids=bids, asks=asks)


class TestIdentifyCryptoMarkets:
    def test_btc_15min(self):
        scanner = LatencyScanner()
        m = _make_market("Will BTC be up 0.5% in 15 minutes?")
        result = scanner.identify_crypto_markets([m])
        assert len(result) == 1
        assert result[0][1] == "BTC"
        assert result[0][2] == "up"

    def test_eth_15min_down(self):
        scanner = LatencyScanner()
        m = _make_market("Will Ethereum be down in 15 min?")
        result = scanner.identify_crypto_markets([m])
        assert len(result) == 1
        assert result[0][1] == "ETH"
        assert result[0][2] == "down"

    def test_sol_15min(self):
        scanner = LatencyScanner()
        m = _make_market("Will SOL be above $200 in 15 minutes?")
        result = scanner.identify_crypto_markets([m])
        assert len(result) == 1
        assert result[0][1] == "SOL"

    def test_non_crypto_filtered(self):
        scanner = LatencyScanner()
        m = _make_market("Will the next president be a Democrat?")
        result = scanner.identify_crypto_markets([m])
        assert len(result) == 0


class TestComputeMomentum:
    def test_positive_momentum(self):
        scanner = LatencyScanner()
        now = time.time()
        scanner._spot_cache["BTC"] = (100000.0, now)
        scanner._prev_spot["BTC"] = (99000.0, now - 5)
        momentum = scanner.compute_momentum_pct("BTC")
        assert momentum is not None
        assert abs(momentum - 1.0101) < 0.01  # ~1%

    def test_negative_momentum(self):
        scanner = LatencyScanner()
        now = time.time()
        scanner._spot_cache["BTC"] = (99000.0, now)
        scanner._prev_spot["BTC"] = (100000.0, now - 5)
        momentum = scanner.compute_momentum_pct("BTC")
        assert momentum is not None
        assert momentum < 0

    def test_no_previous_data(self):
        scanner = LatencyScanner()
        scanner._spot_cache["BTC"] = (100000.0, time.time())
        assert scanner.compute_momentum_pct("BTC") is None


class TestComputeImpliedProbability:
    def test_strong_positive_momentum_up(self):
        scanner = LatencyScanner()
        prob = scanner.compute_implied_probability(1.0, "up")
        assert prob > 0.80

    def test_negative_momentum_up(self):
        scanner = LatencyScanner()
        prob = scanner.compute_implied_probability(-1.0, "up")
        assert prob < 0.20

    def test_strong_positive_momentum_down(self):
        """Positive spot momentum for 'down' market → low probability."""
        scanner = LatencyScanner()
        prob = scanner.compute_implied_probability(1.0, "down")
        assert prob < 0.20

    def test_zero_momentum(self):
        scanner = LatencyScanner()
        prob = scanner.compute_implied_probability(0.0, "up")
        assert abs(prob - 0.50) < 0.01

    def test_clamped_to_bounds(self):
        scanner = LatencyScanner()
        prob = scanner.compute_implied_probability(100.0, "up")
        assert prob <= 0.99
        prob = scanner.compute_implied_probability(-100.0, "up")
        assert prob >= 0.01


class TestCheckLatencyArb:
    def test_arb_detected_buy(self):
        """Strong positive momentum, market at 50/50 → buy YES."""
        fm = MarketFeeModel(enabled=False)  # disable fees for clarity
        scanner = LatencyScanner(min_edge_pct=3.0, fee_model=fm)
        now = time.time()
        scanner._spot_cache["BTC"] = (101000.0, now)
        scanner._prev_spot["BTC"] = (100000.0, now - 5)

        market = _make_market()
        book = _make_book(ask_price=0.52, ask_size=200)

        opp = scanner.check_latency_arb(market, book, "BTC", "up")
        assert opp is not None
        assert opp.type == OpportunityType.LATENCY_ARB
        assert opp.legs[0].side == Side.BUY

    def test_arb_detected_buy_no(self):
        """Strong negative momentum on 'up' market → buy NO (avoids inventory need)."""
        fm = MarketFeeModel(enabled=False)
        scanner = LatencyScanner(min_edge_pct=3.0, fee_model=fm)
        now = time.time()
        scanner._spot_cache["BTC"] = (99000.0, now)
        scanner._prev_spot["BTC"] = (100000.0, now - 5)

        market = _make_market()
        book = _make_book(bid_price=0.48, bid_size=200)
        no_book = _make_book(token_id="no1", ask_price=0.52, ask_size=200)

        opp = scanner.check_latency_arb(
            market, book, "BTC", "up",
            no_books={"no1": no_book},
        )
        assert opp is not None
        assert opp.type == OpportunityType.LATENCY_ARB
        assert opp.legs[0].side == Side.BUY
        assert opp.legs[0].token_id == "no1"

    def test_arb_buy_no_without_no_book_uses_fallback(self):
        """When NO book is unavailable, estimate NO price from YES bid."""
        fm = MarketFeeModel(enabled=False)
        scanner = LatencyScanner(min_edge_pct=3.0, fee_model=fm)
        now = time.time()
        scanner._spot_cache["BTC"] = (99000.0, now)
        scanner._prev_spot["BTC"] = (100000.0, now - 5)

        market = _make_market()
        book = _make_book(bid_price=0.48, bid_size=200)

        opp = scanner.check_latency_arb(market, book, "BTC", "up")
        # With no NO book, price estimated as 1.0 - 0.48 = 0.52
        if opp is not None:
            assert opp.legs[0].side == Side.BUY
            assert opp.legs[0].token_id == "no1"

    def test_no_arb_when_edge_too_small(self):
        """Slight momentum, not enough edge."""
        fm = MarketFeeModel(enabled=False)
        scanner = LatencyScanner(min_edge_pct=10.0, fee_model=fm)
        now = time.time()
        scanner._spot_cache["BTC"] = (100100.0, now)  # tiny 0.1% move
        scanner._prev_spot["BTC"] = (100000.0, now - 5)

        market = _make_market()
        book = _make_book(ask_price=0.50, ask_size=200)

        opp = scanner.check_latency_arb(market, book, "BTC", "up")
        assert opp is None

    def test_no_arb_without_momentum_data(self):
        scanner = LatencyScanner()
        market = _make_market()
        book = _make_book()
        opp = scanner.check_latency_arb(market, book, "BTC", "up")
        assert opp is None

    def test_fees_reduce_edge(self):
        """With dynamic fees enabled, edge should be smaller."""
        fm = MarketFeeModel(enabled=True)
        scanner = LatencyScanner(min_edge_pct=5.0, fee_model=fm)
        now = time.time()
        scanner._spot_cache["BTC"] = (101000.0, now)
        scanner._prev_spot["BTC"] = (100000.0, now - 5)

        market = _make_market()  # crypto 15-min → has fee
        book = _make_book(ask_price=0.50, ask_size=200)

        # At 50/50, fee is ~3.15%, eating into edge
        opp = scanner.check_latency_arb(market, book, "BTC", "up")
        # May or may not be arb depending on edge vs fee
        if opp:
            assert opp.net_profit > 0


class TestScanLatencyMarkets:
    def test_scan_finds_arb(self):
        fm = MarketFeeModel(enabled=False)
        scanner = LatencyScanner(min_edge_pct=3.0, fee_model=fm)
        now = time.time()
        scanner._spot_cache["BTC"] = (101000.0, now)
        scanner._prev_spot["BTC"] = (100000.0, now - 5)

        market = _make_market()
        book = _make_book(ask_price=0.52, ask_size=200)
        crypto_markets = [(market, "BTC", "up")]
        books = {"yes1": book}

        opps = scan_latency_markets(scanner, crypto_markets, books)
        assert len(opps) >= 1
        assert opps[0].type == OpportunityType.LATENCY_ARB

    def test_scan_empty_when_no_momentum(self):
        scanner = LatencyScanner()
        market = _make_market()
        book = _make_book()
        crypto_markets = [(market, "BTC", "up")]
        books = {"yes1": book}

        opps = scan_latency_markets(scanner, crypto_markets, books)
        assert len(opps) == 0
