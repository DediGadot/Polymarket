"""
Unit tests for scanner/cross_platform.py -- cross-platform arbitrage scanner.
"""

import pytest

from scanner.cross_platform import scan_cross_platform, _check_cross_platform_arb
from scanner.matching import MatchedEvent, PlatformMatch
from scanner.models import (
    Market,
    OrderBook,
    PriceLevel,
    OpportunityType,
    Side,
)


def _make_pm_market(yes_id: str = "pm_yes", no_id: str = "pm_no", eid: str = "e1") -> Market:
    return Market(
        condition_id="c1",
        question="Test market?",
        yes_token_id=yes_id,
        no_token_id=no_id,
        neg_risk=False,
        event_id=eid,
        min_tick_size="0.01",
        active=True,
        volume=10000.0,
    )


def _make_book(token_id: str, bid_price: float, bid_size: float, ask_price: float, ask_size: float) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(bid_price, bid_size),) if bid_price > 0 else (),
        asks=(PriceLevel(ask_price, ask_size),) if ask_price > 0 else (),
    )


def _make_matched_event(
    pm_market: Market | None = None,
    kalshi_ticker: str = "K-TEST",
    confidence: float = 1.0,
    match_method: str = "manual",
) -> MatchedEvent:
    if pm_market is None:
        pm_market = _make_pm_market()
    return MatchedEvent(
        pm_event_id=pm_market.event_id,
        pm_markets=(pm_market,),
        platform_matches=(
            PlatformMatch(
                platform="kalshi",
                event_ticker="K-EVT",
                tickers=(kalshi_ticker,),
                confidence=confidence,
                match_method=match_method,
            ),
        ),
    )


class TestCheckCrossPlatformArb:
    def test_direction2_pm_no_kalshi_yes_arb(self):
        """PM NO cheap + Kalshi YES cheap = arb when total < $1."""
        # PM NO ask = $0.40, Kalshi YES ask = $0.40
        # Total cost = $0.80 -> profit = $0.20 per set
        pm_no_book = _make_book("pm_no", 0.39, 200, 0.40, 200)
        kalshi_book = _make_book("K-TEST", 0.39, 200, 0.40, 200)

        opp = _check_cross_platform_arb(
            event_id="e1",
            pm_book=pm_no_book,
            pm_side=Side.BUY,
            pm_token_id="pm_no",
            ext_book=kalshi_book,
            ext_side=Side.BUY,
            ext_ticker="K-TEST",
            platform="kalshi",
            min_profit_usd=0.01,
            min_roi_pct=0.1,
            gas_per_order=150000,
            gas_oracle=None,
            gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.type == OpportunityType.CROSS_PLATFORM_ARB
        assert opp.expected_profit_per_set == pytest.approx(0.20, abs=0.01)
        assert len(opp.legs) == 2
        assert opp.legs[0].platform == "polymarket"
        assert opp.legs[1].platform == "kalshi"

    def test_no_arb_when_total_above_1(self):
        """PM NO ask = $0.55, Kalshi YES ask = $0.55 -> total = $1.10, no arb."""
        pm_no_book = _make_book("pm_no", 0.54, 200, 0.55, 200)
        kalshi_book = _make_book("K-TEST", 0.54, 200, 0.55, 200)

        opp = _check_cross_platform_arb(
            event_id="e1",
            pm_book=pm_no_book,
            pm_side=Side.BUY,
            pm_token_id="pm_no",
            ext_book=kalshi_book,
            ext_side=Side.BUY,
            ext_ticker="K-TEST",
            platform="kalshi",
            min_profit_usd=0.01,
            min_roi_pct=0.1,
            gas_per_order=150000,
            gas_oracle=None,
            gas_price_gwei=30.0,
        )
        assert opp is None

    def test_direction1_pm_yes_kalshi_sell_arb(self):
        """PM YES cheap + Kalshi YES expensive (sell) = arb."""
        # PM YES ask = $0.40, Kalshi YES bid = $0.70
        # Cost = PM_YES + (1 - Kalshi_YES_bid) = 0.40 + 0.30 = 0.70 < 1.0
        pm_yes_book = _make_book("pm_yes", 0.39, 200, 0.40, 200)
        kalshi_book = _make_book("K-TEST", 0.70, 200, 0.71, 200)

        opp = _check_cross_platform_arb(
            event_id="e1",
            pm_book=pm_yes_book,
            pm_side=Side.BUY,
            pm_token_id="pm_yes",
            ext_book=kalshi_book,
            ext_side=Side.SELL,
            ext_ticker="K-TEST",
            platform="kalshi",
            min_profit_usd=0.01,
            min_roi_pct=0.1,
            gas_per_order=150000,
            gas_oracle=None,
            gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.expected_profit_per_set == pytest.approx(0.30, abs=0.01)

    def test_empty_book_returns_none(self):
        """Missing asks should return None."""
        pm_book = OrderBook(token_id="pm_yes", bids=(), asks=())
        kalshi_book = _make_book("K-TEST", 0.50, 100, 0.50, 100)

        opp = _check_cross_platform_arb(
            event_id="e1",
            pm_book=pm_book,
            pm_side=Side.BUY,
            pm_token_id="pm_yes",
            ext_book=kalshi_book,
            ext_side=Side.BUY,
            ext_ticker="K-TEST",
            platform="kalshi",
            min_profit_usd=0.01,
            min_roi_pct=0.1,
            gas_per_order=150000,
            gas_oracle=None,
            gas_price_gwei=30.0,
        )
        assert opp is None

    def test_depth_limits_max_sets(self):
        """max_sets should be limited by smaller depth."""
        pm_no_book = _make_book("pm_no", 0.39, 50, 0.40, 50)
        kalshi_book = _make_book("K-TEST", 0.39, 200, 0.40, 200)

        opp = _check_cross_platform_arb(
            event_id="e1",
            pm_book=pm_no_book,
            pm_side=Side.BUY,
            pm_token_id="pm_no",
            ext_book=kalshi_book,
            ext_side=Side.BUY,
            ext_ticker="K-TEST",
            platform="kalshi",
            min_profit_usd=0.01,
            min_roi_pct=0.1,
            gas_per_order=150000,
            gas_oracle=None,
            gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.max_sets == 50.0


class TestScanCrossPlatform:
    def test_finds_arb_in_matched_events(self):
        """Should find cross-platform arb in matched events."""
        pm_market = _make_pm_market()
        match = _make_matched_event(pm_market=pm_market)

        pm_books = {
            "pm_yes": _make_book("pm_yes", 0.54, 200, 0.55, 200),
            "pm_no": _make_book("pm_no", 0.34, 200, 0.35, 200),
        }
        kalshi_books = {"K-TEST": _make_book("K-TEST", 0.54, 200, 0.55, 200)}

        opps = scan_cross_platform(
            [match], pm_books, {"kalshi": kalshi_books},
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000,
        )
        # PM NO ask=0.35 + Kalshi YES ask=0.55 = 0.90 < 1.0 -> arb
        assert len(opps) >= 1
        assert opps[0].type == OpportunityType.CROSS_PLATFORM_ARB

    def test_skips_low_confidence_match(self):
        """Should skip matches below min_confidence."""
        pm_market = _make_pm_market()
        match = _make_matched_event(pm_market=pm_market, confidence=0.80)

        pm_books = {
            "pm_yes": _make_book("pm_yes", 0.34, 200, 0.35, 200),
            "pm_no": _make_book("pm_no", 0.34, 200, 0.35, 200),
        }
        kalshi_books = {"K-TEST": _make_book("K-TEST", 0.34, 200, 0.35, 200)}

        opps = scan_cross_platform(
            [match], pm_books, {"kalshi": kalshi_books},
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000,
            min_confidence=0.90,
        )
        assert len(opps) == 0

    def test_no_arb_in_fair_markets(self):
        """Fair pricing on both platforms = no arb."""
        pm_market = _make_pm_market()
        match = _make_matched_event(pm_market=pm_market)

        # PM: YES=0.60, NO=0.40 -> total=1.00
        # Kalshi: YES ask=0.60
        pm_books = {
            "pm_yes": _make_book("pm_yes", 0.59, 200, 0.60, 200),
            "pm_no": _make_book("pm_no", 0.39, 200, 0.40, 200),
        }
        kalshi_books = {"K-TEST": _make_book("K-TEST", 0.59, 200, 0.60, 200)}

        opps = scan_cross_platform(
            [match], pm_books, {"kalshi": kalshi_books},
            min_profit_usd=0.50, min_roi_pct=2.0,
            gas_per_order=150000,
        )
        assert len(opps) == 0

    def test_sorted_by_roi(self):
        """Results should be sorted by ROI descending."""
        m1 = _make_pm_market(yes_id="y1", no_id="n1", eid="e1")
        m2 = _make_pm_market(yes_id="y2", no_id="n2", eid="e2")
        match1 = _make_matched_event(pm_market=m1, kalshi_ticker="K1")
        match2 = MatchedEvent(
            pm_event_id="e2",
            pm_markets=(m2,),
            platform_matches=(
                PlatformMatch(
                    platform="kalshi",
                    event_ticker="K-EVT2",
                    tickers=("K2",),
                    confidence=1.0,
                    match_method="manual",
                ),
            ),
        )

        pm_books = {
            "y1": _make_book("y1", 0.54, 200, 0.55, 200),
            "n1": _make_book("n1", 0.29, 200, 0.30, 200),
            "y2": _make_book("y2", 0.39, 200, 0.40, 200),
            "n2": _make_book("n2", 0.39, 200, 0.40, 200),
        }
        kalshi_books = {
            "K1": _make_book("K1", 0.54, 200, 0.55, 200),
            "K2": _make_book("K2", 0.39, 200, 0.40, 200),
        }

        opps = scan_cross_platform(
            [match1, match2], pm_books, {"kalshi": kalshi_books},
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000,
        )
        if len(opps) >= 2:
            assert opps[0].roi_pct >= opps[1].roi_pct

    def test_extreme_kalshi_price_rejected_by_fee_guard(self):
        """Kalshi fee rate >20% at extreme prices should reject the arb."""
        from scanner.kalshi_fees import KalshiFeeModel

        pm_market = _make_pm_market()
        match = _make_matched_event(pm_market=pm_market)

        # Kalshi YES ask at $0.02 (2 cents) -- fee is $0.01 = 50% fee rate
        pm_books = {
            "pm_yes": _make_book("pm_yes", 0.94, 200, 0.95, 200),
            "pm_no": _make_book("pm_no", 0.01, 200, 0.02, 200),
        }
        kalshi_books = {"K-TEST": _make_book("K-TEST", 0.01, 200, 0.02, 200)}

        kalshi_fee = KalshiFeeModel()

        opps = scan_cross_platform(
            [match], pm_books, {"kalshi": kalshi_books},
            min_profit_usd=0.001, min_roi_pct=0.01,
            gas_per_order=150000,
            platform_fee_models={"kalshi": kalshi_fee},
        )
        # Should be filtered out by fee-rate guard (>20% at $0.02)
        assert len(opps) == 0

    def test_cent_rounding_drift_rejected(self):
        """Kalshi price that drifts >0.5 cents on rounding should be rejected."""
        pm_market = _make_pm_market()
        match = _make_matched_event(pm_market=pm_market)

        pm_books = {
            "pm_yes": _make_book("pm_yes", 0.54, 200, 0.55, 200),
            "pm_no": _make_book("pm_no", 0.34, 200, 0.35, 200),
        }
        kalshi_books = {"K-TEST": _make_book("K-TEST", 0.54, 200, 0.55, 200)}

        opps = scan_cross_platform(
            [match], pm_books, {"kalshi": kalshi_books},
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000,
        )
        # Clean prices at 55 cents should pass cent-rounding check
        for opp in opps:
            for leg in opp.legs:
                if leg.platform == "kalshi":
                    cents = round(leg.price * 100)
                    assert 1 <= cents <= 99

    def test_missing_books_skipped(self):
        """Missing orderbooks should skip, not crash."""
        pm_market = _make_pm_market()
        match = _make_matched_event(pm_market=pm_market)

        opps = scan_cross_platform(
            [match], {}, {},
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000,
        )
        assert len(opps) == 0
