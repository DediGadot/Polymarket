"""
Tests for scanner/maker.py -- maker strategy scanner.
"""

import pytest
from scanner.maker import scan_maker_opportunities, MakerPersistenceGate, MakerExecutionModel
from scanner.models import (
    Market,
    OrderBook,
    PriceLevel,
    OpportunityType,
    Side,
)


def _make_book(
    token_id: str,
    best_bid: float | None,
    best_ask: float | None,
    bid_depth: float = 100.0,
    ask_depth: float = 100.0,
) -> OrderBook:
    """Create an OrderBook with given best prices."""
    bids: tuple[PriceLevel, ...] = ()
    asks: tuple[PriceLevel, ...] = ()

    if best_bid is not None:
        bids = (PriceLevel(price=best_bid, size=bid_depth),)
    if best_ask is not None:
        asks = (PriceLevel(price=best_ask, size=ask_depth),)

    return OrderBook(token_id=token_id, bids=bids, asks=asks)


def _make_market(
    yes_id: str = "yes-123",
    no_id: str = "no-123",
    tick_size: str = "0.01",
    volume: float = 1000.0,
) -> Market:
    """Create a test market."""
    return Market(
        condition_id="cond-123",
        question="Will BTC hit $100k?",
        yes_token_id=yes_id,
        no_token_id=no_id,
        neg_risk=False,
        event_id="evt-123",
        min_tick_size=tick_size,
        active=True,
        volume=volume,
    )


class TestMakerScanner:
    def test_wide_spread_creates_opportunity(self):
        """Wide spread (YES bid=0.45, NO bid=0.45) should find maker edge."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 1
        opp = opps[0]
        assert opp.type == OpportunityType.MAKER_REBALANCE
        assert opp.event_id == "evt-123"
        assert opp.net_profit > 0
        # YES bid + 1 tick = 0.45, NO bid + 1 tick = 0.45, cost = 0.90
        # Gross edge = 0.10, minus gas = 0.01 -> net ~ 0.09
        assert opp.net_profit > 0.05

    def test_no_spread_no_opportunity(self):
        """YES bid=0.50, NO bid=0.50 -> cost = 1.00 -> no opportunity."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.49, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.49, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 0

    def test_no_bid_no_opportunity(self):
        """Missing bid prices -> no opportunity."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=None, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 0

    def test_min_spread_filter(self):
        """1-tick spread should be rejected by min_spread_ticks filter."""
        market = _make_market()
        # YES bid=0.48, ask=0.49 -> spread = 1 tick
        yes_book = _make_book("yes-123", best_bid=0.48, best_ask=0.49)
        no_book = _make_book("no-123", best_bid=0.48, best_ask=0.49)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            min_spread_ticks=2,  # Requires 2 tick spread
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 0

    def test_depth_limits_max_sets(self):
        """Smaller side depth should limit max_sets."""
        market = _make_market()
        # YES side has 10 contracts depth
        # NO side has 5 contracts depth
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50, bid_depth=10.0)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50, bid_depth=5.0)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 1
        assert opps[0].max_sets == 5.0  # Limited by NO side depth

    def test_negrisk_market_skipped(self):
        """NegRisk markets should be skipped."""
        market = Market(
            condition_id="cond-123",
            question="Who wins?",
            yes_token_id="yes-123",
            no_token_id="no-123",
            neg_risk=True,  # NegRisk market
            event_id="evt-123",
            min_tick_size="0.01",
            active=True,
            volume=1000.0,
        )
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 0

    def test_stale_market_skipped(self):
        """Stale markets should be skipped."""
        market = Market(
            condition_id="cond-123",
            question="Will BTC hit $100k?",
            yes_token_id="yes-123",
            no_token_id="no-123",
            neg_risk=False,
            event_id="evt-123",
            min_tick_size="0.01",
            active=False,  # Inactive
            volume=1000.0,
            closed=True,  # Closed
        )
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 0

    def test_zero_depth_no_opportunity(self):
        """Zero depth on either side should yield no opportunity."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50, bid_depth=0.0)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50, bid_depth=100.0)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 0

    def test_sorted_by_profit_descending(self):
        """Results should be sorted by net profit descending."""
        market1 = _make_market(yes_id="yes-1", no_id="no-1")
        market2 = _make_market(yes_id="yes-2", no_id="no-2")

        # Market 1: wide spread = big edge
        yes_book1 = _make_book("yes-1", best_bid=0.40, best_ask=0.50)
        no_book1 = _make_book("no-1", best_bid=0.40, best_ask=0.50)

        # Market 2: narrow spread = small edge
        yes_book2 = _make_book("yes-2", best_bid=0.48, best_ask=0.50)
        no_book2 = _make_book("no-2", best_bid=0.48, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market1, market2],
            {
                "yes-1": yes_book1, "no-1": no_book1,
                "yes-2": yes_book2, "no-2": no_book2,
            },
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 2
        assert opps[0].net_profit >= opps[1].net_profit

    def test_min_edge_threshold(self):
        """Opportunities below min_edge_usd should be filtered."""
        market = _make_market()
        # Very narrow spread -> tiny edge
        yes_book = _make_book("yes-123", best_bid=0.49, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.49, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.10,  # Require at least $0.10 net profit
            gas_cost_per_order=0.005,
        )

        # After gas ($0.01), net would be ~$0.01, below threshold
        assert len(opps) == 0

    def test_missing_book_no_opportunity(self):
        """Missing orderbook should be handled gracefully."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50)
        # NO book missing

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book},  # NO book missing
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 0

    def test_combined_cost_exceeds_1_no_opportunity(self):
        """If combined cost >= 1.0, no opportunity exists."""
        market = _make_market()
        # After +tick, cost would be 1.0 or more
        yes_book = _make_book("yes-123", best_bid=0.49, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.49, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 0

    def test_roi_calculation(self):
        """ROI should be correctly calculated."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50, bid_depth=10.0)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50, bid_depth=10.0)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 1
        opp = opps[0]
        # Cost = 0.90, gross profit per set = 0.10
        # Net = (0.10 * 10) - 0.01 = 0.99
        # ROI = 0.99 / (0.90 * 10) * 100 = 11%
        assert opp.roi_pct > 0
        assert opp.roi_pct < 20  # Should be reasonable

    def test_leg_orders_correct(self):
        """Leg orders should have correct parameters."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        assert len(opps) == 1
        opp = opps[0]
        assert len(opp.legs) == 2
        assert opp.legs[0].side == Side.BUY
        assert opp.legs[1].side == Side.BUY
        assert opp.legs[0].token_id == "yes-123"
        assert opp.legs[1].token_id == "no-123"
        # Price should be bid + tick
        assert opp.legs[0].price == 0.45  # 0.44 + 0.01
        assert opp.legs[1].price == 0.45  # 0.44 + 0.01

    def test_tick_size_0_001(self):
        """Scanner should work with 0.001 tick size markets."""
        market = _make_market(tick_size="0.001")
        yes_book = _make_book("yes-123", best_bid=0.449, best_ask=0.460)
        no_book = _make_book("no-123", best_bid=0.449, best_ask=0.460)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
        )

        # With 0.001 tick, bid+tick = 0.450, cost = 0.90, edge exists
        assert len(opps) == 1

    def test_near_certain_market_filtered(self):
        """Near-certain market (one side < min_leg_price) should be filtered."""
        market = _make_market()
        # YES is near-certain (ask=$0.96), NO is cheap (ask=$0.03 < $0.05 threshold)
        # Both sides have valid bid < ask spreads
        yes_book = _make_book("yes-123", best_bid=0.90, best_ask=0.96)
        no_book = _make_book("no-123", best_bid=0.01, best_ask=0.03)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            min_leg_price=0.05,
        )

        assert len(opps) == 0

    def test_micro_depth_filtered(self):
        """Micro-depth books (below min_depth_sets) should be filtered."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50, bid_depth=2.0)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50, bid_depth=2.0)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            min_depth_sets=5.0,
        )

        assert len(opps) == 0

    def test_low_volume_market_filtered(self):
        """Markets with volume below maker_min_volume should be rejected."""
        market = _make_market(volume=100.0)  # Below 500 threshold
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            min_volume=500.0,
        )

        assert len(opps) == 0

    def test_high_volume_market_passes(self):
        """Markets with volume above maker_min_volume should pass."""
        market = _make_market(volume=1000.0)
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            min_volume=500.0,
        )

        assert len(opps) == 1

    def test_raised_depth_filter_rejects_thin_books(self):
        """With min_depth_sets=15.0, thin books (< 15 sets) should be rejected."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50, bid_depth=10.0)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50, bid_depth=10.0)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            min_depth_sets=15.0,
        )

        assert len(opps) == 0

    def test_near_certain_passes_when_disabled(self):
        """Near-certain market passes when min_leg_price=0.0."""
        market = _make_market()
        # YES is near-certain (ask=$0.96), NO is cheap (ask=$0.03)
        # Both sides have valid bid < ask spreads
        yes_book = _make_book("yes-123", best_bid=0.90, best_ask=0.96)
        no_book = _make_book("no-123", best_bid=0.01, best_ask=0.03)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            min_leg_price=0.0,
        )

        # bid+tick: YES=0.91 + NO=0.02 = 0.93, edge=0.07
        assert len(opps) == 1

    def test_rejects_when_taker_cross_too_expensive(self):
        """Even with maker cost < 1, reject if crossing spread implies weak realizability."""
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.10, best_ask=0.80)
        no_book = _make_book("no-123", best_bid=0.10, best_ask=0.70)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            max_taker_cost=1.03,
            max_spread_ticks=200,
        )
        assert len(opps) == 0

    def test_rejects_when_spread_too_wide(self):
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.20, best_ask=0.80)
        no_book = _make_book("no-123", best_bid=0.20, best_ask=0.80)

        opps = scan_maker_opportunities(
            [market],
            {"yes-123": yes_book, "no-123": no_book},
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            max_spread_ticks=8,
        )
        assert len(opps) == 0

    def test_persistence_gate_requires_multiple_cycles(self):
        gate = MakerPersistenceGate(min_consecutive_cycles=3)
        market = _make_market()
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50)

        books = {"yes-123": yes_book, "no-123": no_book}
        for _ in range(2):
            opps = scan_maker_opportunities(
                [market],
                books,
                fee_model=None,
                min_edge_usd=0.001,
                gas_cost_per_order=0.005,
                persistence_gate=gate,
            )
            assert opps == []

        opps = scan_maker_opportunities(
            [market],
            books,
            fee_model=None,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            persistence_gate=gate,
        )
        assert len(opps) == 1

    def test_execution_model_adds_fill_prob_and_expected_ev(self):
        model = MakerExecutionModel()
        market = _make_market(volume=5000.0)
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50, bid_depth=50.0, ask_depth=50.0)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50, bid_depth=50.0, ask_depth=50.0)
        books = {"yes-123": yes_book, "no-123": no_book}

        # Warm two cycles so EWMA state is populated.
        scan_maker_opportunities(
            [market],
            books,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            execution_model=model,
            min_pair_fill_prob=0.0,
            max_toxicity_score=1.0,
            min_expected_ev_usd=0.0,
        )
        opps = scan_maker_opportunities(
            [market],
            books,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            execution_model=model,
            min_pair_fill_prob=0.0,
            max_toxicity_score=1.0,
            min_expected_ev_usd=0.0,
        )

        assert len(opps) == 1
        opp = opps[0]
        assert 0.0 < opp.pair_fill_prob <= 1.0
        assert 0.0 <= opp.toxicity_score <= 1.0
        assert opp.expected_realized_net == pytest.approx(opp.net_profit)
        assert opp.quote_theoretical_net > 0

    def test_execution_model_ev_gate_can_filter(self):
        model = MakerExecutionModel()
        market = _make_market(volume=1000.0)
        yes_book = _make_book("yes-123", best_bid=0.44, best_ask=0.50, bid_depth=15.0, ask_depth=15.0)
        no_book = _make_book("no-123", best_bid=0.44, best_ask=0.50, bid_depth=15.0, ask_depth=15.0)
        books = {"yes-123": yes_book, "no-123": no_book}

        opps = scan_maker_opportunities(
            [market],
            books,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            execution_model=model,
            min_pair_fill_prob=0.0,
            max_toxicity_score=1.0,
            min_expected_ev_usd=10.0,  # deliberately impossible for this setup
        )
        assert opps == []

    def test_execution_model_toxicity_gate_can_filter(self):
        model = MakerExecutionModel()
        market = _make_market(volume=500.0)
        # Highly imbalanced/thin queue profile tends to increase toxicity.
        yes_book = _make_book("yes-123", best_bid=0.30, best_ask=0.40, bid_depth=500.0, ask_depth=2.0)
        no_book = _make_book("no-123", best_bid=0.60, best_ask=0.70, bid_depth=2.0, ask_depth=500.0)
        books = {"yes-123": yes_book, "no-123": no_book}

        opps = scan_maker_opportunities(
            [market],
            books,
            min_edge_usd=0.001,
            gas_cost_per_order=0.005,
            execution_model=model,
            min_pair_fill_prob=0.0,
            max_toxicity_score=0.05,  # strict threshold to force reject
            min_expected_ev_usd=0.0,
            max_taker_cost=1.20,
            max_spread_ticks=20,
        )
        assert opps == []
