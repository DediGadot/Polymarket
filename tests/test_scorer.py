"""
Unit tests for scanner/scorer.py -- composite opportunity scoring.
"""

from scanner.scorer import (
    score_opportunity,
    rank_opportunities,
    ScoringContext,
    ScoredOpportunity,
    _score_profit,
    _score_fill,
    _score_urgency,
    _score_competition,
)
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)


def _make_opp(
    opp_type=OpportunityType.BINARY_REBALANCE,
    net_profit=5.0,
    roi_pct=10.0,
    max_sets=100.0,
):
    return Opportunity(
        type=opp_type,
        event_id="e1",
        legs=(LegOrder("y1", Side.BUY, 0.45, max_sets),),
        expected_profit_per_set=net_profit / max_sets if max_sets > 0 else 0,
        net_profit_per_set=net_profit / max_sets if max_sets > 0 else 0,
        max_sets=max_sets,
        gross_profit=net_profit + 0.01,
        estimated_gas_cost=0.01,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=net_profit / (roi_pct / 100) if roi_pct > 0 else 1.0,
    )


class TestScoreProfit:
    def test_high_profit(self):
        opp = _make_opp(net_profit=50.0)
        assert _score_profit(opp) > 0.5

    def test_low_profit(self):
        opp = _make_opp(net_profit=0.20)
        assert _score_profit(opp) > 0
        assert _score_profit(opp) < 0.5

    def test_zero_profit(self):
        opp = _make_opp(net_profit=0.0)
        assert _score_profit(opp) == 0.0

    def test_negative_profit(self):
        opp = _make_opp(net_profit=-1.0)
        assert _score_profit(opp) == 0.0


class TestScoreFill:
    def test_deep_book(self):
        opp = _make_opp()
        ctx = ScoringContext(book_depth_ratio=2.0)
        assert _score_fill(opp, ctx) == 1.0

    def test_thin_book(self):
        opp = _make_opp()
        ctx = ScoringContext(book_depth_ratio=0.5)
        score = _score_fill(opp, ctx)
        assert 0 < score < 0.5

    def test_zero_depth(self):
        opp = _make_opp()
        ctx = ScoringContext(book_depth_ratio=0.0)
        assert _score_fill(opp, ctx) == 0.0


class TestScoreUrgency:
    def test_spike_max_urgency(self):
        opp = _make_opp(opp_type=OpportunityType.SPIKE_LAG)
        ctx = ScoringContext()
        assert _score_urgency(opp, ctx) == 1.0

    def test_latency_high_urgency(self):
        opp = _make_opp(opp_type=OpportunityType.LATENCY_ARB)
        ctx = ScoringContext()
        assert _score_urgency(opp, ctx) == 0.85

    def test_binary_moderate_urgency(self):
        opp = _make_opp(opp_type=OpportunityType.BINARY_REBALANCE)
        ctx = ScoringContext()
        assert _score_urgency(opp, ctx) == 0.50

    def test_is_spike_context_overrides(self):
        opp = _make_opp(opp_type=OpportunityType.BINARY_REBALANCE)
        ctx = ScoringContext(is_spike=True)
        assert _score_urgency(opp, ctx) == 1.0


class TestScoreCompetition:
    def test_no_trades(self):
        ctx = ScoringContext(recent_trade_count=0)
        assert _score_competition(ctx) == 1.0

    def test_many_trades(self):
        ctx = ScoringContext(recent_trade_count=50)
        score = _score_competition(ctx)
        assert score < 0.20

    def test_moderate_trades(self):
        ctx = ScoringContext(recent_trade_count=10)
        score = _score_competition(ctx)
        assert 0.3 < score < 0.8


class TestScoreOpportunity:
    def test_returns_scored_opportunity(self):
        opp = _make_opp()
        ctx = ScoringContext()
        scored = score_opportunity(opp, ctx)
        assert isinstance(scored, ScoredOpportunity)
        assert scored.total_score > 0
        assert scored.opportunity is opp

    def test_spike_scores_higher_than_binary(self):
        spike_opp = _make_opp(opp_type=OpportunityType.SPIKE_LAG, net_profit=5.0)
        binary_opp = _make_opp(opp_type=OpportunityType.BINARY_REBALANCE, net_profit=5.0)
        ctx = ScoringContext()
        spike_scored = score_opportunity(spike_opp, ctx)
        binary_scored = score_opportunity(binary_opp, ctx)
        assert spike_scored.total_score > binary_scored.total_score


class TestRankOpportunities:
    def test_ranked_by_score(self):
        opp1 = _make_opp(net_profit=1.0, roi_pct=2.0)
        opp2 = _make_opp(net_profit=50.0, roi_pct=15.0)
        ranked = rank_opportunities([opp1, opp2])
        assert ranked[0].opportunity.net_profit >= ranked[1].opportunity.net_profit

    def test_empty_list(self):
        assert rank_opportunities([]) == []

    def test_with_contexts(self):
        opp = _make_opp()
        ctx_good = ScoringContext(book_depth_ratio=3.0, recent_trade_count=0)
        ctx_bad = ScoringContext(book_depth_ratio=0.1, recent_trade_count=100)
        ranked = rank_opportunities([opp, opp], [ctx_good, ctx_bad])
        assert ranked[0].total_score > ranked[1].total_score
