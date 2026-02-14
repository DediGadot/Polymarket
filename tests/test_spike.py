"""
Unit tests for scanner/spike.py -- event-driven spike detection.
"""

import time

from scanner.spike import (
    PriceHistory,
    SpikeDetector,
    SpikeEvent,
    scan_spike_opportunities,
)
from scanner.fees import MarketFeeModel
from scanner.book_cache import BookCache
from scanner.models import (
    Market,
    Event,
    OpportunityType,
)


def _make_market(yes_id="yes1", event_id="evt1", question="Outcome A?"):
    return Market(
        condition_id="cond1",
        question=question,
        yes_token_id=yes_id,
        no_token_id="no_" + yes_id,
        neg_risk=True,
        event_id=event_id,
        min_tick_size="0.01",
        active=True,
        volume=10000.0,
    )


class TestPriceHistory:
    def test_record_and_latest(self):
        h = PriceHistory()
        h.record(0.50, 100.0)
        h.record(0.55, 101.0)
        assert h.latest == (101.0, 0.55)
        assert len(h) == 2

    def test_pct_change(self):
        h = PriceHistory()
        h.record(0.50, 100.0)
        h.record(0.55, 110.0)
        pct = h.pct_change(window_sec=20.0)
        assert pct is not None
        assert abs(pct - 10.0) < 0.1

    def test_pct_change_negative(self):
        h = PriceHistory()
        h.record(0.60, 100.0)
        h.record(0.54, 110.0)
        pct = h.pct_change(window_sec=20.0)
        assert pct is not None
        assert pct < 0

    def test_velocity(self):
        h = PriceHistory()
        h.record(0.50, 100.0)
        h.record(0.55, 105.0)  # +0.05 in 5 seconds
        vel = h.velocity(window_sec=10.0)
        assert vel is not None
        assert abs(vel - 0.01) < 0.001

    def test_insufficient_data(self):
        h = PriceHistory()
        assert h.pct_change(10.0) is None
        assert h.velocity(10.0) is None
        h.record(0.50, 100.0)
        assert h.pct_change(10.0) is None  # need 2 points

    def test_old_points_pruned(self):
        h = PriceHistory(max_window_sec=10.0)
        h.record(0.50, 100.0)
        h.record(0.55, 105.0)
        h.record(0.60, 115.0)  # oldest (100.0) should be pruned
        assert len(h) == 2  # only 105.0 and 115.0 remain


class TestSpikeDetector:
    def test_detects_spike(self):
        d = SpikeDetector(threshold_pct=5.0, window_sec=30.0)
        d.register_token("tok1", "evt1")
        base = time.time()
        d.update("tok1", 0.50, base)
        d.update("tok1", 0.55, base + 10)  # +10% in 10s
        spikes = d.detect_spikes()
        assert len(spikes) == 1
        assert spikes[0].token_id == "tok1"
        assert spikes[0].magnitude_pct >= 5.0

    def test_no_spike_below_threshold(self):
        d = SpikeDetector(threshold_pct=10.0, window_sec=30.0)
        d.register_token("tok1", "evt1")
        base = time.time()
        d.update("tok1", 0.50, base)
        d.update("tok1", 0.52, base + 10)  # +4%, below threshold
        spikes = d.detect_spikes()
        assert len(spikes) == 0

    def test_cooldown_prevents_re_trigger(self):
        d = SpikeDetector(threshold_pct=5.0, window_sec=30.0, cooldown_sec=60.0)
        d.register_token("tok1", "evt1")
        base = time.time()
        d.update("tok1", 0.50, base)
        d.update("tok1", 0.56, base + 10)
        spikes1 = d.detect_spikes()
        assert len(spikes1) == 1

        # Immediate re-check should be in cooldown
        d.update("tok1", 0.60, base + 15)
        spikes2 = d.detect_spikes()
        assert len(spikes2) == 0

    def test_negative_spike(self):
        d = SpikeDetector(threshold_pct=5.0, window_sec=30.0)
        d.register_token("tok1", "evt1")
        base = time.time()
        d.update("tok1", 0.60, base)
        d.update("tok1", 0.54, base + 10)  # -10%
        spikes = d.detect_spikes()
        assert len(spikes) == 1
        assert spikes[0].direction < 0

    def test_get_velocity(self):
        d = SpikeDetector()
        d.update("tok1", 0.50, 100.0)
        d.update("tok1", 0.55, 105.0)
        vel = d.get_velocity("tok1", 10.0)
        assert vel is not None

    def test_get_velocity_unknown_token(self):
        d = SpikeDetector()
        assert d.get_velocity("unknown") is None

    def test_cleanup_stale_removes_old_tokens(self):
        """cleanup_stale() should remove tokens not in active_tokens set."""
        d = SpikeDetector()
        d.register_token("tok1", "evt1")
        d.register_token("tok2", "evt1")
        d.register_token("tok3", "evt1")

        # Add price history for all tokens
        base = time.time()
        d.update("tok1", 0.50, base)
        d.update("tok2", 0.55, base)
        d.update("tok3", 0.60, base)

        # All tokens should have history
        assert len(d._histories) == 3
        assert len(d._token_events) == 3

        # cleanup with only tok1, tok2 active (tok3 stale)
        active_tokens = {"tok1", "tok2"}
        d.cleanup_stale(active_tokens)

        # tok3 should be removed
        assert len(d._histories) == 2
        assert "tok1" in d._histories
        assert "tok2" in d._histories
        assert "tok3" not in d._histories
        assert len(d._token_events) == 2

    def test_cleanup_stale_preserves_active(self):
        """cleanup_stale() should preserve all active tokens."""
        d = SpikeDetector()
        for i in range(5):
            d.register_token(f"tok{i}", "evt1")
            d.update(f"tok{i}", 0.50, time.time())

        assert len(d._histories) == 5

        # All tokens are active
        active_tokens = {f"tok{i}" for i in range(5)}
        d.cleanup_stale(active_tokens)

        # All should remain
        assert len(d._histories) == 5


class TestScanSpikeOpportunities:
    def test_detects_lag_arb(self):
        """After a spike, if sibling sum < 1.0, detect arb."""
        m1 = _make_market("yes1", "evt1", "Outcome A?")
        m2 = _make_market("yes2", "evt1", "Outcome B?")
        m3 = _make_market("yes3", "evt1", "Outcome C?")
        event = Event(event_id="evt1", title="Test Event", markets=(m1, m2, m3), neg_risk=True)

        cache = BookCache()
        # Spike: m1 dropped to 0.10, m2 and m3 haven't adjusted (still at 0.40 each)
        # Sum = 0.10 + 0.40 + 0.40 = 0.90 < 1.0 → 10% arb
        cache.apply_snapshot("yes1", [{"price": "0.08", "size": "100"}], [{"price": "0.10", "size": "100"}])
        cache.apply_snapshot("yes2", [{"price": "0.38", "size": "100"}], [{"price": "0.40", "size": "100"}])
        cache.apply_snapshot("yes3", [{"price": "0.38", "size": "100"}], [{"price": "0.40", "size": "100"}])

        spike = SpikeEvent(
            token_id="yes1", event_id="evt1",
            direction=-20.0, magnitude_pct=20.0, velocity=-0.01, timestamp=time.time(),
        )
        fm = MarketFeeModel(enabled=False)
        opps = scan_spike_opportunities(spike, event, cache, fm, min_profit_usd=0.01)
        assert len(opps) == 1
        assert opps[0].type == OpportunityType.SPIKE_LAG
        assert opps[0].net_profit > 0

    def test_no_arb_when_sum_above_1(self):
        m1 = _make_market("yes1", "evt1", "A?")
        m2 = _make_market("yes2", "evt1", "B?")
        event = Event(event_id="evt1", title="Test", markets=(m1, m2), neg_risk=True)

        cache = BookCache()
        cache.apply_snapshot("yes1", [], [{"price": "0.55", "size": "100"}])
        cache.apply_snapshot("yes2", [], [{"price": "0.50", "size": "100"}])

        spike = SpikeEvent(
            token_id="yes1", event_id="evt1",
            direction=5.0, magnitude_pct=5.0, velocity=0.01, timestamp=time.time(),
        )
        fm = MarketFeeModel(enabled=False)
        opps = scan_spike_opportunities(spike, event, cache, fm)
        assert len(opps) == 0

    def test_non_negrisk_skipped(self):
        m1 = _make_market("yes1", "evt1", "A?")
        event = Event(event_id="evt1", title="Test", markets=(m1,), neg_risk=False)

        cache = BookCache()
        spike = SpikeEvent(
            token_id="yes1", event_id="evt1",
            direction=10.0, magnitude_pct=10.0, velocity=0.01, timestamp=time.time(),
        )
        fm = MarketFeeModel(enabled=False)
        opps = scan_spike_opportunities(spike, event, cache, fm)
        assert len(opps) == 0
