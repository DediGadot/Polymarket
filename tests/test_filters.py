"""
Unit tests for scanner/filters.py -- pre-filters for scan speed optimization.
"""

from datetime import datetime, timezone, timedelta

from scanner.filters import filter_by_volume, filter_about_to_resolve, apply_pre_filters
from scanner.models import Market


def _make_market(
    volume: float = 10000.0,
    neg_risk: bool = False,
    end_date: str = "",
    condition_id: str = "cond1",
) -> Market:
    return Market(
        condition_id=condition_id,
        question="Test market?",
        yes_token_id="yes1",
        no_token_id="no1",
        neg_risk=neg_risk,
        event_id="evt1",
        min_tick_size="0.01",
        active=True,
        volume=volume,
        end_date=end_date,
    )


def _future_iso(hours: float) -> str:
    """Return an ISO 8601 timestamp `hours` from now."""
    dt = datetime.now(timezone.utc) + timedelta(hours=hours)
    return dt.isoformat()


def _past_iso(hours: float) -> str:
    """Return an ISO 8601 timestamp `hours` ago."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()


# ── Volume filter ──────────────────────────────────────────────────


class TestFilterByVolume:
    def test_min_volume_zero_passes_everything(self):
        """min_volume=0 should be a no-op."""
        markets = [_make_market(volume=0.0), _make_market(volume=100.0)]
        result = filter_by_volume(markets, min_volume=0.0)
        assert len(result) == 2

    def test_skips_low_volume_markets(self):
        """Markets below min_volume are excluded."""
        markets = [
            _make_market(volume=50.0, condition_id="low"),
            _make_market(volume=5000.0, condition_id="high"),
        ]
        result = filter_by_volume(markets, min_volume=1000.0)
        assert len(result) == 1
        assert result[0].condition_id == "high"

    def test_preserves_high_volume_markets(self):
        """Markets at or above min_volume are kept."""
        markets = [
            _make_market(volume=1000.0, condition_id="exact"),
            _make_market(volume=2000.0, condition_id="above"),
        ]
        result = filter_by_volume(markets, min_volume=1000.0)
        assert len(result) == 2

    def test_negrisk_markets_never_filtered(self):
        """NegRisk markets bypass volume filter (need complete outcome sets)."""
        markets = [
            _make_market(volume=0.0, neg_risk=True, condition_id="nr"),
            _make_market(volume=0.0, neg_risk=False, condition_id="bin"),
        ]
        result = filter_by_volume(markets, min_volume=1000.0)
        assert len(result) == 1
        assert result[0].condition_id == "nr"

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert filter_by_volume([], min_volume=100.0) == []


# ── About-to-resolve filter ───────────────────────────────────────


class TestFilterAboutToResolve:
    def test_skips_markets_ending_soon(self):
        """Markets resolving within min_hours are excluded."""
        markets = [_make_market(end_date=_future_iso(0.5))]
        result = filter_about_to_resolve(markets, min_hours=1.0)
        assert len(result) == 0

    def test_passes_markets_ending_later(self):
        """Markets resolving after min_hours are kept."""
        markets = [_make_market(end_date=_future_iso(5.0))]
        result = filter_about_to_resolve(markets, min_hours=1.0)
        assert len(result) == 1

    def test_handles_empty_end_date(self):
        """Markets with no end_date pass through (unknown resolution time)."""
        markets = [_make_market(end_date="")]
        result = filter_about_to_resolve(markets, min_hours=1.0)
        assert len(result) == 1

    def test_handles_invalid_end_date(self):
        """Markets with unparseable end_date pass through gracefully."""
        markets = [_make_market(end_date="not-a-date")]
        result = filter_about_to_resolve(markets, min_hours=1.0)
        assert len(result) == 1

    def test_min_hours_zero_passes_everything(self):
        """min_hours=0 should be a no-op (even past-due markets pass)."""
        markets = [_make_market(end_date=_past_iso(1.0))]
        result = filter_about_to_resolve(markets, min_hours=0.0)
        assert len(result) == 1

    def test_z_suffix_parsed(self):
        """ISO 8601 with Z suffix is parsed correctly."""
        future = datetime.now(timezone.utc) + timedelta(hours=5)
        iso_z = future.strftime("%Y-%m-%dT%H:%M:%SZ")
        markets = [_make_market(end_date=iso_z)]
        result = filter_about_to_resolve(markets, min_hours=1.0)
        assert len(result) == 1

    def test_date_only_format(self):
        """Date-only end_date (e.g. '2099-12-31') is parsed correctly."""
        markets = [_make_market(end_date="2099-12-31")]
        result = filter_about_to_resolve(markets, min_hours=1.0)
        assert len(result) == 1

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert filter_about_to_resolve([], min_hours=1.0) == []


# ── Combined filter ───────────────────────────────────────────────


class TestApplyPreFilters:
    def test_combined_filters(self):
        """Both filters applied in sequence."""
        markets = [
            _make_market(volume=5000.0, end_date=_future_iso(5.0), condition_id="good"),
            _make_market(volume=10.0, end_date=_future_iso(5.0), condition_id="low_vol"),
            _make_market(volume=5000.0, end_date=_future_iso(0.3), condition_id="soon"),
            _make_market(volume=10.0, end_date=_future_iso(0.3), condition_id="both_bad"),
        ]
        result = apply_pre_filters(markets, min_volume=100.0, min_hours=1.0)
        assert len(result) == 1
        assert result[0].condition_id == "good"

    def test_defaults_are_noop(self):
        """Default args (min_volume=0, min_hours=0) pass everything."""
        markets = [_make_market(volume=0.0, end_date=_past_iso(1.0))]
        result = apply_pre_filters(markets)
        assert len(result) == 1

    def test_negrisk_survives_combined(self):
        """NegRisk market with low volume passes combined filters."""
        markets = [
            _make_market(volume=0.0, neg_risk=True, end_date=_future_iso(5.0), condition_id="nr"),
            _make_market(volume=0.0, neg_risk=False, end_date=_future_iso(5.0), condition_id="bin"),
        ]
        result = apply_pre_filters(markets, min_volume=1000.0, min_hours=1.0)
        assert len(result) == 1
        assert result[0].condition_id == "nr"
