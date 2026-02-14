"""
Unit tests for scanner/outcome_oracle.py -- outcome determination from external data.
"""

import pytest

from scanner.outcome_oracle import (
    OutcomeOracle,
    OutcomeStatus,
    ParsedThreshold,
    check_outcome_sync,
)
from scanner.models import Market


def _make_market(
    question="Will BTC be above $50,000 at 3pm?",
    yes_token_id="yes1",
    no_token_id="no1",
):
    return Market(
        condition_id="cond1",
        question=question,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        neg_risk=False,
        event_id="evt1",
        min_tick_size="0.01",
        active=True,
        volume=10000.0,
    )


class TestParsedThreshold:
    def test_parse_btc_above_50k(self):
        """Parse 'Will BTC be above $50,000'"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will BTC be above $50,000 at 3pm?")
        assert result is not None
        assert result.symbol == "BTC"
        assert result.direction == "above"
        assert result.threshold == 50000.0
        assert result.confidence >= 0.8

    def test_parse_eth_below_3k(self):
        """Parse 'ETH below 3k'"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will ETH be below 3k at end of day?")
        assert result is not None
        assert result.symbol == "ETH"
        assert result.direction == "below"
        assert result.threshold == 3000.0

    def test_parse_sol_above_threshold(self):
        """Parse 'SOL above $200'"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will SOL be above $200?")
        assert result is not None
        assert result.symbol == "SOL"
        assert result.direction == "above"
        assert result.threshold == 200.0

    def test_parse_bitcoin_full_name(self):
        """Parse 'Bitcoin' instead of BTC"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will Bitcoin exceed $100,000?")
        assert result is not None
        assert result.symbol == "BITCOIN"  # matched from question
        assert result.direction == "above"
        assert result.threshold == 100000.0

    def test_parse_no_threshold_returns_none(self):
        """Question without threshold returns None"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will BTC go up today?")
        assert result is None

    def test_parse_no_direction_returns_none(self):
        """Question without direction returns None"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("BTC at $50,000 at 3pm?")
        assert result is None

    def test_parse_non_crypto_returns_none(self):
        """Non-crypto market returns None"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will the Eagles win on Sunday?")
        assert result is None


class TestSpotPrice:
    def test_cache_used_within_ttl(self):
        """Cached price is returned if within TTL"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        # Manually set cache with recent timestamp
        oracle._spot_cache["BTC"] = (50000.0, time.time())

        # Even with network disabled, cache hit works
        price = oracle._get_spot_price("BTC")
        assert price == 50000.0  # cache hit

    def test_network_disabled_returns_none_on_miss(self):
        """With allow_network=False, uncached fetch returns None"""
        oracle = OutcomeOracle(allow_network=False)
        # No cached price
        price = oracle._get_spot_price("BTC")
        assert price is None

    def test_clear_cache(self):
        """clear_cache() wipes the spot cache"""
        oracle = OutcomeOracle(allow_network=False)
        oracle._spot_cache["BTC"] = (50000.0, 0.0)
        oracle.clear_cache()
        assert "BTC" not in oracle._spot_cache


class TestCryptoOutcome:
    def test_above_threshold_confirmed_yes_when_spot_high(self):
        """BTC above $50K, spot=$55K → CONFIRMED_YES"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        # Mock spot price with current timestamp
        oracle._spot_cache["BTC"] = (55000.0, time.time())

        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.CONFIRMED_YES

    def test_above_threshold_confirmed_no_when_spot_low(self):
        """BTC above $50K, spot=$45K → CONFIRMED_NO"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        oracle._spot_cache["BTC"] = (45000.0, time.time())

        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.CONFIRMED_NO

    def test_above_threshold_unknown_when_spot_near(self):
        """BTC above $50K, spot=$50,200 → UNKNOWN (within 1% buffer)"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        oracle._spot_cache["BTC"] = (50200.0, time.time())

        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.UNKNOWN

    def test_below_threshold_confirmed_yes_when_spot_low(self):
        """BTC below $50K, spot=$45K → CONFIRMED_YES"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        oracle._spot_cache["BTC"] = (45000.0, time.time())

        market = _make_market("Will BTC be below $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.CONFIRMED_YES

    def test_below_threshold_confirmed_no_when_spot_high(self):
        """BTC below $50K, spot=$55K → CONFIRMED_NO"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        oracle._spot_cache["BTC"] = (55000.0, time.time())

        market = _make_market("Will BTC be below $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.CONFIRMED_NO

    def test_non_crypto_market_returns_unknown(self):
        """Non-crypto market → UNKNOWN"""
        oracle = OutcomeOracle(allow_network=False)

        market = _make_market("Will the Eagles win on Sunday?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.UNKNOWN

    def test_no_spot_price_returns_unknown(self):
        """Failed spot fetch → UNKNOWN (graceful degradation)"""
        oracle = OutcomeOracle(allow_network=False)
        # No cached spot price

        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.UNKNOWN


class TestCheckOutcome:
    def test_delegates_to_crypto_checker(self):
        """check_outcome() routes to crypto checker for crypto markets"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        oracle._spot_cache["BTC"] = (55000.0, time.time())

        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = oracle.check_outcome(market)

        assert result == OutcomeStatus.CONFIRMED_YES

    def test_non_market_returns_unknown(self):
        """Non-parseable market → UNKNOWN"""
        oracle = OutcomeOracle(allow_network=False)

        market = _make_market("Will it rain tomorrow?")
        result = oracle.check_outcome(market)

        assert result == OutcomeStatus.UNKNOWN


class TestConvenienceFunction:
    def test_check_outcome_sync_no_network(self):
        """Convenience function with network disabled"""
        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = check_outcome_sync(market, allow_network=False)
        assert result == OutcomeStatus.UNKNOWN

    def test_check_outcome_sync_no_parsing(self):
        """Convenience function with non-crypto market"""
        market = _make_market("Will the Eagles win?")
        result = check_outcome_sync(market, allow_network=False)
        assert result == OutcomeStatus.UNKNOWN


class TestEdgeCases:
    def test_zero_threshold_returns_none(self):
        """Threshold of 0 is invalid"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will BTC be above $0?")
        assert result is None

    def test_negative_threshold_returns_none(self):
        """Negative threshold is invalid"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will BTC be above -$1000?")
        assert result is None

    def test_malformed_number_returns_none(self):
        """Non-numeric threshold returns None"""
        oracle = OutcomeOracle(allow_network=False)
        result = oracle._parse_crypto_threshold("Will BTC be above PRICE?")
        assert result is None

    def test_threshold_at_exactly_buffer_edge(self):
        """Spot exactly at buffer edge → UNKNOWN"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        # $50,000 threshold with 1% buffer = $500 buffer
        # Spot at $50,500 is exactly at the upper buffer edge
        oracle._spot_cache["BTC"] = (50500.0, time.time())

        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        # At exactly buffer edge, should be UNKNOWN (conservative)
        assert result == OutcomeStatus.UNKNOWN

    def test_just_above_buffer_confirms(self):
        """Spot just above buffer → CONFIRMED_YES"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        # $50,000 threshold with 1% buffer = $500 buffer
        # Spot at $50,501 is just above the buffer
        oracle._spot_cache["BTC"] = (50501.0, time.time())

        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.CONFIRMED_YES

    def test_just_below_buffer_confirms_no(self):
        """Spot just below buffer for above-market → CONFIRMED_NO"""
        import time
        oracle = OutcomeOracle(allow_network=False)
        # $50,000 threshold with 1% buffer = $500 buffer
        # Spot at $49,499 is just below the buffer
        oracle._spot_cache["BTC"] = (49499.0, time.time())

        market = _make_market("Will BTC be above $50,000 at 3pm?")
        result = oracle._check_crypto_outcome(market)

        assert result == OutcomeStatus.CONFIRMED_NO
