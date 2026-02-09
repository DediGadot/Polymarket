"""
Unit tests for scanner/fees.py -- market fee model.
"""

from scanner.fees import MarketFeeModel, MAX_CRYPTO_FEE_RATE, RESOLUTION_FEE_RATE
from scanner.models import Market, LegOrder, Side


def _make_market(question="Test market?"):
    return Market(
        condition_id="cond1",
        question=question,
        yes_token_id="yes1",
        no_token_id="no1",
        neg_risk=False,
        event_id="evt1",
        min_tick_size="0.01",
        active=True,
        volume=10000.0,
    )


class TestIsCrypto15Min:
    def test_btc_15min_up(self):
        m = _make_market("Will BTC be up 0.5% in 15 minutes?")
        fm = MarketFeeModel()
        assert fm.is_crypto_15min(m) is True

    def test_eth_15min_down(self):
        m = _make_market("Will Ethereum be down in 15 min?")
        fm = MarketFeeModel()
        assert fm.is_crypto_15min(m) is True

    def test_sol_15min_above(self):
        m = _make_market("Will SOL be above $200 in 15 minutes?")
        fm = MarketFeeModel()
        assert fm.is_crypto_15min(m) is True

    def test_btc_daily_not_matched(self):
        m = _make_market("Will Bitcoin reach $100k by end of year?")
        fm = MarketFeeModel()
        assert fm.is_crypto_15min(m) is False

    def test_political_market_not_matched(self):
        m = _make_market("Will the next president be a Democrat?")
        fm = MarketFeeModel()
        assert fm.is_crypto_15min(m) is False

    def test_fifteen_spelled_out(self):
        m = _make_market("Will BTC be up in fifteen minutes?")
        fm = MarketFeeModel()
        assert fm.is_crypto_15min(m) is True


class TestDynamicCryptoFee:
    def test_fee_at_50_50(self):
        """Max fee at 50/50 odds."""
        fm = MarketFeeModel()
        fee = fm._dynamic_crypto_fee(0.50)
        assert abs(fee - MAX_CRYPTO_FEE_RATE) < 1e-6

    def test_fee_at_extreme_odds(self):
        """Fee near zero at extreme odds."""
        fm = MarketFeeModel()
        fee_at_01 = fm._dynamic_crypto_fee(0.01)
        assert fee_at_01 < 0.002
        fee_at_99 = fm._dynamic_crypto_fee(0.99)
        assert fee_at_99 < 0.002

    def test_fee_at_zero(self):
        fm = MarketFeeModel()
        assert fm._dynamic_crypto_fee(0.0) == 0.0

    def test_fee_at_one(self):
        fm = MarketFeeModel()
        assert fm._dynamic_crypto_fee(1.0) == 0.0

    def test_fee_at_quarter(self):
        """Fee at 25% odds should be ~75% of max."""
        fm = MarketFeeModel()
        fee = fm._dynamic_crypto_fee(0.25)
        expected = MAX_CRYPTO_FEE_RATE * 4.0 * 0.25 * 0.75
        assert abs(fee - expected) < 1e-6

    def test_fee_symmetric(self):
        """Fee should be same at p and 1-p."""
        fm = MarketFeeModel()
        assert abs(fm._dynamic_crypto_fee(0.30) - fm._dynamic_crypto_fee(0.70)) < 1e-10


class TestGetTakerFee:
    def test_standard_market_zero_fee(self):
        m = _make_market("Will Biden win the election?")
        fm = MarketFeeModel()
        assert fm.get_taker_fee(m, 0.50) == 0.0

    def test_crypto_15min_has_fee(self):
        m = _make_market("Will BTC be up 0.5% in 15 minutes?")
        fm = MarketFeeModel()
        fee = fm.get_taker_fee(m, 0.50)
        assert fee > 0.0
        assert abs(fee - MAX_CRYPTO_FEE_RATE) < 1e-6

    def test_disabled_returns_zero(self):
        m = _make_market("Will BTC be up 0.5% in 15 minutes?")
        fm = MarketFeeModel(enabled=False)
        assert fm.get_taker_fee(m, 0.50) == 0.0


class TestAdjustProfit:
    def test_standard_market_only_resolution_fee(self):
        """Standard market: only resolution fee ($0.02 per set)."""
        m = _make_market("Political market?")
        fm = MarketFeeModel()
        legs = (
            LegOrder("yes1", Side.BUY, 0.45, 100),
            LegOrder("no1", Side.BUY, 0.45, 100),
        )
        adjusted = fm.adjust_profit(0.10, legs, market=m)
        expected = 0.10 - 0.02  # only resolution fee
        assert abs(adjusted - expected) < 1e-6

    def test_crypto_market_deducts_taker_and_resolution(self):
        """Crypto 15-min: taker fee on each leg + resolution fee."""
        m = _make_market("Will BTC be up 0.5% in 15 minutes?")
        fm = MarketFeeModel()
        legs = (
            LegOrder("yes1", Side.BUY, 0.50, 100),
            LegOrder("no1", Side.BUY, 0.50, 100),
        )
        fee_rate = fm.get_taker_fee(m, 0.50)
        expected_taker = fee_rate * 0.50 + fee_rate * 0.50
        expected = 0.10 - expected_taker - 0.02  # gross - taker - resolution
        adjusted = fm.adjust_profit(0.10, legs, market=m)
        assert abs(adjusted - expected) < 1e-6

    def test_resolution_fee_constant(self):
        fm = MarketFeeModel()
        assert abs(fm.estimate_resolution_fee(0.10) - RESOLUTION_FEE_RATE) < 1e-6
