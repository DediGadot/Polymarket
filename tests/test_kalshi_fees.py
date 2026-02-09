"""
Unit tests for scanner/kalshi_fees.py -- Kalshi fee model.
"""

import pytest
from scanner.kalshi_fees import KalshiFeeModel
from scanner.models import LegOrder, Side


class TestTakerFeePerContract:
    def test_max_fee_at_50_cents(self):
        """Fee should be highest (~$0.0175) at price=0.50."""
        model = KalshiFeeModel()
        fee = model.taker_fee_per_contract(0.50)
        # ceil(0.07 * 100 * 0.50 * 0.50) = ceil(1.75) = 2 cents = $0.02
        assert fee == pytest.approx(0.02)

    def test_lower_fee_at_extreme_odds(self):
        """Fee should be lower at extreme prices (near 0 or 1)."""
        model = KalshiFeeModel()
        fee_10 = model.taker_fee_per_contract(0.10)
        fee_50 = model.taker_fee_per_contract(0.50)
        fee_90 = model.taker_fee_per_contract(0.90)
        assert fee_10 < fee_50
        assert fee_90 < fee_50
        # Symmetric: fee at 0.10 == fee at 0.90
        assert fee_10 == fee_90

    def test_fee_at_25_cents(self):
        """ceil(0.07 * 100 * 0.25 * 0.75) = ceil(1.3125) = 2 cents."""
        model = KalshiFeeModel()
        fee = model.taker_fee_per_contract(0.25)
        assert fee == pytest.approx(0.02)

    def test_fee_at_10_cents(self):
        """ceil(0.07 * 100 * 0.10 * 0.90) = ceil(0.63) = 1 cent."""
        model = KalshiFeeModel()
        fee = model.taker_fee_per_contract(0.10)
        assert fee == pytest.approx(0.01)

    def test_fee_at_5_cents(self):
        """ceil(0.07 * 100 * 0.05 * 0.95) = ceil(0.3325) = 1 cent."""
        model = KalshiFeeModel()
        fee = model.taker_fee_per_contract(0.05)
        assert fee == pytest.approx(0.01)

    def test_fee_clamps_price(self):
        """Prices outside 0.01-0.99 should be clamped."""
        model = KalshiFeeModel()
        fee_low = model.taker_fee_per_contract(0.0)
        fee_high = model.taker_fee_per_contract(1.0)
        assert fee_low >= 0
        assert fee_high >= 0


class TestTotalFee:
    def test_scales_with_contracts(self):
        model = KalshiFeeModel()
        fee_1 = model.total_fee(0.50, 1)
        fee_10 = model.total_fee(0.50, 10)
        assert fee_10 == pytest.approx(fee_1 * 10)

    def test_zero_contracts(self):
        model = KalshiFeeModel()
        assert model.total_fee(0.50, 0) == 0.0


class TestAdjustProfit:
    def test_deducts_fees_from_profit(self):
        model = KalshiFeeModel()
        legs = (
            LegOrder("t1", Side.BUY, 0.50, 100),
        )
        gross = 0.10
        net = model.adjust_profit(gross, legs)
        # Fee at 0.50: $0.02 per contract
        assert net == pytest.approx(0.10 - 0.02)

    def test_two_legs(self):
        model = KalshiFeeModel()
        legs = (
            LegOrder("t1", Side.BUY, 0.40, 100),
            LegOrder("t2", Side.BUY, 0.40, 100),
        )
        gross = 0.20
        # Fee at 0.40: ceil(7 * 0.4 * 0.6) = ceil(1.68) = 2 cents = $0.02 per leg
        net = model.adjust_profit(gross, legs)
        assert net == pytest.approx(0.20 - 0.04)

    def test_no_resolution_fee(self):
        """Kalshi has no resolution fee, unlike Polymarket."""
        model = KalshiFeeModel()
        legs = (LegOrder("t1", Side.BUY, 0.50, 100),)
        net = model.adjust_profit(0.10, legs)
        # Only taker fee, no 2% resolution fee
        assert net > 0.10 - 0.03  # Must be > gross - (taker + resolution)
