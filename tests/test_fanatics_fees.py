"""
Tests for scanner/fanatics_fees.py -- Fanatics placeholder fee model.
Mirrors test patterns from test_fees.py for Kalshi-equivalent parabolic formula.
"""

from __future__ import annotations

import math
import pytest

from scanner.fanatics_fees import FanaticsFeeModel
from scanner.models import LegOrder, Side


class TestFanaticsFeeModel:
    def test_platform_name(self):
        model = FanaticsFeeModel()
        assert model.platform_name == "fanatics"

    def test_no_resolution_fee(self):
        model = FanaticsFeeModel()
        assert model.has_resolution_fee is False

    def test_fee_at_50_cents(self):
        """Max fee at P=0.50: ceil(0.07 * 100 * 0.5 * 0.5) = ceil(1.75) = 2 cents."""
        model = FanaticsFeeModel()
        fee = model.taker_fee_per_contract(0.50)
        assert fee == 0.02

    def test_fee_at_10_cents(self):
        """At P=0.10: ceil(0.07 * 100 * 0.1 * 0.9) = ceil(0.63) = 1 cent."""
        model = FanaticsFeeModel()
        fee = model.taker_fee_per_contract(0.10)
        assert fee == 0.01

    def test_fee_at_90_cents(self):
        """At P=0.90: ceil(0.07 * 100 * 0.9 * 0.1) = ceil(0.63) = 1 cent."""
        model = FanaticsFeeModel()
        fee = model.taker_fee_per_contract(0.90)
        assert fee == 0.01

    def test_fee_symmetric(self):
        """Fee(p) == Fee(1-p) due to p*(1-p) symmetry."""
        model = FanaticsFeeModel()
        for p in [0.10, 0.20, 0.30, 0.40, 0.50]:
            assert model.taker_fee_per_contract(p) == model.taker_fee_per_contract(1 - p)

    def test_total_fee(self):
        model = FanaticsFeeModel()
        fee_per = model.taker_fee_per_contract(0.50)
        assert model.total_fee(0.50, 10) == fee_per * 10

    def test_adjust_profit(self):
        model = FanaticsFeeModel()
        legs = (
            LegOrder("t1", Side.BUY, 0.40, 10, platform="fanatics"),
            LegOrder("t2", Side.SELL, 0.60, 10, platform="fanatics"),
        )
        gross = 0.10
        net = model.adjust_profit(gross, legs)
        fee1 = model.taker_fee_per_contract(0.40)
        fee2 = model.taker_fee_per_contract(0.60)
        assert net == pytest.approx(gross - fee1 - fee2)

    def test_fee_clamped_at_boundaries(self):
        """Prices outside 0.01-0.99 are clamped."""
        model = FanaticsFeeModel()
        # Below minimum
        fee_low = model.taker_fee_per_contract(0.001)
        expected = math.ceil(0.07 * 100 * 0.01 * 0.99) / 100.0
        assert fee_low == expected
        # Above maximum
        fee_high = model.taker_fee_per_contract(1.5)
        assert fee_high == expected  # clamped to 0.99, symmetric with 0.01
