"""
Unit tests for scanner/validation.py -- price, size, and gas data validation.

Tests validate_price(), validate_size(), and validate_gas_gwei() functions.
"""

import math

import pytest

from scanner.validation import validate_price, validate_size, validate_gas_gwei


class TestValidatePrice:
    """Test price validation at ingestion boundaries."""

    def test_normal_price_valid(self):
        """Normal prices within [0, 1] range should pass."""
        assert validate_price(0.01) == 0.01
        assert validate_price(0.50) == 0.50
        assert validate_price(0.99) == 0.99
        assert validate_price(0.0) == 0.0
        assert validate_price(1.0) == 1.0

    def test_zero_price_valid(self):
        """Exactly 0.0 is valid (represents no bids/asks)."""
        assert validate_price(0.0) == 0.0

    def test_ones_price_valid(self):
        """Exactly 1.0 is valid (certain outcome)."""
        assert validate_price(1.0) == 1.0

    def test_negative_price_invalid(self):
        """Negative prices should raise ValueError."""
        with pytest.raises(ValueError, match="negative"):
            validate_price(-0.01)
        with pytest.raises(ValueError, match="negative"):
            validate_price(-1.0)

    def test_greater_than_one_invalid(self):
        """Prices > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_price(1.01)
        with pytest.raises(ValueError, match="out of range"):
            validate_price(2.0)
        with pytest.raises(ValueError, match="out of range"):
            validate_price(999.0)

    def test_nan_price_invalid(self):
        """NaN prices should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            validate_price(float("nan"))

    def test_inf_price_invalid(self):
        """Infinite prices should raise ValueError."""
        with pytest.raises(ValueError, match="Inf"):
            validate_price(float("inf"))
        with pytest.raises(ValueError, match="Inf"):
            validate_price(float("-inf"))

    def test_with_context(self):
        """Context string should be included in error message."""
        with pytest.raises(ValueError, match="BTC"):
            validate_price(1.5, context="BTC price")
        with pytest.raises(ValueError, match="YES ask"):
            validate_price(-0.1, context="YES ask")

    def test_small_valid_epsilon(self):
        """Very small but valid prices."""
        assert validate_price(1e-9) == 1e-9
        assert validate_price(0.001) == 0.001

    def test_large_negative_invalid(self):
        """Large negative values."""
        with pytest.raises(ValueError, match="negative"):
            validate_price(-100.0)


class TestValidateSize:
    """Test size validation at ingestion boundaries."""

    def test_normal_size_valid(self):
        """Normal positive sizes should pass."""
        assert validate_size(1.0) == 1.0
        assert validate_size(100.0) == 100.0
        assert validate_size(1000000.0) == 1000000.0

    def test_zero_size_valid(self):
        """Size of 0 is valid (represents no liquidity)."""
        assert validate_size(0.0) == 0.0

    def test_negative_size_invalid(self):
        """Negative sizes should raise ValueError."""
        with pytest.raises(ValueError, match="negative"):
            validate_size(-1.0)
        with pytest.raises(ValueError, match="negative"):
            validate_size(-0.01)

    def test_nan_size_invalid(self):
        """NaN sizes should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            validate_size(float("nan"))

    def test_inf_size_invalid(self):
        """Infinite sizes should raise ValueError."""
        with pytest.raises(ValueError, match="Inf"):
            validate_size(float("inf"))
        with pytest.raises(ValueError, match="Inf"):
            validate_size(float("-inf"))

    def test_with_context(self):
        """Context string should be included in error message."""
        with pytest.raises(ValueError, match="bid size"):
            validate_size(-10.0, context="bid size")
        with pytest.raises(ValueError, match="available"):
            validate_size(float("nan"), context="available size")

    def test_very_large_size_valid(self):
        """Very large but finite sizes are valid."""
        assert validate_size(1e12) == 1e12


class TestValidateGasGwei:
    """Test gas price validation."""

    def test_normal_gas_valid(self):
        """Normal gas prices should pass."""
        assert validate_gas_gwei(1.0) == 1.0
        assert validate_gas_gwei(30.0) == 30.0
        assert validate_gas_gwei(100.0) == 100.0
        assert validate_gas_gwei(500.0) == 500.0

    def test_zero_gas_valid(self):
        """Zero gas is technically valid (fallback)."""
        assert validate_gas_gwei(0.0) == 0.0

    def test_negative_gas_invalid(self):
        """Negative gas prices should raise ValueError."""
        with pytest.raises(ValueError, match="negative"):
            validate_gas_gwei(-1.0)
        with pytest.raises(ValueError, match="negative"):
            validate_gas_gwei(-30.0)

    def test_nan_gas_invalid(self):
        """NaN gas prices should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            validate_gas_gwei(float("nan"))

    def test_inf_gas_invalid(self):
        """Infinite gas prices should raise ValueError."""
        with pytest.raises(ValueError, match="Inf"):
            validate_gas_gwei(float("inf"))
        with pytest.raises(ValueError, match="Inf"):
            validate_gas_gwei(float("-inf"))

    def test_very_high_gas_valid(self):
        """Very high gas is technically valid (spike conditions)."""
        assert validate_gas_gwei(1000.0) == 1000.0

    def test_unreasonably_high_gas_invalid(self):
        """Gas prices above 10,000 gwei are likely data errors."""
        with pytest.raises(ValueError, match="out of range"):
            validate_gas_gwei(10001.0)
        with pytest.raises(ValueError, match="out of range"):
            validate_gas_gwei(1e6)

    def test_with_context(self):
        """Context string should be included in error message."""
        with pytest.raises(ValueError, match="Polygon"):
            validate_gas_gwei(-1.0, context="Polygon gas")


class TestValidationIntegration:
    """Integration-style tests for validation functions."""

    def test_api_response_simulated(self):
        """Simulate API returning bad data in orderbook format."""
        # Simulate book levels from API
        raw_levels = [
            {"price": "0.52", "size": "100"},
            {"price": "nan", "size": "50"},  # bad price
            {"price": "-0.01", "size": "200"},  # negative price
            {"price": "0.55", "size": "inf"},  # bad size (price is ok but size fails)
        ]

        # Only first and fourth levels have valid prices (though fourth has bad size)
        valid_prices = []
        for level in raw_levels:
            try:
                price = validate_price(float(level["price"]), context="book price")
                valid_prices.append(price)
            except ValueError:
                pass

        # 0.52 and 0.55 are both valid prices
        assert len(valid_prices) == 2
        assert valid_prices[0] == 0.52
        assert valid_prices[1] == 0.55

    def test_clob_book_snapshot_validation(self):
        """Test validation of CLOB snapshot data."""
        snapshot = {
            "bids": [
                {"price": 0.50, "size": 100.0},
                {"price": 0.49, "size": 200.0},
                {"price": 0.48, "size": 150.0},
            ],
            "asks": [
                {"price": 0.51, "size": 100.0},
                {"price": 0.52, "size": float("nan")},  # bad
            ],
        }

        # Validate bids
        validated_bids = []
        for bid in snapshot["bids"]:
            validated_bids.append((
                validate_price(bid["price"], context="bid price"),
                validate_size(bid["size"], context="bid size"),
            ))

        assert len(validated_bids) == 3
        assert validated_bids[0] == (0.50, 100.0)

        # Validate asks - should fail on NaN
        with pytest.raises(ValueError, match="NaN"):
            for ask in snapshot["asks"]:
                validate_size(ask["size"], context="ask size")

    def test_kalshi_cents_to_dollars_validation(self):
        """Test validation of Kalshi price conversion (cents to dollars)."""
        # Kalshi returns 1-99 cents, convert to 0.01-0.99 dollars
        kalshi_prices = [1, 50, 99]
        converted = [p / 100.0 for p in kalshi_prices]

        for price in converted:
            # All should validate (within generic 0-1 range)
            validated = validate_price(price, context="Kalshi price")
            assert 0.01 <= validated <= 0.99

        # Exactly 1.0 dollar = 100 cents, out of range for generic validation
        with pytest.raises(ValueError, match="out of range"):
            validate_price(1.01, context="Kalshi price")  # 101 cents > $1.00

        # Negative cent value should fail
        with pytest.raises(ValueError, match="negative"):
            validate_price(-0.01, context="Kalshi price")  # negative price
