"""
Validation functions for orderbook/price data at ingestion boundaries.

Provides validate_price(), validate_size(), and validate_gas_gwei() functions
that raise ValueError on invalid data (NaN, Inf, negative, out-of-range).

Import and call these at every float() conversion from external data.
"""

from __future__ import annotations

import math


def validate_price(p: float, context: str = "price") -> float:
    """
    Validate a price value is within [0.0, 1.0] and finite.

    Args:
        p: Price value to validate (typically 0.0-1.0 for prediction markets).
        context: Description of what this value represents (for error messages).

    Returns:
        The validated price value.

    Raises:
        ValueError: If price is NaN, infinite, negative, or > 1.0.
    """
    if math.isnan(p):
        raise ValueError(f"Invalid {context}: NaN")
    if math.isinf(p):
        raise ValueError(f"Invalid {context}: Inf")
    if p < 0.0:
        raise ValueError(f"Invalid {context}: negative value {p}")
    if p > 1.0:
        raise ValueError(f"Invalid {context}: {p} out of range [0.0, 1.0]")
    return p


def validate_size(s: float, context: str = "size") -> float:
    """
    Validate a size/quantity value is non-negative and finite.

    Args:
        s: Size value to validate (must be >= 0).
        context: Description of what this value represents (for error messages).

    Returns:
        The validated size value.

    Raises:
        ValueError: If size is NaN, infinite, or negative.
    """
    if math.isnan(s):
        raise ValueError(f"Invalid {context}: NaN")
    if math.isinf(s):
        raise ValueError(f"Invalid {context}: Inf")
    if s < 0.0:
        raise ValueError(f"Invalid {context}: negative value {s}")
    return s


def validate_gas_gwei(g: float, context: str = "gas price") -> float:
    """
    Validate a gas price in gwei is non-negative and finite.

    Gas prices can spike very high during network congestion, but values
    above 10,000 gwei are likely data errors.

    Args:
        g: Gas price in gwei to validate.
        context: Description of what this value represents (for error messages).

    Returns:
        The validated gas price.

    Raises:
        ValueError: If gas price is NaN, infinite, negative, or > 10,000.
    """
    if math.isnan(g):
        raise ValueError(f"Invalid {context}: NaN")
    if math.isinf(g):
        raise ValueError(f"Invalid {context}: Inf")
    if g < 0.0:
        raise ValueError(f"Invalid {context}: negative value {g}")
    if g > 10000.0:
        raise ValueError(f"Invalid {context}: {g} gwei out of range (max 10000)")
    return g
