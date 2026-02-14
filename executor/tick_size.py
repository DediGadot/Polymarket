"""
Tick size quantization for order execution.
Ensures prices conform to market tick sizes before order placement.
"""

from __future__ import annotations


class TickSizeExceededError(ValueError):
    """Raised when quantization would shift price by more than tick_size / 2."""
    pass


def quantize_price(price: float, tick_size: float) -> float:
    """
    Round a price to the nearest valid tick.

    Args:
        price: The desired price (0.00 to 1.00 inclusive).
        tick_size: The minimum price increment (e.g., 0.01 or 0.001).

    Returns:
        The quantized price, rounded to the nearest tick.

    Raises:
        ValueError: If price is negative, above 1.0, or tick_size is invalid.
        TickSizeExceededError: If quantization would shift price by > tick_size / 2.
    """
    # Validate inputs
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")

    if price < 0:
        raise ValueError(f"price must be non-negative, got {price}")

    if price > 1.0:
        raise ValueError(f"price must not exceed 1.0, got {price}")

    # Round to nearest tick using Python's round() which uses round-half-to-even
    quantized = round(price / tick_size) * tick_size

    # Clamp to [0, 1] range in case of floating point edge cases
    quantized = max(0.0, min(1.0, quantized))

    # Check if quantization shift is acceptable (within tick_size / 2)
    shift = abs(quantized - price)
    max_shift = tick_size / 2.0

    # Allow a small epsilon for floating point precision
    if shift > max_shift + 1e-12:
        raise TickSizeExceededError(
            f"Price {price} quantized to {quantized} (shift {shift:.6f}) "
            f"exceeds tick_size/2 ({max_shift:.6f})"
        )

    return quantized
