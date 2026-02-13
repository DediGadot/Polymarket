"""
Platform fee model protocol. Thin interface for cross-platform fee calculation.

Each exchange has its own fee schedule (taker fees, resolution fees, etc.).
Implementations satisfy this protocol so the scanner can compute net profit
across any platform without knowing the fee details.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scanner.models import LegOrder


@runtime_checkable
class PlatformFeeModel(Protocol):
    """Minimal interface for a platform's fee model."""

    @property
    def platform_name(self) -> str:
        """Short identifier matching the platform client."""
        ...

    @property
    def has_resolution_fee(self) -> bool:
        """True if platform charges a fee on winning positions at resolution."""
        ...

    def taker_fee_per_contract(self, price: float) -> float:
        """Compute taker fee per contract in dollars for a given price."""
        ...

    def total_fee(self, price: float, contracts: int) -> float:
        """Compute total taker fee for an order."""
        ...

    def adjust_profit(
        self,
        gross_profit_per_set: float,
        legs: tuple[LegOrder, ...],
    ) -> float:
        """Adjust profit per set after platform-specific fees."""
        ...
