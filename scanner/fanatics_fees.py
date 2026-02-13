"""
Fanatics fee model.

Placeholder: uses the same parabolic taker fee formula as Kalshi until
the real Fanatics fee schedule is published.

  fee_per_contract = ceil(0.07 * 100 * P * (1 - P)) cents
  No resolution fee (assumed, like Kalshi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scanner.models import LegOrder

# Same formula as Kalshi until real fee schedule published
_FEE_FACTOR = 0.07


@dataclass(frozen=True)
class FanaticsFeeModel:
    """Fanatics taker fee model (placeholder: Kalshi-equivalent parabolic)."""

    @property
    def platform_name(self) -> str:
        return "fanatics"

    @property
    def has_resolution_fee(self) -> bool:
        return False

    def taker_fee_per_contract(self, price: float) -> float:
        """Compute taker fee per contract in dollars."""
        price = max(0.01, min(0.99, price))
        fee_cents = math.ceil(_FEE_FACTOR * 100 * price * (1 - price))
        return fee_cents / 100.0

    def total_fee(self, price: float, contracts: int) -> float:
        """Compute total taker fee for an order."""
        return self.taker_fee_per_contract(price) * contracts

    def adjust_profit(
        self,
        gross_profit_per_set: float,
        legs: tuple[LegOrder, ...],
    ) -> float:
        """Adjust profit per set after taker fees. No resolution fee."""
        total_fee_per_set = 0.0
        for leg in legs:
            total_fee_per_set += self.taker_fee_per_contract(leg.price)
        return gross_profit_per_set - total_fee_per_set
