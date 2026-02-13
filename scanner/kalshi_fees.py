"""
Kalshi fee model.

Kalshi charges a taker fee per contract based on a parabolic formula:
  fee_per_contract = ceil(0.07 * C * P * (1 - P))
where:
  C = number of contracts
  P = execution price (0 to 1)

Max fee: ~1.75 cents per contract at P=0.50
No resolution fee (unlike Polymarket's 2%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scanner.models import LegOrder


# Kalshi fee parameters
KALSHI_FEE_FACTOR = 0.07  # 7% base rate


@dataclass(frozen=True)
class KalshiFeeModel:
    """Kalshi taker fee model: ceil(0.07 * C * P * (1-P)) per contract."""

    @property
    def platform_name(self) -> str:
        return "kalshi"

    @property
    def has_resolution_fee(self) -> bool:
        return False

    def taker_fee_per_contract(self, price: float) -> float:
        """
        Compute taker fee per contract in dollars.

        Args:
            price: Execution price in dollars (0.01-0.99)

        Returns:
            Fee per contract in dollars.
        """
        price = max(0.01, min(0.99, price))
        # Formula: ceil(0.07 * 100 * p * (1-p)) cents -> dollars
        # Actually: fee_cents = ceil(7 * p * (1-p)) where p is 0-1
        fee_cents = math.ceil(KALSHI_FEE_FACTOR * 100 * price * (1 - price))
        return fee_cents / 100.0

    def total_fee(self, price: float, contracts: int) -> float:
        """
        Compute total taker fee for an order.

        Args:
            price: Execution price in dollars (0.01-0.99)
            contracts: Number of contracts

        Returns:
            Total fee in dollars.
        """
        return self.taker_fee_per_contract(price) * contracts

    def adjust_profit(
        self,
        gross_profit_per_set: float,
        legs: tuple[LegOrder, ...],
    ) -> float:
        """
        Adjust profit per set after Kalshi taker fees on each leg.
        No resolution fee on Kalshi.
        """
        total_fee_per_set = 0.0
        for leg in legs:
            total_fee_per_set += self.taker_fee_per_contract(leg.price)
        return gross_profit_per_set - total_fee_per_set
