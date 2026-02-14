"""
Market fee model for Polymarket. Updated with DCM parabolic fee.

Fee types:
- Standard markets: 0% taker fee
- 15-min crypto markets: Dynamic taker fee (up to ~3.15% at p=0.15)
- DCM markets (Daily Crypto): Parabolic taker fee based on price
  - 0.10% at p=0.50 (max uncertainty)
  - 0.01% at p=0.90 (min uncertainty)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from scanner.models import Market, LegOrder


logger = logging.getLogger(__name__)


# Regex patterns for 15-min crypto market detection
_CRYPTO_15MIN_PATTERNS = [
    re.compile(r"\b(BTC|Bitcoin)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\b.*\b(min|minute)", re.IGNORECASE),
    re.compile(r"\b(ETH|Ethereum)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\b.*\b(min|minute)", re.IGNORECASE),
    re.compile(r"\b(SOL|Solana)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\b.*\b(min|minute)", re.IGNORECASE),
]

# Max dynamic fee rate at 50/50 odds for 15-min crypto markets
MAX_CRYPTO_FEE_RATE = 0.0315  # 3.15%

# Polymarket US DCM flat taker fee (10bps)
DCM_FEE_RATE = 0.001  # 0.10%

# Resolution fee on winning positions (2% of $1 payout)
RESOLUTION_FEE_RATE = 0.02


@dataclass
class MarketFeeModel:
    """Determines per-trade taker fees based on market type and trade price."""

    enabled: bool = True

    def is_dcm_market(self, market: Market) -> bool:
        """Detect if market is a DCM (Daily Crypto Market) with parabolic fee."""
        return market.min_tick_size == "0.001"

    def get_dcm_fee_rate(self, price: float) -> float:
        """
        Parabolic taker fee for DCM markets.
        Fee increases as price approaches 50/50 odds (maximum uncertainty).
        At price=0.50: fee = 0.10%. At price=0.90: fee = 0.01%.

        Formula: fee = DCMMAX_FEE * (1 - |price - MIDPOINT| * FACTOR)
        where DCMMAX_FEE = 0.001 (0.10%), MIDPOINT = 0.50, FACTOR = 2.0
        Capped at 0.01% minimum.
        """
        DCM_MAX_FEE = 0.001  # 0.10% at p=0.50
        MIDPOINT = 0.50
        FACTOR = 2.0  # Controls parabolic shape

        # Parabolic curve: highest fee at midpoint, dropping toward edges
        price_distance = abs(price - MIDPOINT)
        dcm_fee = DCM_MAX_FEE * (1 - price_distance * FACTOR)

        # Cap at 0.01% minimum (1bp) and 0.10% maximum (10bps)
        return max(0.0001, min(0.001, dcm_fee))

    def is_crypto_15min(self, market: Market) -> bool:
        """Detect if market is a 15-minute crypto prediction market."""
        question = market.question.lower()
        for pattern in _CRYPTO_15MIN_PATTERNS:
            if pattern.search(question):
                return True
        return False

    def get_crypto_fee_rate(self, price: float) -> float:
        """
        Dynamic fee for 15-min crypto markets.
        Fee increases as price approaches 50/50 odds (maximum uncertainty).

        Formula: fee = MAX_RATE * 4 * price * (1 - price)
        """
        return MAX_CRYPTO_FEE_RATE * 4.0 * price * (1.0 - price)

    def get_taker_fee(self, market: Market, price: float) -> float:
        """
        Return taker fee rate for a given market and trade price.

        Returns:
            - DCM markets: parabolic fee based on price
            - 15-min crypto markets: dynamic fee based on price
            - Standard markets: 0% fee
        """
        if not self.enabled:
            return 0.0

        # DCM markets have parabolic taker fee
        if self.is_dcm_market(market):
            return self.get_dcm_fee_rate(price)

        # 15-min crypto markets have dynamic fee
        if self.is_crypto_15min(market):
            return self.get_crypto_fee_rate(price)

        # Standard markets: zero taker fee
        return 0.0

    def estimate_resolution_fee(self, profit_per_set: float) -> float:
        """
        Estimate resolution fee on winning position.
        The 2% fee applies to winning payout ($1), not just profit.

        For arb: we buy all sides, so exactly one wins $1. Fee = $0.02 per set.
        """
        return RESOLUTION_FEE_RATE * 1.0  # $0.02 per set won

    def adjust_profit(
        self,
        gross_profit_per_set: float,
        legs: tuple[LegOrder, ...],
        market: Market | None = None,
        markets: list[Market] | None = None,
        is_sell: bool = False,
    ) -> float:
        """
        Compute net profit per set after taker fees on each leg.

        For binary arb: market is single market.
        For negrisk arb: markets is list of all event markets.
        """
        total_fee_per_set = 0.0

        if markets is not None and len(markets) == 1:
            # Binary arb: single market
            mkt = markets[0]
            for leg in legs:
                fee_rate = self.get_taker_fee(mkt, leg.price)
                total_fee_per_set += fee_rate * leg.price

        elif markets is not None and len(markets) > 1:
            # NegRisk arb: one market per leg
            for i, leg in enumerate(legs):
                if i >= len(markets):
                    break
                mkt = markets[i]
                fee_rate = self.get_taker_fee(mkt, leg.price)
                total_fee_per_set += fee_rate * leg.price

        else:
            # NegRisk arb with single market or cross-platform (no market info)
            for leg in legs:
                # Get market from leg if possible
                if markets is not None and len(markets) == 1:
                    mkt = markets[0]
                else:
                    mkt = market
                if mkt is not None:
                    fee_rate = self.get_taker_fee(mkt, leg.price)
                else:
                    fee_rate = self.get_taker_fee(mkt, leg.price)
                total_fee_per_set += fee_rate * leg.price

        # Resolution fee only applies to buy-side arbs (holder pays at resolution)
        if is_sell:
            resolution_fee = 0.0
        else:
            resolution_fee = self.estimate_resolution_fee(gross_profit_per_set)

        return gross_profit_per_set - total_fee_per_set - resolution_fee


def compute_dcm_fee_examples():
    """
    Compute example DCM parabolic fees at various price points.
    For verification and testing.
    """
    model = MarketFeeModel()

    examples = [
        (0.10, model.get_dcm_fee_rate(0.10)),   # p=0.10 -> max fee = 0.001 (10bps)
        (0.25, model.get_dcm_fee_rate(0.25)),   # p=0.25 -> mid fee = 0.0005 (5bps)
        (0.40, model.get_dcm_fee_rate(0.40)),   # p=0.40 -> decreasing
        (0.50, model.get_dcm_fee_rate(0.50)),   # p=0.50 -> max fee = 0.001 (10bps)
        (0.60, model.get_dcm_fee_rate(0.60)),   # p=0.60 -> decreasing
        (0.75, model.get_dcm_fee_rate(0.75)),   # p=0.75 -> lower
        (0.90, model.get_dcm_fee_rate(0.90)),   # p=0.90 -> min fee = 0.0001 (1bp)
    ]

    print("DCM Parabolic Fee Examples:")
    print("Price\tFee\tFee%")
    for price, fee in examples:
        print(f"{price:.2f}\t{fee:.4f}\t{fee * 100:.2f}%")

    return examples


if __name__ == "__main__":
    compute_dcm_fee_examples()
