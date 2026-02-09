"""
Market fee model for Polymarket. Detects fee-bearing markets and adjusts
opportunity profitability after per-leg fee deductions.

Fee types:
- Standard markets: 0% taker fee
- 15-min crypto markets (BTC/ETH/SOL): Dynamic taker fee, highest (~3.15%) at 50/50 odds,
  drops toward 0% at extreme odds (0% or 100%)
- Polymarket US DCM: Flat 10bps (0.10%) taker fee
- All markets: 2% fee on winning positions at resolution
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from scanner.models import Market, LegOrder

logger = logging.getLogger(__name__)

# Regex patterns for 15-min crypto market detection
_CRYPTO_15MIN_PATTERNS = [
    re.compile(r"\b(BTC|Bitcoin)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\s*(min|minute)", re.IGNORECASE),
    re.compile(r"\b(ETH|Ethereum)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\s*(min|minute)", re.IGNORECASE),
    re.compile(r"\b(SOL|Solana)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\s*(min|minute)", re.IGNORECASE),
    re.compile(r"\b(15|fifteen)\s*(min|minute).*\b(BTC|ETH|SOL|Bitcoin|Ethereum|Solana)\b", re.IGNORECASE),
]

# Max dynamic fee rate at 50/50 odds for crypto 15-min markets
MAX_CRYPTO_FEE_RATE = 0.0315  # 3.15%

# Polymarket US DCM flat taker fee
DCM_FEE_RATE = 0.0010  # 10bps

# Resolution fee on winning positions
RESOLUTION_FEE_RATE = 0.02  # 2%


@dataclass
class MarketFeeModel:
    """Determines per-trade taker fees based on market type and trade price."""

    enabled: bool = True

    def is_crypto_15min(self, market: Market) -> bool:
        """Detect if market is a 15-minute crypto prediction market."""
        return any(p.search(market.question) for p in _CRYPTO_15MIN_PATTERNS)

    def get_taker_fee(self, market: Market, price: float) -> float:
        """
        Return taker fee rate for a given market and trade price.
        Returns a decimal (e.g., 0.0315 for 3.15%).
        """
        if not self.enabled:
            return 0.0

        if self.is_crypto_15min(market):
            return self._dynamic_crypto_fee(price)

        # Standard markets: zero fee
        return 0.0

    def _dynamic_crypto_fee(self, price: float) -> float:
        """
        Dynamic fee for 15-min crypto markets.
        Highest at price=0.50 (3.15%), drops parabolically toward 0 at price=0 or price=1.
        Formula: fee = MAX_RATE * 4 * price * (1 - price)
        At price=0.50: fee = 0.0315 * 4 * 0.5 * 0.5 = 0.0315
        At price=0.25: fee = 0.0315 * 4 * 0.25 * 0.75 = 0.02363
        At price=0.10: fee = 0.0315 * 4 * 0.1 * 0.9 = 0.01134
        """
        price = max(0.0, min(1.0, price))
        return MAX_CRYPTO_FEE_RATE * 4.0 * price * (1.0 - price)

    def estimate_resolution_fee(self, profit_per_set: float) -> float:
        """
        Estimate resolution fee on winning position.
        The 2% fee applies to the winning payout ($1), not just profit.
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
        For binary arb: market is the single market.
        For negrisk arb: markets is the list of all event markets.
        is_sell: True for sell-side arbs (no resolution fee -- seller exits before resolution).
        """
        total_fee_per_set = 0.0

        if market is not None:
            # Binary: same market for both legs
            for leg in legs:
                fee_rate = self.get_taker_fee(market, leg.price)
                total_fee_per_set += fee_rate * leg.price
        elif markets is not None:
            # NegRisk: one market per leg
            for i, leg in enumerate(legs):
                mkt = markets[i] if i < len(markets) else None
                if mkt:
                    fee_rate = self.get_taker_fee(mkt, leg.price)
                    total_fee_per_set += fee_rate * leg.price

        # Resolution fee only applies to buy-side arbs (holder pays at resolution).
        # Sell-side arbs exit positions before resolution -- no resolution fee.
        if is_sell:
            resolution_fee = 0.0
        else:
            resolution_fee = self.estimate_resolution_fee(gross_profit_per_set)
        return gross_profit_per_set - total_fee_per_set - resolution_fee
