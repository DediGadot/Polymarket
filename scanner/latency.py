"""
Latency arbitrage scanner for 15-minute crypto prediction markets.

These markets (BTC/ETH/SOL up/down in 15 minutes) reprice slowly relative
to spot exchanges. When spot momentum confirms direction, the prediction
market lags -- creating a window where real probability diverges from market price.

The $313 -> $414K bot exploited exactly this pattern with 98% win rate.
Dynamic taker fees (up to 3.15% at 50/50) were introduced to curb this,
so fee-aware execution is critical: only enter when odds have moved away
from 50/50 (where fees drop near zero).
"""

from __future__ import annotations

import logging
import re
import time

import httpx

from scanner.fees import MarketFeeModel
from scanner.models import (
    Market,
    OrderBook,
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 5.0

# Patterns matching crypto 15-min markets
_CRYPTO_PATTERNS = {
    "BTC": re.compile(r"\b(BTC|Bitcoin)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\s*(min|minute)", re.IGNORECASE),
    "ETH": re.compile(r"\b(ETH|Ethereum)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\s*(min|minute)", re.IGNORECASE),
    "SOL": re.compile(r"\b(SOL|Solana)\b.*\b(up|down|above|below)\b.*\b(15|fifteen)\s*(min|minute)", re.IGNORECASE),
}

# Direction patterns
_UP_PATTERN = re.compile(r"\b(up|above|higher|rise|over)\b", re.IGNORECASE)
_DOWN_PATTERN = re.compile(r"\b(down|below|lower|fall|under)\b", re.IGNORECASE)

# Binance spot price endpoints
_BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}


class LatencyScanner:
    """
    Detects latency arb on 15-min crypto markets by comparing spot price
    momentum against prediction market pricing.
    """

    def __init__(
        self,
        min_edge_pct: float = 5.0,
        spot_cache_sec: float = 2.0,
        fee_model: MarketFeeModel | None = None,
    ):
        self._min_edge_pct = min_edge_pct
        self._spot_cache_sec = spot_cache_sec
        self._fee_model = fee_model or MarketFeeModel()

        # Spot price cache: {symbol: (price, timestamp)}
        self._spot_cache: dict[str, tuple[float, float]] = {}
        # Previous spot prices for momentum: {symbol: (price, timestamp)}
        self._prev_spot: dict[str, tuple[float, float]] = {}

    def identify_crypto_markets(self, markets: list[Market]) -> list[tuple[Market, str, str]]:
        """
        Filter markets to 15-min crypto predictions.
        Returns list of (market, crypto_symbol, direction).
        direction is "up" or "down".
        """
        results: list[tuple[Market, str, str]] = []
        for market in markets:
            for symbol, pattern in _CRYPTO_PATTERNS.items():
                if pattern.search(market.question):
                    direction = "up" if _UP_PATTERN.search(market.question) else "down"
                    results.append((market, symbol, direction))
                    break
        return results

    def get_spot_price(self, symbol: str) -> float | None:
        """
        Fetch current spot price from Binance. Uses cache if fresh.
        Returns None on failure.
        """
        now = time.time()
        cached = self._spot_cache.get(symbol)
        if cached and (now - cached[1]) < self._spot_cache_sec:
            return cached[0]

        binance_symbol = _BINANCE_SYMBOLS.get(symbol)
        if not binance_symbol:
            return None

        try:
            resp = httpx.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": binance_symbol},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            price = float(resp.json()["price"])

            # Save previous before updating
            if symbol in self._spot_cache:
                self._prev_spot[symbol] = self._spot_cache[symbol]

            self._spot_cache[symbol] = (price, now)
            return price
        except Exception as e:
            logger.warning("Spot price fetch failed for %s: %s", symbol, e)
            return None

    def compute_momentum_pct(self, symbol: str) -> float | None:
        """
        Compute spot price momentum as % change from previous observation.
        Returns None if insufficient data.
        """
        current = self._spot_cache.get(symbol)
        previous = self._prev_spot.get(symbol)
        if not current or not previous:
            return None
        if previous[0] <= 0:
            return None
        return ((current[0] - previous[0]) / previous[0]) * 100.0

    def compute_implied_probability(
        self,
        momentum_pct: float,
        direction: str,
        threshold_pct: float = 0.0,
    ) -> float:
        """
        Estimate what the YES probability should be given spot momentum.

        If direction is "up" and momentum is positive:
          - Strong momentum (>1%) → ~85% implied probability
          - Moderate (0.3-1%) → ~65-85%
          - Weak (<0.3%) → ~50-60%
        Negative momentum for "up" → implied probability drops below 50%.
        Mirror for "down" direction.
        """
        # Align momentum with market direction
        effective = momentum_pct if direction == "up" else -momentum_pct

        # Logistic-like mapping from momentum to probability
        # Calibrated so 1% momentum ≈ 85% probability
        if effective > 0:
            # Positive momentum favoring YES outcome
            prob = 0.50 + min(effective * 0.35, 0.45)
        else:
            # Negative momentum opposing YES outcome
            prob = 0.50 + max(effective * 0.35, -0.45)

        return max(0.01, min(0.99, prob))

    def check_latency_arb(
        self,
        market: Market,
        book: OrderBook,
        symbol: str,
        direction: str,
        gas_cost_per_order: float = 0.005,
        no_books: dict[str, OrderBook] | None = None,
    ) -> Opportunity | None:
        """
        Check if a latency arb exists on this market.
        Returns Opportunity if market price lags implied probability enough.
        """
        momentum = self.compute_momentum_pct(symbol)
        if momentum is None:
            return None

        implied_prob = self.compute_implied_probability(momentum, direction)

        # Determine which side to trade
        best_ask = book.best_ask
        best_bid = book.best_bid

        if implied_prob > 0.55 and best_ask:
            # We think YES is underpriced, buy YES
            market_price = best_ask.price
            edge_pct = ((implied_prob - market_price) / market_price) * 100.0

            # Check fee-adjusted edge (taker fee + 2% resolution fee)
            fee_rate = self._fee_model.get_taker_fee(market, market_price)
            fee_cost = fee_rate * market_price
            resolution_fee = self._fee_model.estimate_resolution_fee(0)
            net_edge_per_unit = implied_prob - market_price - fee_cost - resolution_fee
            net_edge_pct = (net_edge_per_unit / market_price) * 100.0 if market_price > 0 else 0

            if net_edge_pct < self._min_edge_pct:
                return None

            size = best_ask.size
            profit_per_unit = net_edge_per_unit
            gross_profit = profit_per_unit * size
            net_profit = gross_profit - gas_cost_per_order
            capital = market_price * size
            roi = (net_profit / capital * 100) if capital > 0 else 0

            if net_profit <= 0:
                return None

            logger.info(
                "LATENCY ARB: %s | implied=%.2f market=%.2f edge=%.1f%% fee=%.2f%% net_edge=%.1f%%",
                market.question[:60], implied_prob, market_price,
                edge_pct, fee_rate * 100, net_edge_pct,
            )

            return Opportunity(
                type=OpportunityType.LATENCY_ARB,
                event_id=market.event_id,
                legs=(LegOrder(
                    token_id=market.yes_token_id,
                    side=Side.BUY,
                    price=market_price,
                    size=size,
                ),),
                expected_profit_per_set=profit_per_unit,
                net_profit_per_set=profit_per_unit,  # already fee-adjusted
                max_sets=size,
                gross_profit=gross_profit,
                estimated_gas_cost=gas_cost_per_order,
                net_profit=net_profit,
                roi_pct=roi,
                required_capital=capital,
            )

        elif implied_prob < 0.45 and best_bid:
            # We think YES is overpriced → BUY NO (avoids needing YES inventory)
            # NO price ≈ 1 - YES bid. Use NO book if available, else estimate.
            no_book = no_books.get(market.no_token_id) if no_books else None
            if no_book and no_book.best_ask:
                no_price = no_book.best_ask.price
                no_size = no_book.best_ask.size
            else:
                # Fallback: estimate NO ask from YES bid
                no_price = 1.0 - best_bid.price
                no_size = best_bid.size

            # Edge: implied NO probability vs NO market price
            implied_no = 1.0 - implied_prob
            edge_pct = ((implied_no - no_price) / no_price) * 100.0 if no_price > 0 else 0

            fee_rate = self._fee_model.get_taker_fee(market, no_price)
            fee_cost = fee_rate * no_price
            resolution_fee = self._fee_model.estimate_resolution_fee(0)
            net_edge_per_unit = implied_no - no_price - fee_cost - resolution_fee
            net_edge_pct = (net_edge_per_unit / no_price) * 100.0 if no_price > 0 else 0

            if net_edge_pct < self._min_edge_pct:
                return None

            size = no_size
            profit_per_unit = net_edge_per_unit
            gross_profit = profit_per_unit * size
            net_profit = gross_profit - gas_cost_per_order
            capital = no_price * size
            roi = (net_profit / capital * 100) if capital > 0 else 0

            if net_profit <= 0:
                return None

            logger.info(
                "LATENCY ARB (BUY NO): %s | implied_no=%.2f no_price=%.2f edge=%.1f%% net_edge=%.1f%%",
                market.question[:60], implied_no, no_price, edge_pct, net_edge_pct,
            )

            return Opportunity(
                type=OpportunityType.LATENCY_ARB,
                event_id=market.event_id,
                legs=(LegOrder(
                    token_id=market.no_token_id,
                    side=Side.BUY,
                    price=no_price,
                    size=size,
                ),),
                expected_profit_per_set=profit_per_unit,
                net_profit_per_set=profit_per_unit,  # already fee-adjusted
                max_sets=size,
                gross_profit=gross_profit,
                estimated_gas_cost=gas_cost_per_order,
                net_profit=net_profit,
                roi_pct=roi,
                required_capital=capital,
            )

        return None


def scan_latency_markets(
    latency_scanner: LatencyScanner,
    crypto_markets: list[tuple[Market, str, str]],
    books: dict[str, OrderBook],
    gas_cost_per_order: float = 0.005,
    no_books: dict[str, OrderBook] | None = None,
) -> list[Opportunity]:
    """
    Scan all identified crypto markets for latency arb.
    crypto_markets: output of identify_crypto_markets().
    books: {token_id: OrderBook} for YES tokens.
    no_books: {token_id: OrderBook} for NO tokens (used for BUY NO legs).
    """
    opps: list[Opportunity] = []
    for market, symbol, direction in crypto_markets:
        book = books.get(market.yes_token_id)
        if not book:
            continue
        opp = latency_scanner.check_latency_arb(
            market, book, symbol, direction, gas_cost_per_order,
            no_books=no_books,
        )
        if opp:
            opps.append(opp)
    return opps
