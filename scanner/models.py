"""
Data models for the arbitrage scanner. Pure data, no behavior.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class Platform(Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    FANATICS = "fanatics"


class OpportunityType(Enum):
    BINARY_REBALANCE = "binary_rebalance"
    NEGRISK_REBALANCE = "negrisk_rebalance"
    NEGRISK_VALUE = "negrisk_value"
    LATENCY_ARB = "latency_arb"
    SPIKE_LAG = "spike_lag"
    CROSS_PLATFORM_ARB = "cross_platform_arb"
    RESOLUTION_SNIPE = "resolution_snipe"
    STALE_QUOTE_ARB = "stale_quote_arb"
    MAKER_REBALANCE = "maker_rebalance"
    CORRELATION_ARB = "correlation_arb"


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class PriceLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBook:
    token_id: str
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]

    @property
    def best_bid(self) -> PriceLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> PriceLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def midpoint(self) -> float | None:
        if self.best_bid and self.best_ask:
            return (self.best_ask.price + self.best_bid.price) / 2.0
        return None


# BookFetcher: callable that fetches orderbooks for a list of token IDs.
# Used to decouple scanners from specific platform clients.
BookFetcher = Callable[[list[str]], dict[str, "OrderBook"]]


@dataclass(frozen=True)
class Market:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    neg_risk: bool
    event_id: str
    min_tick_size: str  # "0.01" or "0.001"
    active: bool
    volume: float = 0.0
    end_date: str = ""   # ISO 8601 from Gamma API (empty = unknown)
    closed: bool = False  # True if market already resolved
    neg_risk_market_id: str = ""  # Groups mutually exclusive outcomes (separates moneyline from spread/totals)


_stale_cache: dict[str, tuple[bool, float]] = {}


def is_market_stale(market: Market) -> bool:
    """
    Check if a market is stale (expired or already resolved).
    Returns True if the market should be skipped.

    Caches results by end_date string for 60 seconds. Many markets share
    the same resolution date, so caching eliminates ~93% of datetime
    parsing across 15K+ markets per scan cycle.
    """
    if market.closed:
        return True
    if not market.end_date:
        return False

    now_mono = time.monotonic()
    cached = _stale_cache.get(market.end_date)
    if cached is not None:
        result, cached_at = cached
        if now_mono - cached_at < 60.0:
            return result

    from datetime import datetime, timezone
    try:
        dt_str = market.end_date.replace("Z", "+00:00")
        if "T" not in dt_str:
            dt_str += "T23:59:59+00:00"
        end_dt = datetime.fromisoformat(dt_str)
        now = datetime.now(timezone.utc)
        result = end_dt < now
    except (ValueError, TypeError):
        result = False

    _stale_cache[market.end_date] = (result, now_mono)
    return result


@dataclass(frozen=True)
class Event:
    event_id: str
    title: str
    markets: tuple[Market, ...]
    neg_risk: bool
    neg_risk_market_id: str = ""  # Grouping key for mutually exclusive outcomes


@dataclass(frozen=True)
class LegOrder:
    token_id: str
    side: Side
    price: float
    size: float
    platform: str = ""
    tick_size: str = "0.01"  # "0.01" or "0.001"


@dataclass(frozen=True)
class Opportunity:
    type: OpportunityType
    event_id: str
    legs: tuple[LegOrder, ...]
    expected_profit_per_set: float  # GROSS profit per set (before fees)
    net_profit_per_set: float       # NET profit per set (after taker + resolution fees, before gas)
    max_sets: float
    gross_profit: float
    estimated_gas_cost: float
    net_profit: float
    roi_pct: float
    required_capital: float
    pair_fill_prob: float = 1.0
    toxicity_score: float = 0.0
    expected_realized_net: float = 0.0
    quote_theoretical_net: float = 0.0
    # Optional classifier/debug tags used by research-oriented scanners
    # (e.g., correlation and large-event basket candidates).
    reason_code: str = ""
    risk_flags: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)

    @property
    def is_profitable(self) -> bool:
        return self.net_profit > 0 and self.roi_pct > 0

    @property
    def is_sell_arb(self) -> bool:
        """True if all legs are SELL (requires holding inventory)."""
        return len(self.legs) > 0 and all(leg.side == Side.SELL for leg in self.legs)

    @property
    def is_buy_arb(self) -> bool:
        """True if all legs are BUY (actionable without inventory)."""
        return len(self.legs) > 0 and all(leg.side == Side.BUY for leg in self.legs)


@dataclass(frozen=True)
class TradeResult:
    opportunity: Opportunity
    order_ids: tuple[str, ...]
    fill_prices: tuple[float, ...]
    fill_sizes: tuple[float, ...]
    fees: float
    gas_cost: float
    net_pnl: float
    execution_time_ms: float
    fully_filled: bool
    timestamp: float = field(default_factory=time.time)
