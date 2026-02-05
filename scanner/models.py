"""
Data models for the arbitrage scanner. Pure data, no behavior.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field


class OpportunityType(Enum):
    BINARY_REBALANCE = "binary_rebalance"
    NEGRISK_REBALANCE = "negrisk_rebalance"


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


@dataclass(frozen=True)
class Event:
    event_id: str
    title: str
    markets: tuple[Market, ...]
    neg_risk: bool


@dataclass(frozen=True)
class LegOrder:
    token_id: str
    side: Side
    price: float
    size: float


@dataclass(frozen=True)
class Opportunity:
    type: OpportunityType
    event_id: str
    legs: tuple[LegOrder, ...]
    expected_profit_per_set: float
    max_sets: float
    gross_profit: float
    estimated_gas_cost: float
    net_profit: float
    roi_pct: float
    required_capital: float
    timestamp: float = field(default_factory=time.time)

    @property
    def is_profitable(self) -> bool:
        return self.net_profit > 0 and self.roi_pct > 0


@dataclass
class TradeResult:
    opportunity: Opportunity
    order_ids: list[str]
    fill_prices: list[float]
    fill_sizes: list[float]
    fees: float
    gas_cost: float
    net_pnl: float
    execution_time_ms: float
    fully_filled: bool
    timestamp: float = field(default_factory=time.time)
