"""
NegRisk multi-outcome rebalancing scanner.
Detects when sum of all YES ask prices in a multi-outcome event < 1.0.
This is the highest-value arbitrage type on Polymarket (73% of historical profits).
"""

from __future__ import annotations

import logging
import time

from py_clob_client.client import ClobClient

from client.clob import get_orderbooks
from scanner.models import (
    Event,
    Market,
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
    OrderBook,
)

logger = logging.getLogger(__name__)


def scan_negrisk_events(
    client: ClobClient,
    events: list[Event],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
) -> list[Opportunity]:
    """
    Scan all negRisk events for rebalancing arbitrage.
    A negRisk event has multiple mutually exclusive outcomes.
    If sum(YES_ask) < 1.0, buy all YES tokens for guaranteed profit.
    If sum(YES_bid) > 1.0, sell all YES tokens for guaranteed profit.
    """
    negrisk_events = [e for e in events if e.neg_risk and len(e.markets) >= 2]
    if not negrisk_events:
        return []

    opportunities: list[Opportunity] = []
    for event in negrisk_events:
        # Batch-fetch all YES token orderbooks for this event
        yes_token_ids = [m.yes_token_id for m in event.markets if m.active]
        if len(yes_token_ids) < 2:
            continue

        books = get_orderbooks(client, yes_token_ids)

        opp = _check_buy_all_arb(
            event, books,
            min_profit_usd, min_roi_pct,
            gas_per_order, gas_price_gwei,
        )
        if opp:
            opportunities.append(opp)

        opp = _check_sell_all_arb(
            event, books,
            min_profit_usd, min_roi_pct,
            gas_per_order, gas_price_gwei,
        )
        if opp:
            opportunities.append(opp)

    opportunities.sort(key=lambda o: o.roi_pct, reverse=True)
    return opportunities


def _check_buy_all_arb(
    event: Event,
    books: dict[str, OrderBook],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
) -> Opportunity | None:
    """
    Buy one YES share in every outcome of a negRisk event.
    Exactly one outcome will resolve to $1, so guaranteed payout = $1.
    If total cost < $1, that's arbitrage.
    """
    active_markets = [m for m in event.markets if m.active]
    ask_prices: list[tuple[Market, float, float]] = []  # (market, price, size)

    for market in active_markets:
        book = books.get(market.yes_token_id)
        if not book or not book.best_ask:
            return None  # Can't price all legs -- skip entire event
        ask_prices.append((market, book.best_ask.price, book.best_ask.size))

    total_cost = sum(price for _, price, _ in ask_prices)
    if total_cost >= 1.0:
        return None

    profit_per_set = 1.0 - total_cost
    max_sets = min(size for _, _, size in ask_prices)
    if max_sets <= 0:
        return None

    gross_profit = profit_per_set * max_sets
    n_legs = len(ask_prices)

    gas_cost_wei = n_legs * gas_per_order * gas_price_gwei * 1e9
    gas_cost_matic = gas_cost_wei / 1e18
    gas_cost_usd = gas_cost_matic * 0.50

    net_profit = gross_profit - gas_cost_usd
    required_capital = total_cost * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd or roi_pct < min_roi_pct:
        return None

    logger.info(
        "NEGRISK BUY ARB: %s | %d outcomes | cost=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
        event.title[:50], n_legs, total_cost, profit_per_set, max_sets, net_profit, roi_pct,
    )

    legs = tuple(
        LegOrder(
            token_id=market.yes_token_id,
            side=Side.BUY,
            price=price,
            size=max_sets,
        )
        for market, price, _ in ask_prices
    )

    return Opportunity(
        type=OpportunityType.NEGRISK_REBALANCE,
        event_id=event.event_id,
        legs=legs,
        expected_profit_per_set=profit_per_set,
        max_sets=max_sets,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost_usd,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )


def _check_sell_all_arb(
    event: Event,
    books: dict[str, OrderBook],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
) -> Opportunity | None:
    """
    Sell one YES share in every outcome of a negRisk event.
    If total proceeds > $1, that's arbitrage (requires holding all YES positions).
    """
    active_markets = [m for m in event.markets if m.active]
    bid_prices: list[tuple[Market, float, float]] = []

    for market in active_markets:
        book = books.get(market.yes_token_id)
        if not book or not book.best_bid:
            return None
        bid_prices.append((market, book.best_bid.price, book.best_bid.size))

    total_proceeds = sum(price for _, price, _ in bid_prices)
    if total_proceeds <= 1.0:
        return None

    profit_per_set = total_proceeds - 1.0
    max_sets = min(size for _, _, size in bid_prices)
    if max_sets <= 0:
        return None

    gross_profit = profit_per_set * max_sets
    n_legs = len(bid_prices)

    gas_cost_wei = n_legs * gas_per_order * gas_price_gwei * 1e9
    gas_cost_matic = gas_cost_wei / 1e18
    gas_cost_usd = gas_cost_matic * 0.50

    net_profit = gross_profit - gas_cost_usd
    required_capital = 1.0 * max_sets  # cost basis is $1 per set
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd or roi_pct < min_roi_pct:
        return None

    logger.info(
        "NEGRISK SELL ARB: %s | %d outcomes | proceeds=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
        event.title[:50], n_legs, total_proceeds, profit_per_set, max_sets, net_profit, roi_pct,
    )

    legs = tuple(
        LegOrder(
            token_id=market.yes_token_id,
            side=Side.SELL,
            price=price,
            size=max_sets,
        )
        for market, price, _ in bid_prices
    )

    return Opportunity(
        type=OpportunityType.NEGRISK_REBALANCE,
        event_id=event.event_id,
        legs=legs,
        expected_profit_per_set=profit_per_set,
        max_sets=max_sets,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost_usd,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )
