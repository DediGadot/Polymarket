"""
Maker strategy scanner for binary markets.

Detects wide bid-ask spreads where placing GTC limit orders at bid+1tick
on both YES and NO sides costs less than $1.00. If both fill, guaranteed
$1.00 payout at resolution.

Unlike taker arbs (FAK), this uses GTC limit orders that improve the spread,
earning the maker rebate and capturing wider edges. The tradeoff is fill
uncertainty — orders may not fill or may partially fill.
"""

from __future__ import annotations

import logging

from scanner.fees import MarketFeeModel
from scanner.models import (
    LegOrder,
    Market,
    Opportunity,
    OpportunityType,
    OrderBook,
    Side,
    is_market_stale,
)

logger = logging.getLogger(__name__)


def scan_maker_opportunities(
    markets: list[Market],
    books: dict[str, OrderBook],
    fee_model: MarketFeeModel | None = None,
    min_edge_usd: float = 0.01,
    gas_cost_per_order: float = 0.005,
    min_spread_ticks: int = 2,
    min_leg_price: float = 0.05,
    min_depth_sets: float = 5.0,
) -> list[Opportunity]:
    """
    Scan binary markets for maker spread capture opportunities.

    For each binary market, checks if placing GTC limit orders at bid+1tick
    on both YES and NO sides results in a combined cost below $1.00.

    Args:
        markets: Binary markets to scan.
        books: Pre-fetched orderbooks keyed by token_id.
        fee_model: Fee model for cost calculations.
        min_edge_usd: Minimum net profit after gas.
        gas_cost_per_order: Estimated gas cost per order.
        min_spread_ticks: Minimum spread (in ticks) required to consider market.

    Returns:
        List of Opportunities sorted by net profit descending.
    """
    opps: list[Opportunity] = []

    for market in markets:
        # Skip negRisk (handled by negrisk scanner)
        if market.neg_risk:
            continue

        # Skip inactive or stale markets
        if not market.active or is_market_stale(market):
            continue

        yes_book = books.get(market.yes_token_id)
        no_book = books.get(market.no_token_id)

        if not yes_book or not no_book:
            continue

        opp = _check_maker_arb(
            market, yes_book, no_book,
            fee_model=fee_model,
            min_edge_usd=min_edge_usd,
            gas_cost_per_order=gas_cost_per_order,
            min_spread_ticks=min_spread_ticks,
            min_leg_price=min_leg_price,
            min_depth_sets=min_depth_sets,
        )
        if opp:
            opps.append(opp)

    opps.sort(key=lambda o: o.net_profit, reverse=True)
    return opps


def _check_maker_arb(
    market: Market,
    yes_book: OrderBook,
    no_book: OrderBook,
    fee_model: MarketFeeModel | None,
    min_edge_usd: float,
    gas_cost_per_order: float,
    min_spread_ticks: int,
    min_leg_price: float = 0.05,
    min_depth_sets: float = 5.0,
) -> Opportunity | None:
    """
    Check if placing GTC limit orders inside the spread is profitable.

    Strategy: post BUY YES at (yes_bid + 1tick) and BUY NO at (no_bid + 1tick).
    If both fill, cost = yes_price + no_price. If cost < $1.00, profit = $1 - cost.
    """
    if not yes_book.best_bid or not no_book.best_bid:
        return None

    # Filter near-certain markets: if either side's best ask is below
    # min_leg_price, the low-probability side will never fill a GTC order.
    if min_leg_price > 0:
        if yes_book.best_ask and yes_book.best_ask.price < min_leg_price:
            return None
        if no_book.best_ask and no_book.best_ask.price < min_leg_price:
            return None

    tick_size = float(market.min_tick_size)

    yes_bid = yes_book.best_bid.price
    no_bid = no_book.best_bid.price

    # Check spread width on both sides
    if yes_book.best_ask:
        yes_spread_ticks = round((yes_book.best_ask.price - yes_bid) / tick_size)
        if yes_spread_ticks < min_spread_ticks:
            return None

    if no_book.best_ask:
        no_spread_ticks = round((no_book.best_ask.price - no_bid) / tick_size)
        if no_spread_ticks < min_spread_ticks:
            return None

    # Our limit order prices: 1 tick inside the best bid
    yes_price = round(yes_bid + tick_size, 6)
    no_price = round(no_bid + tick_size, 6)

    # Combined cost must be < $1.00
    total_cost = yes_price + no_price
    if total_cost >= 1.0:
        return None

    profit_per_set = 1.0 - total_cost

    # Depth is limited by the smaller side's bid depth
    yes_depth = yes_book.best_bid.size
    no_depth = no_book.best_bid.size
    max_sets = min(yes_depth, no_depth)
    if max_sets <= 0:
        return None

    # Filter micro-depth phantom arbs
    if max_sets < min_depth_sets:
        return None

    # Gas cost: 2 orders (YES + NO)
    gas_cost = gas_cost_per_order * 2

    # Fee adjustment
    legs = (
        LegOrder(
            token_id=market.yes_token_id,
            side=Side.BUY,
            price=yes_price,
            size=max_sets,
            tick_size=market.min_tick_size,
        ),
        LegOrder(
            token_id=market.no_token_id,
            side=Side.BUY,
            price=no_price,
            size=max_sets,
            tick_size=market.min_tick_size,
        ),
    )

    if fee_model:
        net_profit_per_set = fee_model.adjust_profit(profit_per_set, legs, market=market)
    else:
        net_profit_per_set = profit_per_set

    gross_profit = profit_per_set * max_sets
    net_profit = net_profit_per_set * max_sets - gas_cost
    required_capital = total_cost * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_edge_usd:
        return None

    logger.debug(
        "MAKER SPREAD: %s | yes_bid=%.4f no_bid=%.4f cost=%.4f profit/set=%.4f sets=%.0f net=$%.4f roi=%.2f%%",
        market.question[:50], yes_price, no_price, total_cost,
        profit_per_set, max_sets, net_profit, roi_pct,
    )

    return Opportunity(
        type=OpportunityType.MAKER_REBALANCE,
        event_id=market.event_id,
        legs=legs,
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=net_profit_per_set,
        max_sets=max_sets,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )
