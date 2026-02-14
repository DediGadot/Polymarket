"""
Resolution sniping scanner for nearly-resolved markets.

This scanner identifies markets where the outcome is publicly knowable
(e.g., crypto price thresholds already crossed) but the market hasn't
resolved yet. By buying the winning side at a discount, we capture
guaranteed profit at resolution.

The scanner uses the OutcomeOracle to determine if outcomes are knowable,
then filters by time to resolution and minimum edge.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Callable

from scanner.fees import MarketFeeModel
from scanner.models import (
    BookFetcher,
    LegOrder,
    Market,
    Opportunity,
    OpportunityType,
    OrderBook,
    Side,
    is_market_stale,
)
from scanner.outcome_oracle import OutcomeOracle, OutcomeStatus
from scanner.validation import validate_price, validate_size

logger = logging.getLogger(__name__)


def scan_resolution_opportunities(
    markets: list[Market],
    books: dict[str, OrderBook],
    outcome_checker: Callable[[Market], OutcomeStatus],
    fee_model: MarketFeeModel,
    max_minutes_to_resolution: float = 60.0,
    min_edge_pct: float = 3.0,
    gas_cost_per_order: float = 0.005,
) -> list[Opportunity]:
    """
    Scan markets for resolution sniping opportunities.

    Args:
        markets: List of markets to scan.
        books: Pre-fetched orderbooks keyed by token_id.
        outcome_checker: Function that determines market outcome status.
        fee_model: Fee model for cost calculations.
        max_minutes_to_resolution: Only consider markets resolving within this window.
        min_edge_pct: Minimum profit edge percentage (after fees).
        gas_cost_per_order: Estimated gas cost per order.

    Returns:
        List of Opportunities, sorted by ROI descending.
    """
    opps: list[Opportunity] = []
    cutoff_time = datetime.now(timezone.utc) + timedelta(minutes=max_minutes_to_resolution)

    for market in markets:
        # Skip inactive or stale markets
        if not market.active or is_market_stale(market):
            continue

        # Filter by time to resolution
        if not _is_near_resolution(market, cutoff_time):
            continue

        # Check if outcome is knowable
        outcome = outcome_checker(market)
        if outcome == OutcomeStatus.UNKNOWN:
            continue

        # Get orderbook
        book = books.get(market.yes_token_id)
        if not book:
            continue

        # Check for profitable entry
        opp = _check_resolution_snipe(
            market=market,
            book=book,
            outcome=outcome,
            fee_model=fee_model,
            min_edge_pct=min_edge_pct,
            gas_cost=gas_cost_per_order,
            books=books,
        )
        if opp:
            opps.append(opp)
            logger.info(
                "RESOLUTION SNIPE: %s | outcome=%s price=%.2f edge=%.1f%%",
                market.question[:60], outcome.value, opp.expected_profit_per_set,
                opp.roi_pct,
            )

    # Sort by ROI descending
    opps.sort(key=lambda o: o.roi_pct, reverse=True)
    return opps


def _is_near_resolution(market: Market, cutoff_time: datetime) -> bool:
    """
    Check if market ends before the cutoff time.

    Markets without an end_date are considered not near resolution.
    """
    if not market.end_date:
        return False

    try:
        # Handle various ISO formats
        dt_str = market.end_date.replace("Z", "+00:00")
        if "T" not in dt_str:
            dt_str += "T23:59:59+00:00"
        end_dt = datetime.fromisoformat(dt_str)
        return end_dt <= cutoff_time
    except (ValueError, TypeError):
        logger.debug("Failed to parse end_date for %s: %s", market.question, market.end_date)
        return False


def _check_resolution_snipe(
    market: Market,
    book: OrderBook,
    outcome: OutcomeStatus,
    fee_model: MarketFeeModel,
    min_edge_pct: float,
    gas_cost: float,
    books: dict[str, OrderBook],
) -> Opportunity | None:
    """
    Check if we can profitably snipe the resolution.

    For CONFIRMED_YES: Buy YES if ask < (1.0 - min_edge)
    For CONFIRMED_NO: Buy NO if ask < (1.0 - min_edge)

    Returns Opportunity if profitable, None otherwise.
    """
    if outcome == OutcomeStatus.CONFIRMED_YES:
        return _check_yes_snipe(market, book, fee_model, min_edge_pct, gas_cost)
    elif outcome == OutcomeStatus.CONFIRMED_NO:
        return _check_no_snipe(market, book, fee_model, min_edge_pct, gas_cost, books)
    return None


def _check_yes_snipe(
    market: Market,
    book: OrderBook,
    fee_model: MarketFeeModel,
    min_edge_pct: float,
    gas_cost: float,
) -> Opportunity | None:
    """
    Check if buying YES is profitable when YES outcome is confirmed.

    We pay: ask_price + taker_fee + resolution_fee
    We receive: $1.00 at resolution

    Edge = 1.0 - (ask + fees) must be >= min_edge_pct% of ask
    """
    best_ask = book.best_ask
    if not best_ask:
        return None

    price = best_ask.price
    size = best_ask.size

    # Calculate edge
    taker_fee_rate = fee_model.get_taker_fee(market, price)
    taker_fee = taker_fee_rate * price
    resolution_fee = fee_model.estimate_resolution_fee(0)
    total_cost = price + taker_fee + resolution_fee

    edge = 1.0 - total_cost
    edge_pct = (edge / price) * 100 if price > 0 else 0

    if edge_pct < min_edge_pct:
        return None

    # Profit calculations
    profit_per_set = edge
    gross_profit = profit_per_set * size
    net_profit = gross_profit - gas_cost
    capital = total_cost * size
    roi = (net_profit / capital * 100) if capital > 0 else 0

    if net_profit <= 0:
        return None

    return Opportunity(
        type=OpportunityType.RESOLUTION_SNIPE,
        event_id=market.event_id,
        legs=(LegOrder(
            token_id=market.yes_token_id,
            side=Side.BUY,
            price=price,
            size=size,
        ),),
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=profit_per_set,  # Already fee-adjusted
        max_sets=size,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost,
        net_profit=net_profit,
        roi_pct=roi,
        required_capital=capital,
    )


def _check_no_snipe(
    market: Market,
    book: OrderBook,
    fee_model: MarketFeeModel,
    min_edge_pct: float,
    gas_cost: float,
    books: dict[str, OrderBook],
) -> Opportunity | None:
    """
    Check if buying NO is profitable when NO outcome is confirmed.

    For binary markets, NO price ≈ 1 - YES price.
    We prefer to use the actual NO orderbook if available.
    """
    # Try to get NO orderbook
    no_book = books.get(market.no_token_id)
    if no_book and no_book.best_ask:
        no_ask = no_book.best_ask
        price = no_ask.price
        size = no_ask.size
        token_id = market.no_token_id
    else:
        # Fallback: estimate NO price from YES bid
        best_bid = book.best_bid
        if not best_bid:
            return None
        price = 1.0 - best_bid.price
        size = best_bid.size
        token_id = market.no_token_id

    # Validate price range
    if price <= 0 or price >= 1.0:
        return None

    # Calculate edge (NO pays $1.00 if correct)
    taker_fee_rate = fee_model.get_taker_fee(market, price)
    taker_fee = taker_fee_rate * price
    resolution_fee = fee_model.estimate_resolution_fee(0)
    total_cost = price + taker_fee + resolution_fee

    edge = 1.0 - total_cost
    edge_pct = (edge / price) * 100 if price > 0 else 0

    if edge_pct < min_edge_pct:
        return None

    # Profit calculations
    profit_per_set = edge
    gross_profit = profit_per_set * size
    net_profit = gross_profit - gas_cost
    capital = total_cost * size
    roi = (net_profit / capital * 100) if capital > 0 else 0

    if net_profit <= 0:
        return None

    return Opportunity(
        type=OpportunityType.RESOLUTION_SNIPE,
        event_id=market.event_id,
        legs=(LegOrder(
            token_id=token_id,
            side=Side.BUY,
            price=price,
            size=size,
        ),),
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=profit_per_set,
        max_sets=size,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost,
        net_profit=net_profit,
        roi_pct=roi,
        required_capital=capital,
    )
