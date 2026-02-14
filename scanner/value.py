"""
Partial NegRisk value scanner.
Detects underpriced individual outcomes in multi-outcome NegRisk events.
Finds directional bets where an outcome's ask price is disproportionately low
compared to uniform probability, indicating potential value.

Only activates when sum(best_asks) > 1.0 (no risk-free arb exists).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from client.gas import GasOracle
from scanner.depth import effective_price, sweep_depth, worst_fill_price
from scanner.fees import MarketFeeModel
from scanner.models import (
    BookFetcher,
    Event,
    Market,
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
    OrderBook,
    is_market_stale,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from scanner.book_cache import BookCache


# Kelly odds for directional value bets (conservative since not risk-free)
VALUE_KELLY_ODDS = 0.30

# Default max position size for value opportunities (in sets)
MAX_VALUE_SETS = 10.0


def scan_value_opportunities(
    book_fetcher: BookFetcher,
    events: list[Event],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
    gas_oracle: GasOracle | None = None,
    fee_model: MarketFeeModel | None = None,
    book_cache: "BookCache | None" = None,
    min_volume: float = 0.0,
    max_events: int = 50,
    min_edge_pct: float = 10.0,
    uniform_discount_factor: float = 0.5,
    gas_cost_per_order: float = 0.005,
) -> list[Opportunity]:
    """
    Scan for underpriced individual outcomes in NegRisk events.

    Only processes events where:
    1. sum(best_asks) > 1.0 (no risk-free arb — those go to negrisk scanner)
    2. At least one outcome is priced below half the uniform probability

    For each undervalued outcome, emits a single-leg BUY opportunity with
    type=NEGRISK_VALUE.

    Args:
        min_edge_pct: Minimum edge percentage to consider an outcome undervalued
        uniform_discount_factor: Price must be below (uniform_prob * this factor)
        gas_cost_per_order: Estimated gas cost per order (USD)

    Returns:
        List of value opportunities sorted by ROI descending
    """
    negrisk_events = [e for e in events if e.neg_risk and len(e.markets) >= 2]
    if not negrisk_events:
        return []

    # Limit to max_events to avoid excessive processing
    if len(negrisk_events) > max_events:
        negrisk_events = negrisk_events[:max_events]

    # Collect all YES token IDs from negrisk events
    event_tokens: dict[str, list[str]] = {}
    event_active_markets: dict[str, list[Market]] = {}
    all_yes_token_ids: list[str] = []

    for event in negrisk_events:
        active_markets = [
            m for m in event.markets
            if m.active and not is_market_stale(m)
        ]
        if min_volume > 0:
            active_markets = [m for m in active_markets if m.volume >= min_volume]

        if len(active_markets) < 2:
            continue

        yes_token_ids = [m.yes_token_id for m in active_markets]
        if len(yes_token_ids) < 2:
            continue

        event_tokens[event.event_id] = yes_token_ids
        event_active_markets[event.event_id] = active_markets
        all_yes_token_ids.extend(yes_token_ids)

    if not all_yes_token_ids:
        return []

    dedup_yes_token_ids = list(dict.fromkeys(all_yes_token_ids))
    all_books = book_fetcher(dedup_yes_token_ids)
    if book_cache:
        book_cache.store_books(all_books)

    opportunities: list[Opportunity] = []

    for event in negrisk_events:
        yes_token_ids = event_tokens.get(event.event_id)
        if not yes_token_ids:
            continue

        books = {tid: all_books[tid] for tid in yes_token_ids if tid in all_books}
        active_markets = event_active_markets[event.event_id]

        # Find value opportunities in this event
        opps = _find_value_opportunities(
            event,
            books,
            active_markets,
            min_profit_usd,
            min_roi_pct,
            gas_per_order,
            gas_price_gwei,
            gas_oracle=gas_oracle,
            fee_model=fee_model,
            min_edge_pct=min_edge_pct,
            uniform_discount_factor=uniform_discount_factor,
            gas_cost_per_order=gas_cost_per_order,
        )
        opportunities.extend(opps)

    opportunities.sort(key=lambda o: o.roi_pct, reverse=True)
    return opportunities


def _find_value_opportunities(
    event: Event,
    books: dict[str, OrderBook],
    active_markets: list[Market],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
    gas_oracle: GasOracle | None = None,
    fee_model: MarketFeeModel | None = None,
    min_edge_pct: float = 10.0,
    uniform_discount_factor: float = 0.5,
    gas_cost_per_order: float = 0.005,
) -> list[Opportunity]:
    """
    Find undervalued individual outcomes in a single event.

    An outcome is undervalued if:
    1. The ask price is below uniform_prob * uniform_discount_factor
    2. The expected edge (based on implied probability) exceeds min_edge_pct

    Returns:
        List of Opportunity objects (one per undervalued outcome)
    """
    n_outcomes = len(active_markets)
    if n_outcomes < 2:
        return []

    # Compute sum of best asks to check for risk-free arb
    ask_prices: list[tuple[Market, float, float]] = []
    for market in active_markets:
        book = books.get(market.yes_token_id)
        if not book or not book.best_ask:
            return []  # Can't evaluate with incomplete books

        depth = sweep_depth(book, Side.BUY, max_price=book.best_ask.price * 1.005)
        ask_prices.append((market, book.best_ask.price, depth))

    total_ask = sum(price for _, price, _ in ask_prices)

    # If total_ask < 1.0, risk-free arb exists — let negrisk scanner handle it
    if total_ask < 1.0:
        return []

    # Uniform probability baseline
    uniform_prob = 1.0 / n_outcomes
    uniform_price_threshold = uniform_prob * uniform_discount_factor

    opportunities: list[Opportunity] = []

    for market, ask_price, depth in ask_prices:
        # Skip if not significantly below uniform probability
        if ask_price >= uniform_price_threshold:
            continue

        # Compute expected value edge
        # For a directional bet, edge = (implied_prob - ask_price) / ask_price
        # Implied prob is what the market "should" be at based on uniform distribution
        implied_prob = uniform_prob
        edge_dollars = implied_prob - ask_price
        edge_pct = (edge_dollars / ask_price * 100) if ask_price > 0 else 0

        if edge_pct < min_edge_pct:
            continue

        # This outcome is undervalued — create opportunity
        opp = _create_value_opportunity(
            event,
            market,
            books[market.yes_token_id],
            depth,
            min_profit_usd,
            min_roi_pct,
            gas_per_order,
            gas_price_gwei,
            gas_oracle=gas_oracle,
            fee_model=fee_model,
            gas_cost_per_order=gas_cost_per_order,
        )

        if opp:
            opportunities.append(opp)

    return opportunities


def _create_value_opportunity(
    event: Event,
    market: Market,
    book: OrderBook,
    max_depth: float,
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
    gas_oracle: GasOracle | None = None,
    fee_model: MarketFeeModel | None = None,
    gas_cost_per_order: float = 0.005,
) -> Opportunity | None:
    """
    Create a single-leg value opportunity for an undervalued outcome.

    Expected profit calculation:
    - For a value bet, we're buying one outcome at price P
    - If correct, payout = $1, profit = 1 - P - fees
    - If wrong, loss = P + fees
    - Expected profit = implied_prob * (1 - P - fees) - (1 - implied_prob) * (P + fees)
    - Simplified: expected_profit = implied_prob - P - total_fees

    For conservative sizing, we use Kelly odds = 0.30 (not the true implied prob)
    since this is a directional bet with real risk of loss.
    """
    if max_depth <= 0:
        return None

    best_ask = book.best_ask
    if not best_ask:
        return None

    ask_price = best_ask.price
    n_outcomes = len(event.markets)
    uniform_prob = 1.0 / n_outcomes

    # Compute expected profit per set (before fees)
    # EV = uniform_prob * $1 - cost
    gross_profit_per_set = uniform_prob - ask_price

    if gross_profit_per_set <= 0:
        return None  # No value

    # Size based on depth and conservative Kelly sizing
    max_sets = min(max_depth, MAX_VALUE_SETS)

    # VWAP price for realistic fill estimation
    vwap_price = effective_price(book, Side.BUY, max_sets)
    if vwap_price is None:
        return None

    # Worst-fill price for execution limit
    worst_price = worst_fill_price(book, Side.BUY, max_sets)
    if worst_price is None:
        return None

    # Re-calculate profit using VWAP
    gross_profit_per_set = uniform_prob - vwap_price
    if gross_profit_per_set <= 0:
        return None

    # Single leg
    legs = (
        LegOrder(
            token_id=market.yes_token_id,
            side=Side.BUY,
            price=worst_price,
            size=max_sets,
            tick_size=market.min_tick_size,
        ),
    )

    # Fee adjustment
    if fee_model:
        net_profit_per_set = fee_model.adjust_profit(
            gross_profit_per_set,
            legs,
            market=market,
        )
    else:
        net_profit_per_set = gross_profit_per_set

    # Gas cost (single order)
    if gas_oracle:
        gas_cost_usd = gas_oracle.estimate_cost_usd(1, gas_per_order)
    else:
        gas_cost_usd = gas_cost_per_order

    gross_profit = gross_profit_per_set * max_sets
    net_profit = net_profit_per_set * max_sets - gas_cost_usd
    required_capital = vwap_price * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd or roi_pct < min_roi_pct:
        return None

    logger.debug(
        "NEGRISK VALUE: %s | outcome=%s ask=%.4f uniform=%.4f edge=%.2f%% sets=%.1f net=$%.2f roi=%.2f%%",
        event.title[:50], market.question[:30], vwap_price, uniform_prob,
        ((uniform_prob - vwap_price) / vwap_price * 100) if vwap_price > 0 else 0,
        max_sets, net_profit, roi_pct,
    )

    return Opportunity(
        type=OpportunityType.NEGRISK_VALUE,
        event_id=event.event_id,
        legs=legs,
        expected_profit_per_set=gross_profit_per_set,
        net_profit_per_set=net_profit_per_set,
        max_sets=max_sets,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost_usd,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )
