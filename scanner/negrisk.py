"""
NegRisk multi-outcome rebalancing scanner.
Detects when sum of all YES ask prices in a multi-outcome event < 1.0.
This is the highest-value arbitrage type on Polymarket (73% of historical profits).
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


def scan_negrisk_events(
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
    max_legs: int = 0,
    event_market_counts: dict[str, int] | None = None,
) -> list[Opportunity]:
    """
    Scan all negRisk events for rebalancing arbitrage.
    A negRisk event has multiple mutually exclusive outcomes.
    If sum(YES_ask) < 1.0, buy all YES tokens for guaranteed profit.
    If sum(YES_bid) > 1.0, sell all YES tokens for guaranteed profit.

    max_legs: skip events with more active markets than this (0 = no limit).
    event_market_counts: expected active market count per event from events API.
        Events with fewer markets than expected are skipped (incomplete data).
    """
    negrisk_events = [e for e in events if e.neg_risk and len(e.markets) >= 2]
    if not negrisk_events:
        return []

    # Build one global token set and fetch all books in a single batch pass.
    event_tokens: dict[str, list[str]] = {}
    event_active_markets: dict[str, list[Market]] = {}
    all_yes_token_ids: list[str] = []
    for event in negrisk_events:
        active_markets = [m for m in event.markets if m.active and not is_market_stale(m)]
        if min_volume > 0:
            active_markets = [m for m in active_markets if m.volume >= min_volume]

        # Event completeness check: skip outcome groups where the bot has
        # fewer markets than the total (active + inactive) for that
        # negRiskMarketId.  Inactive markets represent "Other" / placeholder
        # outcomes whose probability is priced into the gap between
        # sum(active asks) and $1.00.  Treating that gap as arb is a false
        # positive.
        if event_market_counts:
            nrm_key = event.neg_risk_market_id or event.event_id
            expected_total = event_market_counts.get(nrm_key, 0)
            if expected_total > 0 and len(active_markets) < expected_total:
                logger.debug(
                    "SKIP incomplete outcome group %s: have %d/%d markets (inactive outcomes exist)",
                    event.title[:50], len(active_markets), expected_total,
                )
                continue

        # Early exit: skip oversized events before fetching books
        if max_legs > 0 and len(active_markets) > max_legs:
            logger.debug(
                "SKIP %s: %d legs exceeds max %d",
                event.title[:50], len(active_markets), max_legs,
            )
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

        opp = _check_buy_all_arb(
            event, books, active_markets,
            min_profit_usd, min_roi_pct,
            gas_per_order, gas_price_gwei,
            gas_oracle=gas_oracle, fee_model=fee_model,
        )
        if opp:
            opportunities.append(opp)

        opp = _check_sell_all_arb(
            event, books, active_markets,
            min_profit_usd, min_roi_pct,
            gas_per_order, gas_price_gwei,
            gas_oracle=gas_oracle, fee_model=fee_model,
        )
        if opp:
            opportunities.append(opp)

    opportunities.sort(key=lambda o: o.roi_pct, reverse=True)
    return opportunities


def _check_buy_all_arb(
    event: Event,
    books: dict[str, OrderBook],
    active_markets: list[Market],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
    gas_oracle: GasOracle | None = None,
    fee_model: MarketFeeModel | None = None,
) -> Opportunity | None:
    """
    Buy one YES share in every outcome of a negRisk event.
    Exactly one outcome will resolve to $1, so guaranteed payout = $1.
    If total cost < $1, that's arbitrage.
    active_markets: pre-filtered list from the outer function (consistent filtering).
    """
    if len(active_markets) < 2:
        return None  # Need at least 2 active outcomes for multi-outcome arb

    ask_prices: list[tuple[Market, float, float]] = []  # (market, price, depth)

    for market in active_markets:
        book = books.get(market.yes_token_id)
        if not book or not book.best_ask:
            return None  # Can't price all legs -- skip entire event
        # Depth-aware: available size within 0.5% above best ask
        depth = sweep_depth(book, Side.BUY, max_price=book.best_ask.price * 1.005)
        ask_prices.append((market, book.best_ask.price, depth))

    # Fast pre-check using best-level prices
    total_cost = sum(price for _, price, _ in ask_prices)
    if total_cost >= 1.0:
        return None

    max_sets = min(depth for _, _, depth in ask_prices)
    if max_sets <= 0:
        return None

    # VWAP-aware cost: compute true fill cost walking the books
    vwap_prices: list[tuple[Market, float]] = []
    for market, _, _ in ask_prices:
        book = books[market.yes_token_id]
        vwap = effective_price(book, Side.BUY, max_sets)
        if vwap is None:
            return None
        vwap_prices.append((market, vwap))

    total_cost = sum(vwap for _, vwap in vwap_prices)
    if total_cost >= 1.0:
        return None

    profit_per_set = 1.0 - total_cost
    n_legs = len(vwap_prices)

    # Worst-fill prices for execution limits
    worst_prices: list[tuple[Market, float]] = []
    for market, _ in vwap_prices:
        book = books[market.yes_token_id]
        worst = worst_fill_price(book, Side.BUY, max_sets)
        if worst is None:
            return None
        worst_prices.append((market, worst))

    # Gas cost
    if gas_oracle:
        gas_cost_usd = gas_oracle.estimate_cost_usd(n_legs, gas_per_order)
    else:
        gas_cost_wei = n_legs * gas_per_order * gas_price_gwei * 1e9
        gas_cost_matic = gas_cost_wei / 1e18
        gas_cost_usd = gas_cost_matic * 0.50

    # Use worst-fill for execution limit, VWAP for profit calc
    legs = tuple(
        LegOrder(
            token_id=market.yes_token_id,
            side=Side.BUY,
            price=worst,
            size=max_sets,
        )
        for market, worst in worst_prices
    )

    # Fee adjustment (use VWAP-based legs for fee calc)
    vwap_legs = tuple(
        LegOrder(token_id=market.yes_token_id, side=Side.BUY, price=vwap, size=max_sets)
        for market, vwap in vwap_prices
    )
    if fee_model:
        event_markets = [m for m, _, _ in ask_prices]
        net_profit_per_set = fee_model.adjust_profit(profit_per_set, vwap_legs, markets=event_markets)
    else:
        net_profit_per_set = profit_per_set

    gross_profit = profit_per_set * max_sets
    net_profit = net_profit_per_set * max_sets - gas_cost_usd
    required_capital = total_cost * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd or roi_pct < min_roi_pct:
        return None

    logger.debug(
        "NEGRISK BUY ARB: %s | %d outcomes | cost=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
        event.title[:50], n_legs, total_cost, profit_per_set, max_sets, net_profit, roi_pct,
    )

    return Opportunity(
        type=OpportunityType.NEGRISK_REBALANCE,
        event_id=event.event_id,
        legs=legs,
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=net_profit_per_set,
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
    active_markets: list[Market],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
    gas_oracle: GasOracle | None = None,
    fee_model: MarketFeeModel | None = None,
) -> Opportunity | None:
    """
    Sell one YES share in every outcome of a negRisk event.
    If total proceeds > $1, that's arbitrage (requires holding all YES positions).
    active_markets: pre-filtered list from the outer function (consistent filtering).
    """
    if len(active_markets) < 2:
        return None  # Need at least 2 active outcomes for multi-outcome arb

    bid_prices: list[tuple[Market, float, float]] = []

    for market in active_markets:
        book = books.get(market.yes_token_id)
        if not book or not book.best_bid:
            return None
        # Depth-aware: available size within 0.5% below best bid
        depth = sweep_depth(book, Side.SELL, max_price=book.best_bid.price * 0.995)
        bid_prices.append((market, book.best_bid.price, depth))

    # Fast pre-check using best-level prices
    total_proceeds = sum(price for _, price, _ in bid_prices)
    if total_proceeds <= 1.0:
        return None

    max_sets = min(depth for _, _, depth in bid_prices)
    if max_sets <= 0:
        return None

    # VWAP-aware proceeds: compute true fill price walking the books
    vwap_prices: list[tuple[Market, float]] = []
    for market, _, _ in bid_prices:
        book = books[market.yes_token_id]
        vwap = effective_price(book, Side.SELL, max_sets)
        if vwap is None:
            return None
        vwap_prices.append((market, vwap))

    total_proceeds = sum(vwap for _, vwap in vwap_prices)
    if total_proceeds <= 1.0:
        return None

    profit_per_set = total_proceeds - 1.0
    n_legs = len(vwap_prices)

    # Worst-fill prices for execution limits
    worst_prices: list[tuple[Market, float]] = []
    for market, _ in vwap_prices:
        book = books[market.yes_token_id]
        worst = worst_fill_price(book, Side.SELL, max_sets)
        if worst is None:
            return None
        worst_prices.append((market, worst))

    # Gas cost
    if gas_oracle:
        gas_cost_usd = gas_oracle.estimate_cost_usd(n_legs, gas_per_order)
    else:
        gas_cost_wei = n_legs * gas_per_order * gas_price_gwei * 1e9
        gas_cost_matic = gas_cost_wei / 1e18
        gas_cost_usd = gas_cost_matic * 0.50

    # Use worst-fill for execution limit, VWAP for profit calc
    legs = tuple(
        LegOrder(
            token_id=market.yes_token_id,
            side=Side.SELL,
            price=worst,
            size=max_sets,
        )
        for market, worst in worst_prices
    )

    # Fee adjustment (use VWAP-based legs for fee calc)
    vwap_legs = tuple(
        LegOrder(token_id=market.yes_token_id, side=Side.SELL, price=vwap, size=max_sets)
        for market, vwap in vwap_prices
    )
    if fee_model:
        event_markets = [m for m, _, _ in bid_prices]
        net_profit_per_set = fee_model.adjust_profit(
            profit_per_set, vwap_legs, markets=event_markets, is_sell=True,
        )
    else:
        net_profit_per_set = profit_per_set

    gross_profit = profit_per_set * max_sets
    net_profit = net_profit_per_set * max_sets - gas_cost_usd
    required_capital = 1.0 * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd or roi_pct < min_roi_pct:
        return None

    logger.debug(
        "NEGRISK SELL ARB: %s | %d outcomes | proceeds=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
        event.title[:50], n_legs, total_proceeds, profit_per_set, max_sets, net_profit, roi_pct,
    )

    return Opportunity(
        type=OpportunityType.NEGRISK_REBALANCE,
        event_id=event.event_id,
        legs=legs,
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=net_profit_per_set,
        max_sets=max_sets,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost_usd,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )
