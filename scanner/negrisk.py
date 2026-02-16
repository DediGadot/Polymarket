"""
NegRisk multi-outcome rebalancing scanner.
Detects when sum of all YES ask prices in a multi-outcome event < 1.0.
This is the highest-value arbitrage type on Polymarket (73% of historical profits).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from client.gas import GasOracle
from scanner.depth import effective_price, slippage_ceiling, sweep_depth, worst_fill_price
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
    slippage_fraction: float = 0.4,
    max_slippage_pct: float = 3.0,
    large_event_subset_enabled: bool = False,
    large_event_max_subset: int = 15,
    large_event_tail_max_prob: float = 0.05,
    should_stop: Callable[[], bool] | None = None,
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
    event_payout_caps: dict[str, float] = {}
    event_risk_flags: dict[str, list[str]] = {}
    all_yes_token_ids: list[str] = []
    event_oversized: dict[str, bool] = {}
    for event in negrisk_events:
        if should_stop and should_stop():
            return []
        active_markets = [m for m in event.markets if m.active and not is_market_stale(m)]
        if min_volume > 0:
            active_markets = [m for m in active_markets if m.volume >= min_volume]

        # Event completeness check: skip outcome groups where the bot has
        # fewer markets than the total (active + inactive) for that
        # negRiskMarketId.  Inactive markets represent "Other" / placeholder
        # outcomes whose probability is priced into the gap between
        # sum(active asks) and $1.00.
        payout_cap = 1.0
        risk_flags_list = []
        if event_market_counts is not None:
            nrm_key = event.neg_risk_market_id or event.event_id
            expected_total = event_market_counts.get(nrm_key, 0)
            if expected_total == 0:
                # Unknown completeness -- skip conservatively to avoid false positives
                logger.debug(
                    "SKIP %s: unknown market count for neg_risk_market_id=%s (expected_total=0)",
                    event.title[:50], nrm_key,
                )
                continue
            if len(active_markets) < expected_total:
                missing = expected_total - len(active_markets)
                # If missing only 1-2 markets, we can still arb if the edge is wide enough
                # to cover the theoretical maximum price of those missing markets.
                # Use $0.01 (min tick) as a conservative buffer per missing market.
                if missing <= 2:
                    payout_cap = 1.0 - (missing * 0.01)
                    risk_flags_list.append(f"incomplete_group:missing_{missing}")
                    logger.debug(
                        "INCOMPLETE outcome group %s: have %d/%d, using payout_cap=%.2f",
                        event.title[:50], len(active_markets), expected_total, payout_cap,
                    )
                else:
                    logger.debug(
                        "SKIP incomplete outcome group %s: have %d/%d markets (too many missing)",
                        event.title[:50], len(active_markets), expected_total,
                    )
                    continue

        # Oversized events (> max_legs) can optionally run in bounded subset mode.
        if max_legs > 0 and len(active_markets) > max_legs:
            if not large_event_subset_enabled:
                logger.debug(
                    "SKIP %s: %d legs exceeds max %d",
                    event.title[:50], len(active_markets), max_legs,
                )
                continue
            event_oversized[event.event_id] = True

        yes_token_ids = [m.yes_token_id for m in active_markets]
        if len(yes_token_ids) < 2:
            continue
        event_tokens[event.event_id] = yes_token_ids
        event_active_markets[event.event_id] = active_markets
        event_payout_caps[event.event_id] = payout_cap
        event_risk_flags[event.event_id] = risk_flags_list
        all_yes_token_ids.extend(yes_token_ids)
    if not all_yes_token_ids:
        return []

    dedup_yes_token_ids = list(dict.fromkeys(all_yes_token_ids))
    all_books = book_fetcher(dedup_yes_token_ids)
    if book_cache:
        book_cache.store_books(all_books)

    opportunities: list[Opportunity] = []
    for event in negrisk_events:
        if should_stop and should_stop():
            return opportunities
        yes_token_ids = event_tokens.get(event.event_id)
        if not yes_token_ids:
            continue

        books = {tid: all_books[tid] for tid in yes_token_ids if tid in all_books}
        active_markets = event_active_markets[event.event_id]
        oversized = event_oversized.get(event.event_id, False)
        payout_cap = event_payout_caps.get(event.event_id, 1.0)
        reason_code = ""
        risk_flags = tuple(event_risk_flags.get(event.event_id, []))

        if oversized:
            subset = _build_large_event_subset(
                active_markets=active_markets,
                books=books,
                max_subset=min(max_legs if max_legs > 0 else large_event_max_subset, large_event_max_subset),
                tail_max_prob=large_event_tail_max_prob,
            )
            if subset is None:
                continue
            active_markets, payout_cap, tail_prob = subset
            reason_code = "negrisk_large_event_subset"
            risk_flags = ("large_event_subset", f"tail_prob<=${tail_prob:.4f}")

        opp = _check_buy_all_arb(
            event, books, active_markets,
            min_profit_usd, min_roi_pct,
            gas_per_order, gas_price_gwei,
            gas_oracle=gas_oracle, fee_model=fee_model,
            slippage_fraction=slippage_fraction, max_slippage_pct=max_slippage_pct,
            payout_cap=payout_cap,
            reason_code=reason_code,
            risk_flags=risk_flags,
        )
        if opp:
            opportunities.append(opp)

        # SELL subsets on large events carry path-dependent downside; keep buy-only.
        if not oversized:
            opp = _check_sell_all_arb(
                event, books, active_markets,
                min_profit_usd, min_roi_pct,
                gas_per_order, gas_price_gwei,
                gas_oracle=gas_oracle, fee_model=fee_model,
                slippage_fraction=slippage_fraction, max_slippage_pct=max_slippage_pct,
            )
            if opp:
                opportunities.append(opp)

    opportunities.sort(key=lambda o: o.roi_pct, reverse=True)
    return opportunities


def _build_large_event_subset(
    *,
    active_markets: list[Market],
    books: dict[str, OrderBook],
    max_subset: int,
    tail_max_prob: float,
) -> tuple[list[Market], float, float] | None:
    """
    Build a bounded subset basket for oversized events.

    Returns:
      (subset_markets, payout_cap, tail_prob)
    where payout_cap = 1 - omitted_tail_prob (conservative proxy).
    """
    priced: list[tuple[Market, float, float]] = []
    for market in active_markets:
        book = books.get(market.yes_token_id)
        if not book or not book.best_ask:
            continue
        priced.append((market, book.best_ask.price, book.best_ask.size))

    if len(priced) < 2:
        return None

    priced.sort(key=lambda x: x[1])  # cheapest asks first
    subset_n = max(2, min(max_subset, len(priced)))
    subset = priced[:subset_n]
    omitted = priced[subset_n:]
    tail_prob = sum(price for _, price, _ in omitted)
    if tail_prob > tail_max_prob:
        return None

    payout_cap = max(0.0, 1.0 - tail_prob)
    if payout_cap <= 0:
        return None

    return [m for m, _, _ in subset], payout_cap, tail_prob


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
    slippage_fraction: float = 0.4,
    max_slippage_pct: float = 3.0,
    payout_cap: float = 1.0,
    reason_code: str = "",
    risk_flags: tuple[str, ...] = (),
) -> Opportunity | None:
    """
    Buy one YES share in every outcome of a negRisk event.
    Exactly one outcome will resolve to $1, so guaranteed payout = $1.
    If total cost < payout_cap, that's arbitrage under the configured payout cap.
    active_markets: pre-filtered list from the outer function (consistent filtering).
    """
    if len(active_markets) < 2:
        return None  # Need at least 2 active outcomes for multi-outcome arb

    # First pass: collect best-ask prices for edge calculation
    best_asks: list[tuple[Market, float]] = []
    for market in active_markets:
        book = books.get(market.yes_token_id)
        if not book or not book.best_ask:
            return None
        best_asks.append((market, book.best_ask.price))

    # Fast pre-check using best-level prices
    total_cost = sum(price for _, price in best_asks)
    if total_cost >= payout_cap:
        return None

    # Edge-proportional slippage: wider edges tolerate more slippage
    edge_pct = ((payout_cap - total_cost) / total_cost) * 100.0

    # Second pass: compute depth with edge-proportional ceiling
    ask_prices: list[tuple[Market, float, float]] = []
    for market, best_price in best_asks:
        book = books[market.yes_token_id]
        ceiling = slippage_ceiling(best_price, edge_pct, Side.BUY, slippage_fraction, max_slippage_pct, fee_pct=2.0)
        depth = sweep_depth(book, Side.BUY, max_price=ceiling)
        ask_prices.append((market, best_price, depth))

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
    if total_cost >= payout_cap:
        return None

    profit_per_set = payout_cap - total_cost
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
        reason_code=reason_code,
        risk_flags=risk_flags,
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
    slippage_fraction: float = 0.4,
    max_slippage_pct: float = 3.0,
) -> Opportunity | None:
    """
    Sell one YES share in every outcome of a negRisk event.
    If total proceeds > $1, that's arbitrage (requires holding all YES positions).
    active_markets: pre-filtered list from the outer function (consistent filtering).
    """
    if len(active_markets) < 2:
        return None  # Need at least 2 active outcomes for multi-outcome arb

    # First pass: collect best-bid prices for edge calculation
    best_bids: list[tuple[Market, float]] = []
    for market in active_markets:
        book = books.get(market.yes_token_id)
        if not book or not book.best_bid:
            return None
        best_bids.append((market, book.best_bid.price))

    # Fast pre-check
    total_proceeds = sum(price for _, price in best_bids)
    if total_proceeds <= 1.0:
        return None

    # Edge-proportional slippage
    edge_pct = ((total_proceeds - 1.0) / 1.0) * 100.0

    bid_prices: list[tuple[Market, float, float]] = []
    for market, best_price in best_bids:
        book = books[market.yes_token_id]
        floor = slippage_ceiling(best_price, edge_pct, Side.SELL, slippage_fraction, max_slippage_pct, fee_pct=2.0)
        depth = sweep_depth(book, Side.SELL, max_price=floor)
        bid_prices.append((market, best_price, depth))

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
