"""
Binary market rebalancing scanner.
Detects when YES_ask + NO_ask < 1.0 (buy arbitrage)
or YES_bid + NO_bid > 1.0 (sell arbitrage).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from client.gas import GasOracle
from scanner.depth import effective_price, sweep_depth, worst_fill_price
from scanner.fees import MarketFeeModel
from scanner.models import (
    BookFetcher,
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


def scan_binary_markets(
    book_fetcher: BookFetcher,
    markets: list[Market],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
    gas_oracle: GasOracle | None = None,
    fee_model: MarketFeeModel | None = None,
    book_cache: "BookCache | None" = None,
    min_volume: float = 0.0,
) -> list[Opportunity]:
    """
    Scan all binary (non-negRisk) markets for rebalancing arbitrage.
    Returns a list of profitable opportunities sorted by ROI descending.
    """
    # Filter to binary-only markets
    binary_markets = [m for m in markets if not m.neg_risk and m.active and not is_market_stale(m)]
    if min_volume > 0:
        binary_markets = [m for m in binary_markets if m.volume >= min_volume]
    if not binary_markets:
        return []

    # Batch-fetch all orderbooks (YES and NO tokens)
    all_token_ids = []
    for m in binary_markets:
        all_token_ids.append(m.yes_token_id)
        all_token_ids.append(m.no_token_id)

    books = book_fetcher(all_token_ids)
    if book_cache:
        book_cache.store_books(books)

    opportunities: list[Opportunity] = []
    for market in binary_markets:
        yes_book = books.get(market.yes_token_id)
        no_book = books.get(market.no_token_id)
        if not yes_book or not no_book:
            continue

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd, min_roi_pct,
            gas_per_order, gas_price_gwei,
            gas_oracle=gas_oracle, fee_model=fee_model,
        )
        if opp:
            opportunities.append(opp)

        opp = _check_sell_arb(
            market, yes_book, no_book,
            min_profit_usd, min_roi_pct,
            gas_per_order, gas_price_gwei,
            gas_oracle=gas_oracle, fee_model=fee_model,
        )
        if opp:
            opportunities.append(opp)

    opportunities.sort(key=lambda o: o.roi_pct, reverse=True)
    return opportunities


def _check_buy_arb(
    market: Market,
    yes_book: OrderBook,
    no_book: OrderBook,
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
    gas_oracle: GasOracle | None = None,
    fee_model: MarketFeeModel | None = None,
) -> Opportunity | None:
    """
    Check if buying both YES and NO is cheaper than $1.00.
    If so, buying one of each guarantees $1.00 at resolution.
    """
    yes_ask = yes_book.best_ask
    no_ask = no_book.best_ask
    if not yes_ask or not no_ask:
        return None

    # Fast pre-check using best-level prices
    cost_per_set = yes_ask.price + no_ask.price
    if cost_per_set >= 1.0:
        return None

    # Depth-aware sizing: use sweep_depth to find actual fillable size
    # within 0.5% above best ask price (slippage ceiling)
    yes_depth = sweep_depth(yes_book, Side.BUY, max_price=yes_ask.price * 1.005)
    no_depth = sweep_depth(no_book, Side.BUY, max_price=no_ask.price * 1.005)
    max_sets = min(yes_depth, no_depth)
    if max_sets <= 0:
        return None

    # VWAP-aware cost: if depth exists beyond best level, compute true fill cost
    yes_vwap = effective_price(yes_book, Side.BUY, max_sets)
    no_vwap = effective_price(no_book, Side.BUY, max_sets)
    if yes_vwap is None or no_vwap is None:
        return None

    cost_per_set = yes_vwap + no_vwap
    if cost_per_set >= 1.0:
        return None

    profit_per_set = 1.0 - cost_per_set

    # Worst-fill prices for execution limit (ensures all levels get swept)
    yes_worst = worst_fill_price(yes_book, Side.BUY, max_sets)
    no_worst = worst_fill_price(no_book, Side.BUY, max_sets)
    if yes_worst is None or no_worst is None:
        return None

    # Gas cost: real-time from oracle or fallback to static estimate
    if gas_oracle:
        gas_cost_usd = gas_oracle.estimate_cost_usd(2, gas_per_order)
    else:
        gas_cost_wei = 2 * gas_per_order * gas_price_gwei * 1e9
        gas_cost_matic = gas_cost_wei / 1e18
        gas_cost_usd = gas_cost_matic * 0.50

    # Fee adjustment -- use worst-fill for execution limit price, VWAP for profit calc
    legs = (
        LegOrder(token_id=market.yes_token_id, side=Side.BUY, price=yes_worst, size=max_sets),
        LegOrder(token_id=market.no_token_id, side=Side.BUY, price=no_worst, size=max_sets),
    )
    if fee_model:
        net_profit_per_set = fee_model.adjust_profit(profit_per_set, legs, market=market)
    else:
        net_profit_per_set = profit_per_set

    gross_profit = profit_per_set * max_sets
    net_profit = net_profit_per_set * max_sets - gas_cost_usd
    required_capital = cost_per_set * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd or roi_pct < min_roi_pct:
        return None

    logger.info(
        "BINARY BUY ARB: %s | cost=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
        market.question[:60], cost_per_set, profit_per_set, max_sets, net_profit, roi_pct,
    )

    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id=market.event_id,
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


def _check_sell_arb(
    market: Market,
    yes_book: OrderBook,
    no_book: OrderBook,
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_price_gwei: float,
    gas_oracle: GasOracle | None = None,
    fee_model: MarketFeeModel | None = None,
) -> Opportunity | None:
    """
    Check if selling both YES and NO yields more than $1.00.
    This requires holding both positions already.
    """
    yes_bid = yes_book.best_bid
    no_bid = no_book.best_bid
    if not yes_bid or not no_bid:
        return None

    # Fast pre-check using best-level prices
    proceeds_per_set = yes_bid.price + no_bid.price
    if proceeds_per_set <= 1.0:
        return None

    # Depth-aware sizing: use sweep_depth to find actual fillable size
    # within 0.5% below best bid price (slippage floor)
    yes_depth = sweep_depth(yes_book, Side.SELL, max_price=yes_bid.price * 0.995)
    no_depth = sweep_depth(no_book, Side.SELL, max_price=no_bid.price * 0.995)
    max_sets = min(yes_depth, no_depth)
    if max_sets <= 0:
        return None

    # VWAP-aware proceeds: compute true fill price walking the book
    yes_vwap = effective_price(yes_book, Side.SELL, max_sets)
    no_vwap = effective_price(no_book, Side.SELL, max_sets)
    if yes_vwap is None or no_vwap is None:
        return None

    proceeds_per_set = yes_vwap + no_vwap
    if proceeds_per_set <= 1.0:
        return None

    profit_per_set = proceeds_per_set - 1.0

    # Worst-fill prices for execution limit
    yes_worst = worst_fill_price(yes_book, Side.SELL, max_sets)
    no_worst = worst_fill_price(no_book, Side.SELL, max_sets)
    if yes_worst is None or no_worst is None:
        return None

    # Gas cost
    if gas_oracle:
        gas_cost_usd = gas_oracle.estimate_cost_usd(2, gas_per_order)
    else:
        gas_cost_wei = 2 * gas_per_order * gas_price_gwei * 1e9
        gas_cost_matic = gas_cost_wei / 1e18
        gas_cost_usd = gas_cost_matic * 0.50

    # Fee adjustment -- use worst-fill for execution limit, VWAP for profit calc
    legs = (
        LegOrder(token_id=market.yes_token_id, side=Side.SELL, price=yes_worst, size=max_sets),
        LegOrder(token_id=market.no_token_id, side=Side.SELL, price=no_worst, size=max_sets),
    )
    if fee_model:
        net_profit_per_set = fee_model.adjust_profit(
            profit_per_set, legs, market=market, is_sell=True,
        )
    else:
        net_profit_per_set = profit_per_set

    gross_profit = profit_per_set * max_sets
    net_profit = net_profit_per_set * max_sets - gas_cost_usd
    required_capital = 1.0 * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd or roi_pct < min_roi_pct:
        return None

    logger.info(
        "BINARY SELL ARB: %s | proceeds=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
        market.question[:60], proceeds_per_set, profit_per_set, max_sets, net_profit, roi_pct,
    )

    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id=market.event_id,
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
