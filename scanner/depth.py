"""
Multi-level orderbook analysis. Looks beyond best bid/ask to find arbs
hiding 2-3 levels deep in the book. Provides VWAP fill pricing for
realistic cost estimation.
"""

from __future__ import annotations

import logging

from scanner.models import OrderBook, Side

logger = logging.getLogger(__name__)


def sweep_cost(book: OrderBook, side: Side, target_size: float) -> float:
    """
    Total cost to fill target_size walking the book.
    For BUY: walks asks from best up.
    For SELL: walks bids from best down.
    Returns total USD cost. Raises ValueError if insufficient depth.
    """
    levels = book.asks if side == Side.BUY else book.bids
    remaining = target_size
    total_cost = 0.0

    for level in levels:
        fill = min(remaining, level.size)
        total_cost += fill * level.price
        remaining -= fill
        if remaining <= 0:
            break

    if remaining > 0:
        raise ValueError(
            f"Insufficient depth for {book.token_id}: need {target_size:.1f}, "
            f"have {target_size - remaining:.1f}"
        )
    return total_cost


def slippage_ceiling(
    base_price: float,
    edge_pct: float,
    side: Side,
    slippage_fraction: float = 0.4,
    max_slippage_pct: float = 3.0,
    fee_pct: float = 0.0,
) -> float:
    """
    Compute a price ceiling (for BUY) or floor (for SELL) proportional to
    the arb edge. Wider edges tolerate more slippage, thin edges stay tight.

    For BUY side: returns max acceptable ask price.
    For SELL side: returns min acceptable bid price.

    slippage_fraction: fraction of edge to allow as slippage (0.4 = 40%).
    max_slippage_pct: hard cap on slippage percentage.
    fee_pct: fee percentage to deduct from edge before computing slippage budget.
        E.g. 2.0 for Polymarket's 2% resolution fee. Tightens slippage ceiling
        so scanners don't accept fills that would be unprofitable after fees.
    """
    net_edge = abs(edge_pct) - fee_pct
    if net_edge <= 0:
        net_edge = 0.0
    allowed_slip_pct = min(net_edge * slippage_fraction, max_slippage_pct)
    slip_ratio = allowed_slip_pct / 100.0
    if side == Side.BUY:
        return base_price * (1.0 + slip_ratio)
    else:
        return base_price * (1.0 - slip_ratio)


def sweep_depth(book: OrderBook, side: Side, max_price: float) -> float:
    """
    Total available size up to price ceiling (for BUY) or floor (for SELL).
    For BUY: sum all ask sizes where price <= max_price.
    For SELL: sum all bid sizes where price >= max_price.
    """
    levels = book.asks if side == Side.BUY else book.bids
    total = 0.0

    for level in levels:
        if side == Side.BUY and level.price > max_price:
            break
        if side == Side.SELL and level.price < max_price:
            break
        total += level.size

    return total


def effective_price(book: OrderBook, side: Side, size: float) -> float | None:
    """
    VWAP fill price for given size, walking the book from best level.
    Returns None if insufficient depth.
    """
    levels = book.asks if side == Side.BUY else book.bids
    if not levels:
        return None

    remaining = size
    total_cost = 0.0

    for level in levels:
        fill = min(remaining, level.size)
        total_cost += fill * level.price
        remaining -= fill
        if remaining <= 0:
            break

    if remaining > 0:
        return None  # insufficient depth

    return total_cost / size


def worst_fill_price(book: OrderBook, side: Side, size: float) -> float | None:
    """
    Price of the last level needed to fill `size`. This is the correct limit price
    for order submission -- unlike VWAP (average), this ensures ALL levels up to
    the target size will be swept.

    For BUY: highest ask price needed. For SELL: lowest bid price needed.
    Returns None if insufficient depth.
    """
    levels = book.asks if side == Side.BUY else book.bids
    if not levels:
        return None

    remaining = size
    last_price = None

    for level in levels:
        fill = min(remaining, level.size)
        last_price = level.price
        remaining -= fill
        if remaining <= 0:
            break

    if remaining > 0:
        return None  # insufficient depth

    return last_price


def depth_profile(book: OrderBook, side: Side) -> list[tuple[float, float]]:
    """
    Cumulative (price, cumulative_size) pairs for the given side.
    For BUY: ascending asks. For SELL: descending bids.
    """
    levels = book.asks if side == Side.BUY else book.bids
    result: list[tuple[float, float]] = []
    cumulative = 0.0

    for level in levels:
        cumulative += level.size
        result.append((level.price, cumulative))

    return result


def find_deep_binary_arb(
    yes_book: OrderBook,
    no_book: OrderBook,
    target_size: float,
) -> tuple[float, float, float] | None:
    """
    Find arb at target_size by walking both books.
    Returns (effective_yes_ask, effective_no_ask, arb_size) or None.
    The arb exists if effective_yes_ask + effective_no_ask < 1.0.
    """
    yes_vwap = effective_price(yes_book, Side.BUY, target_size)
    no_vwap = effective_price(no_book, Side.BUY, target_size)

    if yes_vwap is None or no_vwap is None:
        return None

    if yes_vwap + no_vwap >= 1.0:
        return None

    return (yes_vwap, no_vwap, target_size)


def find_deep_negrisk_arb(
    books: dict[str, OrderBook],
    token_ids: list[str],
    target_size: float,
) -> tuple[list[float], float] | None:
    """
    Find NegRisk arb at target_size by walking all YES books.
    Returns (list_of_vwaps, arb_size) or None.
    Arb exists if sum(vwaps) < 1.0.
    """
    vwaps: list[float] = []
    for tid in token_ids:
        book = books.get(tid)
        if not book:
            return None
        vwap = effective_price(book, Side.BUY, target_size)
        if vwap is None:
            return None
        vwaps.append(vwap)

    if sum(vwaps) >= 1.0:
        return None

    return (vwaps, target_size)
