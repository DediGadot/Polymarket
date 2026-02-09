"""
CLOB REST client wrapper. Thin layer converting SDK types to our domain models.
"""

from __future__ import annotations

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    MarketOrderArgs,
    OrderType,
    BookParams,
    PartialCreateOrderOptions,
)
from py_clob_client.order_builder.constants import BUY, SELL

from scanner.models import OrderBook, PriceLevel, Side


def _sort_book_levels(
    raw_bids: list, raw_asks: list,
) -> tuple[tuple[PriceLevel, ...], tuple[PriceLevel, ...]]:
    """
    Convert raw SDK levels to sorted PriceLevel tuples.
    Asks: ascending by price (best/lowest first at index 0).
    Bids: descending by price (best/highest first at index 0).
    The SDK does NOT guarantee sort order -- we must enforce it.
    """
    bids = tuple(sorted(
        (PriceLevel(price=float(b.price), size=float(b.size)) for b in (raw_bids or [])),
        key=lambda lvl: lvl.price,
        reverse=True,
    ))
    asks = tuple(sorted(
        (PriceLevel(price=float(a.price), size=float(a.size)) for a in (raw_asks or [])),
        key=lambda lvl: lvl.price,
    ))
    return bids, asks


def get_orderbook(client: ClobClient, token_id: str) -> OrderBook:
    """Fetch full orderbook for a token and convert to our OrderBook model."""
    raw = client.get_order_book(token_id)
    bids, asks = _sort_book_levels(raw.bids, raw.asks)
    return OrderBook(token_id=token_id, bids=bids, asks=asks)


BOOK_BATCH_SIZE = 50  # Max token IDs per /books request to avoid payload limit


def get_orderbooks(client: ClobClient, token_ids: list[str]) -> dict[str, OrderBook]:
    """Fetch orderbooks for multiple tokens, chunking to avoid payload limits."""
    result = {}
    for i in range(0, len(token_ids), BOOK_BATCH_SIZE):
        chunk = token_ids[i:i + BOOK_BATCH_SIZE]
        params = [BookParams(token_id=tid) for tid in chunk]
        raws = client.get_order_books(params)
        for raw in raws:
            tid = raw.asset_id
            bids, asks = _sort_book_levels(raw.bids, raw.asks)
            result[tid] = OrderBook(token_id=tid, bids=bids, asks=asks)
    return result


def get_orderbooks_parallel(
    client: ClobClient,
    token_ids: list[str],
    max_workers: int = 8,
) -> dict[str, OrderBook]:
    """
    Fetch orderbooks for multiple tokens using parallel threads.
    Chunks token_ids into BOOK_BATCH_SIZE groups and fetches them concurrently.
    Same output as get_orderbooks() but significantly faster for large token lists.
    """
    if not token_ids:
        return {}

    from concurrent.futures import ThreadPoolExecutor

    chunks: list[list[str]] = []
    for i in range(0, len(token_ids), BOOK_BATCH_SIZE):
        chunks.append(token_ids[i:i + BOOK_BATCH_SIZE])

    def _fetch_chunk(chunk: list[str]) -> dict[str, OrderBook]:
        params = [BookParams(token_id=tid) for tid in chunk]
        raws = client.get_order_books(params)
        result: dict[str, OrderBook] = {}
        for raw in raws:
            tid = raw.asset_id
            bids, asks = _sort_book_levels(raw.bids, raw.asks)
            result[tid] = OrderBook(token_id=tid, bids=bids, asks=asks)
        return result

    result: dict[str, OrderBook] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_chunk, chunk) for chunk in chunks]
        for future in futures:
            # .result() propagates exceptions (fail-fast)
            result.update(future.result())
    return result


def get_midpoint(client: ClobClient, token_id: str) -> float:
    """Get mid-market price for a token."""
    resp = client.get_midpoint(token_id)
    return float(resp["mid"])


def create_limit_order(
    client: ClobClient,
    token_id: str,
    side: Side,
    price: float,
    size: float,
    neg_risk: bool = False,
    tick_size: str = "0.01",
) -> object:
    """Create and sign a limit order. Returns a SignedOrder ready to post."""
    args = OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side=BUY if side == Side.BUY else SELL,
    )
    options = PartialCreateOrderOptions(
        tick_size=tick_size,
        neg_risk=neg_risk,
    )
    return client.create_order(args, options)


def create_market_order(
    client: ClobClient,
    token_id: str,
    side: Side,
    amount: float,
    neg_risk: bool = False,
    tick_size: str = "0.01",
) -> object:
    """Create and sign a FOK market order. Returns a SignedOrder ready to post."""
    args = MarketOrderArgs(
        token_id=token_id,
        amount=amount,
        side=BUY if side == Side.BUY else SELL,
    )
    options = PartialCreateOrderOptions(
        tick_size=tick_size,
        neg_risk=neg_risk,
    )
    return client.create_market_order(args, options)


def post_order(
    client: ClobClient,
    signed_order: object,
    order_type: OrderType = OrderType.GTC,
) -> dict:
    """Post a signed order to the CLOB. Returns the response dict."""
    return client.post_order(signed_order, order_type)


def post_orders(
    client: ClobClient,
    signed_orders: list[tuple[object, OrderType]],
) -> list:
    """
    Post multiple signed orders in a single batch (max 15).
    Each item is (signed_order, order_type).
    """
    from py_clob_client.clob_types import PostOrdersArgs

    args = [
        PostOrdersArgs(order=so, orderType=ot)
        for so, ot in signed_orders
    ]
    return client.post_orders(args)


def cancel_order(client: ClobClient, order_id: str) -> dict:
    """Cancel a single order by ID."""
    return client.cancel(order_id)


def cancel_all(client: ClobClient) -> dict:
    """Cancel all open orders."""
    return client.cancel_all()


def get_balance_allowance(client: ClobClient) -> dict:
    """Get current USDC balance and allowance status."""
    return client.get_balance_allowance()
