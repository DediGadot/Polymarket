"""
Platform client protocol. Thin interface extracted from KalshiClient signatures.

Any exchange client (Kalshi, Fanatics, Robinhood, etc.) that satisfies this
protocol can plug into the cross-platform pipeline with zero changes to
scanner/executor code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scanner.models import BookFetcher, OrderBook


@runtime_checkable
class PlatformClient(Protocol):
    """
    Minimal interface for a cross-platform exchange client.

    Implementations must provide market discovery, orderbook fetching,
    order placement, and account queries.
    """

    @property
    def platform_name(self) -> str:
        """Short identifier: 'kalshi', 'fanatics', etc."""
        ...

    @property
    def book_fetcher(self) -> BookFetcher:
        """Return a BookFetcher-compatible callable for scanner use."""
        ...

    def get_all_markets(self, status: str = "open") -> list:
        """Fetch all active markets. Returns platform-specific market objects."""
        ...

    def get_orderbook(self, ticker: str) -> OrderBook:
        """Fetch orderbook for a single ticker."""
        ...

    def get_orderbooks(self, tickers: list[str]) -> dict[str, OrderBook]:
        """Fetch orderbooks for multiple tickers."""
        ...

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str = "buy",
        count: int = 1,
        type: str = "limit",
        **kwargs,
    ) -> dict:
        """Place an order. Returns order response dict."""
        ...

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a single order by ID."""
        ...

    def get_order(self, order_id: str) -> dict:
        """Get order status by ID."""
        ...

    def get_positions(self) -> list[dict]:
        """Get current positions."""
        ...

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        ...
