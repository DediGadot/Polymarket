"""
Fanatics Markets client (powered by Crypto.com CDNA).

CFTC-regulated prediction market launched Dec 2025. Event contract API
endpoints are TBD -- all trading methods raise NotImplementedError.
The pipeline gracefully skips Fanatics until the API ships.

Satisfies the PlatformClient protocol so cross-platform scanning
automatically includes Fanatics once the stubs are replaced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from client.fanatics_auth import FanaticsAuth
from scanner.models import BookFetcher, OrderBook

logger = logging.getLogger(__name__)

_NOT_AVAILABLE = "Fanatics event contract API not yet available"


@dataclass(frozen=True)
class FanaticsMarket:
    """Minimal representation of a Fanatics market."""
    ticker: str
    event_ticker: str
    title: str
    status: str  # "open", "closed", "settled"


class FanaticsClient:
    """
    Fanatics Markets REST client.

    All trading methods raise NotImplementedError until the event contract
    API is published. Auth scaffolding is ready for immediate integration.
    """

    def __init__(
        self,
        auth: FanaticsAuth,
        host: str = "",
    ) -> None:
        self._auth = auth
        self._host = host.rstrip("/") if host else ""

    @property
    def platform_name(self) -> str:
        return "fanatics"

    @property
    def book_fetcher(self) -> BookFetcher:
        """Return a BookFetcher-compatible callable for scanner use."""
        return self.get_orderbooks

    # -- Market Discovery --

    def get_all_markets(self, status: str = "open") -> list[FanaticsMarket]:
        """Fetch all active markets."""
        raise NotImplementedError(_NOT_AVAILABLE)

    # -- Orderbook --

    def get_orderbook(self, ticker: str) -> OrderBook:
        """Fetch orderbook for a single ticker."""
        raise NotImplementedError(_NOT_AVAILABLE)

    def get_orderbooks(self, tickers: list[str]) -> dict[str, OrderBook]:
        """Fetch orderbooks for multiple tickers."""
        raise NotImplementedError(_NOT_AVAILABLE)

    # -- Orders --

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str = "buy",
        count: int = 1,
        type: str = "limit",
        **kwargs,
    ) -> dict:
        """Place an order on Fanatics."""
        raise NotImplementedError(_NOT_AVAILABLE)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a single order by ID."""
        raise NotImplementedError(_NOT_AVAILABLE)

    def get_order(self, order_id: str) -> dict:
        """Get order status by ID."""
        raise NotImplementedError(_NOT_AVAILABLE)

    # -- Account --

    def get_positions(self) -> list[dict]:
        """Get current positions."""
        raise NotImplementedError(_NOT_AVAILABLE)

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        raise NotImplementedError(_NOT_AVAILABLE)
