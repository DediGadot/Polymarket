"""
Outcome oracle for determining if market outcomes are publicly known.

The oracle checks external data sources to determine whether a market's
outcome is already knowable (e.g., crypto price thresholds already crossed).

This enables "resolution sniping" — buying the winning side of nearly-
resolved markets at a discount before the official resolution occurs.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

import httpx

from scanner.models import Market
from scanner.validation import validate_price, validate_size

logger = logging.getLogger(__name__)

_TIMEOUT = 5.0


class OutcomeStatus(Enum):
    """Outcome determination status for a market."""
    CONFIRMED_YES = "confirmed_yes"
    CONFIRMED_NO = "confirmed_no"
    UNKNOWN = "unknown"


# Crypto symbols and their Binance ticker mappings
_BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "BITCOIN": "BTCUSDT",
    "ETH": "ETHUSDT",
    "ETHEREUM": "ETHUSDT",
    "SOL": "SOLUSDT",
    "SOLANA": "SOLUSDT",
}

# Direction patterns for parsing market questions
_ABOVE_PATTERNS = [
    re.compile(r"\b(above|higher|over|greater than|exceeds?|surpass(?:es|ed)?|up|up to)\b", re.IGNORECASE),
]

_BELOW_PATTERNS = [
    re.compile(r"\b(below|lower|under|less than|drops?|falls?|down)\b", re.IGNORECASE),
]

# Threshold extraction patterns - captures numbers like "$50,000" or "50k"
_THRESHOLD_PATTERNS = [
    # $50,000 or 50,000 or 50000
    re.compile(r"\$?([\d,]+)\s*k(?:\b|\d)", re.IGNORECASE),  # 50k -> 50000
    re.compile(r"\$?([\d,]+)", re.IGNORECASE),  # $50,000 -> 50000
]

# Buffer around threshold to consider "clearly resolved" (1%)
_THRESHOLD_BUFFER_PCT = 0.01


@dataclass(frozen=True)
class ParsedThreshold:
    """Parsed outcome threshold from a market question."""
    symbol: str
    direction: str  # "above" or "below"
    threshold: float
    confidence: float  # 0.0 to 1.0, how confident we are in the parse


class OutcomeOracle:
    """
    Determines if market outcomes are publicly knowable from external data.

    Currently supports:
    - Crypto price markets (BTC/ETH/SOL) via Binance spot prices

    Extensibility: Add new data sources by implementing new _check_* methods
    and calling them from check_outcome().
    """

    def __init__(
        self,
        allow_network: bool = True,
        spot_cache_sec: float = 2.0,
    ):
        """
        Args:
            allow_network: If False, always returns UNKNOWN (for offline testing).
            spot_cache_sec: How long to cache spot prices before re-fetching.
        """
        self._allow_network = allow_network
        self._spot_cache_sec = spot_cache_sec

        # Spot price cache: {symbol: (price, timestamp)}
        self._spot_cache: dict[str, tuple[float, float]] = {}

    def check_outcome(self, market: Market) -> OutcomeStatus:
        """
        Determine if a market's outcome is publicly knowable.

        Returns:
            OutcomeStatus: CONFIRMED_YES/NO if outcome is determinable,
                          UNKNOWN if not yet knowable or oracle can't parse.
        """
        # Try crypto resolution first (fastest, most reliable)
        crypto_status = self._check_crypto_outcome(market)
        if crypto_status != OutcomeStatus.UNKNOWN:
            return crypto_status

        # Future: add sports, elections, etc.
        # Example: sports_status = self._check_sports_outcome(market)

        return OutcomeStatus.UNKNOWN

    def _check_crypto_outcome(self, market: Market) -> OutcomeStatus:
        """
        Check if a crypto price market's outcome is determinable from Binance spot.

        Returns:
            OutcomeStatus: CONFIRMED_YES/NO if clearly resolved, UNKNOWN otherwise.
        """
        # Parse the market question for symbol, direction, threshold
        parsed = self._parse_crypto_threshold(market.question)
        if not parsed or parsed.confidence < 0.8:
            return OutcomeStatus.UNKNOWN

        # Fetch current spot price
        spot = self._get_spot_price(parsed.symbol)
        if spot is None:
            return OutcomeStatus.UNKNOWN

        # Apply buffer zone (1%) to avoid false positives near threshold
        buffer = parsed.threshold * _THRESHOLD_BUFFER_PCT

        if parsed.direction == "above":
            # Spot must be clearly above threshold
            if spot > parsed.threshold + buffer:
                logger.debug(
                    "Crypto outcome CONFIRMED_YES: %s spot=%.2f > threshold=%.2f (buffer=%.2f)",
                    parsed.symbol, spot, parsed.threshold, buffer,
                )
                return OutcomeStatus.CONFIRMED_YES
            # Spot clearly below threshold → NO wins
            elif spot < parsed.threshold - buffer:
                logger.debug(
                    "Crypto outcome CONFIRMED_NO: %s spot=%.2f < threshold=%.2f (buffer=%.2f)",
                    parsed.symbol, spot, parsed.threshold, buffer,
                )
                return OutcomeStatus.CONFIRMED_NO
            else:
                # Within buffer zone - too close to call
                logger.debug(
                    "Crypto outcome UNKNOWN: %s spot=%.2f within buffer of threshold=%.2f",
                    parsed.symbol, spot, parsed.threshold,
                )
                return OutcomeStatus.UNKNOWN

        else:  # direction == "below"
            # Spot must be clearly below threshold
            if spot < parsed.threshold - buffer:
                logger.debug(
                    "Crypto outcome CONFIRMED_YES: %s spot=%.2f < threshold=%.2f (buffer=%.2f)",
                    parsed.symbol, spot, parsed.threshold, buffer,
                )
                return OutcomeStatus.CONFIRMED_YES
            # Spot clearly above threshold → NO wins
            elif spot > parsed.threshold + buffer:
                logger.debug(
                    "Crypto outcome CONFIRMED_NO: %s spot=%.2f > threshold=%.2f (buffer=%.2f)",
                    parsed.symbol, spot, parsed.threshold, buffer,
                )
                return OutcomeStatus.CONFIRMED_NO
            else:
                return OutcomeStatus.UNKNOWN

    def _parse_crypto_threshold(self, question: str) -> ParsedThreshold | None:
        """
        Parse a crypto market question to extract symbol, direction, threshold.

        Examples:
            "Will BTC be above $50,000 at 3pm?" → BTC, above, 50000
            "ETH above 3k?" → ETH, above, 3000
        """
        question_upper = question.upper()

        # Find crypto symbol
        symbol = None
        for token_name, ticker in _BINANCE_SYMBOLS.items():
            if token_name in question_upper:
                symbol = token_name
                break
        if not symbol:
            return None

        # Determine direction
        direction = None
        for pattern in _ABOVE_PATTERNS:
            if pattern.search(question):
                direction = "above"
                break
        if direction is None:
            for pattern in _BELOW_PATTERNS:
                if pattern.search(question):
                    direction = "below"
                    break
        if not direction:
            return None

        # Extract threshold value
        threshold = None
        for pattern in _THRESHOLD_PATTERNS:
            match = pattern.search(question)
            if match:
                raw = match.group(1).replace(",", "")
                try:
                    value = float(raw)
                    # Check if this was a "k" suffix (e.g., 50k = 50000)
                    if "k" in match.group(0).lower():
                        value *= 1000
                    # Check for negative sign in the matched text or just before it
                    matched_text = match.group(0)
                    if "-" in matched_text:
                        value = -value
                    else:
                        # Check for minus sign immediately before the match
                        match_start = match.start()
                        if match_start > 0 and question[match_start - 1] == "-":
                            value = -value
                    threshold = value
                    break
                except ValueError:
                    continue

        if threshold is None or threshold <= 0:
            return None

        # Confidence based on how well we parsed
        confidence = 0.9  # High confidence if we got here

        return ParsedThreshold(
            symbol=symbol,
            direction=direction,
            threshold=threshold,
            confidence=confidence,
        )

    def _get_spot_price(self, symbol: str) -> float | None:
        """
        Fetch current spot price from Binance. Uses cache if fresh.

        Returns None on failure.
        """
        now = time.time()
        cached = self._spot_cache.get(symbol)
        if cached and (now - cached[1]) < self._spot_cache_sec:
            return cached[0]

        if not self._allow_network:
            return None

        ticker = _BINANCE_SYMBOLS.get(symbol.upper())
        if not ticker:
            return None

        try:
            resp = httpx.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": ticker},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            raw_price = float(resp.json()["price"])

            # Validate price
            price = validate_size(raw_price, context=f"Binance spot ({symbol})")

            self._spot_cache[symbol] = (price, now)
            logger.debug("Spot price %s = $%.2f", symbol, price)
            return price
        except Exception as e:
            logger.warning("Spot price fetch failed for %s: %s", symbol, e)
            return None

    def clear_cache(self) -> None:
        """Clear the spot price cache (useful for testing or force refresh)."""
        self._spot_cache.clear()


def check_outcome_sync(
    market: Market,
    allow_network: bool = True,
) -> OutcomeStatus:
    """
    Convenience function for synchronous outcome checking.

    Creates a new OutcomeOracle instance per call (no cache persistence).
    For persistent caching, create and reuse an OutcomeOracle instance.
    """
    oracle = OutcomeOracle(allow_network=allow_network)
    return oracle.check_outcome(market)
