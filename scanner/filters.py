"""
Pre-filters for market scan speed optimization.
Applied before markets reach any scanner to reduce noise and latency.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from scanner.models import Market

logger = logging.getLogger(__name__)


def filter_by_volume(markets: list[Market], min_volume: float) -> list[Market]:
    """Filter out markets below minimum volume. NegRisk markets are never filtered."""
    if min_volume <= 0:
        return markets
    return [m for m in markets if m.neg_risk or m.volume >= min_volume]


def filter_about_to_resolve(markets: list[Market], min_hours: float = 1.0) -> list[Market]:
    """Filter out markets resolving within min_hours. Empty end_date passes through."""
    if min_hours <= 0:
        return markets

    now = datetime.now(timezone.utc)
    result: list[Market] = []
    for m in markets:
        if not m.end_date:
            result.append(m)
            continue
        try:
            dt_str = m.end_date.replace("Z", "+00:00")
            if "T" not in dt_str:
                dt_str += "T23:59:59+00:00"
            end_dt = datetime.fromisoformat(dt_str)
            hours_left = (end_dt - now).total_seconds() / 3600.0
            if hours_left >= min_hours:
                result.append(m)
        except (ValueError, TypeError):
            result.append(m)  # unparseable → keep (safe default)
    return result


def apply_pre_filters(
    markets: list[Market],
    min_volume: float = 0.0,
    min_hours: float = 0.0,
) -> list[Market]:
    """Apply all pre-filters in sequence."""
    filtered = filter_by_volume(markets, min_volume)
    filtered = filter_about_to_resolve(filtered, min_hours)
    return filtered
