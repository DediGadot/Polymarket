"""
Gamma API client for market discovery. Pure REST, no SDK dependency.
"""

from __future__ import annotations

import json
import httpx

from scanner.models import Market, Event


_TIMEOUT = 30.0


def _get(base_url: str, path: str, params: dict | None = None) -> dict | list:
    """Make a GET request to the Gamma API. Raises on non-200."""
    url = f"{base_url}{path}"
    resp = httpx.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_markets(
    gamma_host: str,
    active: bool = True,
    closed: bool = False,
    neg_risk: bool | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[Market]:
    """
    Fetch markets from Gamma API. Returns our Market models.
    Paginates automatically if needed.
    """
    params: dict = {
        "active": str(active).lower(),
        "closed": str(closed).lower(),
        "limit": limit,
        "offset": offset,
    }
    if neg_risk is not None:
        params["neg_risk"] = str(neg_risk).lower()

    raw_markets = _get(gamma_host, "/markets", params)
    markets = []
    for m in raw_markets:
        # clobTokenIds may be a JSON string or a list
        raw_ids = m.get("clobTokenIds") or m.get("clob_token_ids")
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except (json.JSONDecodeError, TypeError):
                continue
        if not raw_ids or len(raw_ids) < 2:
            continue

        # event_id: try top-level eventId, then events[0].id
        event_id = str(m.get("eventId", ""))
        if not event_id:
            events_list = m.get("events") or []
            if events_list and isinstance(events_list, list):
                event_id = str(events_list[0].get("id", ""))

        # min tick size: may be a float or string
        tick_raw = m.get("orderPriceMinTickSize") or m.get("minimumTickSize") or m.get("minimum_tick_size") or "0.01"
        min_tick = str(tick_raw)

        markets.append(Market(
            condition_id=m.get("conditionId", m.get("condition_id", "")),
            question=m.get("question", ""),
            yes_token_id=raw_ids[0],
            no_token_id=raw_ids[1],
            neg_risk=bool(m.get("negRisk", m.get("neg_risk", False))),
            event_id=event_id,
            min_tick_size=min_tick,
            active=bool(m.get("active", True)),
            volume=float(m.get("volumeNum", m.get("volume", 0)) or 0),
        ))
    return markets


def get_all_markets(gamma_host: str, neg_risk: bool | None = None) -> list[Market]:
    """Fetch all active, non-closed markets by paginating through the API."""
    all_markets: list[Market] = []
    offset = 0
    page_size = 500
    while True:
        page = get_markets(
            gamma_host,
            active=True,
            closed=False,
            neg_risk=neg_risk,
            limit=page_size,
            offset=offset,
        )
        all_markets.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return all_markets


def get_events(gamma_host: str, limit: int = 200, offset: int = 0) -> list[dict]:
    """Fetch raw events from Gamma API."""
    params = {"limit": limit, "offset": offset, "active": "true", "closed": "false"}
    return _get(gamma_host, "/events", params)


def group_markets_by_event(markets: list[Market]) -> dict[str, list[Market]]:
    """Group markets by event_id for NegRisk multi-outcome scanning."""
    events: dict[str, list[Market]] = {}
    for m in markets:
        if m.event_id:
            events.setdefault(m.event_id, []).append(m)
    return events


def build_events(markets: list[Market]) -> list[Event]:
    """Build Event objects from a list of markets, grouped by event_id."""
    grouped = group_markets_by_event(markets)
    events = []
    for event_id, mkt_list in grouped.items():
        neg_risk = any(m.neg_risk for m in mkt_list)
        title = mkt_list[0].question if mkt_list else ""
        events.append(Event(
            event_id=event_id,
            title=title,
            markets=tuple(mkt_list),
            neg_risk=neg_risk,
        ))
    return events
