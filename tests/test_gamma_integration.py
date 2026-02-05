"""
Integration tests for client/gamma.py -- market discovery with mocked HTTP.
"""

import json
import pytest
import httpx
import respx

from client.gamma import (
    get_markets,
    get_all_markets,
    group_markets_by_event,
    build_events,
)


GAMMA_HOST = "https://gamma-api.polymarket.com"


def _market_json(cid, eid, yes_id, no_id, neg_risk=False, active=True, volume=1000):
    return {
        "conditionId": cid,
        "question": f"Question for {cid}?",
        "clobTokenIds": [yes_id, no_id],
        "negRisk": neg_risk,
        "eventId": eid,
        "minimumTickSize": "0.01",
        "active": active,
        "volume": volume,
    }


class TestGetMarkets:
    @respx.mock
    def test_basic_fetch(self):
        """Should parse markets from Gamma API response."""
        mock_data = [
            _market_json("c1", "e1", "y1", "n1"),
            _market_json("c2", "e1", "y2", "n2"),
        ]
        respx.get(f"{GAMMA_HOST}/markets").mock(
            return_value=httpx.Response(200, json=mock_data)
        )

        markets = get_markets(GAMMA_HOST)
        assert len(markets) == 2
        assert markets[0].condition_id == "c1"
        assert markets[0].yes_token_id == "y1"
        assert markets[0].no_token_id == "n1"
        assert markets[1].condition_id == "c2"

    @respx.mock
    def test_negrisk_filter(self):
        """Should filter by neg_risk parameter."""
        mock_data = [
            _market_json("c1", "e1", "y1", "n1", neg_risk=True),
        ]
        respx.get(f"{GAMMA_HOST}/markets").mock(
            return_value=httpx.Response(200, json=mock_data)
        )

        markets = get_markets(GAMMA_HOST, neg_risk=True)
        assert len(markets) == 1
        assert markets[0].neg_risk is True

    @respx.mock
    def test_skips_invalid_markets(self):
        """Markets without clobTokenIds should be skipped."""
        mock_data = [
            {"conditionId": "c1", "question": "Q?"},  # no token IDs
            _market_json("c2", "e1", "y2", "n2"),
        ]
        respx.get(f"{GAMMA_HOST}/markets").mock(
            return_value=httpx.Response(200, json=mock_data)
        )

        markets = get_markets(GAMMA_HOST)
        assert len(markets) == 1

    @respx.mock
    def test_http_error_raises(self):
        """Non-200 response should raise."""
        respx.get(f"{GAMMA_HOST}/markets").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(httpx.HTTPStatusError):
            get_markets(GAMMA_HOST)


class TestGetAllMarkets:
    @respx.mock
    def test_pagination(self):
        """Should paginate through all pages."""
        page1 = [_market_json(f"c{i}", "e1", f"y{i}", f"n{i}") for i in range(500)]
        page2 = [_market_json(f"c{i}", "e2", f"y{i}", f"n{i}") for i in range(500, 750)]

        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            offset = int(request.url.params.get("offset", 0))
            if offset == 0:
                return httpx.Response(200, json=page1)
            else:
                return httpx.Response(200, json=page2)

        respx.get(f"{GAMMA_HOST}/markets").mock(side_effect=side_effect)

        markets = get_all_markets(GAMMA_HOST)
        assert len(markets) == 750
        assert call_count == 2  # two pages


class TestGroupMarketsByEvent:
    def test_grouping(self):
        from scanner.models import Market
        m1 = Market("c1", "Q1", "y1", "n1", True, "e1", "0.01", True)
        m2 = Market("c2", "Q2", "y2", "n2", True, "e1", "0.01", True)
        m3 = Market("c3", "Q3", "y3", "n3", True, "e2", "0.01", True)

        groups = group_markets_by_event([m1, m2, m3])
        assert len(groups) == 2
        assert len(groups["e1"]) == 2
        assert len(groups["e2"]) == 1


class TestBuildEvents:
    def test_build(self):
        from scanner.models import Market
        m1 = Market("c1", "Q1", "y1", "n1", True, "e1", "0.01", True)
        m2 = Market("c2", "Q2", "y2", "n2", True, "e1", "0.01", True)
        m3 = Market("c3", "Q3", "y3", "n3", False, "e2", "0.01", True)

        events = build_events([m1, m2, m3])
        assert len(events) == 2

        nr_event = [e for e in events if e.neg_risk][0]
        assert len(nr_event.markets) == 2
        assert nr_event.event_id == "e1"
