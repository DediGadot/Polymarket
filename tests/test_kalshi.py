"""
Unit tests for client/kalshi.py -- Kalshi REST API v2 client.
"""

from unittest.mock import MagicMock
import pytest
import httpx
import respx

from client.kalshi import KalshiClient, dollars_to_cents
from client.kalshi_auth import KalshiAuth
from scanner.models import OrderBook


def _mock_auth() -> KalshiAuth:
    """Create a mock KalshiAuth that returns empty headers."""
    auth = MagicMock(spec=KalshiAuth)
    auth.sign_request.return_value = {
        "KALSHI-ACCESS-KEY": "test-key",
        "KALSHI-ACCESS-SIGNATURE": "test-sig",
        "KALSHI-ACCESS-TIMESTAMP": "1700000000000",
    }
    return auth


class TestGetMarkets:
    @respx.mock
    def test_fetches_markets(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.get("https://test.kalshi.com/trade-api/v2/markets").mock(
            return_value=httpx.Response(200, json={
                "markets": [
                    {
                        "ticker": "PRES-2028-GOP",
                        "event_ticker": "PRES-2028",
                        "title": "2028 Presidential Election - GOP Nominee",
                        "subtitle": "GOP",
                        "yes_sub_title": "Yes",
                        "no_sub_title": "No",
                        "status": "open",
                        "result": "",
                    },
                ],
                "cursor": "",
            })
        )

        markets, cursor = client.get_markets()
        assert len(markets) == 1
        assert markets[0].ticker == "PRES-2028-GOP"
        assert markets[0].event_ticker == "PRES-2028"
        assert cursor is None

    @respx.mock
    def test_pagination(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        # Page 1
        respx.get("https://test.kalshi.com/trade-api/v2/markets").mock(
            side_effect=[
                httpx.Response(200, json={
                    "markets": [{"ticker": f"M{i}", "event_ticker": "E1", "title": "", "subtitle": "", "yes_sub_title": "", "no_sub_title": "", "status": "open", "result": ""} for i in range(3)],
                    "cursor": "page2",
                }),
                httpx.Response(200, json={
                    "markets": [{"ticker": "M3", "event_ticker": "E1", "title": "", "subtitle": "", "yes_sub_title": "", "no_sub_title": "", "status": "open", "result": ""}],
                    "cursor": "",
                }),
            ]
        )

        all_markets = client.get_all_markets()
        assert len(all_markets) == 4


class TestGetOrderbook:
    @respx.mock
    def test_converts_cents_to_dollars(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.get("https://test.kalshi.com/trade-api/v2/markets/PRES-2028/orderbook").mock(
            return_value=httpx.Response(200, json={
                "orderbook": {
                    "yes": [[60, 100], [55, 200]],  # YES bids: 60c ($0.60), 55c ($0.55)
                    "no": [[45, 150], [40, 300]],    # NO bids: 45c, 40c -> YES asks: 55c, 60c
                },
            })
        )

        book = client.get_orderbook("PRES-2028")
        assert book.token_id == "PRES-2028"

        # Bids: YES bids at $0.60 and $0.55, sorted desc
        assert len(book.bids) == 2
        assert book.bids[0].price == pytest.approx(0.60)
        assert book.bids[0].size == 100
        assert book.bids[1].price == pytest.approx(0.55)
        assert book.bids[1].size == 200

        # Asks: derived from NO bids. NO@45c -> YES ask at (100-45)/100 = $0.55
        # NO@40c -> YES ask at (100-40)/100 = $0.60
        assert len(book.asks) == 2
        assert book.asks[0].price == pytest.approx(0.55)
        assert book.asks[0].size == 150
        assert book.asks[1].price == pytest.approx(0.60)
        assert book.asks[1].size == 300

    @respx.mock
    def test_empty_orderbook(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.get("https://test.kalshi.com/trade-api/v2/markets/EMPTY/orderbook").mock(
            return_value=httpx.Response(200, json={
                "orderbook": {"yes": [], "no": []},
            })
        )

        book = client.get_orderbook("EMPTY")
        assert book.bids == ()
        assert book.asks == ()

    @respx.mock
    def test_get_orderbooks_multiple(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.get("https://test.kalshi.com/trade-api/v2/markets/M1/orderbook").mock(
            return_value=httpx.Response(200, json={
                "orderbook": {"yes": [[50, 100]], "no": [[50, 100]]},
            })
        )
        respx.get("https://test.kalshi.com/trade-api/v2/markets/M2/orderbook").mock(
            return_value=httpx.Response(200, json={
                "orderbook": {"yes": [[70, 200]], "no": [[30, 200]]},
            })
        )

        books = client.get_orderbooks(["M1", "M2"])
        assert len(books) == 2
        assert "M1" in books
        assert "M2" in books

    @respx.mock
    def test_get_orderbooks_partial_failure(self):
        """Should skip failed tickers and return what succeeded."""
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.get("https://test.kalshi.com/trade-api/v2/markets/GOOD/orderbook").mock(
            return_value=httpx.Response(200, json={
                "orderbook": {"yes": [[50, 100]], "no": [[50, 100]]},
            })
        )
        respx.get("https://test.kalshi.com/trade-api/v2/markets/BAD/orderbook").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )

        books = client.get_orderbooks(["GOOD", "BAD"])
        assert len(books) == 1
        assert "GOOD" in books


class TestBookFetcher:
    @respx.mock
    def test_book_fetcher_property(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.get("https://test.kalshi.com/trade-api/v2/markets/T1/orderbook").mock(
            return_value=httpx.Response(200, json={
                "orderbook": {"yes": [[50, 100]], "no": [[50, 100]]},
            })
        )

        fetcher = client.book_fetcher
        books = fetcher(["T1"])
        assert "T1" in books
        assert isinstance(books["T1"], OrderBook)


class TestPlaceOrder:
    @respx.mock
    def test_place_limit_order(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.post("https://test.kalshi.com/trade-api/v2/portfolio/orders").mock(
            return_value=httpx.Response(200, json={
                "order": {"order_id": "abc123", "status": "resting"},
            })
        )

        result = client.place_order(
            ticker="PRES-2028",
            side="yes",
            action="buy",
            count=10,
            type="limit",
            yes_price=55,
        )
        assert result["order"]["order_id"] == "abc123"


class TestBatchOrders:
    @respx.mock
    def test_batch_place_orders(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.post("https://test.kalshi.com/trade-api/v2/portfolio/orders/batched").mock(
            return_value=httpx.Response(200, json={
                "orders": [
                    {"order_id": "o1", "status": "resting"},
                    {"order_id": "o2", "status": "resting"},
                ],
            })
        )

        orders = [
            {"ticker": "M1", "side": "yes", "action": "buy", "count": 5, "type": "limit", "yes_price": 50},
            {"ticker": "M2", "side": "no", "action": "buy", "count": 5, "type": "limit", "no_price": 50},
        ]
        result = client.batch_place_orders(orders)
        assert "orders" in result

    def test_batch_over_20_raises(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")
        with pytest.raises(ValueError, match="batch limit is 20"):
            client.batch_place_orders([{}] * 21)


class TestDollarsToCents:
    def test_midrange_price(self):
        assert dollars_to_cents(0.50) == 50

    def test_one_cent(self):
        assert dollars_to_cents(0.01) == 1

    def test_ninety_nine_cents(self):
        assert dollars_to_cents(0.99) == 99

    def test_rounds_correctly(self):
        """0.554 rounds to 55, 0.555 rounds to 56."""
        assert dollars_to_cents(0.554) == 55
        assert dollars_to_cents(0.555) == 56

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            dollars_to_cents(0.0)

    def test_one_dollar_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            dollars_to_cents(1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            dollars_to_cents(-0.10)

    def test_over_one_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            dollars_to_cents(1.50)


class TestGetBalance:
    @respx.mock
    def test_balance_converts_cents_to_dollars(self):
        auth = _mock_auth()
        client = KalshiClient(auth, host="https://test.kalshi.com/trade-api/v2")

        respx.get("https://test.kalshi.com/trade-api/v2/portfolio/balance").mock(
            return_value=httpx.Response(200, json={"balance": 250000})  # $2,500.00
        )

        balance = client.get_balance()
        assert balance == pytest.approx(2500.0)
