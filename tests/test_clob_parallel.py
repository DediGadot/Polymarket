"""
Unit tests for client/clob.py -- get_orderbooks_parallel().
"""

from unittest.mock import MagicMock, patch
import client.clob as clob
from client.clob import get_orderbooks_parallel, BOOK_BATCH_SIZE
from scanner.models import OrderBook


def _make_raw_book(asset_id: str, bid_price: float, ask_price: float):
    """Create a mock raw book response matching SDK structure."""
    raw = MagicMock()
    raw.asset_id = asset_id
    bid = MagicMock()
    bid.price = str(bid_price)
    bid.size = "100"
    ask = MagicMock()
    ask.price = str(ask_price)
    ask.size = "100"
    raw.bids = [bid]
    raw.asks = [ask]
    return raw


class TestGetOrderbooksParallel:
    def test_empty_input(self):
        client = MagicMock()
        result = get_orderbooks_parallel(client, [])
        assert result == {}
        client.get_order_books.assert_not_called()

    def test_single_token(self):
        client = MagicMock()
        client.get_order_books.return_value = [_make_raw_book("tok1", 0.50, 0.52)]
        result = get_orderbooks_parallel(client, ["tok1"], max_workers=1)
        assert "tok1" in result
        assert isinstance(result["tok1"], OrderBook)
        assert result["tok1"].best_bid.price == 0.50
        assert result["tok1"].best_ask.price == 0.52

    def test_multiple_tokens(self):
        client = MagicMock()

        def mock_get_order_books(params):
            return [_make_raw_book(p.token_id, 0.40, 0.60) for p in params]

        client.get_order_books.side_effect = mock_get_order_books
        token_ids = [f"tok{i}" for i in range(10)]
        result = get_orderbooks_parallel(client, token_ids, max_workers=4)
        assert len(result) == 10
        for tid in token_ids:
            assert tid in result

    def test_chunking_respects_batch_size(self):
        client = MagicMock()
        call_sizes: list[int] = []

        def mock_get_order_books(params):
            call_sizes.append(len(params))
            return [_make_raw_book(p.token_id, 0.40, 0.60) for p in params]

        client.get_order_books.side_effect = mock_get_order_books
        # Create more tokens than one batch
        token_ids = [f"tok{i}" for i in range(BOOK_BATCH_SIZE + 5)]
        result = get_orderbooks_parallel(client, token_ids, max_workers=2)
        assert len(result) == BOOK_BATCH_SIZE + 5
        # Should have made 2 calls
        assert len(call_sizes) == 2
        assert call_sizes[0] == BOOK_BATCH_SIZE or call_sizes[1] == BOOK_BATCH_SIZE

    def test_error_propagation(self):
        client = MagicMock()
        client.get_order_books.side_effect = RuntimeError("API down")
        try:
            get_orderbooks_parallel(client, ["tok1"], max_workers=1)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "API down" in str(e)

    def test_sort_order_correct(self):
        """Verify bids are descending and asks are ascending."""
        client = MagicMock()
        raw = MagicMock()
        raw.asset_id = "tok1"
        bid1 = MagicMock(price="0.48", size="100")
        bid2 = MagicMock(price="0.50", size="200")
        ask1 = MagicMock(price="0.55", size="100")
        ask2 = MagicMock(price="0.52", size="200")
        raw.bids = [bid1, bid2]  # out of order
        raw.asks = [ask1, ask2]  # out of order
        client.get_order_books.return_value = [raw]

        result = get_orderbooks_parallel(client, ["tok1"], max_workers=1)
        book = result["tok1"]
        # Bids: highest first
        assert book.bids[0].price == 0.50
        assert book.bids[1].price == 0.48
        # Asks: lowest first
        assert book.asks[0].price == 0.52
        assert book.asks[1].price == 0.55


class TestRetryApiCall:
    @patch("client.clob.time.sleep", return_value=None)
    def test_retries_on_rate_limit_then_succeeds(self, _mock_sleep):
        fn = MagicMock(side_effect=[
            RuntimeError("429 Too Many Requests: rate limit"),
            "ok",
        ])
        result = clob._retry_api_call(fn, "arg1")
        assert result == "ok"
        assert fn.call_count == 2
