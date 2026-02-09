"""
Unit tests for client/kalshi_auth.py -- RSA-based Kalshi API authentication.
"""

import base64
import tempfile

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from client.kalshi_auth import KalshiAuth


def _generate_test_key_pair() -> tuple[rsa.RSAPrivateKey, bytes]:
    """Generate a test RSA key pair and return (private_key, pem_bytes)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return private_key, pem


def _write_temp_key(pem: bytes) -> str:
    """Write PEM bytes to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
    f.write(pem)
    f.close()
    return f.name


class TestKalshiAuth:
    def test_load_private_key(self):
        """Should load a valid RSA private key from PEM file."""
        _, pem = _generate_test_key_pair()
        path = _write_temp_key(pem)
        auth = KalshiAuth(api_key_id="test-key", private_key_path=path)
        assert auth.api_key_id == "test-key"

    def test_load_invalid_key_raises(self):
        """Should raise on invalid PEM data."""
        path = _write_temp_key(b"not a valid PEM file")
        with pytest.raises(Exception):
            KalshiAuth(api_key_id="test-key", private_key_path=path)

    def test_load_missing_file_raises(self):
        """Should raise on missing key file."""
        with pytest.raises(Exception):
            KalshiAuth(api_key_id="test-key", private_key_path="/nonexistent/key.pem")

    def test_sign_request_returns_headers(self):
        """Should return all required auth headers."""
        _, pem = _generate_test_key_pair()
        path = _write_temp_key(pem)
        auth = KalshiAuth(api_key_id="my-key-id", private_key_path=path)

        headers = auth.sign_request("GET", "/trade-api/v2/markets")

        assert headers["KALSHI-ACCESS-KEY"] == "my-key-id"
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        # Signature should be valid base64
        sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
        assert len(sig) > 0

    def test_sign_request_with_explicit_timestamp(self):
        """Should use provided timestamp instead of auto-generating."""
        _, pem = _generate_test_key_pair()
        path = _write_temp_key(pem)
        auth = KalshiAuth(api_key_id="my-key-id", private_key_path=path)

        ts = 1700000000000
        headers = auth.sign_request("POST", "/trade-api/v2/portfolio/orders", timestamp_ms=ts)

        assert headers["KALSHI-ACCESS-TIMESTAMP"] == str(ts)

    def test_signature_is_verifiable(self):
        """The signature should verify against the public key."""
        private_key, pem = _generate_test_key_pair()
        path = _write_temp_key(pem)
        auth = KalshiAuth(api_key_id="my-key-id", private_key_path=path)

        ts = 1700000000000
        method = "GET"
        api_path = "/trade-api/v2/markets"
        headers = auth.sign_request(method, api_path, timestamp_ms=ts)

        # Verify
        message = f"{ts}{method}{api_path}"
        sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
        public_key = private_key.public_key()
        # Should not raise
        public_key.verify(sig, message.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())

    def test_different_methods_produce_different_signatures(self):
        """GET and POST with same path should produce different signatures."""
        _, pem = _generate_test_key_pair()
        path = _write_temp_key(pem)
        auth = KalshiAuth(api_key_id="my-key-id", private_key_path=path)

        ts = 1700000000000
        headers_get = auth.sign_request("GET", "/trade-api/v2/markets", timestamp_ms=ts)
        headers_post = auth.sign_request("POST", "/trade-api/v2/markets", timestamp_ms=ts)

        assert headers_get["KALSHI-ACCESS-SIGNATURE"] != headers_post["KALSHI-ACCESS-SIGNATURE"]

    def test_method_is_uppercased(self):
        """Method should be uppercased in the signed message."""
        _, pem = _generate_test_key_pair()
        path = _write_temp_key(pem)
        auth = KalshiAuth(api_key_id="key", private_key_path=path)

        ts = 1700000000000
        h1 = auth.sign_request("get", "/path", timestamp_ms=ts)
        h2 = auth.sign_request("GET", "/path", timestamp_ms=ts)

        assert h1["KALSHI-ACCESS-SIGNATURE"] == h2["KALSHI-ACCESS-SIGNATURE"]
