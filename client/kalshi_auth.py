"""
Kalshi RSA-based API key authentication.

Kalshi API v2 requires RSA-signed requests:
  - Header: KALSHI-ACCESS-KEY = api_key_id
  - Header: KALSHI-ACCESS-SIGNATURE = base64(RSA_SIGN(timestamp + method + path))
  - Header: KALSHI-ACCESS-TIMESTAMP = unix_ms
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class KalshiAuth:
    """Handles RSA key loading and request signing for Kalshi API v2."""

    def __init__(self, api_key_id: str, private_key_path: str) -> None:
        self.api_key_id = api_key_id
        self._private_key = self._load_private_key(private_key_path)

    @staticmethod
    def _load_private_key(path: str) -> rsa.RSAPrivateKey:
        """Load RSA private key from PEM file."""
        pem_data = Path(path).read_bytes()
        key = serialization.load_pem_private_key(pem_data, password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError(f"Expected RSA private key, got {type(key).__name__}")
        return key

    def sign_request(self, method: str, path: str, timestamp_ms: int | None = None) -> dict[str, str]:
        """
        Generate authentication headers for a Kalshi API request.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: Request path (e.g., /trade-api/v2/markets)
            timestamp_ms: Unix timestamp in milliseconds (auto-generated if None)

        Returns:
            Dict of headers to add to the request.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        message = f"{timestamp_ms}{method.upper()}{path}"
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        sig_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }
