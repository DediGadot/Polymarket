"""
Fanatics/Crypto.com HMAC-SHA256 request signing.

Based on documented Crypto.com Exchange API v1 auth pattern:
  message = method + id + api_key + sorted_params + nonce
  signature = HMAC-SHA256(secret, message)

NOTE: Fanatics event contract API endpoints are TBD. This module implements
the auth pattern so the scaffolding is ready when the API ships.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


@dataclass(frozen=True)
class FanaticsAuth:
    """HMAC-SHA256 signing for Fanatics/Crypto.com event contract API."""

    api_key: str
    api_secret: str

    def sign_request(
        self,
        method: str,
        request_id: str,
        params: dict | None = None,
        nonce: int = 0,
    ) -> dict:
        """
        Build a signed request body per the Crypto.com HMAC-SHA256 pattern.

        Args:
            method: API method name (e.g., "private/get-order-detail")
            request_id: Unique request identifier
            params: Request parameters dict (sorted by key for signing)
            nonce: Unix timestamp in milliseconds

        Returns:
            Complete request body dict with signature included.
        """
        params = params or {}

        # Sort params by key for deterministic signing
        sorted_params = "".join(
            f"{k}{params[k]}" for k in sorted(params.keys())
        )

        # Construct signing payload
        payload = f"{method}{request_id}{self.api_key}{sorted_params}{nonce}"

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "id": request_id,
            "method": method,
            "api_key": self.api_key,
            "params": params,
            "nonce": nonce,
            "sig": signature,
        }
