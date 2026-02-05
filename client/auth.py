"""
Authentication: wallet setup and API credential derivation.
"""

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

from config import Config


def build_clob_client(cfg: Config) -> ClobClient:
    """
    Build an authenticated ClobClient ready for trading.
    Steps:
      1. Create L1 client with private key
      2. Derive or create API credentials (L2)
      3. Return fully authenticated client
    """
    # L1 client -- can sign orders and derive creds
    client = ClobClient(
        host=cfg.clob_host,
        chain_id=cfg.chain_id,
        key=cfg.private_key,
        signature_type=cfg.signature_type,
        funder=cfg.polymarket_profile_address,
    )

    # Derive L2 credentials (creates if first time, derives if already exist)
    creds: ApiCreds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    return client
