"""Gas cost estimation utilities."""

from __future__ import annotations

from client.gas import GasOracle


def estimate_gas_cost(
    gas_oracle: GasOracle | None,
    n_legs: int,
    gas_per_order: int,
    gas_price_gwei: float = 30.0,
) -> float:
    """
    Estimate USD gas cost for an order or batch of orders.

    Args:
        gas_oracle: GasOracle instance (uses cached price)
        n_legs: Number of legs/orders
        gas_per_order: Gas units per order
        gas_price_gwei: Gas price in GWei

    Returns:
        Estimated USD cost
    """
    if gas_oracle:
        return gas_oracle.estimate_cost_usd(n_legs, gas_per_order)
    else:
        # Fallback: simple calculation without oracle
        gas_cost_wei = n_legs * gas_per_order * 1e9
        gas_cost_matic = gas_cost_wei / 1e18
        # Default $0.50 POL/USD price
        return gas_cost_matic * 0.50


__all__ = ["estimate_gas_cost"]
