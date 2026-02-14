"""
Tests for edge-proportional slippage ceiling in scanner/depth.py.
"""

from scanner.depth import slippage_ceiling
from scanner.models import Side


class TestSlippageCeiling:
    """Test the slippage_ceiling helper function."""

    def test_buy_side_5pct_edge(self):
        """5% edge with 0.4 fraction → 2% allowed slippage."""
        result = slippage_ceiling(0.45, edge_pct=5.0, side=Side.BUY)
        # 0.45 * (1 + 0.02) = 0.459
        assert abs(result - 0.459) < 1e-6

    def test_buy_side_1pct_edge(self):
        """1% edge → 0.4% allowed slippage (nearly same as old 0.5%)."""
        result = slippage_ceiling(0.45, edge_pct=1.0, side=Side.BUY)
        # 0.45 * (1 + 0.004) = 0.4518
        assert abs(result - 0.4518) < 1e-6

    def test_sell_side_5pct_edge(self):
        """Sell side: price floor moves DOWN."""
        result = slippage_ceiling(0.55, edge_pct=5.0, side=Side.SELL)
        # 0.55 * (1 - 0.02) = 0.539
        assert abs(result - 0.539) < 1e-6

    def test_capped_at_max_slippage(self):
        """20% edge with 0.4 fraction = 8%, but capped at max_slippage_pct=3%."""
        result = slippage_ceiling(
            0.10, edge_pct=20.0, side=Side.BUY,
            slippage_fraction=0.4, max_slippage_pct=3.0,
        )
        # 0.10 * (1 + 0.03) = 0.103
        assert abs(result - 0.103) < 1e-6

    def test_zero_edge(self):
        """Zero edge → zero slippage → price unchanged."""
        result = slippage_ceiling(0.50, edge_pct=0.0, side=Side.BUY)
        assert abs(result - 0.50) < 1e-6

    def test_custom_fraction(self):
        """Custom slippage_fraction=0.6 → 60% of edge allowed."""
        result = slippage_ceiling(
            0.45, edge_pct=10.0, side=Side.BUY,
            slippage_fraction=0.6, max_slippage_pct=10.0,
        )
        # 0.45 * (1 + 0.06) = 0.477
        assert abs(result - 0.477) < 1e-6

    def test_negative_edge_uses_absolute(self):
        """Negative edge → uses abs(), so same result as positive."""
        result = slippage_ceiling(0.45, edge_pct=-5.0, side=Side.BUY)
        expected = slippage_ceiling(0.45, edge_pct=5.0, side=Side.BUY)
        assert abs(result - expected) < 1e-9

    def test_wider_edge_gives_wider_ceiling(self):
        """Wider edges should produce wider slippage ceilings."""
        narrow = slippage_ceiling(0.45, edge_pct=2.0, side=Side.BUY)
        wide = slippage_ceiling(0.45, edge_pct=10.0, side=Side.BUY)
        assert wide > narrow

    def test_buy_ceiling_above_base(self):
        """BUY ceiling is always >= base price."""
        result = slippage_ceiling(0.45, edge_pct=5.0, side=Side.BUY)
        assert result >= 0.45

    def test_sell_floor_below_base(self):
        """SELL floor is always <= base price."""
        result = slippage_ceiling(0.55, edge_pct=5.0, side=Side.SELL)
        assert result <= 0.55
