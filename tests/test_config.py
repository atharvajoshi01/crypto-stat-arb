"""Tests for strategy configuration."""

import pytest

from cryptoarb.config import (
    StrategyConfig, SignalConfig, CostConfig, PairConfig, RiskConfig,
)


class TestStrategyConfig:
    def test_defaults(self):
        config = StrategyConfig()
        assert config.signals.entry_z == 2.0
        assert config.signals.exit_z == 0.5
        assert config.data.exchange == "binance"
        assert config.pairs.max_pairs == 20

    def test_custom_values(self):
        config = StrategyConfig(
            signals=SignalConfig(entry_z=1.5, exit_z=0.3),
        )
        assert config.signals.entry_z == 1.5

    def test_round_trip_cost(self):
        cost = CostConfig(taker_fee_bps=10, slippage_bps=5)
        assert cost.round_trip_bps == 60  # 2 legs × 2 directions × 15 bps


class TestParameterValidation:
    def test_correlation_out_of_range(self):
        with pytest.raises(ValueError, match="min_correlation"):
            PairConfig(min_correlation=1.5)

    def test_correlation_zero(self):
        with pytest.raises(ValueError, match="min_correlation"):
            PairConfig(min_correlation=0.0)

    def test_adf_pvalue_out_of_range(self):
        with pytest.raises(ValueError, match="adf_pvalue"):
            PairConfig(adf_pvalue=-0.1)

    def test_half_life_ordering(self):
        with pytest.raises(ValueError, match="max_half_life"):
            PairConfig(min_half_life=30.0, max_half_life=10.0)

    def test_entry_z_negative(self):
        with pytest.raises(ValueError, match="entry_z"):
            SignalConfig(entry_z=-1.0)

    def test_stop_below_entry(self):
        with pytest.raises(ValueError, match="stop_z"):
            SignalConfig(entry_z=2.0, stop_z=1.5)

    def test_exit_above_entry(self):
        with pytest.raises(ValueError, match="exit_z"):
            SignalConfig(entry_z=2.0, exit_z=3.0)

    def test_drawdown_out_of_range(self):
        with pytest.raises(ValueError, match="must be in"):
            RiskConfig(max_portfolio_drawdown=1.5)

    def test_valid_config_passes(self):
        config = StrategyConfig(
            pairs=PairConfig(min_correlation=0.5, adf_pvalue=0.05),
            signals=SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0),
            risk=RiskConfig(max_portfolio_drawdown=0.15),
        )
        assert config.pairs.min_correlation == 0.5


class TestVersionConsistency:
    def test_init_matches_pyproject(self):
        """Ensure __version__ in __init__.py matches pyproject.toml."""
        import tomllib
        from pathlib import Path

        import cryptoarb

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        assert cryptoarb.__version__ == data["project"]["version"]
