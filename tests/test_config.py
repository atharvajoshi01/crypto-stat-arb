"""Tests for strategy configuration."""

from cryptoarb.config import StrategyConfig, SignalConfig, CostConfig


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
