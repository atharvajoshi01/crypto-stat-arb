"""Strategy configuration — all parameters in one place."""

from __future__ import annotations


from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data fetching and cleaning parameters."""

    exchange: str = "binance"
    timeframe: str = "1d"
    quote_currency: str = "USDT"
    min_data_pct: float = Field(0.90, description="Min fraction of non-null data to keep a coin")
    min_avg_volume: float = Field(1_000_000, description="Min avg daily USD volume")
    cache_dir: str = "data/cache"


class PairConfig(BaseModel):
    """Pair discovery parameters."""

    min_correlation: float = Field(0.70, description="Pre-filter: min absolute correlation")
    adf_pvalue: float = Field(0.05, description="Max ADF p-value to accept cointegration")
    min_half_life: float = Field(3.0, description="Min half-life in days")
    max_half_life: float = Field(30.0, description="Max half-life in days")
    max_pairs: int = Field(20, description="Max number of pairs to trade")
    formation_window: int = Field(504, description="Days for pair selection (2 years)")


class SignalConfig(BaseModel):
    """Signal generation parameters."""

    entry_z: float = Field(2.0, description="Z-score threshold to enter a trade")
    exit_z: float = Field(0.5, description="Z-score threshold to exit a trade")
    stop_z: float = Field(4.0, description="Z-score stop-loss threshold")
    rolling_window_multiplier: float = Field(
        2.0, description="Rolling window = multiplier × half_life"
    )


class PortfolioConfig(BaseModel):
    """Portfolio construction parameters."""

    max_pair_weight: float = Field(0.20, description="Max allocation to a single pair")
    max_gross_exposure: float = Field(2.0, description="Max gross exposure as multiple of capital")
    rebalance_frequency: str = Field("daily", description="Rebalance frequency")


class CostConfig(BaseModel):
    """Transaction cost parameters."""

    taker_fee_bps: float = Field(10.0, description="Taker fee per leg in basis points")
    slippage_bps: float = Field(5.0, description="Estimated slippage per leg in basis points")

    @property
    def round_trip_bps(self) -> float:
        """Total round-trip cost (both legs, both directions)."""
        return 2 * (self.taker_fee_bps + self.slippage_bps) * 2


class RiskConfig(BaseModel):
    """Risk management parameters."""

    max_portfolio_drawdown: float = Field(0.15, description="Halt trading at this drawdown")
    max_pair_drawdown: float = Field(0.10, description="Flatten pair at this drawdown")
    vol_scaling: bool = Field(True, description="Scale positions by inverse volatility")
    vol_target: float = Field(0.10, description="Target annual portfolio volatility")
    recoint_frequency_days: int = Field(30, description="Re-test cointegration every N days")


class BacktestConfig(BaseModel):
    """Walk-forward backtest parameters."""

    train_window_days: int = Field(504, description="Training window (2 years)")
    test_window_days: int = Field(126, description="Test window (6 months)")
    step_days: int = Field(63, description="Roll forward (3 months)")
    annualization_factor: int = Field(365, description="Crypto trades 365 days/year")


class StrategyConfig(BaseModel):
    """Complete strategy configuration."""

    data: DataConfig = Field(default_factory=DataConfig)
    pairs: PairConfig = Field(default_factory=PairConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: str) -> StrategyConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
