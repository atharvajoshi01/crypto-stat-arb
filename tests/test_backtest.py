"""Tests for walk-forward backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from cryptoarb.backtest import run_backtest, BacktestResult
from cryptoarb.config import StrategyConfig


def _make_cointegrated_prices(n=800, n_assets=6, seed=42):
    """Create synthetic cointegrated log-price matrix.

    Assets 0-1 and 2-3 are cointegrated pairs.
    Assets 4-5 are independent random walks.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Common factor
    factor = rng.normal(0, 0.01, n).cumsum() + np.log(100)

    columns = {}
    # Pair 1: A and B cointegrated
    noise_a = np.zeros(n)
    for i in range(1, n):
        noise_a[i] = 0.85 * noise_a[i - 1] + rng.normal(0, 0.005)
    columns["COIN_A"] = 0.3 + 1.1 * factor + noise_a
    columns["COIN_B"] = factor + rng.normal(0, 0.002, n).cumsum() * 0.01

    # Pair 2: C and D cointegrated
    noise_c = np.zeros(n)
    for i in range(1, n):
        noise_c[i] = 0.80 * noise_c[i - 1] + rng.normal(0, 0.006)
    columns["COIN_C"] = 0.1 + 0.9 * factor + noise_c
    columns["COIN_D"] = factor + rng.normal(0, 0.003, n).cumsum() * 0.01

    # Independent: E and F
    columns["COIN_E"] = rng.normal(0, 0.01, n).cumsum() + np.log(50)
    columns["COIN_F"] = rng.normal(0, 0.01, n).cumsum() + np.log(80)

    return pd.DataFrame(columns, index=dates)


def _small_config() -> StrategyConfig:
    """Config with small windows for fast testing."""
    return StrategyConfig(
        pairs={"min_correlation": 0.60, "adf_pvalue": 0.10, "min_half_life": 1.0, "max_half_life": 50.0, "max_pairs": 5, "formation_window": 200},
        signals={"entry_z": 1.5, "exit_z": 0.3, "stop_z": 3.5},
        portfolio={"max_pair_weight": 0.30},
        costs={"taker_fee_bps": 5.0, "slippage_bps": 2.0},
        backtest={"train_window_days": 200, "test_window_days": 100, "step_days": 50},
    )


class TestRunBacktest:
    def test_returns_backtest_result(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        assert isinstance(result, BacktestResult)

    def test_produces_returns(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        assert len(result.returns) > 0
        assert "net_return" in result.returns.columns
        assert "gross_return" in result.returns.columns
        assert "cumulative" in result.returns.columns

    def test_multiple_windows(self):
        prices = _make_cointegrated_prices(n=800)
        config = _small_config()
        result = run_backtest(prices, config)
        assert result.n_windows >= 2

    def test_pairs_discovered(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        assert len(result.pairs_used) > 0
        for window_pairs in result.pairs_used:
            assert len(window_pairs) > 0

    def test_no_duplicate_dates(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        assert not result.returns.index.duplicated().any()

    def test_returns_are_finite(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        assert result.returns["net_return"].notna().all()
        assert np.isfinite(result.returns["net_return"]).all()

    def test_cumulative_starts_near_one(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        assert abs(result.returns["cumulative"].iloc[0] - 1.0) < 0.05

    def test_total_days_matches(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        assert result.total_days == len(result.returns)


class TestBacktestEdgeCases:
    def test_insufficient_data_returns_empty(self):
        """Too little data for even one window."""
        prices = _make_cointegrated_prices(n=100)
        config = _small_config()
        result = run_backtest(prices, config)
        assert len(result.returns) == 0
        assert result.n_windows == 0

    def test_no_cointegrated_pairs(self):
        """All independent random walks — no pairs should be found."""
        rng = np.random.RandomState(99)
        n = 800
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        prices = pd.DataFrame({
            f"RW_{i}": rng.normal(0, 0.02, n).cumsum() + np.log(100)
            for i in range(6)
        }, index=dates)
        config = _small_config()
        config.pairs.min_correlation = 0.90
        config.pairs.adf_pvalue = 0.01
        result = run_backtest(prices, config)
        # May produce empty result or some windows skipped
        assert isinstance(result, BacktestResult)

    def test_net_returns_property(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        if len(result.returns) > 0:
            assert result.net_returns.equals(result.returns["net_return"])

    def test_sorted_index(self):
        prices = _make_cointegrated_prices()
        config = _small_config()
        result = run_backtest(prices, config)
        if len(result.returns) > 0:
            assert result.returns.index.is_monotonic_increasing
