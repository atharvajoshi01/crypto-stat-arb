"""Tests for factor attribution."""

import numpy as np
import pandas as pd

from cryptoarb.attribution import factor_attribution, compute_factor_returns


class TestFactorAttribution:
    def test_zero_beta_strategy(self):
        """Strategy uncorrelated with factor should have ~zero beta."""
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        strategy = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
        factors = pd.DataFrame({
            "BTC": rng.normal(0.0005, 0.03, n),
        }, index=dates)

        result = factor_attribution(strategy, factors)
        assert abs(result.beta["BTC"]) < 0.2
        assert result.r_squared < 0.1

    def test_high_beta_strategy(self):
        """Strategy correlated with factor should have high beta."""
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        btc = rng.normal(0.001, 0.03, n)
        strategy = pd.Series(0.8 * btc + rng.normal(0, 0.005, n), index=dates)
        factors = pd.DataFrame({"BTC": btc}, index=dates)

        result = factor_attribution(strategy, factors)
        assert abs(result.beta["BTC"] - 0.8) < 0.15
        assert result.r_squared > 0.5

    def test_alpha_positive(self):
        """Strategy with consistent positive drift should show positive alpha."""
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        btc = rng.normal(0, 0.03, n)
        strategy = pd.Series(0.002 + 0.0 * btc + rng.normal(0, 0.01, n), index=dates)
        factors = pd.DataFrame({"BTC": btc}, index=dates)

        result = factor_attribution(strategy, factors)
        assert result.alpha > 0

    def test_multiple_factors(self):
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        strategy = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
        factors = pd.DataFrame({
            "BTC": rng.normal(0.001, 0.03, n),
            "ETH": rng.normal(0.001, 0.04, n),
        }, index=dates)

        result = factor_attribution(strategy, factors)
        assert "BTC" in result.beta
        assert "ETH" in result.beta

    def test_short_series(self):
        strategy = pd.Series([0.01, 0.02], index=pd.date_range("2023-01-01", periods=2))
        factors = pd.DataFrame({"BTC": [0.01, -0.01]}, index=strategy.index)
        result = factor_attribution(strategy, factors)
        assert result.alpha == 0  # too short

    def test_summary_string(self):
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        strategy = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
        factors = pd.DataFrame({"BTC": rng.normal(0, 0.03, n)}, index=dates)
        result = factor_attribution(strategy, factors)
        s = result.summary()
        assert "Alpha" in s
        assert "Factor Betas" in s

    def test_to_dict(self):
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        strategy = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
        factors = pd.DataFrame({"BTC": rng.normal(0, 0.03, n)}, index=dates)
        result = factor_attribution(strategy, factors)
        d = result.to_dict()
        assert "alpha_annualized" in d
        assert "betas" in d


class TestComputeFactorReturns:
    def test_computes_returns(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "BTC_USD": np.random.randn(100).cumsum() + 50000,
            "ETH_USD": np.random.randn(100).cumsum() + 3000,
            "SOL_USD": np.random.randn(100).cumsum() + 100,
        }, index=dates)
        returns = compute_factor_returns(prices, ["BTC_USD", "ETH_USD"])
        assert "BTC_USD" in returns.columns
        assert "ETH_USD" in returns.columns
        assert "SOL_USD" not in returns.columns

    def test_missing_column(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = pd.DataFrame({"BTC_USD": range(10)}, index=dates)
        returns = compute_factor_returns(prices, ["MISSING"])
        assert returns.empty
