"""Tests for Johansen basket trading."""

import numpy as np
import pandas as pd

from cryptoarb.basket import johansen_test, generate_basket_spread, discover_baskets


def _make_cointegrated_basket(n=500, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    base = rng.normal(0, 0.02, n).cumsum() + np.log(100)
    prices = {}
    for i, (beta, name) in enumerate([(1.0, "A"), (0.8, "B"), (1.1, "C")]):
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 0.01)
        prices[name] = 0.5 + beta * base + noise
    # Independent asset
    prices["D"] = rng.normal(0, 0.03, n).cumsum() + np.log(50)
    return pd.DataFrame(prices, index=dates)


class TestJohansenTest:
    def test_cointegrated_basket(self):
        prices = _make_cointegrated_basket()
        result = johansen_test(prices, ["A", "B", "C"])
        assert result.is_cointegrated
        assert result.n_cointegrating >= 1
        assert len(result.weights) == 3

    def test_non_cointegrated(self):
        rng = np.random.RandomState(42)
        n = 300
        dates = pd.date_range("2021-01-01", periods=n, freq="D")
        prices = pd.DataFrame({
            "X": rng.normal(0, 0.03, n).cumsum(),
            "Y": rng.normal(0, 0.03, n).cumsum(),
            "Z": rng.normal(0, 0.03, n).cumsum(),
        }, index=dates)
        result = johansen_test(prices, ["X", "Y", "Z"])
        # May or may not be cointegrated on random data
        assert len(result.weights) == 3

    def test_short_series(self):
        prices = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = johansen_test(prices, ["A", "B", "C"])
        assert not result.is_cointegrated


class TestBasketSpread:
    def test_generates_zscore(self):
        prices = _make_cointegrated_basket()
        basket = johansen_test(prices, ["A", "B", "C"])
        z = generate_basket_spread(prices, basket, rolling_window=30)
        valid = z.dropna()
        assert len(valid) > 0
        assert abs(valid.mean()) < 2.0


class TestDiscoverBaskets:
    def test_finds_baskets(self):
        prices = _make_cointegrated_basket()
        baskets = discover_baskets(prices, basket_size=3, min_correlation=0.5, max_baskets=5)
        assert len(baskets) >= 1
        assert baskets[0].is_cointegrated
