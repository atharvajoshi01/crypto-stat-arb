"""Tests for pair discovery."""

import numpy as np
import pandas as pd

from cryptoarb.pairs import compute_half_life, test_cointegration as run_coint_test, discover_pairs


def _make_cointegrated_pair(n=500, beta=1.5, seed=42):
    """Generate a synthetic cointegrated pair."""
    rng = np.random.RandomState(seed)
    # Random walk for B
    b = rng.normal(0, 0.01, n).cumsum() + 5.0
    # A = alpha + beta * B + stationary noise
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.02)
    a = 1.0 + beta * b + noise

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return (
        pd.Series(a, index=dates, name="COIN_A"),
        pd.Series(b, index=dates, name="COIN_B"),
    )


def _make_independent_pair(n=500, seed=42):
    """Generate two independent random walks."""
    rng = np.random.RandomState(seed)
    a = rng.normal(0, 0.01, n).cumsum() + 5.0
    b = rng.normal(0, 0.01, n).cumsum() + 5.0
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return (
        pd.Series(a, index=dates, name="IND_A"),
        pd.Series(b, index=dates, name="IND_B"),
    )


class TestHalfLife:
    def test_mean_reverting(self):
        rng = np.random.RandomState(42)
        n = 500
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = 0.9 * spread[i - 1] + rng.normal(0, 0.1)
        hl = compute_half_life(pd.Series(spread))
        assert 1 < hl < 30

    def test_random_walk(self):
        rng = np.random.RandomState(42)
        spread = pd.Series(rng.normal(0, 1, 500).cumsum())
        hl = compute_half_life(spread)
        assert hl == float("inf") or hl > 50


class TestCointegration:
    def test_cointegrated_pair(self):
        a, b = _make_cointegrated_pair()
        result = run_coint_test(a, b)
        assert result.cointegrated
        assert result.adf_pvalue < 0.05
        assert abs(result.beta - 1.5) < 0.5
        assert result.half_life < 30

    def test_independent_pair(self):
        a, b = _make_independent_pair()
        result = run_coint_test(a, b)
        assert not result.cointegrated

    def test_short_series(self):
        a = pd.Series([1, 2, 3], name="X")
        b = pd.Series([4, 5, 6], name="Y")
        result = run_coint_test(a, b)
        assert not result.cointegrated


class TestDiscoverPairs:
    def test_finds_cointegrated(self):
        a, b = _make_cointegrated_pair()
        c, _ = _make_independent_pair(seed=99)
        c.name = "COIN_C"
        prices = pd.DataFrame({"COIN_A": a, "COIN_B": b, "COIN_C": c})

        results = discover_pairs(
            prices,
            min_correlation=0.5,
            min_half_life=1,
            max_half_life=50,
        )
        assert len(results) >= 1
        pair_names = {(r.asset_a, r.asset_b) for r in results}
        assert ("COIN_A", "COIN_B") in pair_names or ("COIN_B", "COIN_A") in pair_names

    def test_respects_max_pairs(self):
        a, b = _make_cointegrated_pair()
        prices = pd.DataFrame({"COIN_A": a, "COIN_B": b})
        results = discover_pairs(prices, min_correlation=0.5, max_pairs=1)
        assert len(results) <= 1
