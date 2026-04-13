"""Tests for Kalman filter hedge ratio."""

import numpy as np
import pandas as pd

from cryptoarb.kalman import kalman_hedge_ratio, kalman_zscore


def _make_cointegrated_pair(n=400, beta=1.5, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    b = rng.normal(0, 0.01, n).cumsum() + np.log(100)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.01)
    a = 0.5 + beta * b + noise
    return (
        pd.Series(a, index=dates, name="A"),
        pd.Series(b, index=dates, name="B"),
    )


class TestKalmanHedgeRatio:
    def test_converges_to_true_beta(self):
        a, b = _make_cointegrated_pair(beta=1.5)
        beta, intercept, spread = kalman_hedge_ratio(a, b)
        # After warm-up, beta should converge near 1.5
        late_beta = beta.iloc[-100:].mean()
        assert abs(late_beta - 1.5) < 0.3

    def test_spread_is_stationary(self):
        a, b = _make_cointegrated_pair()
        _, _, spread = kalman_hedge_ratio(a, b)
        # Spread should have bounded variance
        assert spread.std() < 1.0

    def test_output_lengths(self):
        a, b = _make_cointegrated_pair(n=200)
        beta, intercept, spread = kalman_hedge_ratio(a, b)
        assert len(beta) == 200
        assert len(intercept) == 200
        assert len(spread) == 200

    def test_delta_sensitivity(self):
        a, b = _make_cointegrated_pair()
        # Different delta values produce different beta paths
        beta_small, _, _ = kalman_hedge_ratio(a, b, delta=1e-6)
        beta_large, _, _ = kalman_hedge_ratio(a, b, delta=1e-2)
        # They should NOT be identical
        assert (beta_small.iloc[100:] - beta_large.iloc[100:]).abs().mean() > 0.001


class TestKalmanZScore:
    def test_zscore_centered(self):
        a, b = _make_cointegrated_pair()
        _, _, spread = kalman_hedge_ratio(a, b)
        z = kalman_zscore(spread, window=30)
        valid = z.dropna()
        assert abs(valid.mean()) < 1.0

    def test_zscore_scaled(self):
        a, b = _make_cointegrated_pair()
        _, _, spread = kalman_hedge_ratio(a, b)
        z = kalman_zscore(spread, window=30)
        valid = z.dropna()
        assert 0.5 < valid.std() < 2.5
