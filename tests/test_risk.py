"""Tests for risk management."""

import numpy as np
import pandas as pd

from cryptoarb.risk import (
    apply_drawdown_stop,
    apply_volatility_scaling,
    compute_pair_drawdown,
)


class TestDrawdownStop:
    def test_stops_at_threshold(self):
        # Simulate a 20% drawdown then recovery
        returns = pd.Series(
            [-0.05, -0.05, -0.05, -0.05, -0.05, 0.02, 0.02, 0.02, 0.02, 0.02],
            index=pd.date_range("2023-01-01", periods=10, freq="D"),
        )
        adjusted = apply_drawdown_stop(returns, max_drawdown=0.15, cooldown_days=3)
        # After ~3 consecutive -5% days, DD exceeds 15%
        # Subsequent returns should be zeroed during cooldown
        zeroed = (adjusted == 0).sum()
        assert zeroed > 0

    def test_no_stop_within_threshold(self):
        returns = pd.Series(
            [0.01, -0.01, 0.01, -0.01, 0.01],
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )
        adjusted = apply_drawdown_stop(returns, max_drawdown=0.15, cooldown_days=3)
        # Small returns, no drawdown stop
        assert (adjusted == returns).all()


class TestVolScaling:
    def test_reduces_in_high_vol(self):
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.RandomState(42)

        # Weights: constant allocation
        weights = pd.DataFrame({"A": [0.5] * n, "B": [-0.5] * n}, index=dates)

        # Returns: first half calm, second half volatile
        returns_a = np.concatenate([rng.normal(0, 0.01, 100), rng.normal(0, 0.05, 100)])
        returns_b = np.concatenate([rng.normal(0, 0.01, 100), rng.normal(0, 0.05, 100)])
        returns = pd.DataFrame({"A": returns_a, "B": returns_b}, index=dates)

        scaled = apply_volatility_scaling(weights, returns, vol_target=0.10, vol_lookback=30)

        # Scaled weights in high-vol period should be smaller
        avg_weight_calm = scaled["A"].iloc[60:100].abs().mean()
        avg_weight_volatile = scaled["A"].iloc[160:200].abs().mean()
        assert avg_weight_volatile < avg_weight_calm

    def test_scale_capped(self):
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        weights = pd.DataFrame({"A": [1.0] * n}, index=dates)
        # Near-zero vol → scale would explode without cap
        returns = pd.DataFrame({"A": [0.0001] * n}, index=dates)
        scaled = apply_volatility_scaling(weights, returns, vol_target=0.10, vol_lookback=30)
        assert scaled["A"].max() <= 3.0  # capped at 3x


class TestPairDrawdown:
    def test_in_drawdown(self):
        returns = pd.Series([0.05, 0.05, -0.10, -0.10, 0.02])
        dd = compute_pair_drawdown(returns)
        assert dd < 0

    def test_at_high(self):
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        dd = compute_pair_drawdown(returns)
        assert dd == 0.0

    def test_empty(self):
        dd = compute_pair_drawdown(pd.Series([], dtype=float))
        assert dd == 0.0
