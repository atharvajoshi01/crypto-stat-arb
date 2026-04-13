"""Tests for signal generation."""

import numpy as np
import pandas as pd

from cryptoarb.pairs import PairResult
from cryptoarb.signals import (
    compute_rolling_hedge_ratio,
    compute_spread,
    compute_zscore,
    generate_positions,
    generate_pair_signals,
)


def _make_log_prices(n=300, seed=42):
    rng = np.random.RandomState(seed)
    b = rng.normal(0, 0.01, n).cumsum() + np.log(100)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.01)
    a = 0.5 + 1.2 * b + noise
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"A": a, "B": b}, index=dates)


class TestRollingHedgeRatio:
    def test_produces_beta(self):
        prices = _make_log_prices()
        beta, intercept = compute_rolling_hedge_ratio(prices["A"], prices["B"], window=60)
        assert len(beta) == len(prices)
        # First 60 should be NaN (no window yet)
        assert beta.iloc[:60].isna().all()
        # After window, beta should be close to 1.2
        valid_beta = beta.dropna()
        assert abs(valid_beta.mean() - 1.2) < 0.3


class TestComputeSpread:
    def test_spread_is_stationary(self):
        prices = _make_log_prices()
        beta, intercept = compute_rolling_hedge_ratio(prices["A"], prices["B"], window=60)
        spread = compute_spread(prices["A"], prices["B"], beta, intercept)
        valid = spread.dropna()
        # Spread should have low autocorrelation (mean-reverting)
        assert abs(valid.autocorr(lag=1)) < 0.95


class TestZScore:
    def test_zscore_centered(self):
        rng = np.random.RandomState(42)
        spread = pd.Series(rng.normal(0, 1, 200))
        z = compute_zscore(spread, window=50)
        valid = z.dropna()
        # Z-score should be roughly centered around 0
        assert abs(valid.mean()) < 1.0

    def test_zscore_scaled(self):
        rng = np.random.RandomState(42)
        spread = pd.Series(rng.normal(0, 1, 200))
        z = compute_zscore(spread, window=50)
        valid = z.dropna()
        # Std should be close to 1
        assert 0.5 < valid.std() < 2.0


class TestPositions:
    def test_entry_long(self):
        z = pd.Series([0, 0, -2.5, -2.5, -1.0, -0.3, 0])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
        assert pos.iloc[2] == 1  # z < -2 → long
        assert pos.iloc[5] == 0  # |z| < 0.5 → exit

    def test_entry_short(self):
        z = pd.Series([0, 0, 2.5, 2.5, 1.0, 0.3, 0])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
        assert pos.iloc[2] == -1  # z > 2 → short
        assert pos.iloc[5] == 0  # exit

    def test_stop_loss(self):
        z = pd.Series([0, -2.5, -4.5, -3.0])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)
        assert pos.iloc[1] == 1  # enter long
        assert pos.iloc[2] == 0  # stop loss

    def test_holds_position(self):
        z = pd.Series([0, -2.5, -1.8, -1.5, -1.0, -0.8])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
        # Should hold from entry until |z| < 0.5
        assert pos.iloc[1] == 1
        assert pos.iloc[2] == 1  # still above exit threshold
        assert pos.iloc[4] == 1  # |z|=1.0 > 0.5, still holding

    def test_nan_handling(self):
        z = pd.Series([np.nan, np.nan, -2.5, -1.0, np.nan, 0])
        pos = generate_positions(z, entry_z=2.0, exit_z=0.5)
        assert pos.iloc[0] == 0
        assert pos.iloc[4] == 0  # NaN resets to flat


class TestGeneratePairSignals:
    def test_full_pipeline(self):
        prices = _make_log_prices(n=200)
        pair = PairResult(
            asset_a="A", asset_b="B", beta=1.2, intercept=0.5,
            adf_stat=-3.5, adf_pvalue=0.01, half_life=10,
            correlation=0.95, cointegrated=True,
        )
        signals = generate_pair_signals(prices, pair, entry_z=2.0, exit_z=0.5)
        assert len(signals.spread) == 200
        assert len(signals.z_score) == 200
        assert len(signals.position) == 200
        assert signals.asset_a == "A"
        assert signals.asset_b == "B"
        # Position should only be -1, 0, or 1
        assert set(signals.position.dropna().unique()).issubset({-1, 0, 1})
