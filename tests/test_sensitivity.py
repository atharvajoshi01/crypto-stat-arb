"""Tests for parameter sensitivity analysis."""

import numpy as np
import pandas as pd

from cryptoarb.pairs import discover_pairs
from cryptoarb.sensitivity import run_sensitivity


def _make_cointegrated_prices(n=600, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = rng.normal(0, 0.02, n).cumsum() + np.log(100)
    noise_a = np.zeros(n)
    noise_b = np.zeros(n)
    for i in range(1, n):
        noise_a[i] = 0.85 * noise_a[i - 1] + rng.normal(0, 0.01)
        noise_b[i] = 0.80 * noise_b[i - 1] + rng.normal(0, 0.012)
    return pd.DataFrame({
        "A": 0.5 + 1.2 * base + noise_a,
        "B": base,
        "C": 0.3 + 0.9 * base + noise_b,
    }, index=dates)


class TestSensitivity:
    def test_runs_sweep(self):
        prices = _make_cointegrated_prices()
        pairs = discover_pairs(
            prices.iloc[:300],
            min_correlation=0.5,
            min_half_life=1,
            max_half_life=50,
            max_pairs=5,
        )
        if not pairs:
            return  # skip if no pairs found

        report = run_sensitivity(
            prices, pairs,
            entry_z_range=[1.5, 2.0],
            exit_z_range=[0.5],
            cost_bps_range=[40],
            test_start_idx=300,
        )
        assert len(report.results) >= 1
        assert report.best_sharpe is not None

    def test_to_dataframe(self):
        prices = _make_cointegrated_prices()
        pairs = discover_pairs(
            prices.iloc[:300],
            min_correlation=0.5,
            min_half_life=1,
            max_half_life=50,
        )
        if not pairs:
            return

        report = run_sensitivity(
            prices, pairs,
            entry_z_range=[2.0],
            exit_z_range=[0.5],
            cost_bps_range=[40],
            test_start_idx=300,
        )
        df = report.to_dataframe()
        assert "sharpe" in df.columns
        assert "entry_z" in df.columns
        assert len(df) >= 1

    def test_heatmap(self):
        prices = _make_cointegrated_prices()
        pairs = discover_pairs(
            prices.iloc[:300],
            min_correlation=0.5,
            min_half_life=1,
            max_half_life=50,
        )
        if not pairs:
            return

        report = run_sensitivity(
            prices, pairs,
            entry_z_range=[1.5, 2.0, 2.5],
            exit_z_range=[0.25, 0.5],
            cost_bps_range=[40],
            test_start_idx=300,
        )
        heatmap = report.sharpe_heatmap()
        assert heatmap.shape[0] >= 1  # at least 1 entry_z row
        assert heatmap.shape[1] >= 1  # at least 1 exit_z column

    def test_summary(self):
        prices = _make_cointegrated_prices()
        pairs = discover_pairs(
            prices.iloc[:300],
            min_correlation=0.5,
            min_half_life=1,
            max_half_life=50,
        )
        if not pairs:
            return

        report = run_sensitivity(
            prices, pairs,
            entry_z_range=[2.0],
            exit_z_range=[0.5],
            cost_bps_range=[40],
            test_start_idx=300,
        )
        s = report.summary()
        assert "Best Sharpe" in s
        assert "Combinations tested" in s
