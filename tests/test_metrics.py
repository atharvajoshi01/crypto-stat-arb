"""Tests for performance metrics."""

import numpy as np
import pandas as pd

from cryptoarb.metrics import compute_drawdown, compute_max_drawdown_duration, evaluate


class TestDrawdown:
    def test_no_drawdown(self):
        cumulative = pd.Series([1.0, 1.1, 1.2, 1.3])
        dd = compute_drawdown(cumulative)
        assert dd.min() == 0.0

    def test_drawdown_computed(self):
        cumulative = pd.Series([1.0, 1.2, 0.9, 1.0])
        dd = compute_drawdown(cumulative)
        # Max drawdown: (0.9 - 1.2) / 1.2 = -0.25
        assert np.isclose(dd.min(), -0.25)

    def test_drawdown_with_zero_cumulative(self):
        # Strategy that loses all value — running_max hits 0
        cumulative = pd.Series([1.0, 0.5, 0.0, 0.0])
        dd = compute_drawdown(cumulative)
        # Should not produce -inf or NaN
        assert np.all(np.isfinite(dd))
        assert dd.iloc[1] == -0.5  # normal drawdown still works

    def test_drawdown_duration(self):
        cumulative = pd.Series([1.0, 1.2, 1.1, 1.0, 0.9, 1.0, 1.1, 1.3])
        duration = compute_max_drawdown_duration(cumulative)
        assert duration >= 3  # at least 3 days in drawdown


class TestEvaluate:
    def test_positive_strategy(self):
        n = 365
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.RandomState(42)
        net = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
        cumulative = (1 + net).cumprod()
        df = pd.DataFrame({
            "net_return": net,
            "gross_return": net + 0.0001,
            "cost": pd.Series(np.full(n, 0.0001), index=dates),
            "turnover": pd.Series(np.full(n, 0.05), index=dates),
            "cumulative": cumulative,
        })
        metrics = evaluate(df)
        assert metrics.sharpe_ratio > 0
        assert metrics.annual_return > 0
        assert metrics.win_rate > 0.4
        assert metrics.total_days == n

    def test_empty_returns(self):
        df = pd.DataFrame(columns=["net_return", "gross_return", "cost", "turnover", "cumulative"])
        metrics = evaluate(df)
        assert metrics.total_days == 0
        assert metrics.sharpe_ratio == 0

    def test_summary_string(self):
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.RandomState(42)
        net = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
        df = pd.DataFrame({
            "net_return": net,
            "gross_return": net,
            "cost": pd.Series(np.zeros(n), index=dates),
            "turnover": pd.Series(np.zeros(n), index=dates),
            "cumulative": (1 + net).cumprod(),
        })
        metrics = evaluate(df)
        summary = metrics.summary()
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary

    def test_to_dict(self):
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.RandomState(42)
        net = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
        df = pd.DataFrame({
            "net_return": net,
            "gross_return": net,
            "cost": pd.Series(np.zeros(n), index=dates),
            "turnover": pd.Series(np.zeros(n), index=dates),
            "cumulative": (1 + net).cumprod(),
        })
        metrics = evaluate(df)
        d = metrics.to_dict()
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d
