"""Tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

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
        assert "sharpe_ratio_net" in d
        assert "sharpe_ratio_gross" in d
        assert "cost_drag" in d
        assert "sharpe_per_unit_turnover" in d
        assert "max_drawdown" in d


class TestDecisionQualityMetrics:
    """Kolm's argument 4: a model is only as good as the decision it lets you make
    net of implementation cost. These tests pin the contract for gross/net Sharpe,
    cost drag, and Sharpe per unit turnover."""

    def _make_returns(self, n=252, mean=0.001, vol=0.01, cost_bps=0.0, turnover=0.0, seed=42):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        rng = np.random.RandomState(seed)
        gross = pd.Series(rng.normal(mean, vol, n), index=dates)
        cost_series = pd.Series(np.full(n, cost_bps * turnover), index=dates)
        net = gross - cost_series
        return pd.DataFrame({
            "gross_return": gross,
            "net_return": net,
            "cost": cost_series,
            "turnover": pd.Series(np.full(n, turnover), index=dates),
            "cumulative": (1 + net).cumprod(),
        })

    def test_zero_cost_means_gross_equals_net_sharpe(self):
        df = self._make_returns(cost_bps=0.0, turnover=0.0)
        m = evaluate(df)
        assert m.gross_sharpe_ratio == pytest.approx(m.sharpe_ratio, rel=1e-9)
        assert m.cost_drag == pytest.approx(0.0, abs=1e-9)

    def test_positive_cost_drag_when_costs_eat_into_returns(self):
        df = self._make_returns(cost_bps=0.0005, turnover=0.4)
        m = evaluate(df)
        assert m.gross_sharpe_ratio > m.sharpe_ratio
        assert m.cost_drag > 0

    def test_higher_turnover_lowers_decision_quality_score(self):
        low = evaluate(self._make_returns(cost_bps=0.0005, turnover=0.10, seed=1))
        high = evaluate(self._make_returns(cost_bps=0.0005, turnover=0.80, seed=1))
        # Same underlying signal, more trading => decision quality should fall.
        assert high.sharpe_per_unit_turnover < low.sharpe_per_unit_turnover

    def test_zero_turnover_does_not_explode_decision_score(self):
        df = self._make_returns(turnover=0.0)
        m = evaluate(df)
        # Falls back to net Sharpe when turnover is essentially zero.
        assert m.sharpe_per_unit_turnover == pytest.approx(m.sharpe_ratio, rel=1e-9)
