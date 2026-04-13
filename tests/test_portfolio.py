"""Tests for portfolio construction."""

import numpy as np
import pandas as pd

from cryptoarb.portfolio import build_portfolio, compute_portfolio_returns, check_dollar_neutrality
from cryptoarb.signals import PairSignals


def _make_pair_signals(n=100):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    pos = pd.Series(np.zeros(n), index=dates)
    pos.iloc[20:40] = 1.0  # long spread for 20 days
    pos.iloc[60:80] = -1.0  # short spread for 20 days

    return PairSignals(
        asset_a="A",
        asset_b="B",
        spread=pd.Series(np.random.randn(n), index=dates),
        z_score=pd.Series(np.random.randn(n), index=dates),
        position=pos,
        rolling_beta=pd.Series(np.full(n, 1.5), index=dates),
        rolling_intercept=pd.Series(np.full(n, 0.1), index=dates),
    )


def _make_log_prices(n=100):
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "A": rng.normal(0, 0.01, n).cumsum() + np.log(100),
        "B": rng.normal(0, 0.01, n).cumsum() + np.log(50),
    }, index=dates)


class TestBuildPortfolio:
    def test_produces_weights(self):
        signals = [_make_pair_signals()]
        prices = _make_log_prices()
        weights = build_portfolio(signals, prices)
        assert "A" in weights.columns
        assert "B" in weights.columns
        assert len(weights) == 100

    def test_dollar_neutrality(self):
        signals = [_make_pair_signals()]
        prices = _make_log_prices()
        weights = build_portfolio(signals, prices)
        net_exposure = check_dollar_neutrality(weights)
        # Net exposure should be small (not exactly 0 due to beta != 1)
        assert net_exposure.abs().max() < 0.05

    def test_empty_signals(self):
        prices = _make_log_prices()
        weights = build_portfolio([], prices)
        assert weights.empty or weights.shape[1] == 0


class TestPortfolioReturns:
    def test_computes_returns(self):
        signals = [_make_pair_signals()]
        prices = _make_log_prices()
        weights = build_portfolio(signals, prices)
        returns = compute_portfolio_returns(weights, prices, cost_bps=40)
        assert "gross_return" in returns.columns
        assert "net_return" in returns.columns
        assert "cost" in returns.columns
        assert "turnover" in returns.columns
        assert "cumulative" in returns.columns
        assert len(returns) == 100

    def test_costs_reduce_returns(self):
        signals = [_make_pair_signals()]
        prices = _make_log_prices()
        weights = build_portfolio(signals, prices)
        returns = compute_portfolio_returns(weights, prices, cost_bps=40)
        assert returns["cost"].sum() >= 0
        assert returns["net_return"].sum() <= returns["gross_return"].sum()
