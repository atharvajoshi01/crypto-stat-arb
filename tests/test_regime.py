"""Tests for regime detection."""

import numpy as np
import pandas as pd

from cryptoarb.regime import (
    MarketRegime,
    detect_regimes,
    regime_adjusted_weights,
    compute_regime_performance,
)


def _make_returns(n=500, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    # Mix of calm and volatile periods
    returns = np.concatenate([
        rng.normal(0.001, 0.01, 200),   # calm
        rng.normal(-0.002, 0.05, 100),  # volatile
        rng.normal(0.001, 0.015, 200),  # normal
    ])
    return pd.Series(returns, index=dates)


class TestDetectRegimes:
    def test_detects_multiple_regimes(self):
        returns = _make_returns()
        result = detect_regimes(returns, vol_lookback=20)
        unique = result.regimes.unique()
        assert len(unique) >= 2  # should find at least 2 regimes

    def test_crisis_in_volatile_period(self):
        returns = _make_returns()
        result = detect_regimes(returns, vol_lookback=20)
        # Days 200-300 are volatile — should have some HIGH_VOL or CRISIS
        volatile_period = result.regimes.iloc[210:290]
        high_vol_days = (volatile_period == MarketRegime.HIGH_VOL.value).sum()
        crisis_days = (volatile_period == MarketRegime.CRISIS.value).sum()
        assert (high_vol_days + crisis_days) > 10

    def test_current_regime(self):
        returns = _make_returns()
        result = detect_regimes(returns)
        assert result.current_regime() in [r.value for r in MarketRegime]

    def test_to_dict(self):
        returns = _make_returns()
        result = detect_regimes(returns)
        d = result.to_dict()
        assert "current_regime" in d
        assert "thresholds" in d


class TestRegimeAdjustedWeights:
    def test_reduces_in_crisis(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        weights = pd.DataFrame({"A": [0.5] * 100, "B": [-0.5] * 100}, index=dates)
        regimes = pd.Series(MarketRegime.NORMAL.value, index=dates)
        regimes.iloc[50:70] = MarketRegime.CRISIS.value

        scaled = regime_adjusted_weights(weights, regimes)
        # Crisis period should be zero
        assert (scaled.iloc[55]["A"] == 0.0)
        # Normal period should be unchanged
        assert (scaled.iloc[10]["A"] == 0.5)

    def test_increases_in_low_vol(self):
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        weights = pd.DataFrame({"A": [0.5] * 50}, index=dates)
        regimes = pd.Series(MarketRegime.LOW_VOL.value, index=dates)

        scaled = regime_adjusted_weights(weights, regimes)
        assert scaled["A"].iloc[0] == 0.6  # 0.5 * 1.2


class TestRegimePerformance:
    def test_breakdown_by_regime(self):
        returns = _make_returns()
        result = detect_regimes(returns, vol_lookback=20)
        perf = compute_regime_performance(returns, result.regimes)
        assert "low_vol" in perf.index or "normal" in perf.index
        assert "days" in perf.columns
        assert "sharpe" in perf.columns
