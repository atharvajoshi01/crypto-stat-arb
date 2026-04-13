"""Tests for data pipeline."""

import numpy as np
import pandas as pd

from cryptoarb.data import build_price_matrix, clean_price_matrix, log_prices


def _make_ohlcv(n: int = 100, symbol: str = "BTC_USDT") -> pd.DataFrame:
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.5, n),
        "high": close + abs(rng.normal(0, 1, n)),
        "low": close - abs(rng.normal(0, 1, n)),
        "close": close,
        "volume": rng.uniform(1e6, 1e7, n),
    }, index=dates)


class TestBuildPriceMatrix:
    def test_builds_matrix(self):
        ohlcv = {
            "BTC/USDT": _make_ohlcv(100),
            "ETH/USDT": _make_ohlcv(100),
        }
        matrix = build_price_matrix(ohlcv)
        assert matrix.shape == (100, 2)
        assert "BTC_USDT" in matrix.columns
        assert "ETH_USDT" in matrix.columns


class TestCleanPriceMatrix:
    def test_drops_sparse_coins(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "good": np.random.randn(100).cumsum() + 100,
            "bad": [np.nan] * 50 + list(np.random.randn(50).cumsum() + 100),
        }, index=dates)
        cleaned = clean_price_matrix(prices, min_data_pct=0.90)
        assert "good" in cleaned.columns
        assert "bad" not in cleaned.columns

    def test_forward_fills_gaps(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = pd.DataFrame({
            "a": [1, np.nan, np.nan, 4, 5, 6, 7, 8, 9, 10],
        }, index=dates)
        cleaned = clean_price_matrix(prices, min_data_pct=0.5, max_gap_days=3)
        assert cleaned["a"].isna().sum() == 0

    def test_drops_zero_variance(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "alive": np.random.randn(100).cumsum() + 100,
            "dead": [100.0] * 100,
        }, index=dates)
        cleaned = clean_price_matrix(prices, min_data_pct=0.5)
        assert "alive" in cleaned.columns
        assert "dead" not in cleaned.columns


class TestLogPrices:
    def test_log_transform(self):
        prices = pd.DataFrame({"a": [100, 200, 300]})
        result = log_prices(prices)
        assert np.isclose(result["a"].iloc[0], np.log(100))
        assert np.isclose(result["a"].iloc[1], np.log(200))
