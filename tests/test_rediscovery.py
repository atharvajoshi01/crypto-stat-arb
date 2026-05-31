"""Tests for walk-forward pair re-discovery."""

import numpy as np
import pandas as pd
import pytest

from cryptoarb.pairs import PairResult
from cryptoarb.rediscovery import (
    RediscoveryEvent,
    RediscoveryHistory,
    WalkForwardRediscovery,
    _pair_key,
)


def _synthetic_log_prices(seed: int = 0, n_days: int = 1095) -> pd.DataFrame:
    """Generate three cointegrated clusters across ~3 years.

    The first cluster cointegrates throughout. The second cluster cointegrates
    only in the first half. The third cluster is independent. This lets us
    write tests that verify the rediscovery actually drops the second cluster
    in later refreshes.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")

    # Cluster A: cointegrated all the way through.
    base_a = rng.normal(0, 0.02, n_days).cumsum() + np.log(50000)
    cluster_a = {}
    for i in range(3):
        noise = np.zeros(n_days)
        for t in range(1, n_days):
            noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 0.008)
        beta = 0.85 + rng.uniform(-0.2, 0.2)
        cluster_a[f"A{i}"] = 0.4 + beta * base_a + noise

    # Cluster B: cointegrates for the first half, then decouples.
    base_b = rng.normal(0, 0.022, n_days).cumsum() + np.log(3000)
    cluster_b = {}
    split = n_days // 2
    for i in range(3):
        first = np.zeros(split)
        for t in range(1, split):
            first[t] = 0.80 * first[t - 1] + rng.normal(0, 0.012)
        beta = 0.9 + rng.uniform(-0.15, 0.15)
        second = rng.normal(0, 0.035, n_days - split).cumsum()
        series = np.concatenate([0.3 + beta * base_b[:split] + first, 0.3 + base_b[split:] + second])
        cluster_b[f"B{i}"] = series

    # Cluster C: independent random walks
    cluster_c = {}
    for i in range(3):
        cluster_c[f"C{i}"] = rng.normal(0, 0.03, n_days).cumsum() + np.log(100)

    frame = pd.DataFrame({**cluster_a, **cluster_b, **cluster_c}, index=dates)
    return frame


class TestRediscoveryEvent:
    def test_churn_rate(self):
        ev = RediscoveryEvent(
            refresh_date=pd.Timestamp("2024-01-01"),
            pairs_before=[("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")],
            pairs_after=[("A", "B"), ("I", "J")],
            dropped=[("C", "D"), ("E", "F"), ("G", "H")],
            added=[("I", "J")],
            retained=[("A", "B")],
        )
        assert ev.churn_rate == 0.75

    def test_churn_rate_empty_before(self):
        ev = RediscoveryEvent(
            refresh_date=pd.Timestamp("2024-01-01"),
            pairs_before=[],
            pairs_after=[("A", "B")],
        )
        assert ev.churn_rate == 0.0

    def test_to_dict(self):
        ev = RediscoveryEvent(
            refresh_date=pd.Timestamp("2024-06-15"),
            pairs_before=[("A", "B")],
            pairs_after=[("A", "B"), ("C", "D")],
            added=[("C", "D")],
            retained=[("A", "B")],
            n_tested=2,
        )
        d = ev.to_dict()
        assert d["refresh_date"] == "2024-06-15"
        assert d["n_added"] == 1
        assert d["n_retained"] == 1
        assert d["pairs_after"] == [
            {"asset_a": "A", "asset_b": "B"},
            {"asset_a": "C", "asset_b": "D"},
        ]


class TestPairKey:
    def test_canonical_order(self):
        p1 = PairResult(
            asset_a="ETH", asset_b="BTC", beta=0.5, intercept=0,
            adf_stat=-3, adf_pvalue=0.01, half_life=10,
            correlation=0.9, cointegrated=True,
        )
        p2 = PairResult(
            asset_a="BTC", asset_b="ETH", beta=0.5, intercept=0,
            adf_stat=-3, adf_pvalue=0.01, half_life=10,
            correlation=0.9, cointegrated=True,
        )
        assert _pair_key(p1) == _pair_key(p2)


class TestWalkForwardRediscovery:
    @pytest.fixture(scope="class")
    def log_prices(self):
        return _synthetic_log_prices()

    def test_runs_without_initial_pairs(self, log_prices):
        rediscovery = WalkForwardRediscovery(
            refresh_every_days=180,
            lookback_days=300,
            min_correlation=0.5,
            adf_pvalue=0.05,
            min_half_life=2.0,
            max_half_life=60.0,
            max_pairs=10,
        )
        history = rediscovery.run(log_prices, initial_pairs=[])
        assert history.n_refreshes >= 1
        assert isinstance(history.final_pairs, list)

    def test_drops_decoupled_cluster_in_later_refresh(self, log_prices):
        """Cluster B cointegrates only in the first half. Later refreshes
        should drop most B/B pairs."""
        from cryptoarb.pairs import discover_pairs

        train_window = log_prices.iloc[:365]
        initial = discover_pairs(
            train_window, min_correlation=0.5, adf_pvalue=0.10,
            min_half_life=2.0, max_half_life=60.0, max_pairs=10,
        )
        # Should have some B pairs in the initial set.
        initial_b_pairs = [
            p for p in initial
            if p.asset_a.startswith("B") and p.asset_b.startswith("B")
        ]

        rediscovery = WalkForwardRediscovery(
            refresh_every_days=180,
            lookback_days=300,
            min_correlation=0.5,
            adf_pvalue=0.05,
            min_half_life=2.0,
            max_half_life=60.0,
            max_pairs=10,
        )
        history = rediscovery.run(
            log_prices, initial_pairs=initial, start_date=log_prices.index[365],
        )

        # If we found B pairs initially, at least one refresh should have
        # dropped at least one B pair.
        if initial_b_pairs:
            total_dropped_b = 0
            for event in history.events:
                for a, b in event.dropped:
                    if a.startswith("B") and b.startswith("B"):
                        total_dropped_b += 1
            assert total_dropped_b > 0

    def test_history_aggregates(self, log_prices):
        rediscovery = WalkForwardRediscovery(
            refresh_every_days=120, lookback_days=300,
            min_correlation=0.5, max_pairs=8,
        )
        history = rediscovery.run(log_prices, initial_pairs=[])
        d = history.to_dict()
        assert d["n_refreshes"] == history.n_refreshes
        assert 0.0 <= d["avg_churn"] <= 1.0
        assert d["n_final_pairs"] == len(history.final_pairs)

    def test_respects_max_pairs(self, log_prices):
        rediscovery = WalkForwardRediscovery(
            refresh_every_days=180, lookback_days=300,
            min_correlation=0.4, max_pairs=3,
        )
        history = rediscovery.run(log_prices, initial_pairs=[])
        for event in history.events:
            assert len(event.pairs_after) <= 3
        assert len(history.final_pairs) <= 3

    def test_empty_log_prices_raises(self):
        rediscovery = WalkForwardRediscovery()
        with pytest.raises(ValueError, match="empty"):
            rediscovery.run(pd.DataFrame(), initial_pairs=[])

    def test_skips_short_windows(self):
        rediscovery = WalkForwardRediscovery(
            refresh_every_days=10, lookback_days=5, max_pairs=2,
        )
        log_prices = _synthetic_log_prices(n_days=200)
        history = rediscovery.run(log_prices, initial_pairs=[])
        # Should still produce at least one refresh, none should crash on
        # the short windows.
        assert isinstance(history, RediscoveryHistory)
