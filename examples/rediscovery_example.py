"""
Walk-Forward Pair Re-discovery
==============================

Most stat-arb backtests discover a pair set on a training window and hold it
fixed forever. This example shows the alternative: re-test the current pair
set on a rolling window, drop pairs whose cointegration has broken, and
discover fresh candidates.

The output prints the churn at each refresh event, plus aggregate stats:
how many pairs were dropped, how many were added, what the average churn
looks like across the full backtest.

Usage:
    python examples/rediscovery_example.py
"""

import json

import numpy as np
import pandas as pd

from cryptoarb.data import log_prices
from cryptoarb.pairs import discover_pairs
from cryptoarb.rediscovery import WalkForwardRediscovery


def generate_synthetic_prices(n_coins=10, n_days=1095, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    prices = {}

    base1 = rng.normal(0, 0.02, n_days).cumsum() + np.log(50000)
    for i in range(3):
        noise = np.zeros(n_days)
        for t in range(1, n_days):
            noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 0.01)
        beta = 0.8 + rng.uniform(-0.3, 0.3)
        prices[f"COIN_{i}"] = np.exp(0.5 + beta * base1 + noise)

    base2 = rng.normal(0, 0.025, n_days).cumsum() + np.log(3000)
    split = n_days // 2
    for i in range(3, 6):
        first = np.zeros(split)
        for t in range(1, split):
            first[t] = 0.80 * first[t - 1] + rng.normal(0, 0.015)
        beta = 0.9 + rng.uniform(-0.2, 0.2)
        second = rng.normal(0, 0.030, n_days - split).cumsum()
        series = np.concatenate([
            0.3 + beta * base2[:split] + first,
            0.3 + base2[split:] + second,
        ])
        prices[f"COIN_{i}"] = np.exp(series)

    for i in range(6, n_coins):
        prices[f"COIN_{i}"] = np.exp(rng.normal(0, 0.03, n_days).cumsum() + np.log(100))

    return pd.DataFrame(prices, index=dates)


def main():
    print("=" * 72)
    print("WALK-FORWARD PAIR RE-DISCOVERY")
    print("=" * 72)
    print()

    raw = generate_synthetic_prices()
    log_price_matrix = log_prices(raw)
    print(f"Dataset: {log_price_matrix.shape[1]} coins × {len(log_price_matrix)} days")
    print(f"Range: {log_price_matrix.index[0].date()} to {log_price_matrix.index[-1].date()}")
    print()

    # Initial pair discovery on the first 365 days.
    train_window = log_price_matrix.iloc[:365]
    initial_pairs = discover_pairs(
        train_window,
        min_correlation=0.50,
        adf_pvalue=0.05,
        min_half_life=2.0,
        max_half_life=60.0,
        max_pairs=8,
    )
    print(f"Initial pairs discovered on first 365 days: {len(initial_pairs)}")
    for p in initial_pairs:
        print(f"  {p.asset_a}/{p.asset_b}  half-life {p.half_life:.1f}d  ADF {p.adf_stat:.2f}")
    print()

    # Roll forward, refreshing every 120 days on a 300-day lookback.
    rediscovery = WalkForwardRediscovery(
        refresh_every_days=120,
        lookback_days=300,
        min_correlation=0.50,
        adf_pvalue=0.05,
        min_half_life=2.0,
        max_half_life=60.0,
        max_pairs=8,
    )
    history = rediscovery.run(
        log_price_matrix,
        initial_pairs=initial_pairs,
        start_date=log_price_matrix.index[365],
    )

    print(f"Walk-forward refresh schedule: every 120 days, 300-day lookback")
    print(f"Number of refreshes: {history.n_refreshes}")
    print()
    print("Per-refresh churn:")
    for event in history.events:
        print(
            f"  {event.refresh_date.date()}  "
            f"before {len(event.pairs_before):2d}  "
            f"after {len(event.pairs_after):2d}  "
            f"dropped {len(event.dropped):2d}  "
            f"added {len(event.added):2d}  "
            f"churn {event.churn_rate:.0%}"
        )
    print()

    print("Aggregate:")
    print(f"  Average churn per refresh: {history.avg_churn:.1%}")
    print(f"  Total pairs dropped:       {history.total_dropped}")
    print(f"  Total pairs added:         {history.total_added}")
    print(f"  Final pair set size:       {len(history.final_pairs)}")
    print()

    print("Final pair set:")
    for p in history.final_pairs:
        print(f"  {p.asset_a}/{p.asset_b}  half-life {p.half_life:.1f}d  ADF {p.adf_stat:.2f}")
    print()

    # Persist the history for the dashboard or downstream auditing.
    with open("rediscovery_history.json", "w") as f:
        json.dump(history.to_dict(), f, indent=2)
    print("Wrote rediscovery_history.json")


if __name__ == "__main__":
    main()
