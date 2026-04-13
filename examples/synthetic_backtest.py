"""
Synthetic Backtest Example
==========================

Runs the full stat-arb pipeline on synthetic cointegrated data.
No exchange API needed — generates fake price data locally.

Usage:
    python examples/synthetic_backtest.py
"""

import numpy as np
import pandas as pd

from cryptoarb.config import StrategyConfig
from cryptoarb.data import log_prices
from cryptoarb.pairs import discover_pairs
from cryptoarb.signals import generate_pair_signals
from cryptoarb.portfolio import build_portfolio, compute_portfolio_returns
from cryptoarb.metrics import evaluate
from cryptoarb.risk import apply_drawdown_stop


def generate_synthetic_prices(n_coins=10, n_days=1500, seed=42):
    """Generate synthetic crypto prices with some cointegrated pairs."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    prices = {}

    # Create 3 cointegrated clusters
    # Cluster 1: BTC-like coins
    base1 = rng.normal(0, 0.02, n_days).cumsum() + np.log(50000)
    for i in range(3):
        noise = np.zeros(n_days)
        for t in range(1, n_days):
            noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 0.01)
        beta = 0.8 + rng.uniform(-0.3, 0.3)
        prices[f"COIN_{i}"] = np.exp(0.5 + beta * base1 + noise)

    # Cluster 2: ETH-like coins
    base2 = rng.normal(0, 0.025, n_days).cumsum() + np.log(3000)
    for i in range(3, 6):
        noise = np.zeros(n_days)
        for t in range(1, n_days):
            noise[t] = 0.80 * noise[t - 1] + rng.normal(0, 0.015)
        beta = 0.9 + rng.uniform(-0.2, 0.2)
        prices[f"COIN_{i}"] = np.exp(0.3 + beta * base2 + noise)

    # Independent coins (no cointegration)
    for i in range(6, n_coins):
        prices[f"COIN_{i}"] = np.exp(rng.normal(0, 0.03, n_days).cumsum() + np.log(100))

    return pd.DataFrame(prices, index=dates)


def main():
    print("=" * 60)
    print("CRYPTO STAT-ARB: SYNTHETIC BACKTEST")
    print("=" * 60)
    print()

    # Generate data
    print("Generating synthetic price data...")
    raw_prices = generate_synthetic_prices(n_coins=10, n_days=1500)
    log_price_matrix = log_prices(raw_prices)
    print(f"  {len(raw_prices.columns)} coins × {len(raw_prices)} days")
    print()

    # Configure strategy
    config = StrategyConfig()
    config.pairs.min_correlation = 0.60
    config.pairs.min_half_life = 2.0
    config.pairs.max_half_life = 40.0
    config.pairs.max_pairs = 10
    config.signals.entry_z = 2.0
    config.signals.exit_z = 0.5
    config.costs.taker_fee_bps = 10
    config.costs.slippage_bps = 5

    # Discover pairs on first 2 years
    train_data = log_price_matrix.iloc[:504]
    print("Discovering cointegrated pairs...")
    pairs = discover_pairs(
        train_data,
        min_correlation=config.pairs.min_correlation,
        adf_pvalue=config.pairs.adf_pvalue,
        min_half_life=config.pairs.min_half_life,
        max_half_life=config.pairs.max_half_life,
        max_pairs=config.pairs.max_pairs,
    )

    if not pairs:
        print("No cointegrated pairs found!")
        return

    print(f"\nFound {len(pairs)} pairs:")
    for p in pairs:
        print(f"  {p.asset_a}/{p.asset_b}: β={p.beta:.3f}, HL={p.half_life:.1f}d, ADF p={p.adf_pvalue:.4f}")
    print()

    # Generate signals on full dataset
    print("Generating signals...")
    signals = []
    for pair in pairs:
        sig = generate_pair_signals(
            log_price_matrix, pair,
            entry_z=config.signals.entry_z,
            exit_z=config.signals.exit_z,
        )
        signals.append(sig)
        active_days = (sig.position != 0).sum()
        print(f"  {pair.asset_a}/{pair.asset_b}: {active_days} active trading days")
    print()

    # Build portfolio (test on last 2 years only)
    test_start = 504
    test_log_prices = log_price_matrix.iloc[test_start:]

    print("Building portfolio...")
    weights = build_portfolio(signals, log_price_matrix)
    test_weights = weights.iloc[test_start:]

    # Compute returns
    print("Computing returns...")
    returns = compute_portfolio_returns(
        test_weights, test_log_prices,
        cost_bps=config.costs.round_trip_bps,
    )

    # Apply risk management
    print("Applying drawdown stop...")
    managed_returns = returns.copy()
    managed_returns["net_return"] = apply_drawdown_stop(
        returns["net_return"], max_drawdown=0.15, cooldown_days=30
    )
    managed_returns["cumulative"] = (1 + managed_returns["net_return"]).cumprod()

    # Evaluate
    print()
    raw_metrics = evaluate(returns)
    managed_metrics = evaluate(managed_returns)

    print("RAW STRATEGY (no risk management):")
    print(raw_metrics.summary())
    print()
    print("RISK-MANAGED STRATEGY:")
    print(managed_metrics.summary())


if __name__ == "__main__":
    main()
