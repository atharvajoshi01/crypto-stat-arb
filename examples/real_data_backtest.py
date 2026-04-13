"""
Real Data Backtest
==================

Fetches actual crypto data from Binance and runs the full stat-arb pipeline.
Generates equity curve plots and benchmark comparison.

Usage:
    python examples/real_data_backtest.py
"""

import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cryptoarb.config import StrategyConfig
from cryptoarb.data import fetch_ohlcv, build_price_matrix, clean_price_matrix, log_prices
from cryptoarb.pairs import discover_pairs
from cryptoarb.signals import generate_pair_signals
from cryptoarb.portfolio import build_portfolio, compute_portfolio_returns
from cryptoarb.metrics import evaluate, compute_drawdown
from cryptoarb.risk import apply_drawdown_stop

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Top liquid crypto pairs (USD pairs for Kraken/Coinbase compatibility)
SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD",
    "XRP/USD", "ADA/USD", "AVAX/USD", "DOGE/USD",
    "LINK/USD", "DOT/USD",
    "ATOM/USD", "LTC/USD",
]

EXCHANGE = "kraken"  # US-friendly exchange
START = "2021-01-01"
END = "2026-01-01"


def main():
    print("=" * 60)
    print("CRYPTO STAT-ARB: REAL DATA BACKTEST")
    print("=" * 60)
    print()

    # ---- Step 1: Fetch data ----
    print(f"Fetching {len(SYMBOLS)} coins from {EXCHANGE} ({START} to {END})...")
    ohlcv = fetch_ohlcv(SYMBOLS, exchange_id=EXCHANGE, start=START, end=END)
    print(f"  Fetched {len(ohlcv)} symbols")

    if len(ohlcv) < 4:
        print("Not enough data fetched. Check your internet connection.")
        sys.exit(1)

    # Build price and volume matrices
    price_matrix = build_price_matrix(ohlcv, field="close")
    volume_matrix = build_price_matrix(ohlcv, field="volume")

    # Multiply volume by close price for USD volume
    usd_volume = volume_matrix * price_matrix

    # Clean
    price_matrix = clean_price_matrix(
        price_matrix, min_data_pct=0.90,
        min_avg_volume=usd_volume, min_volume_threshold=500_000,
    )
    log_price_matrix = log_prices(price_matrix)
    print(f"  Clean matrix: {len(price_matrix.columns)} coins × {len(price_matrix)} days")
    print()

    # ---- Step 2: Configure ----
    config = StrategyConfig()
    config.pairs.min_correlation = 0.70
    config.pairs.min_half_life = 3.0
    config.pairs.max_half_life = 30.0
    config.pairs.max_pairs = 10
    config.signals.entry_z = 2.0
    config.signals.exit_z = 0.5
    config.signals.stop_z = 4.0
    config.costs.taker_fee_bps = 10
    config.costs.slippage_bps = 5

    # ---- Step 3: Discover pairs (first 2 years as training) ----
    train_days = min(730, len(log_price_matrix) // 2)
    train_data = log_price_matrix.iloc[:train_days]

    print("Discovering cointegrated pairs on training data...")
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
        sys.exit(1)

    print(f"\nFound {len(pairs)} pairs:")
    for p in pairs:
        print(f"  {p.asset_a}/{p.asset_b}: β={p.beta:.3f}, HL={p.half_life:.1f}d, "
              f"ADF p={p.adf_pvalue:.4f}, corr={p.correlation:.3f}")
    print()

    # ---- Step 4: Generate signals ----
    print("Generating signals on full dataset...")
    signals = []
    for pair in pairs:
        sig = generate_pair_signals(
            log_price_matrix, pair,
            entry_z=config.signals.entry_z,
            exit_z=config.signals.exit_z,
            stop_z=config.signals.stop_z,
        )
        signals.append(sig)
        active = (sig.position != 0).sum()
        print(f"  {pair.asset_a}/{pair.asset_b}: {active} active days")
    print()

    # ---- Step 5: Build portfolio & compute returns (OOS only) ----
    test_log_prices = log_price_matrix.iloc[train_days:]

    weights = build_portfolio(signals, log_price_matrix, max_pair_weight=0.20)
    test_weights = weights.loc[test_log_prices.index[0]:]

    returns = compute_portfolio_returns(
        test_weights, test_log_prices,
        cost_bps=config.costs.round_trip_bps,
    )

    # Risk-managed version
    managed = returns.copy()
    managed["net_return"] = apply_drawdown_stop(
        returns["net_return"], max_drawdown=0.15, cooldown_days=30
    )
    managed["cumulative"] = (1 + managed["net_return"]).cumprod()

    # ---- Step 6: Evaluate ----
    raw_metrics = evaluate(returns, annualization=365)
    managed_metrics = evaluate(managed, annualization=365)

    print("OUT-OF-SAMPLE RESULTS (no risk management):")
    print(raw_metrics.summary())
    print()
    print("OUT-OF-SAMPLE RESULTS (risk-managed):")
    print(managed_metrics.summary())
    print()

    # ---- Step 7: Benchmark comparison ----
    btc_col = [c for c in price_matrix.columns if "BTC" in c]
    if btc_col:
        btc_prices = price_matrix[btc_col[0]].loc[test_log_prices.index[0]:]
        btc_returns = btc_prices.pct_change().fillna(0)
        btc_cumulative = (1 + btc_returns).cumprod()

        # Compute correlation with BTC
        corr_with_btc = returns["net_return"].corr(btc_returns)
        print(f"Correlation with BTC: {corr_with_btc:.4f}")
        print()

    # ---- Step 8: Generate plots ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Equity curve vs BTC
    ax1 = axes[0]
    ax1.plot(managed["cumulative"], label="Stat-Arb (risk-managed)", linewidth=1.5, color="blue")
    ax1.plot(returns["cumulative"], label="Stat-Arb (raw)", linewidth=1, color="lightblue", alpha=0.7)
    if btc_col:
        btc_norm = btc_cumulative / btc_cumulative.iloc[0]
        ax1.plot(btc_norm, label="Buy & Hold BTC", linewidth=1, color="orange", alpha=0.7)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Strategy vs Buy & Hold BTC")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Drawdown
    ax2 = axes[1]
    dd = compute_drawdown(managed["cumulative"])
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax2.plot(dd, color="red", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Underwater Plot")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Daily returns
    ax3 = axes[2]
    ax3.bar(managed["net_return"].index, managed["net_return"].values,
            color=["green" if x > 0 else "red" for x in managed["net_return"].values],
            alpha=0.5, width=1)
    ax3.set_ylabel("Daily Return")
    ax3.set_title("Daily Returns Distribution")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/equity_curve.png", dpi=150, bbox_inches="tight")
    print("Saved: results/equity_curve.png")

    # ---- Step 9: Save results ----
    results_data = {
        "strategy": "crypto_stat_arb",
        "test_period": f"{test_log_prices.index[0].date()} to {test_log_prices.index[-1].date()}",
        "pairs_count": len(pairs),
        "pairs": [p.to_dict() for p in pairs],
        "raw_metrics": raw_metrics.to_dict(),
        "managed_metrics": managed_metrics.to_dict(),
    }
    if btc_col:
        results_data["btc_correlation"] = f"{corr_with_btc:.4f}"

    with open("results/backtest_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print("Saved: results/backtest_results.json")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
