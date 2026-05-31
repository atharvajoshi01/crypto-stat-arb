"""
Turnover-Penalty Sweep
======================

Runs the synthetic stat-arb backtest at multiple entry-z thresholds and reports
how the ranking of strategies changes when you score them on:

  1. Net Sharpe (the deployable number, after costs)
  2. Sharpe per unit of average daily turnover (the decision-quality score)

The point is to show, in numbers, the tension Kolm describes in his Future Alpha
piece: a signal that increases accuracy but also increases turnover may reduce
rather than improve the investment process. The "winning" strategy depends on
which metric you optimize for.

Usage:
    python examples/turnover_penalty_sweep.py
"""

import numpy as np
import pandas as pd

from cryptoarb.config import StrategyConfig
from cryptoarb.data import log_prices
from cryptoarb.metrics import evaluate
from cryptoarb.pairs import discover_pairs
from cryptoarb.portfolio import build_portfolio, compute_portfolio_returns
from cryptoarb.signals import generate_pair_signals


def generate_synthetic_prices(n_coins=10, n_days=1500, seed=42):
    """Same generator as the synthetic backtest example for reproducibility."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    prices = {}

    base1 = rng.normal(0, 0.02, n_days).cumsum() + np.log(50000)
    for i in range(3):
        noise = np.zeros(n_days)
        for t in range(1, n_days):
            noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 0.01)
        beta = 0.8 + rng.uniform(-0.3, 0.3)
        prices[f"COIN_{i}"] = np.exp(0.5 + beta * base1 + noise)

    base2 = rng.normal(0, 0.025, n_days).cumsum() + np.log(3000)
    for i in range(3, 6):
        noise = np.zeros(n_days)
        for t in range(1, n_days):
            noise[t] = 0.80 * noise[t - 1] + rng.normal(0, 0.015)
        beta = 0.9 + rng.uniform(-0.2, 0.2)
        prices[f"COIN_{i}"] = np.exp(0.3 + beta * base2 + noise)

    for i in range(6, n_coins):
        prices[f"COIN_{i}"] = np.exp(rng.normal(0, 0.03, n_days).cumsum() + np.log(100))

    return pd.DataFrame(prices, index=dates)


def run_one(log_price_matrix, pairs, config, label):
    """Run a single configuration end-to-end and return the metrics block."""
    signals = []
    for pair in pairs:
        sig = generate_pair_signals(
            log_price_matrix, pair,
            entry_z=config.signals.entry_z,
            exit_z=config.signals.exit_z,
        )
        signals.append(sig)

    test_start = 504
    test_log_prices = log_price_matrix.iloc[test_start:]
    weights = build_portfolio(signals, log_price_matrix)
    test_weights = weights.iloc[test_start:]
    returns = compute_portfolio_returns(
        test_weights, test_log_prices,
        cost_bps=config.costs.round_trip_bps,
    )
    m = evaluate(returns)
    return {
        "label": label,
        "entry_z": config.signals.entry_z,
        "net_sharpe": round(m.sharpe_ratio, 3),
        "gross_sharpe": round(m.gross_sharpe_ratio, 3),
        "cost_drag": round(m.cost_drag, 3),
        "avg_turnover": round(m.avg_daily_turnover, 4),
        "sharpe_per_turnover": round(m.sharpe_per_unit_turnover, 2),
        "max_dd": round(m.max_drawdown, 3),
    }


def main():
    print("=" * 72)
    print("TURNOVER-PENALTY SWEEP · synthetic crypto stat-arb")
    print("=" * 72)
    print()
    print("Same pairs, same signal logic, sweeping entry-z. Lower z = more trades.")
    print("Two ways to rank: net Sharpe vs Sharpe per unit turnover.")
    print()

    raw_prices = generate_synthetic_prices(n_coins=10, n_days=1500)
    log_price_matrix = log_prices(raw_prices)

    base = StrategyConfig()
    base.pairs.min_correlation = 0.60
    base.pairs.min_half_life = 2.0
    base.pairs.max_half_life = 40.0
    base.pairs.max_pairs = 10
    base.signals.exit_z = 0.5
    base.costs.taker_fee_bps = 10
    base.costs.slippage_bps = 5

    train_data = log_price_matrix.iloc[:504]
    pairs = discover_pairs(
        train_data,
        min_correlation=base.pairs.min_correlation,
        adf_pvalue=base.pairs.adf_pvalue,
        min_half_life=base.pairs.min_half_life,
        max_half_life=base.pairs.max_half_life,
        max_pairs=base.pairs.max_pairs,
    )
    if not pairs:
        print("No cointegrated pairs discovered. Try a different seed.")
        return

    print(f"Discovered {len(pairs)} pairs on the first 2 years of synthetic data.")
    print()

    sweep = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    rows = []
    for z in sweep:
        cfg = base.model_copy(deep=True)
        cfg.signals.entry_z = z
        rows.append(run_one(log_price_matrix, pairs, cfg, label=f"entry_z={z}"))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    # Ranking comparison
    by_net = df.sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    by_dq = df.sort_values("sharpe_per_turnover", ascending=False).reset_index(drop=True)

    print("-" * 72)
    print("RANKING BY NET SHARPE (the deployable number):")
    for i, row in by_net.iterrows():
        print(f"  {i + 1}. entry_z={row['entry_z']:<5}  net_sharpe={row['net_sharpe']:.3f}  "
              f"turnover={row['avg_turnover']:.4f}  sharpe_per_turnover={row['sharpe_per_turnover']:.2f}")
    print()
    print("RANKING BY SHARPE PER UNIT TURNOVER (decision-quality score):")
    for i, row in by_dq.iterrows():
        print(f"  {i + 1}. entry_z={row['entry_z']:<5}  net_sharpe={row['net_sharpe']:.3f}  "
              f"turnover={row['avg_turnover']:.4f}  sharpe_per_turnover={row['sharpe_per_turnover']:.2f}")
    print()

    top_net = by_net.iloc[0]["entry_z"]
    top_dq = by_dq.iloc[0]["entry_z"]
    print("-" * 72)
    if top_net == top_dq:
        print(f"  Same winner under both metrics: entry_z={top_net}.")
        print("  No tension surfaced in this sweep.")
    else:
        print(f"  WINNER FLIPS:")
        print(f"    Net Sharpe picks            entry_z={top_net}")
        print(f"    Sharpe per unit turnover picks  entry_z={top_dq}")
        print(f"  The 'best' strategy depends on which metric you trust.")
        print(f"  This is Kolm's argument 4 made measurable on a real backtest.")
    print()


if __name__ == "__main__":
    main()
