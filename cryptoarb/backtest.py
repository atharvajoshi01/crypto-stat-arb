"""Walk-forward backtesting engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

from cryptoarb.config import StrategyConfig
from cryptoarb.pairs import discover_pairs, PairResult
from cryptoarb.portfolio import build_portfolio, compute_portfolio_returns
from cryptoarb.signals import generate_pair_signals

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a walk-forward backtest."""

    returns: pd.DataFrame  # daily returns with gross, net, cost, cumulative
    pairs_used: List[List[PairResult]]  # pairs per window
    n_windows: int = 0
    total_days: int = 0

    @property
    def net_returns(self) -> pd.Series:
        return self.returns["net_return"]

    @property
    def cumulative(self) -> pd.Series:
        return self.returns["cumulative"]


def run_backtest(
    log_prices: pd.DataFrame,
    config: StrategyConfig,
) -> BacktestResult:
    """Run walk-forward backtest.

    For each window:
    1. Use training period to discover pairs
    2. Generate signals on test period using discovered pairs
    3. Build portfolio and compute returns
    4. Roll forward and repeat

    Args:
        log_prices: Log price matrix (dates × symbols).
        config: Strategy configuration.

    Returns:
        BacktestResult with daily returns across all windows.
    """
    train_days = config.backtest.train_window_days
    test_days = config.backtest.test_window_days
    step_days = config.backtest.step_days
    n_total = len(log_prices)

    all_returns = []
    all_pairs = []
    window_count = 0

    start = 0
    while start + train_days + test_days <= n_total:
        train_end = start + train_days
        test_end = min(train_end + test_days, n_total)

        train_data = log_prices.iloc[start:train_end]
        test_data = log_prices.iloc[train_end:test_end]

        logger.info(
            f"Window {window_count + 1}: "
            f"train {train_data.index[0].date()}–{train_data.index[-1].date()}, "
            f"test {test_data.index[0].date()}–{test_data.index[-1].date()}"
        )

        # Discover pairs on training data
        pairs = discover_pairs(
            train_data,
            min_correlation=config.pairs.min_correlation,
            adf_pvalue=config.pairs.adf_pvalue,
            min_half_life=config.pairs.min_half_life,
            max_half_life=config.pairs.max_half_life,
            max_pairs=config.pairs.max_pairs,
        )

        if not pairs:
            logger.warning("  No pairs found, skipping window")
            start += step_days
            continue

        all_pairs.append(pairs)

        # Generate signals on combined train+test (need history for rolling calcs)
        full_window = log_prices.iloc[start:test_end]
        signals = []
        for pair in pairs:
            sig = generate_pair_signals(
                full_window, pair,
                entry_z=config.signals.entry_z,
                exit_z=config.signals.exit_z,
                stop_z=config.signals.stop_z,
                window_multiplier=config.signals.rolling_window_multiplier,
            )
            signals.append(sig)

        # Build portfolio on full window
        weights = build_portfolio(
            signals, full_window,
            max_pair_weight=config.portfolio.max_pair_weight,
        )

        # Only keep test period returns
        test_weights = weights.loc[test_data.index[0]:test_data.index[-1]]
        test_log_prices = log_prices.loc[test_data.index[0]:test_data.index[-1]]

        window_returns = compute_portfolio_returns(
            test_weights, test_log_prices,
            cost_bps=config.costs.round_trip_bps,
        )

        all_returns.append(window_returns)
        window_count += 1
        start += step_days

    if not all_returns:
        empty = pd.DataFrame(columns=["gross_return", "net_return", "cost", "turnover", "cumulative"])
        return BacktestResult(returns=empty, pairs_used=[], n_windows=0, total_days=0)

    # Concatenate all windows (remove overlaps)
    combined = pd.concat(all_returns)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    # Recompute cumulative from combined net returns
    combined["cumulative"] = (1 + combined["net_return"]).cumprod()

    logger.info(
        f"Backtest complete: {window_count} windows, "
        f"{len(combined)} days, "
        f"final equity: {combined['cumulative'].iloc[-1]:.4f}"
    )

    return BacktestResult(
        returns=combined,
        pairs_used=all_pairs,
        n_windows=window_count,
        total_days=len(combined),
    )
