"""Parameter sensitivity analysis — sweep parameters and measure robustness."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import List

import pandas as pd

from cryptoarb.pairs import PairResult
from cryptoarb.signals import generate_pair_signals
from cryptoarb.portfolio import build_portfolio, compute_portfolio_returns
from cryptoarb.metrics import evaluate

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """Result of a single parameter combination."""

    entry_z: float
    exit_z: float
    cost_bps: float
    sharpe: float
    annual_return: float
    max_drawdown: float
    win_rate: float
    n_trades: int


@dataclass
class SensitivityReport:
    """Full parameter sensitivity analysis report."""

    results: List[SweepResult] = field(default_factory=list)

    @property
    def best_sharpe(self) -> SweepResult:
        return max(self.results, key=lambda r: r.sharpe)

    @property
    def best_return(self) -> SweepResult:
        return max(self.results, key=lambda r: r.annual_return)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "entry_z": r.entry_z,
                "exit_z": r.exit_z,
                "cost_bps": r.cost_bps,
                "sharpe": r.sharpe,
                "annual_return": r.annual_return,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "n_trades": r.n_trades,
            }
            for r in self.results
        ])

    def sharpe_heatmap(self, cost_bps: float | None = None) -> pd.DataFrame:
        """Pivot table of Sharpe ratio by entry_z and exit_z.

        Args:
            cost_bps: Filter to a specific cost level. If None, uses the first.

        Returns:
            DataFrame with entry_z as index, exit_z as columns, Sharpe as values.
        """
        df = self.to_dataframe()
        if cost_bps is not None:
            df = df[df["cost_bps"] == cost_bps]
        elif df["cost_bps"].nunique() > 1:
            df = df[df["cost_bps"] == df["cost_bps"].iloc[0]]

        return df.pivot_table(
            index="entry_z", columns="exit_z", values="sharpe", aggfunc="first"
        )

    def summary(self) -> str:
        best = self.best_sharpe
        lines = [
            "=" * 60,
            "PARAMETER SENSITIVITY ANALYSIS",
            "=" * 60,
            f"Combinations tested: {len(self.results)}",
            "",
            f"Best Sharpe: {best.sharpe:.2f}",
            f"  entry_z={best.entry_z}, exit_z={best.exit_z}, cost={best.cost_bps}bps",
            f"  Annual return: {best.annual_return:.2%}",
            f"  Max drawdown: {best.max_drawdown:.2%}",
            f"  Win rate: {best.win_rate:.1%}",
            "",
            f"Sharpe Range: [{min(r.sharpe for r in self.results):.2f}, "
            f"{max(r.sharpe for r in self.results):.2f}]",
            "=" * 60,
        ]
        return "\n".join(lines)


def run_sensitivity(
    log_prices: pd.DataFrame,
    pairs: List[PairResult],
    entry_z_range: List[float] | None = None,
    exit_z_range: List[float] | None = None,
    cost_bps_range: List[float] | None = None,
    test_start_idx: int = 504,
) -> SensitivityReport:
    """Sweep signal parameters and evaluate each combination.

    Uses pre-discovered pairs (to avoid re-running cointegration) and
    sweeps over entry/exit z-score thresholds and cost assumptions.

    Args:
        log_prices: Log price matrix.
        pairs: Pre-discovered pairs from pair discovery.
        entry_z_range: List of entry z-score thresholds to test.
        exit_z_range: List of exit z-score thresholds to test.
        cost_bps_range: List of round-trip costs in bps to test.
        test_start_idx: Index where out-of-sample period begins.

    Returns:
        SensitivityReport with results for all combinations.
    """
    if entry_z_range is None:
        entry_z_range = [1.5, 2.0, 2.5, 3.0]
    if exit_z_range is None:
        exit_z_range = [0.25, 0.5, 0.75, 1.0]
    if cost_bps_range is None:
        cost_bps_range = [20, 40, 60]

    total = len(entry_z_range) * len(exit_z_range) * len(cost_bps_range)
    logger.info(f"Running sensitivity: {total} combinations")

    report = SensitivityReport()
    test_log_prices = log_prices.iloc[test_start_idx:]

    for i, (entry_z, exit_z, cost_bps) in enumerate(
        product(entry_z_range, exit_z_range, cost_bps_range)
    ):
        if exit_z >= entry_z:
            continue  # exit must be below entry

        # Generate signals with these parameters
        signals = []
        for pair in pairs:
            sig = generate_pair_signals(
                log_prices, pair,
                entry_z=entry_z,
                exit_z=exit_z,
            )
            signals.append(sig)

        # Build portfolio and compute returns
        weights = build_portfolio(signals, log_prices)
        test_weights = weights.iloc[test_start_idx:]

        returns = compute_portfolio_returns(
            test_weights, test_log_prices, cost_bps=cost_bps,
        )

        metrics = evaluate(returns, annualization=365)

        # Count trades (position changes)
        total_trades = sum(
            (sig.position.diff().abs() > 0).sum() for sig in signals
        )

        report.results.append(SweepResult(
            entry_z=entry_z,
            exit_z=exit_z,
            cost_bps=cost_bps,
            sharpe=metrics.sharpe_ratio,
            annual_return=metrics.annual_return,
            max_drawdown=metrics.max_drawdown,
            win_rate=metrics.win_rate,
            n_trades=int(total_trades),
        ))

    logger.info(f"Sensitivity complete: {len(report.results)} valid combinations")
    return report
