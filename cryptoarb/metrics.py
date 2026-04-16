"""Performance metrics and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Aggregated strategy performance metrics."""

    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    calmar_ratio: float
    win_rate: float
    avg_daily_turnover: float
    total_cost: float
    total_days: int

    def to_dict(self) -> Dict:
        return {
            "total_return": f"{self.total_return:.2%}",
            "annual_return": f"{self.annual_return:.2%}",
            "annual_volatility": f"{self.annual_volatility:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "max_drawdown_duration_days": self.max_drawdown_duration,
            "calmar_ratio": f"{self.calmar_ratio:.2f}",
            "win_rate": f"{self.win_rate:.1%}",
            "avg_daily_turnover": f"{self.avg_daily_turnover:.4f}",
            "total_cost": f"{self.total_cost:.4f}",
            "total_days": self.total_days,
        }

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "STRATEGY PERFORMANCE",
            "=" * 50,
            f"Total Return:       {self.total_return:.2%}",
            f"Annual Return:      {self.annual_return:.2%}",
            f"Annual Volatility:  {self.annual_volatility:.2%}",
            f"Sharpe Ratio:       {self.sharpe_ratio:.2f}",
            f"Sortino Ratio:      {self.sortino_ratio:.2f}",
            f"Max Drawdown:       {self.max_drawdown:.2%}",
            f"Max DD Duration:    {self.max_drawdown_duration} days",
            f"Calmar Ratio:       {self.calmar_ratio:.2f}",
            f"Win Rate:           {self.win_rate:.1%}",
            f"Avg Daily Turnover: {self.avg_daily_turnover:.4f}",
            f"Total Cost:         {self.total_cost:.4f}",
            f"Trading Days:       {self.total_days}",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_drawdown(cumulative: pd.Series) -> pd.Series:
    """Compute drawdown series from cumulative returns.

    Handles edge case where running max is zero (e.g., strategy loses
    all value) by treating those points as zero drawdown.
    """
    running_max = cumulative.cummax()
    safe_max = running_max.replace(0, np.nan)
    return ((cumulative - running_max) / safe_max).fillna(0.0)


def compute_max_drawdown_duration(cumulative: pd.Series) -> int:
    """Compute the longest drawdown duration in days."""
    running_max = cumulative.cummax()
    is_in_dd = cumulative < running_max

    if not is_in_dd.any():
        return 0

    # Find consecutive drawdown periods
    dd_groups = (~is_in_dd).cumsum()
    dd_groups = dd_groups[is_in_dd]

    if dd_groups.empty:
        return 0

    return int(dd_groups.value_counts().max())


def evaluate(returns_df: pd.DataFrame, annualization: int = 365) -> PerformanceMetrics:
    """Compute performance metrics from backtest returns.

    Args:
        returns_df: DataFrame with columns: net_return, gross_return, cost, turnover, cumulative.
        annualization: Days per year (365 for crypto).

    Returns:
        PerformanceMetrics with all strategy statistics.
    """
    net = returns_df["net_return"].dropna()
    cumulative = returns_df["cumulative"].dropna()

    if len(net) == 0:
        return PerformanceMetrics(
            total_return=0, annual_return=0, annual_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            max_drawdown_duration=0, calmar_ratio=0, win_rate=0,
            avg_daily_turnover=0, total_cost=0, total_days=0,
        )

    n_days = len(net)
    years = n_days / annualization

    # Total and annual return
    total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1 if cumulative.iloc[0] != 0 else 0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility
    daily_vol = net.std()
    annual_vol = daily_vol * np.sqrt(annualization)

    # Sharpe
    daily_mean = net.mean()
    sharpe = (daily_mean / daily_vol * np.sqrt(annualization)) if daily_vol > 0 else 0

    # Sortino (downside deviation)
    downside = net[net < 0]
    downside_vol = downside.std() if len(downside) > 0 else 0
    sortino = (daily_mean / downside_vol * np.sqrt(annualization)) if downside_vol > 0 else 0

    # Drawdown
    dd = compute_drawdown(cumulative)
    max_dd = float(dd.min()) if len(dd) > 0 else 0
    max_dd_duration = compute_max_drawdown_duration(cumulative)

    # Calmar
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (net > 0).mean()

    # Turnover and costs
    avg_turnover = returns_df["turnover"].mean() if "turnover" in returns_df else 0
    total_cost = returns_df["cost"].sum() if "cost" in returns_df else 0

    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        calmar_ratio=calmar,
        win_rate=win_rate,
        avg_daily_turnover=avg_turnover,
        total_cost=total_cost,
        total_days=n_days,
    )
