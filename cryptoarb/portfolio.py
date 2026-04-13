"""Portfolio construction — position sizing and dollar neutrality."""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from cryptoarb.signals import PairSignals

logger = logging.getLogger(__name__)


def build_portfolio(
    all_signals: List[PairSignals],
    log_prices: pd.DataFrame,
    max_pair_weight: float = 0.20,
) -> pd.DataFrame:
    """Build a dollar-neutral portfolio from pair signals.

    Each pair gets equal allocation (1/K). Within each pair, position sizes
    are set using the rolling hedge ratio to achieve dollar neutrality.

    Args:
        all_signals: List of PairSignals from signal generation.
        log_prices: Log price matrix for computing returns.
        max_pair_weight: Max allocation per pair.

    Returns:
        Portfolio weights DataFrame (dates × assets). Positive = long, negative = short.
    """
    if not all_signals:
        return pd.DataFrame(index=log_prices.index)

    n_pairs = len(all_signals)
    pair_weight = min(1.0 / n_pairs, max_pair_weight)

    # Build asset-level weights
    all_assets = set()
    for sig in all_signals:
        all_assets.add(sig.asset_a)
        all_assets.add(sig.asset_b)

    weights = pd.DataFrame(0.0, index=log_prices.index, columns=sorted(all_assets))

    for sig in all_signals:
        # Position in spread: +1 = long A, short B; -1 = short A, long B
        pos = sig.position
        beta = sig.rolling_beta.fillna(1.0)

        # Normalize so that dollar exposure per pair = pair_weight
        # Leg A: position * pair_weight / (1 + |beta|)
        # Leg B: -position * beta * pair_weight / (1 + |beta|)
        normalizer = 1.0 + beta.abs()

        weight_a = pos * pair_weight / normalizer
        weight_b = -pos * beta * pair_weight / normalizer

        weights[sig.asset_a] = weights[sig.asset_a] + weight_a
        weights[sig.asset_b] = weights[sig.asset_b] + weight_b

    return weights


def compute_portfolio_returns(
    weights: pd.DataFrame,
    log_prices: pd.DataFrame,
    cost_bps: float = 40.0,
) -> pd.DataFrame:
    """Compute daily portfolio returns with transaction costs.

    Args:
        weights: Portfolio weights (dates × assets).
        log_prices: Log price matrix.
        cost_bps: Round-trip cost in basis points.

    Returns:
        DataFrame with columns: gross_return, turnover, cost, net_return, cumulative.
    """
    # Compute asset returns from log prices
    asset_returns = log_prices.diff()

    # Align
    common_cols = weights.columns.intersection(asset_returns.columns)
    weights = weights[common_cols]
    asset_returns = asset_returns[common_cols]

    # Gross return: sum of (yesterday's weight × today's return)
    gross = (weights.shift(1) * asset_returns).sum(axis=1)

    # Turnover: sum of absolute weight changes
    turnover = weights.diff().abs().sum(axis=1)

    # Cost: turnover × cost_per_unit
    cost_per_unit = cost_bps / 10_000
    cost = turnover * cost_per_unit

    # Net return
    net = gross - cost

    result = pd.DataFrame({
        "gross_return": gross,
        "turnover": turnover,
        "cost": cost,
        "net_return": net,
    }, index=weights.index)

    result["cumulative"] = (1 + result["net_return"]).cumprod()

    return result


def check_dollar_neutrality(weights: pd.DataFrame) -> pd.Series:
    """Check how close to dollar neutral the portfolio is each day.

    Returns the net exposure (sum of weights) — should be close to 0.

    Args:
        weights: Portfolio weights.

    Returns:
        Net exposure per day.
    """
    return weights.sum(axis=1)
