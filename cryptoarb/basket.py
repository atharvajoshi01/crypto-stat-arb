"""Johansen-based basket trading — trade 3+ coin portfolios."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


@dataclass
class BasketResult:
    """Result of Johansen cointegration test for a basket of assets."""

    assets: List[str]
    weights: np.ndarray  # hedge weights (first eigenvector)
    n_cointegrating: int  # number of cointegrating relationships
    trace_stats: np.ndarray
    critical_values: np.ndarray
    half_life: float
    is_cointegrated: bool

    def to_dict(self) -> dict:
        return {
            "assets": self.assets,
            "weights": self.weights.tolist(),
            "n_cointegrating": self.n_cointegrating,
            "half_life": self.half_life,
            "is_cointegrated": self.is_cointegrated,
        }


def johansen_test(
    log_prices: pd.DataFrame,
    assets: List[str],
    det_order: int = 0,
    k_ar_diff: int = 1,
    significance: int = 1,  # 0=90%, 1=95%, 2=99%
) -> BasketResult:
    """Run Johansen cointegration test on a basket of assets.

    Tests for cointegrating relationships among N assets simultaneously.
    Returns the hedge weights from the first eigenvector.

    Args:
        log_prices: Log price matrix.
        assets: List of asset column names (3+ assets).
        det_order: Deterministic trend order (-1=no const, 0=const, 1=trend).
        k_ar_diff: Number of lagged differences in the VECM.
        significance: Critical value index (0=90%, 1=95%, 2=99%).

    Returns:
        BasketResult with hedge weights and cointegration test results.
    """
    data = log_prices[assets].dropna()

    if len(data) < 60:
        return BasketResult(
            assets=assets, weights=np.zeros(len(assets)),
            n_cointegrating=0, trace_stats=np.array([]),
            critical_values=np.array([]), half_life=float("inf"),
            is_cointegrated=False,
        )

    result = coint_johansen(data.values, det_order=det_order, k_ar_diff=k_ar_diff)

    # Count cointegrating relationships at chosen significance
    trace_stats = result.lr1  # trace statistics
    crit_vals = result.cvt[:, significance]  # critical values at significance level
    n_coint = int((trace_stats > crit_vals).sum())

    is_cointegrated = n_coint >= 1

    # First eigenvector = optimal hedge weights
    weights = result.evec[:, 0]
    # Normalize so first weight = 1
    if abs(weights[0]) > 1e-8:
        weights = weights / weights[0]

    # Compute spread using these weights
    spread = data.values @ weights
    spread_series = pd.Series(spread, index=data.index)

    # Half-life
    from cryptoarb.pairs import compute_half_life
    hl = compute_half_life(spread_series)

    logger.info(
        f"Johansen test on {assets}: "
        f"{n_coint} cointegrating relationships, HL={hl:.1f}d"
    )

    return BasketResult(
        assets=assets,
        weights=weights,
        n_cointegrating=n_coint,
        trace_stats=trace_stats,
        critical_values=crit_vals,
        half_life=hl,
        is_cointegrated=is_cointegrated,
    )


def generate_basket_spread(
    log_prices: pd.DataFrame,
    basket: BasketResult,
    rolling_window: int = 60,
) -> pd.Series:
    """Compute the basket spread using Johansen weights.

    Args:
        log_prices: Log price matrix.
        basket: BasketResult from johansen_test.
        rolling_window: Window for z-score normalization.

    Returns:
        Z-scored basket spread.
    """
    data = log_prices[basket.assets].dropna()
    spread = pd.Series(data.values @ basket.weights, index=data.index, name="basket_spread")

    z = (spread - spread.rolling(rolling_window).mean()) / spread.rolling(rolling_window).std()
    return z


def discover_baskets(
    log_prices: pd.DataFrame,
    basket_size: int = 3,
    min_correlation: float = 0.70,
    max_baskets: int = 5,
) -> List[BasketResult]:
    """Discover cointegrated baskets from a price matrix.

    Pre-filters by average pairwise correlation, then runs Johansen
    on candidate groups.

    Args:
        log_prices: Log price matrix.
        basket_size: Number of assets per basket.
        min_correlation: Min average pairwise correlation to test.
        max_baskets: Max baskets to return.

    Returns:
        List of BasketResult, sorted by number of cointegrating relationships.
    """
    from itertools import combinations

    symbols = log_prices.columns.tolist()
    corr = log_prices.corr()

    results = []

    for combo in combinations(symbols, basket_size):
        combo_list = list(combo)

        # Check average pairwise correlation
        sub_corr = corr.loc[combo_list, combo_list]
        avg_corr = (sub_corr.values.sum() - len(combo_list)) / (len(combo_list) * (len(combo_list) - 1))

        if avg_corr < min_correlation:
            continue

        basket = johansen_test(log_prices, combo_list)
        if basket.is_cointegrated and 2 < basket.half_life < 40:
            results.append(basket)

    # Sort by number of cointegrating relationships (more = better)
    results.sort(key=lambda b: (-b.n_cointegrating, b.half_life))

    return results[:max_baskets]
