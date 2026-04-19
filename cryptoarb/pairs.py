"""Pair discovery via cointegration testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


@dataclass
class PairResult:
    """Result of cointegration test for a single pair."""

    asset_a: str
    asset_b: str
    beta: float
    intercept: float
    adf_stat: float
    adf_pvalue: float
    half_life: float
    correlation: float
    cointegrated: bool

    def to_dict(self) -> dict:
        return {
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "beta": self.beta,
            "intercept": self.intercept,
            "adf_stat": self.adf_stat,
            "adf_pvalue": self.adf_pvalue,
            "half_life": self.half_life,
            "correlation": self.correlation,
            "cointegrated": self.cointegrated,
        }


def compute_half_life(spread: pd.Series) -> float:
    """Compute the half-life of mean reversion for a spread.

    Fits an AR(1) model: spread_t = c + theta * spread_{t-1} + eps
    Half-life = -ln(2) / ln(theta)

    Args:
        spread: The spread time series.

    Returns:
        Half-life in periods (days). Returns inf if non-mean-reverting.
    """
    spread_clean = spread.dropna()
    if len(spread_clean) < 20:
        return float("inf")

    y = spread_clean.values[1:]
    x = spread_clean.values[:-1]
    x_with_const = add_constant(x)

    model = OLS(y, x_with_const).fit()
    theta = model.params[1]

    if theta >= 1.0 or theta <= 0.0:
        logger.debug(f"Non-mean-reverting spread: theta={theta:.4f}")
        return float("inf")

    return -np.log(2) / np.log(theta)


def test_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    adf_pvalue_threshold: float = 0.05,
) -> PairResult:
    """Test cointegration between two log-price series (Engle-Granger method).

    1. Run OLS: log(P_A) = alpha + beta * log(P_B) + residual
    2. Test residuals for stationarity with ADF
    3. Compute half-life of mean reversion

    Args:
        price_a: Log prices of asset A.
        price_b: Log prices of asset B.
        adf_pvalue_threshold: Max p-value to accept cointegration.

    Returns:
        PairResult with test statistics and cointegration decision.
    """
    name_a = price_a.name or "A"
    name_b = price_b.name or "B"

    # Align series
    combined = pd.concat([price_a, price_b], axis=1).dropna()
    if len(combined) < 60:
        return PairResult(
            asset_a=name_a, asset_b=name_b, beta=0, intercept=0,
            adf_stat=0, adf_pvalue=1.0, half_life=float("inf"),
            correlation=0, cointegrated=False,
        )

    a = combined.iloc[:, 0].values
    b = combined.iloc[:, 1].values

    correlation = float(np.corrcoef(a, b)[0, 1])

    # OLS regression: A = alpha + beta * B
    b_with_const = add_constant(b)
    model = OLS(a, b_with_const).fit()
    intercept = model.params[0]
    beta = model.params[1]

    # Spread (residuals)
    spread = pd.Series(a - (intercept + beta * b), index=combined.index)

    # ADF test on spread
    adf_result = adfuller(spread.values, maxlag=int(np.sqrt(len(spread))))
    adf_stat = float(adf_result[0])
    adf_pvalue = float(adf_result[1])

    # Half-life
    hl = compute_half_life(spread)

    cointegrated = adf_pvalue < adf_pvalue_threshold

    return PairResult(
        asset_a=name_a,
        asset_b=name_b,
        beta=float(beta),
        intercept=float(intercept),
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        half_life=hl,
        correlation=correlation,
        cointegrated=cointegrated,
    )


def discover_pairs(
    log_prices: pd.DataFrame,
    min_correlation: float = 0.70,
    adf_pvalue: float = 0.05,
    min_half_life: float = 3.0,
    max_half_life: float = 30.0,
    max_pairs: int = 20,
) -> List[PairResult]:
    """Discover cointegrated pairs from a price matrix.

    Pipeline:
    1. Pre-filter by correlation (reduces combinations)
    2. Test cointegration (Engle-Granger)
    3. Filter by half-life
    4. Rank by ADF statistic and select top pairs

    Args:
        log_prices: Log price matrix (dates × symbols).
        min_correlation: Pre-filter threshold.
        adf_pvalue: Max ADF p-value for cointegration.
        min_half_life: Min half-life in days.
        max_half_life: Max half-life in days.
        max_pairs: Max pairs to return.

    Returns:
        List of PairResult, sorted by ADF statistic (strongest first).
    """
    symbols = log_prices.columns.tolist()
    n = len(symbols)
    total_combinations = n * (n - 1) // 2
    logger.info(f"Testing {n} assets → {total_combinations} possible pairs")

    # Pre-filter by correlation
    corr_matrix = log_prices.corr()
    candidates = []
    for i, j in combinations(range(n), 2):
        corr = abs(corr_matrix.iloc[i, j])
        if corr >= min_correlation:
            candidates.append((symbols[i], symbols[j]))

    logger.info(f"After correlation filter (>{min_correlation}): {len(candidates)} pairs")

    # Test cointegration
    results = []
    for sym_a, sym_b in candidates:
        result = test_cointegration(
            log_prices[sym_a], log_prices[sym_b],
            adf_pvalue_threshold=adf_pvalue,
        )
        if result.cointegrated:
            results.append(result)

    logger.info(f"Cointegrated pairs (p<{adf_pvalue}): {len(results)}")

    # Filter by half-life
    results = [
        r for r in results
        if min_half_life <= r.half_life <= max_half_life
    ]
    logger.info(f"After half-life filter ({min_half_life}-{max_half_life}d): {len(results)}")

    # Sort by ADF statistic (most negative = strongest cointegration)
    results.sort(key=lambda r: r.adf_stat)

    # Take top N
    results = results[:max_pairs]
    logger.info(f"Selected top {len(results)} pairs")

    for r in results:
        logger.info(
            f"  {r.asset_a}/{r.asset_b}: β={r.beta:.4f}, "
            f"ADF={r.adf_stat:.2f} (p={r.adf_pvalue:.4f}), "
            f"HL={r.half_life:.1f}d, corr={r.correlation:.3f}"
        )

    return results
