"""Risk management — drawdown stops, volatility scaling, pair health."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_drawdown_stop(
    returns: pd.Series,
    max_drawdown: float = 0.15,
    cooldown_days: int = 30,
) -> pd.Series:
    """Apply a drawdown-based stop to a return series.

    When cumulative drawdown exceeds the threshold, returns are zeroed
    out for a cooldown period.

    Args:
        returns: Daily return series.
        max_drawdown: Threshold to trigger stop (e.g., 0.15 = 15%).
        cooldown_days: Days to stay flat after stop is triggered.

    Returns:
        Adjusted return series with stopped periods zeroed out.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    adjusted = returns.copy()
    stopped_until = -1

    for i in range(len(returns)):
        if i <= stopped_until:
            adjusted.iloc[i] = 0.0
            continue

        if drawdown.iloc[i] < -max_drawdown:
            logger.info(
                f"Drawdown stop triggered at {returns.index[i]}: "
                f"DD={drawdown.iloc[i]:.2%}, halting for {cooldown_days} days"
            )
            stopped_until = i + cooldown_days
            adjusted.iloc[i] = 0.0

    return adjusted


def apply_volatility_scaling(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    vol_target: float = 0.10,
    vol_lookback: int = 60,
    annualization: int = 365,
) -> pd.DataFrame:
    """Scale portfolio weights to target a specific volatility.

    When realized volatility is high, reduce exposure. When low, increase.

    Args:
        weights: Portfolio weights (dates × assets).
        returns: Asset return matrix (dates × assets).
        vol_target: Target annual volatility (e.g., 0.10 = 10%).
        vol_lookback: Lookback window for realized vol estimation.
        annualization: Days per year.

    Returns:
        Scaled weights DataFrame.
    """
    # Compute portfolio returns from weights
    port_returns = (weights.shift(1) * returns).sum(axis=1)

    # Rolling realized volatility (annualized)
    rolling_vol = port_returns.rolling(window=vol_lookback, min_periods=vol_lookback // 2).std()
    rolling_vol_annual = rolling_vol * np.sqrt(annualization)

    # Scale factor: target_vol / realized_vol
    scale = vol_target / rolling_vol_annual.replace(0, np.nan)
    scale = scale.clip(0.1, 3.0)  # cap scaling between 10% and 300%
    scale = scale.fillna(1.0)

    # Apply scale to all weights
    scaled = weights.multiply(scale, axis=0)

    return scaled


def check_pair_health(
    log_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    window: int = 60,
    adf_threshold: float = 0.10,
) -> bool:
    """Check if a pair is still cointegrated.

    Uses a rolling window ADF test on the spread.

    Args:
        log_prices: Log price matrix.
        asset_a: First asset name.
        asset_b: Second asset name.
        window: Lookback window for the check.
        adf_threshold: Max p-value to consider pair healthy.

    Returns:
        True if pair is still cointegrated, False otherwise.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from statsmodels.tsa.stattools import adfuller

    recent = log_prices[[asset_a, asset_b]].iloc[-window:].dropna()
    if len(recent) < window // 2:
        return False

    a = recent[asset_a].values
    b = recent[asset_b].values

    model = OLS(a, add_constant(b)).fit()
    spread = a - model.predict(add_constant(b))

    try:
        adf_p = adfuller(spread, maxlag=int(np.sqrt(len(spread))))[1]
        return adf_p < adf_threshold
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.warning(f"Pair health check failed for {asset_a}/{asset_b}: {e}")
        return False


def compute_pair_drawdown(
    pair_returns: pd.Series,
) -> float:
    """Compute current drawdown for a single pair's return stream.

    Args:
        pair_returns: Daily returns for the pair.

    Returns:
        Current drawdown as a negative fraction (e.g., -0.08 = 8% drawdown).
    """
    cumulative = (1 + pair_returns).cumprod()
    if len(cumulative) == 0:
        return 0.0
    running_max = cumulative.cummax()
    if running_max.iloc[-1] == 0:
        return 0.0
    current_dd = (cumulative.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
    return float(current_dd)
