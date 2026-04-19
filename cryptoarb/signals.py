"""Signal generation — spread computation, z-scores, entry/exit logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from cryptoarb.pairs import PairResult

logger = logging.getLogger(__name__)


@dataclass
class PairSignals:
    """Signals for a single pair over time."""

    asset_a: str
    asset_b: str
    spread: pd.Series
    z_score: pd.Series
    position: pd.Series  # +1 (long spread), -1 (short spread), 0 (flat)
    rolling_beta: pd.Series
    rolling_intercept: pd.Series


def compute_rolling_hedge_ratio(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    window: int = 60,
) -> tuple[pd.Series, pd.Series]:
    """Compute rolling OLS hedge ratio (beta) and intercept.

    At each time t, fits OLS on [t-window, t]:
        log(P_A) = alpha + beta * log(P_B)

    Args:
        log_price_a: Log prices of asset A.
        log_price_b: Log prices of asset B.
        window: Rolling window size in days.

    Returns:
        Tuple of (rolling_beta, rolling_intercept) as pd.Series.
    """
    n = len(log_price_a)
    betas = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)

    a_vals = log_price_a.values
    b_vals = log_price_b.values

    for i in range(window, n):
        a_win = a_vals[i - window : i]
        b_win = b_vals[i - window : i]
        b_const = add_constant(b_win)

        try:
            model = OLS(a_win, b_const).fit()
            intercepts[i] = model.params[0]
            betas[i] = model.params[1]
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug(f"OLS failed at index {i}: {e}")
            # betas[i] and intercepts[i] remain NaN

    return (
        pd.Series(betas, index=log_price_a.index, name="beta"),
        pd.Series(intercepts, index=log_price_a.index, name="intercept"),
    )


def compute_spread(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    beta: pd.Series,
    intercept: pd.Series,
) -> pd.Series:
    """Compute the spread using rolling hedge ratio.

    spread_t = log(P_A)_t - (intercept_t + beta_t * log(P_B)_t)

    Args:
        log_price_a: Log prices of asset A.
        log_price_b: Log prices of asset B.
        beta: Rolling hedge ratio.
        intercept: Rolling intercept.

    Returns:
        Spread time series.
    """
    return log_price_a - (intercept + beta * log_price_b)


def compute_zscore(
    spread: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling z-score of the spread.

    z_t = (spread_t - rolling_mean) / rolling_std

    Args:
        spread: Spread time series.
        window: Rolling window for mean and std.

    Returns:
        Z-score time series.
    """
    rolling_mean = spread.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = spread.rolling(window=window, min_periods=window // 2).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    return (spread - rolling_mean) / rolling_std


def generate_positions(
    z_score: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> pd.Series:
    """Generate position signals from z-scores.

    Trading rules:
    - z > +entry_z  → SHORT spread (position = -1)
    - z < -entry_z  → LONG spread (position = +1)
    - |z| < exit_z  → FLAT (position = 0)
    - |z| > stop_z  → STOP LOSS, go flat

    Positions are held until an exit or stop signal.

    Args:
        z_score: Z-score time series.
        entry_z: Entry threshold.
        exit_z: Exit threshold.
        stop_z: Stop-loss threshold.

    Returns:
        Position series (+1, -1, or 0).
    """
    n = len(z_score)
    positions = np.zeros(n)
    current_pos = 0

    z_vals = z_score.values

    for i in range(n):
        z = z_vals[i]

        if np.isnan(z):
            positions[i] = 0
            current_pos = 0
            continue

        # Stop loss
        if abs(z) > stop_z and current_pos != 0:
            current_pos = 0

        # Entry signals (only if flat)
        elif current_pos == 0:
            if z > entry_z:
                current_pos = -1  # short the spread
            elif z < -entry_z:
                current_pos = 1  # long the spread

        # Exit signals (only if in a position)
        elif current_pos != 0:
            if abs(z) < exit_z:
                current_pos = 0

        positions[i] = current_pos

    return pd.Series(positions, index=z_score.index, name="position")


def generate_pair_signals(
    log_prices: pd.DataFrame,
    pair: PairResult,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    window_multiplier: float = 2.0,
) -> PairSignals:
    """Generate complete signals for a single pair.

    Args:
        log_prices: Log price matrix.
        pair: PairResult from pair discovery.
        entry_z: Z-score entry threshold.
        exit_z: Z-score exit threshold.
        stop_z: Z-score stop-loss threshold.
        window_multiplier: Rolling window = multiplier × half_life.

    Returns:
        PairSignals with spread, z-score, and position.
    """
    window = max(int(pair.half_life * window_multiplier), 20)

    log_a = log_prices[pair.asset_a]
    log_b = log_prices[pair.asset_b]

    rolling_beta, rolling_intercept = compute_rolling_hedge_ratio(log_a, log_b, window=window)
    spread = compute_spread(log_a, log_b, rolling_beta, rolling_intercept)
    z = compute_zscore(spread, window=window)
    position = generate_positions(z, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)

    return PairSignals(
        asset_a=pair.asset_a,
        asset_b=pair.asset_b,
        spread=spread,
        z_score=z,
        position=position,
        rolling_beta=rolling_beta,
        rolling_intercept=rolling_intercept,
    )
