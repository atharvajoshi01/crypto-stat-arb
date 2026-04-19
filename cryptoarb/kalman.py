"""Kalman filter for adaptive hedge ratio estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """State of the Kalman filter at a point in time."""

    beta: float  # current hedge ratio estimate
    intercept: float  # current intercept estimate
    P: np.ndarray  # state covariance matrix (2×2)


def kalman_hedge_ratio(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    delta: float = 1e-4,
    Ve: float = 1e-3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Estimate time-varying hedge ratio using a Kalman filter.

    Models the relationship as:
        log(P_A)_t = intercept_t + beta_t * log(P_B)_t + noise

    where intercept and beta evolve as random walks:
        [intercept_t]   [intercept_{t-1}]
        [beta_t     ] = [beta_{t-1}     ] + process_noise

    The Kalman filter produces smoother, less noisy estimates than
    rolling OLS, and adapts faster to structural breaks.

    Args:
        log_price_a: Log prices of asset A.
        log_price_b: Log prices of asset B.
        delta: Process noise variance (controls how fast beta can change).
            Smaller = smoother. Typical: 1e-5 to 1e-3.
        Ve: Measurement noise variance (observation uncertainty).

    Returns:
        Tuple of (beta_series, intercept_series, spread_series).
    """
    n = len(log_price_a)
    a = log_price_a.values
    b = log_price_b.values

    # State: [intercept, beta]
    # Initialize
    theta = np.array([0.0, 1.0])  # initial guess: intercept=0, beta=1
    P = np.eye(2)  # initial state covariance
    Vw = delta / (1 - delta) * np.eye(2)  # process noise covariance

    betas = np.zeros(n)
    intercepts = np.zeros(n)
    spreads = np.zeros(n)

    for t in range(n):
        # Observation: y_t = [1, b_t] @ theta + noise
        y = a[t]
        F = np.array([1.0, b[t]])  # observation matrix (1×2)

        # Predict
        # State prediction: theta stays the same (random walk)
        # Covariance prediction: P = P + Vw
        P = P + Vw

        # Innovation
        y_hat = F @ theta
        e = y - y_hat  # innovation (prediction error)

        # Innovation covariance
        S = F @ P @ F + Ve

        # Guard against singular/negative innovation covariance
        if S <= 0:
            logger.warning(f"Kalman filter: non-positive innovation variance at t={t}, resetting P")
            P = np.eye(2)
            continue

        # Kalman gain
        K = P @ F / S  # (2×1)

        # Update
        theta = theta + K * e
        P = P - np.outer(K, F @ P)

        # Ensure P stays positive semi-definite
        P = (P + P.T) / 2
        P = np.maximum(P, 1e-10 * np.eye(2))

        # Store
        intercepts[t] = theta[0]
        betas[t] = theta[1]
        spreads[t] = e  # the residual IS the spread

    return (
        pd.Series(betas, index=log_price_a.index, name="kalman_beta"),
        pd.Series(intercepts, index=log_price_a.index, name="kalman_intercept"),
        pd.Series(spreads, index=log_price_a.index, name="kalman_spread"),
    )


def kalman_zscore(
    spread: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute z-score of the Kalman filter spread.

    Args:
        spread: Kalman filter residual (prediction error) series.
        window: Rolling window for mean and std.

    Returns:
        Z-scored spread.
    """
    rolling_mean = spread.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = spread.rolling(window=window, min_periods=window // 2).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (spread - rolling_mean) / rolling_std
