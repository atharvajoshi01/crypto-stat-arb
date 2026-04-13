"""Regime detection — identify market states to adapt strategy behavior."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""

    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class RegimeResult:
    """Regime detection result."""

    regimes: pd.Series  # MarketRegime per date
    vol_series: pd.Series  # rolling volatility
    thresholds: dict  # vol thresholds used

    @property
    def regime_counts(self) -> dict:
        return self.regimes.value_counts().to_dict()

    def current_regime(self) -> str:
        return self.regimes.iloc[-1]

    def to_dict(self) -> dict:
        return {
            "current_regime": self.current_regime(),
            "regime_counts": {str(k): v for k, v in self.regime_counts.items()},
            "thresholds": self.thresholds,
        }


def detect_regimes(
    returns: pd.Series,
    vol_lookback: int = 30,
    annualization: int = 365,
    low_vol_pct: float = 25.0,
    high_vol_pct: float = 75.0,
    crisis_pct: float = 95.0,
) -> RegimeResult:
    """Detect market regimes based on rolling volatility percentiles.

    Classifies each day into one of four regimes based on where the
    current realized volatility falls in its historical distribution.

    Args:
        returns: Daily return series (e.g., BTC returns).
        vol_lookback: Rolling window for realized volatility.
        annualization: Days per year.
        low_vol_pct: Percentile threshold for low-vol regime.
        high_vol_pct: Percentile threshold for high-vol regime.
        crisis_pct: Percentile threshold for crisis regime.

    Returns:
        RegimeResult with per-day regime labels and volatility series.
    """
    # Rolling realized volatility (annualized)
    rolling_vol = returns.rolling(window=vol_lookback, min_periods=vol_lookback // 2).std()
    rolling_vol_ann = rolling_vol * np.sqrt(annualization)

    # Compute expanding percentile thresholds
    vol_clean = rolling_vol_ann.dropna()
    low_threshold = np.percentile(vol_clean, low_vol_pct)
    high_threshold = np.percentile(vol_clean, high_vol_pct)
    crisis_threshold = np.percentile(vol_clean, crisis_pct)

    # Classify
    regimes = pd.Series(MarketRegime.NORMAL, index=returns.index)

    regimes[rolling_vol_ann < low_threshold] = MarketRegime.LOW_VOL
    regimes[rolling_vol_ann > high_threshold] = MarketRegime.HIGH_VOL
    regimes[rolling_vol_ann > crisis_threshold] = MarketRegime.CRISIS

    # NaN periods default to normal
    regimes[rolling_vol_ann.isna()] = MarketRegime.NORMAL

    logger.info(f"Regime detection: {regimes.value_counts().to_dict()}")

    return RegimeResult(
        regimes=regimes,
        vol_series=rolling_vol_ann,
        thresholds={
            "low_vol": float(low_threshold),
            "high_vol": float(high_threshold),
            "crisis": float(crisis_threshold),
        },
    )


def regime_adjusted_weights(
    weights: pd.DataFrame,
    regimes: pd.Series,
    regime_scales: dict | None = None,
) -> pd.DataFrame:
    """Scale portfolio weights based on detected market regime.

    In high-volatility or crisis regimes, reduce exposure. In low-vol,
    maintain or slightly increase.

    Args:
        weights: Portfolio weights (dates × assets).
        regimes: Per-day regime labels from detect_regimes.
        regime_scales: Multiplier per regime. Defaults to:
            LOW_VOL=1.2, NORMAL=1.0, HIGH_VOL=0.5, CRISIS=0.0

    Returns:
        Scaled weights DataFrame.
    """
    if regime_scales is None:
        regime_scales = {
            MarketRegime.LOW_VOL: 1.2,
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOL: 0.5,
            MarketRegime.CRISIS: 0.0,
        }

    scale = regimes.map(regime_scales).fillna(1.0)
    return weights.multiply(scale, axis=0)


def compute_regime_performance(
    returns: pd.Series,
    regimes: pd.Series,
    annualization: int = 365,
) -> pd.DataFrame:
    """Compute strategy performance broken down by regime.

    Args:
        returns: Daily strategy returns.
        regimes: Per-day regime labels.
        annualization: Days per year.

    Returns:
        DataFrame with per-regime: mean return, volatility, Sharpe, count.
    """
    results = []

    for regime in MarketRegime:
        mask = regimes == regime
        r = returns[mask]

        if len(r) < 5:
            results.append({
                "regime": regime.value,
                "days": len(r),
                "mean_daily": 0,
                "annual_return": 0,
                "annual_vol": 0,
                "sharpe": 0,
            })
            continue

        mean_daily = r.mean()
        vol_daily = r.std()
        annual_ret = mean_daily * annualization
        annual_vol = vol_daily * np.sqrt(annualization)
        sharpe = (mean_daily / vol_daily * np.sqrt(annualization)) if vol_daily > 0 else 0

        results.append({
            "regime": regime.value,
            "days": len(r),
            "mean_daily": float(mean_daily),
            "annual_return": float(annual_ret),
            "annual_vol": float(annual_vol),
            "sharpe": float(sharpe),
        })

    return pd.DataFrame(results).set_index("regime")
