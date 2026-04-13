"""Factor attribution — decompose returns into alpha and factor exposures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Factor attribution result."""

    alpha: float           # annualized alpha (intercept)
    alpha_tstat: float     # t-statistic of alpha
    alpha_pvalue: float    # p-value of alpha
    beta: Dict[str, float]  # factor betas
    r_squared: float       # R² of the regression
    residual_vol: float    # annualized residual volatility
    annualization: int

    @property
    def alpha_significant(self) -> bool:
        """Alpha is statistically significant at 95% confidence."""
        return abs(self.alpha_tstat) > 1.96

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "FACTOR ATTRIBUTION",
            "=" * 50,
            f"Alpha (ann.):     {self.alpha:.2%}  (t={self.alpha_tstat:.2f}, p={self.alpha_pvalue:.4f})",
            f"Alpha significant: {'YES' if self.alpha_significant else 'NO'} (95% CI)",
            f"R²:               {self.r_squared:.4f}",
            f"Residual Vol:     {self.residual_vol:.2%}",
            "",
            "Factor Betas:",
        ]
        for name, b in self.beta.items():
            lines.append(f"  {name}: {b:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "alpha_annualized": f"{self.alpha:.4%}",
            "alpha_tstat": f"{self.alpha_tstat:.2f}",
            "alpha_pvalue": f"{self.alpha_pvalue:.4f}",
            "alpha_significant": self.alpha_significant,
            "betas": {k: f"{v:.4f}" for k, v in self.beta.items()},
            "r_squared": f"{self.r_squared:.4f}",
            "residual_vol": f"{self.residual_vol:.4%}",
        }


def factor_attribution(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    annualization: int = 365,
) -> AttributionResult:
    """Regress strategy returns against factor returns.

    Runs OLS: R_strategy = alpha + beta_1 * F_1 + beta_2 * F_2 + ... + eps

    The intercept (alpha) represents the strategy's return not explained
    by the factors. A significant positive alpha means genuine skill.

    Args:
        strategy_returns: Daily strategy returns.
        factor_returns: DataFrame of daily factor returns (e.g., BTC, ETH).
        annualization: Days per year for annualizing alpha.

    Returns:
        AttributionResult with alpha, betas, and significance tests.
    """
    # Align indexes
    combined = pd.concat([strategy_returns.rename("strategy"), factor_returns], axis=1).dropna()

    if len(combined) < 30:
        return AttributionResult(
            alpha=0, alpha_tstat=0, alpha_pvalue=1, beta={},
            r_squared=0, residual_vol=0, annualization=annualization,
        )

    y = combined["strategy"].values
    X = combined.drop(columns=["strategy"]).values
    X_const = add_constant(X)

    model = OLS(y, X_const).fit()

    # Extract results
    alpha_daily = model.params[0]
    alpha_annualized = alpha_daily * annualization
    alpha_tstat = float(model.tvalues[0])
    alpha_pvalue = float(model.pvalues[0])

    betas = {}
    factor_names = combined.drop(columns=["strategy"]).columns.tolist()
    for i, name in enumerate(factor_names):
        betas[name] = float(model.params[i + 1])

    residual_vol = float(model.resid.std() * np.sqrt(annualization))

    result = AttributionResult(
        alpha=alpha_annualized,
        alpha_tstat=alpha_tstat,
        alpha_pvalue=alpha_pvalue,
        beta=betas,
        r_squared=float(model.rsquared),
        residual_vol=residual_vol,
        annualization=annualization,
    )

    logger.info(f"Alpha: {alpha_annualized:.2%} (t={alpha_tstat:.2f})")
    for name, b in betas.items():
        logger.info(f"  {name} beta: {b:.4f}")

    return result


def compute_factor_returns(
    price_matrix: pd.DataFrame,
    benchmark_columns: list,
) -> pd.DataFrame:
    """Compute daily returns for benchmark/factor assets.

    Args:
        price_matrix: Raw price matrix (not log).
        benchmark_columns: Column names to use as factors.

    Returns:
        DataFrame of daily returns for the specified factors.
    """
    available = [c for c in benchmark_columns if c in price_matrix.columns]
    if not available:
        return pd.DataFrame(index=price_matrix.index)

    return price_matrix[available].pct_change().dropna()
