"""crypto-stat-arb: Statistical arbitrage engine for cryptocurrency markets."""

__version__ = "0.1.0"

from cryptoarb.config import StrategyConfig
from cryptoarb.data import (
    fetch_ohlcv,
    build_price_matrix,
    clean_price_matrix,
    log_prices,
    get_top_symbols,
)
from cryptoarb.pairs import discover_pairs, test_cointegration, PairResult
from cryptoarb.signals import generate_pair_signals, PairSignals
from cryptoarb.portfolio import build_portfolio, compute_portfolio_returns
from cryptoarb.backtest import run_backtest, BacktestResult
from cryptoarb.metrics import evaluate, PerformanceMetrics
from cryptoarb.risk import apply_drawdown_stop, apply_volatility_scaling
from cryptoarb.basket import johansen_test, discover_baskets, BasketResult
from cryptoarb.kalman import kalman_hedge_ratio, kalman_zscore
from cryptoarb.regime import detect_regimes, regime_adjusted_weights, MarketRegime
from cryptoarb.sensitivity import run_sensitivity, SensitivityReport
from cryptoarb.attribution import factor_attribution, AttributionResult

__all__ = [
    "StrategyConfig",
    "fetch_ohlcv",
    "build_price_matrix",
    "clean_price_matrix",
    "log_prices",
    "get_top_symbols",
    "discover_pairs",
    "test_cointegration",
    "PairResult",
    "generate_pair_signals",
    "PairSignals",
    "build_portfolio",
    "compute_portfolio_returns",
    "run_backtest",
    "BacktestResult",
    "evaluate",
    "PerformanceMetrics",
    "apply_drawdown_stop",
    "apply_volatility_scaling",
    "johansen_test",
    "discover_baskets",
    "BasketResult",
    "kalman_hedge_ratio",
    "kalman_zscore",
    "detect_regimes",
    "regime_adjusted_weights",
    "MarketRegime",
    "run_sensitivity",
    "SensitivityReport",
    "factor_attribution",
    "AttributionResult",
]
