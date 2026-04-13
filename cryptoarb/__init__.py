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

__all__ = [
    "StrategyConfig",
    "fetch_ohlcv",
    "build_price_matrix",
    "clean_price_matrix",
    "log_prices",
    "get_top_symbols",
]
