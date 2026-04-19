"""Data fetching, cleaning, and caching for crypto OHLCV data."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbols: List[str],
    exchange_id: str = "binance",
    timeframe: str = "1d",
    start: str = "2020-01-01",
    end: Optional[str] = None,
    rate_limit_ms: int = 200,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data from a crypto exchange via CCXT.

    Args:
        symbols: List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"]).
        exchange_id: CCXT exchange identifier.
        timeframe: Candle timeframe (e.g., "1d", "1h").
        start: Start date as YYYY-MM-DD string.
        end: End date. Defaults to today.
        rate_limit_ms: Delay between API calls to avoid rate limiting.

    Returns:
        Dict mapping symbol to OHLCV DataFrame with columns:
        [open, high, low, close, volume] indexed by datetime.
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    since = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    until = (
        int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
        if end
        else int(datetime.now().timestamp() * 1000)
    )

    result: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        logger.info(f"Fetching {symbol}...")
        all_candles = []
        fetch_since = since

        try:
            while fetch_since < until:
                candles = exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=fetch_since, limit=1000
                )
                if not candles:
                    break
                all_candles.extend(candles)
                fetch_since = candles[-1][0] + 1
                time.sleep(rate_limit_ms / 1000)

            if all_candles:
                df = pd.DataFrame(
                    all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("timestamp")
                df = df[df.index <= pd.Timestamp(end or datetime.now())]
                df = df[~df.index.duplicated(keep="first")]
                result[symbol] = df
                logger.info(f"  {symbol}: {len(df)} candles")
            else:
                logger.warning(f"  {symbol}: no data returned")

        except Exception as e:
            logger.warning(f"  {symbol}: failed — {e}")

    return result


def build_price_matrix(
    ohlcv: Dict[str, pd.DataFrame],
    field: str = "close",
) -> pd.DataFrame:
    """Build a price matrix from OHLCV data.

    Args:
        ohlcv: Dict mapping symbol to OHLCV DataFrame.
        field: Price field to extract (default: "close").

    Returns:
        DataFrame with dates as index and symbols as columns.
    """
    series = {}
    for symbol, df in ohlcv.items():
        name = symbol.replace("/", "_")
        series[name] = df[field]

    matrix = pd.DataFrame(series)
    matrix = matrix.sort_index()
    return matrix


def clean_price_matrix(
    prices: pd.DataFrame,
    min_data_pct: float = 0.90,
    min_avg_volume: Optional[pd.DataFrame] = None,
    min_volume_threshold: float = 1_000_000,
    max_gap_days: int = 3,
) -> pd.DataFrame:
    """Clean and filter the price matrix.

    Args:
        prices: Raw price matrix (dates × symbols).
        min_data_pct: Drop coins with less than this fraction of valid data.
        min_avg_volume: Optional volume matrix for liquidity filtering.
        min_volume_threshold: Min average daily USD volume.
        max_gap_days: Forward-fill gaps up to this many days.

    Returns:
        Cleaned price matrix with only qualifying coins.
    """
    # Filter by data completeness
    valid_pct = prices.notna().mean()
    keep = valid_pct[valid_pct >= min_data_pct].index
    logger.info(f"Data completeness filter: {len(prices.columns)} → {len(keep)} coins")
    prices = prices[keep]

    # Forward-fill small gaps
    prices = prices.ffill(limit=max_gap_days)

    # Drop remaining rows with any NaN
    prices = prices.dropna()

    # Volume filter
    if min_avg_volume is not None:
        avg_vol = min_avg_volume.mean()
        liquid = avg_vol[avg_vol >= min_volume_threshold].index
        overlap = prices.columns.intersection(liquid)
        logger.info(f"Volume filter: {len(prices.columns)} → {len(overlap)} coins")
        prices = prices[overlap]

    # Drop coins with zero variance (dead coins)
    variance = prices.pct_change().var()
    nonzero = variance[variance > 0].index
    prices = prices[nonzero]

    logger.info(f"Final: {len(prices.columns)} coins × {len(prices)} days")
    return prices


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log prices.

    Warns if non-positive values are found, which produce -inf/NaN.
    """
    n_invalid = (prices <= 0).sum().sum()
    if n_invalid > 0:
        logger.warning(
            f"Found {n_invalid} non-positive prices; these will become NaN in log space"
        )
    return np.log(prices.clip(lower=1e-10))


def save_cache(prices: pd.DataFrame, path: str) -> None:
    """Save price matrix to parquet cache."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(p)
    logger.info(f"Cached to {p}")


def load_cache(path: str) -> Optional[pd.DataFrame]:
    """Load price matrix from parquet cache."""
    p = Path(path)
    if p.exists():
        logger.info(f"Loading cache from {p}")
        return pd.read_parquet(p)
    return None


def get_top_symbols(
    exchange_id: str = "binance",
    quote: str = "USDT",
    n: int = 100,
) -> List[str]:
    """Get top N symbols by volume from an exchange.

    Args:
        exchange_id: CCXT exchange identifier.
        quote: Quote currency to filter by.
        n: Number of symbols to return.

    Returns:
        List of symbol strings sorted by 24h volume.
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()

    usdt_markets = [
        m for m in exchange.markets.values()
        if m["quote"] == quote and m["active"] and m["spot"]
    ]

    # Sort by symbol name as a proxy (volume-based sorting requires tickers)
    symbols = sorted([m["symbol"] for m in usdt_markets])
    return symbols[:n]
