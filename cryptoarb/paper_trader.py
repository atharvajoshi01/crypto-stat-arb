"""Paper trading engine — simulate live trading without real money."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single paper trade."""

    timestamp: str
    pair: str
    action: str  # "OPEN_LONG", "OPEN_SHORT", "CLOSE"
    asset_a_qty: float
    asset_b_qty: float
    z_score: float
    spread: float


@dataclass
class PaperPortfolio:
    """Current state of the paper trading portfolio."""

    positions: Dict[str, float] = field(default_factory=dict)  # asset -> quantity
    cash: float = 100_000.0
    initial_cash: float = 100_000.0
    trades: List[Trade] = field(default_factory=list)
    daily_equity: List[dict] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def pnl(self, prices: Dict[str, float]) -> float:
        """Compute current P&L given latest prices."""
        position_value = sum(
            qty * prices.get(asset, 0)
            for asset, qty in self.positions.items()
        )
        return (self.cash + position_value) - self.initial_cash

    def equity(self, prices: Dict[str, float]) -> float:
        """Compute current equity."""
        position_value = sum(
            qty * prices.get(asset, 0)
            for asset, qty in self.positions.items()
        )
        return self.cash + position_value

    def record_equity(self, prices: Dict[str, float]) -> None:
        """Record daily equity snapshot."""
        self.daily_equity.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": self.equity(prices),
            "pnl": self.pnl(prices),
            "n_positions": sum(1 for q in self.positions.values() if abs(q) > 1e-8),
        })

    def save(self, path: str) -> None:
        """Save portfolio state to JSON."""
        data = {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "positions": self.positions,
            "n_trades": self.n_trades,
            "trades": [
                {
                    "timestamp": t.timestamp,
                    "pair": t.pair,
                    "action": t.action,
                    "z_score": t.z_score,
                }
                for t in self.trades[-50:]  # last 50 trades
            ],
            "daily_equity": self.daily_equity[-365:],  # last year
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class PaperTrader:
    """Paper trading engine for stat-arb strategies.

    Simulates live trading by:
    1. Fetching latest prices from exchange
    2. Computing signals from the strategy
    3. Executing trades on paper (no real money)
    4. Logging all decisions for audit

    Args:
        exchange_id: CCXT exchange to fetch prices from.
        pairs: List of (asset_a, asset_b, beta) tuples to trade.
        entry_z: Z-score entry threshold.
        exit_z: Z-score exit threshold.
        position_size: Dollar amount per pair leg.
        portfolio: Initial portfolio state.
    """

    def __init__(
        self,
        exchange_id: str = "kraken",
        pairs: list | None = None,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        position_size: float = 10_000.0,
        portfolio: PaperPortfolio | None = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.pairs = pairs or []
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.position_size = position_size
        self.portfolio = portfolio or PaperPortfolio()

    def fetch_latest_prices(self) -> Dict[str, float]:
        """Fetch latest prices from exchange."""
        import ccxt

        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({"enableRateLimit": True})

        prices = {}
        for asset_a, asset_b, _ in self.pairs:
            for symbol in [asset_a, asset_b]:
                if symbol not in prices:
                    try:
                        ticker = exchange.fetch_ticker(symbol)
                        prices[symbol] = ticker["last"]
                    except Exception as e:
                        logger.warning(f"Failed to fetch {symbol}: {e}")

        return prices

    def compute_signal(
        self,
        price_history: pd.DataFrame,
        asset_a: str,
        asset_b: str,
        beta: float,
        window: int = 60,
    ) -> tuple[float, float]:
        """Compute current z-score for a pair.

        Args:
            price_history: Recent log price history.
            asset_a: First asset.
            asset_b: Second asset.
            beta: Hedge ratio.
            window: Rolling window for z-score.

        Returns:
            Tuple of (z_score, spread_value).
        """
        if asset_a not in price_history.columns or asset_b not in price_history.columns:
            return 0.0, 0.0

        log_a = np.log(price_history[asset_a])
        log_b = np.log(price_history[asset_b])
        spread = log_a - beta * log_b

        if len(spread.dropna()) < window:
            return 0.0, 0.0

        mean = spread.rolling(window).mean().iloc[-1]
        std = spread.rolling(window).std().iloc[-1]

        if std == 0 or np.isnan(std):
            return 0.0, spread.iloc[-1]

        z = (spread.iloc[-1] - mean) / std
        return float(z), float(spread.iloc[-1])

    def execute_signals(
        self,
        prices: Dict[str, float],
        signals: Dict[str, tuple],
    ) -> List[Trade]:
        """Execute paper trades based on signals.

        Args:
            prices: Current prices {symbol: price}.
            signals: {pair_name: (z_score, spread)} for each pair.

        Returns:
            List of trades executed.
        """
        trades = []
        now = datetime.now(timezone.utc).isoformat()

        for (asset_a, asset_b, beta), (z, spread) in zip(self.pairs, signals.values()):
            pair_name = f"{asset_a}/{asset_b}"
            current_pos_a = self.portfolio.positions.get(asset_a, 0)
            is_flat = abs(current_pos_a) < 1e-8

            price_a = prices.get(asset_a, 0)
            price_b = prices.get(asset_b, 0)
            if price_a == 0 or price_b == 0:
                continue

            qty_a = self.position_size / price_a
            qty_b = (self.position_size * abs(beta)) / price_b

            if is_flat:
                # Entry
                if z > self.entry_z:
                    # Short spread: sell A, buy B
                    self.portfolio.positions[asset_a] = self.portfolio.positions.get(asset_a, 0) - qty_a
                    self.portfolio.positions[asset_b] = self.portfolio.positions.get(asset_b, 0) + qty_b
                    self.portfolio.cash += qty_a * price_a - qty_b * price_b
                    trade = Trade(now, pair_name, "OPEN_SHORT", -qty_a, qty_b, z, spread)
                    trades.append(trade)

                elif z < -self.entry_z:
                    # Long spread: buy A, sell B
                    self.portfolio.positions[asset_a] = self.portfolio.positions.get(asset_a, 0) + qty_a
                    self.portfolio.positions[asset_b] = self.portfolio.positions.get(asset_b, 0) - qty_b
                    self.portfolio.cash -= qty_a * price_a - qty_b * price_b
                    trade = Trade(now, pair_name, "OPEN_LONG", qty_a, -qty_b, z, spread)
                    trades.append(trade)

            else:
                # Exit
                if abs(z) < self.exit_z:
                    # Close both legs
                    pos_a = self.portfolio.positions.get(asset_a, 0)
                    pos_b = self.portfolio.positions.get(asset_b, 0)
                    self.portfolio.cash += pos_a * price_a + pos_b * price_b
                    self.portfolio.positions[asset_a] = 0
                    self.portfolio.positions[asset_b] = 0
                    trade = Trade(now, pair_name, "CLOSE", -pos_a, -pos_b, z, spread)
                    trades.append(trade)

        self.portfolio.trades.extend(trades)
        return trades

    def run_once(self, price_history: pd.DataFrame) -> List[Trade]:
        """Run one iteration of the paper trading loop.

        1. Fetch latest prices
        2. Compute signals for all pairs
        3. Execute trades
        4. Record equity

        Args:
            price_history: Recent price history DataFrame.

        Returns:
            List of trades executed in this iteration.
        """
        prices = self.fetch_latest_prices()

        signals = {}
        for asset_a, asset_b, beta in self.pairs:
            pair_name = f"{asset_a}/{asset_b}"
            z, spread = self.compute_signal(price_history, asset_a, asset_b, beta)
            signals[pair_name] = (z, spread)

        trades = self.execute_signals(prices, signals)
        self.portfolio.record_equity(prices)

        for t in trades:
            logger.info(f"  TRADE: {t.action} {t.pair} (z={t.z_score:.2f})")

        return trades
