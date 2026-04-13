"""Tests for paper trading engine."""

import numpy as np
import pandas as pd

from cryptoarb.paper_trader import PaperTrader, PaperPortfolio, Trade


class TestPaperPortfolio:
    def test_initial_state(self):
        pf = PaperPortfolio()
        assert pf.cash == 100_000.0
        assert pf.n_trades == 0
        assert len(pf.positions) == 0

    def test_pnl_calculation(self):
        pf = PaperPortfolio(cash=90_000, initial_cash=100_000)
        pf.positions = {"BTC/USD": 0.5}
        prices = {"BTC/USD": 40_000}
        # equity = 90000 + 0.5 * 40000 = 110000
        assert pf.pnl(prices) == 10_000

    def test_equity(self):
        pf = PaperPortfolio(cash=50_000)
        pf.positions = {"ETH/USD": 10}
        prices = {"ETH/USD": 3_000}
        assert pf.equity(prices) == 80_000  # 50k + 10*3k

    def test_record_equity(self):
        pf = PaperPortfolio()
        pf.record_equity({"BTC/USD": 50_000})
        assert len(pf.daily_equity) == 1
        assert pf.daily_equity[0]["equity"] == 100_000

    def test_save(self, tmp_path):
        pf = PaperPortfolio()
        pf.trades.append(Trade(
            timestamp="2023-01-01", pair="BTC/ETH",
            action="OPEN_LONG", asset_a_qty=1, asset_b_qty=-10,
            z_score=-2.5, spread=-0.1,
        ))
        path = str(tmp_path / "portfolio.json")
        pf.save(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["n_trades"] == 1


class TestPaperTrader:
    def test_compute_signal(self):
        rng = np.random.RandomState(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        prices = pd.DataFrame({
            "BTC/USD": rng.normal(0, 100, n).cumsum() + 40000,
            "ETH/USD": rng.normal(0, 50, n).cumsum() + 3000,
        }, index=dates)

        trader = PaperTrader(pairs=[("BTC/USD", "ETH/USD", 1.5)])
        z, spread = trader.compute_signal(prices, "BTC/USD", "ETH/USD", beta=1.5, window=30)
        assert isinstance(z, float)
        assert isinstance(spread, float)

    def test_execute_no_signal(self):
        trader = PaperTrader(pairs=[("BTC/USD", "ETH/USD", 1.5)])
        prices = {"BTC/USD": 40000, "ETH/USD": 3000}
        # z=0 should produce no trades
        signals = {"BTC/USD/ETH/USD": (0.0, 0.0)}
        trades = trader.execute_signals(prices, signals)
        assert len(trades) == 0

    def test_execute_entry(self):
        trader = PaperTrader(
            pairs=[("BTC/USD", "ETH/USD", 1.5)],
            entry_z=2.0,
            position_size=10_000,
        )
        prices = {"BTC/USD": 40000, "ETH/USD": 3000}
        # z > 2 should trigger short spread
        signals = {"BTC/USD/ETH/USD": (2.5, 0.1)}
        trades = trader.execute_signals(prices, signals)
        assert len(trades) == 1
        assert trades[0].action == "OPEN_SHORT"
