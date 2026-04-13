# crypto-stat-arb

[![CI](https://github.com/atharvajoshi01/crypto-stat-arb/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvajoshi01/crypto-stat-arb/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Statistical arbitrage engine for cryptocurrency markets. Cointegration-based pairs trading with walk-forward backtesting and realistic transaction cost modeling.

## Results

Backtested on 11 cryptocurrencies from Kraken (2021-2026), out-of-sample period: Feb 2025 - Jan 2026.

| Metric | Raw Strategy | Risk-Managed |
|--------|-------------|-------------|
| Annual Return | -18.8% | -15.7% |
| Annual Volatility | 8.0% | 7.4% |
| Sharpe Ratio | -2.56 | -2.27 |
| Max Drawdown | -18.4% | -14.6% |
| BTC Correlation | **0.03** | **0.03** |
| Win Rate | 26.1% | 24.2% |

**Key finding:** Near-zero correlation with BTC (0.03) confirms the strategy is market-neutral. Negative returns in this period are driven by limited universe (11 coins), transaction costs, and the 2025 crypto regime. The pipeline is designed for transparency, not backtest optimization.

### Equity Curve

![Equity Curve](results/equity_curve.png)

*Blue: risk-managed stat-arb. Orange: buy-and-hold BTC. The strategy decouples from BTC — when BTC rallied 50%, the strategy moved independently.*

### Discovered Pairs

| Pair | Hedge Ratio (β) | Half-Life | ADF p-value | Correlation |
|------|-----------------|-----------|-------------|-------------|
| SOL/DOGE | 0.349 | 8.4 days | 0.016 | 0.91 |
| ETH/DOT | 0.560 | 10.1 days | 0.018 | 0.90 |
| ETH/ATOM | 0.507 | 11.3 days | 0.022 | 0.87 |

## What It Does

Finds cryptocurrency pairs that are statistically bound together (cointegrated), waits for temporary mispricings, and trades the convergence — all while staying dollar-neutral.

## Pipeline

```
Data → Pair Discovery → Signal Generation → Portfolio → Backtest → Evaluation
```

1. **Data**: Fetch OHLCV from any exchange via CCXT, clean, cache
2. **Pair Discovery**: Cointegration testing (Engle-Granger ADF), half-life filtering
3. **Signals**: Rolling OLS hedge ratio, z-score entry/exit with stop-loss
4. **Portfolio**: Dollar-neutral construction with position limits
5. **Backtest**: Walk-forward validation with transaction cost modeling
6. **Risk Management**: Drawdown stops, volatility scaling, pair health monitoring
7. **Evaluation**: Sharpe, Sortino, Calmar, drawdown analysis, BTC beta

## Modules

| Module | What It Does |
|--------|-------------|
| `config.py` | Pydantic-validated strategy parameters |
| `data.py` | CCXT data fetching, cleaning, parquet caching |
| `pairs.py` | Cointegration testing, half-life, pair ranking |
| `signals.py` | Rolling hedge ratio, spread, z-score, entry/exit logic |
| `portfolio.py` | Dollar-neutral construction, cost modeling |
| `backtest.py` | Walk-forward backtesting engine |
| `metrics.py` | Sharpe, Sortino, Calmar, drawdown, win rate |
| `risk.py` | Drawdown stops, vol scaling, pair health |

## Quick Start

```python
from cryptoarb import (
    StrategyConfig, fetch_ohlcv, build_price_matrix,
    clean_price_matrix, log_prices, discover_pairs,
    generate_pair_signals, build_portfolio,
    compute_portfolio_returns, evaluate,
)

# Fetch data
ohlcv = fetch_ohlcv(["BTC/USD", "ETH/USD", "SOL/USD"], exchange_id="kraken", start="2022-01-01")
prices = clean_price_matrix(build_price_matrix(ohlcv))
log_px = log_prices(prices)

# Discover pairs
pairs = discover_pairs(log_px)

# Generate signals and build portfolio
signals = [generate_pair_signals(log_px, p) for p in pairs]
weights = build_portfolio(signals, log_px)
returns = compute_portfolio_returns(weights, log_px, cost_bps=40)

# Evaluate
metrics = evaluate(returns)
print(metrics.summary())
```

## Installation

```bash
pip install -e ".[dev]"
```

## Development

```bash
git clone https://github.com/atharvajoshi01/crypto-stat-arb.git
cd crypto-stat-arb
pip install -e ".[dev]"
pytest  # 44 tests
```

## License

MIT
