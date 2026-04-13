# crypto-stat-arb

[![CI](https://github.com/atharvajoshi01/crypto-stat-arb/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvajoshi01/crypto-stat-arb/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Statistical arbitrage engine for cryptocurrency markets. Cointegration-based pairs trading with walk-forward backtesting and realistic transaction cost modeling.

## What It Does

Finds cryptocurrency pairs that are statistically bound together (cointegrated), waits for temporary mispricings, and trades the convergence — all while staying dollar-neutral (market direction doesn't matter).

## Pipeline

```
Data → Pair Discovery → Signal Generation → Portfolio → Backtest → Evaluation
```

1. **Data**: Fetch OHLCV from any exchange via CCXT, clean, cache
2. **Pair Discovery**: Cointegration testing (ADF), half-life filtering
3. **Signals**: Rolling hedge ratio, z-score entry/exit
4. **Portfolio**: Dollar-neutral construction, position limits
5. **Backtest**: Walk-forward validation, transaction cost modeling
6. **Evaluation**: Sharpe, drawdown, alpha, factor attribution

## Installation

```bash
pip install -e ".[dev]"
```

## Development

```bash
git clone https://github.com/atharvajoshi01/crypto-stat-arb.git
cd crypto-stat-arb
pip install -e ".[dev]"
pytest
```

## License

MIT
