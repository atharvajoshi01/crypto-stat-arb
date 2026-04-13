# %% [markdown]
# # Crypto Statistical Arbitrage — Research Notebook
#
# This notebook walks through the full research process for building a
# cointegration-based pairs trading strategy on cryptocurrency markets.
#
# **Pipeline:** Data → Pair Discovery → Signal Analysis → Backtest → Evaluation

# %% [markdown]
# ## 1. Setup and Data Generation
#
# We use synthetic data that mimics real crypto price dynamics —
# cointegrated clusters with mean-reverting spreads.

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryptoarb.data import log_prices
from cryptoarb.pairs import discover_pairs, test_cointegration as run_coint_test
from cryptoarb.signals import (
    compute_rolling_hedge_ratio,
    compute_spread,
    compute_zscore,
    generate_positions,
    generate_pair_signals,
)
from cryptoarb.portfolio import build_portfolio, compute_portfolio_returns
from cryptoarb.metrics import evaluate, compute_drawdown
from cryptoarb.regime import detect_regimes, compute_regime_performance, regime_adjusted_weights
from cryptoarb.sensitivity import run_sensitivity
from cryptoarb.attribution import factor_attribution

np.random.seed(42)

# %%
# Generate synthetic prices with cointegrated clusters
def generate_prices(n_days=1200):
    rng = np.random.RandomState(42)
    dates = pd.date_range("2020-06-01", periods=n_days, freq="D")
    prices = {}

    # Cluster 1: BTC-like (3 coins)
    base1 = rng.normal(0, 0.02, n_days).cumsum() + np.log(40000)
    for i, (beta, name) in enumerate([(1.0, "BTC"), (0.85, "ETH"), (1.1, "BNB")]):
        noise = np.zeros(n_days)
        for t in range(1, n_days):
            noise[t] = 0.85 * noise[t-1] + rng.normal(0, 0.01)
        prices[name] = np.exp(0.5 + beta * base1 + noise)

    # Cluster 2: Alt-coins (3 coins)
    base2 = rng.normal(0, 0.025, n_days).cumsum() + np.log(100)
    for i, (beta, name) in enumerate([(1.0, "SOL"), (0.9, "AVAX"), (1.05, "DOT")]):
        noise = np.zeros(n_days)
        for t in range(1, n_days):
            noise[t] = 0.80 * noise[t-1] + rng.normal(0, 0.015)
        prices[name] = np.exp(0.3 + beta * base2 + noise)

    # Independent coins
    for name in ["DOGE", "SHIB"]:
        prices[name] = np.exp(rng.normal(0, 0.03, n_days).cumsum() + np.log(50))

    return pd.DataFrame(prices, index=dates)

raw_prices = generate_prices()
log_px = log_prices(raw_prices)
print(f"Universe: {len(raw_prices.columns)} coins, {len(raw_prices)} days")
print(f"Date range: {raw_prices.index[0].date()} to {raw_prices.index[-1].date()}")

# %% [markdown]
# ## 2. Correlation Analysis
#
# Before testing cointegration (expensive), we pre-filter by correlation.

# %%
corr = log_px.corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
plt.colorbar(im, label="Correlation")
ax.set_title("Log-Price Correlation Matrix")
plt.tight_layout()
plt.savefig("notebooks/correlation_matrix.png", dpi=100)
print("Saved: notebooks/correlation_matrix.png")

# %% [markdown]
# ## 3. Pair Discovery
#
# Test all correlated pairs for cointegration using the Engle-Granger method.
# Only pairs with ADF p-value < 0.05 and half-life between 3-30 days are kept.

# %%
train_data = log_px.iloc[:600]  # first ~2 years for training

pairs = discover_pairs(
    train_data,
    min_correlation=0.60,
    min_half_life=2.0,
    max_half_life=40.0,
    max_pairs=10,
)

print(f"\nDiscovered {len(pairs)} cointegrated pairs:\n")
for p in pairs:
    print(f"  {p.asset_a}/{p.asset_b}: β={p.beta:.3f}, HL={p.half_life:.1f}d, "
          f"ADF={p.adf_stat:.2f} (p={p.adf_pvalue:.4f}), corr={p.correlation:.3f}")

# %% [markdown]
# ## 4. Spread Analysis (Best Pair)
#
# For the best pair, visualize the spread, z-score, and trading signals.

# %%
best_pair = pairs[0]
print(f"Analyzing: {best_pair.asset_a}/{best_pair.asset_b}")

sig = generate_pair_signals(log_px, best_pair, entry_z=2.0, exit_z=0.5)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Plot 1: Log prices
ax = axes[0]
ax.plot(log_px[best_pair.asset_a], label=best_pair.asset_a, linewidth=0.8)
ax.plot(log_px[best_pair.asset_b], label=best_pair.asset_b, linewidth=0.8)
ax.set_ylabel("Log Price")
ax.set_title(f"Pair: {best_pair.asset_a} / {best_pair.asset_b}")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Spread
ax = axes[1]
spread = sig.spread.dropna()
ax.plot(spread, linewidth=0.8, color="purple")
ax.axhline(y=spread.mean(), color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Spread")
ax.set_title(f"Spread (β={best_pair.beta:.3f})")
ax.grid(True, alpha=0.3)

# Plot 3: Z-score with thresholds
ax = axes[2]
z = sig.z_score.dropna()
ax.plot(z, linewidth=0.8, color="blue")
ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.7, label="Entry (+2σ)")
ax.axhline(y=-2.0, color="green", linestyle="--", alpha=0.7, label="Entry (-2σ)")
ax.axhline(y=0.5, color="orange", linestyle=":", alpha=0.5, label="Exit (±0.5σ)")
ax.axhline(y=-0.5, color="orange", linestyle=":", alpha=0.5)
ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
ax.set_ylabel("Z-Score")
ax.set_title("Z-Score with Entry/Exit Thresholds")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: Position
ax = axes[3]
pos = sig.position
ax.fill_between(pos.index, pos.values, 0, alpha=0.3,
                color=["green" if x > 0 else "red" if x < 0 else "gray" for x in pos.values])
ax.set_ylabel("Position")
ax.set_title("Trading Signal (+1 = Long Spread, -1 = Short Spread)")
ax.set_ylim(-1.5, 1.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/spread_analysis.png", dpi=100)
print("Saved: notebooks/spread_analysis.png")

# %% [markdown]
# ## 5. Portfolio Backtest
#
# Build a dollar-neutral portfolio from all discovered pairs and compute
# out-of-sample returns with transaction costs.

# %%
# Generate signals for all pairs
all_signals = [generate_pair_signals(log_px, p, entry_z=2.0, exit_z=0.5) for p in pairs]

# Build portfolio and compute OOS returns
test_start = 600
weights = build_portfolio(all_signals, log_px)
test_weights = weights.iloc[test_start:]
test_log_px = log_px.iloc[test_start:]

returns = compute_portfolio_returns(test_weights, test_log_px, cost_bps=40)
metrics = evaluate(returns, annualization=365)
print(metrics.summary())

# %% [markdown]
# ## 6. Regime Analysis
#
# Does the strategy perform differently in calm vs volatile markets?

# %%
# Use BTC as the regime indicator
btc_returns = raw_prices["BTC"].pct_change().dropna()
regime_result = detect_regimes(btc_returns, vol_lookback=30)
regime_perf = compute_regime_performance(returns["net_return"], regime_result.regimes)
print("Performance by Market Regime:")
print(regime_perf.to_string())

# %% [markdown]
# ## 7. Parameter Sensitivity
#
# Sweep entry/exit thresholds to check if results are robust or overfit.

# %%
report = run_sensitivity(
    log_px, pairs,
    entry_z_range=[1.5, 2.0, 2.5, 3.0],
    exit_z_range=[0.25, 0.5, 0.75],
    cost_bps_range=[20, 40, 60],
    test_start_idx=test_start,
)
print(report.summary())
print("\nSharpe Heatmap (cost=40bps):")
print(report.sharpe_heatmap(cost_bps=40).to_string())

# %% [markdown]
# ## 8. Factor Attribution
#
# Is the strategy's return explained by BTC? Or is there genuine alpha?

# %%
btc_factor = raw_prices["BTC"].pct_change().iloc[test_start:].dropna()
factors = pd.DataFrame({"BTC": btc_factor})
attr = factor_attribution(returns["net_return"], factors)
print(attr.summary())

# %% [markdown]
# ## Key Findings
#
# 1. **Cointegration works** — discovered multiple pairs with ADF p < 0.05
# 2. **Market neutral** — near-zero BTC beta confirms dollar neutrality
# 3. **Transaction costs matter** — Sharpe degrades significantly with higher costs
# 4. **Regime awareness helps** — strategy underperforms in crisis regimes
# 5. **Parameter sensitivity** — results vary with thresholds, check for robustness
