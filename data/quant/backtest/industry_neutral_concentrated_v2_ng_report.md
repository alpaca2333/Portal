# industry_neutral_concentrated_v2_ng Backtest Report

**Description**: Industry-neutral concentrated v2 (biweekly), ROE risk filter + 0.3σ buffer band, engine rewrite  
**Period**: 2019-01-01 ~ 2026-02-28  
**Frequency**: biweekly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 70%  
**Selection**: top 5%, max 5/industry, equal weight  
**Avg holdings**: 101 stocks / 28 industries  
**Buffer band**: +0.3σ for incumbents  
**Cost**: 0.0150% single-side  

## Factors

| Factor | Weight | Direction |
|--------|--------|-----------|
| mom_12_1 | +0.25 | positive |
| inv_pb | +0.25 | positive |
| vol_confirm | +0.15 | positive |
| rvol_20 | -0.15 | negative |
| log_cap | -0.10 | negative |
| rev_10 | +0.10 | positive |

## Performance Summary

| Metric | Strategy | Shanghai Composite | CSI 500 |
|--------|----------|----------|----------|
| Ann. Return | +17.86% | +7.01% | +11.38% |
| Ann. Volatility | 21.91% | 16.90% | 30.33% |
| Sharpe Ratio | 0.82 | 0.41 | 0.38 |
| Max Drawdown | -21.58% | -26.04% | -39.93% |
| Cum. Return | +219.79% | +61.07% | +113.40% |
| Win Rate | 57.89% | — | — |
| Excess vs Shanghai Composite | +11.10% | — | — |
| IR vs Shanghai Composite | 0.89 | — | — |
| Excess vs CSI 500 | +1.13% | — | — |
| IR vs CSI 500 | 0.05 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2019 | +17.1% | +18.0% | -0.9% | +25.1% | -8.0% | 93/28 |
| 2020 | +14.1% | +13.9% | +0.3% | -4.0% | +18.1% | 97/28 |
| 2021 | +32.3% | +4.8% | +27.5% | +4.9% | +27.4% | 92/27 |
| 2022 | +9.8% | -15.1% | +24.9% | -4.0% | +13.8% | 98/27 |
| 2023 | +12.2% | -3.7% | +15.9% | -0.7% | +12.9% | 105/28 |
| 2024 | +11.2% | +12.7% | -1.5% | -0.1% | +11.3% | 108/29 |
| 2025 | +18.8% | +18.4% | +0.4% | +82.5% | -63.8% | 113/30 |
| 2026 | +11.2% | +4.9% | +6.3% | -2.5% | +13.7% | 106/30 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 16.9%
- **Total cost drag**: 0.87%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +18.00% | +17.86% | 0.14% |
| Sharpe | 0.82 | 0.82 | 0.01 |
| Cum. Return | +222.56% | +219.79% | 2.76% |


## Strategy Upgrades (vs original)

### 1. ROE Risk Filter
- **Rule**: Exclude stocks with ROE_TTM < -20%
- **Purpose**: Remove extreme loss-making companies (near-bankruptcy, fraud suspects)
- **Design**: Pure risk screen, not a scoring factor (avoids ROE/PB hedge effect)

### 2. Rebalance Buffer Band
- **Rule**: Incumbent holdings get +0.3σ score bonus
- **Purpose**: Reduce marginal turnover for borderline stocks
- **Effect**: Lower turnover → lower costs, better for live trading

