# multifactor_ng Backtest Report

**Description**: Multi-factor strategy (mom + low vol + low turnover), monthly rebalance, industry-neutral, engine rewrite  
**Period**: 2019-01-01 ~ 2026-02-28  
**Frequency**: monthly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 90%  
**Selection**: top 10%, max 0/industry, equal weight  
**Avg holdings**: 357 stocks / 31 industries  
**Cost**: 0.0150% single-side  

## Factors

| Factor | Weight | Direction |
|--------|--------|-----------|
| mom_12_1 | +0.33 | positive |
| rvol_20 | -0.33 | negative |
| turnover | -0.33 | negative |

## Performance Summary

| Metric | Strategy | Shanghai Composite | CSI 500 |
|--------|----------|----------|----------|
| Ann. Return | +11.01% | +5.15% | +8.82% |
| Ann. Volatility | 20.19% | 14.58% | 27.36% |
| Sharpe Ratio | 0.55 | 0.35 | 0.32 |
| Max Drawdown | -32.44% | -23.39% | -37.16% |
| Cum. Return | +107.70% | +41.55% | +79.44% |
| Win Rate | 62.35% | — | — |
| Excess vs Shanghai Composite | +3.95% | — | — |
| IR vs Shanghai Composite | 0.37 | — | — |
| Excess vs CSI 500 | -4.07% | — | — |
| IR vs CSI 500 | -0.18 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2019 | +21.3% | +3.7% | +17.6% | +5.2% | +16.1% | 281/30 |
| 2020 | +19.6% | +13.9% | +5.7% | -4.0% | +23.6% | 301/31 |
| 2021 | +23.4% | +4.8% | +18.6% | +4.9% | +18.5% | 315/31 |
| 2022 | -13.3% | -15.1% | +1.8% | -4.0% | -9.4% | 339/31 |
| 2023 | -4.9% | -3.7% | -1.2% | -0.7% | -4.2% | 374/31 |
| 2024 | +1.4% | +12.7% | -11.3% | -0.1% | +1.5% | 430/31 |
| 2025 | +23.4% | +18.4% | +5.0% | +82.5% | -59.2% | 441/31 |
| 2026 | +12.5% | +4.9% | +7.6% | -2.5% | +15.1% | 444/30 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 48.2%
- **Total cost drag**: 1.23%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +11.20% | +11.01% | 0.19% |
| Sharpe | 0.56 | 0.55 | 0.01 |
| Cum. Return | +110.25% | +107.70% | 2.56% |

