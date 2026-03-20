# ic_reweighted_2015 Backtest Report

**Description**: IC引导因子重新定权，2015年起完整市场周期回测  
**Period**: 2015-01-01 ~ 2026-02-28  
**Frequency**: biweekly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 70%  
**Selection**: top 5%, max 5/industry, equal weight  
**Avg holdings**: 81 stocks / 26 industries  
**Cost**: 0.0150% single-side  

## Factors

| Factor | Weight | Direction |
|--------|--------|-----------|
| inv_pb | +0.25 | positive |
| rev_10 | +0.30 | positive |
| rvol_20 | -0.35 | negative |
| log_cap | -0.10 | negative |

## Performance Summary

| Metric | Strategy | Shanghai Composite | CSI 500 |
|--------|----------|----------|----------|
| Ann. Return | +13.62% | +2.38% | -2.43% |
| Ann. Volatility | 28.15% | 20.05% | 39.79% |
| Sharpe Ratio | 0.48 | 0.12 | -0.06 |
| Max Drawdown | -51.84% | -50.74% | -78.38% |
| Cum. Return | +311.60% | +29.67% | -23.76% |
| Win Rate | 56.55% | — | — |
| Excess vs Shanghai Composite | +11.71% | — | — |
| IR vs Shanghai Composite | 0.77 | — | — |
| Excess vs CSI 500 | +7.17% | — | — |
| IR vs CSI 500 | 0.26 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2015 | +77.6% | +10.2% | +67.3% | -20.5% | +98.0% | 56/23 |
| 2016 | -3.5% | -12.3% | +8.8% | -22.8% | +19.3% | 70/24 |
| 2017 | -8.4% | +6.6% | -14.9% | +2.7% | -11.1% | 73/25 |
| 2018 | -31.7% | -24.6% | -7.1% | -42.5% | +10.8% | 78/26 |
| 2019 | +31.3% | +22.3% | +9.0% | +23.3% | +8.0% | 87/26 |
| 2020 | +19.4% | +13.9% | +5.5% | -4.0% | +23.4% | 80/25 |
| 2021 | +33.6% | +4.8% | +28.8% | +4.9% | +28.7% | 80/25 |
| 2022 | -1.0% | -15.1% | +14.1% | -4.0% | +3.0% | 89/26 |
| 2023 | +14.9% | -3.7% | +18.6% | -0.7% | +15.6% | 92/27 |
| 2024 | +16.2% | +12.7% | +3.5% | -0.1% | +16.3% | 94/27 |
| 2025 | +26.2% | +18.4% | +7.8% | +82.5% | -56.3% | 91/27 |
| 2026 | +9.9% | +4.9% | +5.0% | -2.5% | +12.4% | 93/28 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 58.2%
- **Total cost drag**: 4.66%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +14.10% | +13.62% | 0.48% |
| Sharpe | 0.50 | 0.48 | 0.02 |
| Cum. Return | +331.15% | +311.60% | 19.55% |

