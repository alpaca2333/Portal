# industry_neutral_concentrated_ng Backtest Report

**Description**: Industry-neutral concentrated strategy (biweekly), top 5% + max 5/industry, engine rewrite  
**Period**: 2019-01-01 ~ 2026-02-28  
**Frequency**: biweekly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 70%  
**Selection**: top 5%, max 5/industry, equal weight  
**Avg holdings**: 90 stocks / 27 industries  
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
| Ann. Return | +20.00% | +7.01% | +11.38% |
| Ann. Volatility | 22.53% | 16.90% | 30.33% |
| Sharpe Ratio | 0.89 | 0.41 | 0.38 |
| Max Drawdown | -24.06% | -26.04% | -39.93% |
| Cum. Return | +263.20% | +61.07% | +113.40% |
| Win Rate | 59.65% | — | — |
| Excess vs Shanghai Composite | +13.20% | — | — |
| IR vs Shanghai Composite | 1.01 | — | — |
| Excess vs CSI 500 | +3.02% | — | — |
| IR vs CSI 500 | 0.12 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2019 | +15.5% | +18.0% | -2.5% | +25.1% | -9.6% | 86/27 |
| 2020 | +13.1% | +13.9% | -0.7% | -4.0% | +17.2% | 89/27 |
| 2021 | +39.4% | +4.8% | +34.6% | +4.9% | +34.5% | 84/26 |
| 2022 | +10.4% | -15.1% | +25.6% | -4.0% | +14.4% | 89/26 |
| 2023 | +16.1% | -3.7% | +19.8% | -0.7% | +16.8% | 93/27 |
| 2024 | +8.1% | +12.7% | -4.6% | -0.1% | +8.2% | 95/27 |
| 2025 | +30.0% | +18.4% | +11.5% | +82.5% | -52.6% | 97/28 |
| 2026 | +10.7% | +4.9% | +5.8% | -2.5% | +13.2% | 89/27 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 48.8%
- **Total cost drag**: 2.50%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +20.42% | +20.00% | 0.42% |
| Sharpe | 0.91 | 0.89 | 0.02 |
| Cum. Return | +272.34% | +263.20% | 9.14% |

