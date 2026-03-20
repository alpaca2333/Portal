# ic_reweighted_buf3 Backtest Report

**Description**: IC重新定权 + 换仓缓冲带 buffer_sigma=0.3  

## 策略背景与改动原因

## 策略背景

在 `ic_reweighted` 基础上加入换仓缓冲带（buffer band）。

**问题**：`ic_reweighted` 平均单边换手率 57.8%，每双周约换掉一半持仓，
边界附近的股票频繁进出，产生不必要的交易成本（累计拖累 ~3%）。

**方案**：给当前持仓股的综合得分加一个 bonus（+buffer_sigma σ），
让它们在得分略低于新候选股时仍然保留，只有明显落后的才被替换。

**参数 buffer_sigma = 0.3**：相当于持仓股享有 0.3σ 的"留任优势"，
理论上能把换手率从 ~58% 降至 ~40% 以下，同时减少摩擦成本。

**Period**: 2019-01-01 ~ 2026-02-28  
**Frequency**: biweekly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 70%  
**Selection**: top 5%, max 5/industry, equal weight  
**Avg holdings**: 97 stocks / 28 industries  
**Buffer band**: +0.3σ for incumbents  
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
| Ann. Return | +18.84% | +7.01% | +11.38% |
| Ann. Volatility | 22.52% | 16.90% | 30.33% |
| Sharpe Ratio | 0.84 | 0.41 | 0.38 |
| Max Drawdown | -22.08% | -26.04% | -39.93% |
| Cum. Return | +239.06% | +61.07% | +113.40% |
| Win Rate | 57.89% | — | — |
| Excess vs Shanghai Composite | +12.21% | — | — |
| IR vs Shanghai Composite | 0.96 | — | — |
| Excess vs CSI 500 | +2.11% | — | — |
| IR vs CSI 500 | 0.09 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2019 | +19.2% | +18.0% | +1.2% | +25.1% | -5.9% | 90/28 |
| 2020 | +19.7% | +13.9% | +5.9% | -4.0% | +23.7% | 89/27 |
| 2021 | +30.1% | +4.8% | +25.3% | +4.9% | +25.2% | 89/27 |
| 2022 | +1.1% | -15.1% | +16.2% | -4.0% | +5.1% | 99/28 |
| 2023 | +14.3% | -3.7% | +18.0% | -0.7% | +15.0% | 103/28 |
| 2024 | +18.0% | +12.7% | +5.4% | -0.1% | +18.2% | 104/28 |
| 2025 | +22.7% | +18.4% | +4.3% | +82.5% | -59.9% | 103/29 |
| 2026 | +9.1% | +4.9% | +4.2% | -2.5% | +11.6% | 108/28 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 26.6%
- **Total cost drag**: 1.37%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +19.07% | +18.84% | 0.23% |
| Sharpe | 0.85 | 0.84 | 0.01 |
| Cum. Return | +243.70% | +239.06% | 4.63% |

