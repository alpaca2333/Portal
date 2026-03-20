# ic_reweighted_buf5 Backtest Report

**Description**: IC重新定权 + 换仓缓冲带 buffer_sigma=0.5  

## 策略背景与改动原因

## 策略背景

在 `ic_reweighted` 基础上加入换仓缓冲带（buffer band）。

**问题**：`ic_reweighted` 平均单边换手率 57.8%，每双周约换掉一半持仓，
边界附近的股票频繁进出，产生不必要的交易成本（累计拖累 ~3%）。

**方案**：给当前持仓股的综合得分加一个 bonus（+buffer_sigma σ），
让它们在得分略低于新候选股时仍然保留，只有明显落后的才被替换。

**参数 buffer_sigma = 0.5**：相当于持仓股享有 0.5σ 的"留任优势"，
理论上能把换手率从 ~58% 降至 ~40% 以下，同时减少摩擦成本。

**Period**: 2019-01-01 ~ 2026-02-28  
**Frequency**: biweekly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 70%  
**Selection**: top 5%, max 5/industry, equal weight  
**Avg holdings**: 103 stocks / 29 industries  
**Buffer band**: +0.5σ for incumbents  
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
| Ann. Return | +18.19% | +7.01% | +11.38% |
| Ann. Volatility | 22.24% | 16.90% | 30.33% |
| Sharpe Ratio | 0.82 | 0.41 | 0.38 |
| Max Drawdown | -22.31% | -26.04% | -39.93% |
| Cum. Return | +226.19% | +61.07% | +113.40% |
| Win Rate | 57.31% | — | — |
| Excess vs Shanghai Composite | +11.60% | — | — |
| IR vs Shanghai Composite | 0.95 | — | — |
| Excess vs CSI 500 | +1.48% | — | — |
| IR vs CSI 500 | 0.06 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2019 | +17.3% | +18.0% | -0.7% | +25.1% | -7.8% | 94/29 |
| 2020 | +20.0% | +13.9% | +6.1% | -4.0% | +24.0% | 93/29 |
| 2021 | +28.6% | +4.8% | +23.8% | +4.9% | +23.8% | 95/28 |
| 2022 | +1.3% | -15.1% | +16.5% | -4.0% | +5.3% | 105/29 |
| 2023 | +14.6% | -3.7% | +18.3% | -0.7% | +15.3% | 108/28 |
| 2024 | +17.9% | +12.7% | +5.2% | -0.1% | +18.0% | 110/29 |
| 2025 | +20.6% | +18.4% | +2.2% | +82.5% | -62.0% | 109/29 |
| 2026 | +9.2% | +4.9% | +4.3% | -2.5% | +11.7% | 117/30 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 16.7%
- **Total cost drag**: 0.86%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +18.33% | +18.19% | 0.14% |
| Sharpe | 0.82 | 0.82 | 0.01 |
| Cum. Return | +228.98% | +226.19% | 2.79% |

