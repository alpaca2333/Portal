# ic_reweighted_buf8 Backtest Report

**Description**: IC重新定权 + 换仓缓冲带 buffer_sigma=0.8  

## 策略背景与改动原因

## 策略背景

在 `ic_reweighted` 基础上加入换仓缓冲带（buffer band）。

**问题**：`ic_reweighted` 平均单边换手率 57.8%，每双周约换掉一半持仓，
边界附近的股票频繁进出，产生不必要的交易成本（累计拖累 ~3%）。

**方案**：给当前持仓股的综合得分加一个 bonus（+buffer_sigma σ），
让它们在得分略低于新候选股时仍然保留，只有明显落后的才被替换。

**参数 buffer_sigma = 0.8**：相当于持仓股享有 0.8σ 的"留任优势"，
理论上能把换手率从 ~58% 降至 ~40% 以下，同时减少摩擦成本。

**Period**: 2019-01-01 ~ 2026-02-28  
**Frequency**: biweekly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 70%  
**Selection**: top 5%, max 5/industry, equal weight  
**Avg holdings**: 109 stocks / 29 industries  
**Buffer band**: +0.8σ for incumbents  
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
| Ann. Return | +16.71% | +7.01% | +11.38% |
| Ann. Volatility | 21.47% | 16.90% | 30.33% |
| Sharpe Ratio | 0.78 | 0.41 | 0.38 |
| Max Drawdown | -22.28% | -26.04% | -39.93% |
| Cum. Return | +198.33% | +61.07% | +113.40% |
| Win Rate | 58.48% | — | — |
| Excess vs Shanghai Composite | +10.09% | — | — |
| IR vs Shanghai Composite | 0.87 | — | — |
| Excess vs CSI 500 | +0.07% | — | — |
| IR vs CSI 500 | 0.00 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2019 | +15.7% | +18.0% | -2.3% | +25.1% | -9.5% | 98/29 |
| 2020 | +15.5% | +13.9% | +1.6% | -4.0% | +19.5% | 98/30 |
| 2021 | +28.4% | +4.8% | +23.6% | +4.9% | +23.6% | 100/29 |
| 2022 | +0.4% | -15.1% | +15.5% | -4.0% | +4.4% | 110/30 |
| 2023 | +14.2% | -3.7% | +17.9% | -0.7% | +14.8% | 116/29 |
| 2024 | +17.2% | +12.7% | +4.5% | -0.1% | +17.3% | 120/30 |
| 2025 | +18.0% | +18.4% | -0.4% | +82.5% | -64.6% | 121/30 |
| 2026 | +9.7% | +4.9% | +4.8% | -2.5% | +12.2% | 122/30 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 9.4%
- **Total cost drag**: 0.48%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +16.79% | +16.71% | 0.08% |
| Sharpe | 0.78 | 0.78 | 0.00 |
| Cum. Return | +199.75% | +198.33% | 1.43% |

