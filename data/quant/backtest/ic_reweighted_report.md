# ic_reweighted Backtest Report

**Description**: IC引导因子重新定权：移除动量和量价，加大反转和低波动  

## 策略背景与改动原因

## 为什么做这次改动

通过 `factor_ic_analysis.py` 对 2022-2026 年所有因子的 Rank IC 进行了系统测算，发现原版策略存在两个"方向错误"的因子：

- **`mom_12_1`（12-1月动量，权重 +25%）**：全期 IC 均值 = -0.014，ICIR = -0.10，每年都是负向或接近零。
  原本假设"过去涨的股票未来继续涨"，但在 A 股双周频率下，动量完全无效，反而略微拖累。
- **`vol_confirm`（量价确认，权重 +15%）**：全期 IC 均值 = -0.064，ICIR = -0.54，全年都是反向。
  这是最强的"负向因子"——选高量价比的股票反而亏钱，方向完全搞反了。

真正有效的因子（ICIR > 0.3）：`rev_10`（+0.37）、`log_cap`（+0.35）、`rvol_20`（+0.34）、`inv_pb`（+0.33）。

## 改动内容

仅修改因子权重，其他参数（股票池、调仓频率、选股比例、成本）与 `industry_neutral_concentrated_ng` 完全一致：

| 因子 | 原权重 | 新权重 | 原因 |
|------|--------|--------|------|
| `mom_12_1` | +0.25 | 移除 | 全期IC负向 |
| `vol_confirm` | +0.15 | 移除 | ICIR=-0.54，方向相反 |
| `rvol_20` | -0.15 | **-0.35** | 最稳定因子，加大 |
| `rev_10` | +0.10 | **+0.30** | 2025年IC=+0.08，加大 |
| `inv_pb` | +0.25 | +0.25 | 稳定，保持 |
| `log_cap` | -0.10 | -0.10 | 稳定，保持 |

**Period**: 2019-01-01 ~ 2026-02-28  
**Frequency**: biweekly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 70%  
**Selection**: top 5%, max 5/industry, equal weight  
**Avg holdings**: 87 stocks / 26 industries  
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
| Ann. Return | +19.81% | +7.01% | +11.38% |
| Ann. Volatility | 23.29% | 16.90% | 30.33% |
| Sharpe Ratio | 0.85 | 0.41 | 0.38 |
| Max Drawdown | -24.49% | -26.04% | -39.93% |
| Cum. Return | +259.09% | +61.07% | +113.40% |
| Win Rate | 56.14% | — | — |
| Excess vs Shanghai Composite | +13.20% | — | — |
| IR vs Shanghai Composite | 0.97 | — | — |
| Excess vs CSI 500 | +2.93% | — | — |
| IR vs CSI 500 | 0.12 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2019 | +22.9% | +18.0% | +4.9% | +25.1% | -2.2% | 84/26 |
| 2020 | +19.4% | +13.9% | +5.5% | -4.0% | +23.4% | 80/25 |
| 2021 | +33.5% | +4.8% | +28.7% | +4.9% | +28.6% | 80/25 |
| 2022 | -1.0% | -15.1% | +14.1% | -4.0% | +3.0% | 89/26 |
| 2023 | +14.9% | -3.7% | +18.6% | -0.7% | +15.6% | 92/27 |
| 2024 | +16.2% | +12.7% | +3.5% | -0.1% | +16.3% | 94/27 |
| 2025 | +26.2% | +18.4% | +7.8% | +82.5% | -56.3% | 91/27 |
| 2026 | +9.9% | +4.9% | +5.0% | -2.5% | +12.4% | 93/28 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 57.8%
- **Total cost drag**: 2.96%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +20.31% | +19.81% | 0.50% |
| Sharpe | 0.87 | 0.85 | 0.02 |
| Cum. Return | +269.81% | +259.09% | 10.73% |

