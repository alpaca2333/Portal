# regime_switching_2015 Backtest Report

**Description**: 市场状态切换策略：4状态(BULL/BEAR/HIGH_VOL/NEUTRAL) + 动态因子权重  

## 策略背景与改动原因

## 策略背景

`ic_reweighted` 策略使用固定因子权重，在不同市场环境下表现差异明显：
- 2022熊市：低波动权重够，防守好（+1%）
- 2025政策急涨：固定权重无法适应反转行情

本策略在 `ic_reweighted` 基础上增加**市场状态判别层**，根据当期市场状态动态调整因子权重。

## 市场状态定义

用3个纯价格信号，无前视偏差：

| 信号 | 计算方式 |
|------|---------|
| `mkt_mom_20` | 全市场等权日收益的20日滚动均值 |
| `mkt_vol_20` | 全市场等权日收益的20日滚动标准差 |
| `mkt_breadth` | 当日上涨股占比 |

状态判别（优先级从高到低）：
1. `HIGH_VOL`：mkt_vol_20 > 历史75分位
2. `BULL`：mkt_mom_20 > 0 且 mkt_breadth > 0.55
3. `BEAR`：mkt_mom_20 < 0 且 mkt_breadth < 0.45
4. `NEUTRAL`：其余

加平滑：连续2期同状态才切换，防止噪音频繁切换。

## 因子权重切换逻辑

| 状态 | inv_pb | rev_10 | rvol_20 | log_cap | 设计逻辑 |
|------|--------|--------|---------|---------|---------|
| BULL | +0.30 | +0.15 | -0.25 | -0.30 | 趋势市，价值+小盘，反转信号弱化 |
| BEAR | +0.20 | +0.15 | -0.50 | -0.15 | 熊市防御，低波动最大 |
| HIGH_VOL | +0.20 | +0.45 | -0.25 | -0.10 | 震荡市，反转最有效 |
| NEUTRAL | +0.25 | +0.30 | -0.35 | -0.10 | IC分析默认最优权重 |

## 实现方式

- `compute_factors_fn`：标准因子计算 + 注入 `regime` 列
- `post_select`：根据 `regime` 选对应权重重新打分选股
- 其他参数（股票池、调仓频率、成本、buffer）与 `ic_reweighted` 完全一致

**Period**: 2015-01-01 ~ 2026-02-28  
**Frequency**: biweekly  
**Benchmarks**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**Universe**: SH+SZ, free_market_cap top 70%  
**Selection**: top 5%, max 5/industry, equal weight  
**Avg holdings**: 88 stocks / 27 industries  
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
| Ann. Return | +13.24% | +2.38% | -2.43% |
| Ann. Volatility | 28.92% | 20.05% | 39.79% |
| Sharpe Ratio | 0.46 | 0.12 | -0.06 |
| Max Drawdown | -50.92% | -50.74% | -78.38% |
| Cum. Return | +296.28% | +29.67% | -23.76% |
| Win Rate | 57.30% | — | — |
| Excess vs Shanghai Composite | +11.45% | — | — |
| IR vs Shanghai Composite | 0.71 | — | — |
| Excess vs CSI 500 | +7.04% | — | — |
| IR vs CSI 500 | 0.25 | — | — |

## Year-by-Year

| Year | Strategy | Shanghai Composite | Excess | CSI 500 | Excess | Holdings |
|------|----------|----------|--------|----------|--------|----------|
| 2015 | +83.5% | +10.2% | +73.3% | -20.5% | +104.0% | 57/24 |
| 2016 | -3.3% | -12.3% | +9.0% | -22.8% | +19.5% | 73/25 |
| 2017 | -8.3% | +6.6% | -14.9% | +2.7% | -11.0% | 78/26 |
| 2018 | -30.3% | -24.6% | -5.7% | -42.5% | +12.2% | 84/27 |
| 2019 | +22.9% | +22.3% | +0.6% | +23.3% | -0.3% | 92/28 |
| 2020 | +20.9% | +13.9% | +7.0% | -4.0% | +24.9% | 87/27 |
| 2021 | +34.1% | +4.8% | +29.3% | +4.9% | +29.2% | 88/26 |
| 2022 | +0.6% | -15.1% | +15.7% | -4.0% | +4.6% | 94/28 |
| 2023 | +13.0% | -3.7% | +16.7% | -0.7% | +13.7% | 102/28 |
| 2024 | +16.9% | +12.7% | +4.2% | -0.1% | +17.0% | 102/28 |
| 2025 | +20.8% | +18.4% | +2.4% | +82.5% | -61.7% | 103/29 |
| 2026 | +9.2% | +4.9% | +4.3% | -2.5% | +11.7% | 109/30 |

## Transaction Cost

- **Single-side cost**: 0.0150%
- **Avg single-side turnover**: 34.2%
- **Total cost drag**: 2.74%

| Metric | Gross | Net | Diff |
|--------|-------|-----|------|
| Ann. Return | +13.52% | +13.24% | 0.28% |
| Sharpe | 0.47 | 0.46 | 0.01 |
| Cum. Return | +307.24% | +296.28% | 10.95% |

