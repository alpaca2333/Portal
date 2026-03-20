# lgb_stock_selection 回测报告

**策略描述**: LightGBM cross-sectional stock ranking with rolling retraining. 15 features, 3-year rolling window, biweekly rebalance.  

## 策略背景与改动原因

### Motivation

The linear multi-factor model (v2) uses fixed weights that cannot capture non-linear interactions between factors (e.g., momentum works differently in high-vol vs low-vol regimes). LightGBM learns these conditional effects automatically from data.

### Key Design Choices

1. **Cross-sectional rank labels**: predict relative ranking, not absolute returns. This removes non-stationarity of return distributions.
2. **Rolling 3-year training window**: adapts to regime changes without look-ahead bias.
3. **Industry-aware selection**: top 5% by ML score, capped at 5 per industry (same as v2 for fair comparison).
4. **15 features**: 6 base factors (same as v2) + 9 extended features (short/mid momentum, micro-vol, turnover, price distance, etc.).
5. **Feature rank normalization**: all features mapped to [0,1] percentile within each period, ensuring cross-period comparability.

**回测区间**: 2022-02-28 ~ 2023-02-28  
**调仓频率**: biweekly  
**基准指数**: Shanghai Composite (sh000001) / CSI 500 (sz000905)  
**选股范围**: 沪深主板, 自由流通市值前 70%  
**选股规则**: 前 5%，每行业最多 5 只，等权  
**平均持仓**: 70 只 / 26 个行业  
**缓冲带**: 持仓 +0.3σ 加分  
**交易成本**: 单边 0.0150%  

## 因子列表

| 因子 | 权重 | 方向 |
|------|------|------|
| mom_12_1 | +0.00 | 反向 |
| rev_10 | +0.00 | 反向 |
| rvol_20 | +0.00 | 反向 |
| vol_confirm | +0.00 | 反向 |
| inv_pb | +0.00 | 反向 |
| log_cap | +0.00 | 反向 |
| pe_ttm | +0.00 | 反向 |
| roe_ttm | +0.00 | 反向 |
| turnover_20 | +0.00 | 反向 |
| mom_3_1 | +0.00 | 反向 |
| mom_6_1 | +0.00 | 反向 |
| ret_5d_std | +0.00 | 反向 |
| volume_chg | +0.00 | 反向 |
| high_low_20 | +0.00 | 反向 |
| close_to_high_60 | +0.00 | 反向 |

## 业绩概览

| 指标 | 策略 | Shanghai Composite | CSI 500 |
|------|------|------|------|
| 年化收益 | +126.55% | +7.69% | +7.41% |
| 年化波动 | 27.62% | 17.66% | 31.43% |
| 夏普比率 | 4.58 | 0.44 | 0.24 |
| 最大回撤 | -8.60% | -14.86% | -16.04% |
| 累计收益 | +118.94% | +7.04% | +6.78% |
| 胜率 | 79.17% | — | — |
| 超额vsShanghai Composite | +134.48% | — | — |
| IR vsShanghai Composite | 7.98 | — | — |
| 超额vsCSI 500 | +125.29% | — | — |
| IR vsCSI 500 | 5.81 | — | — |

## 逐年表现

| 年份 | 策略 | Shanghai Composite | 超额 | CSI 500 | 超额 | 持仓 |
|------|----------|----------|--------|----------|--------|----------|
| 2022 | +78.8% | +0.8% | +77.9% | +3.6% | +75.1% | 70/26 |
| 2023 | +22.5% | +6.2% | +16.3% | +3.0% | +19.4% | 68/26 |

## 交易成本

- **单边费率**: 0.0150%
- **平均单边换手**: 57.9%
- **累计成本拖累**: 0.42%

| 指标 | 毛收益 | 净收益 | 差值 |
|------|--------|--------|------|
| 年化收益 | +127.50% | +126.55% | 0.96% |
| 夏普比率 | 4.62 | 4.58 | 0.03 |
| 累计收益 | +119.83% | +118.94% | 0.89% |

## 时间布局

- **滚动训练/验证/测试**: 截至 **2022-02-28**
- **留出期（纯样本外）**: **2022-02-28 ~ 2023-02-28** （12个月）
- 留出期使用**最后一个训练好的模型**推理，**零数据泄露**。

## 模型质量: Rank IC（样本外）

| 指标 | 数值 |
|------|------|
| 平均 Rank IC | 0.0914 |
| Rank IC 标准差 | 0.1224 |
| IC IR（均值/标准差） | 0.75 |
| IC > 0 占比 | 84.6% |
| 调仓期数 | 14 |

### 逐年 Rank IC

| 年份 | 平均IC | IC标准差 | 期数 |
|------|--------|----------|------|
| 2021 | 0.0961 | 0.1230 | 9 |
| 2022 | 0.0808 | 0.1560 | 4 |

## 特征重要性（跨窗口平均）

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | rev_10 | 333.0 |
| 2 | high_low_20 | 305.5 |
| 3 | mom_12_1 | 291.0 |
| 4 | log_cap | 276.0 |
| 5 | close_to_high_60 | 247.5 |
| 6 | turnover_20 | 232.0 |
| 7 | inv_pb | 227.5 |
| 8 | pe_ttm | 227.5 |
| 9 | mom_6_1 | 222.0 |
| 10 | vol_confirm | 221.0 |
| 11 | ret_5d_std | 218.5 |
| 12 | mom_3_1 | 199.0 |
| 13 | volume_chg | 189.0 |
| 14 | roe_ttm | 186.5 |
| 15 | rvol_20 | 167.5 |

## 模型架构

- **模型**: LightGBM（梯度提升决策树）
- **训练窗口**: 3 年滚动
- **验证窗口**: 6 个月
- **重训频率**: 每 6 个月
- **特征数**: 15 个，截面排序归一化
- **标签**: 下期收益率，排序归一化至 [0,1]
- **训练窗口数**: 2

### 关键超参数

| 参数 | 值 |
|------|-----|
| num_leaves | 31 |
| max_depth | 6 |
| learning_rate | 0.05 |
| n_estimators | 1000 |
| min_child_samples | 100 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |

## 对比: LightGBM vs 线性多因子 (v2)

| 维度 | 线性v2 | LightGBM |
|------|--------|----------|
| 打分方式 | 固定线性权重 | 非线性树集成 |
| 因子交互 | 无（仅加法） | 自动（条件分裂） |
| 市场适应 | 无（静态权重） | 每6个月滚动重训 |
| 特征数量 | 6 | 15 |
| 特征归一化 | 缩尾z-score | 截面百分位排序 |
| 选股方式 | 得分前5% | ML得分前5% |
| 行业约束 | 每行业最多5只 | 每行业最多5只 |
| 缓冲带 | +0.3σ | +0.3σ（按分数标准差缩放） |

