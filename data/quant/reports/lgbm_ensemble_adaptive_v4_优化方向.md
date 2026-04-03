# LightGBM Ensemble Adaptive V4 — 优化方向与改进路线图

> **策略代号**：`lgbm_ensemble_adaptive_v4`  
> **策略文件**：`strategies/lgbm_ensemble_adaptive_v4.py`（1366 行）  
> **文档日期**：2026-04-03  
> **基准回测**：2018-01-01 ~ 2025-12-31，初始资金 30 万，双周调仓

---

## 0. 当前 V4 表现基线

| 指标 | 数值 |
|------|------|
| 年化收益率 | +16.03% |
| 年化波动率 | 26.55% |
| 夏普比率 | 0.60 |
| 最大回撤 | -29.69% |
| 胜率 | 58.7% |
| 累计收益 | +228.59% |
| 超额收益 vs 沪深300 | +14.88% |

**已知短板**：

1. **2018 年超额为负**（-2.16% vs 上证，-0.29% vs 沪深 300）：熊市防守不足
2. **2025 年跑输中证 500 达 63.44%**：未捕捉小盘暴涨行情
3. **30 万小资金整手约束**导致持仓分散度受限，累计收益损耗
4. 行业动量模块虽贡献 17.3%，但实现粗糙（±0.02 离散加分）

---

## 1. 模型层改进

### 1.1 ⭐ 训练目标升级：Regression → LambdaRank（P0）

**现状**：两个子模型（SM/CS）均使用 `objective: regression` + `metric: rmse` 拟合 forward return 的 rank percentile。

**问题**：选股本质是**排序任务**，回归目标关注绝对预测精度，而我们只关心"谁排在前面"。RMSE 对中间排名的股票投入同等优化资源，但选股只关心 Top 5%。

**改进方案**：

```python
# 当前（V4）
params = {
    "objective": "regression",
    "metric": "rmse",
    ...
}

# 改进后
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10, 20],
    "label_gain": [0, 1, 3, 7, 15, 31],  # 自定义增益：Top 档位大幅加权
}
```

**注意事项**：
- LambdaRank 需要提供 query group（按日期分组），即每个调仓日的全部股票为一个 group
- 标签需要改为整数等级（如 5 档：0-4），而非连续 rank percentile
- 建议先在 SM 模型上试验，对比 IC / NDCG@10 的变化

**预期收益**：选股 IC 提升 5-15%，头部股票排序更精准，年化预计 +1~3%。

---

### 1.2 模型超参差异化（P2）

**现状**：SM 和 CS 两个模型使用完全相同的超参数：

```
num_leaves=31, max_depth=4, learning_rate=0.05
lambda_l1=0.5, lambda_l2=5.0
feature_fraction=0.8, bagging_fraction=0.8
```

**问题**：模型融合的核心价值在于**多样性**。相同超参数 + 不同特征仅提供有限的多样性。

**改进方案**：

| 参数 | SM 模型（保守） | CS 模型（激进） |
|------|----------------|----------------|
| num_leaves | 31 | 63 |
| max_depth | 4 | 6 |
| learning_rate | 0.05 | 0.03 |
| n_estimators | 300 | 500 |
| lambda_l1 | 0.5 | 0.1 |
| lambda_l2 | 5.0 | 1.0 |

**进阶方案**：引入第三个异质模型（XGBoost 或 CatBoost），三模型融合进一步提升多样性。

---

### 1.3 标签工程优化：行业中性化（P1）

**现状**：直接使用两个调仓日之间的原始收益率，然后 `rank(pct=True)` 作为训练标签。

**问题**：模型可能学到的是"哪个行业好"而非"行业内谁好"，在行业轮动剧烈时容易失效。

**改进方案**：

```python
# 当前
label = fwd_ret.rank(pct=True)

# 改进：行业中性化标签
industry_median = fwd_ret.groupby(industry_map).transform('median')
fwd_ret_neutral = fwd_ret - industry_median
label = fwd_ret_neutral.rank(pct=True)
```

**替代方案**：将 fwd_ret 分为 5 档（Top 20% = 4, Bottom 20% = 0），转为分类/排序问题，减少极端收益噪音。

---

### 1.4 训练窗口差异化（P2）

**现状**：SM 和 CS 均使用 `train_window_years=3`（约 750 个交易日）。

**问题**：截面因子（CS 模型）的有效性衰减较快，3 年窗口可能包含过多过时数据；而量价因子（SM 模型）的模式相对稳定。

**改进方案**：
- SM 模型：保持 3 年窗口
- CS 模型：缩短至 2 年窗口
- 或实现**指数加权训练**：近期样本权重更高

---

## 2. 因子层改进

### 2.1 ⭐ 新增关键大类因子（P1）

**现状**：SM 22 个因子 + CS 22 个因子，共 44 个因子（有重叠），主要覆盖量价、技术面、市值因子。

**缺失的关键因子大类**：

| 缺失因子 | 具体指标 | 有效性来源 | 实现难度 |
|----------|---------|-----------|---------|
| **分析师预期** | `ep_fwd_12m` 一致预期 PE, `rev_3m_chg` 盈利预测修正, `coverage_chg` 覆盖变动 | A 股最有效 alpha 因子之一 | ⭐⭐⭐（需外部数据） |
| **资金流** | `north_flow_20d` 北向资金, `inst_net_buy_20d` 机构净买入 | 聪明资金信号 | ⭐⭐⭐（需外部数据） |
| **特质波动率** | `ivol` = 回归残差的标准差 | Fama 经典负溢价因子 | ⭐⭐ |
| **彩票因子** | `max_ret_20d` = 过去 20 日最大单日涨幅 | 文献验证：高彩票效应股票未来收益低 | ⭐ |
| **偏度因子** | `skew_20d` 收益率偏度 | 与彩票因子互补 | ⭐ |
| **风格动量** | `size_momentum`, `value_momentum` | 解决 2025 年风格切换问题的关键 | ⭐⭐ |

**优先建议**：先实现 `ivol`、`max_ret_20d`、`skew_20d` 三个无需外部数据的因子，再考虑接入分析师预期和资金流数据。

---

### 2.2 行业动量连续化（P1）

**现状**：二元分类（强势 top 10 / 弱势 bottom 10），加分仅 ±0.02/-0.01，无衰减。

```python
# 当前实现
if ind in strong_set:
    ensemble[code] += 0.02   # 离散加分
elif ind in weak_set:
    ensemble[code] += -0.01  # 离散减分
```

**问题**：
- 排名第 1 和第 10 的行业获得相同的加分，信息损失严重
- 缺乏动量反转信号：强势 6 个月后反转概率显著上升
- 相关行业同时入选时集中度风险未控制

**改进方案**：

```python
# 改进：连续化 + 动量加速度
rank_normalized = 1 - (rank - 1) / (n_industries - 1)  # [0, 1]
bonus = (rank_normalized - 0.5) * scale  # 连续加分

# 动量加速度（二阶导）：检测动量拐点
mom_3m = compute_industry_momentum(lookback=3)
mom_6m = compute_industry_momentum(lookback=6)
acceleration = mom_3m - mom_6m / 2  # 正值 = 加速，负值 = 减速
if acceleration < 0 and mom_6m > 0:
    bonus *= 0.5  # 动量见顶衰减
```

---

## 3. 风控层改进

### 3.1 ⭐ 市场择时升级（P2）

**现状**：3 维度（趋势 / 广度 / 波动率）→ 5 档离散仓位系数，硬编码阈值。

```python
# 当前择时矩阵
if is_uptrend:
    if above_ratio > 0.60: coeff = 1.00
    else:                   coeff = 0.80
else:
    if above_ratio >= 0.30: coeff = 0.50
    elif not is_high_vol:   coeff = 0.30
    else:                   coeff = 0.10
```

**问题**：
- 阈值（0.60、0.30、1.5）未经优化，硬编码一刀切
- 2018 年熊市未及时降到 0.1，导致超额为负
- 2025 年可能过早降仓错过反弹
- 缺乏前瞻性信号（全部是同步/滞后指标）

**改进方案**：

#### 方案 A：连续仓位函数

```python
# 连续化：避免仓位跳变
raw_score = w1 * trend_signal + w2 * breadth_signal + w3 * vol_signal + bias
coeff = sigmoid(raw_score)  # 输出 (0, 1)
coeff = max(0.1, min(1.0, coeff))  # 限制范围
```

#### 方案 B：Hidden Markov Model（进阶）

```python
# 用 HMM 拟合市场隐状态转移
from hmmlearn import GaussianHMM
hmm = GaussianHMM(n_components=3)  # 牛 / 震荡 / 熊
hmm.fit(market_features)
state = hmm.predict(current_features)
coeff = {0: 1.0, 1: 0.6, 2: 0.2}[state]
```

#### 方案 C：增加宏观前瞻维度

| 维度 | 指标 | 作用 |
|------|------|------|
| 信用利差 | AAA-国债利差变化 | 提前感知风险偏好收缩 |
| 社融脉冲 | 社融增速 - 名义GDP增速 | 领先经济周期 3-6 月 |
| M2 同比 | M2 增速变化 | 流动性信号 |

---

### 3.2 止损机制升级（P2）

**现状**：固定 -12% 阈值，冷却 2 期（约 1 个月）。

```python
StopLossTracker(threshold=-0.12, cooldown_periods=2)
```

**问题**：
- 高波动股票频繁触发止损（假信号），低波动股票止损太宽（真亏损）
- 止损即全部卖出，没有分级减仓
- 无移动止损，盈利股票没有锁利机制

**改进方案**：

```python
# ATR 自适应止损
atr_20d = talib.ATR(high, low, close, timeperiod=20)
stop_loss_threshold = -k * atr_20d[-1] / close[-1]  # k=2~3

# 分级止损
if drawdown <= -0.5 * threshold:
    sell_ratio = 0.5   # 半仓止损
elif drawdown <= threshold:
    sell_ratio = 1.0   # 全部止损

# 移动止损（Trailing Stop）
trailing_high = max(trailing_high, current_price)
trailing_stop = trailing_high * (1 + threshold)
if current_price < trailing_stop:
    trigger_stop_loss()
```

---

## 4. 组合构建层改进

### 4.1 ⭐ 动态融合权重（P0）

**现状**：`weight_model_a=0.4, weight_model_b=0.6`（SM:CS = 0.4:0.6），写死不变。

**历史实验数据**：

| 配比 | 年化收益 | 夏普 | 备注 |
|------|---------|------|------|
| SM:CS = 0.6:0.4 | 18.28% | 0.66 | **最优静态配比** |
| SM:CS = 0.5:0.5 | 16.03% | 0.60 | 当前版本 |
| SM:CS = 0.0:1.0 | 最差 | — | 纯 CS 模型 |

**问题**：0.6:0.4 明显优于 0.5:0.5（年化差 2.25%），但即使最优配比也可能在不同市场环境下非最优。

**改进方案**：

```python
# 基于滚动 IC 动态调整
ic_a = rolling_rank_ic(model_a_scores, fwd_returns, window=6)
ic_b = rolling_rank_ic(model_b_scores, fwd_returns, window=6)
weight_a = max(0.2, ic_a / (ic_a + ic_b))
weight_b = 1 - weight_a
```

**进阶方案**：按市场状态切换
- 趋势行情（coeff ≥ 0.8）：偏 SM（量价信号强）
- 震荡行情（coeff = 0.5）：偏 CS（截面基本面稳）
- 熊市（coeff ≤ 0.3）：均衡或偏 CS

---

### 4.2 权重方案升级：风险平价（P3）

**现状**：`softmax(temperature=5.0)` + 单股 8% 上限。

**问题**：纯信号强度加权，忽略了个股波动率差异。高波动股票可能获得高权重，导致组合风险集中。

**改进方案**：

```python
# 信号-波动率复合加权
raw_weight = softmax(scores * temperature)
vol_20d = stock_returns.rolling(20).std()
risk_adjusted_weight = raw_weight / vol_20d
risk_adjusted_weight /= risk_adjusted_weight.sum()
```

---

## 5. 回测工程层改进

### 5.1 ⭐ 涨跌停不可交易过滤（P0）

**现状**：回测中涨停股可以买入，失真。

**改进**：
```python
# 在选股后过滤当日涨停股
is_limit_up = daily_df["pct_chg"] >= 9.8  # 主板涨停
is_limit_up_20 = daily_df["pct_chg"] >= 19.8  # 创业板/科创板涨停
candidates = candidates[~(is_limit_up | is_limit_up_20)]
```

### 5.2 滑点模型优化（P3）

**现状**：固定 0.15% 滑点。

**改进**：基于成交量的动态滑点
```python
slippage = base_bps + k / np.sqrt(avg_volume_20d)
```

### 5.3 买卖手续费分离（P3）

**现状**：买卖统一 0.015% commission。

**改进**：
- 买入：0.015%（佣金）
- 卖出：0.015%（佣金）+ 0.05%（印花税，2025 年减半后）

### 5.4 事件驱动调仓（P2）

**现状**：固定双周调仓，即使触发止损也要等到下一个调仓日。

**改进**：支持盘中止损立即执行，不等调仓日。

---

## 6. 健壮性验证（P3）

当前缺乏策略健壮性的系统性验证，建议补充：

### 6.1 滚动样本外测试

将 2018-2025 按年份分段，做 Walk-Forward Analysis：
- 训练：2015-2017 → 测试：2018
- 训练：2015-2018 → 测试：2019
- ...逐年滚动

### 6.2 参数敏感性分析

对以下核心参数做 ±20% 扰动，观察策略表现变化：

| 参数 | 当前值 | 扰动范围 |
|------|--------|---------|
| consensus_top_pct | 0.05 | [0.03, 0.07] |
| softmax_temperature | 5.0 | [3.0, 7.0] |
| buffer_sigma | 0.5 | [0.3, 0.7] |
| stop_loss_threshold | -0.12 | [-0.08, -0.16] |
| industry_momentum_bonus | 0.02 | [0.01, 0.04] |

若参数小幅变动导致策略表现大幅波动，说明存在过拟合风险。

### 6.3 随机打乱测试

将因子值随机打乱后跑回测，若策略仍有正收益，则信号可能是虚假的。

---

## 7. 实施路线图

### Phase 1：快速见效（1-2 天）

| 编号 | 改进项 | 难度 | 预期效果 |
|:---:|--------|:---:|---------|
| 1.1 | LambdaRank 训练目标 | ⭐⭐ | 年化 +1~3% |
| 4.1 | 动态融合权重（先恢复 0.6:0.4） | ⭐ | 年化 +1~2% |
| 5.1 | 涨跌停过滤 | ⭐ | 回撤改善 |

### Phase 2：因子增强（3-5 天）

| 编号 | 改进项 | 难度 | 预期效果 |
|:---:|--------|:---:|---------|
| 2.1 | 新增 ivol / max_ret_20d / skew_20d 因子 | ⭐⭐ | 年化 +1~2% |
| 1.3 | 行业中性化标签 | ⭐⭐ | 年化 +1~2% |
| 2.2 | 行业动量连续化 + 加速度 | ⭐ | 年化 +0.5~1% |

### Phase 3：风控与组合优化（1-2 周）

| 编号 | 改进项 | 难度 | 预期效果 |
|:---:|--------|:---:|---------|
| 3.1 | 市场择时连续化 / HMM | ⭐⭐⭐⭐ | 年化 +1~3%，回撤改善 |
| 3.2 | ATR 自适应止损 + 移动止损 | ⭐⭐ | 最大回撤 -2~3% |
| 1.2 | 模型超参差异化 | ⭐ | 年化 +0.5~1% |

### Phase 4：进阶与验证（2-4 周）

| 编号 | 改进项 | 难度 | 预期效果 |
|:---:|--------|:---:|---------|
| 2.1+ | 接入分析师预期 / 资金流数据 | ⭐⭐⭐ | 年化 +2~4% |
| 4.2 | 风险平价加权 | ⭐⭐⭐ | 夏普提升 |
| 6 | 健壮性验证框架 | ⭐⭐⭐ | 间接效果 |

---

## 8. 乐观估计：改进后 V5 目标

在 Phase 1-3 全部落地后，预期指标：

| 指标 | V4 当前 | V5 目标 | 变化 |
|------|--------|--------|------|
| 年化收益率 | +16.03% | +22~25% | +6~9% |
| 夏普比率 | 0.60 | 0.80~0.95 | +0.20~0.35 |
| 最大回撤 | -29.69% | -22~25% | 改善 5~8% |
| 2018 超额(vs上证) | -2.16% | > 0% | 转正 |
| 2025 超额(vs中证500) | -63.44% | -30~40% | 大幅改善 |

> ⚠️ 以上为乐观估计，实际效果取决于因子有效性和市场环境，需通过回测验证。

---

*文档作者：Knot AI 助手 | 基于 V4 源码（1366 行）及 7 轮回测数据分析*
