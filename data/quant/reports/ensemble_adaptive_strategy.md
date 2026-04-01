# 自适应集成选股 + 宏观择时 组合策略设计文档

> **策略代号**：`lgbm_ensemble_adaptive`  
> **策略文件**：`strategies/lgbm_ensemble_adaptive.py`  
> **版本**：v1.0 | **日期**：2026-03-31  
> **目标**：年化≥15%，夏普≥1.0，最大回撤≤25%

---

## 1. 策略概述

### 1.1 核心架构

```
┌─────────────────────────────────────────────────────┐
│              自适应集成选股 + 宏观择时                  │
├─────────────────────────────────────────────────────┤
│  模块1: 多模型集成选股 (提升收益)                      │
│    ├─ Model A: Smart Money 大资金追踪 (22因子+LGBM)   │
│    ├─ Model B: Cross-Sectional 截面排名 (22因子+LGBM) │
│    └─ 加权融合: 0.6×rank(A) + 0.4×rank(B)           │
├─────────────────────────────────────────────────────┤
│  模块2: 市场状态自适应仓位 (降低回撤, 提升夏普)         │
│    ├─ 趋势维度: 全市场中位数 MA20 vs MA60             │
│    ├─ 广度维度: 站上20日均线的股票占比                  │
│    └─ 波动维度: 20日/60日年化波动率之比                 │
├─────────────────────────────────────────────────────┤
│  模块3: 组合构建与风控 (优化风险调整后收益)              │
│    ├─ 信号强度加权 (softmax)                          │
│    ├─ 行业约束 + 个股止损                              │
│    └─ 换手控制 (buffer_sigma=0.5)                    │
└─────────────────────────────────────────────────────┘
```

### 1.2 与现有策略的差异

| 维度 | smart_money | cross_sectional V4 | 本策略 |
|------|------------|-------------------|--------|
| 选股模型 | 单模型 | 单模型 | 双模型融合 |
| 择时 | 无 | 无 | 三维度市场状态判断 |
| 权重 | 等权 | 等权 | 信号强度softmax加权 |
| 止损 | 无 | 无 | 个股跌12%强制止损 |
| 调仓频率 | 周频(W) | 月频(M) | 双周频(BW) |
| 最大持仓 | 30只 | ~60只 | 满仓25只,随状态缩减 |

---

## 2. 模块1：多模型集成选股

### 2.1 Model A：Smart Money 大资金追踪模型

**复用来源**：`lgbm_smart_money.py` 的 `compute_features_from_memory()` 函数

**22个因子列表**（与现有 smart_money 完全一致）：

| # | 因子名 | 大类 | 计算公式 |
|---|--------|------|---------|
| 1 | vol_surge_ratio | 吸筹 | vol_5d_mean / vol_60d_mean |
| 2 | shrink_pullback | 洗盘 | mean(vol[pct_chg<0]) / mean(vol[pct_chg>0])，20日窗口 |
| 3 | vol_compression | 蓄势 | std(daily_ret_10d) / std(daily_ret_60d) |
| 4 | ma_convergence | 蓄势 | std(MA5, MA10, MA20, MA60) / close |
| 5 | obv_slope | 吸筹 | OBV 20日线性回归斜率 |
| 6 | lower_shadow_ratio | 吸筹 | mean((close-low)/(high-low))，20日 |
| 7 | money_flow_strength | 吸筹 | sum(amount[ret>0]) / sum(amount[ret<0])，20日 |
| 8 | bottom_deviation | 位置 | close / min(low_120d) - 1 |
| 9 | turnover_concentration | 筹码 | std(turnover_20d) / mean(turnover_20d) |
| 10 | illiq_change | 筹码 | amihud_10d / amihud_60d |
| 11 | mom_12_1 | 动量 | close[-20] / close[-240] - 1 |
| 12 | rev_10 | 反转 | close[-1] / close[-10] - 1 |
| 13 | rvol_20 | 风险 | std(daily_ret_20d) |
| 14 | vol_confirm | 量价 | corr(ret_20d, vol_20d) |
| 15 | inv_pb | 价值 | 1 / PB |
| 16 | log_cap | 规模 | log(circ_mv) |
| 17 | roe_ttm | 质量 | ROE |
| 18 | turnover_20 | 量价 | mean(turnover_rate_f_20d) |
| 19 | volume_chg | 量价 | mean(vol_20d) / mean(vol_60d) |
| 20 | close_to_high_60 | 技术 | close / max(high_60d) |
| 21 | growth_revenue | 成长 | tr_yoy |
| 22 | growth_profit | 成长 | op_yoy |

**特征预处理**：全部做截面 rank 归一化到 [0, 1]

**模型训练**：独立 LightGBM，标签为截面收益 rank 百分位

### 2.2 Model B：Cross-Sectional 截面排名模型

**复用来源**：`lgbm_cross_sectional.py` 的 `compute_features_from_memory()` 函数

**22个因子列表**（与现有 cross_sectional V4 完全一致）：

| # | 因子名 | 大类 | 计算公式 |
|---|--------|------|---------|
| 1 | mom_12_1 | 动量 | close[-20] / close[-240] - 1 |
| 2 | rev_10 | 反转 | close[-1] / close[-10] - 1 |
| 3 | rvol_20 | 风险 | std(daily_ret_20d) |
| 4 | vol_confirm | 量价 | corr(ret_20d, vol_20d) |
| 5 | inv_pb | 价值 | 1 / PB |
| 6 | log_cap | 规模 | log(circ_mv) |
| 7 | pe_ttm | 价值 | PE_TTM |
| 8 | roe_ttm | 质量 | ROE |
| 9 | turnover_20 | 量价 | mean(turnover_rate_f_20d) |
| 10 | mom_3_1 | 动量 | close[-20] / close[-60] - 1 |
| 11 | mom_6_1 | 动量 | close[-20] / close[-120] - 1 |
| 12 | ret_5d_std | 风险 | std(daily_ret_5d) |
| 13 | volume_chg | 量价 | mean(vol_20d) / mean(vol_60d) |
| 14 | high_low_20 | 风险 | (max(high_20d) - min(low_20d)) / close |
| 15 | close_to_high_60 | 技术 | close / max(high_60d) |
| 16 | dv_ttm | 分红 | 股息率 TTM |
| 17 | roa_ttm | 质量 | ROA |
| 18 | gross_margin | 质量 | 毛利率 |
| 19 | low_leverage | 质量 | -debt_to_assets |
| 20 | growth_revenue | 成长 | tr_yoy |
| 21 | growth_profit | 成长 | op_yoy |
| 22 | illiq_20 | 流动性 | mean(|pct_chg| / amount)，20日 |

### 2.3 融合机制

**伪代码**：

```python
# 1. 两个模型独立预测
score_A = model_A.predict(features_A)  # Smart Money 分数
score_B = model_B.predict(features_B)  # Cross-Sectional 分数

# 2. 截面内 rank 归一化
rank_A = rank_pct(score_A)  # [0, 1]
rank_B = rank_pct(score_B)  # [0, 1]

# 3. 加权融合
w_A, w_B = 0.6, 0.4  # smart_money alpha更高，给更大权重
ensemble_score = w_A * rank_A + w_B * rank_B

# 4. 共识过滤：综合分前5%，且至少被一个模型评为前10%
top_5pct = ensemble_score >= quantile(ensemble_score, 0.95)
model_a_top10 = rank_A >= 0.90
model_b_top10 = rank_B >= 0.90
consensus_filter = top_5pct & (model_a_top10 | model_b_top10)

candidates = stocks[consensus_filter]
```

### 2.4 选股池定义

- **市场**：全市场沪深主板（6xxxxx.SH, 000xxx.SZ, 001xxx.SZ, 003xxx.SZ）
- **流通市值**：前85%（`mv_pct_upper=0.85`）
- **过滤**：剔除ST/*ST、停牌、close为空/0
- **偏向中小盘**：log_cap 截面后70%分位以下的股票在 ensemble_score 上获得 +0.02 的加分

---

## 3. 模块2：市场状态自适应仓位管理

> **这是从夏普0.33提升到1.0+的最关键杠杆点**

### 3.1 三维度市场状态判断

所有指标基于全市场（universe内）数据计算，**不引入新的外部数据源**。

#### 维度1：趋势维度

```python
# 计算全市场中位数收盘价的 MA20 和 MA60
# 使用 close_pivot（已有的价格数据透视表）
market_median_close = close_pivot.median(axis=1)  # 每日全市场中位数收盘价
ma20_market = market_median_close.rolling(20).mean().iloc[-1]
ma60_market = market_median_close.rolling(60).mean().iloc[-1]

is_uptrend = ma20_market > ma60_market  # True = 多头排列
```

#### 维度2：广度维度

```python
# 站上20日均线的股票占比
ma20_per_stock = close_pivot.rolling(20).mean()
last_close = close_pivot.iloc[-1]
last_ma20 = ma20_per_stock.iloc[-1]
above_ma20_ratio = (last_close > last_ma20).mean()  # [0, 1]
```

#### 维度3：波动维度

```python
# 全市场中位数20日年化波动率 vs 60日年化波动率
daily_ret = close_pivot.pct_change()
vol_20d = daily_ret.iloc[-20:].std() * np.sqrt(252)  # 每只股票的20日年化波动率
vol_60d = daily_ret.iloc[-60:].std() * np.sqrt(252)
median_vol_ratio = (vol_20d / vol_60d).median()  # 全市场中位数

is_high_vol = median_vol_ratio > 1.5
```

### 3.2 仓位矩阵

| 状态 | 趋势 | 广度(above_ma20_ratio) | 波动(is_high_vol) | 仓位系数 | 对应最大持仓 |
|------|------|----------------------|------------------|---------|------------|
| 强牛 | 多头 | >60% | 正常 | 1.00 | 25只 |
| 普牛 | 多头 | 40-60% | 正常 | 0.80 | 20只 |
| 震荡 | 空头 | 30-60% | 正常 | 0.50 | 12只 |
| 弱熊 | 空头 | <30% | 正常 | 0.30 | 7只 |
| 恐慌 | 空头 | <30% | 高 | 0.10 | 2只 |

### 3.3 判断逻辑伪代码

```python
def compute_position_coefficient(close_pivot: pd.DataFrame) -> float:
    """返回仓位系数 [0.1, 1.0]"""
    # 趋势
    market_median = close_pivot.median(axis=1)
    ma20 = market_median.rolling(20).mean().iloc[-1]
    ma60 = market_median.rolling(60).mean().iloc[-1]
    is_uptrend = ma20 > ma60
    
    # 广度
    stock_ma20 = close_pivot.rolling(20).mean()
    above_ratio = (close_pivot.iloc[-1] > stock_ma20.iloc[-1]).mean()
    
    # 波动
    daily_ret = close_pivot.pct_change()
    vol_20 = daily_ret.iloc[-20:].std() * np.sqrt(252)
    vol_60 = daily_ret.iloc[-60:].std() * np.sqrt(252)
    vol_ratio_median = (vol_20 / vol_60).median()
    is_high_vol = vol_ratio_median > 1.5
    
    # 仓位矩阵匹配
    if is_uptrend:
        if above_ratio > 0.60:
            coeff = 1.00  # 强牛
        else:
            coeff = 0.80  # 普牛
    else:
        if above_ratio >= 0.30:
            coeff = 0.50  # 震荡
        elif not is_high_vol:
            coeff = 0.30  # 弱熊
        else:
            coeff = 0.10  # 恐慌
    
    return coeff
```

### 3.4 仓位如何体现

**关键设计**：仓位系数通过**缩减持仓数量**体现，而非降低个股权重。

```python
max_positions_base = 25  # 满仓时最大持仓数
effective_max_positions = max(2, int(max_positions_base * position_coeff))
# 例如：恐慌期 effective_max = max(2, int(25 * 0.1)) = 2
# 选股时只取 top effective_max_positions 只
# 框架的 sum=1 归一化自动生效
```

这样做的好处：
- 熊市自动降仓到极低水平，大幅减少回撤
- 牛市自动满仓，不错过机会
- 与现有框架的权重归一化完全兼容

---

## 4. 模块3：组合构建与风控

### 4.1 权重方案：信号强度 Softmax 加权

**不再使用等权**，改为基于 ensemble_score 的 softmax 权重：

```python
def compute_weights(ensemble_scores: pd.Series, temperature: float = 5.0) -> pd.Series:
    """
    信号强度加权: softmax(score * temperature)
    temperature 控制权重集中度：
    - temperature=1: 接近等权
    - temperature=5: 中等集中（推荐）
    - temperature=10: 高度集中
    """
    scaled = ensemble_scores * temperature
    scaled = scaled - scaled.max()  # 数值稳定性
    exp_scores = np.exp(scaled)
    weights = exp_scores / exp_scores.sum()
    return weights
```

### 4.2 单股权重上限

```python
MAX_SINGLE_WEIGHT = 0.08  # 单只股票最大权重 8%

# 应用上限后重新归一化
weights = weights.clip(upper=MAX_SINGLE_WEIGHT)
weights = weights / weights.sum()  # 重新归一化
```

### 4.3 行业约束

```python
MAX_PER_INDUSTRY = 3  # 每个申万一级行业最多3只

# 在选股阶段实施：按 ensemble_score 降序遍历
# 每个行业计数，超过3只就跳过
```

### 4.4 个股止损

```python
STOP_LOSS_THRESHOLD = -0.12  # 持仓股距买入价跌超12%强制止损

# 实现方式：在 generate_target_weights 中
# 1. 记录每只股票的买入价（首次进入持仓时的收盘价）
# 2. 当前收盘价 / 买入价 - 1 < -0.12 时，不再将该股票纳入目标权重
# 3. 被止损的股票设置冷却期：2个调仓周期内不再买入
```

**伪代码**：

```python
class StopLossTracker:
    def __init__(self, threshold=-0.12, cooldown_periods=2):
        self.threshold = threshold
        self.cooldown = cooldown_periods
        self.entry_prices = {}   # {ts_code: entry_price}
        self.cooldown_map = {}   # {ts_code: remaining_cooldown}
    
    def update(self, date, current_holdings, prices):
        """每期调仓时调用"""
        stopped_out = set()
        
        # 检查止损
        for code in current_holdings:
            if code in self.entry_prices and code in prices:
                ret = prices[code] / self.entry_prices[code] - 1
                if ret < self.threshold:
                    stopped_out.add(code)
                    self.cooldown_map[code] = self.cooldown
        
        # 更新冷却期
        expired = []
        for code, remaining in self.cooldown_map.items():
            if remaining <= 1:
                expired.append(code)
            else:
                self.cooldown_map[code] = remaining - 1
        for code in expired:
            del self.cooldown_map[code]
        
        # 清理已卖出股票的入场价
        for code in stopped_out:
            if code in self.entry_prices:
                del self.entry_prices[code]
        
        return stopped_out
    
    def is_in_cooldown(self, code):
        return code in self.cooldown_map
    
    def record_entry(self, code, price):
        if code not in self.entry_prices:
            self.entry_prices[code] = price
    
    def record_exit(self, code):
        if code in self.entry_prices:
            del self.entry_prices[code]
```

### 4.5 换手控制

```python
BUFFER_SIGMA = 0.5  # 持仓股在 ensemble_score 上加 0.5 个标准差的优势

# 实现：
score_std = ensemble_scores.std()
for code in current_holdings:
    if code in ensemble_scores.index:
        ensemble_scores[code] += BUFFER_SIGMA * score_std
```

### 4.6 调仓频率

- **双周频（BW）**：每14个自然日取最后一个交易日调仓
- 比月频更灵活捕捉短周期 alpha
- 比周频更节省交易成本
- 框架已原生支持 `rebalance_freq="BW"`

---

## 5. 完整选股流程（Step-by-Step）

```
每个调仓日执行以下流程：

Step 1: 数据准备
  ├─ 从 bulk_data 中提取当日截面数据
  ├─ 过滤：剔除ST/停牌/无效价格
  └─ 市场范围：沪深主板，流通市值前85%

Step 2: 特征计算（两组独立计算）
  ├─ Model A 特征：调用 smart_money 的 compute_features_from_memory()
  │   → 得到 22 个因子，rank 归一化到 [0,1]
  └─ Model B 特征：调用 cross_sectional 的 compute_features_from_memory()
      → 得到 22 个因子，rank 归一化到 [0,1]

Step 3: 模型预测（两个模型独立预测）
  ├─ Model A: score_A = model_A.predict(features_A)
  └─ Model B: score_B = model_B.predict(features_B)

Step 4: 分数融合
  ├─ rank_A = rank_pct(score_A)
  ├─ rank_B = rank_pct(score_B)
  ├─ ensemble_score = 0.6 * rank_A + 0.4 * rank_B
  └─ 中小盘加分: log_cap 后70%分位 → ensemble_score += 0.02

Step 5: 市场状态判断
  ├─ 计算趋势维度 (MA20 vs MA60)
  ├─ 计算广度维度 (站上MA20比例)
  ├─ 计算波动维度 (20d/60d波动率比)
  └─ 查表得到 position_coeff (0.1 ~ 1.0)

Step 6: 持仓数量确定
  └─ effective_max = max(2, int(25 * position_coeff))

Step 7: 共识过滤
  ├─ 综合分排名前5%
  └─ 至少被一个模型评为前10%

Step 8: 止损与冷却处理
  ├─ 检查持仓股是否触发止损（跌超12%）
  ├─ 触发止损的股票不纳入候选池
  └─ 冷却期内的股票不纳入候选池

Step 9: 换手控制
  └─ 持仓股 ensemble_score += 0.5 * std(ensemble_score)

Step 10: 行业约束 + 数量裁剪
  ├─ 按 ensemble_score 降序排列
  ├─ 每行业最多3只
  └─ 总数不超过 effective_max

Step 11: 权重计算
  ├─ softmax(ensemble_score * 5.0)
  ├─ 单股权重上限 8%
  └─ 重新归一化使权重和=1

Step 12: 输出目标权重
  └─ return {ts_code: weight}
```

---

## 6. LightGBM 超参数

两个子模型使用**相同的超参数**（沿用现有策略验证过的参数，不大改）：

| 参数 | 值 | 说明 |
|------|-----|------|
| objective | regression | 回归任务 |
| metric | rmse | 均方根误差 |
| boosting_type | gbdt | 梯度提升决策树 |
| num_leaves | 31 | 叶子节点数 |
| learning_rate | 0.05 | 学习率 |
| feature_fraction | 0.7 | 特征采样比例 |
| bagging_fraction | 0.8 | 样本采样比例 |
| bagging_freq | 5 | 采样频率 |
| max_depth | 4 | 最大深度 |
| min_child_samples | 100 | 叶子最小样本数 |
| lambda_l1 | 0.5 | L1 正则化 |
| lambda_l2 | 5.0 | L2 正则化 |
| num_boost_round | 300 | 最大迭代轮数 |
| early_stopping_rounds | 20 | 早停轮数 |
| seed | 42 | 随机种子 |

**训练配置**：
- 训练窗口：3年滚动
- 重训练频率：每4个调仓期重训练一次（约2个月）
- 训练/验证分割：80%/20%（按时间顺序）
- 标签：截面收益 rank 百分位 [0, 1]

---

## 7. 回测配置建议

```python
cfg = BacktestConfig(
    initial_capital=1_000_000,
    commission_rate=1.5e-4,    # 1.5 bps 单边佣金
    slippage=0.0015,           # 15 bps 滑点
    start_date="2018-01-01",
    end_date="2025-12-31",
    rebalance_freq="BW",       # 双周频
    db_path="data/quant/data/quant.db",
    baseline_dir="data/quant/baseline",
    output_dir="data/quant/backtest",
)

strategy = LGBMEnsembleAdaptive(
    # 模型参数
    train_window_years=3,
    retrain_interval=4,        # 每4期重训练
    # 集成参数
    weight_model_a=0.6,        # smart_money 权重
    weight_model_b=0.4,        # cross_sectional 权重
    consensus_top_pct=0.05,    # 综合分前5%
    consensus_single_top_pct=0.10,  # 单模型前10%
    # 仓位参数
    max_positions=25,          # 满仓最大持仓
    # 权重参数
    softmax_temperature=5.0,
    max_single_weight=0.08,
    # 风控参数
    max_per_industry=3,
    stop_loss_threshold=-0.12,
    stop_loss_cooldown=2,
    buffer_sigma=0.5,
    # 选股池参数
    mv_pct_upper=0.85,
    small_cap_bonus=0.02,
    small_cap_quantile=0.70,
    feature_lookback=260,
    backtest_end_date="2025-12-31",
)
```

---

## 8. 程序员实现清单

### 8.1 需要新建的文件

| 文件 | 说明 |
|------|------|
| `strategies/lgbm_ensemble_adaptive.py` | 主策略文件 |

### 8.2 需要导入/复用的模块

| 来源文件 | 复用内容 | 方式 |
|----------|---------|------|
| `lgbm_smart_money.py` | `FEATURE_COLUMNS`, `FEATURE_NAMES` (as SM_*), `compute_features_from_memory`, `rank_normalize`, `compute_forward_return_from_memory`, `train_lgbm_model` | 直接 import |
| `lgbm_cross_sectional.py` | `FEATURE_COLUMNS`, `FEATURE_NAMES` (as CS_*), `compute_features_from_memory`, `rank_normalize`, `compute_forward_return_from_memory`, `train_lgbm_model` | 直接 import |
| `strategies/utils.py` | `prefetch_bulk_data` | 直接 import |
| `engine/` | `BacktestConfig`, `StrategyBase`, `run_backtest`, `DataAccessor` | 直接 import |

### 8.3 类结构

```python
class LGBMEnsembleAdaptive(StrategyBase):
    """
    自适应集成选股 + 宏观择时策略
    继承 StrategyBase，实现 generate_target_weights
    """
    
    def __init__(self, ...):
        super().__init__("lgbm_ensemble_adaptive")
        # 存储所有参数
        # 初始化两个子模型的状态
        # 初始化 StopLossTracker
    
    def describe(self) -> str:
        """策略描述，嵌入回测报告"""
    
    # ── 内部方法 ──
    def _warmup_training_cache(self, date, accessor):
        """预热：加载历史数据，计算两组特征+标签"""
    
    def _compute_model_a_features(self, date, bulk_data, st_codes):
        """计算 Model A (Smart Money) 的特征"""
        # 调用 smart_money 的 compute_features_from_memory
    
    def _compute_model_b_features(self, date, bulk_data, st_codes):
        """计算 Model B (Cross-Sectional) 的特征"""
        # 调用 cross_sectional 的 compute_features_from_memory
    
    def _train_model(self, model_id, train_cache, date):
        """训练指定子模型"""
    
    def _compute_market_state(self, close_pivot):
        """计算市场状态，返回 position_coeff"""
    
    def _compute_ensemble_score(self, score_a, score_b, log_cap_series):
        """融合两个模型的分数"""
    
    def _apply_stop_loss(self, candidates, current_holdings, prices):
        """应用止损逻辑"""
    
    def _select_and_weight(self, ensemble_scores, feat_df, 
                           current_holdings, effective_max):
        """选股 + 行业约束 + softmax 加权"""
    
    # ── 主入口 ──
    def generate_target_weights(self, date, accessor, current_holdings):
        """回测引擎调用的主接口"""
```

### 8.4 关键实现注意事项

1. **两组特征独立计算**：Model A 和 Model B 的 `FEATURE_NAMES` 不同（各22个），特征计算函数也不同，但 `FEATURE_COLUMNS`（数据库字段）基本相同，可以共享 `bulk_data`。

2. **两个模型独立训练**：各自维护 `_model_a` 和 `_model_b`，各自有 `_train_cache_a` 和 `_train_cache_b`。训练周期可以同步（同一期同时重训练）。

3. **特征命名空间**：import 时需要区分：
   ```python
   from strategies.lgbm_smart_money import (
       FEATURE_NAMES as SM_FEATURE_NAMES,
       compute_features_from_memory as sm_compute_features,
       train_lgbm_model as sm_train_model,
       rank_normalize,
   )
   from strategies.lgbm_cross_sectional import (
       FEATURE_NAMES as CS_FEATURE_NAMES,
       compute_features_from_memory as cs_compute_features,
       train_lgbm_model as cs_train_model,
   )
   ```

4. **bulk_data 共享**：两个模型共享同一份 `bulk_data`（数据库字段是两者的并集），只需一次 prefetch。`FEATURE_COLUMNS` 取两者的并集：
   ```python
   ALL_FEATURE_COLUMNS = list(set(SM_FEATURE_COLUMNS + CS_FEATURE_COLUMNS))
   ```

5. **市场状态计算**：需要 `close_pivot`（全市场收盘价透视表）。可以在特征计算阶段顺带构建（smart_money 的 `compute_features_from_memory` 内部已经构建了 `close_pivot`，但未暴露）。建议：在主策略中独立构建一个用于择时的 close_pivot：
   ```python
   # 从 bulk_data 中提取
   window_dates = all_dates[all_dates <= date][-60:]
   window = bulk_data[bulk_data["trade_date"].isin(window_dates)]
   close_pivot = window.pivot_table(
       index="trade_date", columns="ts_code", values="close"
   ).sort_index()
   ```

6. **止损逻辑**：`StopLossTracker` 需要在策略对象中持久化（因为 `generate_target_weights` 会被多次调用）。买入价记录：每期调仓后，新增持仓的买入价 = 当日收盘价。

7. **数据字段确认**：以下字段在数据库中已存在且被两个策略使用过：
   - `ts_code, trade_date, open, high, low, close, pre_close`
   - `pct_chg, vol, amount, turnover_rate_f`
   - `pb, pe_ttm, circ_mv`
   - `roe, roa, grossprofit_margin, debt_to_assets`
   - `dv_ttm, tr_yoy, op_yoy`
   - `sw_l1, is_suspended`
   - 无需引入任何新的外部数据源。

8. **`_bulk_date_index_cache` 冲突**：两个子模型各自有模块级别的 `_bulk_date_index_cache` dict。由于共享同一个 `bulk_data` DataFrame（同一个 `id()`），只要两个模块都基于同一个 DataFrame 对象计算，缓存不会冲突。但注意：每次更新 `bulk_data` 后都要清除两个模块的缓存：
   ```python
   import strategies.lgbm_smart_money as sm_module
   import strategies.lgbm_cross_sectional as cs_module
   sm_module._bulk_date_index_cache.clear()
   cs_module._bulk_date_index_cache.clear()
   ```

---

## 附录：预期效果分析

| 年份 | smart_money 实际 | 预期改善 | 改善来源 |
|------|-----------------|---------|---------|
| 2018 | -33.79% | → -15~-20% | 择时降仓至10-30% |
| 2019 | +46.75% | → +40~50% | 集成选股基本维持 |
| 2020 | +6.70% | → +15~25% | 双模型互补捕捉结构机会 |
| 2021 | +19.51% | → +15~20% | 基本维持 |
| 2022 | -4.75% | → +0~5% | 择时+止损控制亏损 |
| 2023 | +3.81% | → +5~10% | 集成选股小幅提升 |
| 2024 | +10.38% | → +10~15% | 基本维持 |

**综合预期**：
- 年化收益：13~17%（目标≥15%）
- 夏普比率：0.7~1.1（目标≥1.0）
- 最大回撤：-20~-25%（目标≤-25%）
- 胜率：55~65%

**核心预期逻辑**：
- 择时模块对夏普的贡献最大：2018年从-33.79%改善到-15%~-20%，直接贡献年化约+2%和夏普约+0.3
- 止损机制：在震荡/下跌市中进一步降低回撤约3-5个百分点
- 双模型融合：通过共识过滤提高选股胜率，预计贡献年化约+1~2%
