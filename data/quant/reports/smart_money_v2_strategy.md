# Smart Money V2 — 聪明资金进阶版 策略设计文档

> 作者：策略研究员 | 日期：2026-03-31
> 基线策略：`lgbm_smart_money`（年化 +9.42%，夏普 0.33，最大回撤 -35.68%）
> 策略文件：`strategies/lgbm_smart_money_v2.py`
> 目标：**夏普 ≥ 0.5，最大回撤 ≤ -30%**

---

## 1. 策略概述

Smart Money V2 在 V1 的基础上进行三大核心升级：**市场状态感知风控**、**信号增强**和**权重优化**。策略继续使用 LightGBM 模型识别大资金吸筹行为，但新增了市场状态判断模块，在系统性下跌时自动降仓；新增 3 个增强因子（筹码集中度变化率、行业相对动量、沪深300权重变化）提升信号质量；从等权分配改为 ML 分数强度加权 + 波动率倒数加权的混合方案，降低高波动个股对组合的冲击。同时提高共振确认阈值从 3 个到 4 个，并引入个股止损和组合波动率上限，全面控制尾部风险。

---

## 2. 与 V1 的主要区别

| 维度 | V1 (lgbm_smart_money) | V2 (smart_money_v2) |
|------|----------------------|---------------------|
| **风控** | 无风控模块 | 市场状态感知仓位管理 + 个股止损 + 组合波动率上限 |
| **因子数量** | 22 个（10 大资金 + 12 基础） | 25 个（10 大资金 + 3 新增 + 12 基础） |
| **新增因子** | — | 筹码集中度变化率、行业相对动量、沪深300权重变化 |
| **共振确认** | ≥ 3 个大资金信号 | ≥ 4 个大资金信号（更严格） |
| **权重分配** | 等权 (1/N) | ML 分数强度加权 × 波动率倒数加权（混合） |
| **单股权重上限** | 无 | 10%（防过度集中） |
| **个股止损** | 无 | 持仓股距买入价跌超 15% 强制卖出 |
| **市场仓位** | 始终满仓（有信号时） | 市场下跌趋势时自动半仓，极端情况下 20% 仓位 |
| **组合波动率控制** | 无 | 组合近 20 日波动率超阈值时缩减至高确信度持仓 |
| **换手率缓冲** | buffer_sigma = 0.3 | buffer_sigma = 0.5（更稳定） |
| **最大持仓数** | 30 | 25（更集中于高确信度标的） |
| **每行业上限** | 3 | 3（不变） |

---

## 3. 因子设计

### 3.1 保留的 V1 因子（22 个，不变）

V1 的全部 22 个因子保持不变，包括 10 个大资金追踪因子和 12 个基础上下文因子。具体参见 V1 策略文档。

### 3.2 新增因子（3 个）

#### 因子 #23：筹码集中度变化率 (turnover_cv_change)

| 属性 | 值 |
|------|-----|
| **名称** | `turnover_cv_change` |
| **大类** | 筹码 |
| **含义** | 换手率变异系数（CV）的时序变化，衡量筹码从分散到集中的趋势 |
| **计算公式** | `CV_10d / CV_60d`，其中 `CV = std(turnover_rate_f) / mean(turnover_rate_f)` |
| **方向** | ↓ 越低越好（CV 在缩小 = 筹码正在从分散走向集中 = 锁仓加强） |
| **所需 DB 字段** | `turnover_rate_f` |
| **回看窗口** | 60 日 |

**计算逻辑**：
```python
# 近 10 日换手率 CV
turn_10 = turn_pivot.iloc[-10:]
cv_10 = turn_10.std() / turn_10.mean().replace(0, np.nan)

# 近 60 日换手率 CV
turn_60 = turn_pivot.iloc[-60:]
cv_60 = turn_60.std() / turn_60.mean().replace(0, np.nan)

# 变化率：< 1 表示筹码趋于集中
turnover_cv_change = cv_10 / cv_60.replace(0, np.nan)
```

**信号逻辑**：当 `turnover_cv_change < 0.8` 时视为"筹码加速集中"信号（rank 归一化后 ≤ 0.30 触发）。

---

#### 因子 #24：行业相对动量 (industry_relative_momentum)

| 属性 | 值 |
|------|-----|
| **名称** | `industry_relative_momentum` |
| **大类** | 动量 |
| **含义** | 个股动量减去其所在行业的平均动量，捕捉行业内的相对强势 |
| **计算公式** | `stock_mom_20d - industry_avg_mom_20d` |
| **方向** | ↑ 越高越好（个股在行业中相对更强） |
| **所需 DB 字段** | `close`, `sw_l1` |
| **回看窗口** | 20 日 |

**计算逻辑**：
```python
# 个股 20 日收益率
stock_ret_20 = close_pivot.iloc[-1] / close_pivot.iloc[-20] - 1

# 按 sw_l1 行业分组，计算行业平均动量
# 注意：需从 snap 中获取每只股票的 sw_l1，构建 {ts_code: sw_l1} 映射
stock_industry = snap_indexed["sw_l1"]  # Series: ts_code -> sw_l1
ret_with_ind = pd.DataFrame({"ret": stock_ret_20, "sw_l1": stock_industry})
industry_avg = ret_with_ind.groupby("sw_l1")["ret"].transform("mean")

industry_relative_momentum = stock_ret_20 - industry_avg
```

**注意**：该因子不参与大资金信号计数（它是增强因子，不是大资金信号）。

---

#### 因子 #25：沪深300权重变化 (hs300_weight_change)

| 属性 | 值 |
|------|-----|
| **名称** | `hs300_weight_change` |
| **大类** | 资金流 |
| **含义** | 个股在沪深300指数中的权重变化，间接反映被动资金和机构资金的配置方向变化 |
| **计算公式** | `weight_current - weight_previous`（当前最新权重 - 上一期权重） |
| **方向** | ↑ 越高越好（权重上升 = 被纳入/增配 = 更多资金流入） |
| **所需 DB 表** | `index_weight` 表（`index_code='000300.SH'`） |
| **前视偏差** | 使用 `accessor.get_index_weights(date)` 获取当日或之前最近的权重数据，安全 |

**计算逻辑**：
```python
# 获取当期权重
current_weights = accessor.get_index_weights(date, index_code="000300.SH")
# current_weights: DataFrame with columns [con_code, weight]

# 获取上一期权重（回看约 30-60 天）
# index_weight 表通常月度/季度更新，取前一期即可
prev_date = date - pd.DateOffset(days=60)  # 足够回溯到上一个公布日
prev_weights = accessor.get_index_weights(prev_date, index_code="000300.SH")

if current_weights is not None and prev_weights is not None:
    curr_w = current_weights.set_index("con_code")["weight"]
    prev_w = prev_weights.set_index("con_code")["weight"]
    # 合并：当前有、之前无的视为新纳入（prev = 0）
    all_codes = curr_w.index.union(prev_w.index)
    curr_aligned = curr_w.reindex(all_codes, fill_value=0)
    prev_aligned = prev_w.reindex(all_codes, fill_value=0)
    hs300_weight_change = curr_aligned - prev_aligned
else:
    # 无数据时填 0（中性）
    hs300_weight_change = pd.Series(0, index=valid_codes)
```

**注意**：
- 非沪深300成份股的 `weight_change` 为 0（中性值），不影响其选股。
- 该因子不参与大资金信号计数（作为增强信号使用）。
- `get_index_weights` API 已有前视偏差保护，安全使用。

---

### 3.3 更新后的完整因子列表

```python
FEATURE_NAMES = [
    # ── Smart money accumulation features (10) ── [不变]
    "vol_surge_ratio",        # 底部放量比
    "shrink_pullback",        # 缩量回调比
    "vol_compression",        # 波动率压缩
    "ma_convergence",         # 均线粘合度
    "obv_slope",              # OBV 20日斜率
    "lower_shadow_ratio",     # 下影线比例
    "money_flow_strength",    # 资金流强度
    "bottom_deviation",       # 底部偏离度
    "turnover_concentration", # 换手率集中度
    "illiq_change",           # 流动性变化

    # ── NEW: Enhanced features (3) ── [新增]
    "turnover_cv_change",     # 筹码集中度变化率
    "industry_relative_momentum",  # 行业相对动量
    "hs300_weight_change",    # 沪深300权重变化

    # ── Base context features (12) ── [不变]
    "mom_12_1",
    "rev_10",
    "rvol_20",
    "vol_confirm",
    "inv_pb",
    "log_cap",
    "roe_ttm",
    "turnover_20",
    "volume_chg",
    "close_to_high_60",
    "growth_revenue",
    "growth_profit",
]
```

### 3.4 需要新增读取的 DB 列

在 `FEATURE_COLUMNS` 中无需新增列——`turnover_rate_f`、`close`、`sw_l1` 已在 V1 中包含。唯一新增的是通过 `accessor.get_index_weights()` API 读取 `index_weight` 表，该调用独立于 `FEATURE_COLUMNS`。

---

## 4. 风控规则

### 4.1 市场状态感知仓位管理（最核心改进）

**目的**：在市场系统性下跌时自动降低仓位，避免 2018 年 -33.79%、2020 年小牛市踏空等问题。

#### 市场状态判断

使用沪深300指数（或上证指数）的均线系统判断市场状态。由于策略的 `bulk_data` 中包含个股数据，我们需要用**全市场截面统计**来间接推断市场状态（因为没有直接的指数日线数据）。

**方案：用全市场中位数收盘价的均线系统**

```python
def compute_market_state(bulk_data, date, lookback=60):
    """
    计算市场状态，返回仓位调整系数 [0.2, 1.0]。
    
    使用全市场中位数收益率和波动率来判断：
    1. 趋势信号：全市场中位数股票的 MA20 vs MA60
    2. 波动率信号：全市场近 20 日波动率 vs 历史 60 日波动率
    3. 广度信号：站上 20 日均线的股票占比
    """
    # 获取窗口数据
    all_dates, date_to_rows = _get_bulk_date_index(bulk_data)
    date_ts = pd.Timestamp(date)
    valid_dates = all_dates[all_dates <= date_ts]
    
    if len(valid_dates) < 60:
        return 1.0  # 数据不足，默认满仓
    
    window_dates = valid_dates[-60:]
    # ... 构建 close_pivot（与 compute_features_from_memory 同理）
    
    # --- 信号 1：趋势 ---
    # 全市场每日中位数收盘价序列
    market_median = close_pivot.median(axis=1)  # 每日全市场中位数
    ma20 = market_median.iloc[-20:].mean()
    ma60 = market_median.iloc[-60:].mean()
    trend_bullish = ma20 >= ma60  # True = 上升趋势
    
    # --- 信号 2：波动率 ---
    market_ret = close_pivot.pct_change(fill_method=None).iloc[1:]
    market_median_ret = market_ret.median(axis=1)
    vol_20 = market_median_ret.iloc[-20:].std() * np.sqrt(252)
    vol_60 = market_median_ret.iloc[-60:].std() * np.sqrt(252)
    vol_elevated = vol_20 > vol_60 * 1.5  # 近期波动率显著高于长期
    
    # --- 信号 3：广度 ---
    last_close = close_pivot.iloc[-1]
    ma20_stock = close_pivot.iloc[-20:].mean()
    breadth = (last_close > ma20_stock).mean()  # 站上20日均线的比例
    breadth_weak = breadth < 0.3  # 不到 30% 的股票在均线之上
    
    # --- 综合判断 ---
    if not trend_bullish and vol_elevated and breadth_weak:
        # 三重看跌：极端防御
        return 0.2
    elif not trend_bullish and (vol_elevated or breadth_weak):
        # 两重看跌：半仓防御
        return 0.5
    elif not trend_bullish:
        # 单一看跌：七成仓位
        return 0.7
    else:
        # 上升趋势：满仓
        return 1.0
```

**在 `generate_target_weights` 中使用**：
```python
# 计算市场仓位系数
position_scale = self.compute_market_state(self._bulk_data, date)
print(f"      [风控] 市场仓位系数 = {position_scale:.1f}")

# 最终权重 = 原始选股权重 × 仓位系数
# 框架会自动归一化权重，所以 position_scale 需要体现在持仓数量上
if position_scale < 1.0:
    # 只保留 ML 分数最高的 N 只，N = max_positions × position_scale
    effective_max = max(3, int(self.max_positions * position_scale))
    # 在 _select_stocks 中使用 effective_max 替代 self.max_positions
```

#### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| MA 短期窗口 | 20 日 | 约 1 个月的趋势 |
| MA 长期窗口 | 60 日 | 约 3 个月的趋势 |
| 波动率放大阈值 | 1.5 倍 | vol_20 > vol_60 × 1.5 视为异常 |
| 广度阈值 | 30% | 低于 30% 股票在均线之上视为弱市 |
| 最低仓位 | 20% | 极端情况下仍保留最低仓位 |

---

### 4.2 个股止损

**规则**：持仓股票的当前价格距上次买入日的收盘价跌超 15% 时，强制卖出（不纳入下期目标权重）。

**实现方式**：

```python
# 在策略类中维护买入价格记录
self._entry_prices: Dict[str, float] = {}  # {ts_code: entry_close_price}

def generate_target_weights(self, date, accessor, current_holdings):
    # ... 正常选股后得到 weights: Dict[str, float]
    
    # 获取当日价格
    snap = accessor.get_date(date, columns=["close"])
    current_prices = snap.set_index("ts_code")["close"].to_dict()
    
    # 检查止损
    stopped_out = set()
    for code in list(weights.keys()):
        if code in self._entry_prices:
            entry_price = self._entry_prices[code]
            current_price = current_prices.get(code, 0)
            if current_price > 0 and entry_price > 0:
                drawdown = (current_price - entry_price) / entry_price
                if drawdown <= -0.15:  # 跌超15%
                    stopped_out.add(code)
                    del weights[code]
                    print(f"      [止损] {code} 触发止损: "
                          f"买入价={entry_price:.2f} 现价={current_price:.2f} "
                          f"跌幅={drawdown:.1%}")
    
    # 更新买入价格记录
    # 新买入的股票记录当前价格
    for code in weights:
        if code not in self._entry_prices:
            self._entry_prices[code] = current_prices.get(code, 0)
    # 已卖出的股票清除记录
    for code in list(self._entry_prices.keys()):
        if code not in weights:
            del self._entry_prices[code]
    
    return weights
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 止损阈值 | -15% | 距买入价跌超 15% 强制卖出 |
| 买入价基准 | 买入日收盘价 | 用 `accessor.get_date(date, columns=["close"])` 获取 |

---

### 4.3 组合波动率上限

**规则**：当组合近 20 日年化波动率超过 35%（阈值）时，缩减持仓至当期 ML 分数最高的 top-K 只，K = `max_positions × 0.5`。

**实现方式**：

```python
def compute_portfolio_volatility(self, date, accessor, current_holdings):
    """
    计算当前持仓组合的近 20 日年化波动率。
    """
    if not current_holdings:
        return 0.0
    
    held_codes = list(current_holdings.keys())
    window = accessor.get_window(date, lookback=21, ts_codes=held_codes,
                                  columns=["close"])
    if window is None or window.empty:
        return 0.0
    
    pivot = window.pivot(index="trade_date", columns="ts_code", values="close")
    daily_ret = pivot.pct_change(fill_method=None).iloc[1:]
    
    # 等权组合日收益
    portfolio_ret = daily_ret.mean(axis=1)
    vol_annual = portfolio_ret.std() * np.sqrt(252)
    return vol_annual
```

**在选股中使用**：
```python
port_vol = self.compute_portfolio_volatility(date, accessor, current_holdings)
if port_vol > 0.35:  # 年化波动率 > 35%
    # 缩减持仓：只保留高确信度标的
    effective_max = max(5, int(self.max_positions * 0.5))
    print(f"      [风控] 组合波动率 {port_vol:.1%} 超阈值，"
          f"缩减最大持仓至 {effective_max}")
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 波动率阈值 | 35% 年化 | 超过时触发缩仓 |
| 缩仓比例 | 50% | 缩减至 max_positions 的一半 |
| 最低持仓数 | 5 | 即使触发缩仓也保留至少 5 只 |

---

### 4.4 风控规则优先级与叠加

三项风控规则**独立计算、取最严**：

```
effective_max_positions = min(
    self.max_positions,                              # 基础上限 25
    int(self.max_positions * position_scale),         # 市场状态调整
    int(self.max_positions * vol_scale),              # 波动率调整（超阈值时 0.5，否则 1.0）
)
effective_max_positions = max(3, effective_max_positions)  # 绝对下限

# 个股止损在最终权重中单独处理（从 weights 中删除触发止损的标的）
```

---

## 5. 选股逻辑

### 5.1 完整选股流程

```
┌─────────────────────────────────────────────────────┐
│ Step 0: 预热与模型训练（同 V1，首次调仓时执行）       │
│   - 加载 bulk_data（含历史 + 回测区间全量数据）        │
│   - 遍历历史调仓日，计算特征 + 标签，缓存训练数据      │
│   - 训练 LightGBM 模型                               │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: 计算全市场特征（25 个）                       │
│   - 从 bulk_data 内存计算 22 个 V1 因子              │
│   - 新增计算 3 个增强因子                             │
│   - 过滤选股池：排除 ST、停牌、无价格                  │
│   - 正则匹配主板/中小板代码                           │
│   - 市值过滤（mv_pct_upper = 0.85）                  │
│   - Rank 归一化至 [0, 1]                             │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: 风控评估                                     │
│   ① 市场状态 → position_scale ∈ [0.2, 1.0]          │
│   ② 组合波动率 → vol_scale ∈ {0.5, 1.0}              │
│   ③ effective_max = min(25, 25×position_scale,        │
│                         25×vol_scale)                 │
│      effective_max = max(3, effective_max)             │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: ML 预测                                      │
│   - 如需重训练（每 4 周）：构建训练集 → 训练模型       │
│   - 对当期全市场特征进行预测 → ml_score               │
│   - 冷启动时使用等权因子均值替代                       │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: 双重过滤（宁缺毋滥，比 V1 更严格）            │
│   ① ML 分数阈值：ml_score ≥ 截面 85% 分位数          │
│   ② 大资金信号共振：10 个大资金因子中                  │
│      ≥ 4 个处于前 30% 分位（V1 是 ≥ 3 个）           │
│   ③ 两个条件必须同时满足                              │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 5: 持仓缓冲                                     │
│   - buffer_sigma = 0.5（V1 是 0.3）                  │
│   - 已持有股票的 ml_score += 0.5 × score_std         │
│   - 减少不必要的换手                                  │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 6: 排序 + 行业约束 + 持仓上限                    │
│   - 按调整后的 ml_score 降序排列                      │
│   - 每行业最多 3 只                                   │
│   - 总持仓最多 effective_max 只                       │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 7: 个股止损                                     │
│   - 检查已入选股票中，是否有距买入价跌超 15% 的        │
│   - 触发止损的股票从权重中删除                         │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 8: 权重分配（见下节详述）                        │
│   - ML 分数强度加权 × 波动率倒数加权                  │
│   - 单股权重上限 10%                                  │
│   - 归一化至 sum = 1                                  │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ Step 9: 输出目标权重 Dict[str, float]                │
│   - 返回给回测引擎执行调仓                            │
│   - 不在字典中的持仓自动卖出                          │
└─────────────────────────────────────────────────────┘
```

### 5.2 大资金信号计数规则（V2 更新）

10 个大资金因子的信号触发条件**与 V1 完全一致**：

| 因子 | 触发方向 | 阈值（rank 归一化后） |
|------|---------|---------------------|
| vol_surge_ratio | HIGH is good | ≥ 0.70 |
| obv_slope | HIGH is good | ≥ 0.70 |
| lower_shadow_ratio | HIGH is good | ≥ 0.70 |
| money_flow_strength | HIGH is good | ≥ 0.70 |
| illiq_change | HIGH is good | ≥ 0.70 |
| shrink_pullback | LOW is good | ≤ 0.30 |
| vol_compression | LOW is good | ≤ 0.30 |
| ma_convergence | LOW is good | ≤ 0.30 |
| turnover_concentration | LOW is good | ≤ 0.30 |
| bottom_deviation | LOW is good (relaxed) | ≤ 0.40 |

**V2 变化**：最低共振确认数从 **3 → 4**。

---

## 6. 权重分配

### 6.1 方案：ML 分数强度 × 波动率倒数 混合加权

V1 使用简单等权分配（1/N），V2 改为混合加权方案：

```python
def compute_weights(self, selected_codes, result_df, feat_ranked, date):
    """
    混合加权：50% ML 分数强度 + 50% 波动率倒数。
    """
    n = len(selected_codes)
    if n == 0:
        return {}
    
    # --- 组件 1：ML 分数强度权重 ---
    ml_scores = result_df.set_index("ts_code").loc[selected_codes, "ml_score"]
    # 线性映射：分数越高权重越大
    ml_min = ml_scores.min()
    ml_range = ml_scores.max() - ml_min
    if ml_range > 0:
        ml_weights = (ml_scores - ml_min) / ml_range + 0.5  # [0.5, 1.5] 范围
    else:
        ml_weights = pd.Series(1.0, index=selected_codes)
    ml_weights = ml_weights / ml_weights.sum()  # 归一化
    
    # --- 组件 2：波动率倒数权重 ---
    rvol = feat_ranked.set_index("ts_code").loc[selected_codes, "rvol_20"]
    rvol = rvol.replace(0, np.nan).fillna(rvol.median())
    inv_vol = 1.0 / rvol
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median())
    vol_weights = inv_vol / inv_vol.sum()  # 归一化
    
    # --- 混合 ---
    combined = 0.5 * ml_weights + 0.5 * vol_weights
    
    # --- 单股权重上限 10% ---
    max_single = 0.10
    combined = combined.clip(upper=max_single)
    
    # --- 重新归一化 ---
    combined = combined / combined.sum()
    
    return combined.to_dict()
```

### 6.2 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| ML 权重占比 | 50% | 高分股票获得更多权重 |
| 波动率倒数权重占比 | 50% | 低波动股票获得更多权重，降低组合整体波动 |
| ML 分数映射范围 | [0.5, 1.5] | 防止最低分股票权重为 0 |
| 单股权重上限 | 10% | 防止过度集中于少数标的 |

### 6.3 权重分配示例

假设选中 10 只股票：

| 股票 | ML 分数 | 20 日波动率 | ML 权重 | 波动率倒数权重 | 混合权重 | 截断后 |
|------|--------|-----------|---------|-------------|---------|--------|
| A | 0.95 | 15% | 15.0% | 8.0% | 11.5% | **10.0%** |
| B | 0.92 | 20% | 13.0% | 6.0% | 9.5% | 9.5% |
| C | 0.88 | 12% | 11.0% | 10.0% | 10.5% | **10.0%** |
| ... | ... | ... | ... | ... | ... | ... |

截断后重新归一化至 sum = 1。

---

## 7. 回测配置建议

### 7.1 主回测配置

```python
cfg = BacktestConfig(
    initial_capital=1_000_000,
    commission_rate=1.5e-4,     # 万1.5 单边佣金
    slippage=0.0015,             # 15bps 滑点
    start_date="2018-01-01",
    end_date="2025-12-31",
    rebalance_freq="W",          # 周频调仓（与 V1 一致）
    db_path="data/quant/data/quant.db",
    baseline_dir="data/quant/baseline",
    output_dir="data/quant/backtest",
)
```

### 7.2 策略参数

```python
strategy = LGBMSmartMoneyV2(
    # --- 训练相关 ---
    train_window_years=3,        # 训练窗口 3 年
    retrain_interval=4,          # 每 4 周重训练
    feature_lookback=260,        # 特征回看 260 日
    
    # --- 选股相关 ---
    score_quantile=0.85,         # ML 分数 85% 分位阈值
    min_signal_count=4,          # 大资金信号共振 ≥ 4 个（V1 是 3）
    max_per_industry=3,          # 每行业最多 3 只
    max_positions=25,            # 最大持仓 25 只（V1 是 30）
    buffer_sigma=0.5,            # 持仓缓冲 0.5（V1 是 0.3）
    mv_pct_upper=0.85,           # 市值过滤（前 85%）
    
    # --- 风控相关 ---
    stop_loss_pct=-0.15,         # 个股止损 -15%
    market_vol_threshold=1.5,    # 市场波动率放大阈值
    breadth_threshold=0.30,      # 市场广度阈值
    portfolio_vol_cap=0.35,      # 组合波动率上限（年化）
    min_position_scale=0.2,      # 最低仓位系数
    
    # --- 权重相关 ---
    ml_weight_ratio=0.5,         # ML 分数权重占比
    vol_weight_ratio=0.5,        # 波动率倒数权重占比
    max_single_weight=0.10,      # 单股权重上限
    
    backtest_end_date=cfg.end_date,
)
```

### 7.3 LightGBM 超参数（与 V1 一致）

```python
lgbm_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 4,
    "min_child_samples": 100,
    "lambda_l1": 0.5,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": 42,
}
num_boost_round = 300
early_stopping_rounds = 20
```

---

## 8. 预期改进

### 8.1 各项指标预期

| 指标 | V1 实际值 | V2 预期值 | 改善来源 |
|------|----------|----------|---------|
| **年化收益率** | +9.42% | +10~13% | 信号增强（+1~2%）+ 权重优化（+0.5~1%） |
| **年化波动率** | 28.26% | 22~25% | 市场状态降仓（-3~4%）+ 波动率加权（-1~2%） |
| **夏普比率** | 0.33 | **0.45~0.55** | 收益小幅提升 + 波动率显著下降 |
| **最大回撤** | -35.68% | **-25~30%** | 2018 年降仓 + 个股止损 + 组合波动率控制 |
| **胜率** | 56.4% | 57~60% | 更严格的共振确认 + 止损截断亏损 |
| **超额收益 vs 上证** | +7.42% | +8~10% | 信号增强 + 风控减损 |

### 8.2 分年度预期改善

| 年份 | V1 收益 | V2 预期改善 | 改善逻辑 |
|------|--------|-----------|---------|
| **2018** | -33.79% | → -20~25% | 市场状态模块在熊市自动降仓至 20~50% |
| **2019** | +19.29% | → +18~22% | 基本持平，市场上升趋势时满仓 |
| **2020** | +6.70% | → +10~15% | 结构性牛市中，行业相对动量因子有效 |
| **2021** | +15.27% | → +15~18% | 基本持平，波动率加权小幅改善 |
| **2022** | -4.75% | → -2~+2% | 市场状态降仓 + 止损减损 |
| **2023** | +11.53% | → +12~15% | 信号增强小幅改善 |
| **2024** | +30.69% | → +28~32% | 基本持平（V1 已表现优秀） |
| **2025** | +50.29% | → +45~52% | 基本持平（可能因更严格筛选略降） |

### 8.3 风险提示

1. **V2 可能在极端牛市中略微跑输 V1**：因为风控模块在波动率上升时会降仓，可能在 V 型反弹中错过部分收益
2. **行业相对动量因子在行业轮动加速时可能失效**：如果资金快速在行业间轮转，20 日窗口可能滞后
3. **沪深300权重变化因子的更新频率较低**：index_weight 数据通常月度/季度更新，对短期信号贡献有限
4. **止损规则在 V 型走势中可能导致"割在最低点"**：止损后该股在同一调仓期内不会回买

### 8.4 后续迭代方向（留待 V3）

- 标签优化：从单期 forward return 改为多期平滑收益
- 目标函数：从 regression 改为 LambdaRank
- 多窗口模型集成：训练多个不同窗口长度的模型取平均
- 动态参数调整：根据市场状态动态调整 score_quantile（牛市放宽到 0.80，熊市收紧到 0.90）

---

## 9. 实现清单（给程序员）

### 9.1 文件结构

```
strategies/
├── lgbm_smart_money.py       # V1（不动）
├── lgbm_smart_money_v2.py    # V2（新建，从 V1 复制并修改）
└── utils.py                  # 公共工具（不动）
```

### 9.2 需要修改的部分

从 `lgbm_smart_money.py` 复制为 `lgbm_smart_money_v2.py`，然后：

1. **类名**：`LGBMSmartMoney` → `LGBMSmartMoneyV2`
2. **策略名**：`"lgbm_smart_money"` → `"lgbm_smart_money_v2"`
3. **`FEATURE_NAMES`**：新增 3 个因子名称
4. **`compute_features_from_memory`**：新增 3 个因子的计算逻辑
5. **`__init__`**：新增风控参数（stop_loss_pct, market_vol_threshold 等）和权重参数
6. **新增方法 `compute_market_state`**：市场状态判断
7. **新增方法 `compute_portfolio_volatility`**：组合波动率计算
8. **新增方法 `compute_weights`**：混合加权
9. **修改 `_select_stocks`**：使用 effective_max 替代固定 max_positions
10. **修改 `generate_target_weights`**：
    - 调用风控模块计算 effective_max
    - 实现个股止损逻辑
    - 调用 compute_weights 替代等权
11. **修改 `_count_smart_money_signals`**：不需要改（规则不变）
12. **修改 `min_signal_count`**：默认值从 3 → 4
13. **修改 `max_positions`**：默认值从 30 → 25
14. **修改 `buffer_sigma`**：默认值从 0.3 → 0.5
15. **修改 `describe`**：更新策略描述
16. **`__main__` 区块**：更新参数

### 9.3 不需要修改的部分

- `engine/` 目录下的所有文件（回测框架不动）
- `strategies/utils.py`（公共工具不动）
- `train_lgbm_model` 函数（训练逻辑不动）
- `rank_normalize` 函数（不动）
- `compute_forward_return_from_memory` 函数（不动）

### 9.4 特别注意

- `hs300_weight_change` 因子需要在 `compute_features_from_memory` 中额外接收 `accessor` 参数（或在 `generate_target_weights` 中单独计算后合并到 feat_df），因为需要调用 `accessor.get_index_weights()`
- 所有新增代码必须在 `accessor.set_current_date(date)` 之后执行，确保不存在前视偏差
- `self._entry_prices` 字典需要在 `__init__` 中初始化
- 风控的 `compute_market_state` 使用的是 `bulk_data` 中的数据（已加载到内存），不需要额外的 SQL 查询
