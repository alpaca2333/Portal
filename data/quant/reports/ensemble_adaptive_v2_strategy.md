# 自适应集成选股 V2：波动率目标制 + 多级择时 改进策略

> **策略代号**：`lgbm_ensemble_adaptive_v2`  
> **策略文件**：`strategies/lgbm_ensemble_adaptive_v2.py`  
> **版本**：v2.0 | **日期**：2026-03-31  
> **基准策略**：`lgbm_ensemble_adaptive` (v1.0, 夏普 0.57, 年化 +14.60%, 最大回撤 -34.27%)  
> **目标**：夏普 >= 0.8，年化 >= 15%，最大回撤 <= -25%

---

## 1. V1 策略诊断：从 0.57 到 0.8 的改进空间

### 1.1 分年度弱点分析

| 年份 | V1收益 | 沪深300收益 | 超额 | 诊断 | 改进潜力 |
|------|--------|-----------|------|------|---------|
| **2018** | -34.67% | -28.75% | -5.93% | 择时模块完全失效：MA20/MA60在单边熊市中反复震荡，持仓数在7~20只间频繁切换 | **高**：若能控制在-15%~-20%，年化+2%，夏普+0.15 |
| 2019 | +36.38% | +35.57% | +0.80% | 跟随大盘反弹，选股无明显alpha | 低 |
| **2020** | +9.46% | +24.08% | -14.61% | 大盘蓝筹领涨，策略中小盘偏好被严重拖累 | **中**：若能自适应风格切换至+15%，夏普+0.05 |
| 2021 | +53.16% | -2.86% | +56.03% | 最强年份，中小盘行情+选股alpha爆发 | 维持 |
| 2022 | +8.71% | -21.87% | +30.58% | 熊市防御优秀 | 维持 |
| 2023 | +6.55% | -12.91% | +19.46% | 稳健alpha | 维持 |
| 2024 | +21.62% | +17.51% | +4.11% | 中等表现 | 低 |
| 2025 | +41.35% | +17.70% | +23.64% | 强劲表现 | 维持 |

### 1.2 核心问题诊断

**问题1：择时模块在2018年适得其反（夏普损失约 -0.15）**

月度数据揭示：
- 2018-01-29：持仓8只，亏损-13.97%（初始即重仓）
- 2018-04-09：持仓扩大到20只（死猫反弹时加仓）
- 2018-04-23：急缩到8只（亏-8.28%后缩仓）
- 2018-12月：又扩到20只（年底抄底失败，-3.9%~-5.6%）

根因：MA20/MA60 是趋势跟踪型指标，在2018年这种无趋势单边阴跌+频繁反弹的市场中，**反复被假突破欺骗**，导致"高位加仓、低位减仓"的逆向操作。

**问题2：波动率过高（25.66%）拖累夏普**

夏普 = (14.60% - Rf) / 25.66%。即使收益不变，若能将波动率从25.66%降至20%，夏普将从0.57提升到0.73。波动率高的原因：
- 持仓数动态变化（2~25只），导致组合波动巨大
- 单一止损阈值（-12%）在高波动期频繁触发，加大换手和波动
- softmax权重集中度使少数个股波动传导到整体组合

**问题3：缺乏风格自适应（2020年亏损-14.61%超额）**

V1使用固定的 `small_cap_bonus=0.02` 和 `mv_pct_upper=0.85`，在2020年大盘蓝筹主导的行情下，中小盘偏好成为拖累。策略缺乏对市场风格的识别和适应能力。

### 1.3 改进路径量化

| 改进项 | 预期夏普增量 | 机制 |
|--------|------------|------|
| 波动率目标制替代离散仓位档 | +0.10~0.15 | 直接降低组合波动率至目标水平 |
| 多级择时：加入中期趋势确认 | +0.05~0.10 | 消除2018年式的假突破交易 |
| 自适应止损 | +0.03~0.05 | 减少高波动期的错误止损 |
| 风格自适应（大/小盘动态切换） | +0.03~0.05 | 改善2020年类似行情 |
| 模型训练优化 | +0.02~0.03 | 提高IC稳定性 |
| **合计预期** | **+0.23~0.38** | **目标夏普 0.80~0.95** |

---

## 2. 改进方案一：波动率目标制仓位管理（核心改进）

### 2.1 设计思路

V1的五档离散仓位（1.0/0.8/0.5/0.3/0.1）有两个根本问题：
1. 档位跳变大（如0.8→0.3），组合波动剧烈
2. MA20/MA60择时信号滞后且容易被假突破欺骗

**解决方案**：用**波动率目标制（Volatility Targeting）**替代离散仓位档。核心思想：设定组合年化波动率目标（如18%），根据近期实际波动率动态调整仓位，使组合波动率始终维持在目标附近。

### 2.2 波动率目标制公式

```python
# 核心参数
TARGET_VOL = 0.18          # 目标年化波动率 18%
VOL_LOOKBACK = 40          # 波动率估算窗口（交易日）
VOL_SCALE_MIN = 0.15       # 最低缩放比例（防止完全空仓）
VOL_SCALE_MAX = 1.20       # 最高缩放比例（允许轻度加杠杆效果，但不超配）
VOL_SMOOTHING = 0.7        # 指数平滑系数（防止仓位剧烈跳变）

def compute_vol_target_scale(portfolio_returns: pd.Series, prev_scale: float) -> float:
    """
    根据近期组合波动率计算仓位缩放系数。
    
    参数:
        portfolio_returns: 最近 VOL_LOOKBACK 个交易日的日收益率
        prev_scale: 上一期的缩放系数（用于平滑）
    
    返回:
        scale: 仓位缩放系数 [VOL_SCALE_MIN, VOL_SCALE_MAX]
    """
    if len(portfolio_returns) < 20:
        return 1.0  # 数据不足时不调整
    
    # 使用指数加权标准差，给近期更高权重
    recent_vol = portfolio_returns.ewm(span=VOL_LOOKBACK, min_periods=20).std().iloc[-1]
    annualized_vol = recent_vol * np.sqrt(252)
    
    if annualized_vol <= 0 or np.isnan(annualized_vol):
        return prev_scale
    
    # 原始缩放比
    raw_scale = TARGET_VOL / annualized_vol
    
    # 平滑处理：避免仓位剧烈跳变
    smoothed_scale = VOL_SMOOTHING * prev_scale + (1 - VOL_SMOOTHING) * raw_scale
    
    # 限制在合理范围内
    return np.clip(smoothed_scale, VOL_SCALE_MIN, VOL_SCALE_MAX)
```

### 2.3 与V1仓位系统的结合

波动率目标制**不完全取代**V1的市场状态判断，而是**叠加**使用：

```python
# Step 1: 市场状态判断（简化为三档，去掉容易误判的"强牛"和"恐慌"）
market_regime = compute_market_regime(close_pivot)  # "bull" / "neutral" / "bear"

# Step 2: 市场状态修正系数
regime_modifier = {
    "bull": 1.0,     # 牛市不做额外限制
    "neutral": 0.85, # 震荡市小幅降低上限
    "bear": 0.60,    # 熊市限制仓位上限
}[market_regime]

# Step 3: 波动率目标缩放
vol_scale = compute_vol_target_scale(recent_portfolio_returns, prev_vol_scale)

# Step 4: 综合仓位系数
final_scale = min(vol_scale, regime_modifier)  # 取较保守值
effective_max_positions = max(3, int(MAX_POSITIONS_BASE * final_scale))
```

### 2.4 预期效果

以2018年为例分析：
- 2018年1-3月市场开始下跌，组合波动率上升至30%+
- 波动率目标制自动将仓位缩放至 18%/30% = 0.6
- 叠加 bear regime_modifier=0.6 → final_scale=0.6
- 持仓从25只降到约15只
- 关键区别：**不会因为短期反弹就扩回25只**（因为波动率不会瞬间下降）
- 预期2018年亏损从-34.67%改善至-18%~-22%

---

## 3. 改进方案二：多级趋势确认择时

### 3.1 V1择时的问题

V1仅使用 MA20 vs MA60 判断趋势方向，这是一个单一级别的均线系统。问题：
- 在2018年式的"阴跌+反弹"行情中，MA20频繁上穿下穿MA60
- 每次"多头排列"信号都导致扩大持仓，但紧接着就是新一轮下跌

### 3.2 改进：三级趋势确认

```python
def compute_market_regime(close_pivot: pd.DataFrame) -> str:
    """
    三级趋势确认：
    Level 1: 短期 — MA5 vs MA20（日级别）
    Level 2: 中期 — MA20 vs MA60（周级别）  
    Level 3: 长期 — 60日新高/新低比率
    
    只有两个级别以上确认，才判定为牛/熊。
    """
    n = len(close_pivot)
    if n < 60:
        return "neutral"
    
    market_median = close_pivot.median(axis=1)
    
    # Level 1: 短期趋势
    ma5 = market_median.rolling(5).mean().iloc[-1]
    ma20 = market_median.rolling(20).mean().iloc[-1]
    short_up = ma5 > ma20
    
    # Level 2: 中期趋势（保留V1的判断）
    ma60 = market_median.rolling(60).mean().iloc[-1]
    mid_up = ma20 > ma60
    
    # Level 3: 长期趋势 — 60日内创新高的股票占比 vs 创新低的占比
    high_60 = close_pivot.rolling(60).max()
    low_60 = close_pivot.rolling(60).min()
    last_close = close_pivot.iloc[-1]
    
    new_high_pct = (last_close >= high_60.iloc[-1] * 0.98).mean()  # 接近60日新高
    new_low_pct = (last_close <= low_60.iloc[-1] * 1.02).mean()   # 接近60日新低
    
    long_up = new_high_pct > new_low_pct + 0.05  # 新高比例显著大于新低
    long_down = new_low_pct > new_high_pct + 0.05
    
    # 多级确认
    bull_score = sum([short_up, mid_up, long_up])
    bear_score = sum([not short_up, not mid_up, long_down])
    
    if bull_score >= 2:
        return "bull"
    elif bear_score >= 2:
        return "bear"
    else:
        return "neutral"
```

### 3.3 广度指标改进

V1的广度指标（站上MA20的股票比例）保留，但增加**广度动量**：

```python
# 广度动量：广度的变化方向比广度本身更重要
breadth_current = (close_pivot.iloc[-1] > stock_ma20.iloc[-1]).mean()
breadth_5d_ago = (close_pivot.iloc[-5] > stock_ma20.iloc[-5]).mean()
breadth_momentum = breadth_current - breadth_5d_ago

# 广度正在恶化时额外降低仓位
if breadth_momentum < -0.10:
    regime_modifier *= 0.8  # 广度快速恶化，额外降低20%仓位
```

---

## 4. 改进方案三：自适应止损

### 4.1 V1止损的问题

V1使用固定 -12% 止损阈值。问题：
- 高波动市场（如2018、2020年3月）：很多优质股票正常波动就超过12%，导致频繁错误止损
- 低波动市场（如2023年）：12%的阈值太宽松，不能及时止损

### 4.2 改进：基于个股波动率的自适应止损

```python
class AdaptiveStopLoss:
    """
    自适应止损：止损阈值 = max(固定下限, -k * 个股N日波动率)
    
    - 高波动股票：更宽的止损带，避免正常波动触发止损
    - 低波动股票：更窄的止损带，及时止损
    """
    
    def __init__(
        self,
        vol_multiplier: float = 2.5,     # 止损阈值 = -k * annualized_vol / sqrt(252) * holding_days
        min_threshold: float = -0.08,     # 最窄止损带（即使低波动也不低于-8%）
        max_threshold: float = -0.20,     # 最宽止损带（即使高波动也不超过-20%）
        cooldown_periods: int = 2,
    ):
        self.vol_multiplier = vol_multiplier
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.cooldown = cooldown_periods
        self.entry_prices: Dict[str, float] = {}
        self.entry_vols: Dict[str, float] = {}      # 记录入场时的波动率
        self.cooldown_map: Dict[str, int] = {}
    
    def compute_threshold(self, stock_vol_annual: float) -> float:
        """根据个股年化波动率计算止损阈值"""
        # 假设持仓约10个交易日（双周频），估算合理波动幅度
        holding_period_vol = stock_vol_annual * np.sqrt(10) / np.sqrt(252)
        threshold = -self.vol_multiplier * holding_period_vol
        return np.clip(threshold, self.max_threshold, self.min_threshold)
    
    def update(self, current_holdings, prices, stock_vols=None):
        """
        检查止损。stock_vols: {ts_code: annual_vol} 当前各股的年化波动率。
        """
        stopped_out = set()
        for code in current_holdings:
            if code in self.entry_prices and code in prices:
                ret = prices[code] / self.entry_prices[code] - 1
                # 使用入场时记录的波动率（避免前视偏差）
                vol = self.entry_vols.get(code, 0.30)
                threshold = self.compute_threshold(vol)
                if ret < threshold:
                    stopped_out.add(code)
                    self.cooldown_map[code] = self.cooldown
        
        # 更新冷却期
        expired = [c for c, r in self.cooldown_map.items() if r <= 1]
        for c in expired:
            del self.cooldown_map[c]
        for c in self.cooldown_map:
            if c not in expired:
                self.cooldown_map[c] -= 1
        
        for code in stopped_out:
            self.entry_prices.pop(code, None)
            self.entry_vols.pop(code, None)
        
        return stopped_out
    
    def record_entry(self, code, price, annual_vol):
        if code not in self.entry_prices:
            self.entry_prices[code] = price
            self.entry_vols[code] = annual_vol
```

---

## 5. 改进方案四：市场风格自适应

### 5.1 问题描述

2020年大盘蓝筹跑赢中小盘约30%，V1的固定 `small_cap_bonus=0.02` 反向加码，导致-14.61%超额亏损。

### 5.2 改进：动态风格因子

```python
def compute_style_factor(close_pivot: pd.DataFrame, circ_mv_series: pd.Series) -> float:
    """
    计算大/小盘风格因子：正值偏大盘，负值偏小盘。
    
    方法：过去60个交易日，大盘股中位数收益 vs 小盘股中位数收益的累计差值。
    """
    if len(close_pivot) < 60:
        return 0.0
    
    # 按市值中位数分组
    mv_median = circ_mv_series.median()
    large_caps = circ_mv_series[circ_mv_series >= mv_median].index
    small_caps = circ_mv_series[circ_mv_series < mv_median].index
    
    # 60日累计收益
    ret_60 = close_pivot.iloc[-1] / close_pivot.iloc[-60] - 1
    
    large_ret = ret_60.reindex(large_caps).median()
    small_ret = ret_60.reindex(small_caps).median()
    
    # 正值 = 大盘领先，负值 = 小盘领先
    return large_ret - small_ret

# 根据风格因子调整 small_cap_bonus
style_factor = compute_style_factor(close_pivot, circ_mv_series)

if style_factor > 0.05:
    # 大盘领涨环境，取消小盘加分，甚至给大盘加分
    small_cap_bonus = -0.01  # 轻微偏好大盘
elif style_factor < -0.05:
    # 小盘领涨环境，增加小盘加分
    small_cap_bonus = 0.03
else:
    # 风格均衡
    small_cap_bonus = 0.0
```

---

## 6. 改进方案五：模型训练优化

### 6.1 样本加权：近期样本更重要

```python
def compute_sample_weights(dates: pd.Series, half_life_months: int = 12) -> np.ndarray:
    """
    指数衰减样本权重：近期样本权重更高。
    half_life_months: 权重衰减到一半的月数。
    """
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values
    half_life_days = half_life_months * 21  # 约交易日数
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    return weights / weights.mean()  # 归一化使均值为1
```

在 LightGBM 训练时传入 `weight` 参数：
```python
train_data = lgb.Dataset(
    X_train, label=y_train,
    weight=compute_sample_weights(train_dates),
    feature_name=feature_names,
)
```

### 6.2 增大训练窗口

V1使用3年滚动窗口。在A股市场周期通常为3-5年的背景下，3年窗口可能不包含一个完整的牛熊周期。

改进：训练窗口从3年增大到**4年**，确保模型至少见过一轮完整牛熊：
```python
train_window_years = 4
```

配合样本加权（近期样本权重更高），增大窗口不会导致过拟合远古数据。

### 6.3 重训练频率优化

V1每4期重训练（约2个月）。改为**每6期**（约3个月），减少过拟合风险：
```python
retrain_interval = 6
```

---

## 7. 改进方案六：组合构建优化

### 7.1 单股权重上限调整

V1使用 `max_single_weight=0.08`（8%）。在25只满仓时，平均权重4%，最大8%，集中度适中。但在低仓位（如10只股票）时，平均权重10%，可能导致个股风险暴露过大。

改进：动态调整单股权重上限
```python
# 根据持仓数量动态调整
effective_max_weight = min(0.08, 2.0 / effective_max_positions)
# 10只时：min(0.08, 0.20) = 0.08
# 5只时：min(0.08, 0.40) = 0.08
# 保持不变，但确保在极少持仓时有限制
```

### 7.2 行业约束放宽

V1每行业最多3只。在某些年份（如2021年新能源、2024年AI），单一行业可能贡献大量alpha。

改进：将 `max_per_industry` 从3调整为**4**，同时增加行业数量下限：
```python
max_per_industry = 4
min_industries = 4  # 至少分布在4个行业中

# 在选股循环中增加行业分散约束
if len(selected) >= effective_max and len(industry_count) < min_industries:
    # 如果行业不够分散，继续选不同行业的股票
    continue
```

### 7.3 softmax温度动态调整

V1使用固定 `temperature=5.0`。改进：根据市场状态调整温度，牛市更集中（高温度），熊市更分散（低温度）：

```python
# 牛市集中持仓（信号更可靠），熊市分散持仓（降低风险）
regime_temperature = {
    "bull": 6.0,     # 更集中
    "neutral": 4.0,  # 中等
    "bear": 2.0,     # 更分散（接近等权）
}[market_regime]
```

---

## 8. 完整改进架构

```
┌─────────────────────────────────────────────────────────────┐
│                  lgbm_ensemble_adaptive_v2                    │
├─────────────────────────────────────────────────────────────┤
│  选股层（沿用V1 + 风格自适应改进）                             │
│    ├─ Model A: Smart Money (权重0.6)                         │
│    ├─ Model B: Cross-Sectional (权重0.4)                     │
│    ├─ 共识过滤（综合前5% 且至少一个模型前10%）                  │
│    └─ [新] 动态风格因子替代固定 small_cap_bonus                │
│         大盘领涨时偏好大盘，小盘领涨时偏好小盘                   │
├─────────────────────────────────────────────────────────────┤
│  仓位层（核心改进：波动率目标制 + 多级择时）                     │
│    ├─ [新] 波动率目标制：target_vol=18%，连续平滑调仓           │
│    │   → 自动降低高波动期仓位，无需离散档位                      │
│    ├─ [改] 三级趋势确认（短/中/长期），替代单一 MA20/MA60       │
│    │   → 消除2018年式假突破频繁切换                             │
│    ├─ [新] 广度动量：广度快速恶化时额外降仓                      │
│    └─ 综合仓位 = min(vol_target_scale, regime_modifier)       │
├─────────────────────────────────────────────────────────────┤
│  风控层（改进止损 + 组合优化）                                  │
│    ├─ [改] 自适应止损：阈值 = f(个股波动率)                     │
│    │   → 高波动股宽止损，低波动股窄止损                          │
│    ├─ [改] 行业约束：3只→4只，同时要求至少4个行业                │
│    ├─ [改] softmax温度动态化：牛市集中/熊市分散                  │
│    └─ [保留] 换手控制 buffer_sigma=0.5                        │
├─────────────────────────────────────────────────────────────┤
│  模型层（训练优化）                                            │
│    ├─ [改] 训练窗口 3年→4年                                   │
│    ├─ [新] 指数衰减样本加权（half_life=12个月）                 │
│    └─ [改] 重训练频率 4期→6期                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 具体因子设计

### 9.1 沿用因子（44个因子，双模型不变）

所有44个因子（Model A 22个 + Model B 22个）**完全沿用V1**，不做修改。因子本身不是V1的瓶颈——V1的选股能力（2021年+53%、2022年+8.7%、2025年+41%）已经很好，改进重心在仓位管理和风控。

### 9.2 新增因子：无

不增加新的ML特征因子。V1的因子集已足够丰富（覆盖动量、价值、质量、成长、吸筹、风险6大类），且两个模型已有足够的差异化。新增因子的边际收益有限，反而可能增加过拟合风险。

### 9.3 新增计算指标（非ML因子，仅用于仓位/风控决策）

| 指标 | 计算方式 | 用途 | 所需DB字段 |
|------|---------|------|-----------|
| portfolio_vol | 组合日收益的EWM标准差 × sqrt(252) | 波动率目标制 | 策略内部计算，不需DB字段 |
| breadth_momentum | 广度变化率（当前-5日前） | 广度恶化检测 | close（已有）|
| new_high_pct | 接近60日新高的股票比例 | 长期趋势确认 | high, close（已有）|
| new_low_pct | 接近60日新低的股票比例 | 长期趋势确认 | low, close（已有）|
| style_factor | 大盘vs小盘60日累计收益差 | 风格自适应 | close, circ_mv（已有）|
| stock_vol | 个股20日年化波动率 | 自适应止损 | close（已有）|

**不需要任何新的数据库字段**，全部基于已有数据计算。

---

## 10. 选股逻辑完整流程（V2）

```
每个调仓日执行以下流程：

Step 1: 数据准备（同V1）
  ├─ 从 bulk_data 中提取当日截面数据
  ├─ 过滤：剔除ST/停牌/无效价格
  └─ 市场范围：沪深主板，流通市值前85%

Step 2: 特征计算（同V1）
  ├─ Model A 特征：22个因子，rank归一化
  └─ Model B 特征：22个因子，rank归一化

Step 3: 模型预测（同V1）
  ├─ Model A: score_A = model_A.predict(features_A)
  └─ Model B: score_B = model_B.predict(features_B)

Step 4: 分数融合（改进：动态风格加分）
  ├─ rank_A = rank_pct(score_A)
  ├─ rank_B = rank_pct(score_B)
  ├─ ensemble_score = 0.6 * rank_A + 0.4 * rank_B
  └─ [新] 风格自适应加分：
       ├─ 计算 style_factor = large_cap_60d_ret - small_cap_60d_ret
       ├─ style_factor > 0.05: small_cap_bonus = -0.01（偏大盘）
       ├─ style_factor < -0.05: small_cap_bonus = 0.03（偏小盘）
       └─ 否则: small_cap_bonus = 0.0（中性）

Step 5: 仓位确定（核心改进）
  ├─ [新] 波动率目标制：
  │   ├─ 计算组合近期实际年化波动率
  │   ├─ vol_scale = target_vol / actual_vol（指数平滑）
  │   └─ 限制在 [0.15, 1.20] 范围
  ├─ [改] 三级趋势确认择时：
  │   ├─ Level 1: MA5 vs MA20（短期）
  │   ├─ Level 2: MA20 vs MA60（中期）
  │   ├─ Level 3: 新高/新低比率（长期）
  │   └─ 2个以上级别确认 → bull/bear，否则 neutral
  ├─ [新] 广度动量修正：
  │   └─ breadth_momentum < -0.10 时，regime_modifier × 0.8
  └─ final_scale = min(vol_scale, regime_modifier)
     effective_max = max(3, int(25 * final_scale))

Step 6: 共识过滤（同V1）
  ├─ 综合分排名前5%
  └─ 至少被一个模型评为前10%

Step 7: 止损与冷却处理（改进）
  ├─ [改] 自适应止损：阈值 = f(个股入场时波动率)
  │   ├─ threshold = -2.5 * annual_vol * sqrt(10) / sqrt(252)
  │   └─ clip到 [-0.20, -0.08] 范围
  └─ 冷却期：2个调仓周期

Step 8: 换手控制（同V1）
  └─ 持仓股 ensemble_score += 0.5 * std(ensemble_score)

Step 9: 行业约束 + 数量裁剪（调整）
  ├─ 按 ensemble_score 降序排列
  ├─ [改] 每行业最多4只（原3只）
  ├─ [新] 至少分布在4个行业
  └─ 总数不超过 effective_max

Step 10: 权重计算（改进）
  ├─ [改] softmax温度动态化：
  │   ├─ bull: temperature=6.0
  │   ├─ neutral: temperature=4.0
  │   └─ bear: temperature=2.0
  ├─ 单股权重上限 8%
  └─ 重新归一化

Step 11: 输出目标权重
  └─ return {ts_code: weight}
```

---

## 11. 回测配置建议

```python
cfg = BacktestConfig(
    initial_capital=1_000_000,
    commission_rate=1.5e-4,    # 1.5 bps 单边佣金
    slippage=0.0015,           # 15 bps 滑点
    start_date="2018-01-01",
    end_date="2025-12-31",
    rebalance_freq="BW",       # 双周频（沿用V1）
    db_path="data/quant/data/quant.db",
    baseline_dir="data/quant/baseline",
    output_dir="data/quant/backtest",
)

strategy = LGBMEnsembleAdaptiveV2(
    # 模型参数（改进）
    train_window_years=4,          # 3→4年，覆盖完整牛熊
    retrain_interval=6,            # 4→6期，降低过拟合
    sample_weight_halflife=12,     # 新增：样本权重衰减半衰期（月）
    
    # 集成参数（沿用V1）
    weight_model_a=0.6,
    weight_model_b=0.4,
    consensus_top_pct=0.05,
    consensus_single_top_pct=0.10,
    
    # 仓位参数（核心改进）
    max_positions=25,
    target_vol=0.18,               # 新增：目标年化波动率
    vol_lookback=40,               # 新增：波动率估算窗口
    vol_scale_min=0.15,            # 新增：最低仓位缩放
    vol_scale_max=1.20,            # 新增：最高仓位缩放
    vol_smoothing=0.7,             # 新增：仓位平滑系数
    
    # 权重参数（改进）
    softmax_temperature_bull=6.0,  # 新增：牛市温度
    softmax_temperature_neutral=4.0,
    softmax_temperature_bear=2.0,
    max_single_weight=0.08,
    
    # 风控参数（改进）
    max_per_industry=4,            # 3→4只/行业
    min_industries=4,              # 新增：最少行业数
    stop_loss_vol_multiplier=2.5,  # 新增：止损波动率倍数
    stop_loss_min=-0.08,           # 新增：最窄止损
    stop_loss_max=-0.20,           # 新增：最宽止损
    stop_loss_cooldown=2,
    buffer_sigma=0.5,
    
    # 选股池参数（改进）
    mv_pct_upper=0.85,
    # 去掉固定small_cap_bonus/small_cap_quantile，改为动态
    style_threshold=0.05,          # 新增：风格切换阈值
    feature_lookback=260,
    backtest_end_date="2025-12-31",
)
```

---

## 12. 程序员实现清单

### 12.1 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `strategies/lgbm_ensemble_adaptive_v2.py` | **新建** | V2策略主文件 |

### 12.2 类结构

```python
class LGBMEnsembleAdaptiveV2(StrategyBase):
    """
    V2: 波动率目标制 + 多级择时 + 自适应止损
    基于 V1 改进，核心变化在仓位管理和风控层
    """
    
    def __init__(self, ...):
        super().__init__("lgbm_ensemble_adaptive_v2")
        # V1 所有参数 + V2 新增参数
        # 初始化 AdaptiveStopLoss 替代 StopLossTracker
        # 新增状态变量：
        self._prev_vol_scale = 1.0          # 上一期波动率缩放系数
        self._portfolio_returns = []        # 历史组合收益率（用于波动率估算）
        self._prev_portfolio_value = None   # 上一期组合价值
    
    # ── 新增/修改的内部方法 ──
    
    def _compute_market_regime(self, close_pivot) -> str:
        """[改] 三级趋势确认，返回 "bull"/"neutral"/"bear" """
    
    def _compute_vol_target_scale(self) -> float:
        """[新] 波动率目标制缩放系数"""
    
    def _compute_breadth_momentum(self, close_pivot) -> float:
        """[新] 广度动量"""
    
    def _compute_style_factor(self, close_pivot, circ_mv_series) -> float:
        """[新] 大/小盘风格因子"""
    
    def _compute_sample_weights(self, dates) -> np.ndarray:
        """[新] 指数衰减样本权重"""
    
    # ── 修改的内部方法 ──
    
    def _train_sub_model(self, ...):
        """[改] 增加 sample_weights 传入 LightGBM"""
    
    def _select_and_weight(self, ...):
        """[改] 动态温度 + 行业约束放宽"""
    
    def _compute_market_state(self, close_pivot) -> Tuple[str, float]:
        """[改] 返回 (regime, regime_modifier) 替代原来的单一 coeff"""
    
    # ── 沿用的内部方法（可直接复制V1）──
    
    def _warmup_training_cache(self, ...):  # 仅修改 train_window_years 默认值
    def _append_walk_forward_data(self, ...):  # 不变
    def _compute_ensemble_score(self, ...):  # 修改 small_cap_bonus 为动态
    def _compute_weights(self, ...):  # 修改温度参数为动态
    
    # ── 主入口 ──
    
    def generate_target_weights(self, date, accessor, current_holdings):
        """
        主入口。关键变化：
        1. 在返回权重前，记录组合价值用于下一期波动率计算
        2. 使用 vol_target_scale 和 regime_modifier 综合决定仓位
        3. 传入动态 softmax 温度
        """
```

### 12.3 需要导入/复用的模块

与V1完全相同：

| 来源文件 | 复用内容 | 方式 |
|----------|---------|------|
| `lgbm_smart_money.py` | `FEATURE_COLUMNS`, `FEATURE_NAMES`, `compute_features_from_memory`, `rank_normalize`, `compute_forward_return_from_memory`, `train_lgbm_model` | 直接 import |
| `lgbm_cross_sectional.py` | `FEATURE_COLUMNS`, `FEATURE_NAMES`, `compute_features_from_memory`, `compute_forward_return_from_memory`, `train_lgbm_model` | 直接 import |
| `strategies/utils.py` | `prefetch_bulk_data` | 直接 import |
| `engine/` | `BacktestConfig`, `StrategyBase`, `run_backtest`, `DataAccessor` | 直接 import |

### 12.4 关键实现注意事项

1. **波动率目标制的组合收益率追踪**：每期调仓后需要估算组合的日收益率。由于回测引擎不直接暴露每日净值，策略需要在内部用近似方式追踪：
   ```python
   # 方案A（推荐）：利用 accessor 获取持仓股价格变化
   # 在 generate_target_weights 开始时：
   if self._prev_weights and self._prev_prices:
       # 计算上一期到本期的组合收益
       port_ret = sum(
           w * (current_prices.get(code, prev_price) / prev_price - 1)
           for code, (w, prev_price) in self._prev_holdings_info.items()
       )
       self._portfolio_returns.append(port_ret)
   ```
   
   注意：这是**期间收益率**（双周级别），不是日收益率。需要将其转换为等效日波动率：
   ```python
   # 假设每期约10个交易日
   period_vol = np.std(self._portfolio_returns[-20:])  # 20期 ≈ 约40周
   daily_vol_approx = period_vol / np.sqrt(10)
   annual_vol_approx = daily_vol_approx * np.sqrt(252)
   ```

2. **AdaptiveStopLoss 需要个股波动率**：在 record_entry 时，需要计算入场个股的20日年化波动率：
   ```python
   # 从 bulk_data 中获取该股票近20日的 close 数据
   stock_data = bulk_data[(bulk_data["ts_code"] == code)].tail(20)
   daily_ret = stock_data["close"].pct_change().dropna()
   annual_vol = daily_ret.std() * np.sqrt(252)
   self._stop_loss.record_entry(code, price, annual_vol)
   ```

3. **样本加权传入 LightGBM**：
   ```python
   # 在 _train_sub_model 中
   # 构建训练集时记录每个样本的日期
   sample_dates = []
   for d_str, f_df, labels in valid_cache:
       merged = ...
       sample_dates.extend([pd.Timestamp(d_str)] * len(merged))
   
   sample_dates_series = pd.Series(sample_dates)
   weights = compute_sample_weights(sample_dates_series, half_life_months=self.sample_weight_halflife)
   
   train_data = lgb.Dataset(train_X, label=train_y, weight=weights[:len(train_X)])
   ```

4. **整体代码结构建议**：V2以V1的 `lgbm_ensemble_adaptive.py` 为骨架，约80%代码可复用。主要修改点：
   - `__init__`：新增参数，替换 StopLossTracker 为 AdaptiveStopLoss
   - `_compute_market_state`：重写为三级确认 + 返回 regime 字符串
   - `generate_target_weights`：加入波动率目标制计算、风格因子计算、动态温度
   - `_train_sub_model`：加入样本加权
   - `_select_and_weight`：修改行业约束参数、动态温度

5. **数据库字段**：无需新增任何数据库字段，所有新计算指标均基于 `close`, `high`, `low`, `circ_mv` 等已有字段。

---

## 13. 预期效果分析

| 年份 | V1实际 | V2预期 | 改善来源 |
|------|--------|--------|---------|
| 2018 | -34.67% | -18%~-22% | 波动率目标制+三级择时：高波动率自动大幅降仓，不受假突破影响 |
| 2019 | +36.38% | +30%~38% | 基本维持，波动率降低后可能略减收益 |
| 2020 | +9.46% | +15%~20% | 风格自适应：检测到大盘领涨后取消小盘加分 |
| 2021 | +53.16% | +45%~55% | 基本维持，小盘行情下风格因子自动偏小盘 |
| 2022 | +8.71% | +5%~10% | 自适应止损减少误止损，波动率控制更平滑 |
| 2023 | +6.55% | +8%~12% | 训练窗口加大+样本加权提升IC |
| 2024 | +21.62% | +18%~22% | 基本维持 |
| 2025 | +41.35% | +35%~42% | 基本维持 |

**综合预期**：

| 指标 | V1实际 | V2预期 | 改善 |
|------|--------|--------|------|
| 年化收益 | 14.60% | 15%~18% | 小幅提升（主要来自2018/2020改善） |
| 年化波动率 | 25.66% | 18%~21% | **大幅降低**（波动率目标制核心贡献） |
| 夏普比率 | 0.57 | **0.75~0.95** | **显著提升** |
| 最大回撤 | -34.27% | -20%~-25% | 大幅改善（2018年从-34%降至-20%级别） |
| 胜率 | 57.7% | 58%~62% | 小幅提升 |

---

## 附录A：参数敏感性说明

以下参数需要特别关注其敏感性：

| 参数 | 推荐值 | 敏感区间 | 说明 |
|------|--------|---------|------|
| target_vol | 0.18 | [0.15, 0.22] | 太低会过度减仓损失收益，太高会失去波控效果 |
| vol_smoothing | 0.7 | [0.5, 0.9] | 太低仓位跳变大，太高反应迟钝 |
| vol_multiplier (止损) | 2.5 | [2.0, 3.0] | 太低止损太紧频繁触发，太高形同虚设 |
| train_window_years | 4 | [3, 5] | 太短数据不足，太长引入噪声 |
| style_threshold | 0.05 | [0.03, 0.08] | 太低频繁切换风格，太高反应迟钝 |
