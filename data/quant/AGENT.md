# Quant 项目手册

## 1. 数据拉取

### 1.1 用法

```bash
# 全量下载
python scripts/download_all.py

# 增量追加 (从指定日期开始)
python scripts/download_all.py --since 20260301

# 调整并发线程数 (默认5)
python scripts/download_all.py --workers 10

# 跳过 Wave 1 (stock_basic/trade_cal 已存在时)
python scripts/download_all.py --wave 2

# 组合使用
python scripts/download_all.py --since 20260301 --workers 10 --wave 2
```

### 1.2 输出结构

```
data/quant/data/
├── stock_basic/stock_basic.csv              # 股票列表 (含退市)，全量覆盖
├── calendar/trade_cal.csv                   # 交易日历 2010-2026，全量覆盖
├── daily/{ts_code}.csv                      # 日线 OHLCV，每股一文件
├── daily_basic/{ts_code}.csv                # 每日基本面 (pe/pb/市值/换手率等)，每股一文件
├── adj_factor/{ts_code}.csv                 # 复权因子，每股一文件
├── fina_indicator/{ts_code}.csv             # 财务指标 (roe/roa/eps/bps等)，每股一文件
├── index_weight/{index_code}.csv            # 指数成分股权重 (如 000300.SH)，每指数一文件
├── industry/
│   ├── sw_industry_index.csv                # 申万行业分类 (L1/L2/L3)
│   ├── sw_industry_member.csv               # L1 成分股映射
│   └── sw_industry_member_l2.csv            # L2 成分股映射
└── suspend/suspend_d.csv                    # 停牌记录，单文件
```

### 1.3 依赖

```
tushare, pandas, tqdm
```

## 2. 数据导入 (CSV → SQLite)

### 2.1 用法

```bash
# 全量重建
python scripts/build_db.py

# 增量导入 (只处理某日期之后的交易日)
python scripts/build_db.py --since 20260301
```

### 2.2 数据库表结构

`data/quant/data/quant.db` 包含 4 张表：

| 表名 | 主键 | 说明 |
|------|------|------|
| `stock_daily` | (ts_code, trade_date) | 宽表：价格+基本面+复权因子+财务指标+行业 |
| `industry_info` | industry_code | 行业代码 → 名称 (申万 L1/L2/L3) |
| `stock_info` | ts_code | 股票代码 → 名称/上市日期等 |
| `index_weight` | (index_code, trade_date, con_code) | 指数成分股权重（如沪深300、中证500） |

### 2.3 前视偏差处理

- **财务指标 (fina_indicator)**：以 `ann_date` (公告日) 的**下一个交易日**为生效日，向前填充至下次公告
- **行业分类**：按 `in_date / out_date` 时间范围匹配

---

## 3. 回测框架 (`engine/`)

### 3.1 目录结构

```
engine/
├── __init__.py          # 公开 API: BacktestConfig, StrategyBase, DataAccessor, LookAheadError, run_backtest
├── config.py            # 回测参数配置（dataclass）
├── data_loader.py       # 懒加载数据访问器 DataAccessor + 交易日历 / 基准加载
├── strategy_base.py     # 策略抽象基类
├── broker.py            # 模拟券商（订单执行 / 资金管理）
├── backtest.py          # 回测主循环引擎
├── analyzer.py          # 绩效分析、CSV/Markdown 输出
└── examples/
    └── equal_weight_demo.py   # 示例策略：等权 Top-N
```

### 3.2 运行方式

```bash
# 运行示例策略
python -m data.quant.engine.examples.equal_weight_demo
```

### 3.3 回测配置 (`BacktestConfig`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_capital` | float | 1,000,000 | 初始资金 |
| `commission_rate` | float | 1.5e-4 | 单边佣金率（万1.5） |
| `slippage` | float | 0.0 | 滑点比例（买入加价、卖出减价） |
| `lot_size` | int | 100 | 整手股数（A 股 = 100） |
| `start_date` | str | "2010-01-01" | 回测起始日 |
| `end_date` | str | "2026-12-31" | 回测结束日 |
| `rebalance_freq` | str | "M" | 调仓频率：M=月末 / W=周末 / Q=季末 / D=每日 |
| `baseline_dir` | str | "data/quant/baseline" | 基准目录，自动扫描目录下所有 `.csv` 文件作为基准 |
| `db_path` | str | "data/quant/data/quant.db" | SQLite 数据库路径 |
| `output_dir` | str | "data/quant/backtest" | 输出目录 |

### 3.4 策略编写

继承 `StrategyBase`，实现两个方法：

```python
from engine import BacktestConfig, StrategyBase, run_backtest

class MyStrategy(StrategyBase):
    def __init__(self):
        super().__init__("my_strategy")  # 策略名（用于文件命名）

    def describe(self) -> str:
        """可选：返回策略描述，写入 report.md"""
        return "这是一个动量轮动策略..."

    def generate_target_weights(self, date, accessor, current_holdings):
        """
        必须实现：返回 {ts_code: weight} 目标权重字典。
        - 权重自动归一化，无需手动确保 sum=1
        - 不在字典中的持仓会被卖出
        - 返回空字典 = 清仓
        """
        snap = accessor.get_date(date, columns=["close", "circ_mv"])
        top = snap.nlargest(30, "circ_mv")
        return {row.ts_code: 1/30 for row in top.itertuples()}

cfg = BacktestConfig(start_date="2020-01-01", end_date="2025-12-31")
result = run_backtest(MyStrategy(), cfg)
```

### 3.5 DataAccessor API

懒加载数据访问器，全程只持有一个 SQLite 连接，按需查询。

| 方法 | 说明 | 典型用途 |
|------|------|----------|
| `get_date(date, columns=None)` | 查询单日全市场快照 | 选股 |
| `get_prices(date)` | 返回 `{ts_code: close}` 字典 | 引擎内部估值 |
| `get_window(end_date, lookback, ts_codes=None, columns=None)` | 回溯 N 个交易日的数据 | 动量/均线策略 |
| `get_stocks_on_date(date, ts_codes, columns=None)` | 查询指定股票在指定日期的数据 | 精确查询 |
| `get_index_weights(date, index_code="000300.SH")` | 查询指定日期（或之前最近一次）的指数成分股权重 | 指数增强基准对齐 |

**注意**：`columns` 参数可指定只查询需要的字段，进一步节省内存。

### 3.6 前视偏差防护

引擎在每个调仓日自动调用 `accessor.set_current_date(rebal_date)`。此后策略通过 accessor 访问数据时，若请求的日期 > 当前模拟日期，会抛出 `LookAheadError`：

```
LookAheadError: 前视偏差检测: get_date() 试图访问 2024-02-28 的数据,
但当前模拟日期为 2024-01-31。策略不允许访问未来数据。
```

### 3.7 交易执行规则

- **整手约束**：买卖股数 `(shares // 100) * 100`，不足一手的部分丢弃
- **单边佣金**：买卖各收 `amount × commission_rate`
- **滑点**：买入价 `price × (1 + slippage)`，卖出价 `price × (1 - slippage)`
- **资金不足处理**：自动降档到可负担的最大手数；仍不足一手则订单 REJECTED
- **调仓顺序**：先执行全部 SELL（释放资金），再执行 BUY

### 3.8 输出产物

回测完成后，在 `backtest/` 目录下生成以下文件（格式遵循 `OUTPUT.md` 规范）：

| 文件 | 说明 |
|------|------|
| `{strategy}_nav.csv` | 累计净值曲线（策略 + 所有基准），初始 = 1.0 |
| `{strategy}_monthly_returns.csv` | 每期收益明细（port_ret / n_stocks / 每个基准的 bench_ret_{name} / excess_{name}） |
| `{strategy}-trade.csv` | 全部成交记录（含 REJECTED），字段见下表 |
| `{strategy}_report.md` | 策略报告（策略概述 + 回测配置 + 绩效汇总 + 分年度表现 + 总结） |

成交记录字段：

| 字段 | 说明 |
|------|------|
| `date` | 交易日期 |
| `ts_code` | 股票代码 |
| `direction` | BUY / SELL |
| `price` | 成交价（含滑点） |
| `shares` | 成交股数 |
| `amount` | 成交金额 |
| `commission` | 佣金 |
| `status` | FILLED / REJECTED |
| `reason` | 拒绝原因（如"资金不足"） |

### 3.9 回测主循环流程

```
[1/5] 加载交易日历 → 确定调仓日序列
[2/5] 初始化 DataAccessor（懒加载，不读全量数据）
[3/5] 加载基准数据（小 CSV，安全驻留内存）
[4/5] 运行主循环:
       对每个调仓日:
         ① set_current_date → 激活前视偏差守卫
         ② get_prices → 按需查询当日价格
         ③ strategy.generate_target_weights → 策略生成目标权重
         ④ broker.rebalance → 先卖后买，执行订单
         ⑤ 记录净值快照 & 成交记录
[5/5] 生成 CSV / Markdown 报告，打印绩效汇总表
```

---

## 4. 通用因子框架 (`engine/factors.py`)

### 4.1 架构概览

```
Factor (ABC)                      # 抽象基类：compute(date, accessor) → Series
├── CrossSectionalFactor          # 截面因子：只需当日快照
│   └── compute_from_snapshot()
├── TimeSeriesFactor              # 时序因子：需要回看窗口
│   └── compute_from_window()
└── FactorEngine                  # 编排器：winsorize → z-score → composite
```

### 4.2 内置因子一览

共 **23 个** 内置因子，覆盖 7 大类：

| 大类 | 因子类名 | `name` 属性 | 原始指标 | 方向 | 所需 DB 字段 |
|------|---------|------------|---------|------|-------------|
| **动量** | `Momentum(lookback)` | `momentum_{N}d` | N 日收益率 | ↑ 越高越好 | `close` |
| | `Reversal(lookback)` | `reversal_{N}d` | N 日收益率（反转） | ↓ 越低越好 | `close` |
| **价值** | `ValueBP()` | `value_bp` | 1/PB | ↑ 越高越便宜 | `pb` |
| | `ValueEP()` | `value_ep` | 1/PE_TTM | ↑ | `pe_ttm` |
| | `ValueSP()` | `value_sp` | 1/PS_TTM | ↑ | `ps_ttm` |
| | `ValueDP()` | `value_dp` | 股息率 TTM | ↑ | `dv_ttm` |
| | `ValueCFTP()` | `value_cftp` | CFPS/close | ↑ | `cfps`, `close` |
| **质量** | `QualityROE()` | `quality_roe` | ROE | ↑ | `roe` |
| | `QualityROA()` | `quality_roa` | ROA | ↑ | `roa` |
| | `QualityGrossMargin()` | `quality_gross_margin` | 毛利率 | ↑ | `grossprofit_margin` |
| | `QualityNetMargin()` | `quality_net_margin` | 净利率 | ↑ | `netprofit_margin` |
| | `QualityCurrentRatio()` | `quality_current_ratio` | 流动比率 | ↑ | `current_ratio` |
| | `QualityDebtToAssets()` | `quality_low_leverage` | 资产负债率 | ↓ 越低越安全 | `debt_to_assets` |
| | `QualityAssetTurnover()` | `quality_asset_turnover` | 总资产周转率 | ↑ | `assets_turn` |
| **成长** | `GrowthRevenue()` | `growth_revenue` | 营收同比增长 | ↑ | `tr_yoy` |
| | `GrowthProfit()` | `growth_profit` | 营业利润同比 | ↑ | `op_yoy` |
| | `GrowthEquity()` | `growth_equity` | 净资产同比 | ↑ | `equity_yoy` |
| | `GrowthEarnings()` | `growth_earnings` | 利润总额同比 | ↑ | `ebt_yoy` |
| **风险** | `Volatility(lookback)` | `volatility_{N}d` | 日收益率标准差 | ↓ 越低越好 | `close` |
| | `TurnoverStability(lookback)` | `turnover_stability_{N}d` | 换手率变异系数 | ↓ | `turnover_rate_f` |
| **技术** | `Illiquidity(lookback)` | `illiquidity_{N}d` | Amihud ILLIQ | ↓ 默认偏好流动性 | `pct_chg`, `amount` |
| | `VolumePriceDivergence(lookback)` | `vol_price_corr_{N}d` | 量价相关性 | ↑ | `pct_chg`, `vol` |
| **规模** | `SizeLogMV()` | `size_log_mv` | ln(流通市值) | ↓ 默认偏好小盘 | `circ_mv` |

> **方向说明**：↑ 表示原始值越高越好（`ascending=False`）；↓ 表示原始值越低越好（`ascending=True`）。FactorEngine 内部会自动根据方向翻转 z-score 符号。

### 4.3 FactorEngine 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `factors` | `List[Factor]` | *必填* | 参与打分的因子列表 |
| `winsorize_pct` | `(float, float)` 或 `None` | `(0.01, 0.99)` | 缩尾处理分位数。设为 `None` 跳过 |
| `neutralize_by` | `str` 或 `None` | `"sw_l1"` | 行业中性化列名。`None` 则用全局 z-score |
| `min_industry_size` | `int` | `5` | 行业内股票数 < 此值时整组剔除 |
| `universe_filter` | `Dict[str, Callable]` 或 `None` | `None` | 自定义选股池过滤条件 |

### 4.4 FactorEngine 处理流程

```
① 各因子独立 compute() → 原始值 Series
② 加载当日快照，过滤选股池（剔除停牌、无价格）
③ inner join：只保留选股池内且所有因子均有值的股票
④ 行业过滤：剔除人数不足 min_industry_size 的行业
⑤ Winsorize：各因子按 [1%, 99%] 分位数缩尾
⑥ Z-Score：行业内标准化（或全局标准化）
⑦ 方向调整：ascending=True 的因子 z-score 取负
⑧ 综合评分：composite = Σ(weight × z_score) / Σ(weight)
⑨ 按 composite 降序排列返回 DataFrame
```

返回 DataFrame 列结构：`ts_code, [sw_l1], z_factor1, z_factor2, ..., composite`

### 4.5 基本用法

#### 4.5.1 在策略中使用 FactorEngine

```python
from engine import (
    BacktestConfig, StrategyBase, run_backtest,
    FactorEngine, ValueBP, QualityROE, Momentum,
)

class MyMultiFactorStrategy(StrategyBase):
    def __init__(self):
        super().__init__("my_mf_strategy")
        self.engine = FactorEngine(
            factors=[
                ValueBP(weight=1.0),
                QualityROE(weight=1.0),
                Momentum(lookback=20, weight=1.0),
            ],
            winsorize_pct=(0.01, 0.99),
            neutralize_by="sw_l1",
        )

    def generate_target_weights(self, date, accessor, current_holdings):
        scores = self.engine.run(date, accessor)
        if scores.empty:
            return {}
        top = scores.head(30)
        w = 1.0 / len(top)
        return {row.ts_code: w for row in top.itertuples()}
```

#### 4.5.2 使用便捷函数快速组合

```python
from engine.factors import (
    FactorEngine, value_factors, quality_factors,
    momentum_factors, Volatility,
)

engine = FactorEngine(
    factors=[
        *value_factors(weight=2.0),      # BP + EP + SP + DP, 各 weight=2.0
        *quality_factors(weight=1.5),     # ROE + ROA + 毛利率 + 低杠杆
        *momentum_factors(20, weight=1.0),# 20日动量 + 5日反转
        Volatility(20, weight=0.5),       # 低波因子
    ],
)
```

#### 4.5.3 自定义选股池过滤

```python
engine = FactorEngine(
    factors=[ValueBP(), QualityROE()],
    universe_filter={
        # 只选流通市值 > 100 亿的大盘股
        "circ_mv": lambda s: s > 1_000_000,
        # 只选换手率 > 0.5% 的活跃股
        "turnover_rate_f": lambda s: s > 0.5,
    },
)
```

#### 4.5.4 关闭行业中性化（全局 z-score）

```python
engine = FactorEngine(
    factors=[Momentum(60), ValueEP()],
    neutralize_by=None,  # 不做行业中性，直接全局 z-score
)
```

### 4.6 自定义因子

#### 4.6.1 截面因子（只需当日数据）

```python
from engine.factors import CrossSectionalFactor

class MyPEGFactor(CrossSectionalFactor):
    """PEG = PE / 盈利增速。越低越好。"""

    def __init__(self, weight=1.0):
        super().__init__(
            name="peg",
            columns=["pe_ttm", "ebt_yoy"],  # 所需 DB 字段
            ascending=True,                   # 越低越好
            weight=weight,
        )

    def compute_from_snapshot(self, snap):
        pe = snap["pe_ttm"]
        growth = snap["ebt_yoy"]
        peg = pe / growth.replace(0, float("nan"))
        peg[peg < 0] = float("nan")  # 负值无意义
        return peg.rename(self.name)
```

#### 4.6.2 时序因子（需要回看窗口）

```python
from engine.factors import TimeSeriesFactor

class MeanReversion(TimeSeriesFactor):
    """价格偏离 N 日均线的幅度。越低 = 越超跌 = 均值回复机会。"""

    def __init__(self, lookback=20, weight=1.0):
        super().__init__(
            name=f"mean_reversion_{lookback}d",
            lookback=lookback,
            columns=["close"],
            ascending=True,  # 越低于均线越好
            weight=weight,
        )

    def compute_from_window(self, window, date):
        pivot = window.pivot(
            index="trade_date", columns="ts_code", values="close"
        )
        ma = pivot.mean()
        last = pivot.iloc[-1]
        deviation = (last - ma) / ma
        return deviation.rename(self.name)
```

#### 4.6.3 在策略中使用自定义因子

```python
engine = FactorEngine(
    factors=[
        MyPEGFactor(weight=2.0),
        MeanReversion(20, weight=1.0),
        QualityROE(weight=1.0),
    ],
)
```

### 4.7 因子注册表 `ALL_FACTORS`

`ALL_FACTORS` 是一个 `Dict[str, Type[Factor]]` 字典，可用于枚举所有内置因子：

```python
from engine.factors import ALL_FACTORS

for key, cls in ALL_FACTORS.items():
    print(f"{key:25s} → {cls.__name__}")
```

输出：

```
momentum                  → Momentum
reversal                  → Reversal
value_bp                  → ValueBP
value_ep                  → ValueEP
...（共 23 个）
```

### 4.8 Factor 基类属性

每个 Factor 实例都有以下属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 因子名，作为 DataFrame 列名 |
| `ascending` | `bool` | `True` = 原始值越低越好；`False` = 越高越好 |
| `weight` | `float` | 在 composite 中的权重 |
| `columns` | `List[str]` | （截面/时序因子）所需的 DB 字段列表 |
| `lookback` | `int` | （仅时序因子）回看窗口天数 |

---

## 5. 批量数据 Parquet 缓存

### 5.1 背景

策略的 `prefetch_bulk_data()` 函数需要从 SQLite 一次性加载数百万行数据（如 2018~2026 约 888 万行），首次 SQL 查询耗时 ~160s。为避免每次运行都重复执行相同的慢查询，引入了基于 Parquet 的文件系统缓存。

### 5.2 缓存位置

```
data/quant/data/.cache/
├── bulk_20140630_20171231_a1b2c3d4_1711234567.parquet   # 训练窗口
├── bulk_20171231_20260101_a1b2c3d4_1711234567.parquet   # 回测窗口
└── ...
```

该目录已加入 `.gitignore`，不会提交到版本库。

### 5.3 缓存 Key 设计

文件名格式：`bulk_{start}_{end}_{col_hash}_{db_mtime}.parquet`

| 组成部分 | 来源 | 作用 |
|----------|------|------|
| `start` / `end` | 日期范围 `YYYYMMDD` | 日期范围变化 → 缓存失效 |
| `col_hash` | `md5(FEATURE_COLUMNS)[:8]` | 特征列变化 → 缓存失效 |
| `db_mtime` | `int(quant.db.stat().st_mtime)` | 数据库更新（如 `build_db.py` 增量导入）→ 缓存失效 |

任何一个组成部分变化，都会生成不同的文件名，旧缓存自然失效（不会被命中）。

### 5.4 缓存匹配策略

缓存查找采用**两级匹配**机制：

1. **精确匹配**：请求的 `[start, end]` 与缓存文件名中的日期范围完全一致 → 直接加载
2. **超集匹配**：扫描 `.cache/` 目录下同 `col_hash` + `db_mtime` 的所有缓存文件，如果某个缓存的日期范围 `[cache_start, cache_end]` 完全覆盖请求范围（即 `cache_start <= start` 且 `cache_end >= end`），则加载该缓存并按 `trade_date` 切片过滤

这意味着：如果之前已缓存了 `2014~2026` 的全量数据，后续请求 `2018~2025` 的子区间时无需重新查询 SQL，直接从超集缓存中切片即可。

### 5.5 工作流程

```
prefetch_bulk_data(accessor, start_date, end_date)
│
├─ 计算 cache_file 路径 (精确匹配文件名)
├─ 精确匹配命中？
│   ├─ YES → pd.read_parquet(cache_file) → 返回 (~2-3s)
│   └─ NO  → 扫描 .cache/ 目录，寻找超集缓存
│            ├─ 超集命中？
│            │   ├─ YES → pd.read_parquet(superset) → 按日期切片 → 返回 (~3-5s)
│            │   └─ NO  → SQL 查询 → 构建 DataFrame → 写入 .parquet → 返回 (~110-160s)
```

### 5.6 涉及文件

| 文件 | 说明 |
|------|------|
| `strategies/utils.py` | **公共模块**，`prefetch_bulk_data()` 的唯一实现 |
| `strategies/lgbm_cross_sectional.py` | v4 策略，调用 `utils.prefetch_bulk_data()` |
| `strategies/lgbm_cross_sectional_v5.py` | v5 策略，同上 |
| `strategies/lgbm_cross_sectional_v6.py` | v6 策略，同上 |
| `strategies/lgbm_cross_sectional_v7.py` | v7 策略，同上 |

`prefetch_bulk_data()` 已从各策略文件中提取到 `strategies/utils.py`，接受 `feature_columns` 参数。各策略通过 `from strategies.utils import prefetch_bulk_data` 导入，调用时传入各自的 `FEATURE_COLUMNS`（因各版本特征列可能不同，`col_hash` 会自动区分缓存文件）。

### 5.6 性能对比

| 场景 | 耗时 | 说明 |
|------|------|------|
| 首次运行（无缓存） | ~110-160s | SQL 查询 + 写入 Parquet（写入本身 < 5s） |
| 后续运行（命中缓存） | **~2-3s** | 直接读取 Parquet，跳过 SQL |

### 5.7 注意事项

- **手动清理**：旧的缓存文件不会自动删除。如磁盘空间紧张，可手动清空 `data/quant/data/.cache/` 目录
- **数据库更新后**：`build_db.py` 增量导入会改变 `quant.db` 的修改时间，缓存自动失效，下次运行会重新生成
- **依赖 pyarrow**：Parquet 读写依赖 `pyarrow` 库，需确保已安装（`pip install pyarrow`）
