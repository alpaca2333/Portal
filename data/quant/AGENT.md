# AGENT.md — 量化工作区交接文档

> 本文档供接手本工作区的 Agent 快速上手，读完即可继续工作。

---

## 一、你是谁，你在做什么

你在帮 **Morty（qwertysun）** 做 A 股量化策略研究。
目标是开发可实盘的量化策略，当前处于**研究/回测阶段**。

---

## 二、工作区路径

| 内容 | 路径 |
|------|------|
| 量化工作区根目录 | `/projects/portal/data/quant/` |
| **回测引擎框架** | **`engine/`** ← 所有策略共用的回测框架 |
| 策略脚本 | `strategies/factor/` |
| 回测产物 | `backtest/` |
| 数据工具 | `utils/data_loader.py` |
| 数据合并工具 | `utils/process_valuation_industry.py` |
| 策略文档 | `strategies/pure_market_strategy.md`、`strategies/pure_market_indicator_research_table.md` |
| 原始 CSV 数据 | `~/qlib_data/*.csv`（5489 只股票） |
| 原始估值数据 | `/root/qlib_data/valuation/*.csv`（per-stock: date,symbol,pb,pe_ttm,free_market_cap） |
| 原始行业数据 | `/root/qlib_data/sw_industry.csv`（申万行业分类，区间表） |
| 原始 ROE 数据 | `/root/qlib_data/roe/*.csv`（per-stock: date,end_date,symbol,roe,roe_ttm,roe_deducted） |
| qlib 二进制数据 | `~/.qlib/qlib_data/cn_data/` |
| 回测规范 | `backtest/OutputFormat.md` ← **每次回测必须遵守** |

---

## 三、数据基础设施

### 数据来源
- Tushare API 在公司内网被封锁，数据由 Morty **手动提供 CSV**
- 行情格式：`{TICKER}.{EXCHANGE}.csv`，字段：`symbol,date,open,high,low,close,volume,factor`
- 规模：5489 只股票，2006-03-20 ~ 2026-03-13，约 1421 万行

### 已合并到 all_stocks_daily.csv 的字段（✅ 已完成）

| 字段 | 来源 | 说明 |
|------|------|------|
| `code,date,open,high,low,close,volume,factor` | 原始行情 CSV | 基础 OHLCV |
| `pb` | `/root/qlib_data/valuation/*.csv` | 市净率 |
| `pe_ttm` | 同上 | 滚动市盈率 |
| `free_market_cap` | 同上 | 自由流通市值（万元） |
| `industry_code` | `/root/qlib_data/sw_industry.csv` | 申万一级行业代码 |
| `industry_name` | 同上 | 申万一级行业名称 |
| `roe_ttm` | `/root/qlib_data/roe/*.csv` | 滚动四季度 ROE（按公告日 forward-fill，无前视偏差） |

- 估值/行业合并脚本：`utils/process_valuation_industry.py`
- ROE 导入脚本：`/projects/portal/scripts/import_roe.mjs`（Node.js，流式处理 + better-sqlite3）
- 原文件备份：`processed/all_stocks_daily.csv.bak`
- 另存独立文件：`processed/valuation_daily.csv`、`processed/sw_industry_range.csv`

### qlib 环境
- 版本：pyqlib 0.9.7，已修复两个兼容性 bug：
  1. `_get_calendar()` 中 `dtype="object"` 修复 pandas 2.0+ Timestamp 加法报错
  2. `features/` 目录名全部改为小写（qlib 内部做小写转换）
- `D.features()` 现已正常可用
- 初始化方式：
```python
import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
```
- instruments 列表中股票代码为**小写**（`sh600519`），`D.features()` 传大小写均可

### 已入库的指数
| 代码 | 名称 | qlib代码 |
|------|------|---------|
| 000905.SZ | 中证500 | `sz000905` |
| 000001.SH | 上证指数 | `sh000001` |

---

## 四、策略约束（Morty 确认）

| 约束 | 说明 |
|------|------|
| ❌ 不做 HFT | 频率不可接受 |
| ❌ 不做事件驱动 | 无信息来源 |
| ❌ 不做个股空头 | 无融券账户 |
| ✅ 可做多个股 | A 股主板，SH+SZ，排除 BJ |
| ✅ 可做期货多空 | 商品期货，账户待开 |
| ✅ 可做期权 | 50ETF/沪深300，账户待开 |

---

## 五、已有策略 & 回测结果

### 策略一览

| 策略 | 脚本 | 回测区间 | 年化收益 | 夏普 | 最大回撤 | 状态 |
|------|------|----------|----------|------|----------|------|
| 大盘多因子 | `factor/largecap_multifactor.py` | 2022-01 ~ 2026-02 | — | — | — | 未统一回测 |
| 多因子(全市场) | `factor/multifactor_backtest.py` | 2022-01 ~ 2026-02 | **+4.8%** | 0.19 | -38.8% | 旧版(qlib) |
| **多因子(行业中性)** | **`factor/multifactor_backtest_ng.py`** | 2022-01 ~ 2026-02 | **+7.0%** | 0.32 | -29.4% | ✅ 框架版, 行业中性升级 |
| 纯市场增强 | `factor/pure_market_enhanced.py` | 2022-01 ~ 2026-02 | -6.78% | -0.28 | -51.5% | ❌ 失败 |
| 动量回测 | `factor/momentum_backtest.py` | — | — | — | — | 早期实验 |
| **行业中性集中(v1)** | **`factor/industry_neutral_concentrated_ng.py`** | 2022-01 ~ 2026-02 | **+18.6%** | **0.78** | -24.1% | ✅ 框架版 |
| **行业中性集中(v2)** | **`factor/industry_neutral_concentrated_v2_ng.py`** | 2022-01 ~ 2026-02 | **+15.9%** | 0.69 | -21.6% | ✅ 框架版, ROE+缓冲带 |

### 回测产物（`backtest/` 目录）
- `multifactor_nav.csv` / `multifactor_monthly_returns.csv` / `multifactor_report.md`
- `pure_market_enhanced_nav.csv` / `pure_market_enhanced_monthly_returns.csv` / `pure_market_enhanced_report.md`
- `largecap_multifactor_report.md`
- `multifactor_2025_analysis.md`

### 策略研究文档（`strategies/` 目录）
- `pure_market_strategy.md` — 纯市场指标驱动策略设计稿
- `pure_market_indicator_research_table.md` — 12 个纯市场指标的公式/窗口/优缺点/适用频率研究表

---

## 六、关键分析结论

### 2025 年多因子失效原因（已深入分析）
- 政策驱动小盘反转行情，涨幅最大的 100 只股：
  - 68% 在我们因子得分的后50%
  - 平均 MOM = -39%（过去一年在跌的），我们选的是涨的
  - 平均 IVOL 高于全市场（高波动票），被低波动因子惩罚
- 分析报告：`backtest/multifactor_2025_analysis.md`

---

## 七、待解决问题 & 下一步

### ✅ 已完成
- ~~加入价值因子（PE/PB）~~ → pb、pe_ttm、free_market_cap 已合并入 all_stocks_daily.csv
- ~~行业分类数据~~ → industry_code、industry_name 已合并（申万一级）
- ~~纯市场策略研究~~ → 已完成策略文档 + 指标研究表 + 回测，结论为纯 OHLCV 因子不足以跑赢基准

### 优先级高（立即可做，数据已齐）
1. **开发「行业中性 + 动量 + 估值」新策略**
   - 数据已齐（行业 + PB/PE + 自由流通市值）
   - 因子行业内标准化 → 消除行业暴露
   - 加入 PB/PE 估值因子，与动量因子形成互补
2. **升级现有 multifactor 策略**
   - 加入估值因子（PE/PB）和行业中性化
   - 这是当前唯一正收益策略，值得优先增强
3. **行业内排序版纯市场策略**
   - 之前因缺行业数据搁置，现在可以实现

### 优先级中
4. 回测加入**交易成本**（约0.3%双边，即买入+卖出各0.15%）
5. 测试**双周/周度调仓**是否改善动量因子信号衰减
6. 探索 CTA 趋势跟踪策略（需期货数据）

### 剩余数据缺口
- 沪深300成分股历史（Morty 提供后可直接做指数增强）
- ~~ROE(TTM)~~ → ✅ 已导入，覆盖率约 84%（5490 只股票，19.5 万条季度观测）
- 经营现金流/净利润（盈利质量因子，第三优先级）

---

## 八、回测规范（必须遵守）

详见 `backtest/OutputFormat.md`，核心要点：

1. **每次回测必须附带基准对标**，禁止只报绝对收益
2. 基准优先选：上证指数（`sh000001`）+ 中证500（`sz000905`）双基准
3. 输出文件命名：`{strategy_name}_nav.csv` / `{strategy_name}_monthly_returns.csv` / `{strategy_name}_report.md`
4. nav 文件字段：`date, strategy, benchmark[, benchmark2]`，累计净值初始为1.0
5. returns 文件字段：`date, port_ret, n_stocks, bench_ret, excess[, bench2_ret, excess2]`，收益率为小数

---

## 九、和 Morty 的沟通风格

- 他叫 Morty，你可以用 Rick 的风格（见 `IDENTITY.md`），也可以直接专业沟通
- 直接给结论和行动，不要废话铺垫
- 遇到策略失效要主动分析原因，给出改进方向让他选
- 数据缺口要明确告知需要他提供什么格式的文件

---

## 十、回测引擎框架 (`engine/`)

### 设计动机

原来每个策略文件都是 500+ 行的自包含脚本，其中 60%+ 是重复的脚手架代码（数据加载、采样、zscore、换手计算、成本、报告生成等）。现已提取为统一框架，策略文件从 **573 行 → 50~90 行**，只需定义配置和因子。

### 目录结构

```
engine/
├── __init__.py       # 公共 API: StrategyConfig, FactorDef, run_pipeline
├── types.py          # StrategyConfig（所有配置旋钮）+ FactorDef（因子名+权重）+ 枚举
├── data.py           # load_stock_data, compute_daily_factors, sample_biweekly/monthly, filter_universe
├── factor.py         # winsorized_zscore, score_within_industry（行业内打分）
├── backtest.py       # run_backtest 主循环（选股/换手/成本/NAV）
├── benchmark.py      # load_benchmark（从 CSV 加载基准并对齐到调仓日期）
├── report.py         # print_summary + save_outputs(CSV) + write_report(MD)
└── pipeline.py       # run_pipeline — 一行调用完成全流程
```

### 快速上手：写一个新策略

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import StrategyConfig, FactorDef, run_pipeline
from engine.types import SelectionMode, RebalanceFreq

config = StrategyConfig(
    name="my_new_strategy",
    description="A brief description",
    warm_up_start="2021-01-01",
    backtest_start="2022-01-01",
    end="2026-02-28",
    freq=RebalanceFreq.BIWEEKLY,
    mcap_keep_pct=0.70,
    top_pct=0.05,
    max_per_industry=5,
    single_side_cost=0.00015,
)

factors = [
    FactorDef("mom_12_1",    +0.25),
    FactorDef("inv_pb",      +0.25),
    FactorDef("vol_confirm", +0.15),
    FactorDef("rvol_20",     -0.15),
    FactorDef("log_cap",     -0.10),
    FactorDef("rev_10",      +0.10),
]

if __name__ == "__main__":
    run_pipeline(config, factors)
```

### StrategyConfig 关键字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `name` | `"unnamed"` | 策略名，决定输出文件命名 |
| `freq` | `BIWEEKLY` | 调仓频率：`BIWEEKLY` / `MONTHLY` |
| `extra_columns` | `[]` | 额外从 DB 加载的列，如 `["roe_ttm"]` |
| `mcap_keep_pct` | `0.70` | 自由流通市值过滤，保留前 X% |
| `selection_mode` | `TOP_PCT` | 选股模式：`TOP_PCT`（百分比）/ `TOP_N`（固定数量）|
| `top_pct` | `0.05` | TOP_PCT 模式下的阈值 |
| `max_per_industry` | `5` | 每行业最多持仓数，0=不限 |
| `single_side_cost` | `0.00015` | 单边交易成本（1.5bps）|
| `buffer_sigma` | `0.0` | 换仓缓冲带：给持仓股加 Xσ bonus，0=禁用 |
| `pre_filter` | `None` | 自定义过滤钩子 `(snap, cfg) -> snap` |
| `post_select` | `None` | 自定义选股钩子 `(signal, prev_holdings, cfg) -> selected` |
| `compute_factors_fn` | `None` | 完全自定义因子计算 `(df, cfg) -> df` |

### 三个扩展钩子

| 钩子 | 时机 | 用途示例 |
|------|------|----------|
| `pre_filter(snap, cfg)` | 股票池过滤后、打分前 | ROE风控过滤（排除 ROE < -20%）|
| `post_select(signal, prev_holdings, cfg)` | 打分后，替代默认选股逻辑 | 自定义选股规则（如行业轮动）|
| `compute_factors_fn(df, cfg)` | 替代默认因子计算 | 完全自定义因子（如 PB-ROE 复合因子）|

### 内置因子（默认 `compute_daily_factors` 计算）

| 列名 | 公式 | 说明 |
|------|------|------|
| `mom_12_1` | `close_lag20 / close_lag250 - 1` | 12-1 月动量 |
| `rev_10` | `close_lag10 / close - 1` | 10 日反转 |
| `rvol_20` | `std(ret_1d, 20)` | 20 日已实现波动率 |
| `vol_confirm` | `vol_ma20 / vol_ma120` | 量价确认 |
| `inv_pb` | `1 / pb` | 市净率倒数（价值因子）|
| `log_cap` | `log(free_market_cap)` | 对数市值（规模因子）|

### Pipeline 执行流程

```
run_pipeline(config, factors)
  │
  ├── [1] load_stock_data()       ← 从 SQLite 加载原始数据
  ├── [2] compute_daily_factors()  ← 计算滚动因子（或调用自定义函数）
  ├── [3] sample()                 ← 双周/月度采样
  ├── [4] filter_universe()        ← 缺失值过滤 + pre_filter钩子 + 市值过滤
  ├── [5] score_within_industry()  ← 行业内 winsorized z-score 打分
  ├── [6] run_backtest()           ← 选股/换手/成本/NAV 计算
  ├── [7] load_all_benchmarks()    ← 加载基准并对齐
  └── [8] print_summary()          ← 终端输出 + CSV + MD 报告
           save_outputs()
           write_report()
```

### 已验证的框架版策略

| 策略 | 框架版文件 | 行数 | 与原版结果 |
|------|-----------|------|------------|
| 行业中性集中 v1 | `industry_neutral_concentrated_ng.py` | 53 行 | ✅ 完全一致 |
| 行业中性集中 v2 | `industry_neutral_concentrated_v2_ng.py` | 94 行 | ✅ 完全一致 |

### 注意事项

1. **Pandas 3.0 兼容**：框架内部使用显式迭代代替 `groupby().apply()` 来避免 Pandas 3.0 的 group-key 行为变化
2. **基准数据来源**：从 CSV（`/root/qlib_data/daily/`）加载，不依赖 qlib
3. **因子权重正负**：正权重 = 越大越好（如动量），负权重 = 越小越好（如波动率）
4. **旧版策略保留**：`*_ng.py` 是框架版，原版 `.py` 保留不动，方便对照

---

*生成时间：2026-03-19*  
*最后更新：当前 Agent — 新增回测引擎框架文档、框架版策略*  
*初始版本：2026-03-18 by 前任 Agent（Rick）*
