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

`data/quant/data/quant.db` 包含 3 张表：

| 表名 | 主键 | 说明 |
|------|------|------|
| `stock_daily` | (ts_code, trade_date) | 宽表：价格+基本面+复权因子+财务指标+行业 |
| `industry_info` | industry_code | 行业代码 → 名称 (申万 L1/L2/L3) |
| `stock_info` | ts_code | 股票代码 → 名称/上市日期等 |

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
