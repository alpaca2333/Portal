# Backtest Output Format Specification

> 本文档规定了所有回测结果的标准输出格式，**每次新策略回测必须严格遵守**。

---

## 目录结构约定

每个策略的回测产物统一放在 `backtest/` 目录下，命名规则：

```
backtest/
├── {strategy_name}_nav.csv               ← 净值曲线（必须）
├── {strategy_name}_monthly_returns.csv   ← 月度收益明细（必须）
└── OUTPUT_FORMAT.md                      ← 本规范文档
```

`{strategy_name}` 使用小写下划线命名，例如：`momentum`、`multifactor`、`cta_trend`。

---

## 文件格式规范

### 1. 净值曲线文件 `{strategy_name}_nav.csv`

记录每个调仓日的累计净值（初始净值 = 1.0）。

**字段定义：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | string | 调仓日期，格式 `YYYY-MM-DD` |
| `strategy` | float | 策略累计净值，初始为 1.0 |
| `benchmark` | float | 主基准累计净值，初始为 1.0（无基准时省略） |
| `benchmark2` | float | 第二基准累计净值，初始为 1.0（有第二基准时提供） |

**示例（含双基准）：**

```csv
date,strategy,benchmark,benchmark2
2022-03-31,0.9155493,1.0065359,0.9863200
2022-06-30,0.9922171,0.9454975,0.9612400
2022-09-30,1.0341820,0.9102341,0.9387100
```

> ⚠️ `strategy` 和 `benchmark` 均为**累计净值**（非收益率），1.2 表示累计盈利 20%。

---

### 2. 月度收益文件 `{strategy_name}_monthly_returns.csv`

记录每个调仓周期的收益率明细。

**字段定义：**

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `date` | string | ✅ | 调仓日期，格式 `YYYY-MM-DD`（通常为月末） |
| `port_ret` | float | ✅ | 策略当期收益率，**小数形式**（0.05 = 5%） |
| `n_stocks` | int | ✅ | 当期持仓股票数量 |
| `bench_ret` | float | ⬜ | 主基准当期收益率，**小数形式**（有基准时必须提供） |
| `excess` | float | ⬜ | 超额收益率 = `port_ret - bench_ret`，**小数形式** |
| `bench2_ret` | float | ⬜ | 第二基准当期收益率，**小数形式**（有第二基准时提供，如同时对标上证+中证500） |
| `excess2` | float | ⬜ | 超额收益率 = `port_ret - bench2_ret`，**小数形式** |

**示例（含双基准）：**

```csv
date,port_ret,n_stocks,bench_ret,excess,bench2_ret,excess2
2022-03-31,-0.08445071,401,0.00653589,-0.09098660,-0.01234500,-0.07210571
2022-06-30,0.08373978,413,-0.06064206,0.14438185,-0.03210000,0.11583978
```

**示例（含基准）：**

```csv
date,port_ret,n_stocks,bench_ret,excess
2022-03-31,-0.08445071,401,0.00653589,-0.09098660
2022-06-30,0.08373978,413,-0.06064206,0.14438185
2022-09-30,0.04218340,398,-0.02134500,0.06352840
```

**示例（无基准）：**

```csv
date,port_ret,n_stocks
2016-03-31,0.18781512,230
2016-06-30,0.03633235,248
```

> ⚠️ 所有收益率字段均为**小数形式**，不是百分比。`0.187` 表示 +18.7%，`-0.084` 表示 -8.4%。

### 3. 回测总结文件 `{strategy_name}_report.md`

记录该策略的详细思路，技术细节。并对他进行总结，反思改进空间等必要信息。

---

## 汇总指标输出规范

每次回测完成后，**必须在终端打印以下汇总表**，并在策略 README 或注释中记录：

```
========================================
策略名称：{strategy_name}
回测区间：{start_date} ~ {end_date}
基    准：{benchmark_name}（如：沪深300 CSI300）
========================================
指标                  策略          基准
----------------------------------------
年化收益率            +12.9%        +8.3%
年化波动率            23.4%         18.7%
夏普比率              0.55          0.44
最大回撤              -25.1%        -33.2%
月胜率                54.5%         —
累计收益              +56.0%        +32.1%
超额收益（Alpha）     +4.6%         —
信息比率（IR）        0.31          —
========================================
```

**基准选择规则：**

| 策略类型 | 基准 | 数据代码 |
|----------|------|----------|
| A 股多头因子策略 | 沪深300 | `sh000300` |
| 中小盘因子策略 | 中证500 | `sh000905` |
| 全市场策略 | 万得全A | — |
| CTA / 期货策略 | 南华商品指数 | — |

> ❌ **禁止只报策略绝对收益，必须附带基准对标。**

---

## Python 输出模板

在回测脚本末尾统一调用以下函数输出结果：

```python
import pandas as pd
import numpy as np

def save_backtest_results(strategy_name: str, nav_df: pd.DataFrame, returns_df: pd.DataFrame):
    """
    Save backtest results to backtest/ directory.

    Parameters
    ----------
    strategy_name : str
        Strategy identifier, e.g. 'momentum', 'multifactor'
    nav_df : pd.DataFrame
        Columns: date, strategy[, benchmark]  — cumulative NAV starting at 1.0
    returns_df : pd.DataFrame
        Columns: date, port_ret, n_stocks[, bench_ret, excess]  — decimal fractions
    """
    base = f"backtest/{strategy_name}"
    nav_df.to_csv(f"{base}_nav.csv", index=False, float_format="%.7f")
    returns_df.to_csv(f"{base}_monthly_returns.csv", index=False, float_format="%.8f")
    print(f"[backtest] Saved: {base}_nav.csv, {base}_monthly_returns.csv")


def print_summary(strategy_name: str, returns_df: pd.DataFrame,
                  benchmark_name: str = "N/A", start: str = None, end: str = None):
    """Print standardised performance summary to stdout."""
    rets = returns_df["port_ret"].values
    ann_ret = (1 + rets).prod() ** (12 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    win_rate = (rets > 0).mean()

    nav = (1 + rets).cumprod()
    drawdown = nav / np.maximum.accumulate(nav) - 1
    max_dd = drawdown.min()

    cum_ret = nav[-1] - 1

    print("=" * 44)
    print(f"Strategy : {strategy_name}")
    print(f"Period   : {start or returns_df['date'].iloc[0]} ~ {end or returns_df['date'].iloc[-1]}")
    print(f"Benchmark: {benchmark_name}")
    print("=" * 44)
    print(f"Ann. Return  : {ann_ret:+.1%}")
    print(f"Ann. Vol     : {ann_vol:.1%}")
    print(f"Sharpe       : {sharpe:.2f}")
    print(f"Max Drawdown : {max_dd:.1%}")
    print(f"Win Rate     : {win_rate:.1%}")
    print(f"Cum. Return  : {cum_ret:+.1%}")
    print("=" * 44)
```

---

*Last updated: 2026-03-18 — 新增 `bench2_ret` / `excess2` / `benchmark2` 字段说明（支持双基准对标）*
