"""
动量因子策略 (Momentum Factor)
---
逻辑: 过去N天涨得好的股票，未来继续涨的概率更高
经典参数: 过去12个月收益率，跳过最近1个月（避免短期反转）
回测框架: 月度调仓，等权做多TOP分位
规范: 每次回测必须附带指数基准对标
"""
import sys
sys.path.insert(0, "/root/quant_workspace/utils")

import pandas as pd
import numpy as np
from data_loader import get_close, get_data
import warnings
warnings.filterwarnings("ignore")

# =====================
# 参数配置
# =====================
START = "2015-01-01"
END   = "2025-12-31"
LOOKBACK_DAYS = 250
SKIP_DAYS     = 20
HOLD_TOP_PCT  = 0.1
REBAL_FREQ    = "ME"
BENCHMARK_CODE = "sz000905"   # 中证500（数据里有）

print("=" * 60)
print("动量因子策略回测")
print("=" * 60)

# =====================
# 1. 加载数据
# =====================
print("\n[1/6] 加载数据...")
close = get_close(start=START, end=END)
# 过滤掉北交所
sh_sz_cols = [c for c in close.columns if c.startswith("SH") or c.startswith("SZ")]
close = close[sh_sz_cols]
print(f"股票池 (仅SH+SZ): {close.shape}")

# 加载基准（中证500）
bench_df = get_data(codes=[BENCHMARK_CODE.upper()], start=START, end=END, fields=["close"])
bench_close = bench_df["close"].unstack("code").iloc[:, 0]
bench_monthly = bench_close.resample(REBAL_FREQ).last()
bench_ret = bench_monthly.pct_change().dropna()
bench_ret.name = "bench_ret"
print(f"基准 {BENCHMARK_CODE} 已加载: {len(bench_ret)} 个月")

# =====================
# 2. 计算动量信号
# =====================
print("\n[2/6] 计算动量信号...")

def calc_momentum(close_df, lookback=250, skip=20):
    ret_long  = close_df.pct_change(lookback)
    ret_short = close_df.pct_change(skip)
    return (1 + ret_long) / (1 + ret_short) - 1

momentum = calc_momentum(close, lookback=LOOKBACK_DAYS, skip=SKIP_DAYS)
print(f"动量信号非NaN比例: {momentum.notna().mean().mean():.1%}")

# =====================
# 3. 月末选股 + 计算下月收益
# =====================
print("\n[3/6] 月度选股...")

monthly_dates = close.resample(REBAL_FREQ).last().index
monthly_dates = monthly_dates[(monthly_dates >= pd.Timestamp(START)) &
                              (monthly_dates <= pd.Timestamp(END))]

portfolio_returns = []
for i in range(len(monthly_dates) - 1):
    signal_date = monthly_dates[i]
    next_date   = monthly_dates[i + 1]

    if signal_date not in momentum.index:
        continue
    signal = momentum.loc[signal_date].dropna()
    if len(signal) < 50:
        continue

    threshold = signal.quantile(1 - HOLD_TOP_PCT)
    selected  = signal[signal >= threshold].index.tolist()

    try:
        ret_period = close.loc[next_date, selected] / close.loc[signal_date, selected] - 1
        port_ret   = ret_period.mean()
    except:
        continue

    portfolio_returns.append({
        "date":     next_date,
        "port_ret": port_ret,
        "n_stocks": len(selected),
    })

port_df = pd.DataFrame(portfolio_returns).set_index("date")
print(f"有效调仓月数: {len(port_df)}")

# =====================
# 4. 合并基准，计算超额
# =====================
print("\n[4/6] 合并基准收益...")
combined = port_df.join(bench_ret, how="left")
combined["excess"] = combined["port_ret"] - combined["bench_ret"]

# =====================
# 5. 业绩统计（含对标）
# =====================
print("\n[5/6] 计算绩效指标...")

def calc_metrics(returns_series, freq=12):
    r = returns_series.dropna()
    cum     = (1 + r).cumprod()
    n_years = len(r) / freq
    ann_ret = cum.iloc[-1] ** (1 / n_years) - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / rolling_max
    max_dd      = drawdown.min()
    win_rate    = (r > 0).mean()
    return dict(年化收益=ann_ret, 年化波动=ann_vol, 夏普比率=sharpe,
                最大回撤=max_dd, 月胜率=win_rate, 累计收益=cum.iloc[-1]-1)

m_port  = calc_metrics(combined["port_ret"])
m_bench = calc_metrics(combined["bench_ret"].dropna())
m_excess = calc_metrics(combined["excess"].dropna())

# 信息比率 IR = 年化超额收益 / 超额收益波动
ir = m_excess["年化收益"] / m_excess["年化波动"] if m_excess["年化波动"] > 0 else 0

print("\n📊 动量因子策略绩效 (月度调仓, TOP10%, 等权)")
print(f"{'指标':<12} {'策略':>10} {'基准(中证500)':>14} {'超额':>10}")
print("-" * 50)
for k in ["年化收益", "年化波动", "夏普比率", "最大回撤", "月胜率", "累计收益"]:
    p = m_port[k]
    b = m_bench.get(k, float('nan'))
    e = m_excess.get(k, float('nan'))
    fmt = ".1%" if k not in ["夏普比率"] else ".2f"
    print(f"{k:<12} {p:{fmt}>10} {b:{fmt}>14} {e:{fmt}>10}")
print(f"{'信息比率(IR)':<12} {'':>10} {'':>14} {ir:>10.2f}")

# 逐年对标
print("\n逐年收益对比:")
port_df_copy = port_df.copy()
port_df_copy["year"] = port_df_copy.index.year
bench_copy = combined["bench_ret"].dropna().copy()
bench_copy.index = pd.to_datetime(bench_copy.index)
bench_copy_yr = bench_copy.copy()
bench_copy_yr.index = bench_copy_yr.index.year

for yr, grp in port_df_copy.groupby("year"):
    yr_port  = (1 + grp["port_ret"]).prod() - 1
    yr_bench_vals = combined.loc[combined.index.year == yr, "bench_ret"].dropna()
    yr_bench = (1 + yr_bench_vals).prod() - 1 if len(yr_bench_vals) > 0 else float("nan")
    yr_excess = yr_port - yr_bench
    print(f"  {yr}: 策略{yr_port:+.1%}  基准{yr_bench:+.1%}  超额{yr_excess:+.1%}")

# =====================
# 6. 保存结果
# =====================
print("\n[6/6] 保存结果...")
out_dir = "/root/quant_workspace/backtest"
combined.to_csv(f"{out_dir}/momentum_monthly_returns.csv")
nav_port  = (1 + combined["port_ret"]).cumprod()
nav_bench = (1 + combined["bench_ret"].fillna(0)).cumprod()
pd.DataFrame({"strategy": nav_port, "benchmark": nav_bench}).to_csv(f"{out_dir}/momentum_nav.csv")
print(f"已保存到: {out_dir}/momentum_*.csv")
print("\n" + "=" * 60)
print("回测完成！")
print("=" * 60)
