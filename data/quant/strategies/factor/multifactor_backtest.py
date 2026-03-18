"""
多因子策略回测 —— 使用 qlib 原生表达式引擎
因子：动量 + 低波动 + 换手率反转
周期：2022-01-01 ~ 2025-12-31，月度调仓
规范：每次回测必须附带指数基准对标
"""
import sys
sys.path.insert(0, "/root/quant_workspace/utils")

import qlib
import pandas as pd
import numpy as np
from qlib.constant import REG_CN
from qlib.data import D
import warnings
warnings.filterwarnings("ignore")

# ── 初始化 qlib ──────────────────────────────────────────
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

START = "2021-01-01"   # 多留一年用于因子预热
END   = "2025-12-31"

BENCHMARK_CODE = "sz000905"   # 中证500

print("=" * 60)
print("多因子策略回测（qlib 原生表达式）")
print("回测区间：2022-01-01 ~ 2025-12-31")
print("=" * 60)

# ── 1. 拉取因子数据 ──────────────────────────────────────
print("\n[1/6] 拉取因子数据（全 A 股，SH+SZ）...")

# 获取股票池（沪深主板，排除北交所）
inst = D.list_instruments(
    D.instruments("all"),
    start_time=START, end_time=END, as_list=True
)
inst = [c for c in inst if c.startswith("sh") or c.startswith("sz")]
print(f"股票池：{len(inst)} 只")

# qlib 表达式：
#   MOM12_1  = 过去 250 天 / 过去 20 天 累计收益（动量，跳过最近1月）
#   IVOL20   = 过去 20 天日收益率标准差（低波动因子，取负）
#   TURN20   = 过去 20 天平均换手率（换手率反转，取负）
#   ILLIQ20  = 过去 20 天非流动性（Amihud，价格影响）

fields = [
    "Ref($close, 1)",                                   # 昨收（确认数据可读）
    "$close",
    "Ref($close, 250) / Ref($close, 20) - 1",           # MOM: 12月动量去掉最近1月
    "Std(Ref($close,1)/$close-1, 20)",                   # IVOL: 20日波动率
    "Mean($volume, 20) / $close",                        # TURN: 换手率代理（量价比）
    "Mean(Abs($close/Ref($close,1)-1)/$volume, 20)",     # ILLIQ: Amihud非流动性
]
col_names = ["ref_close", "close", "mom", "ivol", "turn", "illiq"]

df_raw = D.features(inst, fields, start_time=START, end_time=END)
df_raw.columns = col_names
print(f"原始数据：{df_raw.shape}，非空率：{df_raw.notna().mean().mean():.1%}")

# 加载基准（中证500）
bench_raw = D.features([BENCHMARK_CODE], ["$close"], start_time=START, end_time=END)
bench_raw.columns = ["bench_close"]
bench_raw = bench_raw.reset_index()
bench_raw["date"] = pd.to_datetime(bench_raw["datetime"])
bench_monthly_close = bench_raw.set_index("date")["bench_close"].resample("ME").last()
bench_ret = bench_monthly_close.pct_change().dropna()
bench_ret.name = "bench_ret"
print(f"基准 {BENCHMARK_CODE} 已加载: {len(bench_ret)} 个月")

# ── 2. 构造月度截面信号 ───────────────────────────────────
print("\n[2/6] 构造月度截面信号...")

# 取每月最后一个交易日的截面
df_raw = df_raw.reset_index()
df_raw["date"] = pd.to_datetime(df_raw["datetime"])
df_raw["ym"]   = df_raw["date"].dt.to_period("M")

# 每月末截面
monthly = (
    df_raw.sort_values("date")
    .groupby(["ym", "instrument"])
    .last()
    .reset_index()
)

def zscore(s):
    """截面 z-score 标准化，去极值 ±3σ"""
    s = s.clip(s.mean() - 3*s.std(), s.mean() + 3*s.std())
    return (s - s.mean()) / (s.std() + 1e-8)

# 合成因子（等权）：动量 - 低波动 - 换手率反转 + 流动性惩罚
def calc_composite(group):
    g = group.copy()
    # 有效数据 >= 30 只才计算
    mask = g[["mom","ivol","turn","illiq"]].notna().all(axis=1)
    if mask.sum() < 30:
        g["score"] = np.nan
        return g
    g.loc[mask, "score"] = (
          zscore(g.loc[mask, "mom"])        # 高动量 → 高分
        - zscore(g.loc[mask, "ivol"])       # 低波动 → 高分
        - zscore(g.loc[mask, "turn"])       # 低换手 → 高分（反转）
    ) / 3.0
    g.loc[~mask, "score"] = np.nan
    return g

monthly = monthly.groupby("ym", group_keys=False).apply(calc_composite)
n_valid = monthly.groupby("ym")["score"].count().gt(0).sum()
print(f"合成因子计算完成，有效截面月数：{n_valid}")

# ── 3. 月度调仓回测 ──────────────────────────────────────
print("\n[3/6] 月度调仓回测（2022-2025）...")

# 收盘价矩阵，用于计算下月收益
close_pivot = (
    df_raw.pivot(index="date", columns="instrument", values="close")
    .resample("ME").last()
)

results = []
periods = sorted(monthly["ym"].unique())

for i, ym in enumerate(periods):
    # 只回测 2022 年以后
    if ym < pd.Period("2022-01", "M"):
        continue
    if i + 1 >= len(periods):
        break

    signal_period = ym
    hold_period   = periods[i + 1]

    slice_df = monthly[monthly["ym"] == signal_period].dropna(subset=["score","close"])
    if len(slice_df) < 50:
        continue

    # 选 TOP 10%（得分最高）
    threshold = slice_df["score"].quantile(0.90)
    selected  = slice_df[slice_df["score"] >= threshold]["instrument"].tolist()

    # 计算下月收益：用月末收盘价
    signal_date = slice_df["date"].max()
    try:
        hold_date   = monthly[monthly["ym"] == hold_period]["date"].max()
        p0 = close_pivot.loc[signal_date, selected].dropna()
        p1 = close_pivot.loc[hold_date,   p0.index].dropna()
        common = p0.index.intersection(p1.index)
        if len(common) < 5:
            continue
        port_ret = ((p1[common] / p0[common]) - 1).mean()
    except Exception:
        continue

    results.append({
        "date":     hold_date,
        "port_ret": port_ret,
        "n_stocks": len(common),
    })

port_df = pd.DataFrame(results).set_index("date")
print(f"有效调仓月数：{len(port_df)}")

# ── 4. 合并基准，计算超额 ─────────────────────────────────
print("\n[4/6] 合并基准收益...")
port_df.index = pd.to_datetime(port_df.index)
bench_ret.index = pd.to_datetime(bench_ret.index)
combined = port_df.join(bench_ret, how="left")
combined["excess"] = combined["port_ret"] - combined["bench_ret"]

# ── 5. 绩效统计（含对标）────────────────────────────────
print("\n[5/6] 计算绩效...")

def calc_metrics(r, freq=12):
    r = r.dropna()
    cum     = (1 + r).cumprod()
    n_years = len(r) / freq
    ann_ret = cum.iloc[-1] ** (1/n_years) - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    drawdown = (cum - cum.cummax()) / cum.cummax()
    max_dd  = drawdown.min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (r > 0).mean()
    return dict(年化收益=ann_ret, 年化波动=ann_vol, 夏普比率=sharpe,
                最大回撤=max_dd, Calmar=calmar, 月胜率=win_rate,
                累计收益=cum.iloc[-1]-1)

m_port  = calc_metrics(combined["port_ret"])
m_bench = calc_metrics(combined["bench_ret"].dropna())
m_excess = calc_metrics(combined["excess"].dropna())
ir = m_excess["年化收益"] / m_excess["年化波动"] if m_excess["年化波动"] > 0 else 0

print("\n📊 多因子策略绩效（2022-2025，TOP10%，等权）")
print(f"{'指标':<12} {'策略':>10} {'基准(中证500)':>14} {'超额':>10}")
print("-" * 50)
for k in ["年化收益", "年化波动", "夏普比率", "最大回撤", "月胜率", "累计收益"]:
    p = m_port[k]
    b = m_bench.get(k, float("nan"))
    e = m_excess.get(k, float("nan"))
    if k in ["夏普比率", "Calmar"]:
        print(f"{k:<12} {p:>10.2f} {b:>14.2f} {e:>10.2f}")
    else:
        print(f"{k:<12} {p:>10.1%} {b:>14.1%} {e:>10.1%}")
print(f"{'信息比率(IR)':<12} {'':>10} {'':>14} {ir:>10.2f}")

# 逐年对标
print("\n逐年收益对比:")
for yr in sorted(combined.index.year.unique()):
    grp = combined[combined.index.year == yr]
    yr_port  = (1 + grp["port_ret"].dropna()).prod() - 1
    yr_bench = (1 + grp["bench_ret"].dropna()).prod() - 1
    yr_excess = yr_port - yr_bench
    n = grp["n_stocks"].mean() if "n_stocks" in grp else 0
    print(f"  {yr}: 策略{yr_port:+.1%}  基准{yr_bench:+.1%}  超额{yr_excess:+.1%}  (平均持仓{n:.0f}只)")

# ── 6. 保存 ──────────────────────────────────────────────
out = "/root/quant_workspace/backtest"
combined.to_csv(f"{out}/multifactor_monthly_returns.csv")
nav_port  = (1 + combined["port_ret"]).cumprod()
nav_bench = (1 + combined["bench_ret"].fillna(0)).cumprod()
pd.DataFrame({"strategy": nav_port, "benchmark": nav_bench}).to_csv(f"{out}/multifactor_nav.csv")
print(f"\n[6/6] 已保存到 {out}/multifactor_*.csv")
print("=" * 60)
