"""
多因子策略回测 —— 使用 qlib 原生表达式引擎
因子：动量 + 低波动 + 换手率反转
周期：2022-01-01 ~ 2026-02-28，月度调仓
规范：严格遵守 backtest/OutputFormat.md
"""
import sys
sys.path.insert(0, "/projects/portal/data/quant/utils")

import qlib
import pandas as pd
import numpy as np
from qlib.constant import REG_CN
from qlib.data import D
import warnings
warnings.filterwarnings("ignore")

# ── 初始化 ───────────────────────────────────────────────
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

STRATEGY_NAME  = "multifactor"
START          = "2021-01-01"   # 预热期（因子需要 250 天历史）
BACKTEST_START = "2022-01-01"
END            = "2026-02-28"
BENCHMARK_CODE  = "sz000905"
BENCHMARK_NAME  = "中证500 (sz000905)"
BENCHMARK2_CODE = "sh000001"
BENCHMARK2_NAME = "上证指数 (sh000001)"
OUTDIR         = "/projects/portal/data/quant/backtest"

# ── OutputFormat.md 标准函数 ─────────────────────────────
def save_backtest_results(strategy_name, nav_df, returns_df):
    base = f"{OUTDIR}/{strategy_name}"
    nav_df.to_csv(f"{base}_nav.csv", index=False, float_format="%.7f")
    returns_df.to_csv(f"{base}_monthly_returns.csv", index=False, float_format="%.8f")
    print(f"[backtest] Saved: {base}_nav.csv")
    print(f"[backtest] Saved: {base}_monthly_returns.csv")

def print_summary(strategy_name, returns_df, benchmark_name="N/A", start=None, end=None):
    rets  = returns_df["port_ret"].dropna().values
    ann_ret = (1 + rets).prod() ** (12 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    win_rate = (rets > 0).mean()
    nav = (1 + rets).cumprod()
    drawdown = nav / np.maximum.accumulate(nav) - 1
    max_dd  = drawdown.min()
    cum_ret = nav[-1] - 1

    b_rets   = returns_df["bench_ret"].dropna().values if "bench_ret" in returns_df else None
    b_ann_ret = (1 + b_rets).prod() ** (12 / len(b_rets)) - 1 if b_rets is not None else float("nan")
    b_ann_vol = b_rets.std() * np.sqrt(12) if b_rets is not None else float("nan")
    b_sharpe  = b_ann_ret / b_ann_vol if b_ann_vol > 0 else float("nan")
    b_nav     = (1 + b_rets).cumprod() if b_rets is not None else None
    b_max_dd  = (b_nav / np.maximum.accumulate(b_nav) - 1).min() if b_nav is not None else float("nan")
    b_cum_ret = b_nav[-1] - 1 if b_nav is not None else float("nan")

    excess_rets = returns_df["excess"].dropna().values if "excess" in returns_df else None
    alpha_ann   = (1 + excess_rets).prod() ** (12 / len(excess_rets)) - 1 if excess_rets is not None else float("nan")
    ex_vol      = excess_rets.std() * np.sqrt(12) if excess_rets is not None else float("nan")
    ir          = alpha_ann / ex_vol if ex_vol > 0 else float("nan")

    print("=" * 44)
    print(f"策略名称：{strategy_name}")
    print(f"回测区间：{start or returns_df['date'].iloc[0]} ~ {end or returns_df['date'].iloc[-1]}")
    print(f"基    准：{benchmark_name}")
    print("=" * 44)
    print(f"{'指标':<14} {'策略':>10} {'基准':>10}")
    print("-" * 36)
    print(f"{'年化收益率':<14} {ann_ret:>+10.1%} {b_ann_ret:>+10.1%}")
    print(f"{'年化波动率':<14} {ann_vol:>10.1%} {b_ann_vol:>10.1%}")
    print(f"{'夏普比率':<14} {sharpe:>10.2f} {b_sharpe:>10.2f}")
    print(f"{'最大回撤':<14} {max_dd:>10.1%} {b_max_dd:>10.1%}")
    print(f"{'月胜率':<14} {win_rate:>10.1%} {'—':>10}")
    print(f"{'累计收益':<14} {cum_ret:>+10.1%} {b_cum_ret:>+10.1%}")
    print(f"{'超额收益(Alpha)':<14} {alpha_ann:>+10.1%} {'—':>10}")
    print(f"{'信息比率(IR)':<14} {ir:>10.2f} {'—':>10}")
    print("=" * 44)

    # 逐年对比
    print("\n逐年收益对比：")
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    for yr in sorted(returns_df["date"].dt.year.unique()):
        g = returns_df[returns_df["date"].dt.year == yr]
        yr_p = (1 + g["port_ret"].dropna()).prod() - 1
        yr_b = (1 + g["bench_ret"].dropna()).prod() - 1 if "bench_ret" in g else float("nan")
        yr_e = yr_p - yr_b
        print(f"  {yr}: 策略 {yr_p:+.1%}  基准 {yr_b:+.1%}  超额 {yr_e:+.1%}")


# ════════════════════════════════════════════════════════
print("=" * 55)
print("多因子策略回测（qlib 原生表达式）")
print(f"回测区间：{BACKTEST_START} ~ {END}")
print("=" * 55)

# ── 1. 拉取因子数据 ──────────────────────────────────────
print("\n[1/6] 拉取因子数据（全 A 股，SH+SZ）...")
inst = D.list_instruments(D.instruments("all"), start_time=START, end_time=END, as_list=True)
inst = [c for c in inst if c.startswith("sh") or c.startswith("sz")]
print(f"股票池：{len(inst)} 只")

fields = [
    "Ref($close, 1)",
    "$close",
    "Ref($close, 250) / Ref($close, 20) - 1",
    "Std(Ref($close,1)/$close-1, 20)",
    "Mean($volume, 20) / $close",
]
col_names = ["ref_close", "close", "mom", "ivol", "turn"]

df_raw = D.features(inst, fields, start_time=START, end_time=END)
df_raw.columns = col_names
print(f"原始数据：{df_raw.shape}，非空率：{df_raw.notna().mean().mean():.1%}")

def build_benchmark_returns(code, series_name):
    bench_raw = D.features([code], ["$close"], start_time=START, end_time=END)
    bench_raw.columns = ["close"]
    bench_raw = bench_raw.reset_index()
    bench_raw["date"] = pd.to_datetime(bench_raw["datetime"])
    bench_raw["ym"] = bench_raw["date"].dt.to_period("M")
    bench_monthly = (
        bench_raw.sort_values("date")
        .groupby("ym", as_index=False)
        .last()[["date", "close"]]
        .sort_values("date")
    )
    bench_monthly["ret"] = bench_monthly["close"].pct_change()
    ret = bench_monthly.dropna(subset=["ret"]).set_index("date")["ret"]
    ret.name = series_name
    print(f"基准 {code} 已加载：{len(ret)} 个月")
    return ret


# 加载双基准（月内最后一个实际交易日）
bench_ret = build_benchmark_returns(BENCHMARK_CODE, "bench_ret")
bench2_ret = build_benchmark_returns(BENCHMARK2_CODE, "bench2_ret")

# ── 2. 构造月度截面信号 ───────────────────────────────────
print("\n[2/6] 构造月度截面信号...")
df_raw = df_raw.reset_index()
df_raw["date"] = pd.to_datetime(df_raw["datetime"])
df_raw["ym"]   = df_raw["date"].dt.to_period("M")

monthly = (
    df_raw.sort_values("date")
    .groupby(["ym", "instrument"])
    .last()
    .reset_index()
)

def zscore(s):
    s = s.clip(s.mean() - 3*s.std(), s.mean() + 3*s.std())
    return (s - s.mean()) / (s.std() + 1e-8)

def calc_composite(group):
    g = group.copy()
    mask = g[["mom", "ivol", "turn"]].notna().all(axis=1)
    if mask.sum() < 30:
        g["score"] = np.nan
        return g
    g.loc[mask, "score"] = (
          zscore(g.loc[mask, "mom"])
        - zscore(g.loc[mask, "ivol"])
        - zscore(g.loc[mask, "turn"])
    ) / 3.0
    g.loc[~mask, "score"] = np.nan
    return g

monthly = monthly.groupby("ym", group_keys=False).apply(calc_composite)
n_valid = monthly.groupby("ym")["score"].count().gt(0).sum()
print(f"有效截面月数：{n_valid}")

# ── 3. 月度调仓回测 ──────────────────────────────────────
print(f"\n[3/6] 月度调仓回测（{BACKTEST_START} ~ {END}）...")
results = []
periods = sorted(monthly["ym"].unique())

for i, ym in enumerate(periods):
    if ym < pd.Period(BACKTEST_START[:7], "M"):
        continue
    if i + 1 >= len(periods):
        break

    hold_period = periods[i + 1]
    slice_df = monthly[monthly["ym"] == ym].dropna(subset=["score", "close"])
    if len(slice_df) < 50:
        continue

    threshold = slice_df["score"].quantile(0.90)
    selected = (
        slice_df[slice_df["score"] >= threshold][["instrument", "close"]]
        .copy()
        .rename(columns={"close": "entry_close"})
    )

    next_month = monthly[monthly["ym"] == hold_period][["instrument", "close", "date"]].copy()
    next_month = next_month.rename(columns={"close": "exit_close", "date": "exit_date"})

    merged = selected.merge(next_month, on="instrument", how="inner").dropna(subset=["entry_close", "exit_close"])
    if len(merged) < 5:
        continue

    merged["ret"] = merged["exit_close"] / merged["entry_close"] - 1
    signal_date = pd.to_datetime(slice_df["date"].max()).strftime("%Y-%m-%d")
    hold_date = pd.to_datetime(next_month["exit_date"].max()).strftime("%Y-%m-%d")

    results.append({
        "signal_date": signal_date,
        "date": hold_date,
        "port_ret": merged["ret"].mean(),
        "n_stocks": int(len(merged)),
    })

port_df = pd.DataFrame(results)
print(f"有效调仓月数：{len(port_df)}")

# ── 4. 合并基准 ───────────────────────────────────────────
print("\n[4/6] 合并基准收益...")
port_df["date"] = pd.to_datetime(port_df["date"])
port_df["signal_date"] = pd.to_datetime(port_df["signal_date"])
port_df = port_df.set_index("date")
bench_ret.index  = pd.to_datetime(bench_ret.index)
bench2_ret.index = pd.to_datetime(bench2_ret.index)
combined = port_df.join(bench_ret, how="left").join(bench2_ret, how="left")
combined["excess"]  = combined["port_ret"] - combined["bench_ret"]
combined["excess2"] = combined["port_ret"] - combined["bench2_ret"]
combined = combined.reset_index()
combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

# ── 5. 输出汇总（OutputFormat.md 规范）──────────────────
print("\n[5/6] 绩效汇总...")

def calc_ann(rets):
    r = rets.dropna().values
    if len(r) == 0: return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    nav     = (1 + r).cumprod()
    max_dd  = (nav / np.maximum.accumulate(nav) - 1).min()
    cum_ret = nav[-1] - 1
    return ann_ret, ann_vol, sharpe, max_dd, cum_ret

p_ret, p_vol, p_sh, p_dd, p_cum   = calc_ann(combined["port_ret"])
b1_ret, b1_vol, b1_sh, b1_dd, b1_cum = calc_ann(combined["bench_ret"])
b2_ret, b2_vol, b2_sh, b2_dd, b2_cum = calc_ann(combined["bench2_ret"])
ex1_ret, ex1_vol, _, _, _  = calc_ann(combined["excess"])
ex2_ret, ex2_vol, _, _, _  = calc_ann(combined["excess2"])
ir1 = ex1_ret / ex1_vol if ex1_vol > 0 else float("nan")
ir2 = ex2_ret / ex2_vol if ex2_vol > 0 else float("nan")
win = (combined["port_ret"].dropna() > 0).mean()

print("=" * 56)
print(f"策略名称：{STRATEGY_NAME}")
print(f"回测区间：{BACKTEST_START} ~ {END}")
print(f"基    准：{BENCHMARK_NAME} / {BENCHMARK2_NAME}")
print("=" * 56)
print(f"{'指标':<16} {'策略':>10} {'中证500':>10} {'上证指数':>10}")
print("-" * 50)
print(f"{'年化收益率':<16} {p_ret:>+10.1%} {b1_ret:>+10.1%} {b2_ret:>+10.1%}")
print(f"{'年化波动率':<16} {p_vol:>10.1%} {b1_vol:>10.1%} {b2_vol:>10.1%}")
print(f"{'夏普比率':<16} {p_sh:>10.2f} {b1_sh:>10.2f} {b2_sh:>10.2f}")
print(f"{'最大回撤':<16} {p_dd:>10.1%} {b1_dd:>10.1%} {b2_dd:>10.1%}")
print(f"{'月胜率':<16} {win:>10.1%} {'—':>10} {'—':>10}")
print(f"{'累计收益':<16} {p_cum:>+10.1%} {b1_cum:>+10.1%} {b2_cum:>+10.1%}")
print(f"{'超额(vs中证500)':<16} {ex1_ret:>+10.1%} {'—':>10} {'—':>10}")
print(f"{'IR(vs中证500)':<16} {ir1:>10.2f} {'—':>10} {'—':>10}")
print(f"{'超额(vs上证)':<16} {ex2_ret:>+10.1%} {'—':>10} {'—':>10}")
print(f"{'IR(vs上证)':<16} {ir2:>10.2f} {'—':>10} {'—':>10}")
print("=" * 56)

print("\n逐年收益对比：")
combined["date_dt"] = pd.to_datetime(combined["date"])
for yr in sorted(combined["date_dt"].dt.year.unique()):
    g = combined[combined["date_dt"].dt.year == yr]
    yr_p  = (1 + g["port_ret"].dropna()).prod() - 1
    yr_b1 = (1 + g["bench_ret"].dropna()).prod() - 1
    yr_b2 = (1 + g["bench2_ret"].dropna()).prod() - 1
    print(f"  {yr}: 策略 {yr_p:+.1%}  中证500 {yr_b1:+.1%}  上证 {yr_b2:+.1%}  "
          f"超额(中证) {yr_p-yr_b1:+.1%}  超额(上证) {yr_p-yr_b2:+.1%}")
combined.drop(columns=["date_dt"], inplace=True)

# ── 6. 保存文件（OutputFormat.md 规范）──────────────────
print("\n[6/6] 保存结果...")

nav_period_df = pd.DataFrame({
    "date": combined["date"],
    "strategy": (1 + combined["port_ret"]).cumprod(),
    "benchmark": (1 + combined["bench_ret"].fillna(0)).cumprod(),
    "benchmark2": (1 + combined["bench2_ret"].fillna(0)).cumprod(),
})
if not combined.empty:
    initial_nav_row = pd.DataFrame({
        "date": [combined["signal_date"].iloc[0].strftime("%Y-%m-%d")],
        "strategy": [1.0],
        "benchmark": [1.0],
        "benchmark2": [1.0],
    })
    nav_df = pd.concat([initial_nav_row, nav_period_df], ignore_index=True)
else:
    nav_df = nav_period_df

returns_df = combined[["date", "port_ret", "n_stocks", "bench_ret", "excess", "bench2_ret", "excess2"]].copy()
returns_df["n_stocks"] = returns_df["n_stocks"].astype(int)

save_backtest_results(STRATEGY_NAME, nav_df, returns_df)

# report.md
report_path = f"{OUTDIR}/{STRATEGY_NAME}_report.md"
with open(report_path, "w") as f:
    f.write(f"# 多因子策略回测报告\n\n")
    f.write(f"**回测区间**：{BACKTEST_START} ~ {END}  \n")
    f.write(f"**基准**：{BENCHMARK_NAME}  \n")
    f.write(f"**调仓频率**：月度  \n")
    f.write(f"**持仓数量**：TOP 10%（约 {int(returns_df['n_stocks'].mean())} 只）  \n\n")
    f.write("## 因子组合\n\n")
    f.write("| 因子 | 表达式 | 方向 |\n")
    f.write("|------|--------|------|\n")
    f.write("| 动量 | `Ref($close,250)/Ref($close,20)-1` | 正向 |\n")
    f.write("| 低波动 | `Std(Ref($close,1)/$close-1, 20)` | 负向 |\n")
    f.write("| 低换手 | `Mean($volume,20)/$close` | 负向 |\n\n")
    f.write("## 改进空间\n\n")
    f.write("- 2025 年跑输基准：市场切换至高 Beta 小盘风格，低波动因子压制了高弹性标的\n")
    f.write("- 可增加估值因子（PB/PE）提升防御性\n")
    f.write("- 可增加市值因子暴露以捕捉小盘行情\n")
    f.write("- 调仓周期可测试双周/周度，减少信号衰减\n")
print(f"[backtest] Saved: {report_path}")
print("\n完成。")
