"""
Large-cap quality-momentum strategy
----------------------------------
Hypothesis:
- In large-cap A shares, medium-term trend persistence is cleaner.
- Stable price behaviour is a usable quality proxy when fundamentals are unavailable.
- Volatility is only a risk-control input, not the main alpha source.

Universe:
- SH + SZ only
- Dynamic large-cap proxy: top 5% by 60-day average traded value proxy

Factors:
- 12-1 momentum (positive)
- 60-day price stability (positive)
- 20-day realised volatility (negative, low weight)

Rebalance:
- Monthly

Benchmarks:
- Shanghai Composite: sh000001
- CSI 500: sz000905

Output:
- Strictly follows backtest/OutputFormat.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/projects/portal/data/quant/utils")

import warnings
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

STRATEGY_NAME = "largecap_multifactor"
START = "2021-01-01"          # warm-up for 250-day momentum
BACKTEST_START = "2022-01-01"
END = "2026-02-28"
OUTDIR = Path("/projects/portal/data/quant/backtest")

LARGECAP_PCT = 0.05            # strict large-cap proxy bucket
TOP_PCT = 0.20                 # select top 20% inside the large-cap bucket
MIN_LARGECAP_COUNT = 80
MIN_HOLDING_COUNT = 8

BENCHMARK_CODE = "sh000001"
BENCHMARK_NAME = "上证指数 (sh000001)"
BENCHMARK2_CODE = "sz000905"
BENCHMARK2_NAME = "中证500 (sz000905)"


# ── Output helpers ───────────────────────────────────────
def save_backtest_results(strategy_name: str, nav_df: pd.DataFrame, returns_df: pd.DataFrame) -> None:
    base = OUTDIR / strategy_name
    nav_df.to_csv(f"{base}_nav.csv", index=False, float_format="%.7f")
    returns_df.to_csv(f"{base}_monthly_returns.csv", index=False, float_format="%.8f")
    print(f"[backtest] Saved: {base}_nav.csv")
    print(f"[backtest] Saved: {base}_monthly_returns.csv")


def calc_return_metrics(returns: pd.Series, freq: int = 12) -> dict[str, float]:
    r = pd.Series(returns).dropna()
    if r.empty:
        return {
            "ann_ret": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "cum_ret": float("nan"),
            "win_rate": float("nan"),
        }

    nav = (1 + r).cumprod()
    ann_ret = nav.iloc[-1] ** (freq / len(r)) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    drawdown = nav / nav.cummax() - 1

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": drawdown.min(),
        "cum_ret": nav.iloc[-1] - 1,
        "win_rate": (r > 0).mean(),
    }


def print_summary(returns_df: pd.DataFrame) -> None:
    port = calc_return_metrics(returns_df["port_ret"])
    bench1 = calc_return_metrics(returns_df["bench_ret"])
    bench2 = calc_return_metrics(returns_df["bench2_ret"])
    excess1 = calc_return_metrics(returns_df["excess"])
    excess2 = calc_return_metrics(returns_df["excess2"])
    ir1 = excess1["ann_ret"] / excess1["ann_vol"] if excess1["ann_vol"] > 0 else float("nan")
    ir2 = excess2["ann_ret"] / excess2["ann_vol"] if excess2["ann_vol"] > 0 else float("nan")

    print("=" * 64)
    print(f"策略名称：{STRATEGY_NAME}")
    print(f"回测区间：{BACKTEST_START} ~ {END}")
    print(f"股票池  ：每期流动性代理前 {LARGECAP_PCT:.0%}（严格大盘代理）")
    print(f"基    准：{BENCHMARK_NAME} / {BENCHMARK2_NAME}")
    print("=" * 64)
    print(f"{'指标':<18} {'策略':>10} {'上证指数':>10} {'中证500':>10}")
    print("-" * 52)
    print(f"{'年化收益率':<18} {port['ann_ret']:>+10.1%} {bench1['ann_ret']:>+10.1%} {bench2['ann_ret']:>+10.1%}")
    print(f"{'年化波动率':<18} {port['ann_vol']:>10.1%} {bench1['ann_vol']:>10.1%} {bench2['ann_vol']:>10.1%}")
    print(f"{'夏普比率':<18} {port['sharpe']:>10.2f} {bench1['sharpe']:>10.2f} {bench2['sharpe']:>10.2f}")
    print(f"{'最大回撤':<18} {port['max_dd']:>10.1%} {bench1['max_dd']:>10.1%} {bench2['max_dd']:>10.1%}")
    print(f"{'月胜率':<18} {port['win_rate']:>10.1%} {'—':>10} {'—':>10}")
    print(f"{'累计收益':<18} {port['cum_ret']:>+10.1%} {bench1['cum_ret']:>+10.1%} {bench2['cum_ret']:>+10.1%}")
    print(f"{'超额(vs上证)':<18} {excess1['ann_ret']:>+10.1%} {'—':>10} {'—':>10}")
    print(f"{'IR(vs上证)':<18} {ir1:>10.2f} {'—':>10} {'—':>10}")
    print(f"{'超额(vs中证500)':<18} {excess2['ann_ret']:>+10.1%} {'—':>10} {'—':>10}")
    print(f"{'IR(vs中证500)':<18} {ir2:>10.2f} {'—':>10} {'—':>10}")
    print("=" * 64)

    print("\n逐年收益对比：")
    yearly_df = returns_df.copy()
    yearly_df["date"] = pd.to_datetime(yearly_df["date"])
    for year in sorted(yearly_df["date"].dt.year.unique()):
        year_slice = yearly_df[yearly_df["date"].dt.year == year]
        year_port = (1 + year_slice["port_ret"].dropna()).prod() - 1
        year_b1 = (1 + year_slice["bench_ret"].dropna()).prod() - 1
        year_b2 = (1 + year_slice["bench2_ret"].dropna()).prod() - 1
        year_n = year_slice["n_stocks"].mean()
        print(
            f"  {year}: 策略 {year_port:+.1%}  上证 {year_b1:+.1%}  中证500 {year_b2:+.1%}  "
            f"超额(上证) {year_port - year_b1:+.1%}  超额(中证) {year_port - year_b2:+.1%}  "
            f"(持仓 {year_n:.0f} 只)"
        )


# ── Data preparation helpers ─────────────────────────────
def winsorized_zscore(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=s.index)
    lower = valid.quantile(0.02)
    upper = valid.quantile(0.98)
    clipped = s.clip(lower, upper)
    std = clipped.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (clipped - clipped.mean()) / std


def load_equity_monthly_snapshot() -> pd.DataFrame:
    instruments = D.list_instruments(D.instruments("all"), start_time=START, end_time=END, as_list=True)
    instruments = [inst for inst in instruments if inst.startswith("sh") or inst.startswith("sz")]
    print(f"原始股票池：{len(instruments)} 只")

    fields = [
        "$close",
        "Ref($close, 250) / Ref($close, 20) - 1",   # 12-1 momentum
        "Std($close / Ref($close, 1) - 1, 20)",     # 20-day realised volatility
        "Mean($close, 60) / Std($close, 60)",       # price stability proxy
        "Mean($close * $volume, 60)",               # large-cap / liquidity proxy
    ]
    col_names = ["close", "mom_12_1", "vol_20", "stability_60", "liq_proxy_60"]

    raw = D.features(instruments, fields, start_time=START, end_time=END)
    raw.columns = col_names
    print(f"原始特征数据：{raw.shape}，整体非空率：{raw.notna().mean().mean():.1%}")

    monthly = raw.reset_index()
    monthly["date"] = pd.to_datetime(monthly["datetime"])
    monthly["ym"] = monthly["date"].dt.to_period("M")
    monthly = (
        monthly.sort_values("date")
        .groupby(["ym", "instrument"], as_index=False)
        .last()
    )
    return monthly


def build_benchmark_returns(code: str, name: str) -> pd.Series:
    bench = D.features([code], ["$close"], start_time=START, end_time=END)
    bench.columns = ["close"]
    bench = bench.reset_index()
    bench["date"] = pd.to_datetime(bench["datetime"])
    bench["ym"] = bench["date"].dt.to_period("M")
    bench_monthly = (
        bench.sort_values("date")
        .groupby("ym", as_index=False)
        .last()[["date", "close"]]
        .sort_values("date")
    )
    bench_monthly["ret"] = bench_monthly["close"].pct_change()
    ret = bench_monthly.dropna(subset=["ret"]).set_index("date")["ret"]
    ret.name = name
    print(f"基准 {name} 已加载：{len(ret)} 个月")
    return ret


def add_cross_section_score(monthly: pd.DataFrame) -> pd.DataFrame:
    def score_one_month(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["in_largecap"] = False
        g["score"] = np.nan

        cap_base = g.dropna(subset=["liq_proxy_60"])
        if len(cap_base) < MIN_LARGECAP_COUNT:
            return g

        cap_cut = cap_base["liq_proxy_60"].quantile(1 - LARGECAP_PCT)
        g.loc[g["liq_proxy_60"] >= cap_cut, "in_largecap"] = True

        largecap = g[g["in_largecap"]].copy()
        mask = largecap[["mom_12_1", "vol_20", "stability_60"]].notna().all(axis=1)
        if mask.sum() < MIN_HOLDING_COUNT * 2:
            return g

        scored = largecap.loc[mask].copy()
        scored["score"] = (
            0.55 * winsorized_zscore(scored["mom_12_1"])
            + 0.30 * winsorized_zscore(scored["stability_60"])
            - 0.15 * winsorized_zscore(scored["vol_20"])
        )
        g = g.merge(scored[["instrument", "score"]], on="instrument", how="left", suffixes=("", "_new"))
        g["score"] = g["score_new"].combine_first(g["score"])
        g = g.drop(columns=["score_new"])
        return g

    scored = monthly.groupby("ym", group_keys=False).apply(score_one_month)
    valid_months = scored.groupby("ym")["score"].count().gt(0).sum()
    avg_largecap = scored[scored["in_largecap"]].groupby("ym")["instrument"].count().mean()
    print(f"有效截面月数：{valid_months}，严格大盘代理平均每月：{avg_largecap:.0f} 只")
    return scored


# ── Backtest core ────────────────────────────────────────
def run_backtest(monthly: pd.DataFrame) -> pd.DataFrame:
    periods = sorted(monthly["ym"].unique())
    start_period = pd.Period(BACKTEST_START[:7], "M")
    results: list[dict[str, object]] = []

    for idx in range(len(periods) - 1):
        signal_period = periods[idx]
        hold_period = periods[idx + 1]
        if signal_period < start_period:
            continue

        current = monthly[
            (monthly["ym"] == signal_period) &
            (monthly["in_largecap"] == True)
        ].dropna(subset=["score", "close"])

        if len(current) < MIN_HOLDING_COUNT * 2:
            continue

        score_cut = current["score"].quantile(1 - TOP_PCT)
        selected = current[current["score"] >= score_cut][["instrument", "close"]].copy()
        selected = selected.rename(columns={"close": "entry_close"})

        next_month = monthly[monthly["ym"] == hold_period][["instrument", "close", "date"]].copy()
        next_month = next_month.rename(columns={"close": "exit_close", "date": "exit_date"})

        merged = selected.merge(next_month, on="instrument", how="inner").dropna(subset=["entry_close", "exit_close"])
        if len(merged) < MIN_HOLDING_COUNT:
            continue

        merged["ret"] = merged["exit_close"] / merged["entry_close"] - 1
        hold_date = pd.to_datetime(next_month["exit_date"].max()).strftime("%Y-%m-%d")

        results.append({
            "date": hold_date,
            "port_ret": merged["ret"].mean(),
            "n_stocks": int(len(merged)),
        })

    result_df = pd.DataFrame(results)
    print(f"有效调仓月数：{len(result_df)}")
    return result_df


def write_report(returns_df: pd.DataFrame) -> None:
    port = calc_return_metrics(returns_df["port_ret"])
    bench1 = calc_return_metrics(returns_df["bench_ret"])
    bench2 = calc_return_metrics(returns_df["bench2_ret"])
    excess1 = calc_return_metrics(returns_df["excess"])
    excess2 = calc_return_metrics(returns_df["excess2"])

    report_path = OUTDIR / f"{STRATEGY_NAME}_report.md"
    avg_n = returns_df["n_stocks"].mean() if not returns_df.empty else float("nan")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 严格大盘质量动量策略回测报告\n\n")
        f.write(f"**策略文件**：`strategies/factor/{STRATEGY_NAME}.py`  \n")
        f.write(f"**回测区间**：{BACKTEST_START} ~ {END}  \n")
        f.write(f"**调仓频率**：月度  \n")
        f.write(f"**基准**：{BENCHMARK_NAME} / {BENCHMARK2_NAME}  \n")
        f.write(f"**严格大盘代理**：每期按 `Mean($close * $volume, 60)` 取前 {LARGECAP_PCT:.0%}  \n")
        f.write(f"**平均持仓数**：{avg_n:.1f} 只  \n\n")

        f.write("## 策略逻辑\n\n")
        f.write("先在全市场 SH+SZ 股票中，按 60 日成交额代理筛出严格大盘桶，再在桶内按质量动量综合分数选股。分数不追求复杂，而是突出大盘股更稳的中期趋势与稳定性特征。\n\n")

        f.write("## 因子设计\n\n")
        f.write("| 因子 | 表达式 | 权重 | 方向 | 说明 |\n")
        f.write("|------|--------|------|------|------|\n")
        f.write("| 12-1 动量 | `Ref($close, 250) / Ref($close, 20) - 1` | 55% | 正向 | 捕捉机构持续定价的中期趋势 |\n")
        f.write("| 60 日稳定性 | `Mean($close, 60) / Std($close, 60)` | 30% | 正向 | 作为质量代理，偏好更稳的大盘龙头 |\n")
        f.write("| 20 日波动率 | `Std($close / Ref($close, 1) - 1, 20)` | 15% | 负向 | 仅做风险约束，不主导选股 |\n\n")

        f.write("## 回测汇总\n\n")
        f.write("| 指标 | 策略 | 上证指数 | 中证500 |\n")
        f.write("|------|------|----------|---------|\n")
        f.write(f"| 年化收益率 | {port['ann_ret']:+.2%} | {bench1['ann_ret']:+.2%} | {bench2['ann_ret']:+.2%} |\n")
        f.write(f"| 年化波动率 | {port['ann_vol']:.2%} | {bench1['ann_vol']:.2%} | {bench2['ann_vol']:.2%} |\n")
        f.write(f"| 夏普比率 | {port['sharpe']:.2f} | {bench1['sharpe']:.2f} | {bench2['sharpe']:.2f} |\n")
        f.write(f"| 最大回撤 | {port['max_dd']:.2%} | {bench1['max_dd']:.2%} | {bench2['max_dd']:.2%} |\n")
        f.write(f"| 月胜率 | {port['win_rate']:.2%} | — | — |\n")
        f.write(f"| 累计收益 | {port['cum_ret']:+.2%} | {bench1['cum_ret']:+.2%} | {bench2['cum_ret']:+.2%} |\n")
        f.write(f"| 超额收益（vs上证） | {excess1['ann_ret']:+.2%} | — | — |\n")
        f.write(f"| 信息比率（vs上证） | {(excess1['ann_ret'] / excess1['ann_vol']) if excess1['ann_vol'] > 0 else float('nan'):.2f} | — | — |\n")
        f.write(f"| 超额收益（vs中证500） | {excess2['ann_ret']:+.2%} | — | — |\n")
        f.write(f"| 信息比率（vs中证500） | {(excess2['ann_ret'] / excess2['ann_vol']) if excess2['ann_vol'] > 0 else float('nan'):.2f} | — | — |\n\n")

        f.write("## 反思与改进\n\n")
        f.write("- 当前没有 PE/PB、ROE、现金流等基本面历史，因此质量因子只能先用价格稳定性做代理。\n")
        f.write("- 当前大盘定义本质上是**流动性/成交额代理**，不是严格自由流通市值；若后续有市值历史，可直接替换筛选条件。\n")
        f.write("- 下一步最值得补的是估值与盈利质量历史数据，这样可以把大盘策略真正升级为 `质量 + 预期/盈利 + 趋势`。\n")

    print(f"[backtest] Saved: {report_path}")


# ── Main ────────────────────────────────────────────────
print("=" * 64)
print("严格大盘质量动量策略回测")
print(f"回测区间：{BACKTEST_START} ~ {END}")
print(f"严格大盘代理：每期前 {LARGECAP_PCT:.0%}  |  池内选股：前 {TOP_PCT:.0%}")
print("=" * 64)

print("\n[1/6] 拉取股票月度快照...")
monthly_snapshot = load_equity_monthly_snapshot()

print("\n[2/6] 构造严格大盘截面分数...")
monthly_snapshot = add_cross_section_score(monthly_snapshot)

print("\n[3/6] 运行月度回测...")
portfolio_df = run_backtest(monthly_snapshot)
if portfolio_df.empty:
    raise RuntimeError("Backtest produced no valid monthly observations.")

print("\n[4/6] 加载双基准并合并收益...")
bench_ret = build_benchmark_returns(BENCHMARK_CODE, "bench_ret")
bench2_ret = build_benchmark_returns(BENCHMARK2_CODE, "bench2_ret")

portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
combined = (
    portfolio_df.set_index("date")
    .join(bench_ret, how="left")
    .join(bench2_ret, how="left")
    .reset_index()
)
combined["excess"] = combined["port_ret"] - combined["bench_ret"]
combined["excess2"] = combined["port_ret"] - combined["bench2_ret"]
combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")

print("\n[5/6] 打印绩效汇总...")
print_summary(combined.copy())

print("\n[6/6] 保存回测结果与报告...")
nav_base = combined.copy()
nav_df = pd.DataFrame({
    "date": nav_base["date"],
    "strategy": (1 + nav_base["port_ret"]).cumprod(),
    "benchmark": (1 + nav_base["bench_ret"].fillna(0)).cumprod(),
    "benchmark2": (1 + nav_base["bench2_ret"].fillna(0)).cumprod(),
})
returns_df = combined[["date", "port_ret", "n_stocks", "bench_ret", "excess", "bench2_ret", "excess2"]].copy()
returns_df["n_stocks"] = returns_df["n_stocks"].astype(int)

save_backtest_results(STRATEGY_NAME, nav_df, returns_df)
write_report(returns_df)

print("\n完成。")
