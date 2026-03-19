from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

warnings.filterwarnings("ignore")
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

STRATEGY_NAME = "pure_market_enhanced"
START = "2020-01-01"
BACKTEST_START = "2022-01-01"
END = "2026-02-28"
OUTDIR = Path("/projects/portal/data/quant/backtest")

BENCHMARK_CODE = "sh000001"
BENCHMARK_NAME = "上证指数 (sh000001)"
BENCHMARK2_CODE = "sz000905"
BENCHMARK2_NAME = "中证500 (sz000905)"

LIQUIDITY_KEEP_PCT = 0.50
TOP_PCT = 0.10
MIN_UNIVERSE_COUNT = 300
MIN_HOLDING_COUNT = 20


def save_backtest_results(strategy_name: str, nav_df: pd.DataFrame, returns_df: pd.DataFrame) -> None:
    base = OUTDIR / strategy_name
    nav_df.to_csv(f"{base}_nav.csv", index=False, float_format="%.7f")
    returns_df.to_csv(f"{base}_monthly_returns.csv", index=False, float_format="%.8f")
    print(f"[backtest] Saved: {base}_nav.csv")
    print(f"[backtest] Saved: {base}_monthly_returns.csv")


def calc_metrics(returns: pd.Series, freq: int = 12) -> dict[str, float]:
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
    port = calc_metrics(returns_df["port_ret"])
    bench1 = calc_metrics(returns_df["bench_ret"])
    bench2 = calc_metrics(returns_df["bench2_ret"])
    excess1 = calc_metrics(returns_df["excess"])
    excess2 = calc_metrics(returns_df["excess2"])
    ir1 = excess1["ann_ret"] / excess1["ann_vol"] if excess1["ann_vol"] > 0 else float("nan")
    ir2 = excess2["ann_ret"] / excess2["ann_vol"] if excess2["ann_vol"] > 0 else float("nan")

    print("=" * 66)
    print(f"策略名称：{STRATEGY_NAME}")
    print(f"回测区间：{BACKTEST_START} ~ {END}")
    print(f"股票池  ：SH+SZ，按 60 日成交额代理保留前 {LIQUIDITY_KEEP_PCT:.0%}")
    print(f"持仓规则：池内综合得分前 {TOP_PCT:.0%}，月度调仓")
    print(f"基    准：{BENCHMARK_NAME} / {BENCHMARK2_NAME}")
    print("=" * 66)
    print(f"{'指标':<18} {'策略':>10} {'上证指数':>10} {'中证500':>10}")
    print("-" * 54)
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
    print("=" * 66)


def winsorized_zscore(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan)
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


def load_monthly_snapshot() -> pd.DataFrame:
    instruments = D.list_instruments(D.instruments("all"), start_time=START, end_time=END, as_list=True)
    instruments = [inst for inst in instruments if inst.startswith("sh") or inst.startswith("sz")]
    print(f"原始股票池：{len(instruments)} 只")

    fields = [
        "$close",
        "Ref($close, 20) / Ref($close, 250) - 1",
        "$close / Ref($close, 120) - 1",
        "Mean($close * $volume, 20) / Mean($close * $volume, 120)",
        "Abs($close / Ref($close, 60) - 1) / Mean(Abs($close / Ref($close, 1) - 1), 60)",
        "Ref($close, 10) / $close - 1",
        "Std($close / Ref($close, 1) - 1, 20)",
        "Mean($close * $volume, 60)",
    ]
    col_names = [
        "close",
        "mom_12_1",
        "rs_6m",
        "vol_confirm",
        "eff_60",
        "rev_10",
        "vol_20",
        "liq_proxy_60",
    ]

    raw = D.features(instruments, fields, start_time=START, end_time=END)
    raw.columns = col_names
    print(f"原始特征数据：{raw.shape}，整体非空率：{raw.notna().mean().mean():.1%}")

    monthly = raw.reset_index()
    monthly["date"] = pd.to_datetime(monthly["datetime"])
    monthly["ym"] = monthly["date"].dt.to_period("M")
    monthly = monthly.sort_values("date").groupby(["ym", "instrument"], as_index=False).last()
    return monthly


def build_benchmark_returns(code: str, name: str) -> pd.Series:
    bench = D.features([code], ["$close"], start_time=START, end_time=END)
    bench.columns = ["close"]
    bench = bench.reset_index()
    bench["date"] = pd.to_datetime(bench["datetime"])
    bench["ym"] = bench["date"].dt.to_period("M")
    bench_monthly = bench.sort_values("date").groupby("ym", as_index=False).last()[["date", "close"]].sort_values("date")
    bench_monthly["ret"] = bench_monthly["close"].pct_change()
    ret = bench_monthly.dropna(subset=["ret"]).set_index("date")["ret"]
    ret.name = name
    print(f"基准 {name} 已加载：{len(ret)} 个月")
    return ret


def add_cross_section_score(monthly: pd.DataFrame) -> pd.DataFrame:
    def score_one_month(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["in_universe"] = False
        g["score"] = np.nan

        cap_base = g.dropna(subset=["liq_proxy_60"])
        if len(cap_base) < MIN_UNIVERSE_COUNT:
            return g

        cap_cut = cap_base["liq_proxy_60"].quantile(1 - LIQUIDITY_KEEP_PCT)
        g.loc[g["liq_proxy_60"] >= cap_cut, "in_universe"] = True

        liquid = g[g["in_universe"]].copy()
        factor_cols = ["mom_12_1", "rs_6m", "vol_confirm", "eff_60", "rev_10", "vol_20"]
        mask = liquid[factor_cols].notna().all(axis=1)
        if mask.sum() < MIN_HOLDING_COUNT * 4:
            return g

        scored = liquid.loc[mask].copy()
        scored["score"] = (
            0.35 * winsorized_zscore(scored["mom_12_1"])
            + 0.20 * winsorized_zscore(scored["rs_6m"])
            + 0.15 * winsorized_zscore(scored["vol_confirm"])
            + 0.15 * winsorized_zscore(scored["eff_60"])
            + 0.15 * winsorized_zscore(scored["rev_10"])
            - 0.20 * winsorized_zscore(scored["vol_20"])
        )
        g = g.merge(scored[["instrument", "score"]], on="instrument", how="left", suffixes=("", "_new"))
        g["score"] = g["score_new"].combine_first(g["score"])
        g = g.drop(columns=["score_new"])
        return g

    scored = monthly.groupby("ym", group_keys=False).apply(score_one_month)
    valid_months = scored.groupby("ym")["score"].count().gt(0).sum()
    avg_universe = scored[scored["in_universe"]].groupby("ym")["instrument"].count().mean()
    print(f"有效截面月数：{valid_months}，平均候选股票数：{avg_universe:.0f} 只")
    return scored


def run_backtest(monthly: pd.DataFrame) -> pd.DataFrame:
    periods = sorted(monthly["ym"].unique())
    start_period = pd.Period(BACKTEST_START[:7], "M")
    results: list[dict[str, object]] = []

    for idx in range(len(periods) - 1):
        signal_period = periods[idx]
        hold_period = periods[idx + 1]
        if signal_period < start_period:
            continue

        current = monthly[(monthly["ym"] == signal_period) & (monthly["in_universe"] == True)].dropna(subset=["score", "close"])
        if len(current) < MIN_HOLDING_COUNT * 4:
            continue

        score_cut = current["score"].quantile(1 - TOP_PCT)
        selected = current[current["score"] >= score_cut][["instrument", "close"]].copy().rename(columns={"close": "entry_close"})

        next_month = monthly[monthly["ym"] == hold_period][["instrument", "close", "date"]].copy()
        next_month = next_month.rename(columns={"close": "exit_close", "date": "exit_date"})

        merged = selected.merge(next_month, on="instrument", how="inner").dropna(subset=["entry_close", "exit_close"])
        if len(merged) < MIN_HOLDING_COUNT:
            continue

        merged["ret"] = merged["exit_close"] / merged["entry_close"] - 1
        hold_date = pd.to_datetime(next_month["exit_date"].max()).strftime("%Y-%m-%d")
        signal_date = pd.to_datetime(current["date"].max()).strftime("%Y-%m-%d")
        results.append({
            "signal_date": signal_date,
            "date": hold_date,
            "port_ret": merged["ret"].mean(),
            "n_stocks": int(len(merged)),
        })

    result_df = pd.DataFrame(results)
    print(f"有效调仓月数：{len(result_df)}")
    return result_df


def write_report(returns_df: pd.DataFrame) -> None:
    port = calc_metrics(returns_df["port_ret"])
    bench1 = calc_metrics(returns_df["bench_ret"])
    bench2 = calc_metrics(returns_df["bench2_ret"])
    excess1 = calc_metrics(returns_df["excess"])
    excess2 = calc_metrics(returns_df["excess2"])
    ir1 = excess1["ann_ret"] / excess1["ann_vol"] if excess1["ann_vol"] > 0 else float("nan")
    ir2 = excess2["ann_ret"] / excess2["ann_vol"] if excess2["ann_vol"] > 0 else float("nan")
    avg_n = returns_df["n_stocks"].mean() if not returns_df.empty else float("nan")

    report_path = OUTDIR / f"{STRATEGY_NAME}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 纯市场增强策略回测报告\n\n")
        f.write(f"**策略文件**：`strategies/factor/{STRATEGY_NAME}.py`  \n")
        f.write(f"**回测区间**：{BACKTEST_START} ~ {END}  \n")
        f.write("**调仓频率**：月度  \n")
        f.write(f"**基准**：{BENCHMARK_NAME} / {BENCHMARK2_NAME}  \n")
        f.write(f"**流动性过滤**：按 60 日成交额代理保留前 {LIQUIDITY_KEEP_PCT:.0%}  \n")
        f.write(f"**选股方式**：综合得分前 {TOP_PCT:.0%}，等权持有  \n")
        f.write("**交易成本**：未纳入本版回测  \n")
        f.write(f"**平均持仓数**：{avg_n:.1f} 只  \n\n")

        f.write("## 因子设计\n\n")
        f.write("| 因子 | 表达式 | 权重 | 方向 |\n")
        f.write("|------|--------|------|------|\n")
        f.write("| 12-1 动量 | `Ref($close, 20) / Ref($close, 250) - 1` | 35% | 正向 |\n")
        f.write("| 6M 强弱 | `$close / Ref($close, 120) - 1` | 20% | 正向 |\n")
        f.write("| 量价确认 | `Mean($close*$volume,20) / Mean($close*$volume,120)` | 15% | 正向 |\n")
        f.write("| 趋势效率 | `Abs($close / Ref($close,60)-1) / Mean(Abs($close / Ref($close,1)-1),60)` | 15% | 正向 |\n")
        f.write("| 短期反转 | `Ref($close,10) / $close - 1` | 15% | 正向 |\n")
        f.write("| 20 日波动率 | `Std($close / Ref($close,1) - 1, 20)` | 20% | 负向 |\n\n")

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
        f.write(f"| 信息比率（vs上证） | {ir1:.2f} | — | — |\n")
        f.write(f"| 超额收益（vs中证500） | {excess2['ann_ret']:+.2%} | — | — |\n")
        f.write(f"| 信息比率（vs中证500） | {ir2:.2f} | — | — |\n\n")

        f.write("## 改进方向\n\n")
        f.write("- 下一步应加入双边交易成本检验换手敏感性。\n")
        f.write("- 若补齐行业分类，可升级为行业内相对强弱版本。\n")
        f.write("- 若补齐估值和质量历史，可扩展为市场行为 + 基本面混合策略。\n")
    print(f"[backtest] Saved: {report_path}")


print("=" * 66)
print("纯市场增强策略回测")
print(f"回测区间：{BACKTEST_START} ~ {END}")
print(f"流动性过滤：前 {LIQUIDITY_KEEP_PCT:.0%}  |  池内选股：前 {TOP_PCT:.0%}")
print("=" * 66)

print("\n[1/6] 拉取股票月度快照...")
monthly_snapshot = load_monthly_snapshot()

print("\n[2/6] 构造纯市场增强分数...")
monthly_snapshot = add_cross_section_score(monthly_snapshot)

print("\n[3/6] 运行月度回测...")
portfolio_df = run_backtest(monthly_snapshot)
if portfolio_df.empty:
    raise RuntimeError("Backtest produced no valid monthly observations.")

print("\n[4/6] 加载双基准并合并收益...")
bench_ret = build_benchmark_returns(BENCHMARK_CODE, "bench_ret")
bench2_ret = build_benchmark_returns(BENCHMARK2_CODE, "bench2_ret")
portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
combined = portfolio_df.set_index("date").join(bench_ret, how="left").join(bench2_ret, how="left").reset_index()
combined["signal_date"] = pd.to_datetime(combined["signal_date"])
combined["excess"] = combined["port_ret"] - combined["bench_ret"]
combined["excess2"] = combined["port_ret"] - combined["bench2_ret"]
combined["date"] = pd.to_datetime(combined["date"])

print("\n[5/6] 打印绩效汇总...")
print_summary(combined.assign(date=combined["date"].dt.strftime("%Y-%m-%d")).copy())

print("\n[6/6] 保存回测结果与报告...")
nav_period_df = pd.DataFrame({
    "date": combined["date"].dt.strftime("%Y-%m-%d"),
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
returns_df["date"] = pd.to_datetime(returns_df["date"]).dt.strftime("%Y-%m-%d")
returns_df["n_stocks"] = returns_df["n_stocks"].astype(int)

save_backtest_results(STRATEGY_NAME, nav_df, returns_df)
write_report(returns_df)

print("\n完成。")
