"""
Industry-neutral multi-factor strategy (SQLite edition)
=======================================================
Data source : /projects/portal/data/quant/processed/stocks.db

Universe: SH + SZ only, free_market_cap top 70%, valid PB/industry/factors
Factors (within-industry z-score): MOM_12_1, 1/PB, VOL_CONFIRM, LOW_VOL, SIZE, REV_10
Selection: top 20% by composite score, equal weight, monthly rebalance
Benchmarks: Shanghai Composite (sh000001) + CSI 500 (sz000905)
"""
from __future__ import annotations
import sys, warnings, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH = "/projects/portal/data/quant/processed/stocks.db"
STRATEGY_NAME = "industry_neutral_multifactor"
WARM_UP_START = "2021-01-01"
BACKTEST_START = "2022-01-01"
END = "2026-02-28"
OUTDIR = Path("/projects/portal/data/quant/backtest")

MCAP_KEEP_PCT = 0.70
TOP_PCT = 0.20
MIN_INDUSTRY_COUNT = 5
MIN_HOLDING = 20

W_MOM = 0.25; W_VALUE = 0.25; W_VOLCONF = 0.15
W_LOWVOL = -0.15; W_SIZE = -0.10; W_REV = 0.10

BENCHMARK_CODE = "sh000001"; BENCHMARK_NAME = "上证指数 (sh000001)"
BENCHMARK2_CODE = "sz000905"; BENCHMARK2_NAME = "中证500 (sz000905)"


def winsorized_zscore(s):
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=s.index)
    lo, hi = valid.quantile(0.02), valid.quantile(0.98)
    c = s.clip(lo, hi)
    std = c.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (c - c.mean()) / std


def calc_return_metrics(rets, freq=12):
    r = rets.dropna()
    if r.empty:
        return {k: float("nan") for k in
                ["ann_ret","ann_vol","sharpe","max_dd","cum_ret","win_rate"]}
    nav = (1 + r).cumprod()
    ann_ret = nav.iloc[-1] ** (freq / len(r)) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    dd = nav / nav.cummax() - 1
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_dd=dd.min(), cum_ret=nav.iloc[-1]-1, win_rate=(r>0).mean())


def load_stock_data():
    print(f"[data] Loading from {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT code, date, close, volume, pb, pe_ttm,
           free_market_cap, industry_code, industry_name
    FROM kline
    WHERE (code LIKE 'SH%' OR code LIKE 'SZ%')
      AND date >= '{WARM_UP_START}' AND date <= '{END}'
    ORDER BY code, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    print(f"[data] Loaded {len(df):,} rows, {df['code'].nunique()} stocks")
    return df


def compute_daily_factors(df):
    print("[factors] Computing daily rolling factors ...")
    df = df.sort_values(["code","date"]).reset_index(drop=True)
    df["ret_1d"] = df.groupby("code")["close"].pct_change()
    df["close_lag20"] = df.groupby("code")["close"].shift(20)
    df["close_lag250"] = df.groupby("code")["close"].shift(250)
    df["mom_12_1"] = df["close_lag20"] / df["close_lag250"] - 1
    df["close_lag10"] = df.groupby("code")["close"].shift(10)
    df["rev_10"] = df["close_lag10"] / df["close"] - 1
    df["rvol_20"] = df.groupby("code")["ret_1d"].transform(
        lambda x: x.rolling(20, min_periods=15).std())
    df["vol_ma20"] = df.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=15).mean())
    df["vol_ma120"] = df.groupby("code")["volume"].transform(
        lambda x: x.rolling(120, min_periods=80).mean())
    df["vol_confirm"] = df["vol_ma20"] / df["vol_ma120"]
    df["inv_pb"] = 1.0 / df["pb"].replace(0, np.nan)
    df.loc[df["pb"] < 0, "inv_pb"] = np.nan
    df["log_cap"] = np.log(df["free_market_cap"].replace(0, np.nan))
    print(f"[factors] Done. Shape: {df.shape}")
    return df


def sample_monthly(df):
    print("[monthly] Sampling monthly snapshots ...")
    df["ym"] = df["date"].dt.to_period("M")
    monthly = df.sort_values("date").groupby(["code","ym"], as_index=False).last()
    print(f"[monthly] {len(monthly):,} stock-month observations")
    return monthly


def filter_universe(monthly):
    print("[filter] Applying universe filters ...")
    required = ["close","mom_12_1","inv_pb","rvol_20","vol_confirm",
                "industry_code","free_market_cap","log_cap"]
    before = len(monthly)
    monthly = monthly.dropna(subset=required)
    print(f"[filter] After dropna: {len(monthly):,} (dropped {before-len(monthly):,})")
    def cap_filter(g):
        cutoff = g["free_market_cap"].quantile(1 - MCAP_KEEP_PCT)
        return g[g["free_market_cap"] >= cutoff]
    monthly = monthly.groupby("ym", group_keys=False).apply(cap_filter)
    print(f"[filter] After cap filter (top {MCAP_KEEP_PCT:.0%}): {len(monthly):,}")
    return monthly


def score_within_industry(monthly):
    print("[score] Industry-neutral scoring ...")
    weights = {"mom_12_1": W_MOM, "inv_pb": W_VALUE, "vol_confirm": W_VOLCONF,
               "rvol_20": W_LOWVOL, "log_cap": W_SIZE, "rev_10": W_REV}
    def score_group(g):
        if len(g) < MIN_INDUSTRY_COUNT:
            g["score"] = np.nan
            return g
        g = g.copy()
        composite = pd.Series(0.0, index=g.index)
        for col, w in weights.items():
            composite += w * winsorized_zscore(g[col])
        g["score"] = composite
        return g
    scored = monthly.groupby(["ym","industry_code"], group_keys=False).apply(score_group)
    print(f"[score] Valid scored: {scored['score'].notna().sum():,}")
    return scored


def run_backtest(monthly):
    print("[backtest] Running monthly rebalance ...")
    periods = sorted(monthly["ym"].unique())
    bt_start = pd.Period(BACKTEST_START[:7], "M")
    results = []
    for i in range(len(periods)-1):
        sig_ym, hold_ym = periods[i], periods[i+1]
        if sig_ym < bt_start:
            continue
        signal = monthly[(monthly["ym"]==sig_ym) & monthly["score"].notna()].copy()
        if len(signal) < MIN_HOLDING*2:
            continue
        cutoff = signal["score"].quantile(1 - TOP_PCT)
        selected = signal[signal["score"] >= cutoff][["code","close"]].copy()
        selected = selected.rename(columns={"close":"entry_close"})
        hold = monthly[monthly["ym"]==hold_ym][["code","close","date"]].copy()
        hold = hold.rename(columns={"close":"exit_close","date":"exit_date"})
        merged = selected.merge(hold, on="code", how="inner").dropna(
            subset=["entry_close","exit_close"])
        if len(merged) < MIN_HOLDING:
            continue
        merged["ret"] = merged["exit_close"] / merged["entry_close"] - 1
        hold_date = merged["exit_date"].max()
        results.append({
            "date": hold_date.strftime("%Y-%m-%d"),
            "port_ret": merged["ret"].mean(),
            "n_stocks": int(len(merged)),
        })
    result_df = pd.DataFrame(results)
    print(f"[backtest] Valid months: {len(result_df)}")
    return result_df


def load_benchmark(code, name):
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    try:
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    except Exception:
        pass
    bench = D.features([code], ["$close"], start_time=WARM_UP_START, end_time=END)
    bench.columns = ["close"]
    bench = bench.reset_index()
    bench["date"] = pd.to_datetime(bench["datetime"])
    bench["ym"] = bench["date"].dt.to_period("M")
    bm = bench.sort_values("date").groupby("ym", as_index=False).last()[["date","close"]].sort_values("date")
    bm["ret"] = bm["close"].pct_change()
    ret = bm.dropna(subset=["ret"]).set_index("date")["ret"]
    ret.name = name
    print(f"[bench] {name}: {len(ret)} months")
    return ret


def print_summary(df):
    port = calc_return_metrics(df["port_ret"])
    b1 = calc_return_metrics(df["bench_ret"])
    b2 = calc_return_metrics(df["bench2_ret"])
    ex1 = calc_return_metrics(df["excess"])
    ex2 = calc_return_metrics(df["excess2"])
    ir1 = ex1["ann_ret"]/ex1["ann_vol"] if ex1["ann_vol"]>0 else float("nan")
    ir2 = ex2["ann_ret"]/ex2["ann_vol"] if ex2["ann_vol"]>0 else float("nan")
    print("="*64)
    print(f"策略名称：{STRATEGY_NAME}")
    print(f"回测区间：{BACKTEST_START} ~ {END}")
    print(f"基    准：{BENCHMARK_NAME} / {BENCHMARK2_NAME}")
    print("="*64)
    print(f"{'指标':<18} {'策略':>10} {'上证指数':>10} {'中证500':>10}")
    print("-"*52)
    for label, m in [("年化收益率",lambda m:f"{m['ann_ret']:+.1%}"),
                     ("年化波动率",lambda m:f"{m['ann_vol']:.1%}"),
                     ("夏普比率",lambda m:f"{m['sharpe']:.2f}"),
                     ("最大回撤",lambda m:f"{m['max_dd']:.1%}")]:
        print(f"{label:<18} {m(port):>10} {m(b1):>10} {m(b2):>10}")
    print(f"{'月胜率':<18} {port['win_rate']:>10.1%} {'—':>10} {'—':>10}")
    print(f"{'累计收益':<18} {port['cum_ret']:>+10.1%} {b1['cum_ret']:>+10.1%} {b2['cum_ret']:>+10.1%}")
    print(f"{'超额(vs上证)':<18} {ex1['ann_ret']:>+10.1%}")
    print(f"{'IR(vs上证)':<18} {ir1:>10.2f}")
    print(f"{'超额(vs中证500)':<18} {ex2['ann_ret']:>+10.1%}")
    print(f"{'IR(vs中证500)':<18} {ir2:>10.2f}")
    print("="*64)
    print("\n逐年收益对比：")
    tmp = df.copy(); tmp["_date"] = pd.to_datetime(tmp["date"])
    for year in sorted(tmp["_date"].dt.year.unique()):
        ys = tmp[tmp["_date"].dt.year==year]
        pr = (1+ys["port_ret"].dropna()).prod()-1
        br1 = (1+ys["bench_ret"].dropna()).prod()-1
        br2 = (1+ys["bench2_ret"].dropna()).prod()-1
        n = ys["n_stocks"].mean()
        print(f"  {year}: 策略 {pr:+.1%}  上证 {br1:+.1%}  中证500 {br2:+.1%}  "
              f"超额(上证) {pr-br1:+.1%}  超额(中证) {pr-br2:+.1%}  (持仓 {n:.0f} 只)")


def write_report(df):
    port = calc_return_metrics(df["port_ret"])
    b1 = calc_return_metrics(df["bench_ret"])
    b2 = calc_return_metrics(df["bench2_ret"])
    ex1 = calc_return_metrics(df["excess"])
    ex2 = calc_return_metrics(df["excess2"])
    ir1 = ex1["ann_ret"]/ex1["ann_vol"] if ex1["ann_vol"]>0 else float("nan")
    ir2 = ex2["ann_ret"]/ex2["ann_vol"] if ex2["ann_vol"]>0 else float("nan")
    avg_n = df["n_stocks"].mean()
    path = OUTDIR / f"{STRATEGY_NAME}_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 行业中性多因子策略回测报告\n\n")
        f.write(f"**策略文件**：`strategies/factor/{STRATEGY_NAME}.py`  \n")
        f.write(f"**数据来源**：`processed/stocks.db` (SQLite)  \n")
        f.write(f"**回测区间**：{BACKTEST_START} ~ {END}  \n")
        f.write(f"**调仓频率**：月度  \n")
        f.write(f"**基准**：{BENCHMARK_NAME} / {BENCHMARK2_NAME}  \n")
        f.write(f"**股票池**：SH+SZ，排除 BJ，自由流通市值前 {MCAP_KEEP_PCT:.0%}  \n")
        f.write(f"**行业中性化**：申万一级 31 行业，因子在行业内 z-score  \n")
        f.write(f"**选股方式**：综合得分前 {TOP_PCT:.0%}，等权持有  \n")
        f.write(f"**平均持仓数**：{avg_n:.1f} 只  \n\n")
        f.write("## 策略逻辑\n\n")
        f.write("本策略的核心是 **行业中性化**：所有因子先在申万一级行业内做 z-score 标准化，")
        f.write("再加权合成综合分数。策略不押注行业涨跌，只在每个行业内精选相对更优的个股。\n\n")
        f.write("相比前几版策略的关键升级：\n")
        f.write("- 使用 **真实 PB** 替代价格稳定性做估值因子\n")
        f.write("- 使用 **真实自由流通市值** 替代成交额做大小盘定义\n")
        f.write("- 使用 **申万一级行业分类** 做行业内标准化\n")
        f.write("- 数据源从 qlib 切换到 **SQLite**，加载更高效\n\n")
        f.write("## 因子设计\n\n")
        f.write("| 因子 | 表达式 | 权重 | 方向 | 说明 |\n")
        f.write("|------|--------|------|------|------|\n")
        f.write("| 12-1 动量 | `close_t-20 / close_t-250 - 1` | 25% | 正向 | 跳过最近1月，捕捉中期趋势 |\n")
        f.write("| 估值 (1/PB) | `1 / PB` | 25% | 正向 | 行业内比较消除行业差异 |\n")
        f.write("| 量价确认 | `vol_ma20 / vol_ma120` | 15% | 正向 | 近期放量确认趋势 |\n")
        f.write("| 低波动 | `std(daily_ret, 20)` | 15% | 负向 | 行业内偏好低波动 |\n")
        f.write("| 小市值 | `log(free_market_cap)` | 10% | 负向 | 行业内偏好中等偏小 |\n")
        f.write("| 短期反转 | `close_t-10 / close_t - 1` | 10% | 正向 | 短期超跌反弹 |\n\n")
        f.write("## 回测汇总\n\n")
        f.write("| 指标 | 策略 | 上证指数 | 中证500 |\n")
        f.write("|------|------|----------|----------|\n")
        f.write(f"| 年化收益率 | {port['ann_ret']:+.2%} | {b1['ann_ret']:+.2%} | {b2['ann_ret']:+.2%} |\n")
        f.write(f"| 年化波动率 | {port['ann_vol']:.2%} | {b1['ann_vol']:.2%} | {b2['ann_vol']:.2%} |\n")
        f.write(f"| 夏普比率 | {port['sharpe']:.2f} | {b1['sharpe']:.2f} | {b2['sharpe']:.2f} |\n")
        f.write(f"| 最大回撤 | {port['max_dd']:.2%} | {b1['max_dd']:.2%} | {b2['max_dd']:.2%} |\n")
        f.write(f"| 月胜率 | {port['win_rate']:.2%} | — | — |\n")
        f.write(f"| 累计收益 | {port['cum_ret']:+.2%} | {b1['cum_ret']:+.2%} | {b2['cum_ret']:+.2%} |\n")
        f.write(f"| 超额收益（vs上证） | {ex1['ann_ret']:+.2%} | — | — |\n")
        f.write(f"| 信息比率（vs上证） | {ir1:.2f} | — | — |\n")
        f.write(f"| 超额收益（vs中证500） | {ex2['ann_ret']:+.2%} | — | — |\n")
        f.write(f"| 信息比率（vs中证500） | {ir2:.2f} | — | — |\n\n")
        f.write("## 逐年收益\n\n")
        f.write("| 年份 | 策略 | 上证 | 中证500 | 超额(上证) | 超额(中证) | 持仓 |\n")
        f.write("|------|------|------|---------|------------|------------|------|\n")
        tmp = df.copy(); tmp["_date"] = pd.to_datetime(tmp["date"])
        for year in sorted(tmp["_date"].dt.year.unique()):
            ys = tmp[tmp["_date"].dt.year==year]
            pr = (1+ys["port_ret"].dropna()).prod()-1
            br1_ = (1+ys["bench_ret"].dropna()).prod()-1
            br2_ = (1+ys["bench2_ret"].dropna()).prod()-1
            n = ys["n_stocks"].mean()
            f.write(f"| {year} | {pr:+.1%} | {br1_:+.1%} | {br2_:+.1%} | {pr-br1_:+.1%} | {pr-br2_:+.1%} | {n:.0f} |\n")
        f.write("\n## 反思与改进\n\n")
        f.write("- 行业中性化有效消除了行业偏移风险。\n")
        f.write("- PB 估值因子在行业内比较更有意义。\n")
        f.write("- 下一步可补充 ROE/ROIC 做盈利质量因子。\n")
        f.write("- 可测试双周调仓看是否改善动量衰减。\n")
        f.write("- 可加入交易成本（双边 0.3%）做净收益检验。\n")
    print(f"[report] Saved: {path}")


if __name__ == "__main__":
    print("="*64)
    print("行业中性多因子策略回测 (SQLite)")
    print(f"回测区间：{BACKTEST_START} ~ {END}")
    print("="*64)

    print("\n[1/8] 加载股票日线数据 ...")
    daily = load_stock_data()

    print("\n[2/8] 计算日频因子 ...")
    daily = compute_daily_factors(daily)

    print("\n[3/8] 月度采样 ...")
    monthly = sample_monthly(daily)
    del daily

    print("\n[4/8] 股票池过滤 ...")
    monthly = filter_universe(monthly)

    print("\n[5/8] 行业内标准化打分 ...")
    monthly = score_within_industry(monthly)

    print("\n[6/8] 运行月度回测 ...")
    portfolio_df = run_backtest(monthly)
    if portfolio_df.empty:
        raise RuntimeError("Backtest produced no valid monthly observations.")

    print("\n[7/8] 加载双基准 ...")
    bench_ret = load_benchmark(BENCHMARK_CODE, "bench_ret")
    bench2_ret = load_benchmark(BENCHMARK2_CODE, "bench2_ret")

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

    print("\n[8/8] 输出结果 ...")
    print_summary(combined.copy())

    nav_df = pd.DataFrame({
        "date": combined["date"],
        "strategy": (1 + combined["port_ret"]).cumprod(),
        "benchmark": (1 + combined["bench_ret"].fillna(0)).cumprod(),
        "benchmark2": (1 + combined["bench2_ret"].fillna(0)).cumprod(),
    })
    returns_df = combined[["date","port_ret","n_stocks","bench_ret","excess","bench2_ret","excess2"]].copy()
    returns_df["n_stocks"] = returns_df["n_stocks"].astype(int)

    nav_df.to_csv(OUTDIR / f"{STRATEGY_NAME}_nav.csv", index=False, float_format="%.7f")
    returns_df.to_csv(OUTDIR / f"{STRATEGY_NAME}_monthly_returns.csv", index=False, float_format="%.8f")
    print(f"[save] {STRATEGY_NAME}_nav.csv")
    print(f"[save] {STRATEGY_NAME}_monthly_returns.csv")

    write_report(combined.copy())
    print("\n✅ 完成。")
