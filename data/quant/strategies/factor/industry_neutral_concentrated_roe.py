"""
Industry-neutral concentrated strategy with PB-ROE composite factor (biweekly rebalance)
========================================================================================
Based on industry_neutral_concentrated.py with one key upgrade:
- Replaced independent VALUE(1/PB) + ROE with a composite PB-ROE factor: ROE/PB
  This rewards "high ROE at low PB" stocks — truly undervalued quality companies.

Data source : /projects/portal/data/quant/processed/stocks.db
"""
from __future__ import annotations
import sys, warnings, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH = "/projects/portal/data/quant/processed/stocks.db"
STRATEGY_NAME = "industry_neutral_concentrated_roe"
WARM_UP_START = "2021-01-01"
BACKTEST_START = "2022-01-01"
END = "2026-02-28"
OUTDIR = Path("/projects/portal/data/quant/backtest")

MCAP_KEEP_PCT = 0.70
TOP_PCT = 0.05            # Concentrated: top 5%
MAX_PER_INDUSTRY = 5       # Cap per industry
MIN_INDUSTRY_COUNT = 5
MIN_HOLDING = 20

# Factor weights — replaced VALUE + ROE with composite PB_ROE factor
# Original: MOM=0.25, VALUE=0.25, VOLCONF=0.15, LOWVOL=-0.15, SIZE=-0.10, REV=0.10
# New:      MOM=0.25, PB_ROE=0.30, VOLCONF=0.15, LOWVOL=-0.10, SIZE=-0.10, REV=0.10
# PB_ROE gets the combined budget of VALUE(25%) + bonus 5% for being a stronger signal
W_MOM = 0.25; W_PB_ROE = 0.30; W_VOLCONF = 0.15
W_LOWVOL = -0.10; W_SIZE = -0.10; W_REV = 0.10

# Transaction cost: single-side 1.5 bps
SINGLE_SIDE_COST = 0.00015

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


def calc_return_metrics(rets, freq=None):
    """Calculate return metrics. freq=None means auto-detect from length."""
    r = rets.dropna()
    if r.empty:
        return {k: float("nan") for k in
                ["ann_ret","ann_vol","sharpe","max_dd","cum_ret","win_rate"]}
    nav = (1 + r).cumprod()
    # For biweekly, we have ~24 periods/year
    if freq is None:
        n_years = len(r) / 24.0  # biweekly assumption
    else:
        n_years = len(r) / freq
    ann_ret = nav.iloc[-1] ** (1.0 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = r.std(ddof=0) * np.sqrt(1.0 / n_years * len(r)) if n_years > 0 else 0.0
    # Simpler: use periods per year
    periods_per_year = len(r) / n_years if n_years > 0 else 24
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    dd = nav / nav.cummax() - 1
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_dd=dd.min(), cum_ret=nav.iloc[-1]-1, win_rate=(r>0).mean())


def load_stock_data():
    print(f"[data] Loading from {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT code, date, close, volume, pb, pe_ttm, roe_ttm,
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
    roe_coverage = df["roe_ttm"].notna().mean()
    print(f"[data] ROE coverage: {roe_coverage:.1%}")
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
    df["log_cap"] = np.log(df["free_market_cap"].replace(0, np.nan))

    # PB-ROE composite factor: ROE / PB
    # Higher ROE at lower PB = more attractive (truly undervalued quality)
    # Only valid when PB > 0 and ROE is reasonable
    pb_valid = df["pb"].replace(0, np.nan)
    pb_valid = pb_valid.where(pb_valid > 0, np.nan)
    roe_valid = df["roe_ttm"].copy()
    roe_valid = roe_valid.where(roe_valid.abs() <= 200, np.nan)  # filter data errors
    # Only include positive ROE (negative ROE firms are low quality, should not benefit)
    roe_valid = roe_valid.where(roe_valid > 0, np.nan)
    df["pb_roe"] = roe_valid / pb_valid  # unit: %/x = ROE per unit of PB

    print(f"[factors] Done. Shape: {df.shape}")
    pb_roe_valid = df['pb_roe'].notna().sum()
    print(f"[factors] PB-ROE composite valid: {pb_roe_valid:,} / {len(df):,} ({pb_roe_valid/len(df):.1%})")
    return df


def sample_biweekly(df):
    """Sample at biweekly frequency: mid-month (~15th) and month-end snapshots."""
    print("[biweekly] Sampling biweekly snapshots ...")
    df = df.sort_values(["code", "date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Assign each row to a half-month period
    # half=1: day 1-15, half=2: day 16+
    df["half"] = np.where(df["day"] <= 15, 1, 2)
    # Create a period key: e.g. "2022-01-H1", "2022-01-H2"
    df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-H" + df["half"].astype(str)

    # Take last trading day of each half-month per stock
    biweekly = df.sort_values("date").groupby(["code", "period"], as_index=False).last()

    # Create a sortable period index for ordering
    biweekly["period_sort"] = biweekly["year"] * 100 + biweekly["month"]
    biweekly["period_sort"] = biweekly["period_sort"] * 10 + biweekly["half"]

    print(f"[biweekly] {len(biweekly):,} stock-period observations, "
          f"{biweekly['period'].nunique()} periods")
    return biweekly


def filter_universe(snap):
    print("[filter] Applying universe filters ...")
    required = ["close","mom_12_1","pb_roe","rvol_20","vol_confirm",
                "industry_code","free_market_cap","log_cap"]
    before = len(snap)
    snap = snap.dropna(subset=required)
    print(f"[filter] After dropna: {len(snap):,} (dropped {before-len(snap):,})")
    # Compute per-period market-cap cutoff and filter (avoid groupby().apply() for Pandas 3.0 compat)
    cutoffs = snap.groupby("period")["free_market_cap"].quantile(1 - MCAP_KEEP_PCT).rename("_cap_cutoff")
    snap = snap.join(cutoffs, on="period")
    snap = snap[snap["free_market_cap"] >= snap["_cap_cutoff"]].drop(columns=["_cap_cutoff"]).reset_index(drop=True)
    print(f"[filter] After cap filter (top {MCAP_KEEP_PCT:.0%}): {len(snap):,}")
    return snap


def score_within_industry(snap):
    print("[score] Industry-neutral scoring (with PB-ROE composite) ...")
    weights = {"mom_12_1": W_MOM, "pb_roe": W_PB_ROE,
               "vol_confirm": W_VOLCONF, "rvol_20": W_LOWVOL,
               "log_cap": W_SIZE, "rev_10": W_REV}
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
    # Pandas 3.0 compatible: iterate groups instead of groupby().apply()
    parts = []
    for (per, ind), g in snap.groupby(["period", "industry_code"]):
        parts.append(score_group(g))
    scored = pd.concat(parts, ignore_index=True)
    print(f"[score] Valid scored: {scored['score'].notna().sum():,}")
    return scored


def select_concentrated(signal):
    """
    Concentrated selection:
    1. Global top 5% by composite score
    2. Max 5 stocks per industry
    """
    cutoff = signal["score"].quantile(1 - TOP_PCT)
    selected = signal[signal["score"] >= cutoff].copy()

    # Cap per industry: keep top MAX_PER_INDUSTRY by score within each industry
    selected = (selected
                .sort_values("score", ascending=False)
                .groupby("industry_code", group_keys=False)
                .head(MAX_PER_INDUSTRY))

    return selected


def run_backtest(snap):
    print(f"[backtest] Running biweekly rebalance "
          f"(top {TOP_PCT:.0%}, max {MAX_PER_INDUSTRY}/industry, "
          f"cost {SINGLE_SIDE_COST:.4%}) ...")

    periods = sorted(snap["period"].unique())
    bt_start_sort = 2022 * 100 * 10 + 1 * 10 + 1  # 2022-01-H1

    # Build period -> sort key mapping
    period_info = snap.groupby("period").agg(
        period_sort=("period_sort", "first")
    ).reset_index().sort_values("period_sort")
    ordered_periods = period_info["period"].tolist()

    results = []
    prev_holdings = set()

    for i in range(len(ordered_periods) - 1):
        sig_period = ordered_periods[i]
        hold_period = ordered_periods[i + 1]

        sig_sort = period_info[period_info["period"] == sig_period]["period_sort"].values[0]
        if sig_sort < bt_start_sort:
            continue

        signal = snap[(snap["period"] == sig_period) & snap["score"].notna()].copy()
        if len(signal) < MIN_HOLDING * 2:
            continue

        # Concentrated selection
        selected = select_concentrated(signal)[["code", "close"]].copy()
        selected = selected.rename(columns={"close": "entry_close"})

        hold = snap[snap["period"] == hold_period][["code", "close", "date"]].copy()
        hold = hold.rename(columns={"close": "exit_close", "date": "exit_date"})

        merged = selected.merge(hold, on="code", how="inner").dropna(
            subset=["entry_close", "exit_close"])
        if len(merged) < MIN_HOLDING:
            continue

        curr_holdings = set(merged["code"].tolist())
        n_curr = len(curr_holdings)

        # Turnover
        if len(prev_holdings) == 0:
            turnover_buy = 1.0
            turnover_sell = 0.0
        else:
            sold = prev_holdings - curr_holdings
            bought = curr_holdings - prev_holdings
            n_prev = len(prev_holdings)
            turnover_sell = len(sold) / n_prev if n_prev > 0 else 0.0
            turnover_buy = len(bought) / n_curr if n_curr > 0 else 0.0

        tc = SINGLE_SIDE_COST * (turnover_sell + turnover_buy)

        merged["ret"] = merged["exit_close"] / merged["entry_close"] - 1
        gross_ret = merged["ret"].mean()
        net_ret = gross_ret - tc
        hold_date = merged["exit_date"].max()

        # Count industry distribution
        n_industries = signal[signal["code"].isin(curr_holdings)]["industry_code"].nunique()

        results.append({
            "date": hold_date.strftime("%Y-%m-%d"),
            "period": hold_period,
            "port_ret_gross": gross_ret,
            "port_ret": net_ret,
            "n_stocks": int(n_curr),
            "n_industries": int(n_industries),
            "turnover_sell": turnover_sell,
            "turnover_buy": turnover_buy,
            "tc": tc,
        })
        prev_holdings = curr_holdings

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        avg_to = (result_df["turnover_sell"] + result_df["turnover_buy"]).mean() / 2
        avg_tc = result_df["tc"].mean()
        total_tc = result_df["tc"].sum()
        avg_n = result_df["n_stocks"].mean()
        avg_ind = result_df["n_industries"].mean()
        print(f"[backtest] Valid periods: {len(result_df)}")
        print(f"[backtest] Avg holdings: {avg_n:.0f} stocks across {avg_ind:.0f} industries")
        print(f"[backtest] Avg single-side turnover: {avg_to:.1%}")
        print(f"[backtest] Avg period cost: {avg_tc:.4%}")
        print(f"[backtest] Total cost drag: {total_tc:.4%}")
    return result_df


CSV_DATA_DIR = Path("/root/qlib_data/daily")

# Map qlib-style code to CSV filename
_CODE_TO_CSV = {
    "sh000001": "000001.SH.csv",
    "sz000905": "000905.SZ.csv",
}


def load_benchmark_biweekly(code, name, snap_dates):
    """Load benchmark returns aligned to biweekly rebalance dates."""
    csv_file = CSV_DATA_DIR / _CODE_TO_CSV.get(code, f"{code}.csv")
    bench = pd.read_csv(csv_file, usecols=["date", "close"])
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.sort_values("date")
    bench = bench[(bench["date"] >= WARM_UP_START) & (bench["date"] <= END)]

    # For each snap_date, find the closest benchmark date
    snap_dates_dt = pd.to_datetime(snap_dates)
    bench_vals = []
    for dt in snap_dates_dt:
        mask = bench["date"] <= dt
        if mask.any():
            bench_vals.append(bench.loc[mask, "close"].iloc[-1])
        else:
            bench_vals.append(np.nan)

    bdf = pd.DataFrame({"date": snap_dates_dt, "close": bench_vals})
    bdf = bdf.sort_values("date")
    bdf["ret"] = bdf["close"].pct_change()
    ret = bdf.dropna(subset=["ret"]).set_index("date")["ret"]
    ret.name = name
    print(f"[bench] {name}: {len(ret)} periods")
    return ret


def print_summary(df, periods_per_year):
    port = calc_return_metrics(df["port_ret"], freq=periods_per_year)
    b1 = calc_return_metrics(df["bench_ret"], freq=periods_per_year)
    b2 = calc_return_metrics(df["bench2_ret"], freq=periods_per_year)
    ex1 = calc_return_metrics(df["excess"], freq=periods_per_year)
    ex2 = calc_return_metrics(df["excess2"], freq=periods_per_year)
    ir1 = ex1["ann_ret"]/ex1["ann_vol"] if ex1["ann_vol"]>0 else float("nan")
    ir2 = ex2["ann_ret"]/ex2["ann_vol"] if ex2["ann_vol"]>0 else float("nan")
    print("="*64)
    print(f"策略名称：{STRATEGY_NAME}")
    print(f"回测区间：{BACKTEST_START} ~ {END}")
    print(f"调仓频率：双周（biweekly）")
    print(f"选股方式：top {TOP_PCT:.0%} + 每行业最多 {MAX_PER_INDUSTRY} 只")
    print(f"因子：MOM={W_MOM} PB_ROE={W_PB_ROE} VOLCONF={W_VOLCONF} "
          f"LOWVOL={W_LOWVOL} SIZE={W_SIZE} REV={W_REV}")
    print(f"基    准：{BENCHMARK_NAME} / {BENCHMARK2_NAME}")
    print("="*64)
    print(f"{'指标':<18} {'策略':>10} {'上证指数':>10} {'中证500':>10}")
    print("-"*52)
    for label, m in [("年化收益率",lambda m:f"{m['ann_ret']:+.1%}"),
                     ("年化波动率",lambda m:f"{m['ann_vol']:.1%}"),
                     ("夏普比率",lambda m:f"{m['sharpe']:.2f}"),
                     ("最大回撤",lambda m:f"{m['max_dd']:.1%}")]:
        print(f"{label:<18} {m(port):>10} {m(b1):>10} {m(b2):>10}")
    print(f"{'胜率':<18} {port['win_rate']:>10.1%} {'—':>10} {'—':>10}")
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
        n_ind = ys["n_industries"].mean() if "n_industries" in ys.columns else 0
        print(f"  {year}: 策略 {pr:+.1%}  上证 {br1:+.1%}  中证500 {br2:+.1%}  "
              f"超额(上证) {pr-br1:+.1%}  超额(中证) {pr-br2:+.1%}  "
              f"(持仓 {n:.0f} 只/{n_ind:.0f} 行业)")


def write_report(df, periods_per_year):
    port = calc_return_metrics(df["port_ret"], freq=periods_per_year)
    b1 = calc_return_metrics(df["bench_ret"], freq=periods_per_year)
    b2 = calc_return_metrics(df["bench2_ret"], freq=periods_per_year)
    ex1 = calc_return_metrics(df["excess"], freq=periods_per_year)
    ex2 = calc_return_metrics(df["excess2"], freq=periods_per_year)
    ir1 = ex1["ann_ret"]/ex1["ann_vol"] if ex1["ann_vol"]>0 else float("nan")
    ir2 = ex2["ann_ret"]/ex2["ann_vol"] if ex2["ann_vol"]>0 else float("nan")
    avg_n = df["n_stocks"].mean()
    avg_ind = df["n_industries"].mean() if "n_industries" in df.columns else 0

    path = OUTDIR / f"{STRATEGY_NAME}_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 行业中性集中选股策略回测报告（双周调仓 + PB-ROE 复合因子）\n\n")
        f.write(f"**策略文件**：`strategies/factor/{STRATEGY_NAME}.py`  \n")
        f.write(f"**数据来源**：`processed/stocks.db` (SQLite)  \n")
        f.write(f"**回测区间**：{BACKTEST_START} ~ {END}  \n")
        f.write(f"**调仓频率**：双周（每月 15 日 + 月末）  \n")
        f.write(f"**基准**：{BENCHMARK_NAME} / {BENCHMARK2_NAME}  \n")
        f.write(f"**股票池**：SH+SZ，排除 BJ，自由流通市值前 {MCAP_KEEP_PCT:.0%}  \n")
        f.write(f"**行业中性化**：申万一级行业，因子在行业内 z-score  \n")
        f.write(f"**选股方式**：综合得分前 {TOP_PCT:.0%}，每行业最多 {MAX_PER_INDUSTRY} 只，等权持有  \n")
        f.write(f"**平均持仓**：{avg_n:.0f} 只 / {avg_ind:.0f} 个行业  \n\n")

        f.write("## 策略逻辑\n\n")
        f.write("本策略将原版的独立 VALUE(1/PB) 因子替换为 **PB-ROE 复合因子**：\n\n")
        f.write("### 核心改进：PB-ROE 复合因子\n\n")
        f.write('- **公式**：`ROE_TTM / PB`（ROE per unit of PB）\n')
        f.write('- **目的**：直接选出"高性价比"公司——盈利能力强但估值不贵\n')
        f.write("- **优势**：避免独立 ROE + 1/PB 的对冲效应（高ROE公司通常PB也高）\n")
        f.write("- **过滤**：ROE<=0 或 PB<=0 的公司不参与打分（排除亏损和破净异常）\n")
        f.write(f"- **权重**：{W_PB_ROE:.0%}（获得原 VALUE 25% 的预算 + 额外 5%）\n\n")

        f.write("## 因子设计\n\n")
        f.write("| 因子 | 表达式 | 权重 | 方向 | 变化 |\n")
        f.write("|------|--------|------|------|------|\n")
        f.write(f"| 12-1 动量 | `close_t-20 / close_t-250 - 1` | {W_MOM:.0%} | 正向 | 不变 |\n")
        f.write(f"| **PB-ROE 复合** | **`roe_ttm / PB`** | **{W_PB_ROE:.0%}** | **正向** | **替换 VALUE+ROE** |\n")
        f.write(f"| 量价确认 | `vol_ma20 / vol_ma120` | {abs(W_VOLCONF):.0%} | 正向 | 不变 |\n")
        f.write(f"| 低波动 | `std(daily_ret, 20)` | {abs(W_LOWVOL):.0%} | 负向 | 15%→10% |\n")
        f.write(f"| 小市值 | `log(free_market_cap)` | {abs(W_SIZE):.0%} | 负向 | 不变 |\n")
        f.write(f"| 短期反转 | `close_t-10 / close_t - 1` | {abs(W_REV):.0%} | 正向 | 不变 |\n\n")

        f.write("## 回测汇总\n\n")
        f.write("| 指标 | 策略 | 上证指数 | 中证500 |\n")
        f.write("|------|------|----------|----------|\n")
        f.write(f"| 年化收益率 | {port['ann_ret']:+.2%} | {b1['ann_ret']:+.2%} | {b2['ann_ret']:+.2%} |\n")
        f.write(f"| 年化波动率 | {port['ann_vol']:.2%} | {b1['ann_vol']:.2%} | {b2['ann_vol']:.2%} |\n")
        f.write(f"| 夏普比率 | {port['sharpe']:.2f} | {b1['sharpe']:.2f} | {b2['sharpe']:.2f} |\n")
        f.write(f"| 最大回撤 | {port['max_dd']:.2%} | {b1['max_dd']:.2%} | {b2['max_dd']:.2%} |\n")
        f.write(f"| 胜率 | {port['win_rate']:.2%} | — | — |\n")
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
            n_ind = ys["n_industries"].mean() if "n_industries" in ys.columns else 0
            f.write(f"| {year} | {pr:+.1%} | {br1_:+.1%} | {br2_:+.1%} | "
                    f"{pr-br1_:+.1%} | {pr-br2_:+.1%} | {n:.0f}/{n_ind:.0f}行业 |\n")

        # Transaction cost section
        gross_m = calc_return_metrics(df["port_ret_gross"], freq=periods_per_year)
        avg_tc = df["tc"].mean()
        total_tc = df["tc"].sum()
        avg_to = (df["turnover_sell"] + df["turnover_buy"]).mean() / 2
        f.write("\n## 交易成本分析\n\n")
        f.write(f"- **单边成本**：{SINGLE_SIDE_COST:.4%}（万分之{SINGLE_SIDE_COST*10000:.1f}）\n")
        f.write(f"- **平均单边换手率**：{avg_to:.1%}（每半月）\n")
        f.write(f"- **平均半月交易成本**：{avg_tc:.4%}\n")
        f.write(f"- **累计交易成本拖累**：{total_tc:.2%}\n\n")
        f.write("| 指标 | 毛收益 | 净收益 | 差异 |\n")
        f.write("|------|--------|--------|------|\n")
        f.write(f"| 年化收益率 | {gross_m['ann_ret']:+.2%} | {port['ann_ret']:+.2%} | {gross_m['ann_ret']-port['ann_ret']:.2%} |\n")
        f.write(f"| 夏普比率 | {gross_m['sharpe']:.2f} | {port['sharpe']:.2f} | {gross_m['sharpe']-port['sharpe']:.2f} |\n")
        f.write(f"| 累计收益 | {gross_m['cum_ret']:+.2%} | {port['cum_ret']:+.2%} | {gross_m['cum_ret']-port['cum_ret']:.2%} |\n")

        f.write("\n## 与原版对比\n\n")
        f.write("| 指标 | 原版(独立1/PB) | 独立ROE版 | 本版(PB-ROE复合) |\n")
        f.write("|------|---------------|-----------|-----------------|\n")
        f.write(f"| 年化收益率 | +18.6% | +13.2% | {port['ann_ret']:+.1%} |\n")
        f.write(f"| 夏普比率 | 0.78 | 0.56 | {port['sharpe']:.2f} |\n")
        f.write(f"| 最大回撤 | -24.1% | -26.2% | {port['max_dd']:.1%} |\n")
        f.write(f"| IR(vs上证) | 1.20 | 0.90 | {ir1:.2f} |\n")
        f.write(f"| 平均持仓 | ~93只 | ~89只 | {avg_n:.0f}只 |\n\n")

        f.write("## PB-ROE 复合因子的作用机制\n\n")
        f.write("1. **消除对冲效应**：独立的 1/PB 和 ROE 因子会互相抵消（高ROE→高PB→低1/PB），复合因子直接衡量性价比\n")
        f.write("2. **过滤低估值陷阱**：低PB但ROE也低的公司（破产边缘/周期底部），ROE/PB 不会很高\n")
        f.write("3. **过滤泡沫股**：高ROE但PB极高的公司（如100x PB的概念股），ROE/PB 也不会突出\n")
        f.write("4. **选出真正的价值股**：ROE=20% + PB=2x → ROE/PB=10，远优于 ROE=5% + PB=0.5x → ROE/PB=10\n\n")

        f.write("## 下一步改进方向\n\n")
        f.write("- 换仓缓冲带（Buffer）：进入 top 5%，退出 top 8%，减少边界股票频繁进出\n")
        f.write("- 因子 IC 监控：每期计算各因子 Rank IC，追踪因子有效性\n")
        f.write("- 得分加权：高分股票给更大权重，进一步集中 alpha\n")
        f.write("- ROE 变化率因子：用 ΔROE 环比改善替代/补充静态 ROE\n")

    print(f"[report] Saved: {path}")


if __name__ == "__main__":
    print("="*64)
    print("行业中性集中选股策略回测（双周调仓 + ROE 质量因子）")
    print(f"回测区间：{BACKTEST_START} ~ {END}")
    print(f"选股方式：top {TOP_PCT:.0%} + 每行业最多 {MAX_PER_INDUSTRY} 只")
    print(f"因子权重：MOM={W_MOM} PB_ROE={W_PB_ROE} VOLCONF={W_VOLCONF} "
          f"LOWVOL={W_LOWVOL} SIZE={W_SIZE} REV={W_REV}")
    print("="*64)
    print("\n[1/8] 加载股票日线数据 ...")
    daily = load_stock_data()

    print("\n[2/8] 计算日频因子（含 PB-ROE 复合因子）...")
    daily = compute_daily_factors(daily)

    print("\n[3/8] 双周采样 ...")
    biweekly = sample_biweekly(daily)
    del daily

    print("\n[4/8] 股票池过滤 ...")
    biweekly = filter_universe(biweekly)

    print("\n[5/8] 行业内标准化打分 ...")
    biweekly = score_within_industry(biweekly)

    print("\n[6/8] 运行双周回测 ...")
    portfolio_df = run_backtest(biweekly)
    if portfolio_df.empty:
        raise RuntimeError("Backtest produced no valid observations.")

    # Compute actual periods per year
    n_periods = len(portfolio_df)
    date_range_years = (pd.to_datetime(portfolio_df["date"].iloc[-1]) -
                        pd.to_datetime(portfolio_df["date"].iloc[0])).days / 365.25
    periods_per_year = n_periods / date_range_years if date_range_years > 0 else 24
    print(f"[info] {n_periods} periods over {date_range_years:.1f} years = "
          f"{periods_per_year:.1f} periods/year")

    print("\n[7/8] 加载双基准 ...")
    bench_ret = load_benchmark_biweekly(
        BENCHMARK_CODE, "bench_ret", portfolio_df["date"].tolist())
    bench2_ret = load_benchmark_biweekly(
        BENCHMARK2_CODE, "bench2_ret", portfolio_df["date"].tolist())

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
    print_summary(combined.copy(), periods_per_year)

    # Gross vs net summary
    gross_m = calc_return_metrics(combined["port_ret_gross"], freq=periods_per_year)
    net_m = calc_return_metrics(combined["port_ret"], freq=periods_per_year)
    avg_tc = combined["tc"].mean()
    total_tc = combined["tc"].sum()
    avg_to = (combined["turnover_sell"] + combined["turnover_buy"]).mean() / 2
    print(f"\n{'='*64}")
    print("交易成本影响分析")
    print(f"单边成本：{SINGLE_SIDE_COST:.4%}（万分之{SINGLE_SIDE_COST*10000:.1f}）")
    print(f"平均单边换手率：{avg_to:.1%}（每半月）")
    print(f"平均半月交易成本：{avg_tc:.4%}")
    print(f"累计交易成本拖累：{total_tc:.2%}")
    print(f"{'':18} {'毛收益':>12} {'净收益':>12} {'差异':>12}")
    print(f"{'年化收益率':18} {gross_m['ann_ret']:>+12.2%} {net_m['ann_ret']:>+12.2%} {gross_m['ann_ret']-net_m['ann_ret']:>12.2%}")
    print(f"{'夏普比率':18} {gross_m['sharpe']:>12.2f} {net_m['sharpe']:>12.2f} {gross_m['sharpe']-net_m['sharpe']:>12.2f}")
    print(f"{'累计收益':18} {gross_m['cum_ret']:>+12.2%} {net_m['cum_ret']:>+12.2%} {gross_m['cum_ret']-net_m['cum_ret']:>12.2%}")
    print("="*64)

    # Save outputs
    nav_df = pd.DataFrame({
        "date": combined["date"],
        "strategy": (1 + combined["port_ret"]).cumprod(),
        "strategy_gross": (1 + combined["port_ret_gross"]).cumprod(),
        "benchmark": (1 + combined["bench_ret"].fillna(0)).cumprod(),
        "benchmark2": (1 + combined["bench2_ret"].fillna(0)).cumprod(),
    })
    returns_df = combined[["date","port_ret","port_ret_gross","n_stocks","n_industries",
                           "turnover_sell","turnover_buy","tc",
                           "bench_ret","excess","bench2_ret","excess2"]].copy()
    returns_df["n_stocks"] = returns_df["n_stocks"].astype(int)

    nav_df.to_csv(OUTDIR / f"{STRATEGY_NAME}_nav.csv", index=False, float_format="%.7f")
    returns_df.to_csv(OUTDIR / f"{STRATEGY_NAME}_monthly_returns.csv", index=False, float_format="%.8f")
    print(f"[save] {STRATEGY_NAME}_nav.csv")
    print(f"[save] {STRATEGY_NAME}_monthly_returns.csv")

    write_report(combined.copy(), periods_per_year)
    print("\n✅ 完成。")
