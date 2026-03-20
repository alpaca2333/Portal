"""
factor_ic_analysis.py
=====================
计算每个因子的逐期 Rank IC，输出：
1. backtest/factor_ic_analysis.csv       — 逐期每因子 IC 时间序列
2. backtest/factor_ic_report.md          — 汇总表 + 逐年 IC 热力图
"""
from __future__ import annotations
import sys, warnings, sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
DB_PATH = str(BASE / "processed/stocks.db")
OUTDIR  = BASE / "backtest"
OUTDIR.mkdir(exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────────────────────────
WARM_UP_START  = "2021-01-01"
BACKTEST_START = "2022-01-01"
END            = "2026-02-28"
MCAP_KEEP_PCT  = 0.70       # 市值过滤，保留前 70%

# 要分析的因子（列名 → 方向，+1=越大越好，-1=越小越好）
FACTORS = {
    "mom_12_1":   +1,
    "inv_pb":     +1,
    "vol_confirm":+1,
    "rvol_20":    -1,
    "log_cap":    -1,
    "rev_10":     +1,
}

# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"[data] Connecting to {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT code, date, close, volume, pb, free_market_cap, industry_code
        FROM kline
        WHERE (code LIKE 'SH%' OR code LIKE 'SZ%')
          AND date >= '{WARM_UP_START}' AND date <= '{END}'
        ORDER BY code, date
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    print(f"[data] {len(df):,} rows, {df['code'].nunique()} stocks")
    return df

# ── 因子计算 ──────────────────────────────────────────────────────────────────
def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    print("[factors] Computing ...")
    df = df.sort_values(["code","date"]).reset_index(drop=True)
    g = df.groupby("code")

    df["ret_1d"]      = g["close"].pct_change()
    df["close_lag20"] = g["close"].shift(20)
    df["close_lag250"]= g["close"].shift(250)
    df["mom_12_1"]    = df["close_lag20"] / df["close_lag250"] - 1
    df["close_lag10"] = g["close"].shift(10)
    df["rev_10"]      = df["close_lag10"] / df["close"] - 1
    df["rvol_20"]     = g["ret_1d"].transform(
        lambda x: x.rolling(20, min_periods=15).std())
    df["vol_ma20"]    = g["volume"].transform(
        lambda x: x.rolling(20, min_periods=15).mean())
    df["vol_ma120"]   = g["volume"].transform(
        lambda x: x.rolling(120, min_periods=80).mean())
    df["vol_confirm"] = df["vol_ma20"] / df["vol_ma120"]
    pb = df["pb"].replace(0, np.nan)
    df["inv_pb"]      = np.where(pb > 0, 1.0 / pb, np.nan)
    df["log_cap"]     = np.log(df["free_market_cap"].replace(0, np.nan))
    print(f"[factors] Done. Shape: {df.shape}")
    return df

# ── 双周采样 ──────────────────────────────────────────────────────────────────
def sample_biweekly(df: pd.DataFrame) -> pd.DataFrame:
    print("[sample] Biweekly ...")
    df = df.copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["half"]  = np.where(df["date"].dt.day <= 15, 1, 2)
    df["period"] = (df["year"].astype(str) + "-"
                    + df["month"].astype(str).str.zfill(2)
                    + "-H" + df["half"].astype(str))
    snap = df.sort_values("date").groupby(["code","period"], as_index=False).last()
    snap["period_sort"] = (snap["year"]*100 + snap["month"])*10 + snap["half"]
    print(f"[sample] {snap['period'].nunique()} periods")
    return snap

# ── 股票池过滤 ────────────────────────────────────────────────────────────────
def filter_universe(snap: pd.DataFrame) -> pd.DataFrame:
    required = ["close","mom_12_1","inv_pb","rvol_20","vol_confirm","log_cap",
                "free_market_cap","industry_code"]
    snap = snap.dropna(subset=required)
    cutoffs = (snap.groupby("period")["free_market_cap"]
               .quantile(1 - MCAP_KEEP_PCT).rename("_cut"))
    snap = snap.join(cutoffs, on="period")
    snap = snap[snap["free_market_cap"] >= snap["_cut"]].drop(columns=["_cut"])
    return snap.reset_index(drop=True)

# ── 计算未来收益（下一期） ────────────────────────────────────────────────────
def compute_forward_returns(snap: pd.DataFrame) -> pd.DataFrame:
    """
    每只股票 t 期的 forward return = (t+1 期收盘价) / (t 期收盘价) - 1
    通过 shift(-1) 在 period_sort 维度对齐。
    """
    snap = snap.sort_values(["code","period_sort"]).copy()
    snap["next_close"] = snap.groupby("code")["close"].shift(-1)
    snap["fwd_ret"] = snap["next_close"] / snap["close"] - 1
    return snap

# ── Rank IC 计算 ──────────────────────────────────────────────────────────────
def compute_rank_ic(snap: pd.DataFrame) -> pd.DataFrame:
    """逐期计算每个因子的 Rank IC（Spearman 相关系数）"""
    print("[IC] Computing Rank IC per period ...")

    bt_start = int(BACKTEST_START.replace("-","")[:6]) * 10 + 1  # 2022-01-H1

    periods_info = (snap[["period","period_sort"]]
                    .drop_duplicates()
                    .sort_values("period_sort"))

    records = []
    for _, row in periods_info.iterrows():
        period = row["period"]
        psort  = row["period_sort"]
        if psort < bt_start:
            continue

        grp = snap[(snap["period"] == period) & snap["fwd_ret"].notna()].copy()
        if len(grp) < 30:
            continue

        rec = {"period": period, "period_sort": int(psort), "n_stocks": len(grp)}
        for fac, direction in FACTORS.items():
            col = grp[fac].dropna()
            if len(col) < 20:
                rec[f"{fac}_ic"] = np.nan
                continue
            fwd = grp.loc[col.index, "fwd_ret"]
            # Spearman rank correlation
            ic_raw, _ = spearmanr(col, fwd)
            # 乘以方向，正向因子 IC > 0 才算有效
            rec[f"{fac}_ic"] = float(ic_raw) * direction
        records.append(rec)

    ic_df = pd.DataFrame(records)
    print(f"[IC] Computed {len(ic_df)} periods")
    return ic_df

# ── 年份提取 ──────────────────────────────────────────────────────────────────
def period_to_year(period: str) -> int:
    return int(period[:4])

# ── 报告生成 ──────────────────────────────────────────────────────────────────
def write_report(ic_df: pd.DataFrame):
    ic_cols = [c for c in ic_df.columns if c.endswith("_ic")]
    factor_names = [c.replace("_ic","") for c in ic_cols]

    # ── 全期汇总统计 ──────────────────────────────────────────────────────────
    summary = []
    for col, fname in zip(ic_cols, factor_names):
        s = ic_df[col].dropna()
        mean_ic  = s.mean()
        std_ic   = s.std()
        icir     = mean_ic / std_ic if std_ic > 0 else np.nan
        positive_pct = (s > 0).mean()
        summary.append({
            "因子": fname,
            "IC均值": mean_ic,
            "IC标准差": std_ic,
            "ICIR": icir,
            "IC>0占比": positive_pct,
        })
    summary_df = pd.DataFrame(summary).set_index("因子")

    # ── 逐年 IC 均值 ──────────────────────────────────────────────────────────
    ic_df["year"] = ic_df["period"].apply(period_to_year)
    yearly = {}
    for year in sorted(ic_df["year"].unique()):
        yg = ic_df[ic_df["year"] == year]
        yearly[year] = {fname: yg[f"{fname}_ic"].mean()
                        for fname in factor_names}
    yearly_df = pd.DataFrame(yearly).T

    # ── 写 Markdown ───────────────────────────────────────────────────────────
    path = OUTDIR / "factor_ic_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 因子 IC 分析报告\n\n")
        f.write(f"**分析区间**：{BACKTEST_START} ~ {END}  \n")
        f.write(f"**调仓频率**：双周（biweekly）  \n")
        f.write(f"**股票池**：SH+SZ，自由流通市值前 {MCAP_KEEP_PCT:.0%}  \n")
        f.write(f"**IC类型**：Rank IC（Spearman，已乘以因子方向，IC>0 代表有效）  \n\n")

        # 全期汇总
        f.write("## 全期汇总（2022-2026）\n\n")
        f.write("| 因子 | IC均值 | IC标准差 | ICIR | IC>0占比 | 评级 |\n")
        f.write("|------|--------|---------|------|---------|------|\n")
        for _, r in summary_df.iterrows():
            fname = r.name
            rating = ("🟢 优秀" if abs(r["ICIR"]) >= 0.5
                      else "🟡 一般" if abs(r["ICIR"]) >= 0.3
                      else "🔴 较弱")
            f.write(f"| {fname} | {r['IC均值']:+.4f} | {r['IC标准差']:.4f} | "
                    f"{r['ICIR']:+.2f} | {r['IC>0占比']:.1%} | {rating} |\n")

        # 逐年 IC 热力图（文字版）
        f.write("\n## 逐年 IC 均值\n\n")
        f.write("> IC > +0.03 = 🟢有效，IC < -0.01 = 🔴反向失效，其余 = 🟡弱\n\n")
        header = "| 年份 |" + "".join(f" {n} |" for n in factor_names)
        sep    = "|------|" + "".join("--------|" for _ in factor_names)
        f.write(header + "\n" + sep + "\n")
        for year in sorted(yearly_df.index):
            row_str = f"| {year} |"
            for fname in factor_names:
                v = yearly_df.loc[year, fname]
                if np.isnan(v):
                    cell = "  N/A  "
                elif v > 0.03:
                    cell = f"🟢{v:+.3f}"
                elif v < -0.01:
                    cell = f"🔴{v:+.3f}"
                else:
                    cell = f"🟡{v:+.3f}"
                row_str += f" {cell} |"
            f.write(row_str + "\n")

        # 逐年详细统计
        f.write("\n## 逐年详细统计\n\n")
        for year in sorted(ic_df["year"].unique()):
            yg = ic_df[ic_df["year"] == year]
            f.write(f"### {year} 年（{len(yg)} 个调仓期）\n\n")
            f.write("| 因子 | IC均值 | IC标准差 | ICIR | IC>0占比 |\n")
            f.write("|------|--------|---------|------|----------|\n")
            for fname in factor_names:
                s = yg[f"{fname}_ic"].dropna()
                if len(s) < 2:
                    f.write(f"| {fname} | N/A | N/A | N/A | N/A |\n")
                    continue
                mean_ic = s.mean(); std_ic = s.std()
                icir = mean_ic / std_ic if std_ic > 0 else np.nan
                f.write(f"| {fname} | {mean_ic:+.4f} | {std_ic:.4f} | "
                        f"{icir:+.2f} | {(s>0).mean():.1%} |\n")
            f.write("\n")

        # 结论
        f.write("## 关键结论\n\n")
        # 找最强 / 最弱因子
        best  = summary_df["ICIR"].idxmax()
        worst = summary_df["ICIR"].idxmin()
        # 找 2025 年最反转的因子
        if 2025 in yearly_df.index:
            worst_2025 = yearly_df.loc[2025].idxmin()
            best_2025  = yearly_df.loc[2025].idxmax()
            f.write(f"- **全期最强因子**：`{best}`（ICIR = {summary_df.loc[best,'ICIR']:+.2f}）\n")
            f.write(f"- **全期最弱因子**：`{worst}`（ICIR = {summary_df.loc[worst,'ICIR']:+.2f}）\n")
            f.write(f"- **2025年最失效因子**：`{worst_2025}`（年均IC = {yearly_df.loc[2025, worst_2025]:+.4f}）\n")
            f.write(f"- **2025年最有效因子**：`{best_2025}`（年均IC = {yearly_df.loc[2025, best_2025]:+.4f}）\n")
        else:
            f.write(f"- **全期最强因子**：`{best}`（ICIR = {summary_df.loc[best,'ICIR']:+.2f}）\n")
            f.write(f"- **全期最弱因子**：`{worst}`（ICIR = {summary_df.loc[worst,'ICIR']:+.2f}）\n")

        # 权重建议
        f.write("\n## 因子权重调整建议\n\n")
        f.write("基于 ICIR 排序，建议权重调整方向：\n\n")
        f.write("| 因子 | 当前权重 | 建议方向 | 依据 |\n")
        f.write("|------|---------|---------|------|\n")
        current_weights = {
            "mom_12_1": "+25%", "inv_pb": "+25%", "vol_confirm": "+15%",
            "rvol_20": "-15%", "log_cap": "-10%", "rev_10": "+10%",
        }
        for _, r in summary_df.sort_values("ICIR", ascending=False).iterrows():
            fname = r.name
            icir  = r["ICIR"]
            w     = current_weights.get(fname, "?")
            if icir >= 0.5:
                advice = "⬆️ 可加大"
            elif icir >= 0.3:
                advice = "➡️ 保持"
            elif icir >= 0.0:
                advice = "⬇️ 考虑降低"
            else:
                advice = "❌ 考虑移除或反向"
            f.write(f"| {fname} | {w} | {advice} | ICIR={icir:+.2f} |\n")

    print(f"[report] Saved: {path}")
    return summary_df, yearly_df

# ── 控制台输出 ────────────────────────────────────────────────────────────────
def print_summary(summary_df: pd.DataFrame, yearly_df: pd.DataFrame):
    print("\n" + "="*70)
    print("因子 IC 分析汇总（全期 2022-2026）")
    print("="*70)
    print(f"{'因子':<15} {'IC均值':>8} {'IC标准差':>8} {'ICIR':>7} {'IC>0%':>7}  评级")
    print("-"*70)
    for _, r in summary_df.sort_values("ICIR", ascending=False).iterrows():
        rating = ("优秀✅" if abs(r["ICIR"]) >= 0.5
                  else "一般⚠️ " if abs(r["ICIR"]) >= 0.3
                  else "弱  ❌")
        print(f"{r.name:<15} {r['IC均值']:>+8.4f} {r['IC标准差']:>8.4f} "
              f"{r['ICIR']:>+7.2f} {r['IC>0占比']:>7.1%}  {rating}")

    print("\n逐年 IC 均值热力图：")
    factor_names = summary_df.index.tolist()
    header = f"{'年份':<6}" + "".join(f"{n:>12}" for n in factor_names)
    print(header)
    print("-" * (6 + 12 * len(factor_names)))
    for year in sorted(yearly_df.index):
        row = f"{year:<6}"
        for fname in factor_names:
            v = yearly_df.loc[year, fname]
            marker = "🟢" if v > 0.03 else ("🔴" if v < -0.01 else "🟡")
            row += f"{marker}{v:>+9.3f} "
        print(row)
    print("="*70)

# ── 主流程 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("因子 IC 分析")
    print(f"区间：{BACKTEST_START} ~ {END}，双周频率")
    print("="*70)

    print("\n[1/6] 加载数据 ...")
    daily = load_data()

    print("\n[2/6] 计算因子 ...")
    daily = compute_factors(daily)

    print("\n[3/6] 双周采样 ...")
    snap = sample_biweekly(daily)
    del daily

    print("\n[4/6] 过滤股票池 ...")
    snap = filter_universe(snap)

    print("\n[5/6] 计算未来收益 & Rank IC ...")
    snap = compute_forward_returns(snap)
    ic_df = compute_rank_ic(snap)

    # 保存原始 IC 时间序列
    ic_out = OUTDIR / "factor_ic_analysis.csv"
    ic_df.to_csv(ic_out, index=False, float_format="%.6f")
    print(f"[save] {ic_out}")

    print("\n[6/6] 生成报告 ...")
    summary_df, yearly_df = write_report(ic_df)
    print_summary(summary_df, yearly_df)

    print("\n✅ 完成。")
    print(f"   CSV : backtest/factor_ic_analysis.csv")
    print(f"   报告: backtest/factor_ic_report.md")
