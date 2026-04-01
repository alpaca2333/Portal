"""
行业轮动因子研究脚本
=====================
验证 A 股申万一级行业 (31个) 在 2015-2025 年间是否存在可利用的轮动规律。

模块:
  1. 行业月收益率面板构建
  2. 行业动量效应检验
  3. 行业反转效应检验 (formation × holding IC 热力图)
  4. 行业拥挤度指标构建与检验
  5. 行业估值离散度与均值回归
  6. 综合评估汇总

输出:
  data/quant/research/ 目录下的 CSV 文件 + 终端汇总
"""

import os
import sys
import warnings
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Global config
# ──────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "quant.db"
OUTPUT_DIR = Path(__file__).resolve().parent
RESEARCH_START = "20150101"
RESEARCH_END = "20251231"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_conn():
    """Get a SQLite connection."""
    return sqlite3.connect(str(DB_PATH))


def load_industry_map() -> dict:
    """Load industry_code -> industry_name mapping for L1."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT industry_code, industry_name FROM industry_info WHERE level = 'L1'",
        conn,
    )
    conn.close()
    return dict(zip(df["industry_code"], df["industry_name"]))


# ══════════════════════════════════════════════
# Module 1: Industry Monthly Returns Panel
# ══════════════════════════════════════════════

def build_industry_monthly_returns() -> pd.DataFrame:
    """
    Build a monthly return panel for 31 SW-L1 industries.

    Method:
      - For each month-end, compute circ_mv-weighted average pct_chg
        for each industry, aggregated from daily returns.
      - Monthly return = product of (1 + daily_ret) - 1

    Returns:
        DataFrame with columns: [trade_month, industry, ret, n_stocks]
    """
    print("\n" + "=" * 70)
    print("模块一：行业月收益率面板构建")
    print("=" * 70)

    conn = get_conn()

    # Load daily data: only needed columns
    print("  加载日线数据 (pct_chg, circ_mv, sw_l1) ...")
    query = f"""
        SELECT trade_date, ts_code, pct_chg, circ_mv, sw_l1,
               turnover_rate_f, amount, pe_ttm, pb, vol
        FROM stock_daily
        WHERE trade_date >= {RESEARCH_START}
          AND trade_date <= {RESEARCH_END}
          AND sw_l1 IS NOT NULL
          AND sw_l1 != ''
          AND is_suspended = 0
          AND pct_chg IS NOT NULL
          AND circ_mv IS NOT NULL
          AND circ_mv > 0
    """
    df = pd.read_sql(query, conn)
    conn.close()

    print(f"  加载完成: {len(df):,} 行, 日期范围 {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    # Convert trade_date to datetime for month grouping
    df["date"] = pd.to_datetime(df["trade_date"].astype(str))
    df["trade_month"] = df["date"].dt.to_period("M")

    # Daily return as fraction
    df["ret_daily"] = df["pct_chg"] / 100.0

    # ── Compute industry monthly returns (circ_mv weighted) ──
    # For each (month, industry), compound daily returns weighted by circ_mv
    print("  计算行业月收益率 (流通市值加权) ...")

    records = []
    for (month, ind), grp in df.groupby(["trade_month", "sw_l1"]):
        # For each day in this month-industry group, compute weighted return
        daily_rets = []
        for day, day_grp in grp.groupby("trade_date"):
            w = day_grp["circ_mv"].values
            r = day_grp["ret_daily"].values
            w_sum = w.sum()
            if w_sum > 0:
                daily_rets.append(np.dot(w, r) / w_sum)

        if len(daily_rets) > 0:
            # Compound daily returns
            monthly_ret = np.prod([1 + r for r in daily_rets]) - 1
        else:
            monthly_ret = 0.0

        records.append({
            "trade_month": str(month),
            "industry": ind,
            "ret": monthly_ret,
            "n_stocks": grp["ts_code"].nunique(),
        })

    ind_ret = pd.DataFrame(records)

    # Save
    out_path = OUTPUT_DIR / "industry_monthly_returns.csv"
    ind_ret.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {out_path}")

    # Summary
    ind_map = load_industry_map()
    pivot = ind_ret.pivot(index="trade_month", columns="industry", values="ret")
    ann_ret = pivot.mean() * 12
    ann_vol = pivot.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol

    summary = pd.DataFrame({
        "行业名称": [ind_map.get(c, c) for c in pivot.columns],
        "年化收益": ann_ret.values,
        "年化波动": ann_vol.values,
        "夏普比率": sharpe.values,
        "月均股票数": ind_ret.groupby("industry")["n_stocks"].mean().reindex(pivot.columns).values,
    }, index=pivot.columns)
    summary = summary.sort_values("年化收益", ascending=False)

    print("\n  ── 行业年化收益排名 (2015-2025) ──")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n  共 {len(pivot.columns)} 个行业, {len(pivot)} 个月")

    # Also save the raw daily data for other modules
    return ind_ret, df


# ══════════════════════════════════════════════
# Module 2: Momentum Effect Test
# ══════════════════════════════════════════════

def test_momentum_effect(ind_ret: pd.DataFrame):
    """
    Test industry momentum effect:
      - Sort industries by past 1/3/6/12 month returns
      - Divide into 5 quintiles
      - Compute next-month average return for each quintile
      - Report Top-Bottom spread and t-stat

    Also compute momentum decay curve.
    """
    print("\n" + "=" * 70)
    print("模块二：行业动量效应检验")
    print("=" * 70)

    ind_map = load_industry_map()

    # Pivot to wide format: rows=month, cols=industry, values=ret
    pivot = ind_ret.pivot(index="trade_month", columns="industry", values="ret")
    pivot = pivot.sort_index()
    months = pivot.index.tolist()

    lookbacks = [1, 3, 6, 12]
    n_groups = 5  # quintiles

    all_results = {}

    for lb in lookbacks:
        print(f"\n  ── 回看 {lb} 个月动量 ──")

        group_returns = {g: [] for g in range(1, n_groups + 1)}
        ic_list = []

        for i in range(lb, len(months) - 1):
            # Formation: past lb months cumulative return
            formation_ret = pivot.iloc[i - lb:i].add(1).prod() - 1
            # Next month return
            next_ret = pivot.iloc[i + 1] if i + 1 < len(months) else None
            if next_ret is None:
                break

            # Drop NaN
            valid = formation_ret.dropna().index.intersection(next_ret.dropna().index)
            if len(valid) < n_groups * 2:
                continue

            form = formation_ret[valid]
            nxt = next_ret[valid]

            # IC (rank correlation)
            ic = stats.spearmanr(form, nxt)[0]
            ic_list.append(ic)

            # Sort into quintiles
            ranks = form.rank(method="first")
            n = len(ranks)
            for ind_code in valid:
                rank_val = ranks[ind_code]
                group = min(int((rank_val - 1) / (n / n_groups)) + 1, n_groups)
                group_returns[group].append(nxt[ind_code])

        # Compute average return per group
        group_avg = {}
        for g in range(1, n_groups + 1):
            rets = group_returns[g]
            group_avg[g] = np.mean(rets) if len(rets) > 0 else 0

        # Top - Bottom spread
        spread = group_avg[n_groups] - group_avg[1]
        ic_mean = np.mean(ic_list) if ic_list else 0
        ic_std = np.std(ic_list) if ic_list else 1
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_list))) if ic_list else 0

        print(f"    分组月均收益 (Q1=最弱 → Q{n_groups}=最强):")
        for g in range(1, n_groups + 1):
            label = "最弱" if g == 1 else ("最强" if g == n_groups else f"Q{g}")
            print(f"      {label}: {group_avg[g]*100:+.2f}%  (n={len(group_returns[g])})")
        print(f"    多空利差 (Q{n_groups}-Q1): {spread*100:+.2f}%/月")
        print(f"    IC均值: {ic_mean:.4f}, IC_IR: {ic_ir:.4f}, t值: {ic_tstat:.2f}")

        all_results[lb] = {
            "group_avg": group_avg,
            "spread": spread,
            "ic_mean": ic_mean,
            "ic_ir": ic_ir,
            "ic_tstat": ic_tstat,
            "n_months": len(ic_list),
        }

    # ── Momentum Decay Curve ──
    print(f"\n  ── 动量衰减曲线 (formation=1M, holding=1~12M) ──")
    decay_records = []
    for hold in range(1, 13):
        cum_spreads = []
        for i in range(1, len(months) - hold):
            # Formation: past 1 month
            formation_ret = pivot.iloc[i - 1:i].iloc[0]
            # Holding: next 'hold' months cumulative
            if i + hold >= len(months):
                break
            holding_ret = pivot.iloc[i:i + hold].add(1).prod() - 1

            valid = formation_ret.dropna().index.intersection(holding_ret.dropna().index)
            if len(valid) < 10:
                continue

            form = formation_ret[valid]
            hold_r = holding_ret[valid]

            # Top quintile vs bottom quintile
            ranks = form.rank(method="first")
            n = len(ranks)
            cutoff_low = n * 0.2
            cutoff_high = n * 0.8
            top = hold_r[ranks > cutoff_high].mean()
            bottom = hold_r[ranks <= cutoff_low].mean()
            cum_spreads.append(top - bottom)

        avg_spread = np.mean(cum_spreads) if cum_spreads else 0
        decay_records.append({
            "holding_months": hold,
            "avg_cumulative_spread": avg_spread,
            "n_obs": len(cum_spreads),
        })
        direction = "动量" if avg_spread > 0 else "反转"
        print(f"    持有{hold:2d}个月: 累计利差 {avg_spread*100:+.2f}% ({direction})")

    decay_df = pd.DataFrame(decay_records)
    out_path = OUTPUT_DIR / "momentum_decay_curve.csv"
    decay_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {out_path}")

    # Save momentum summary
    mom_summary = pd.DataFrame([
        {
            "回看窗口": f"{lb}M",
            "多空利差(月)": f"{all_results[lb]['spread']*100:+.2f}%",
            "IC均值": f"{all_results[lb]['ic_mean']:.4f}",
            "IC_IR": f"{all_results[lb]['ic_ir']:.4f}",
            "t值": f"{all_results[lb]['ic_tstat']:.2f}",
            "样本月数": all_results[lb]["n_months"],
        }
        for lb in lookbacks
    ])
    out_path = OUTPUT_DIR / "momentum_summary.csv"
    mom_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {out_path}")

    return all_results


# ══════════════════════════════════════════════
# Module 3: Reversal Effect & Formation×Holding IC Heatmap
# ══════════════════════════════════════════════

def test_reversal_and_heatmap(ind_ret: pd.DataFrame):
    """
    Test reversal effect across multiple formation × holding windows.
    Output: IC heatmap (formation=1~12M × holding=1~12M).
    """
    print("\n" + "=" * 70)
    print("模块三：行业反转效应检验 (Formation × Holding IC 热力图)")
    print("=" * 70)

    pivot = ind_ret.pivot(index="trade_month", columns="industry", values="ret")
    pivot = pivot.sort_index()
    months = pivot.index.tolist()

    formations = list(range(1, 13))
    holdings = list(range(1, 13))

    ic_matrix = pd.DataFrame(index=formations, columns=holdings, dtype=float)
    ir_matrix = pd.DataFrame(index=formations, columns=holdings, dtype=float)
    tstat_matrix = pd.DataFrame(index=formations, columns=holdings, dtype=float)

    print("  计算中 (12×12=144 个组合) ...")

    for form in formations:
        for hold in holdings:
            ic_list = []
            for i in range(form, len(months) - hold):
                # Formation return
                formation_ret = pivot.iloc[i - form:i].add(1).prod() - 1
                # Holding return (skip 1 month gap to avoid microstructure)
                end_idx = i + hold
                if end_idx >= len(months):
                    break
                holding_ret = pivot.iloc[i:end_idx].add(1).prod() - 1

                valid = formation_ret.dropna().index.intersection(holding_ret.dropna().index)
                if len(valid) < 10:
                    continue

                ic = stats.spearmanr(formation_ret[valid], holding_ret[valid])[0]
                ic_list.append(ic)

            if len(ic_list) > 5:
                ic_mean = np.mean(ic_list)
                ic_std = np.std(ic_list)
                ic_matrix.loc[form, hold] = ic_mean
                ir_matrix.loc[form, hold] = ic_mean / ic_std if ic_std > 0 else 0
                tstat_matrix.loc[form, hold] = (
                    ic_mean / (ic_std / np.sqrt(len(ic_list))) if ic_std > 0 else 0
                )

    # Print IC heatmap
    print("\n  ── IC 均值热力图 (行=Formation月数, 列=Holding月数) ──")
    print("  " + "  ".join([f"H{h:2d}" for h in holdings]))
    for form in formations:
        row_str = f"  F{form:2d}"
        for hold in holdings:
            val = ic_matrix.loc[form, hold]
            if pd.notna(val):
                marker = "+" if val > 0 else ""
                row_str += f" {marker}{val:.3f}"[: 6].rjust(6)
            else:
                row_str += "   N/A"
        print(row_str)

    # Find best momentum (positive IC) and best reversal (negative IC)
    ic_flat = ic_matrix.stack().astype(float)
    best_mom_idx = ic_flat.idxmax()
    best_rev_idx = ic_flat.idxmin()

    print(f"\n  最强动量组合: Formation={best_mom_idx[0]}M, Holding={best_mom_idx[1]}M, "
          f"IC={ic_flat[best_mom_idx]:.4f}")
    print(f"  最强反转组合: Formation={best_rev_idx[0]}M, Holding={best_rev_idx[1]}M, "
          f"IC={ic_flat[best_rev_idx]:.4f}")

    # Identify significant cells (|t| > 2)
    sig_count = (tstat_matrix.stack().astype(float).abs() > 2).sum()
    total = tstat_matrix.stack().astype(float).notna().sum()
    print(f"  显著性 (|t|>2): {sig_count}/{total} 个组合 ({sig_count/total*100:.1f}%)")

    # Save
    ic_matrix.index.name = "formation"
    ic_matrix.columns.name = "holding"
    out_path = OUTPUT_DIR / "formation_holding_ic_heatmap.csv"
    ic_matrix.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n  已保存: {out_path}")

    # Also save t-stat matrix
    tstat_matrix.index.name = "formation"
    tstat_matrix.columns.name = "holding"
    out_path2 = OUTPUT_DIR / "formation_holding_tstat_heatmap.csv"
    tstat_matrix.to_csv(out_path2, encoding="utf-8-sig")
    print(f"  已保存: {out_path2}")

    return ic_matrix, tstat_matrix


# ══════════════════════════════════════════════
# Module 4: Crowding Indicators
# ══════════════════════════════════════════════

def test_crowding_signals(ind_ret: pd.DataFrame, raw_daily: pd.DataFrame):
    """
    Build and test 3 crowding indicators:
      1. Turnover deviation: industry recent turnover / historical avg
      2. Volume concentration: industry volume share change
      3. Volatility clustering: 20d vol / 60d vol
    """
    print("\n" + "=" * 70)
    print("模块四：行业拥挤度指标构建与检验")
    print("=" * 70)

    ind_map = load_industry_map()

    # We need daily-level data for crowding signals
    df = raw_daily.copy()
    df["date"] = pd.to_datetime(df["trade_date"].astype(str))
    df["trade_month"] = df["date"].dt.to_period("M")

    # ── Signal 1: Turnover Deviation ──
    print("\n  ── 信号1: 换手率偏离度 ──")
    # Monthly avg turnover per industry
    ind_turnover = (
        df.groupby(["trade_month", "sw_l1"])["turnover_rate_f"]
        .mean()
        .reset_index()
        .rename(columns={"turnover_rate_f": "avg_turnover"})
    )
    ind_turnover["trade_month"] = ind_turnover["trade_month"].astype(str)
    ind_turnover = ind_turnover.sort_values(["sw_l1", "trade_month"])

    # Rolling 12-month mean as baseline
    turnover_records = []
    for ind, grp in ind_turnover.groupby("sw_l1"):
        grp = grp.sort_values("trade_month").reset_index(drop=True)
        grp["baseline"] = grp["avg_turnover"].rolling(12, min_periods=6).mean().shift(1)
        grp["turnover_dev"] = grp["avg_turnover"] / grp["baseline"]
        turnover_records.append(grp)
    turnover_df = pd.concat(turnover_records, ignore_index=True)

    # ── Signal 2: Volume Concentration ──
    print("  ── 信号2: 成交额集中度变化 ──")
    # Monthly total amount per industry
    ind_amount = (
        df.groupby(["trade_month", "sw_l1"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total_amount"})
    )
    ind_amount["trade_month"] = ind_amount["trade_month"].astype(str)

    # Market total per month
    mkt_amount = ind_amount.groupby("trade_month")["total_amount"].sum().rename("mkt_total")
    ind_amount = ind_amount.merge(mkt_amount, on="trade_month")
    ind_amount["vol_share"] = ind_amount["total_amount"] / ind_amount["mkt_total"]

    # Change in volume share vs 3-month ago
    vol_share_records = []
    for ind, grp in ind_amount.groupby("sw_l1"):
        grp = grp.sort_values("trade_month").reset_index(drop=True)
        grp["vol_share_3m_ago"] = grp["vol_share"].shift(3)
        grp["vol_share_chg"] = grp["vol_share"] - grp["vol_share_3m_ago"]
        vol_share_records.append(grp)
    vol_share_df = pd.concat(vol_share_records, ignore_index=True)

    # ── Signal 3: Volatility Clustering ──
    print("  ── 信号3: 波动率聚集 ──")
    # Compute daily industry return (simple average for speed)
    ind_daily_ret = (
        df.groupby(["trade_date", "sw_l1"])
        .apply(lambda g: np.average(g["ret_daily"], weights=g["circ_mv"]) if g["circ_mv"].sum() > 0 else 0)
        .reset_index()
        .rename(columns={0: "ind_ret"})
    )
    ind_daily_ret = ind_daily_ret.sort_values(["sw_l1", "trade_date"])

    vol_records = []
    for ind, grp in ind_daily_ret.groupby("sw_l1"):
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        grp["vol_20d"] = grp["ind_ret"].rolling(20, min_periods=15).std()
        grp["vol_60d"] = grp["ind_ret"].rolling(60, min_periods=40).std()
        grp["vol_ratio"] = grp["vol_20d"] / grp["vol_60d"]

        # Aggregate to monthly (last value of month)
        grp["date"] = pd.to_datetime(grp["trade_date"].astype(str))
        grp["trade_month"] = grp["date"].dt.to_period("M").astype(str)
        monthly = grp.groupby("trade_month").last()[["vol_ratio"]].reset_index()
        monthly["sw_l1"] = ind
        vol_records.append(monthly)
    vol_df = pd.concat(vol_records, ignore_index=True)

    # ── Merge all signals with next-month returns ──
    print("\n  合并信号与下月收益 ...")
    ind_ret_copy = ind_ret.copy()

    # Create next-month return
    pivot = ind_ret_copy.pivot(index="trade_month", columns="industry", values="ret").sort_index()
    next_ret = pivot.shift(-1)  # next month return
    next_ret_long = next_ret.stack().reset_index()
    next_ret_long.columns = ["trade_month", "sw_l1", "next_ret"]

    # Merge signals
    signals = turnover_df[["trade_month", "sw_l1", "turnover_dev"]].merge(
        vol_share_df[["trade_month", "sw_l1", "vol_share_chg"]], on=["trade_month", "sw_l1"], how="outer"
    ).merge(
        vol_df[["trade_month", "sw_l1", "vol_ratio"]], on=["trade_month", "sw_l1"], how="outer"
    ).merge(
        next_ret_long, on=["trade_month", "sw_l1"], how="inner"
    )

    # ── Test each signal ──
    signal_names = {
        "turnover_dev": "换手率偏离度",
        "vol_share_chg": "成交额集中度变化",
        "vol_ratio": "波动率聚集度",
    }

    crowding_results = []
    for sig_col, sig_name in signal_names.items():
        print(f"\n  ── {sig_name} ({sig_col}) ──")
        valid = signals.dropna(subset=[sig_col, "next_ret"])
        if len(valid) < 50:
            print(f"    样本不足 ({len(valid)}), 跳过")
            continue

        # Monthly IC
        ic_list = []
        for month, grp in valid.groupby("trade_month"):
            if len(grp) < 10:
                continue
            ic = stats.spearmanr(grp[sig_col], grp["next_ret"])[0]
            ic_list.append(ic)

        ic_mean = np.mean(ic_list)
        ic_std = np.std(ic_list)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_list))) if ic_std > 0 and len(ic_list) > 0 else 0

        # Quintile test
        n_groups = 5
        valid["rank"] = valid.groupby("trade_month")[sig_col].rank(pct=True)
        group_rets = {}
        for g in range(1, n_groups + 1):
            low = (g - 1) / n_groups
            high = g / n_groups
            mask = (valid["rank"] > low) & (valid["rank"] <= high)
            group_rets[g] = valid.loc[mask, "next_ret"].mean()

        spread = group_rets[n_groups] - group_rets[1]

        print(f"    IC均值: {ic_mean:.4f}, IC_IR: {ic_ir:.4f}, t值: {ic_tstat:.2f}")
        print(f"    分组月均收益 (Q1=最低 → Q5=最高):")
        for g in range(1, n_groups + 1):
            print(f"      Q{g}: {group_rets[g]*100:+.2f}%")
        print(f"    Q5-Q1利差: {spread*100:+.2f}%/月")

        # Interpretation
        if ic_mean < -0.05 and abs(ic_tstat) > 1.5:
            print(f"    → 结论: 拥挤度越高, 下月收益越差 (反向预测有效)")
        elif ic_mean > 0.05 and abs(ic_tstat) > 1.5:
            print(f"    → 结论: 拥挤度越高, 下月收益越好 (正向动量)")
        else:
            print(f"    → 结论: 预测能力不显著")

        crowding_results.append({
            "信号": sig_name,
            "IC均值": ic_mean,
            "IC_IR": ic_ir,
            "t值": ic_tstat,
            "Q5-Q1利差(月)": spread,
            "样本月数": len(ic_list),
        })

    crowding_df = pd.DataFrame(crowding_results)
    out_path = OUTPUT_DIR / "crowding_signal_test.csv"
    crowding_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {out_path}")

    return crowding_df


# ══════════════════════════════════════════════
# Module 5: Valuation Dispersion & Mean Reversion
# ══════════════════════════════════════════════

def test_valuation_mean_reversion(ind_ret: pd.DataFrame, raw_daily: pd.DataFrame):
    """
    Test whether low-valuation industries outperform high-valuation ones.
    Uses PE_TTM and PB percentile within 3-year rolling window.
    """
    print("\n" + "=" * 70)
    print("模块五：行业估值离散度与均值回归")
    print("=" * 70)

    ind_map = load_industry_map()
    df = raw_daily.copy()
    df["date"] = pd.to_datetime(df["trade_date"].astype(str))
    df["trade_month"] = df["date"].dt.to_period("M").astype(str)

    # ── Compute monthly industry median PE/PB ──
    print("  计算行业月度中位数 PE/PB ...")
    # Use median to be robust to outliers
    ind_val = (
        df[df["pe_ttm"].notna() & (df["pe_ttm"] > 0) & (df["pe_ttm"] < 500)]
        .groupby(["trade_month", "sw_l1"])
        .agg(
            median_pe=("pe_ttm", "median"),
            median_pb=("pb", "median"),
        )
        .reset_index()
    )
    ind_val = ind_val.sort_values(["sw_l1", "trade_month"])

    # ── Rolling 3-year percentile ──
    print("  计算3年滚动估值分位数 ...")
    val_records = []
    for ind, grp in ind_val.groupby("sw_l1"):
        grp = grp.sort_values("trade_month").reset_index(drop=True)
        for i in range(36, len(grp)):  # need 3 years history
            window_pe = grp["median_pe"].iloc[i - 36:i].dropna()
            window_pb = grp["median_pb"].iloc[i - 36:i].dropna()
            current_pe = grp["median_pe"].iloc[i]
            current_pb = grp["median_pb"].iloc[i]

            pe_pct = (window_pe < current_pe).mean() if len(window_pe) > 10 and pd.notna(current_pe) else np.nan
            pb_pct = (window_pb < current_pb).mean() if len(window_pb) > 10 and pd.notna(current_pb) else np.nan

            val_records.append({
                "trade_month": grp["trade_month"].iloc[i],
                "sw_l1": ind,
                "pe_percentile": pe_pct,
                "pb_percentile": pb_pct,
            })

    val_pct_df = pd.DataFrame(val_records)

    # ── Merge with future returns (6M and 12M) ──
    print("  合并估值分位与未来收益 ...")
    pivot = ind_ret.pivot(index="trade_month", columns="industry", values="ret").sort_index()

    for horizon_name, horizon in [("6M", 6), ("12M", 12)]:
        # Future cumulative return
        future_ret = pivot.rolling(horizon).apply(lambda x: np.prod(1 + x) - 1).shift(-horizon)
        future_long = future_ret.stack().reset_index()
        future_long.columns = ["trade_month", "sw_l1", f"future_ret_{horizon_name}"]

        val_pct_df = val_pct_df.merge(future_long, on=["trade_month", "sw_l1"], how="left")

    # ── Test PE percentile ──
    for val_col, val_name in [("pe_percentile", "PE分位数"), ("pb_percentile", "PB分位数")]:
        print(f"\n  ── {val_name} vs 未来收益 ──")

        for horizon_name in ["6M", "12M"]:
            ret_col = f"future_ret_{horizon_name}"
            valid = val_pct_df.dropna(subset=[val_col, ret_col])
            if len(valid) < 50:
                continue

            # Monthly IC
            ic_list = []
            for month, grp in valid.groupby("trade_month"):
                if len(grp) < 10:
                    continue
                ic = stats.spearmanr(grp[val_col], grp[ret_col])[0]
                ic_list.append(ic)

            ic_mean = np.mean(ic_list) if ic_list else 0
            ic_std = np.std(ic_list) if ic_list else 1
            ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_list))) if ic_list and ic_std > 0 else 0

            # Quintile test
            n_groups = 5
            valid["rank"] = valid.groupby("trade_month")[val_col].rank(pct=True)
            group_rets = {}
            for g in range(1, n_groups + 1):
                low = (g - 1) / n_groups
                high = g / n_groups
                mask = (valid["rank"] > low) & (valid["rank"] <= high)
                group_rets[g] = valid.loc[mask, ret_col].mean()

            spread = group_rets[n_groups] - group_rets[1]

            print(f"    {val_name} → 未来{horizon_name}收益:")
            print(f"      IC均值: {ic_mean:.4f}, t值: {ic_tstat:.2f}")
            for g in range(1, n_groups + 1):
                label = "最低估值" if g == 1 else ("最高估值" if g == n_groups else f"Q{g}")
                print(f"      {label}: {group_rets[g]*100:+.2f}%")
            print(f"      高估-低估利差: {spread*100:+.2f}%")

            if ic_mean < -0.05 and abs(ic_tstat) > 1.5:
                print(f"      → 结论: 低估值行业未来跑赢高估值行业 (均值回归有效)")
            elif ic_mean > 0.05 and abs(ic_tstat) > 1.5:
                print(f"      → 结论: 高估值行业继续跑赢 (估值动量)")
            else:
                print(f"      → 结论: 估值预测能力不显著")

    # Save
    out_path = OUTPUT_DIR / "valuation_quintile_test.csv"
    val_pct_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {out_path}")

    return val_pct_df


# ══════════════════════════════════════════════
# Module 6: Comprehensive Summary
# ══════════════════════════════════════════════

def comprehensive_summary(
    momentum_results: dict,
    ic_matrix: pd.DataFrame,
    crowding_df: pd.DataFrame,
    ind_ret: pd.DataFrame,
):
    """
    Summarize all findings and provide strategy recommendations.
    """
    print("\n" + "=" * 70)
    print("模块六：综合评估汇总")
    print("=" * 70)

    # ── 1. Signal IC Summary ──
    print("\n  ── 各信号 IC 汇总 ──")
    summary_rows = []

    # Momentum signals
    for lb, res in momentum_results.items():
        summary_rows.append({
            "信号类别": "动量",
            "信号名称": f"{lb}M动量",
            "IC均值": res["ic_mean"],
            "IC_IR": res["ic_ir"],
            "t值": res["ic_tstat"],
            "多空利差(月)": res["spread"],
            "样本月数": res["n_months"],
            "是否显著": "✓" if abs(res["ic_tstat"]) > 2 else "✗",
        })

    # Best reversal from heatmap
    ic_flat = ic_matrix.stack().astype(float)
    best_rev_idx = ic_flat.idxmin()
    best_rev_ic = ic_flat[best_rev_idx]
    summary_rows.append({
        "信号类别": "反转",
        "信号名称": f"F{best_rev_idx[0]}M-H{best_rev_idx[1]}M反转",
        "IC均值": best_rev_ic,
        "IC_IR": np.nan,
        "t值": np.nan,
        "多空利差(月)": np.nan,
        "样本月数": np.nan,
        "是否显著": "待确认",
    })

    # Crowding signals
    for _, row in crowding_df.iterrows():
        summary_rows.append({
            "信号类别": "拥挤度",
            "信号名称": row["信号"],
            "IC均值": row["IC均值"],
            "IC_IR": row["IC_IR"],
            "t值": row["t值"],
            "多空利差(月)": row["Q5-Q1利差(月)"],
            "样本月数": row["样本月数"],
            "是否显著": "✓" if abs(row["t值"]) > 2 else "✗",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 2. Yearly Stability ──
    print("\n  ── 分年度稳定性 (1M动量 IC) ──")
    pivot = ind_ret.pivot(index="trade_month", columns="industry", values="ret").sort_index()
    months = pivot.index.tolist()

    yearly_ic = {}
    for i in range(1, len(months) - 1):
        formation_ret = pivot.iloc[i - 1:i].iloc[0]
        next_ret = pivot.iloc[i + 1] if i + 1 < len(months) else None
        if next_ret is None:
            break
        valid = formation_ret.dropna().index.intersection(next_ret.dropna().index)
        if len(valid) < 10:
            continue
        ic = stats.spearmanr(formation_ret[valid], next_ret[valid])[0]
        year = months[i][:4]
        if year not in yearly_ic:
            yearly_ic[year] = []
        yearly_ic[year].append(ic)

    print(f"    {'年份':>6s}  {'IC均值':>8s}  {'IC标准差':>8s}  {'IC>0占比':>8s}  {'月数':>4s}")
    for year in sorted(yearly_ic.keys()):
        ics = yearly_ic[year]
        ic_m = np.mean(ics)
        ic_s = np.std(ics)
        pos_ratio = np.mean([1 if ic > 0 else 0 for ic in ics])
        print(f"    {year:>6s}  {ic_m:>+8.4f}  {ic_s:>8.4f}  {pos_ratio:>8.1%}  {len(ics):>4d}")

    # ── 3. Recommendations ──
    print("\n  ── 策略建议 ──")

    # Find the most effective signals
    effective = summary_df[summary_df["是否显著"] == "✓"].sort_values("IC_IR", ascending=False)
    if len(effective) > 0:
        print(f"  显著有效的信号 ({len(effective)} 个):")
        for _, row in effective.iterrows():
            print(f"    • {row['信号类别']}/{row['信号名称']}: IC={row['IC均值']:.4f}, IC_IR={row['IC_IR']:.4f}")
    else:
        print("  未发现统计显著的信号 (|t|>2)")

    # Check if momentum or reversal dominates
    mom_1m = momentum_results.get(1, {})
    mom_3m = momentum_results.get(3, {})
    if mom_1m.get("ic_mean", 0) > 0.05:
        print("\n  → 短期动量 (1M) 存在正向预测力，建议纳入策略")
    if mom_3m.get("ic_mean", 0) < -0.05:
        print("  → 中期反转 (3M) 存在负向预测力，建议作为反向信号")

    # Save summary
    out_path = OUTPUT_DIR / "industry_rotation_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {out_path}")

    return summary_df


# ══════════════════════════════════════════════
# Main Entry
# ══════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           行业轮动因子研究 (申万一级, 2015-2025)                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  数据库: {DB_PATH}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  研究区间: {RESEARCH_START} ~ {RESEARCH_END}")

    # Module 1: Build industry monthly returns
    ind_ret, raw_daily = build_industry_monthly_returns()

    # Module 2: Momentum effect
    momentum_results = test_momentum_effect(ind_ret)

    # Module 3: Reversal & IC heatmap
    ic_matrix, tstat_matrix = test_reversal_and_heatmap(ind_ret)

    # Module 4: Crowding signals
    crowding_df = test_crowding_signals(ind_ret, raw_daily)

    # Module 5: Valuation mean reversion
    val_df = test_valuation_mean_reversion(ind_ret, raw_daily)

    # Module 6: Comprehensive summary
    summary_df = comprehensive_summary(momentum_results, ic_matrix, crowding_df, ind_ret)

    print("\n" + "=" * 70)
    print("研究完成！所有结果已保存至:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
