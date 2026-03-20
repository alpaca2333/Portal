"""
LightGBM 最小手数策略（小资金实盘模拟）
===================================================
基于 lgb_strategy 的已有预测结果，使用"每只股票买1手"的
真实资金分配方式计算收益，用于评估小资金实盘可行性。

与等权策略的关键区别:
- 等权策略: 每只股票分配相同金额 → ret = mean(individual_rets)
- 本策略:   每只股票买100股 → 高价股占更多资金权重
  ret = sum(price_i * 100 * ret_i) / sum(price_i * 100)

直接复用 lgb_stock_selection_predictions.csv，无需重新训练模型。

用法:
    cd /projects/portal/data/quant
    python strategies/ml/lgb_strategy_minlot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.types import StrategyConfig, FactorDef, SelectionMode, RebalanceFreq
from engine.benchmark import load_all_benchmarks
from engine.report import (
    compute_periods_per_year,
    print_summary,
    save_outputs,
    write_report,
    calc_return_metrics,
)

# ─────────────────────── Paths ───────────────────────────────

PREDICTIONS_PATH = PROJECT_ROOT / "backtest" / "lgb_stock_selection_predictions.csv"

# ─────────────────────── Time layout (same as lgb_strategy) ──

DATA_END = "2026-02-28"
HOLDOUT_MONTHS = 12

_data_end_dt = pd.Timestamp(DATA_END)
_model_cutoff_dt = _data_end_dt - pd.DateOffset(months=HOLDOUT_MONTHS)
MODEL_CUTOFF = _model_cutoff_dt.strftime("%Y-%m-%d")
BACKTEST_START = MODEL_CUTOFF

# ─────────────────────── Min-lot config ──────────────────────

LOT_SIZE = 100           # A-share minimum trading unit
MIN_PRICE = 2.0          # Stocks below this price buy extra lots

# ─────────────────────── Strategy config ─────────────────────

config = StrategyConfig(
    name="lgb_minlot",
    description=(
        "LightGBM 最小手数策略：与 lgb_stock_selection 相同选股，"
        "但每只股票仅买1手(100股)，高价股占更大权重。用于评估小资金实盘表现。"
    ),
    rationale=(
        "### 动机\n\n"
        "lgb_stock_selection 策略在回测中使用等权配置（每只股票分配相同金额），\n"
        "但实盘中小资金无法做到严格等权——A股最小交易单位是100股（1手），\n"
        "当资金有限时，每只股票只能买1手，导致高价股自然占更大的资金权重。\n\n"
        "### 核心差异\n\n"
        "1. **权重分配**：按 `price × 100` 的实际投入金额加权，而非等权。\n"
        "2. **股价影响**：高价股（如50元/股→5000元/手）权重远大于低价股（5元→500元/手）。\n"
        "3. **资金测算**：每期记录所需最低资金 = Σ(entry_price × 100)。\n"
        "4. **选股逻辑完全相同**：复用 lgb_strategy 的 ML 选股，仅改变权重计算。\n"
    ),
    warm_up_start="2010-01-01",
    backtest_start=BACKTEST_START,
    end=DATA_END,
    freq=RebalanceFreq.BIWEEKLY,
    mcap_keep_pct=0.70,
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.05,
    max_per_industry=5,
    min_industry_count=5,
    min_holding=20,
    single_side_cost=0.00015,
    buffer_sigma=0.3,
)


# ─────────────────────── Selection (same as lgb_strategy) ────

def ml_select(
    signal: pd.DataFrame,
    prev_holdings: set,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """ML-based stock selection (identical to lgb_strategy.ml_select)."""
    signal = signal.copy()

    if cfg.buffer_sigma > 0 and len(prev_holdings) > 0:
        score_std = signal["score"].std()
        if score_std > 0:
            is_incumbent = signal["code"].isin(prev_holdings)
            signal.loc[is_incumbent, "score"] += cfg.buffer_sigma * score_std

    if cfg.selection_mode == SelectionMode.TOP_PCT:
        cutoff = signal["score"].quantile(1 - cfg.top_pct)
        selected = signal[signal["score"] >= cutoff].copy()
    else:
        selected = signal.nlargest(cfg.top_n, "score").copy()

    if cfg.max_per_industry > 0:
        selected = (
            selected.sort_values("score", ascending=False)
            .groupby("industry_code", group_keys=False)
            .head(cfg.max_per_industry)
        )

    return selected


# ─────────────────────── Min-lot backtest engine ─────────────

def run_minlot_backtest(
    snap: pd.DataFrame,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """
    Custom backtest loop: each stock gets 1 lot (100 shares).
    High-price stocks naturally have higher portfolio weight.

    Returns DataFrame with standard + min-lot specific columns.
    """
    print(f"[最小手数回测] 开始调仓 (每只{LOT_SIZE}股, 最低股价阈值={MIN_PRICE}元) ...")

    entry_price_col = "next_open" if "next_open" in snap.columns else "close"
    entry_date_col = "next_date" if "next_date" in snap.columns else "date"
    exit_price_col = entry_price_col
    exit_date_col = entry_date_col

    # Build ordered period list
    period_info = (
        snap.groupby("period")
        .agg(period_sort=("period_sort", "first"))
        .reset_index()
        .sort_values("period_sort")
    )
    ordered_periods = period_info["period"].tolist()

    # Backtest start sort key
    bt_year = int(cfg.backtest_start[:4])
    bt_month = int(cfg.backtest_start[5:7])
    bt_start_sort = bt_year * 100 * 10 + bt_month * 10 + 1

    select_fn = ml_select
    results = []
    prev_holdings: set = set()

    for i in range(len(ordered_periods) - 1):
        sig_period = ordered_periods[i]
        hold_period = ordered_periods[i + 1]

        sig_sort = period_info.loc[
            period_info["period"] == sig_period, "period_sort"
        ].values[0]
        if sig_sort < bt_start_sort:
            continue

        signal = snap[
            (snap["period"] == sig_period) & snap["score"].notna()
        ].copy()
        if len(signal) < cfg.min_holding * 2:
            continue

        # Selection
        selected = select_fn(signal, prev_holdings, cfg)[
            ["code", entry_price_col, entry_date_col]
        ].copy()
        selected = selected.rename(columns={
            entry_price_col: "entry_open",
            entry_date_col: "entry_date",
        })

        hold = snap[snap["period"] == hold_period][
            ["code", exit_price_col, exit_date_col]
        ].copy()
        hold = hold.rename(columns={
            exit_price_col: "exit_open",
            exit_date_col: "exit_date",
        })

        merged = selected.merge(hold, on="code", how="inner").dropna(
            subset=["entry_open", "exit_open", "entry_date", "exit_date"]
        )
        merged = merged[(merged["entry_open"] > 0) & (merged["exit_open"] > 0)]
        if len(merged) < cfg.min_holding:
            continue

        curr_holdings = set(merged["code"].tolist())
        n_curr = len(curr_holdings)

        # ── Determine lots per stock ──
        merged["lots"] = LOT_SIZE
        cheap_mask = merged["entry_open"] < MIN_PRICE
        if cheap_mask.any():
            merged.loc[cheap_mask, "lots"] = (
                np.ceil(MIN_PRICE * LOT_SIZE / (merged.loc[cheap_mask, "entry_open"] * LOT_SIZE))
                * LOT_SIZE
            ).astype(int)

        # ── Capital allocation ──
        merged["invested"] = merged["entry_open"] * merged["lots"]
        total_capital = merged["invested"].sum()

        # ── Weight by invested capital ──
        merged["weight"] = merged["invested"] / total_capital

        # ── Individual returns ──
        merged["ret"] = merged["exit_open"] / merged["entry_open"] - 1

        # ── Weighted portfolio return ──
        gross_ret = (merged["weight"] * merged["ret"]).sum()

        # ── Equal-weight return for comparison ──
        eq_ret = merged["ret"].mean()

        # ── Turnover ──
        if len(prev_holdings) == 0:
            turnover_buy = 1.0
            turnover_sell = 0.0
        else:
            sold = prev_holdings - curr_holdings
            bought = curr_holdings - prev_holdings
            n_prev = len(prev_holdings)
            turnover_sell = len(sold) / n_prev if n_prev > 0 else 0.0
            turnover_buy = len(bought) / n_curr if n_curr > 0 else 0.0

        tc = cfg.single_side_cost * (turnover_sell + turnover_buy)
        net_ret = gross_ret - tc
        hold_date = pd.to_datetime(merged["exit_date"]).max()

        n_industries = (
            signal[signal["code"].isin(curr_holdings)]["industry_code"].nunique()
        )

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
            # Min-lot specific
            "total_capital": total_capital,
            "avg_capital_per_stock": total_capital / n_curr if n_curr > 0 else 0,
            "max_single_position": merged["invested"].max(),
            "min_single_position": merged["invested"].min(),
            "max_weight": merged["weight"].max(),
            "eq_port_ret_gross": eq_ret,
            "eq_port_ret": eq_ret - tc,
        })
        prev_holdings = curr_holdings

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        avg_to = (result_df["turnover_sell"] + result_df["turnover_buy"]).mean() / 2
        avg_tc = result_df["tc"].mean()
        total_tc = result_df["tc"].sum()
        avg_n = result_df["n_stocks"].mean()
        avg_ind = result_df["n_industries"].mean()
        avg_cap = result_df["total_capital"].mean()
        min_cap = result_df["total_capital"].min()
        max_cap = result_df["total_capital"].max()
        avg_max_wt = result_df["max_weight"].mean()

        print(f"[最小手数回测] 有效调仓期数: {len(result_df)}")
        print(f"[最小手数回测] 平均持仓: {avg_n:.0f} 只 / {avg_ind:.0f} 个行业")
        print(f"[最小手数回测] 平均单边换手: {avg_to:.1%}")
        print(f"[最小手数回测] 平均每期成本: {avg_tc:.4%}，累计拖累: {total_tc:.4%}")
        print(f"[最小手数回测] ── 资金需求 ──")
        print(f"  平均每期所需资金: ¥{avg_cap:>12,.0f} (≈{avg_cap/10000:.1f}万)")
        print(f"  最少一期所需资金: ¥{min_cap:>12,.0f} (≈{min_cap/10000:.1f}万)")
        print(f"  最多一期所需资金: ¥{max_cap:>12,.0f} (≈{max_cap/10000:.1f}万)")
        print(f"  最大单股权重(均): {avg_max_wt:.2%}")

    return result_df


# ─────────────────────── Main flow ───────────────────────────

def run_lgb_minlot_strategy() -> pd.DataFrame:
    """
    Load pre-computed LGB predictions and run min-lot backtest.
    No model training — reuses lgb_stock_selection_predictions.csv.
    """
    print("=" * 64)
    print(f"策略名称 : {config.name}")
    print(f"  {config.description}")
    print(f"  每只股票买 {LOT_SIZE} 股, 股价 < {MIN_PRICE}元 时加倍")
    print(f"回测区间 : {config.backtest_start} ~ {DATA_END}")
    print("=" * 64)

    # ── Step 1: Load predictions ──
    print(f"\n[1] 加载已有预测结果: {PREDICTIONS_PATH.name} ...")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"])
    print(f"    加载 {len(predictions):,} 行, "
          f"{predictions['code'].nunique()} 只股票, "
          f"{predictions['period'].nunique()} 个调仓期")

    # ── Step 2: Prepare backtest data ──
    print("\n[2] 准备回测数据 ...")
    snap = predictions.rename(columns={"pred_score": "score"})
    bt_start = pd.Timestamp(config.backtest_start)
    snap = snap[snap["date"] >= bt_start].reset_index(drop=True)

    if "period_sort" not in snap.columns:
        parts = snap["period"].str.extract(r"(\d{4})-(\d{2})-H(\d)")
        snap["period_sort"] = (
            parts[0].astype(int) * 100 + parts[1].astype(int)
        ) * 10 + parts[2].astype(int)

    n_periods = snap["period"].nunique()
    n_stocks = snap["code"].nunique()
    print(f"    回测快照: {len(snap):,} 行, {n_periods} 个调仓期, {n_stocks} 只股票")

    # ── Step 3: Run min-lot backtest ──
    print("\n[3] 运行最小手数回测 ...")
    portfolio_df = run_minlot_backtest(snap, config)
    if portfolio_df.empty:
        raise RuntimeError("回测未产生有效数据。")

    ppy = compute_periods_per_year(portfolio_df)

    # ── Step 4: Load benchmarks ──
    print("\n[4] 加载基准指数 ...")
    combined = load_all_benchmarks(portfolio_df, config)

    # ── Step 5: Generate report ──
    print("\n[5] 生成报告 ...")
    factors_for_report = []  # No factor weights in ML strategy
    extra_sections = _build_minlot_report(portfolio_df, combined, ppy)

    print_summary(combined.copy(), config, factors_for_report, ppy)
    save_outputs(combined, config)
    write_report(combined.copy(), config, factors_for_report, ppy, extra_sections)

    # ── Step 6: Capital summary ──
    _print_capital_summary(portfolio_df, combined, ppy)

    print(f"\n✅ {config.name} 策略运行完成。")
    return combined


# ─────────────────────── Report helpers ──────────────────────

def _build_minlot_report(
    portfolio_df: pd.DataFrame,
    combined: pd.DataFrame,
    ppy: float,
) -> str:
    """Build min-lot specific Markdown report sections."""
    sections = []

    # ── Capital requirements ──
    sections.append("## 最小手数资金需求\n")
    sections.append(f"- **交易单位**: 每只股票 {LOT_SIZE} 股（1手）")
    sections.append(f"- **最低股价阈值**: {MIN_PRICE} 元（低于此价格加倍手数）")
    sections.append("")

    avg_cap = portfolio_df["total_capital"].mean()
    min_cap = portfolio_df["total_capital"].min()
    max_cap = portfolio_df["total_capital"].max()
    sections.append("| 指标 | 金额 |")
    sections.append("|------|------|")
    sections.append(f"| 平均每期所需资金 | ¥{avg_cap:,.0f}（≈{avg_cap/10000:.1f}万） |")
    sections.append(f"| 最少一期所需资金 | ¥{min_cap:,.0f}（≈{min_cap/10000:.1f}万） |")
    sections.append(f"| 最多一期所需资金 | ¥{max_cap:,.0f}（≈{max_cap/10000:.1f}万） |")
    sections.append("")

    # ── Weight concentration ──
    avg_max_wt = portfolio_df["max_weight"].mean()
    avg_n = portfolio_df["n_stocks"].mean()
    ideal_wt = 1.0 / avg_n if avg_n > 0 else 0
    sections.append("### 权重集中度\n")
    sections.append(f"- 理想等权权重: {ideal_wt:.2%}")
    sections.append(f"- 实际最大单股权重（均值）: {avg_max_wt:.2%}")
    sections.append(f"- 偏离度: {avg_max_wt - ideal_wt:+.2%}")
    sections.append("")

    # ── Min-lot vs Equal-weight comparison ──
    if "eq_port_ret" in portfolio_df.columns:
        ml_rets = portfolio_df["port_ret"]
        eq_rets = portfolio_df["eq_port_ret"]

        ml_m = calc_return_metrics(ml_rets, freq=ppy)
        eq_m = calc_return_metrics(eq_rets, freq=ppy)

        sections.append("## 最小手数 vs 等权对比\n")
        sections.append("| 指标 | 最小手数 | 等权 | 差异 |")
        sections.append("|------|----------|------|------|")
        for label, key, fmt in [
            ("年化收益", "ann_ret", "+.2%"),
            ("年化波动", "ann_vol", ".2%"),
            ("夏普比率", "sharpe", ".2f"),
            ("最大回撤", "max_dd", ".2%"),
            ("累计收益", "cum_ret", "+.2%"),
        ]:
            ml_v = ml_m[key]
            eq_v = eq_m[key]
            diff = ml_v - eq_v
            if key == "sharpe":
                sections.append(
                    f"| {label} | {ml_v:{fmt}} | {eq_v:{fmt}} | {diff:+.2f} |"
                )
            else:
                sections.append(
                    f"| {label} | {ml_v:{fmt}} | {eq_v:{fmt}} | {diff:+.2%} |"
                )
        sections.append("")

    # ── Capital time series ──
    sections.append("## 逐期资金需求\n")
    sections.append("| 期间 | 持仓数 | 所需资金 | 最大单股仓位 | 最大权重 |")
    sections.append("|------|--------|----------|-------------|----------|")
    for _, row in portfolio_df.iterrows():
        sections.append(
            f"| {row['period']} | {int(row['n_stocks'])} | "
            f"¥{row['total_capital']:,.0f} | "
            f"¥{row['max_single_position']:,.0f} | "
            f"{row['max_weight']:.2%} |"
        )
    sections.append("")

    return "\n".join(sections)


def _print_capital_summary(
    portfolio_df: pd.DataFrame,
    combined: pd.DataFrame,
    ppy: float,
) -> None:
    """Print detailed capital and performance comparison to terminal."""

    print("\n" + "=" * 72)
    print("📊 最小手数策略 — 资金需求与盈利能力分析")
    print("=" * 72)

    # Capital stats
    avg_cap = portfolio_df["total_capital"].mean()
    min_cap = portfolio_df["total_capital"].min()
    max_cap = portfolio_df["total_capital"].max()
    avg_n = portfolio_df["n_stocks"].mean()

    print(f"\n📌 资金需求（每只股票买{LOT_SIZE}股）:")
    print(f"   平均每期所需: ¥{avg_cap:>12,.0f} ≈ {avg_cap/10000:.1f} 万")
    print(f"   最少一期所需: ¥{min_cap:>12,.0f} ≈ {min_cap/10000:.1f} 万")
    print(f"   最多一期所需: ¥{max_cap:>12,.0f} ≈ {max_cap/10000:.1f} 万")
    print(f"   平均持仓数量: {avg_n:.0f} 只")

    # Performance comparison
    if "eq_port_ret" in portfolio_df.columns:
        ml_rets = portfolio_df["port_ret"]
        eq_rets = portfolio_df["eq_port_ret"]

        ml_m = calc_return_metrics(ml_rets, freq=ppy)
        eq_m = calc_return_metrics(eq_rets, freq=ppy)

        print(f"\n📌 盈利能力对比:")
        print(f"   {'指标':<16} {'最小手数':>12} {'等权':>12} {'差异':>12}")
        print(f"   {'-'*16} {'-'*12} {'-'*12} {'-'*12}")

        for label, key, fmt in [
            ("年化收益", "ann_ret", ".2%"),
            ("年化波动", "ann_vol", ".2%"),
            ("夏普比率", "sharpe", ".2f"),
            ("最大回撤", "max_dd", ".2%"),
            ("累计收益", "cum_ret", ".2%"),
            ("胜率", "win_rate", ".1%"),
        ]:
            ml_v = ml_m[key]
            eq_v = eq_m[key]
            diff = ml_v - eq_v
            if key == "sharpe":
                print(f"   {label:<16} {ml_v:>12{fmt}} {eq_v:>12{fmt}} {diff:>+12.2f}")
            else:
                print(f"   {label:<16} {ml_v:>12{fmt}} {eq_v:>12{fmt}} {diff:>+12.2%}")

    # Weight concentration
    avg_max_wt = portfolio_df["max_weight"].mean()
    ideal_wt = 1.0 / avg_n if avg_n > 0 else 0
    print(f"\n📌 权重分布:")
    print(f"   理想等权权重  : {ideal_wt:.2%}")
    print(f"   实际最大权重  : {avg_max_wt:.2%}")
    if ideal_wt > 0:
        print(f"   最大/理想比   : {avg_max_wt/ideal_wt:.1f}x")

    # Year-by-year comparison
    if "eq_port_ret" in portfolio_df.columns:
        print(f"\n📌 逐年对比 (最小手数 vs 等权):")
        tmp = portfolio_df.copy()
        tmp["_year"] = pd.to_datetime(tmp["date"]).dt.year
        for year in sorted(tmp["_year"].unique()):
            ys = tmp[tmp["_year"] == year]
            ml_yr = (1 + ys["port_ret"]).prod() - 1
            eq_yr = (1 + ys["eq_port_ret"]).prod() - 1
            diff = ml_yr - eq_yr
            avg_cap_yr = ys["total_capital"].mean()
            print(f"   {year}: 最小手数 {ml_yr:+.1%}  等权 {eq_yr:+.1%}  "
                  f"差异 {diff:+.1%}  资金≈{avg_cap_yr/10000:.1f}万")

    print("\n" + "=" * 72)
    print(f"💡 结论: 运行此策略最低需要约 ¥{max_cap:,.0f} "
          f"(≈{max_cap/10000:.1f}万) 的资金")
    print(f"   建议预留 20% 缓冲: ¥{max_cap*1.2:,.0f} (≈{max_cap*1.2/10000:.1f}万)")
    print("=" * 72)


# ─────────────────────── Entry point ─────────────────────────

if __name__ == "__main__":
    run_lgb_minlot_strategy()
