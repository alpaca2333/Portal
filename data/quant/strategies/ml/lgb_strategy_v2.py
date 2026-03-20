"""
LightGBM v2 策略入口 — 行业均衡选股 + 双头集成模型
===================================================
完全独立于 lgb_strategy.py，不复用其任何代码。

核心差异 vs v1:
1. **行业均衡选股**: 先在每个行业内部排名，再跨行业等比例选取，
   消除行业暴露（v1 是全局 top 5% + 行业上限）。
2. **Expanding window 训练**: 训练集随时间扩展（v1 是固定滚动窗口）。
3. **双头集成**: 回归 + LambdaRank 混合打分（v1 只有单一回归）。
4. **行业中性化特征**: z-score 而非 rank 归一化。
5. **更严格的 purge + embargo**: 防止标签前视。
6. **分位组合分析**: 报告中包含 Q1-Q5 收益和多空价差。

用法:
    python strategies/ml/lgb_strategy_v2.py
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
from engine.backtest import run_backtest
from engine.benchmark import load_all_benchmarks
from engine.report import compute_periods_per_year, print_summary, save_outputs, write_report

from strategies.ml.data_prep import build_ml_dataset, ALL_FEATURES
from strategies.ml.lgb_model_v2 import (
    ExpandingLGBEnsemble,
    summarize_ic,
    zscore_cross_section,
)


# ─────────────────────── Time Layout ────────────────────────────
# V2 uses a simpler layout: a fixed data horizon + holdout split.
#
# |--- warm-up ---|--- expanding train/test windows ---|--- holdout ---|
#                 DATA_MIN                         HOLDOUT_START    DATA_END
#
# DATA_MIN is set by data availability + warm-up for feature computation.
# HOLDOUT_START = DATA_END - HOLDOUT_MONTHS

DATA_END = "2026-02-28"
HOLDOUT_MONTHS = 12

# Model architecture constants
_INITIAL_TRAIN_YEARS = 4      # minimum training history
_STEP_MONTHS = 6              # expanding step size
_EMBARGO_MONTHS = 1           # gap between train end and test start
_WARM_UP_YEARS = 3            # for feature computation (need ~250 trading days)
_DECAY_HALF_LIFE = 540        # exponential decay half-life in days

# Data availability anchor
_DATA_AVAILABLE_FROM = pd.Timestamp("2010-01-01")
_DATA_MIN = _DATA_AVAILABLE_FROM + pd.DateOffset(years=_WARM_UP_YEARS)

_data_end_dt = pd.Timestamp(DATA_END)
_holdout_start_dt = _data_end_dt - pd.DateOffset(months=HOLDOUT_MONTHS)
_warm_up_start_dt = _DATA_AVAILABLE_FROM

# Validate: enough room for at least one expanding window
_min_needed = _INITIAL_TRAIN_YEARS * 12 + _EMBARGO_MONTHS + _STEP_MONTHS
_available = (
    (_holdout_start_dt.year - _DATA_MIN.year) * 12
    + _holdout_start_dt.month - _DATA_MIN.month
)
if _available < _min_needed:
    raise ValueError(
        f"Not enough data for expanding window training.\n"
        f"  DATA_MIN={_DATA_MIN.date()}, HOLDOUT_START={_holdout_start_dt.date()}\n"
        f"  Available: {_available} months, Needed: {_min_needed} months"
    )

HOLDOUT_START = _holdout_start_dt.strftime("%Y-%m-%d")
DATA_MIN = _DATA_MIN.strftime("%Y-%m-%d")
WARM_UP_START = _warm_up_start_dt.strftime("%Y-%m-%d")

print(f"[v2 时间布局] DATA_END={DATA_END}")
print(f"  预热起始      : {WARM_UP_START}")
print(f"  训练数据起始  : {DATA_MIN}")
print(f"  留出期起始    : {HOLDOUT_START}")
print(f"  回测 / 留出期 : {HOLDOUT_START} ~ {DATA_END}（{HOLDOUT_MONTHS}个月）")


# ─────────────────────── Strategy Config ────────────────────────

config = StrategyConfig(
    name="lgb_v2_industry_neutral",
    description=(
        "LightGBM v2: 行业均衡选股 + Expanding window + "
        "回归/排序双头集成 + 样本衰减加权。"
    ),
    rationale=(
        "### 动机\n\n"
        "V1 策略的核心问题：\n"
        "1. 全局 top 5% 选股导致行业集中度过高，在行业轮动中回撤严重。\n"
        "2. 固定滚动窗口丢弃大量历史数据，样本量不足以支撑 tree model。\n"
        "3. 单一回归模型只优化 MSE，无法保证排序质量。\n\n"
        "### V2 核心设计\n\n"
        "1. **行业均衡选股**: 每个行业内取 ML 得分前 N 名，跨行业等比例选取。"
        "消除行业 beta 暴露。\n"
        "2. **Expanding window + 衰减加权**: 训练集随时间扩展，但对远期样本"
        "施加指数衰减权重（半衰期 ~1.5 年），兼顾样本量与时效性。\n"
        "3. **双头集成**: 回归头优化 Huber loss（对极端值鲁棒），排序头优化 "
        "LambdaRank（直接优化 NDCG 排序质量），最终 rank-space 混合。\n"
        "4. **行业中性化标签**: 训练目标是行业内超额收益，而非绝对收益。\n"
        "5. **Purge + Embargo**: 训练窗口与测试窗口间设 1 个月隔离带。\n"
        "6. **Z-score 特征归一化**: 保留距离信息（v1 的 rank 只保留序关系）。\n"
    ),
    warm_up_start=WARM_UP_START,
    backtest_start=HOLDOUT_START,
    end=DATA_END,
    freq=RebalanceFreq.BIWEEKLY,
    mcap_keep_pct=0.70,
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.05,
    max_per_industry=0,         # V2: no global cap, industry balance handled in select fn
    min_industry_count=5,
    min_holding=20,
    single_side_cost=0.00015,
    buffer_sigma=0.25,          # slightly lighter buffer than v1
)

FEATURE_COLS = ALL_FEATURES


# ─────────────────────── Industry-Balanced Selection ────────────

def industry_balanced_select(
    signal: pd.DataFrame,
    prev_holdings: set,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """
    Industry-balanced stock selection — V2's core selection logic.

    Algorithm:
    1. Within each industry, rank stocks by ML score.
    2. From each industry with >= min_industry_count stocks,
       select the top `per_industry_k` stocks.
    3. Pool all selected stocks; if total < min_holding, relax constraints.
    4. Apply buffer band for incumbents (reduce turnover).

    This ensures the portfolio has balanced industry exposure,
    preventing style concentration risk.
    """
    signal = signal.copy()

    # Buffer band: give incumbents a score boost (but in cross-sectional stdev units)
    if cfg.buffer_sigma > 0 and len(prev_holdings) > 0:
        score_std = signal["score"].std()
        if score_std > 0:
            is_incumbent = signal["code"].isin(prev_holdings)
            signal.loc[is_incumbent, "score"] += cfg.buffer_sigma * score_std

    # Count stocks per industry
    industry_counts = signal.groupby("industry_code").size()
    valid_industries = industry_counts[
        industry_counts >= cfg.min_industry_count
    ].index

    signal_valid = signal[signal["industry_code"].isin(valid_industries)].copy()

    # Determine per-industry quota
    n_industries = len(valid_industries)
    target_total = max(int(len(signal_valid) * cfg.top_pct), cfg.min_holding)

    # Distribute quota across industries (equal allocation)
    base_k = max(target_total // n_industries, 1) if n_industries > 0 else 1
    remainder = target_total - base_k * n_industries

    # Select top-k per industry
    selected_parts = []
    for ind_code in valid_industries:
        ind_stocks = signal_valid[signal_valid["industry_code"] == ind_code]
        k = base_k
        # Give extra slots to industries with more stocks (proportional)
        if remainder > 0 and len(ind_stocks) > base_k:
            k += 1
            remainder -= 1
        k = min(k, len(ind_stocks))
        top_k = ind_stocks.nlargest(k, "score")
        selected_parts.append(top_k)

    if not selected_parts:
        # Fallback: global top
        return signal.nlargest(cfg.min_holding, "score")

    selected = pd.concat(selected_parts, ignore_index=True)

    # Safety check: ensure minimum portfolio size
    if len(selected) < cfg.min_holding:
        remaining = signal[~signal["code"].isin(selected["code"])]
        extra = remaining.nlargest(cfg.min_holding - len(selected), "score")
        selected = pd.concat([selected, extra], ignore_index=True)

    return selected


# ─────────────────────── Main Pipeline ──────────────────────────

def run_lgb_v2_strategy() -> pd.DataFrame:
    """
    LightGBM V2 strategy full pipeline:
    1. Build ML dataset (with z-score normalization instead of rank)
    2. Expanding-window dual-ensemble training & prediction
    3. Holdout-period inference with the last model
    4. Industry-balanced selection → engine backtest
    5. Reports and artifacts

    Returns combined DataFrame (portfolio + benchmarks).
    """
    print("=" * 64)
    print(f"策略: {config.name}")
    print(f"  {config.description}")
    print(f"数据范围    : {WARM_UP_START} ~ {DATA_END}")
    print(f"训练数据起始: {DATA_MIN}")
    print(f"留出期起始  : {HOLDOUT_START}")
    print(f"特征数量    : {len(FEATURE_COLS)}")
    print(f"选股方式    : 行业均衡（每行业内排名 → 等比例选取）")
    print("=" * 64)

    # ── Step 1: Build ML dataset ──
    # Use rank_normalize=False; we'll apply z-score ourselves
    print("\n[1] 构建ML数据集 (z-score归一化) ...")
    dataset = build_ml_dataset(
        warm_up_start=WARM_UP_START,
        backtest_end=DATA_END,
        feature_cols=FEATURE_COLS,
        mcap_keep_pct=config.mcap_keep_pct,
        rank_normalize=False,   # V2: do NOT use rank normalization
    )

    # Apply cross-sectional z-score normalization
    dataset = zscore_cross_section(dataset, FEATURE_COLS, period_col="period", clip=3.0)

    print(f"    数据集: {len(dataset):,} 行, "
          f"{dataset['code'].nunique()} 只股票, "
          f"{dataset['period'].nunique()} 个调仓期")

    # ── Step 2: Expanding-window training & prediction ──
    print(f"\n[2] Expanding-window 训练 (max_date={HOLDOUT_START}) ...")
    ensemble = ExpandingLGBEnsemble(
        initial_train_years=_INITIAL_TRAIN_YEARS,
        step_months=_STEP_MONTHS,
        embargo_months=_EMBARGO_MONTHS,
        decay_half_life_days=_DECAY_HALF_LIFE,
        use_ranker=True,
        ensemble_alpha=0.6,
    )

    predictions = ensemble.expanding_train_predict(
        dataset=dataset,
        feature_cols=FEATURE_COLS,
        label_col="label",
        fwd_ret_col="fwd_ret",
        min_date=DATA_MIN,
        max_date=HOLDOUT_START,
    )
    print(f"    预测结果: {len(predictions):,} 行, "
          f"{predictions['period'].nunique()} 个调仓期")

    # ── Step 2b: Holdout inference ──
    holdout_mask = pd.to_datetime(dataset["date"]) > pd.Timestamp(HOLDOUT_START)
    holdout_data = dataset[holdout_mask].copy()

    if len(holdout_data) > 0 and len(ensemble.reg_models) > 0:
        print(f"\n[2b] 留出期推理: {len(holdout_data):,} 行, "
              f"{holdout_data['period'].nunique()} 个调仓期 ...")

        holdout_data["pred_score"] = ensemble.predict_with_last_model(
            holdout_data, FEATURE_COLS
        )
        # Also store individual head predictions for analysis
        holdout_data["reg_pred"] = ensemble.reg_models[-1].predict(
            holdout_data[FEATURE_COLS]
        )

        keep_cols = [c for c in predictions.columns if c in holdout_data.columns]
        holdout_preds = holdout_data[keep_cols].copy()
        predictions = pd.concat([predictions, holdout_preds], ignore_index=True)
        print(f"    总预测量: {len(predictions):,} 行, "
              f"{predictions['period'].nunique()} 个调仓期")
    else:
        print("\n[2b] 无留出期数据 — 跳过。")

    # ── Step 3: Prepare backtest data ──
    print("\n[3] 准备回测数据 ...")
    snap = predictions.rename(columns={"pred_score": "score"})

    snap["_date"] = pd.to_datetime(snap["date"])
    bt_start = pd.Timestamp(config.backtest_start)
    snap = snap[snap["_date"] >= bt_start].drop(columns=["_date"]).reset_index(drop=True)

    # Ensure period_sort exists
    if "period_sort" not in snap.columns:
        parts = snap["period"].str.extract(r"(\d{4})-(\d{2})-H(\d)")
        snap["period_sort"] = (
            parts[0].astype(int) * 100 + parts[1].astype(int)
        ) * 10 + parts[2].astype(int)

    n_periods = snap["period"].nunique()
    print(f"    回测快照: {len(snap):,} 行, {n_periods} 个调仓期")

    # ── Step 4: Run backtest ──
    print("\n[4] 运行回测 (行业均衡选股) ...")
    config.post_select = industry_balanced_select
    portfolio_df = run_backtest(snap, config)
    if portfolio_df.empty:
        raise RuntimeError("Backtest produced no valid data.")

    ppy = compute_periods_per_year(portfolio_df)
    print(f"    {len(portfolio_df)} 个调仓期, "
          f"跨度 {len(portfolio_df)/ppy:.1f} 年")

    # ── Step 5: Load benchmarks ──
    print("\n[5] 加载基准指数 ...")
    combined = load_all_benchmarks(portfolio_df, config)

    # ── Step 6: Generate report ──
    print("\n[6] 生成报告 ...")
    factors_for_report = [FactorDef(f, 0.0) for f in FEATURE_COLS]
    extra_sections = _build_v2_report_sections(ensemble, predictions, snap)
    print_summary(combined.copy(), config, factors_for_report, ppy)
    save_outputs(combined, config)
    write_report(combined.copy(), config, factors_for_report, ppy, extra_sections)

    # ── Step 7: Save ML artifacts ──
    print("\n[7] 保存ML产物 ...")
    _save_v2_artifacts(ensemble, predictions, config)

    print(f"\n✅ {config.name} V2策略运行完成。")
    return combined


# ─────────────────────── Report Builder ─────────────────────────

def _build_v2_report_sections(
    ensemble: ExpandingLGBEnsemble,
    predictions: pd.DataFrame,
    snap: pd.DataFrame,
) -> str:
    """Build V2-specific Markdown report sections."""
    sections = []

    # ── Architecture Overview ──
    sections.append("## V2 模型架构\n")
    sections.append("- **训练方式**: Expanding window（训练集随时间扩展，不丢弃历史数据）")
    sections.append(f"- **初始训练**: {ensemble.initial_train_years} 年，"
                    f"每 {ensemble.step_months} 个月扩展一次")
    sections.append(f"- **Embargo**: {ensemble.embargo_months} 个月隔离带")
    sections.append(f"- **样本权重**: 指数衰减，半衰期 {ensemble.decay_half_life_days} 天")
    sections.append(f"- **特征数**: {len(ensemble.feature_cols)} 个，截面 z-score 归一化")
    sections.append(f"- **训练步数**: {len(ensemble.reg_models)}")
    sections.append("")

    # Dual head info
    sections.append("### 双头集成\n")
    sections.append("| 头 | 目标函数 | 用途 | 混合权重 |")
    sections.append("|------|----------|------|----------|")
    sections.append(f"| 回归头 | Huber Loss | 预测行业中性化超额收益 | "
                    f"{ensemble.ensemble_alpha:.0%} |")
    sections.append(f"| 排序头 | LambdaRank (NDCG) | 优化截面排序质量 | "
                    f"{1 - ensemble.ensemble_alpha:.0%} |")
    sections.append("")

    # ── IC Summary ──
    edf = ensemble.get_eval_dataframe()
    if len(edf) > 0:
        from strategies.ml.lgb_model_v2 import summarize_ic as _summ

        sections.append("## 模型质量: Rank IC（样本外）\n")

        # Ensemble IC
        summary = _summ(edf["rank_ic"])
        sections.append("### 集成模型\n")
        sections.append("| 指标 | 数值 |")
        sections.append("|------|------|")
        sections.append(f"| 平均 Rank IC | {summary['mean_ic']:.4f} |")
        sections.append(f"| 中位数 Rank IC | {summary['median_ic']:.4f} |")
        sections.append(f"| IC 标准差 | {summary['std_ic']:.4f} |")
        sections.append(f"| IC IR | {summary['ir']:.2f} |")
        sections.append(f"| IC > 0 占比 | {summary['ic_gt0_rate']:.1%} |")
        sections.append(f"| IC > 0.05 占比 | {summary['ic_gt005_rate']:.1%} |")
        sections.append(f"| 调仓期数 | {int(summary['n_periods'])} |")
        sections.append("")

        # Per-head IC comparison
        sections.append("### 各头对比\n")
        sections.append("| 头 | 平均IC | IR |")
        sections.append("|------|--------|-----|")
        reg_sum = _summ(edf["reg_rank_ic"])
        sections.append(f"| 回归头 | {reg_sum['mean_ic']:.4f} | {reg_sum['ir']:.2f} |")
        if "ranker_rank_ic" in edf.columns:
            rank_sum = _summ(edf["ranker_rank_ic"])
            sections.append(f"| 排序头 | {rank_sum['mean_ic']:.4f} | {rank_sum['ir']:.2f} |")
        sections.append(f"| **集成** | **{summary['mean_ic']:.4f}** | **{summary['ir']:.2f}** |")
        sections.append("")

        # Yearly IC
        edf_copy = edf.copy()
        edf_copy["_year"] = edf_copy["period"].str[:4]
        yearly = edf_copy.groupby("_year")["rank_ic"].agg(["mean", "std", "count"])
        if len(yearly) > 0:
            sections.append("### 逐年 Rank IC\n")
            sections.append("| 年份 | 平均IC | IC标准差 | 期数 |")
            sections.append("|------|--------|----------|------|")
            for year, row in yearly.iterrows():
                sections.append(
                    f"| {year} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |"
                )
            sections.append("")

    # ── Quantile Portfolio Analysis ──
    if len(edf) > 0 and "long_short_spread" in edf.columns:
        sections.append("## 分位组合分析（5分位）\n")
        q_cols = [f"Q{i}" for i in range(1, 6)]
        available = [c for c in q_cols if c in edf.columns]
        if available:
            sections.append("| 分位 | 平均收益 | 标准差 |")
            sections.append("|------|----------|--------|")
            for qc in available:
                vals = edf[qc].dropna()
                if len(vals) > 0:
                    sections.append(f"| {qc} | {vals.mean():.4f} | {vals.std():.4f} |")
            sections.append("")

        spreads = edf["long_short_spread"].dropna()
        if len(spreads) > 0:
            sections.append(f"- **平均多空价差 (Q5-Q1)**: {spreads.mean():.4f}")
            sections.append(f"- **多空价差 > 0 占比**: {(spreads > 0).mean():.1%}")
        mono = edf.get("monotonicity", pd.Series()).dropna()
        if len(mono) > 0:
            sections.append(f"- **平均单调性分数**: {mono.mean():.2f}")
        sections.append("")

    # ── Feature Importance ──
    imp_df = ensemble.get_importance_dataframe()
    if len(imp_df) > 0:
        mean_imp = imp_df.mean().sort_values(ascending=False)
        sections.append("## 特征重要性（回归头，跨步骤平均）\n")
        sections.append("| 排名 | 特征 | 重要性 |")
        sections.append("|------|------|--------|")
        for i, (feat, score) in enumerate(mean_imp.items(), 1):
            sections.append(f"| {i} | {feat} | {score:.1f} |")
        sections.append("")

    # ── Key Hyperparameters ──
    sections.append("## 关键超参数\n")
    sections.append("### 回归头\n")
    sections.append("| 参数 | 值 |")
    sections.append("|------|-----|")
    for key in ["objective", "alpha", "num_leaves", "max_depth", "learning_rate",
                 "n_estimators", "min_child_samples", "subsample",
                 "colsample_bytree", "reg_alpha", "reg_lambda"]:
        if key in ensemble.reg_params:
            sections.append(f"| {key} | {ensemble.reg_params[key]} |")
    sections.append("")

    if ensemble.use_ranker:
        sections.append("### 排序头\n")
        sections.append("| 参数 | 值 |")
        sections.append("|------|-----|")
        for key in ["objective", "num_leaves", "max_depth", "learning_rate",
                     "n_estimators", "min_child_samples"]:
            if key in ensemble.rank_params:
                sections.append(f"| {key} | {ensemble.rank_params[key]} |")
        sections.append("")

    # ── V1 vs V2 Comparison ──
    sections.append("## V1 vs V2 对比\n")
    sections.append(
        "| 维度 | V1 | V2 |\n"
        "|------|-----|-----|\n"
        "| 训练窗口 | 3年固定滚动 | Expanding + 衰减加权 |\n"
        "| 标签 | 截面 rank 归一化 | 行业中性化超额收益 |\n"
        "| 特征归一化 | 截面 percentile rank | 截面 z-score (±3σ clip) |\n"
        "| 模型 | 单一 LGBMRegressor (MSE) | 回归(Huber) + 排序(LambdaRank) 集成 |\n"
        "| 选股 | 全局 top 5% + 行业上限5只 | 行业内排名 → 等比例跨行业选取 |\n"
        "| 样本权重 | 无 | 指数衰减（半衰期 540 天） |\n"
        "| Purge | label_end_date 检查 | label_end_date + 1个月 embargo |\n"
        "| 评估指标 | Rank IC only | Rank IC + 分位价差 + 单调性 |\n"
    )

    return "\n".join(sections)


def _save_v2_artifacts(
    ensemble: ExpandingLGBEnsemble,
    predictions: pd.DataFrame,
    cfg: StrategyConfig,
) -> None:
    """Save V2 ML artifacts."""
    outdir = Path(cfg.output_dir)

    # Evaluation history (IC + quantile returns per period)
    edf = ensemble.get_eval_dataframe()
    if len(edf) > 0:
        eval_path = outdir / f"{cfg.name}_eval_history.csv"
        edf.to_csv(eval_path, index=False, float_format="%.6f")
        print(f"[保存] {eval_path.name}")

    # Feature importance
    imp_df = ensemble.get_importance_dataframe()
    if len(imp_df) > 0:
        imp_path = outdir / f"{cfg.name}_feature_importance.csv"
        imp_df.to_csv(imp_path, index=False, float_format="%.1f")
        print(f"[保存] {imp_path.name}")

    # Predictions
    pred_path = outdir / f"{cfg.name}_predictions.csv"
    predictions.to_csv(pred_path, index=False, float_format="%.6f")
    print(f"[保存] {pred_path.name}")


# ─────────────────────── Entry Point ────────────────────────────

if __name__ == "__main__":
    run_lgb_v2_strategy()
