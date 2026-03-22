"""
LightGBM 截面选股策略 — 周度调仓版
===================================================
基于 lgb_strategy.py，将调仓频率从双周改为周度。

流程:
1. 构建 ML 数据集 (data_prep, 周度采样)
2. 滚动窗口 LightGBM 训练 & 预测 (lgb_model)
3. 将 pred_score 注入为 'score' → 引擎回测循环
4. 生成标准报告（双基准，符合 OutputFormat.md 规范）

用法:
    python strategies/ml/lgb_strategy_weekly.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.types import StrategyConfig, FactorDef, SelectionMode, RebalanceFreq
from engine.backtest import run_backtest
from engine.benchmark import load_all_benchmarks
from engine.report import compute_periods_per_year, print_summary, save_outputs, write_report

from strategies.ml.data_prep import build_ml_dataset, ALL_FEATURES
from strategies.ml.lgb_model import RollingLGBModel, ic_summary


# ─────────────────────── 时间布局 ────────────────────────────
# 与 biweekly 版完全相同的推导链，复用同一套参数。

DATA_END = "2016-01-01"
HOLDOUT_MONTHS = 12

_TRAIN_YEARS = 3
_VAL_MONTHS = 6
_STEP_MONTHS = 6
_WARM_UP_YEARS = 3

_data_end_dt = pd.Timestamp(DATA_END)
_model_cutoff_dt = _data_end_dt - pd.DateOffset(months=HOLDOUT_MONTHS)

_DATA_AVAILABLE_FROM = pd.Timestamp("2008-01-01")
_min_date_dt = _DATA_AVAILABLE_FROM + pd.DateOffset(years=_WARM_UP_YEARS)
_warm_up_start_dt = _min_date_dt - pd.DateOffset(years=_WARM_UP_YEARS)

# Validation
_min_needed_months = _TRAIN_YEARS * 12 + _VAL_MONTHS + _STEP_MONTHS
_available_months = (
    (_model_cutoff_dt.year - _min_date_dt.year) * 12
    + _model_cutoff_dt.month - _min_date_dt.month
)
if _available_months < _min_needed_months:
    raise ValueError(
        f"MODEL_CUTOFF={_model_cutoff_dt.date()} 与 MIN_DATE={_min_date_dt.date()} "
        f"之间只有 {_available_months} 个月，至少需要 {_min_needed_months} 个月。\n"
        f"请减小 HOLDOUT_MONTHS（当前={HOLDOUT_MONTHS}）或调整 _DATA_AVAILABLE_FROM。"
    )

_n_windows_approx = (_available_months - _TRAIN_YEARS * 12 - _VAL_MONTHS) // _STEP_MONTHS
if _n_windows_approx < 3:
    import warnings
    warnings.warn(
        f"预计滚动窗口数 ≈ {_n_windows_approx}，IC 统计代表性较弱。\n"
        f"建议 HOLDOUT_MONTHS ≤ 18 以确保至少 5+ 个窗口。",
        UserWarning,
    )

MODEL_CUTOFF = _model_cutoff_dt.strftime("%Y-%m-%d")
BACKTEST_START = MODEL_CUTOFF
MIN_DATE = _min_date_dt.strftime("%Y-%m-%d")
WARM_UP_START = _warm_up_start_dt.strftime("%Y-%m-%d")

print(f"[时间布局] DATA_END={DATA_END}")
print(f"  预热起始  : {WARM_UP_START}")
print(f"  训练起始  : {MIN_DATE}")
print(f"  模型截止  : {MODEL_CUTOFF}")
print(f"  回测/留出 : {MODEL_CUTOFF} ~ {DATA_END}（{HOLDOUT_MONTHS}个月）")

# ─────────────────────── 策略配置 ────────────────────────
# Key difference: freq=RebalanceFreq.WEEKLY

config = StrategyConfig(
    name="lgb_stock_selection_weekly",
    description="LightGBM 截面选股策略（周度调仓版）：15个特征，3年滚动窗口，每周调仓。",
    rationale=(
        "### 动机\n\n"
        "基于 biweekly 版本 (lgb_stock_selection)，将调仓频率从双周提升到每周。\n\n"
        "### 周度调仓的预期优劣\n\n"
        "**优势**:\n"
        "1. 更快捕捉 ML 信号衰减前的 alpha（短期因子如 rev_10, ret_5d_std 受益显著）。\n"
        "2. 更及时的止损/换仓，减少单期大幅回撤。\n"
        "3. 更平滑的 NAV 曲线，降低单期波动。\n\n"
        "**劣势/风险**:\n"
        "1. 换手率约为双周版的 2 倍，交易成本拖累更大。\n"
        "2. 短周期内 ML 信号信噪比可能更低（周度收益更嘈杂）。\n"
        "3. 每周采样导致数据集更大，训练更耗时。\n\n"
        "### 核心设计选择\n\n"
        "1. **截面排序标签**：预测相对排名而非绝对收益。\n"
        "2. **3年滚动训练窗口**：与 biweekly 版一致，便于公平对比。\n"
        "3. **行业约束选股**：ML分数前5%，每行业最多5只。\n"
        "4. **15个特征**：与 biweekly 版完全一致。\n"
        "5. **特征排序归一化**：截面百分位排序 [0,1]。\n"
        "6. **可交易收益定义**：next-open 入场/离场。\n"
    ),
    warm_up_start=WARM_UP_START,
    backtest_start=BACKTEST_START,
    end=DATA_END,
    freq=RebalanceFreq.WEEKLY,
    mcap_keep_pct=0.70,
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.05,
    max_per_industry=5,
    min_industry_count=5,
    min_holding=20,
    single_side_cost=0.00015,
    buffer_sigma=0.3,
)

FEATURE_COLS = ALL_FEATURES


# ─────────────────────── 选股逻辑 ────────────────────────

def ml_select(
    signal: pd.DataFrame,
    prev_holdings: set,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """
    ML-based stock selection (replaces linear scoring).

    Uses 'score' column (= LightGBM pred_score) for ranking.
    Applies buffer band bonus to incumbent holdings, then selects
    top-scoring stocks with per-industry cap.
    """
    signal = signal.copy()

    # Buffer band: score bonus for incumbent holdings
    if cfg.buffer_sigma > 0 and len(prev_holdings) > 0:
        score_std = signal["score"].std()
        if score_std > 0:
            is_incumbent = signal["code"].isin(prev_holdings)
            signal.loc[is_incumbent, "score"] += cfg.buffer_sigma * score_std

    # Top N% selection
    if cfg.selection_mode == SelectionMode.TOP_PCT:
        cutoff = signal["score"].quantile(1 - cfg.top_pct)
        selected = signal[signal["score"] >= cutoff].copy()
    else:
        selected = signal.nlargest(cfg.top_n, "score").copy()

    # Per-industry cap
    if cfg.max_per_industry > 0:
        selected = (
            selected.sort_values("score", ascending=False)
            .groupby("industry_code", group_keys=False)
            .head(cfg.max_per_industry)
        )

    return selected


# ─────────────────────── 主流程 ──────────────────────────

def run_lgb_strategy_weekly() -> pd.DataFrame:
    """
    LightGBM weekly strategy full pipeline:
    1. Build ML dataset (weekly sampling)
    2. Rolling train & predict (up to MODEL_CUTOFF)
    3. Holdout inference (MODEL_CUTOFF ~ DATA_END)
    4. Inject pred_score into engine backtest
    5. Generate report

    Returns combined DataFrame with portfolio + benchmark returns.
    """
    model_cutoff = MODEL_CUTOFF

    print("=" * 64)
    print(f"策略名称 : {config.name}")
    print(f"  {config.description}")
    print(f"数据范围 : {WARM_UP_START} ~ {DATA_END}")
    print(f"训练起始 : {MIN_DATE}")
    print(f"模型截止 : 滚动窗口截至 {model_cutoff}")
    print(f"留出期   : {model_cutoff} ~ {DATA_END}（{HOLDOUT_MONTHS}个月，纯样本外）")
    print(f"回测区间 : {config.backtest_start} ~ {DATA_END}")
    print(f"调仓频率 : 周度 (weekly)")
    print(f"特征数量 : {len(FEATURE_COLS)}")
    print("=" * 64)

    # ── Step 1: Build ML dataset (weekly sampling) ──
    print("\n[1] 构建ML数据集（周度采样）...")
    dataset = build_ml_dataset(
        warm_up_start=WARM_UP_START,
        backtest_end=DATA_END,
        feature_cols=FEATURE_COLS,
        mcap_keep_pct=config.mcap_keep_pct,
        rank_normalize=True,
        freq="weekly",
    )
    print(f"    数据集: {len(dataset):,} 行, "
          f"{dataset['code'].nunique()} 只股票, "
          f"{dataset['period'].nunique()} 个调仓期")

    # ── Step 2: Rolling LightGBM train & predict ──
    print(f"\n[2] 滚动LightGBM训练（max_date={model_cutoff}）...")
    model = RollingLGBModel(
        train_years=_TRAIN_YEARS,
        val_months=_VAL_MONTHS,
        step_months=_STEP_MONTHS,
    )
    predictions = model.rolling_train_predict(
        dataset=dataset,
        feature_cols=FEATURE_COLS,
        label_col="label",
        min_date=MIN_DATE,
        max_date=model_cutoff,
    )
    print(f"    预测结果: {len(predictions):,} 行, "
          f"{predictions['period'].nunique()} 个调仓期")

    # ── Step 2b: Holdout inference ──
    holdout_mask = pd.to_datetime(dataset["date"]) > pd.Timestamp(model_cutoff)
    holdout_data = dataset[holdout_mask].copy()

    if len(holdout_data) > 0 and len(model.models) > 0:
        print(f"\n[2b] 留出期推理: {len(holdout_data):,} 行, "
              f"{holdout_data['period'].nunique()} 个调仓期 ...")
        last_model = model.models[-1]
        X_holdout = holdout_data[FEATURE_COLS]
        holdout_data["pred_score"] = last_model.predict(X_holdout)

        keep_cols = [c for c in predictions.columns if c in holdout_data.columns]
        holdout_preds = holdout_data[keep_cols].copy()

        predictions = pd.concat([predictions, holdout_preds], ignore_index=True)
        print(f"    预测总计（滚动 + 留出期）: {len(predictions):,} 行, "
              f"{predictions['period'].nunique()} 个调仓期")
    else:
        print("\n[2b] 无留出期数据或无已训练模型 — 跳过。")

    # ── Step 3: Prepare backtest data ──
    print("\n[3] 准备回测数据 ...")

    snap = predictions.rename(columns={"pred_score": "score"})

    # Filter to backtest period only
    snap["_date"] = pd.to_datetime(snap["date"])
    bt_start = pd.Timestamp(config.backtest_start)
    snap = snap[snap["_date"] >= bt_start].drop(columns=["_date"]).reset_index(drop=True)

    # Ensure period_sort column exists (YYYY-WXX format)
    if "period_sort" not in snap.columns:
        parts = snap["period"].str.extract(r"(\d{4})-W(\d{2})")
        snap["period_sort"] = (
            parts[0].astype(int) * 100 + parts[1].astype(int)
        )

    n_periods = snap["period"].nunique()
    n_stocks = snap["code"].nunique()
    avg_stocks_per_period = len(snap) / n_periods if n_periods > 0 else 0
    print(f"    回测快照: {len(snap):,} 行, "
          f"{n_periods} 个调仓期, {n_stocks} 只股票")
    print(f"    平均每期股票数: {avg_stocks_per_period:.0f}")

    # ── Step 4: Run backtest ──
    print("\n[4] 运行回测 ...")

    config.post_select = ml_select

    portfolio_df = run_backtest(snap, config)
    if portfolio_df.empty:
        raise RuntimeError("回测未产生有效数据。")

    ppy = compute_periods_per_year(portfolio_df)
    print(f"    {len(portfolio_df)} 个调仓期, 跨度 "
          f"{len(portfolio_df)/ppy:.1f} 年 = {ppy:.1f} 期/年")

    # ── Step 5: Load benchmarks ──
    print("\n[5] 加载基准指数 ...")
    combined = load_all_benchmarks(portfolio_df, config)

    # ── Step 6: Generate report ──
    print("\n[6] 生成报告 ...")

    factors_for_report = [FactorDef(f, 0.0) for f in FEATURE_COLS]
    extra_sections = _build_ml_report_sections(model, predictions, snap, model_cutoff)

    print_summary(combined.copy(), config, factors_for_report, ppy)
    save_outputs(combined, config)
    write_report(combined.copy(), config, factors_for_report, ppy, extra_sections)

    # ── Step 7: Save ML artifacts ──
    print("\n[7] 保存ML产物 ...")
    _save_ml_artifacts(model, predictions, config)

    print(f"\n✅ {config.name} 策略运行完成。")
    return combined


# ─────────────────────── 报告辅助函数 ─────────────────────────

def _build_ml_report_sections(
    model: RollingLGBModel,
    predictions: pd.DataFrame,
    snap: pd.DataFrame,
    model_cutoff: str,
) -> str:
    """Build ML-specific Markdown report sections."""
    sections = []

    # Time layout
    sections.append("## 时间布局\n")
    sections.append(f"- **滚动训练/验证/测试**: 截至 **{model_cutoff}**")
    sections.append(f"- **留出期（纯样本外）**: **{model_cutoff} ~ {DATA_END}** "
                    f"（{HOLDOUT_MONTHS}个月）")
    sections.append("- 训练/验证样本按 `label_end_date` purge，避免标签跨窗口泄露。")
    sections.append("- 回测收益按 `next_open → next_open` 计算，避免同K线收盘成交偏差。")
    sections.append(f"- **调仓频率**: 周度（每周最后一个交易日信号，下周一开盘执行）")
    sections.append("")

    # Rank IC summary
    ic_df = model.get_ic_dataframe()
    if len(ic_df) > 0:
        summary = ic_summary(ic_df["rank_ic"])
        sections.append("## 模型质量: Rank IC（样本外）\n")
        sections.append("| 指标 | 数值 |")
        sections.append("|------|------|")
        sections.append(f"| 平均 Rank IC | {summary['mean_ic']:.4f} |")
        sections.append(f"| Rank IC 标准差 | {summary['std_ic']:.4f} |")
        sections.append(f"| IC IR（均值/标准差） | {summary['ir']:.2f} |")
        sections.append(f"| IC > 0 占比 | {summary['ic_positive_rate']:.1%} |")
        sections.append(f"| 调仓期数 | {len(ic_df)} |")
        sections.append("")

        # Yearly IC
        ic_df = ic_df.copy()
        ic_df["_date"] = ic_df["period"].str[:4]
        yearly_ic = ic_df.groupby("_date")["rank_ic"].agg(["mean", "std", "count"])
        if len(yearly_ic) > 0:
            sections.append("### 逐年 Rank IC\n")
            sections.append("| 年份 | 平均IC | IC标准差 | 期数 |")
            sections.append("|------|--------|----------|------|")
            for year, row in yearly_ic.iterrows():
                sections.append(
                    f"| {year} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |"
                )
            sections.append("")

    # Feature importance
    imp_df = model.get_importance_dataframe()
    if len(imp_df) > 0:
        mean_imp = imp_df.mean().sort_values(ascending=False)
        sections.append("## 特征重要性（跨窗口平均）\n")
        sections.append("| 排名 | 特征 | 重要性 |")
        sections.append("|------|------|--------|")
        for i, (feat, score) in enumerate(mean_imp.items(), 1):
            sections.append(f"| {i} | {feat} | {score:.1f} |")
        sections.append("")

    # Model architecture
    sections.append("## 模型架构\n")
    sections.append("- **模型**: LightGBM（梯度提升决策树）")
    sections.append(f"- **训练窗口**: {model.train_years} 年滚动")
    sections.append(f"- **验证窗口**: {model.val_months} 个月")
    sections.append(f"- **重训频率**: 每 {model.step_months} 个月")
    sections.append(f"- **特征数**: {len(model.feature_cols)} 个，截面排序归一化")
    sections.append(f"- **标签**: next-open 到 next-open 的下期收益率，排序归一化至 [0,1]")
    sections.append(f"- **调仓频率**: 周度（vs biweekly 版的双周）")
    sections.append(f"- **训练窗口数**: {len(model.models)}")
    sections.append("")

    # Key hyperparameters
    sections.append("### 关键超参数\n")
    sections.append("| 参数 | 值 |")
    sections.append("|------|-----|")
    for key in ["num_leaves", "max_depth", "learning_rate", "n_estimators",
                 "min_child_samples", "subsample", "colsample_bytree",
                 "reg_alpha", "reg_lambda"]:
        if key in model.params:
            sections.append(f"| {key} | {model.params[key]} |")
    sections.append("")

    # Comparison with biweekly
    sections.append("## 对比: Weekly vs Biweekly LightGBM\n")
    sections.append(
        "| 维度 | Biweekly | Weekly |\n"
        "|------|----------|--------|\n"
        "| 调仓频率 | 每半月 | 每周 |\n"
        "| 预期换手率 | 基准 | ~2x |\n"
        "| 信号时效性 | 中 | 高 |\n"
        "| 短期因子受益 | 中 | 高 |\n"
        "| 成本拖累 | 低 | 中 |\n"
        "| 数据集大小 | 基准 | ~2x |\n"
        "| 训练/预测耗时 | 基准 | ~2x |\n"
    )

    return "\n".join(sections)


def _save_ml_artifacts(
    model: RollingLGBModel,
    predictions: pd.DataFrame,
    cfg: StrategyConfig,
) -> None:
    """Save ML artifacts (IC history, feature importance, predictions)."""
    outdir = Path(cfg.output_dir)

    # Save IC history
    ic_df = model.get_ic_dataframe()
    if len(ic_df) > 0:
        ic_path = outdir / f"{cfg.name}_rank_ic.csv"
        ic_df.to_csv(ic_path, index=False, float_format="%.6f")
        print(f"[保存] {ic_path.name}")

    # Save feature importance
    imp_df = model.get_importance_dataframe()
    if len(imp_df) > 0:
        imp_path = outdir / f"{cfg.name}_feature_importance.csv"
        imp_df.to_csv(imp_path, index=False, float_format="%.1f")
        print(f"[保存] {imp_path.name}")

    # Save predictions
    pred_path = outdir / f"{cfg.name}_predictions.csv"
    predictions.to_csv(pred_path, index=False, float_format="%.6f")
    print(f"[保存] {pred_path.name}")


# ─────────────────────── 入口 ────────────────────────────

if __name__ == "__main__":
    run_lgb_strategy_weekly()
