"""
LightGBM Cross-Sectional Stock Selection Strategy
===================================================
Strategy entry point that wires ML predictions into the engine
backtest / benchmark / report framework.

Pipeline:
1. Build ML dataset (data_prep)
2. Rolling-window LightGBM training & prediction (lgb_model)
3. Inject pred_score as 'score' → engine backtest loop
4. Generate standard report (dual benchmark, OutputFormat.md compliant)

Usage:
    python strategies/ml/lgb_strategy.py
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
# 所有时间节点均由 DATA_END 反向推导，仅需修改下面两个参数。
#
# |--- 预热期(3年) ---|------ 滚动 训练/验证/测试 ------|--- 留出期 ---|
# WARM_UP_START      MIN_DATE  训练窗口在此滑动   MODEL_CUTOFF     DATA_END
#
# 推导链: DATA_END → MODEL_CUTOFF → MIN_DATE → WARM_UP_START

DATA_END = "2023-02-28"           # 数据集最后日期（唯一需要手动设置的日期）
HOLDOUT_MONTHS = 12               # 留出最后 n 个月作为纯样本外

# ── 模型超参数（影响时间布局） ──
_TRAIN_YEARS = 3                  # 训练窗口长度（年）
_VAL_MONTHS = 6                   # 验证窗口长度（月）
_STEP_MONTHS = 6                  # 滚动步长（月）
_MIN_ROLLING_WINDOWS = 1          # 最少需要的滚动窗口数
_WARM_UP_YEARS = 3                # 预热期长度（年），特征计算需要 ~250 日历史

# ── 从 DATA_END 反向推导全部时间节点 ──
_data_end_dt = pd.Timestamp(DATA_END)

# 1. MODEL_CUTOFF = DATA_END - HOLDOUT_MONTHS
_model_cutoff_dt = _data_end_dt - pd.DateOffset(months=HOLDOUT_MONTHS)

# 2. MIN_DATE: 从 MODEL_CUTOFF 反推，须容纳 train + val + test(至少1个step)
#    第一个窗口: train_end = MIN_DATE + train_years → val_end = +val_months → test = +step_months
#    整个 test 区间须 <= MODEL_CUTOFF，因此需要 train + val + step 的空间。
#    额外 +1 个 step 作为余量，防止月末日期算术边界导致测试集为空。
_min_span_months = (
    _TRAIN_YEARS * 12 + _VAL_MONTHS + _STEP_MONTHS  # 至少一个完整窗口（含test）
    + (_MIN_ROLLING_WINDOWS - 1) * _STEP_MONTHS      # 额外窗口
)
_min_date_dt = _model_cutoff_dt - pd.DateOffset(months=_min_span_months)

# 3. WARM_UP_START = MIN_DATE - WARM_UP_YEARS（特征预热）
_warm_up_start_dt = _min_date_dt - pd.DateOffset(years=_WARM_UP_YEARS)

# ── 校验 ──
if _model_cutoff_dt <= _min_date_dt:
    raise ValueError(
        f"DATA_END={DATA_END} 配合 HOLDOUT_MONTHS={HOLDOUT_MONTHS} "
        f"空间不足，MODEL_CUTOFF={_model_cutoff_dt.date()} <= MIN_DATE={_min_date_dt.date()}。"
        f"请增大 DATA_END 或减小 HOLDOUT_MONTHS。"
    )

MODEL_CUTOFF = _model_cutoff_dt.strftime("%Y-%m-%d")
BACKTEST_START = MODEL_CUTOFF      # 回测区间 = 留出期
MIN_DATE = _min_date_dt.strftime("%Y-%m-%d")
WARM_UP_START = _warm_up_start_dt.strftime("%Y-%m-%d")

print(f"[时间布局] DATA_END={DATA_END}")
print(f"  预热起始  : {WARM_UP_START}")
print(f"  训练起始  : {MIN_DATE}")
print(f"  模型截止  : {MODEL_CUTOFF}")
print(f"  回测/留出 : {MODEL_CUTOFF} ~ {DATA_END}（{HOLDOUT_MONTHS}个月）")

# ─────────────────────── Strategy Config ────────────────────────

config = StrategyConfig(
    name="lgb_stock_selection",
    description="LightGBM cross-sectional stock ranking with rolling retraining. "
                "15 features, 3-year rolling window, biweekly rebalance.",
    rationale=(
        "### Motivation\n\n"
        "The linear multi-factor model (v2) uses fixed weights that cannot capture "
        "non-linear interactions between factors (e.g., momentum works differently "
        "in high-vol vs low-vol regimes). LightGBM learns these conditional effects "
        "automatically from data.\n\n"
        "### Key Design Choices\n\n"
        "1. **Cross-sectional rank labels**: predict relative ranking, not absolute returns. "
        "This removes non-stationarity of return distributions.\n"
        "2. **Rolling 3-year training window**: adapts to regime changes without look-ahead bias.\n"
        "3. **Industry-aware selection**: top 5% by ML score, capped at 5 per industry "
        "(same as v2 for fair comparison).\n"
        "4. **15 features**: 6 base factors (same as v2) + 9 extended features "
        "(short/mid momentum, micro-vol, turnover, price distance, etc.).\n"
        "5. **Feature rank normalization**: all features mapped to [0,1] percentile "
        "within each period, ensuring cross-period comparability.\n"
    ),
    warm_up_start=WARM_UP_START,
    backtest_start=BACKTEST_START,  # Synced with holdout period
    end=DATA_END,
    freq=RebalanceFreq.BIWEEKLY,
    mcap_keep_pct=0.70,
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.05,
    max_per_industry=5,
    min_industry_count=5,
    min_holding=20,
    single_side_cost=0.00015,
    buffer_sigma=0.3,  # Rebalance buffer band (same as v2)
)

# Features used by the model (defined in data_prep.py)
FEATURE_COLS = ALL_FEATURES


# ─────────────────────── Selection Logic ────────────────────────

def ml_select(
    signal: pd.DataFrame,
    prev_holdings: set,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """
    ML-based stock selection (replaces linear scoring).

    Uses 'score' column (= pred_score from LightGBM) for ranking.
    Applies buffer band for incumbent holdings, then selects top stocks
    with industry cap.

    This function has the same signature as engine's _default_select,
    so it can be plugged into StrategyConfig.post_select.
    """
    signal = signal.copy()

    # Buffer band: give incumbents a score bonus
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

    # Cap per industry
    if cfg.max_per_industry > 0:
        selected = (
            selected.sort_values("score", ascending=False)
            .groupby("industry_code", group_keys=False)
            .head(cfg.max_per_industry)
        )

    return selected


# ─────────────────────── Main Pipeline ──────────────────────────

def run_lgb_strategy() -> pd.DataFrame:
    """
    Full LightGBM strategy pipeline:
    1. 构建ML数据集
    2. 滚动训练 & 预测（截至 MODEL_CUTOFF）
    3. 留出期推理（MODEL_CUTOFF ~ DATA_END，使用最后一个模型）
    4. 注入预测分数到引擎回测
    5. 生成报告

    Returns combined DataFrame with portfolio + benchmark returns.
    """
    # ── 使用模块级计算好的 MODEL_CUTOFF ──
    model_cutoff = MODEL_CUTOFF

    print("=" * 64)
    print(f"策略名称 : {config.name}")
    print(f"  {config.description}")
    print(f"数据范围 : {WARM_UP_START} ~ {DATA_END}")
    print(f"训练起始 : {MIN_DATE}")
    print(f"模型截止 : 滚动窗口截至 {model_cutoff}")
    print(f"留出期   : {model_cutoff} ~ {DATA_END}（{HOLDOUT_MONTHS}个月，纯样本外）")
    print(f"回测区间 : {config.backtest_start} ~ {DATA_END}")
    print(f"特征数量 : {len(FEATURE_COLS)}")
    print("=" * 64)

    # ── 第1步: 构建ML数据集（全范围，留出期也需要特征） ──
    print("\n[1] 构建ML数据集 ...")
    dataset = build_ml_dataset(
        warm_up_start=WARM_UP_START,
        backtest_end=DATA_END,
        feature_cols=FEATURE_COLS,
        mcap_keep_pct=config.mcap_keep_pct,
        rank_normalize=True,
    )
    print(f"    数据集: {len(dataset):,} 行, "
          f"{dataset['code'].nunique()} 只股票, "
          f"{dataset['period'].nunique()} 个调仓期")

    # ── 第2步: 滚动LightGBM训练 & 预测（截至 MODEL_CUTOFF） ──
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

    # ── 第2b步: 留出期推理 ──
    # 使用最后一个训练好的模型预测留出期数据
    holdout_mask = pd.to_datetime(dataset["date"]) > pd.Timestamp(model_cutoff)
    holdout_data = dataset[holdout_mask].copy()

    if len(holdout_data) > 0 and len(model.models) > 0:
        print(f"\n[2b] 留出期推理: {len(holdout_data):,} 行, "
              f"{holdout_data['period'].nunique()} 个调仓期 ...")
        last_model = model.models[-1]
        X_holdout = holdout_data[FEATURE_COLS]
        holdout_data["pred_score"] = last_model.predict(X_holdout)

        # Keep same columns as rolling predictions
        keep_cols = [c for c in predictions.columns if c in holdout_data.columns]
        holdout_preds = holdout_data[keep_cols].copy()

        predictions = pd.concat([predictions, holdout_preds], ignore_index=True)
        print(f"    预测总计（滚动 + 留出期）: {len(predictions):,} 行, "
              f"{predictions['period'].nunique()} 个调仓期")
    else:
        print("\n[2b] 无留出期数据或无已训练模型 — 跳过。")

    # ── 第3步: 准备回测数据 ──
    print("\n[3] 准备回测数据 ...")

    # Rename pred_score -> score (engine expects 'score' column)
    snap = predictions.rename(columns={"pred_score": "score"})

    # Filter to backtest period only
    snap["_date"] = pd.to_datetime(snap["date"])
    bt_start = pd.Timestamp(config.backtest_start)
    snap = snap[snap["_date"] >= bt_start].drop(columns=["_date"]).reset_index(drop=True)

    # Ensure period_sort exists and is correct
    if "period_sort" not in snap.columns:
        # Reconstruct from period string (e.g., "2022-07-H2")
        parts = snap["period"].str.extract(r"(\d{4})-(\d{2})-H(\d)")
        snap["period_sort"] = (
            parts[0].astype(int) * 100 + parts[1].astype(int)
        ) * 10 + parts[2].astype(int)

    n_periods = snap["period"].nunique()
    n_stocks = snap["code"].nunique()
    avg_stocks_per_period = len(snap) / n_periods if n_periods > 0 else 0
    print(f"    回测快照: {len(snap):,} 行, "
          f"{n_periods} 个调仓期, {n_stocks} 只股票")
    print(f"    平均每期股票数: {avg_stocks_per_period:.0f}")

    # ── 第4步: 运行回测 ──
    print("\n[4] 运行回测 ...")

    # Use custom selection function
    config.post_select = ml_select

    portfolio_df = run_backtest(snap, config)
    if portfolio_df.empty:
        raise RuntimeError("回测未产生有效数据。")

    ppy = compute_periods_per_year(portfolio_df)
    print(f"    {len(portfolio_df)} 个调仓期, 跨度 "
          f"{len(portfolio_df)/ppy:.1f} 年 = {ppy:.1f} 期/年")

    # ── 第5步: 加载基准 ──
    print("\n[5] 加载基准指数 ...")
    combined = load_all_benchmarks(portfolio_df, config)

    # ── 第6步: 生成报告 ──
    print("\n[6] 生成报告 ...")

    # Build factor list for report display (not used for scoring, just documentation)
    factors_for_report = [FactorDef(f, 0.0) for f in FEATURE_COLS]

    # Build extra report sections with ML-specific info
    extra_sections = _build_ml_report_sections(model, predictions, snap, model_cutoff)

    print_summary(combined.copy(), config, factors_for_report, ppy)
    save_outputs(combined, config)
    write_report(combined.copy(), config, factors_for_report, ppy, extra_sections)

    # ── 第7步: 保存ML产物 ──
    print("\n[7] 保存ML产物 ...")
    _save_ml_artifacts(model, predictions, config)

    print(f"\n✅ {config.name} 策略运行完成。")
    return combined


# ─────────────────────── Report Helpers ─────────────────────────

def _build_ml_report_sections(
    model: RollingLGBModel,
    predictions: pd.DataFrame,
    snap: pd.DataFrame,
    model_cutoff: str,
) -> str:
    """Build extra Markdown sections for the ML-specific report."""
    sections = []

    # ── 留出期标注 ──
    sections.append("## 时间布局\n")
    sections.append(f"- **滚动训练/验证/测试**: 截至 **{model_cutoff}**")
    sections.append(f"- **留出期（纯样本外）**: **{model_cutoff} ~ {DATA_END}** "
                    f"（{HOLDOUT_MONTHS}个月）")
    sections.append(f"- 留出期使用**最后一个训练好的模型**推理，**零数据泄露**。")
    sections.append("")

    # ── Rank IC 汇总 ──
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

        # IC by year
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

    # ── 特征重要性 ──
    imp_df = model.get_importance_dataframe()
    if len(imp_df) > 0:
        mean_imp = imp_df.mean().sort_values(ascending=False)
        sections.append("## 特征重要性（跨窗口平均）\n")
        sections.append("| 排名 | 特征 | 重要性 |")
        sections.append("|------|------|--------|")
        for i, (feat, score) in enumerate(mean_imp.items(), 1):
            sections.append(f"| {i} | {feat} | {score:.1f} |")
        sections.append("")

    # ── 模型架构 ──
    sections.append("## 模型架构\n")
    sections.append("- **模型**: LightGBM（梯度提升决策树）")
    sections.append(f"- **训练窗口**: {model.train_years} 年滚动")
    sections.append(f"- **验证窗口**: {model.val_months} 个月")
    sections.append(f"- **重训频率**: 每 {model.step_months} 个月")
    sections.append(f"- **特征数**: {len(model.feature_cols)} 个，截面排序归一化")
    sections.append(f"- **标签**: 下期收益率，排序归一化至 [0,1]")
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

    # ── 对比说明 ──
    sections.append("## 对比: LightGBM vs 线性多因子 (v2)\n")
    sections.append(
        "| 维度 | 线性v2 | LightGBM |\n"
        "|------|--------|----------|\n"
        "| 打分方式 | 固定线性权重 | 非线性树集成 |\n"
        "| 因子交互 | 无（仅加法） | 自动（条件分裂） |\n"
        "| 市场适应 | 无（静态权重） | 每6个月滚动重训 |\n"
        "| 特征数量 | 6 | 15 |\n"
        "| 特征归一化 | 缩尾z-score | 截面百分位排序 |\n"
        "| 选股方式 | 得分前5% | ML得分前5% |\n"
        "| 行业约束 | 每行业最多5只 | 每行业最多5只 |\n"
        "| 缓冲带 | +0.3σ | +0.3σ（按分数标准差缩放） |\n"
    )

    return "\n".join(sections)


def _save_ml_artifacts(
    model: RollingLGBModel,
    predictions: pd.DataFrame,
    cfg: StrategyConfig,
) -> None:
    """Save ML-specific artifacts (IC history, feature importance, predictions)."""
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

    # Save predictions (for further analysis)
    pred_path = outdir / f"{cfg.name}_predictions.csv"
    predictions.to_csv(pred_path, index=False, float_format="%.6f")
    print(f"[保存] {pred_path.name}")


# ─────────────────────── Entry Point ────────────────────────────

if __name__ == "__main__":
    run_lgb_strategy()
