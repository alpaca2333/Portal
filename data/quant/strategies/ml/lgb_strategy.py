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
    warm_up_start="2016-01-01",
    backtest_start="2022-07-01",  # First valid test window output
    end="2026-02-28",
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
    1. Build ML dataset
    2. Rolling train & predict
    3. Inject scores into engine backtest
    4. Generate report

    Returns combined DataFrame with portfolio + benchmark returns.
    """
    print("=" * 64)
    print(f"Strategy: {config.name}")
    print(f"  {config.description}")
    print(f"Period  : {config.backtest_start} ~ {config.end}")
    print(f"Features: {len(FEATURE_COLS)}")
    print("=" * 64)

    # ── Step 1: Build ML dataset ──
    print("\n[1] Building ML dataset ...")
    dataset = build_ml_dataset(
        warm_up_start="2016-01-01",
        backtest_end=config.end,
        feature_cols=FEATURE_COLS,
        mcap_keep_pct=config.mcap_keep_pct,
        rank_normalize=True,
    )
    print(f"    Dataset: {len(dataset):,} rows, "
          f"{dataset['code'].nunique()} stocks, "
          f"{dataset['period'].nunique()} periods")

    # ── Step 2: Rolling LightGBM training & prediction ──
    print("\n[2] Rolling LightGBM training ...")
    model = RollingLGBModel(
        train_years=3,
        val_months=6,
        step_months=6,
    )
    predictions = model.rolling_train_predict(
        dataset=dataset,
        feature_cols=FEATURE_COLS,
        label_col="label",
        min_date="2019-01-01",
        max_date=config.end,
    )
    print(f"    Predictions: {len(predictions):,} rows, "
          f"{predictions['period'].nunique()} periods")

    # ── Step 3: Prepare snap for engine backtest ──
    print("\n[3] Preparing data for engine backtest ...")

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
    print(f"    Backtest snap: {len(snap):,} rows, "
          f"{n_periods} periods, {n_stocks} stocks")
    print(f"    Avg stocks/period: {avg_stocks_per_period:.0f}")

    # ── Step 4: Run backtest ──
    print("\n[4] Running backtest ...")

    # Use custom selection function
    config.post_select = ml_select

    portfolio_df = run_backtest(snap, config)
    if portfolio_df.empty:
        raise RuntimeError("Backtest produced no valid observations.")

    ppy = compute_periods_per_year(portfolio_df)
    print(f"    {len(portfolio_df)} periods over "
          f"{len(portfolio_df)/ppy:.1f} years = {ppy:.1f} periods/year")

    # ── Step 5: Load benchmarks ──
    print("\n[5] Loading benchmarks ...")
    combined = load_all_benchmarks(portfolio_df, config)

    # ── Step 6: Generate report ──
    print("\n[6] Generating report ...")

    # Build factor list for report display (not used for scoring, just documentation)
    factors_for_report = [FactorDef(f, 0.0) for f in FEATURE_COLS]

    # Build extra report sections with ML-specific info
    extra_sections = _build_ml_report_sections(model, predictions, snap)

    print_summary(combined.copy(), config, factors_for_report, ppy)
    save_outputs(combined, config)
    write_report(combined.copy(), config, factors_for_report, ppy, extra_sections)

    # ── Step 7: Save ML-specific artifacts ──
    print("\n[7] Saving ML artifacts ...")
    _save_ml_artifacts(model, predictions, config)

    print(f"\n✅ {config.name} strategy complete.")
    return combined


# ─────────────────────── Report Helpers ─────────────────────────

def _build_ml_report_sections(
    model: RollingLGBModel,
    predictions: pd.DataFrame,
    snap: pd.DataFrame,
) -> str:
    """Build extra Markdown sections for the ML-specific report."""
    sections = []

    # ── Rank IC Summary ──
    ic_df = model.get_ic_dataframe()
    if len(ic_df) > 0:
        summary = ic_summary(ic_df["rank_ic"])
        sections.append("## Model Quality: Rank IC (Out-of-Sample)\n")
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        sections.append(f"| Mean Rank IC | {summary['mean_ic']:.4f} |")
        sections.append(f"| Std Rank IC | {summary['std_ic']:.4f} |")
        sections.append(f"| IC IR (Mean/Std) | {summary['ir']:.2f} |")
        sections.append(f"| IC > 0 Rate | {summary['ic_positive_rate']:.1%} |")
        sections.append(f"| Num Periods | {len(ic_df)} |")
        sections.append("")

        # IC by year
        ic_df = ic_df.copy()
        ic_df["_date"] = ic_df["period"].str[:4]
        yearly_ic = ic_df.groupby("_date")["rank_ic"].agg(["mean", "std", "count"])
        if len(yearly_ic) > 0:
            sections.append("### Rank IC by Year\n")
            sections.append("| Year | Mean IC | Std IC | Periods |")
            sections.append("|------|---------|--------|---------|")
            for year, row in yearly_ic.iterrows():
                sections.append(
                    f"| {year} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |"
                )
            sections.append("")

    # ── Feature Importance ──
    imp_df = model.get_importance_dataframe()
    if len(imp_df) > 0:
        mean_imp = imp_df.mean().sort_values(ascending=False)
        sections.append("## Feature Importance (Avg Across Windows)\n")
        sections.append("| Rank | Feature | Importance |")
        sections.append("|------|---------|------------|")
        for i, (feat, score) in enumerate(mean_imp.items(), 1):
            sections.append(f"| {i} | {feat} | {score:.1f} |")
        sections.append("")

    # ── Model Architecture ──
    sections.append("## Model Architecture\n")
    sections.append("- **Model**: LightGBM (Gradient Boosted Decision Trees)")
    sections.append(f"- **Training window**: {model.train_years} years rolling")
    sections.append(f"- **Validation window**: {model.val_months} months")
    sections.append(f"- **Retrain frequency**: every {model.step_months} months")
    sections.append(f"- **Features**: {len(model.feature_cols)} cross-sectional rank-normalized")
    sections.append(f"- **Label**: next-period forward return, rank-normalized to [0,1]")
    sections.append(f"- **Num training windows**: {len(model.models)}")
    sections.append("")

    # Key hyperparameters
    sections.append("### Key Hyperparameters\n")
    sections.append("| Parameter | Value |")
    sections.append("|-----------|-------|")
    for key in ["num_leaves", "max_depth", "learning_rate", "n_estimators",
                 "min_child_samples", "subsample", "colsample_bytree",
                 "reg_alpha", "reg_lambda"]:
        if key in model.params:
            sections.append(f"| {key} | {model.params[key]} |")
    sections.append("")

    # ── Comparison note ──
    sections.append("## Comparison: LightGBM vs Linear Multi-Factor (v2)\n")
    sections.append(
        "| Dimension | Linear v2 | LightGBM |\n"
        "|-----------|-----------|----------|\n"
        "| Scoring method | Fixed linear weights | Non-linear tree ensemble |\n"
        "| Factor interactions | None (additive only) | Automatic (conditional splits) |\n"
        "| Regime adaptation | None (static weights) | Rolling retrain every 6 months |\n"
        "| Num features | 6 | 15 |\n"
        "| Feature normalization | Winsorized z-score | Cross-sectional percentile rank |\n"
        "| Selection | Top 5% by score | Top 5% by ML score |\n"
        "| Industry constraint | Max 5/industry | Max 5/industry |\n"
        "| Buffer band | +0.3σ | +0.3σ (scaled by score std) |\n"
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
        print(f"[save] {ic_path.name}")

    # Save feature importance
    imp_df = model.get_importance_dataframe()
    if len(imp_df) > 0:
        imp_path = outdir / f"{cfg.name}_feature_importance.csv"
        imp_df.to_csv(imp_path, index=False, float_format="%.1f")
        print(f"[save] {imp_path.name}")

    # Save predictions (for further analysis)
    pred_path = outdir / f"{cfg.name}_predictions.csv"
    predictions.to_csv(pred_path, index=False, float_format="%.6f")
    print(f"[save] {pred_path.name}")


# ─────────────────────── Entry Point ────────────────────────────

if __name__ == "__main__":
    run_lgb_strategy()
