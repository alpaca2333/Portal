"""
LightGBM 截面选股策略 V2 —— 完全独立复现版
============================================
目的：数据管道 + 模型训练全部独立实现，与 lgb_strategy.py 交叉验证。

复现边界：
  - 数据管道：data_prep_v2.py（从零独立实现，不引用任何原始代码或字节码）
  - 模型训练/滚动窗口/IC 计算：独立实现，不引用 lgb_model.py

用法：
    python strategies/ml/lgb_strategy_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
from engine.types import StrategyConfig, FactorDef, SelectionMode, RebalanceFreq
from engine.backtest import run_backtest
from engine.benchmark import load_all_benchmarks
from engine.report import compute_periods_per_year, print_summary, save_outputs, write_report

# ── 使用完全独立实现的数据管道 ──
from strategies.ml.data_prep_v2 import build_ml_dataset, ALL_FEATURES

# ──────────────────────── 时间参数（与 V1 完全一致）────────────
DATA_END        = "2026-02-28"
HOLDOUT_MONTHS  = 12
TRAIN_YEARS     = 3
VAL_MONTHS      = 6
STEP_MONTHS     = 6
WARM_UP_YEARS   = 3

_end    = pd.Timestamp(DATA_END)
_cutoff = _end - pd.DateOffset(months=HOLDOUT_MONTHS)
_data_from = pd.Timestamp("2010-01-01")
_min_date  = _data_from + pd.DateOffset(years=WARM_UP_YEARS)

MODEL_CUTOFF   = _cutoff.strftime("%Y-%m-%d")
BACKTEST_START = MODEL_CUTOFF
MIN_DATE       = _min_date.strftime("%Y-%m-%d")
WARM_UP_START  = _data_from.strftime("%Y-%m-%d")

print("[V2 时间布局]")
print(f"  数据起始 : {WARM_UP_START}")
print(f"  训练起始 : {MIN_DATE}")
print(f"  模型截止 : {MODEL_CUTOFF}")
print(f"  留出期   : {MODEL_CUTOFF} ~ {DATA_END} ({HOLDOUT_MONTHS}个月)")

# ──────────────────────── LGB 超参数（与 V1 完全一致）──────────
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


# ──────────────────────── 滚动训练（独立实现）─────────────────

def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
           purge: bool = False) -> pd.DataFrame:
    m = (df["_dt"] >= start) & (df["_dt"] <= end)
    if purge and "_led" in df.columns:
        m &= df["_led"].notna() & (df["_led"] <= end)
    return df[m].copy()


def rolling_train_predict_v2(
    dataset: pd.DataFrame,
    feature_cols: list,
    min_date: str,
    max_date: str,
) -> tuple[pd.DataFrame, list, list]:
    """
    独立实现的滚动窗口训练 & 预测。

    窗口结构（对齐 V1）：
      train : [T - TRAIN_YEARS,  T)               —— purge by label_end_date
      val   : [T,  T + VAL_MONTHS)                —— purge by label_end_date
      test  : [T + VAL_MONTHS,  T + VAL_MONTHS + STEP_MONTHS)
    """
    ds = dataset.copy()
    ds["_dt"] = pd.to_datetime(ds["date"])
    if "label_end_date" in ds.columns:
        ds["_led"] = pd.to_datetime(ds["label_end_date"])

    min_dt = pd.Timestamp(min_date)
    max_dt = pd.Timestamp(max_date)

    models, ic_history, all_preds = [], [], []

    t = min_dt + pd.DateOffset(years=TRAIN_YEARS)
    win_idx = 0

    while t <= max_dt:
        win_idx += 1
        ts = t - pd.DateOffset(years=TRAIN_YEARS)
        te = t - pd.DateOffset(days=1)
        vs = t
        ve = t + pd.DateOffset(months=VAL_MONTHS) - pd.DateOffset(days=1)
        xs = t + pd.DateOffset(months=VAL_MONTHS)
        xe = min(t + pd.DateOffset(months=VAL_MONTHS + STEP_MONTHS) - pd.DateOffset(days=1), max_dt)

        if xs > max_dt:
            break

        tr = _slice(ds, ts, te, purge=True).dropna(subset=["label"])
        va = _slice(ds, vs, ve, purge=True).dropna(subset=["label"])
        te_ = _slice(ds, xs, xe, purge=False)

        if len(tr) == 0 or len(te_) == 0:
            t += pd.DateOffset(months=STEP_MONTHS)
            continue

        if len(va) == 0:
            sorted_tr = tr.sort_values("_dt")
            split = int(len(sorted_tr) * 0.8)
            va = sorted_tr.iloc[split:]
            tr = sorted_tr.iloc[:split]

        print(f"\n[V2] 窗口 {win_idx}: "
              f"训练 {ts.date()}~{te.date()} ({len(tr):,}), "
              f"验证 ({len(va):,}), "
              f"测试 {xs.date()}~{xe.date()} ({len(te_):,})")

        # ── 训练（独立：直接用 sklearn API，不封装成 RollingLGBModel）──
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            tr[feature_cols], tr["label"].values,
            eval_set=[(va[feature_cols], va["label"].values)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        best = model.best_iteration_ if (model.best_iteration_ or 0) > 0 else LGB_PARAMS["n_estimators"]
        print(f"  最佳迭代: {best}")
        models.append(model)

        # ── 预测 ──
        te_ = te_.copy()
        te_["pred_score"] = model.predict(te_[feature_cols])

        # ── Rank IC（独立计算：不调用 rank_ic_by_period）──
        for period, g in te_.groupby("period"):
            valid = g[["fwd_ret", "pred_score"]].dropna()
            if len(valid) < 10:
                continue
            rho, _ = stats.spearmanr(valid["fwd_ret"], valid["pred_score"])
            ic_history.append({
                "window": win_idx, "period": period,
                "rank_ic": rho, "n_stocks": len(valid),
            })

        keep = [c for c in [
            "code", "date", "period", "period_sort",
            "close", "next_open", "next_date", "label_end_date",
            "free_market_cap", "industry_code", "industry_name",
            "pred_score", "fwd_ret", "label",
        ] if c in te_.columns]
        all_preds.append(te_[keep])

        t += pd.DateOffset(months=STEP_MONTHS)

    if not all_preds:
        raise RuntimeError("没有生成任何有效预测窗口。")

    preds = pd.concat(all_preds, ignore_index=True)
    preds = (preds.sort_values(["period", "code"])
             .drop_duplicates(subset=["period", "code"], keep="last")
             .reset_index(drop=True))

    # ── IC 汇总 ──
    ic_df = pd.DataFrame(ic_history)
    if len(ic_df) > 0:
        vals = ic_df["rank_ic"].dropna()
        m, s = vals.mean(), vals.std(ddof=0)
        print("\n" + "=" * 50)
        print("Rank IC 汇总（V2 样本外）")
        print("=" * 50)
        print(f"  平均 IC     : {m:.4f}")
        print(f"  IC 标准差   : {s:.4f}")
        print(f"  IC IR       : {m/s if s>0 else float('nan'):.2f}")
        print(f"  IC > 0 占比 : {(vals > 0).mean():.1%}")
        print(f"  调仓期数    : {len(ic_df)}")
        print("=" * 50)

    return preds, models, ic_history


# ──────────────────────── 选股（独立实现）─────────────────────

def ml_select_v2(signal: pd.DataFrame, prev_holdings: set,
                 cfg: StrategyConfig) -> pd.DataFrame:
    signal = signal.copy()
    if cfg.buffer_sigma > 0 and prev_holdings:
        std = signal["score"].std()
        if std > 0:
            signal.loc[signal["code"].isin(prev_holdings), "score"] += cfg.buffer_sigma * std
    cutoff = signal["score"].quantile(1 - cfg.top_pct)
    sel = signal[signal["score"] >= cutoff].copy()
    if cfg.max_per_industry > 0:
        sel = (sel.sort_values("score", ascending=False)
               .groupby("industry_code", group_keys=False)
               .head(cfg.max_per_industry))
    return sel


# ──────────────────────── 主流程 ──────────────────────────────

def run_lgb_v2() -> pd.DataFrame:
    print("=" * 64)
    print("策略名称 : lgb_stock_selection_v2  (独立复现)")
    print(f"数据范围 : {WARM_UP_START} ~ {DATA_END}")
    print(f"训练起始 : {MIN_DATE}  |  模型截止 : {MODEL_CUTOFF}")
    print("=" * 64)

    # ── 1. 数据管道（复用 data_prep 确保输入一致）──
    print("\n[1] 构建 ML 数据集 ...")
    dataset = build_ml_dataset(
        warm_up_start=WARM_UP_START,
        backtest_end=DATA_END,
        feature_cols=ALL_FEATURES,
        mcap_keep_pct=0.70,
        rank_normalize=True,
    )
    print(f"    数据集: {len(dataset):,} 行, "
          f"{dataset['code'].nunique()} 只股票, "
          f"{dataset['period'].nunique()} 个调仓期")

    # ── 2. 滚动训练（V2 独立实现）──
    print(f"\n[2] 滚动训练 (max_date={MODEL_CUTOFF}) ...")
    predictions, models, ic_history = rolling_train_predict_v2(
        dataset, ALL_FEATURES, MIN_DATE, MODEL_CUTOFF
    )
    print(f"    滚动预测: {len(predictions):,} 行, "
          f"{predictions['period'].nunique()} 个调仓期")

    # ── 3. 留出期推理 ──
    holdout = dataset[pd.to_datetime(dataset["date"]) > pd.Timestamp(MODEL_CUTOFF)].copy()
    if len(holdout) > 0 and models:
        holdout["pred_score"] = models[-1].predict(holdout[ALL_FEATURES])
        keep = [c for c in predictions.columns if c in holdout.columns]
        predictions = pd.concat([predictions, holdout[keep]], ignore_index=True)
        print(f"    加入留出期后: {len(predictions):,} 行, "
              f"{predictions['period'].nunique()} 个调仓期")

    # ── 4. 回测数据准备 ──
    bt = predictions.rename(columns={"pred_score": "score"}).copy()
    bt = bt[pd.to_datetime(bt["date"]) >= pd.Timestamp(BACKTEST_START)].reset_index(drop=True)
    if "period_sort" not in bt.columns:
        p = bt["period"].str.extract(r"(\d{4})-(\d{2})-H(\d)")
        bt["period_sort"] = (p[0].astype(int) * 100 + p[1].astype(int)) * 10 + p[2].astype(int)

    print(f"\n[3] 回测快照: {len(bt):,} 行, {bt['period'].nunique()} 个调仓期")

    # ── 5. 策略配置 & 回测 ──
    config = StrategyConfig(
        name="lgb_stock_selection_v2",
        description="LightGBM 截面选股 V2（独立复现）：15特征，3年滚动，双周调仓。",
        rationale=(
            "独立复现 lgb_stock_selection，用于结果交叉验证。\n"
            "数据管道复用 data_prep，模型训练逻辑独立实现。"
        ),
        warm_up_start=WARM_UP_START,
        backtest_start=BACKTEST_START,
        end=DATA_END,
        freq=RebalanceFreq.BIWEEKLY,
        mcap_keep_pct=0.70,
        selection_mode=SelectionMode.TOP_PCT,
        top_pct=0.05,
        max_per_industry=5,
        min_holding=20,
        single_side_cost=0.00015,
        buffer_sigma=0.3,
        post_select=ml_select_v2,
    )

    print("\n[4] 运行回测 ...")
    portfolio_df = run_backtest(bt, config)
    if portfolio_df.empty:
        raise RuntimeError("回测未产生有效数据。")

    ppy = compute_periods_per_year(portfolio_df)
    combined = load_all_benchmarks(portfolio_df, config)

    print("\n[5] 生成报告 ...")
    factors_for_report = [FactorDef(f, 0.0) for f in ALL_FEATURES]
    print_summary(combined.copy(), config, factors_for_report, ppy)
    save_outputs(combined, config)

    ic_df = pd.DataFrame(ic_history)
    extra = _build_ic_section(ic_df, len(models))
    write_report(combined.copy(), config, factors_for_report, ppy, extra)

    print("\n✅ lgb_stock_selection_v2 完成。")
    return combined


# ──────────────────────── 报告辅助 ────────────────────────────

def _build_ic_section(ic_df: pd.DataFrame, n_models: int) -> str:
    lines = []
    lines.append("## 时间布局\n")
    lines.append(f"- 滚动训练: {MIN_DATE} ~ {MODEL_CUTOFF}")
    lines.append(f"- 留出期: {MODEL_CUTOFF} ~ {DATA_END} ({HOLDOUT_MONTHS}个月)")
    lines.append(f"- 滚动窗口数: {n_models}")
    lines.append("")
    if len(ic_df) > 0:
        vals = ic_df["rank_ic"].dropna()
        m, s = vals.mean(), vals.std(ddof=0)
        ir = m / s if s > 0 else float("nan")
        lines.append("## Rank IC（样本外）\n")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| 平均 IC | {m:.4f} |")
        lines.append(f"| IC IR | {ir:.2f} |")
        lines.append(f"| IC>0 占比 | {(vals>0).mean():.1%} |")
        lines.append(f"| 期数 | {len(ic_df)} |")
        lines.append("")
        ic_df2 = ic_df.copy()
        ic_df2["year"] = ic_df2["period"].str[:4]
        yearly = ic_df2.groupby("year")["rank_ic"].agg(["mean", "count"])
        lines.append("### 逐年 IC\n")
        lines.append("| 年份 | 均值 | 期数 |")
        lines.append("|---|---|---|")
        for yr, row in yearly.iterrows():
            lines.append(f"| {yr} | {row['mean']:.4f} | {int(row['count'])} |")
    lines.append("\n**注**: 本策略为独立复现版，模型训练逻辑不依赖 lgb_model.py。")
    return "\n".join(lines)


if __name__ == "__main__":
    run_lgb_v2()
