"""
lgb_v2_2018.py — 2018年回测段（去杠杆熊市验证）
DATA_END = 2018-12-31，留出期 = 2018 全年
"""
import sys, pandas as pd
sys.path.insert(0, '/projects/portal/data/quant')

from strategies.ml.data_prep_v2 import build_ml_dataset, ALL_FEATURES
from strategies.ml.lgb_strategy_v2 import (
    rolling_train_predict_v2, ml_select_v2, _build_ic_section,
)
from engine.types import StrategyConfig, FactorDef, SelectionMode, RebalanceFreq
from engine.backtest import run_backtest
from engine.benchmark import load_all_benchmarks
from engine.report import compute_periods_per_year, print_summary, save_outputs, write_report

DATA_END       = "2018-12-31"
HOLDOUT_MONTHS = 12
WARM_UP_START  = "2010-01-01"
MIN_DATE       = "2013-01-01"
MODEL_CUTOFF   = (pd.Timestamp(DATA_END) - pd.DateOffset(months=HOLDOUT_MONTHS)).strftime("%Y-%m-%d")
BACKTEST_START = MODEL_CUTOFF

print("=" * 64)
print(f"[2018段] DATA_END={DATA_END}")
print(f"  MODEL_CUTOFF = {MODEL_CUTOFF}")
print(f"  留出期 = {MODEL_CUTOFF} ~ {DATA_END}")
print("=" * 64)

dataset = build_ml_dataset(
    warm_up_start=WARM_UP_START,
    backtest_end=DATA_END,
    feature_cols=ALL_FEATURES,
    mcap_keep_pct=0.70,
    rank_normalize=True,
)

predictions, models, ic_history = rolling_train_predict_v2(
    dataset, ALL_FEATURES, MIN_DATE, MODEL_CUTOFF
)

holdout = dataset[pd.to_datetime(dataset["date"]) > pd.Timestamp(MODEL_CUTOFF)].copy()
if len(holdout) > 0 and models:
    holdout["pred_score"] = models[-1].predict(holdout[ALL_FEATURES])
    keep = [c for c in predictions.columns if c in holdout.columns]
    predictions = pd.concat([predictions, holdout[keep]], ignore_index=True)
    print(f"留出期推理: {len(holdout):,} 行, {holdout['period'].nunique()} 个调仓期")

bt = predictions.rename(columns={"pred_score": "score"}).copy()
bt = bt[pd.to_datetime(bt["date"]) >= pd.Timestamp(BACKTEST_START)].reset_index(drop=True)
if "period_sort" not in bt.columns:
    p = bt["period"].str.extract(r"(\d{4})-(\d{2})-H(\d)")
    bt["period_sort"] = (p[0].astype(int)*100 + p[1].astype(int))*10 + p[2].astype(int)

config = StrategyConfig(
    name="lgb_v2_2018",
    description="LightGBM V2 回测段：2018年（去杠杆熊市验证）",
    rationale=(
        "验证策略在 2018 年去杠杆熊市中的表现。\n"
        f"DATA_END={DATA_END}，MODEL_CUTOFF={MODEL_CUTOFF}，留出期=2018全年。\n"
        "滚动窗口训练至 2017-12，用最后一个模型推理整个 2018 年。"
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

portfolio_df = run_backtest(bt, config)
if portfolio_df.empty:
    raise RuntimeError("回测无数据")

ppy = compute_periods_per_year(portfolio_df)
combined = load_all_benchmarks(portfolio_df, config)
factors_for_report = [FactorDef(f, 0.0) for f in ALL_FEATURES]
print_summary(combined.copy(), config, factors_for_report, ppy)
save_outputs(combined, config)

ic_df = pd.DataFrame(ic_history)
extra = _build_ic_section(ic_df, len(models))
write_report(combined.copy(), config, factors_for_report, ppy, extra)
print("\n✅ 2018段完成")
