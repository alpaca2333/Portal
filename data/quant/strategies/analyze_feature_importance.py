"""
Feature Importance Analysis for LightGBM Cross-Sectional Strategy (V4)
======================================================================
Trains the V4 model once and extracts feature importance rankings.

Usage:
    cd <project_root>
    python -m data.quant.strategies.analyze_feature_importance
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import lightgbm as lgb

from engine import BacktestConfig
from engine.data_loader import DataAccessor

from data.quant.strategies.lgbm_cross_sectional import (
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    prefetch_bulk_data,
    compute_features_from_memory,
    compute_forward_return_from_memory,
    rank_normalize,
    train_lgbm_model,
    _bulk_date_index_cache,
)


def main():
    print("=" * 70)
    print("  LightGBM V4 特征重要性分析")
    print("=" * 70)

    # ── Step 1: Connect to DB
    db_path = "data/quant/data/quant.db"
    cfg = BacktestConfig(db_path=db_path)
    accessor = DataAccessor(cfg)
    accessor.open()
    print(f"\n[1/5] 数据库连接成功: {db_path}")

    # ── Step 2: Load ST codes
    st_df = pd.read_sql_query(
        "SELECT ts_code, name FROM stock_info WHERE name LIKE '%ST%'",
        accessor.conn,
    )
    st_codes = set(st_df["ts_code"].tolist())
    print(f"[2/5] ST 股票: {len(st_codes)} 只")

    # ── Step 3: Bulk load data
    # Use 2015-01-01 ~ 2025-06-30 to cover 3-year training + backtest
    data_start = pd.Timestamp("2014-01-01")
    data_end = pd.Timestamp("2025-06-30")
    print(f"[3/5] 批量加载数据 {data_start.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')} ...")
    bulk_data = prefetch_bulk_data(accessor, data_start, data_end)
    _bulk_date_index_cache.clear()

    # ── Step 4: Build training dataset
    # Use biweekly dates from 2015-01-01 to 2025-01-01 as training window
    print(f"[4/5] 构建训练数据集 ...")

    all_dates = np.sort(bulk_data["trade_date"].unique())
    # Pick biweekly rebalance dates from 2015-06 to 2025-01
    mask = (all_dates >= pd.Timestamp("2015-06-01")) & (all_dates <= pd.Timestamp("2025-01-01"))
    trade_dates = pd.DatetimeIndex(all_dates[mask])

    origin = trade_dates[0]
    day_offsets = (trade_dates - origin).days
    block_ids = day_offsets // 14
    s = pd.Series(trade_dates, index=trade_dates)
    groups = s.groupby(block_ids)
    rebal_dates = pd.DatetimeIndex(groups.last().values)

    print(f"      调仓日: {len(rebal_dates)} 个")

    all_X = []
    all_y = []
    n_success = 0
    n_skip = 0

    for i in range(len(rebal_dates) - 1):
        d = rebal_dates[i]
        d_next = rebal_dates[i + 1]

        feat_df = compute_features_from_memory(
            d, bulk_data, lookback=260, st_codes=st_codes,
        )
        if feat_df is None or feat_df.empty:
            n_skip += 1
            continue

        # Market cap filter: top 85%
        if "circ_mv" in feat_df.columns:
            mv = feat_df["circ_mv"].dropna()
            if len(mv) > 0:
                lower_bound = mv.quantile(1.0 - 0.85)
                feat_df = feat_df[feat_df["circ_mv"] >= lower_bound].copy()

        if len(feat_df) < 50:
            n_skip += 1
            continue

        feat_ranked = rank_normalize(feat_df, FEATURE_NAMES)

        fwd_ret = compute_forward_return_from_memory(d, d_next, bulk_data)
        if fwd_ret is None:
            n_skip += 1
            continue

        merged = feat_ranked.set_index("ts_code").join(
            fwd_ret.rename("label"), how="inner"
        )
        if len(merged) >= 50:
            merged["label"] = merged["label"].rank(pct=True, method="average")
            all_X.append(merged[FEATURE_NAMES])
            all_y.append(merged["label"])
            n_success += 1

        if n_success % 50 == 0 and n_success > 0:
            print(f"      已处理 {n_success} 期 ...")

    print(f"      训练数据: {n_success} 期成功, {n_skip} 期跳过")
    print(f"      总样本量: {sum(len(x) for x in all_X):,}")

    # ── Step 5: Train model and extract importance
    print(f"[5/5] 训练 LightGBM 模型 ...")

    n = len(all_X)
    split = max(1, int(n * 0.8))

    train_X = pd.concat(all_X[:split])
    train_y = pd.concat(all_y[:split])
    val_X = pd.concat(all_X[split:]) if split < n else None
    val_y = pd.concat(all_y[split:]) if split < n else None

    print(f"      训练集: {len(train_y):,} 样本 ({split} 期)")
    print(f"      验证集: {len(val_y):,} 样本 ({n - split} 期)" if val_y is not None else "      验证集: 无")

    model = train_lgbm_model(train_X, train_y, val_X, val_y)
    best_iter = model.best_iteration if hasattr(model, 'best_iteration') else '?'
    print(f"      模型训练完成, 最优轮次: {best_iter}")

    # ── Extract feature importance
    imp_split = model.feature_importance(importance_type="split")
    imp_gain = model.feature_importance(importance_type="gain")

    imp_df = pd.DataFrame({
        "特征名": FEATURE_NAMES,
        "分裂次数(split)": imp_split,
        "信息增益(gain)": imp_gain,
    })

    # Normalize to percentage
    imp_df["split占比(%)"] = (imp_df["分裂次数(split)"] / imp_df["分裂次数(split)"].sum() * 100).round(2)
    imp_df["gain占比(%)"] = (imp_df["信息增益(gain)"] / imp_df["信息增益(gain)"].sum() * 100).round(2)

    # Add category labels
    category_map = {
        "mom_12_1": "动量", "mom_3_1": "动量", "mom_6_1": "动量",
        "rev_10": "反转",
        "rvol_20": "风险", "ret_5d_std": "风险", "high_low_20": "风险",
        "vol_confirm": "量价", "turnover_20": "量价", "volume_chg": "量价",
        "inv_pb": "价值", "pe_ttm": "价值",
        "log_cap": "规模",
        "roe_ttm": "质量", "roa_ttm": "质量", "gross_margin": "质量", "low_leverage": "质量",
        "dv_ttm": "分红",
        "growth_revenue": "成长", "growth_profit": "成长",
        "illiq_20": "流动性",
        "close_to_high_60": "技术",
    }
    imp_df["大类"] = imp_df["特征名"].map(category_map).fillna("其他")

    # Sort by gain importance
    imp_df = imp_df.sort_values("信息增益(gain)", ascending=False).reset_index(drop=True)
    imp_df.index = imp_df.index + 1
    imp_df.index.name = "排名"

    # ── Print results
    print("\n" + "=" * 70)
    print("  特征重要性排名 (按信息增益 gain 降序)")
    print("=" * 70)
    print()

    # Print table
    header = f"{'排名':>4}  {'特征名':<20}  {'大类':<6}  {'split占比':>10}  {'gain占比':>10}  {'重要程度':<10}"
    print(header)
    print("-" * len(header))

    total_gain = imp_df["gain占比(%)"].sum()
    for idx, row in imp_df.iterrows():
        gain_pct = row["gain占比(%)"]
        # Visual bar
        bar_len = int(gain_pct / 2)
        bar = "█" * bar_len
        level = ""
        if gain_pct >= 8:
            level = "★★★ 核心"
        elif gain_pct >= 5:
            level = "★★ 重要"
        elif gain_pct >= 3:
            level = "★ 一般"
        else:
            level = "○ 边缘"

        print(f"{idx:>4}  {row['特征名']:<20}  {row['大类']:<6}  {row['split占比(%)']:>8.2f}%  {gain_pct:>8.2f}%  {level:<10}  {bar}")

    # ── Category summary
    print("\n" + "=" * 70)
    print("  按大类汇总")
    print("=" * 70)
    cat_summary = imp_df.groupby("大类").agg({
        "split占比(%)": "sum",
        "gain占比(%)": "sum",
        "特征名": "count",
    }).rename(columns={"特征名": "因子数"})
    cat_summary = cat_summary.sort_values("gain占比(%)", ascending=False)

    print(f"\n{'大类':<8}  {'因子数':>6}  {'split占比':>10}  {'gain占比':>10}")
    print("-" * 45)
    for cat, row in cat_summary.iterrows():
        print(f"{cat:<8}  {int(row['因子数']):>6}  {row['split占比(%)']:>8.2f}%  {row['gain占比(%)']:>8.2f}%")

    # ── Top / Bottom analysis
    print("\n" + "=" * 70)
    print("  关键发现")
    print("=" * 70)

    top3 = imp_df.head(3)
    bottom3 = imp_df.tail(3)

    print(f"\n  🔝 最重要的 3 个因子 (占总 gain {top3['gain占比(%)'].sum():.1f}%):")
    for _, row in top3.iterrows():
        print(f"     - {row['特征名']} ({row['大类']}): gain {row['gain占比(%)']:.2f}%")

    print(f"\n  🔻 最不重要的 3 个因子 (占总 gain {bottom3['gain占比(%)'].sum():.1f}%):")
    for _, row in bottom3.iterrows():
        print(f"     - {row['特征名']} ({row['大类']}): gain {row['gain占比(%)']:.2f}%")

    # Check redundancy
    mom_factors = imp_df[imp_df["大类"] == "动量"]
    risk_factors = imp_df[imp_df["大类"] == "风险"]
    value_factors = imp_df[imp_df["大类"] == "价值"]

    print(f"\n  📊 冗余分析:")
    print(f"     动量类 ({len(mom_factors)} 个因子): 合计 gain {mom_factors['gain占比(%)'].sum():.2f}%")
    for _, row in mom_factors.iterrows():
        print(f"       - {row['特征名']}: {row['gain占比(%)']:.2f}%")

    print(f"     风险类 ({len(risk_factors)} 个因子): 合计 gain {risk_factors['gain占比(%)'].sum():.2f}%")
    for _, row in risk_factors.iterrows():
        print(f"       - {row['特征名']}: {row['gain占比(%)']:.2f}%")

    print(f"     价值类 ({len(value_factors)} 个因子): 合计 gain {value_factors['gain占比(%)'].sum():.2f}%")
    for _, row in value_factors.iterrows():
        print(f"       - {row['特征名']}: {row['gain占比(%)']:.2f}%")

    print("\n" + "=" * 70)
    print("  分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
