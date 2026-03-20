"""
Data preparation for ML-based stock selection.
================================================
Responsibilities:
1. Load raw data from SQLite (reusing engine.data)
2. Compute extended features (beyond the 6 base factors)
3. Build forward-return labels (configurable horizon)
4. Cross-sectional rank normalization per rebalance date
5. Time-series split (rolling or expanding window)

Usage:
    from strategies.ml.data_prep import build_ml_dataset, time_split
"""
from __future__ import annotations

import gc
import sqlite3
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────── Constants ──────────────────────────────

DB_PATH = "/projects/portal/data/quant/processed/stocks.db"

# Standard feature list (column names after compute)
BASE_FEATURES = [
    "mom_12_1",    # 12-1 momentum
    "rev_10",      # 10-day reversal
    "rvol_20",     # 20-day realized vol
    "vol_confirm", # volume confirmation (20d/120d MA)
    "inv_pb",      # 1/PB (value)
    "log_cap",     # log(free_market_cap) (size)
]

EXTRA_FEATURES = [
    "pe_ttm",      # PE ratio (TTM)
    "roe_ttm",     # ROE (TTM)
    "turnover_20", # 20-day average turnover rate
    "mom_3_1",     # 3-1 month short momentum
    "mom_6_1",     # 6-1 month medium momentum
    "ret_5d_std",  # 5-day return std (short-term vol)
    "volume_chg",  # volume change ratio (20d vs 60d)
    "high_low_20", # 20-day high-low range / close
    "close_to_high_60", # distance from 60-day high
]

ALL_FEATURES = BASE_FEATURES + EXTRA_FEATURES


# ─────────────────────── Helpers ────────────────────────────────

def _grouped_rolling(series: pd.Series, group: pd.Series, window: int,
                     min_periods: int, func: str = "mean") -> pd.Series:
    """
    Vectorized grouped rolling that avoids transform(lambda ...).
    Uses sort-based groupby().rolling() which is 5-10x faster and
    uses much less memory than per-group lambda calls.
    """
    result = getattr(
        series.groupby(group, sort=False).rolling(window, min_periods=min_periods),
        func
    )()
    # groupby().rolling() produces a MultiIndex; droplevel to align back
    result = result.droplevel(0).sort_index()
    return result


def _slice_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    require_realized_label: bool = False,
) -> pd.DataFrame:
    """
    Slice rows by signal date and optionally purge rows whose label horizon
    extends beyond the window end.
    """
    mask = (df["_date"] >= start) & (df["_date"] <= end)
    if require_realized_label and "_label_end_date" in df.columns:
        mask &= df["_label_end_date"].notna() & (df["_label_end_date"] <= end)
    return df.loc[mask]


# ─────────────────────── Data Loading ───────────────────────────

def load_raw_data(
    start: str = "2016-01-01",
    end: str = "2026-02-28",
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    Load raw daily data from SQLite.
    Only SH/SZ stocks (no BJ). Sorted by code, date.
    Downcasts floats to float32 to save ~50% memory.
    """
    print(f"[数据] 加载原始数据 {start} ~ {end} ...")
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT code, date, open, high, low, close, volume,
           pb, pe_ttm, roe_ttm, free_market_cap,
           industry_code, industry_name
    FROM kline
    WHERE (code LIKE 'SH%' OR code LIKE 'SZ%')
      AND date >= '{start}' AND date <= '{end}'
    ORDER BY code, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])

    # Downcast float columns to float32 (saves ~50% RAM)
    float_cols = df.select_dtypes(include=["float64"]).columns
    for c in float_cols:
        df[c] = df[c].astype(np.float32)

    print(f"[数据] 已加载 {len(df):,} 行, {df['code'].nunique()} 只股票 "
          f"({df.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    return df


# ─────────────────────── Feature Engineering ────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from raw daily data.
    Returns the same DataFrame with new feature columns appended.

    Memory-optimized: uses vectorized rolling instead of transform(lambda).
    Intermediate columns are deleted as soon as they are no longer needed.
    """
    print("[数据] 计算特征 ...")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    code = df["code"]  # reference, not copy
    g_daily = df.groupby("code", sort=False)

    # --- Daily return & next-trade execution fields ---
    g_close = g_daily["close"]
    df["ret_1d"] = g_close.pct_change().astype(np.float32)
    df["next_open"] = g_daily["open"].shift(-1).astype(np.float32)
    df["next_date"] = g_daily["date"].shift(-1)

    # --- Shifted close prices (reused by multiple factors) ---
    df["close_lag10"] = g_close.shift(10)
    df["close_lag20"] = g_close.shift(20)
    df["close_lag60"] = g_close.shift(60)
    df["close_lag120"] = g_close.shift(120)
    df["close_lag250"] = g_close.shift(250)

    # --- Base factors ---
    # 12-1 momentum (skip most recent month)
    df["mom_12_1"] = (df["close_lag20"] / df["close_lag250"] - 1).astype(np.float32)
    del df["close_lag250"]  # no longer needed

    # 10-day reversal
    df["rev_10"] = (df["close_lag10"] / df["close"] - 1).astype(np.float32)
    del df["close_lag10"]

    # 20-day realized volatility (vectorized rolling)
    df["rvol_20"] = _grouped_rolling(
        df["ret_1d"], code, 20, 15, "std"
    ).astype(np.float32)

    # Volume rolling means (reused by vol_confirm and volume_chg)
    df["vol_ma20"] = _grouped_rolling(
        df["volume"], code, 20, 15, "mean"
    ).astype(np.float32)
    df["vol_ma120"] = _grouped_rolling(
        df["volume"], code, 120, 80, "mean"
    ).astype(np.float32)

    # Volume confirmation (20d MA / 120d MA)
    df["vol_confirm"] = (df["vol_ma20"] / df["vol_ma120"]).astype(np.float32)

    # Value: 1/PB
    pb = df["pb"].replace(0, np.nan)
    df["inv_pb"] = (1.0 / pb).astype(np.float32)
    df.loc[df["pb"] < 0, "inv_pb"] = np.nan

    # Size: log(free_market_cap)
    df["log_cap"] = np.log(df["free_market_cap"].replace(0, np.nan)).astype(np.float32)

    # --- Extended features ---
    # Turnover rate proxy: volume / free_market_cap
    turnover_rate = (df["volume"] / df["free_market_cap"].replace(0, np.nan)).astype(np.float32)
    df["turnover_20"] = _grouped_rolling(
        turnover_rate, code, 20, 15, "mean"
    ).astype(np.float32)
    del turnover_rate

    # Short momentum: 3-1 month
    df["mom_3_1"] = (df["close_lag20"] / df["close_lag60"] - 1).astype(np.float32)
    del df["close_lag60"]

    # Medium momentum: 6-1 month
    df["mom_6_1"] = (df["close_lag20"] / df["close_lag120"] - 1).astype(np.float32)
    del df["close_lag120"]
    del df["close_lag20"]

    # 5-day return std (short-term micro-volatility)
    df["ret_5d_std"] = _grouped_rolling(
        df["ret_1d"], code, 5, 4, "std"
    ).astype(np.float32)

    # Volume change ratio (20d vs 60d)
    df["vol_ma60"] = _grouped_rolling(
        df["volume"], code, 60, 40, "mean"
    ).astype(np.float32)
    df["volume_chg"] = (df["vol_ma20"] / df["vol_ma60"]).astype(np.float32)
    del df["vol_ma60"]

    # 20-day high-low range / close
    df["high_20"] = _grouped_rolling(
        df["high"], code, 20, 15, "max"
    ).astype(np.float32)
    df["low_20"] = _grouped_rolling(
        df["low"], code, 20, 15, "min"
    ).astype(np.float32)
    df["high_low_20"] = ((df["high_20"] - df["low_20"]) / df["close"]).astype(np.float32)
    del df["high_20"], df["low_20"]

    # Distance from 60-day high
    df["high_60"] = _grouped_rolling(
        df["high"], code, 60, 40, "max"
    ).astype(np.float32)
    df["close_to_high_60"] = (df["close"] / df["high_60"] - 1).astype(np.float32)
    del df["high_60"]

    # Clean up remaining intermediates
    for tmp in ["vol_ma20", "vol_ma120", "ret_1d"]:
        if tmp in df.columns:
            del df[tmp]

    gc.collect()
    print(f"[数据] 特征计算完成。形状: {df.shape} "
          f"({df.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    return df


# ─────────────────────── Sampling ───────────────────────────────

def sample_biweekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample at biweekly frequency (mid-month + month-end).
    Uses last trading day of each half-month.

    Memory-optimized: avoids string concatenation for period key;
    uses integer period_sort directly, then reconstructs period string
    only on the much smaller sampled DataFrame.
    """
    print("[数据] 双周采样 ...")
    df = df.sort_values(["code", "date"])

    # Build integer period key (no string allocation on 10M rows)
    year = df["date"].dt.year.values
    month = df["date"].dt.month.values
    day = df["date"].dt.day.values
    half = np.where(day <= 15, 1, 2)
    # period_sort: YYYYMMH format as integer (e.g., 2024011 = 2024-01-H1)
    period_sort = (year * 100 + month) * 10 + half
    df["period_sort"] = period_sort

    # Take last row per (code, period_sort) — already sorted by date
    biweekly = df.groupby(["code", "period_sort"], sort=False).tail(1).copy()

    # Reconstruct period string on the smaller DataFrame
    ps = biweekly["period_sort"]
    h = (ps % 10).astype(str)
    m = ((ps // 10) % 100).astype(str).str.zfill(2)
    y = (ps // 1000).astype(str)
    biweekly["period"] = y + "-" + m + "-H" + h

    # Free the big DataFrame
    del df
    gc.collect()

    print(f"[数据] {len(biweekly):,} 条股票-调仓期记录, "
          f"{biweekly['period'].nunique()} periods "
          f"({biweekly.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    return biweekly


# ─────────────────────── Label Construction ─────────────────────

def build_forward_returns(
    snap: pd.DataFrame,
    horizon: str = "next_period",
) -> pd.DataFrame:
    """
    Build forward return labels using next-open execution.

    Parameters
    ----------
    snap : DataFrame with 'code', 'period', 'period_sort', 'next_open', 'next_date' columns
    horizon : 'next_period' — enter at next open after signal, exit at the
        following rebalance open

    Returns
    -------
    snap with 'fwd_ret' and 'label_end_date' columns added
    """
    print(f"[数据] 构建前瞻收益 (horizon={horizon}) ...")
    if horizon != "next_period":
        raise ValueError(f"Unsupported horizon: {horizon}")

    snap = snap.sort_values(["code", "period_sort"]).copy()
    entry_open = snap["next_open"].replace(0, np.nan)
    snap["exit_open"] = snap.groupby("code")["next_open"].shift(-1)
    snap["label_end_date"] = snap.groupby("code")["next_date"].shift(-1)
    snap["fwd_ret"] = (snap["exit_open"] / entry_open - 1).astype(np.float32)
    del snap["exit_open"]

    n_valid = snap["fwd_ret"].notna().sum()
    print(f"[数据] 前瞻收益: {n_valid:,} 有效")
    return snap


# ─────────────────────── Cross-Sectional Normalization ──────────

def cross_sectional_rank_norm(
    snap: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Rank-normalize features within each period (cross-sectional).
    Maps each feature to [0, 1] range using percentile rank.

    This is crucial for LightGBM because:
    1. Removes outlier effects without losing ordering information
    2. Makes features comparable across different time periods
    3. Robust to regime changes (feature distributions shift over time)
    """
    print("[数据] 截面排序归一化 ...")
    snap = snap.copy()
    for col in feature_cols:
        if col not in snap.columns:
            print(f"  [警告] 特征 '{col}' 不存在，跳过")
            continue
        # Rank within each period, normalized to [0, 1]
        snap[col] = snap.groupby("period")[col].rank(pct=True, na_option="keep").astype(np.float32)
    print(f"[数据] 已归一化 {len(feature_cols)} 个特征")
    return snap


# ─────────────────────── Universe Filter ────────────────────────

def filter_universe(
    snap: pd.DataFrame,
    mcap_keep_pct: float = 0.70,
    roe_floor: float = -20.0,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply standard universe filters:
    1. Drop rows with missing required features
    2. Market-cap filter (keep top X%)
    3. ROE risk filter (exclude extreme negatives)
    """
    print("[数据] 过滤股票池 ...")
    before = len(snap)

    # Drop NaN in critical columns
    required = ["close", "free_market_cap", "industry_code"]
    snap = snap.dropna(subset=required)
    print(f"  删除缺失值后: {len(snap):,}（删除 {before - len(snap):,}）")

    # Market cap filter
    cutoffs = (
        snap.groupby("period")["free_market_cap"]
        .quantile(1 - mcap_keep_pct)
        .rename("_cap_cutoff")
    )
    snap = snap.join(cutoffs, on="period")
    snap = (
        snap[snap["free_market_cap"] >= snap["_cap_cutoff"]]
        .drop(columns=["_cap_cutoff"])
        .reset_index(drop=True)
    )
    print(f"  市值过滤后 (前{mcap_keep_pct:.0%}): {len(snap):,}")

    # ROE risk filter
    if "roe_ttm" in snap.columns:
        mask = (snap["roe_ttm"].isna()) | (snap["roe_ttm"] >= roe_floor)
        dropped = (~mask).sum()
        snap = snap[mask].reset_index(drop=True)
    print(f"  ROE过滤后 (>= {roe_floor}%): {len(snap):,}（删除 {dropped:,}）")

    return snap


# ─────────────────────── Label Normalization ────────────────────

def normalize_label(snap: pd.DataFrame, label_col: str = "fwd_ret") -> pd.DataFrame:
    """
    Cross-sectional rank-normalize the label.
    Maps forward returns to [0, 1] via percentile rank within each period.
    This makes the model predict relative ranking, not absolute returns.
    """
    snap = snap.copy()
    snap["label"] = snap.groupby("period")[label_col].rank(pct=True, na_option="keep").astype(np.float32)
    return snap


# ─────────────────────── Time Split ─────────────────────────────

def time_split(
    snap: pd.DataFrame,
    train_start: str = "2019-01-01",
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2024-12-31",
    test_start: str = "2025-01-01",
    test_end: str = "2026-02-28",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological train/val/test split.
    Uses the 'date' column for splitting and purges train/val rows whose
    labels are not fully realized within each window.

    Returns (train, val, test) DataFrames.
    """
    snap = snap.copy()
    snap["_date"] = pd.to_datetime(snap["date"])
    temp_cols = ["_date"]
    if "label_end_date" in snap.columns:
        snap["_label_end_date"] = pd.to_datetime(snap["label_end_date"])
        temp_cols.append("_label_end_date")

    train = _slice_window(
        snap,
        pd.Timestamp(train_start),
        pd.Timestamp(train_end),
        require_realized_label=True,
    ).drop(columns=temp_cols, errors="ignore")
    val = _slice_window(
        snap,
        pd.Timestamp(val_start),
        pd.Timestamp(val_end),
        require_realized_label=True,
    ).drop(columns=temp_cols, errors="ignore")
    test = _slice_window(
        snap,
        pd.Timestamp(test_start),
        pd.Timestamp(test_end),
        require_realized_label=False,
    ).drop(columns=temp_cols, errors="ignore")

    print(f"[数据] 切分: 训练={len(train):,} 验证={len(val):,} 测试={len(test):,}")
    return train, val, test


def rolling_time_split(
    snap: pd.DataFrame,
    train_years: int = 3,
    val_months: int = 6,
    step_months: int = 6,
    min_date: str = "2019-01-01",
    max_date: str = "2026-02-28",
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Generate rolling train/val/test windows.

    Each window:
    - train: [T - train_years, T)
    - val  : [T, T + val_months)
    - test : [T + val_months, T + val_months + step_months)

    Train/validation windows are purged so labels never extend beyond the
    corresponding window end date.

    Returns list of (train_df, val_df, test_df) tuples.
    """
    snap = snap.copy()
    snap["_date"] = pd.to_datetime(snap["date"])
    temp_cols = ["_date"]
    if "label_end_date" in snap.columns:
        snap["_label_end_date"] = pd.to_datetime(snap["label_end_date"])
        temp_cols.append("_label_end_date")

    min_dt = pd.Timestamp(min_date)
    max_dt = pd.Timestamp(max_date)

    # First possible train start: enough for train_years
    first_train_end = min_dt + pd.DateOffset(years=train_years)

    windows = []
    t = first_train_end
    while t + pd.DateOffset(months=val_months) <= max_dt:
        train_start = t - pd.DateOffset(years=train_years)
        train_end = t - pd.DateOffset(days=1)
        val_start = t
        val_end = t + pd.DateOffset(months=val_months) - pd.DateOffset(days=1)
        test_start = t + pd.DateOffset(months=val_months)
        test_end = min(
            t + pd.DateOffset(months=val_months + step_months) - pd.DateOffset(days=1),
            max_dt,
        )

        tr = _slice_window(snap, train_start, train_end, require_realized_label=True)
        va = _slice_window(snap, val_start, val_end, require_realized_label=True)
        te = _slice_window(snap, test_start, test_end, require_realized_label=False)

        if len(tr) > 0 and len(te) > 0:
            windows.append((
                tr.drop(columns=temp_cols, errors="ignore"),
                va.drop(columns=temp_cols, errors="ignore"),
                te.drop(columns=temp_cols, errors="ignore"),
            ))
            print(f"  窗口 {len(windows)}: 训练 {train_start.date()}~{train_end.date()} "
                  f"({len(tr):,}), 验证 ({len(va):,}), 测试 {test_start.date()}~{test_end.date()} ({len(te):,})")

        t += pd.DateOffset(months=step_months)

    print(f"[数据] 已生成 {len(windows)} 个滚动窗口")
    return windows


# ─────────────────────── High-Level API ─────────────────────────

def build_ml_dataset(
    warm_up_start: str = "2016-01-01",
    backtest_end: str = "2026-02-28",
    feature_cols: Optional[List[str]] = None,
    mcap_keep_pct: float = 0.70,
    rank_normalize: bool = True,
) -> pd.DataFrame:
    """
    End-to-end dataset builder.

    Steps:
    1. Load raw data (with warm-up period for factor computation)
    2. Compute all features
    3. Sample biweekly
    4. Filter universe
    5. Build forward returns (next-period)
    6. (Optional) Cross-sectional rank normalization
    7. Normalize label

    Returns a DataFrame ready for model training/prediction.
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    # 1. Load (keep a small buffer after backtest_end for next-open execution)
    load_end = (pd.Timestamp(backtest_end) + pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    df = load_raw_data(start=warm_up_start, end=load_end)

    # 2. Features
    df = compute_features(df)

    # 3. Sample biweekly (this also deletes the big daily df internally)
    snap = sample_biweekly(df)
    snap = snap[snap["date"] <= pd.Timestamp(backtest_end)].copy()
    del df  # free memory
    gc.collect()

    # 4. Filter universe
    snap = filter_universe(snap, mcap_keep_pct=mcap_keep_pct)

    # 5. Forward returns
    snap = build_forward_returns(snap)

    # 6. Rank normalize features
    if rank_normalize:
        snap = cross_sectional_rank_norm(snap, feature_cols)

    # 7. Normalize label
    snap = normalize_label(snap)

    # Clean up: drop intermediate columns
    keep_cols = (
        ["code", "date", "period", "period_sort",
         "close", "next_open", "next_date", "label_end_date",
         "free_market_cap", "industry_code", "industry_name"]
        + feature_cols
        + ["fwd_ret", "label"]
    )
    keep_cols = [c for c in keep_cols if c in snap.columns]
    snap = snap[keep_cols].copy()
    gc.collect()

    # Encode industry as integer
    snap["industry_id"] = snap["industry_code"].astype("category").cat.codes

    n_periods = snap["period"].nunique()
    n_stocks = snap["code"].nunique()
    n_valid = snap["label"].notna().sum()
    print(f"\n[数据] 最终数据集: {len(snap):,} 行, "
          f"{n_stocks} stocks, {n_periods} periods, {n_valid:,} labeled "
          f"({snap.memory_usage(deep=True).sum() / 1e6:.0f} MB)")
    return snap


# ─────────────────────── CLI Test ───────────────────────────────

if __name__ == "__main__":
    dataset = build_ml_dataset()
    print("\n数据集预览:")
    print(dataset.head(10).to_string())
    print(f"\n特征缺失率:")
    for col in ALL_FEATURES:
        if col in dataset.columns:
            nan_rate = dataset[col].isna().mean()
            print(f"  {col}: {nan_rate:.1%}")
    print(f"\n标签缺失率: {dataset['label'].isna().mean():.1%}")
