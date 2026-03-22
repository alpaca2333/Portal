"""
Data loading, factor computation, sampling, and universe filtering.
"""
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
from engine.types import StrategyConfig, RebalanceFreq


def load_stock_data(cfg: StrategyConfig) -> pd.DataFrame:
    """Load raw daily data from SQLite."""
    print(f"[data] Loading from {cfg.db_path} ...")
    conn = sqlite3.connect(cfg.db_path)

    # Build column list — always need these base columns
    base_cols = [
        "code", "date", "close", "volume", "pb", "pe_ttm",
        "free_market_cap", "industry_code", "industry_name",
    ]
    # Add any extra columns the strategy requested (e.g. roe_ttm)
    all_cols = list(dict.fromkeys(base_cols + cfg.extra_columns))
    col_str = ", ".join(all_cols)

    query = f"""
    SELECT {col_str}
    FROM kline
    WHERE (code LIKE 'SH%' OR code LIKE 'SZ%')
      AND date >= '{cfg.warm_up_start}' AND date <= '{cfg.end}'
    ORDER BY code, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    print(f"[data] Loaded {len(df):,} rows, {df['code'].nunique()} stocks")
    return df


def compute_daily_factors(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Compute standard daily rolling factors.

    If cfg.compute_factors_fn is set, delegates to that instead.
    """
    if cfg.compute_factors_fn is not None:
        print("[factors] Using custom factor computation ...")
        return cfg.compute_factors_fn(df, cfg)

    print("[factors] Computing daily rolling factors ...")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")

    # Daily return
    df["ret_1d"] = g["close"].pct_change()

    # 12-1 momentum (skip most recent month)
    df["close_lag20"] = g["close"].shift(20)
    df["close_lag250"] = g["close"].shift(250)
    df["mom_12_1"] = df["close_lag20"] / df["close_lag250"] - 1

    # 10-day reversal
    df["close_lag10"] = g["close"].shift(10)
    df["rev_10"] = df["close_lag10"] / df["close"] - 1

    # 20-day realized volatility
    df["rvol_20"] = g["ret_1d"].transform(
        lambda x: x.rolling(20, min_periods=15).std())

    # Volume confirmation (20d MA / 120d MA)
    df["vol_ma20"] = g["volume"].transform(
        lambda x: x.rolling(20, min_periods=15).mean())
    df["vol_ma120"] = g["volume"].transform(
        lambda x: x.rolling(120, min_periods=80).mean())
    df["vol_confirm"] = df["vol_ma20"] / df["vol_ma120"]

    # Value: 1/PB
    df["inv_pb"] = 1.0 / df["pb"].replace(0, np.nan)
    df.loc[df["pb"] < 0, "inv_pb"] = np.nan

    # Size: log(free_market_cap)
    df["log_cap"] = np.log(df["free_market_cap"].replace(0, np.nan))

    print(f"[factors] Done. Shape: {df.shape}")
    return df


def sample_biweekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample at biweekly frequency: mid-month (~15th) and month-end.
    Creates 'period' and 'period_sort' columns.
    """
    print("[sample] Biweekly sampling ...")
    df = df.sort_values(["code", "date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["half"] = np.where(df["day"] <= 15, 1, 2)
    df["period"] = (
        df["year"].astype(str) + "-"
        + df["month"].astype(str).str.zfill(2) + "-H"
        + df["half"].astype(str)
    )
    biweekly = df.sort_values("date").groupby(
        ["code", "period"], as_index=False
    ).last()
    biweekly["period_sort"] = (
        biweekly["year"] * 100 + biweekly["month"]
    ) * 10 + biweekly["half"]
    print(f"[sample] {len(biweekly):,} stock-period obs, "
          f"{biweekly['period'].nunique()} periods")
    return biweekly


def sample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample at monthly frequency.
    Creates 'period' (YYYY-MM) and 'period_sort' columns.
    """
    print("[sample] Monthly sampling ...")
    df = df.sort_values(["code", "date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["period"] = (
        df["year"].astype(str) + "-"
        + df["month"].astype(str).str.zfill(2)
    )
    monthly = df.sort_values("date").groupby(
        ["code", "period"], as_index=False
    ).last()
    monthly["period_sort"] = monthly["year"] * 100 + monthly["month"]
    print(f"[sample] {len(monthly):,} stock-period obs, "
          f"{monthly['period'].nunique()} periods")
    return monthly


def sample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample at weekly frequency: last trading day of each ISO week.
    Creates 'period' (YYYY-WXX) and 'period_sort' columns.
    """
    print("[sample] Weekly sampling ...")
    df = df.sort_values(["code", "date"]).copy()
    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["period"] = (
        df["year"].astype(str) + "-W"
        + df["week"].astype(str).str.zfill(2)
    )
    weekly = df.sort_values("date").groupby(
        ["code", "period"], as_index=False
    ).last()
    weekly["period_sort"] = weekly["year"] * 100 + weekly["week"]
    print(f"[sample] {len(weekly):,} stock-period obs, "
          f"{weekly['period'].nunique()} periods")
    return weekly


def sample(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Dispatch to the correct sampler based on config."""
    if cfg.freq == RebalanceFreq.WEEKLY:
        return sample_weekly(df)
    elif cfg.freq == RebalanceFreq.BIWEEKLY:
        return sample_biweekly(df)
    else:
        return sample_monthly(df)


def filter_universe(snap: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Apply standard universe filters:
    1. Drop rows missing required factor columns
    2. Call strategy pre_filter hook (e.g. ROE risk filter)
    3. Market-cap filter (keep top X%)
    """
    print("[filter] Applying universe filters ...")

    # Required columns: all factor columns that exist + basic fields
    required = [
        "close", "mom_12_1", "inv_pb", "rvol_20", "vol_confirm",
        "industry_code", "free_market_cap", "log_cap",
    ]
    # Only require columns that actually exist in the DataFrame
    required = [c for c in required if c in snap.columns]

    before = len(snap)
    snap = snap.dropna(subset=required)
    print(f"[filter] After dropna: {len(snap):,} (dropped {before - len(snap):,})")

    # Strategy-specific pre-filter hook (e.g. ROE risk filter)
    if cfg.pre_filter is not None:
        before_hook = len(snap)
        snap = cfg.pre_filter(snap, cfg)
        dropped = before_hook - len(snap)
        print(f"[filter] After pre_filter hook: {len(snap):,} (dropped {dropped:,})")

    # Market-cap filter: keep top X%
    cutoffs = (
        snap.groupby("period")["free_market_cap"]
        .quantile(1 - cfg.mcap_keep_pct)
        .rename("_cap_cutoff")
    )
    snap = snap.join(cutoffs, on="period")
    snap = (
        snap[snap["free_market_cap"] >= snap["_cap_cutoff"]]
        .drop(columns=["_cap_cutoff"])
        .reset_index(drop=True)
    )
    print(f"[filter] After cap filter (top {cfg.mcap_keep_pct:.0%}): {len(snap):,}")
    return snap
