"""
LightGBM Cross-Sectional Stock Selection Strategy (V4)
======================================================

Motivation
----------
Linear multi-factor models (v2) use fixed weights and cannot capture
non-linear interactions between factors (e.g. momentum behaves
differently in high-vol vs. low-vol regimes).  LightGBM automatically
learns these conditional effects from data.

Core Design Choices
-------------------
1. **Cross-sectional rank label**: predict relative ranking rather than
   absolute returns, eliminating non-stationarity of return distributions.
2. **3-year rolling training window**: train/val sets are purged by label
   end date to avoid cross-window look-ahead.
3. **Industry-constrained selection**: ML score top 5%, max 5 per industry
   (consistent with v2 for fair comparison).
4. **22 features**: 15 base factors (v3) + 7 new factors covering
   dividend yield, quality (ROA/gross margin/leverage), growth
   (revenue/profit YoY), and liquidity (Amihud ILLIQ).
5. **Feature rank normalization**: all features mapped to [0,1] percentile
   within each cross-section, ensuring cross-period comparability.
6. **Reduced model complexity**: num_leaves=31, max_depth=4, stronger
   regularization to prevent overfitting.
7. **Wider market cap coverage**: top 85% by circ_mv (was 70%).

Usage
-----
cd <project_root>
python -m data.quant.strategies.lgbm_cross_sectional
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError(
        "LightGBM is required. Install with: pip install lightgbm"
    )

from engine import BacktestConfig, StrategyBase, run_backtest
from engine.data_loader import DataAccessor

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ===================================================================
# Feature engineering
# ===================================================================

# All DB columns needed by the 15 features
FEATURE_COLUMNS = [
    "ts_code", "trade_date",
    "open", "high", "low", "close", "pre_close",
    "pct_chg", "vol", "amount",
    "turnover_rate_f",
    "pb", "pe_ttm", "circ_mv",
    "roe", "roa", "grossprofit_margin", "debt_to_assets",
    "dv_ttm",
    "tr_yoy", "op_yoy",
    "sw_l1", "is_suspended",
]

# Feature names in output order (must match computation)
FEATURE_NAMES = [
    "mom_12_1",          # 12-month momentum, skip most recent month
    "rev_10",            # 10-day reversal
    "rvol_20",           # 20-day realized volatility
    "vol_confirm",       # volume-price correlation (20d)
    "inv_pb",            # 1/PB (value)
    "log_cap",           # log(circulating market value)
    "pe_ttm",            # PE_TTM
    "roe_ttm",           # ROE
    "turnover_20",       # 20-day average turnover rate
    "mom_3_1",           # 3-month momentum, skip 1 month
    "mom_6_1",           # 6-month momentum, skip 1 month
    "ret_5d_std",        # 5-day return standard deviation
    "volume_chg",        # volume change ratio (20d vs 60d)
    "high_low_20",       # 20-day high-low range / close
    "close_to_high_60",  # close / 60-day high
    # ── New factors (V4 optimization) ──
    "dv_ttm",            # dividend yield TTM
    "roa_ttm",           # return on assets
    "gross_margin",      # gross profit margin
    "low_leverage",      # -debt_to_assets (lower is better)
    "growth_revenue",    # revenue YoY growth
    "growth_profit",     # operating profit YoY growth
    "illiq_20",          # Amihud illiquidity (20d)
]


def _compute_momentum(close_pivot: pd.DataFrame, start_offset: int,
                       end_offset: int) -> pd.Series:
    """
    Compute momentum: close[end_offset] / close[start_offset] - 1.
    Offsets are in number of rows from the end (0 = last row).
    Positive start_offset means further back in time.
    """
    n = len(close_pivot)
    if n <= start_offset:
        return pd.Series(dtype=float)
    end_idx = n - 1 - end_offset if end_offset > 0 else n - 1
    start_idx = n - 1 - start_offset
    if start_idx < 0 or end_idx < 0 or start_idx >= n or end_idx >= n:
        return pd.Series(dtype=float)
    end_prices = close_pivot.iloc[end_idx]
    start_prices = close_pivot.iloc[start_idx]
    mom = end_prices / start_prices - 1
    return mom.replace([np.inf, -np.inf], np.nan)


# -------------------------------------------------------------------
# Bulk data prefetch — load entire date range in ONE SQL query
# -------------------------------------------------------------------

def prefetch_bulk_data(
    accessor: DataAccessor,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load all stock_daily data for [start_date, end_date] in a single SQL
    query.  Returns a DataFrame with all FEATURE_COLUMNS, sorted by
    (trade_date, ts_code).  This is the key to avoiding per-date DB hits.
    """
    s = start_date.strftime("%Y%m%d")
    e = end_date.strftime("%Y%m%d")
    col_sql = ", ".join(FEATURE_COLUMNS)
    sql = f"""
        SELECT {col_sql} FROM stock_daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date, ts_code
    """
    print(f"      [预取] 执行批量 SQL 加载 {s} ~ {e} ...")
    df = pd.read_sql_query(sql, accessor.conn, params=(s, e))
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    print(f"      [预取] ✓ 加载完成: {len(df):,} 行, "
          f"{df['ts_code'].nunique()} 只股票, "
          f"{df['trade_date'].nunique()} 个交易日")
    return df


# Pre-computed date index for bulk_data to avoid repeated full-table scans
_bulk_date_index_cache: Dict[int, Tuple[np.ndarray, Dict]] = {}


def _get_bulk_date_index(bulk_data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """
    Build or retrieve a cached date index for bulk_data.
    Returns (sorted_unique_dates_as_Timestamps, {pd.Timestamp: row_indices}).

    IMPORTANT: All dict keys are pd.Timestamp to ensure hash-compatible
    lookups (np.datetime64 and pd.Timestamp have different hashes!).
    """
    cache_key = id(bulk_data)
    if cache_key in _bulk_date_index_cache:
        return _bulk_date_index_cache[cache_key]

    dates = bulk_data["trade_date"].values
    unique_dates_raw = np.sort(np.unique(dates))
    # Convert to pd.Timestamp array for consistent hashing
    unique_dates = pd.DatetimeIndex(unique_dates_raw)
    # Build a dict mapping each pd.Timestamp to its row indices
    date_to_rows = {}
    for d_ts in unique_dates:
        # Use boolean mask on the original numpy array for speed
        date_to_rows[d_ts] = np.where(dates == d_ts)[0]

    _bulk_date_index_cache[cache_key] = (unique_dates, date_to_rows)
    return unique_dates, date_to_rows


def compute_features_from_memory(
    date: pd.Timestamp,
    bulk_data: pd.DataFrame,
    lookback: int = 260,
    st_codes: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute all 15 features using in-memory bulk_data instead of DB queries.
    This is the memory-based equivalent of compute_features_for_date().

    Uses a cached date index to avoid repeated full-table scans.

    Parameters
    ----------
    date : pd.Timestamp
        The date to compute features for.
    bulk_data : pd.DataFrame
        Pre-loaded DataFrame containing all needed historical data.
    lookback : int
        Number of trading days to look back.

    Returns
    -------
    pd.DataFrame or None
    """
    # Use cached date index for O(1) date lookups
    all_dates, date_to_rows = _get_bulk_date_index(bulk_data)

    # Normalize date to pd.Timestamp (must match dict key type)
    date_ts = pd.Timestamp(date)

    # Extract window dates: all trade dates <= date, take last `lookback` dates
    valid_dates = all_dates[all_dates <= date_ts]
    if len(valid_dates) < 60:
        return None
    window_dates = valid_dates[-lookback:] if len(valid_dates) >= lookback else valid_dates

    # Gather rows for all window dates using the index (much faster than isin)
    row_indices = []
    for wd in window_dates:
        if wd in date_to_rows:
            row_indices.append(date_to_rows[wd])
    if not row_indices:
        return None
    all_row_idx = np.concatenate(row_indices)
    window = bulk_data.iloc[all_row_idx]

    # Get the snapshot for the target date
    if date_ts in date_to_rows:
        snap = bulk_data.iloc[date_to_rows[date_ts]].copy()
    else:
        return None
    if snap.empty:
        return None

    snap = snap[
        ["ts_code", "close", "pb", "pe_ttm", "circ_mv",
         "roe", "roa", "grossprofit_margin", "debt_to_assets",
         "dv_ttm", "tr_yoy", "op_yoy",
         "sw_l1", "is_suspended", "turnover_rate_f"]
    ].copy()

    # Filter universe: not suspended, valid close, SH/SZ main board, not ST
    snap = snap[
        (snap["is_suspended"] != 1)
        & (snap["close"].notna())
        & (snap["close"] > 0)
    ].copy()

    snap = snap[
        snap["ts_code"].str.match(r"^(6\d{5}\.SH|00[013]\d{3}\.SZ)$")
    ].copy()

    # Filter out ST / *ST stocks (based on stock_info.name)
    if st_codes:
        n_before = len(snap)
        snap = snap[~snap["ts_code"].isin(st_codes)].copy()
        n_removed = n_before - len(snap)
        if n_removed > 0:
            pass  # silently remove ST stocks

    if len(snap) < 100:
        return None

    universe_codes = set(snap["ts_code"].tolist())

    # Build ALL pivots from window data at once (only universe stocks)
    # This avoids repeated pivot_table calls on the same large DataFrame
    w = window[window["ts_code"].isin(universe_codes)].copy()
    w.sort_values(["trade_date", "ts_code"], inplace=True)

    # Build all needed pivots in one pass using a single set_index + unstack
    w_indexed = w.set_index(["trade_date", "ts_code"])
    pivot_cols = ["close", "vol", "high", "low", "turnover_rate_f"]
    pivots = {}
    for col in pivot_cols:
        if col in w_indexed.columns:
            pivots[col] = w_indexed[col].unstack("ts_code").sort_index()

    close_pivot = pivots["close"]
    vol_pivot = pivots.get("vol")
    high_pivot = pivots.get("high")
    low_pivot = pivots.get("low")
    turn_pivot = pivots.get("turnover_rate_f")

    n_dates = len(close_pivot)
    features = {}
    valid_codes = close_pivot.columns.tolist()

    # --- 1. mom_12_1 ---
    if n_dates >= 240:
        features["mom_12_1"] = _compute_momentum(close_pivot, 240, 20)
    else:
        features["mom_12_1"] = pd.Series(np.nan, index=valid_codes)

    # --- 2. rev_10 ---
    if n_dates >= 10:
        features["rev_10"] = _compute_momentum(close_pivot, 10, 0)
    else:
        features["rev_10"] = pd.Series(np.nan, index=valid_codes)

    # --- 3. rvol_20 ---
    if n_dates >= 21:
        daily_ret = close_pivot.iloc[-21:].pct_change(fill_method=None).iloc[1:]
        features["rvol_20"] = daily_ret.std()
    else:
        features["rvol_20"] = pd.Series(np.nan, index=valid_codes)

    # --- 4. vol_confirm ---
    if n_dates >= 20 and vol_pivot is not None:
        ret_20 = close_pivot.iloc[-20:].pct_change(fill_method=None).iloc[1:]
        vp_20 = vol_pivot.iloc[-20:]
        common_dates = ret_20.index.intersection(vp_20.index)
        if len(common_dates) >= 10:
            r = ret_20.loc[common_dates]
            v = vp_20.loc[common_dates]
            # Vectorized correlation via numpy
            r_arr = r.values
            v_arr = v.values
            mask = ~(np.isnan(r_arr) | np.isnan(v_arr))
            corr_vals = {}
            for j, code in enumerate(r.columns):
                m = mask[:, j]
                if m.sum() >= 10:
                    rr = r_arr[m, j]
                    vv = v_arr[m, j]
                    rr_dm = rr - rr.mean()
                    vv_dm = vv - vv.mean()
                    denom = np.sqrt((rr_dm ** 2).sum() * (vv_dm ** 2).sum())
                    corr_vals[code] = (rr_dm * vv_dm).sum() / denom if denom > 0 else np.nan
            features["vol_confirm"] = pd.Series(corr_vals)
        else:
            features["vol_confirm"] = pd.Series(np.nan, index=valid_codes)
    else:
        features["vol_confirm"] = pd.Series(np.nan, index=valid_codes)

    # --- 5. inv_pb ---
    snap_indexed = snap.set_index("ts_code")
    pb = snap_indexed["pb"]
    features["inv_pb"] = pd.Series(
        np.where(pb > 0, 1.0 / pb, np.nan), index=snap_indexed.index
    )

    # --- 6. log_cap ---
    mv = snap_indexed["circ_mv"]
    features["log_cap"] = pd.Series(
        np.where(mv > 0, np.log(mv), np.nan), index=snap_indexed.index
    )

    # --- 7. pe_ttm ---
    features["pe_ttm"] = snap_indexed["pe_ttm"]

    # --- 8. roe_ttm ---
    features["roe_ttm"] = snap_indexed["roe"]

    # --- 9. turnover_20 ---
    if n_dates >= 20 and turn_pivot is not None:
        features["turnover_20"] = turn_pivot.iloc[-20:].mean()
    else:
        features["turnover_20"] = pd.Series(np.nan, index=valid_codes)

    # --- 10. mom_3_1 ---
    if n_dates >= 60:
        features["mom_3_1"] = _compute_momentum(close_pivot, 60, 20)
    else:
        features["mom_3_1"] = pd.Series(np.nan, index=valid_codes)

    # --- 11. mom_6_1 ---
    if n_dates >= 120:
        features["mom_6_1"] = _compute_momentum(close_pivot, 120, 20)
    else:
        features["mom_6_1"] = pd.Series(np.nan, index=valid_codes)

    # --- 12. ret_5d_std ---
    if n_dates >= 6:
        daily_ret_5 = close_pivot.iloc[-6:].pct_change(fill_method=None).iloc[1:]
        features["ret_5d_std"] = daily_ret_5.std()
    else:
        features["ret_5d_std"] = pd.Series(np.nan, index=valid_codes)

    # --- 13. volume_chg ---
    if n_dates >= 60 and vol_pivot is not None:
        vol_20 = vol_pivot.iloc[-20:].mean()
        vol_60 = vol_pivot.iloc[-60:].mean()
        features["volume_chg"] = (vol_20 / vol_60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["volume_chg"] = pd.Series(np.nan, index=valid_codes)

    # --- 14. high_low_20 ---
    if n_dates >= 20 and high_pivot is not None and low_pivot is not None:
        h20 = high_pivot.iloc[-20:].max()
        l20 = low_pivot.iloc[-20:].min()
        last_close = close_pivot.iloc[-1]
        features["high_low_20"] = ((h20 - l20) / last_close.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["high_low_20"] = pd.Series(np.nan, index=valid_codes)

    # --- 15. close_to_high_60 ---
    if n_dates >= 60 and high_pivot is not None:
        h60 = high_pivot.iloc[-60:].max()
        last_close = close_pivot.iloc[-1]
        features["close_to_high_60"] = (last_close / h60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["close_to_high_60"] = pd.Series(np.nan, index=valid_codes)

    # --- 16. dv_ttm: dividend yield TTM ---
    features["dv_ttm"] = snap_indexed["dv_ttm"]

    # --- 17. roa_ttm: return on assets ---
    features["roa_ttm"] = snap_indexed["roa"]

    # --- 18. gross_margin: gross profit margin ---
    features["gross_margin"] = snap_indexed["grossprofit_margin"]

    # --- 19. low_leverage: negative debt-to-assets (lower leverage is better) ---
    dta = snap_indexed["debt_to_assets"]
    features["low_leverage"] = pd.Series(
        np.where(dta.notna(), -dta, np.nan), index=snap_indexed.index
    )

    # --- 20. growth_revenue: revenue YoY growth ---
    features["growth_revenue"] = snap_indexed["tr_yoy"]

    # --- 21. growth_profit: operating profit YoY growth ---
    features["growth_profit"] = snap_indexed["op_yoy"]

    # --- 22. illiq_20: Amihud illiquidity (20d) ---
    if n_dates >= 20:
        # Use window data for pct_chg and amount
        w_recent = window[window["trade_date"].isin(window_dates[-20:])].copy()
        if "pct_chg" in w_recent.columns and "amount" in w_recent.columns:
            pctchg_pivot = w_recent.pivot_table(
                index="trade_date", columns="ts_code", values="pct_chg"
            ).sort_index()
            amount_pivot = w_recent.pivot_table(
                index="trade_date", columns="ts_code", values="amount"
            ).sort_index()
            # Amihud ILLIQ = mean(|ret| / amount)
            abs_ret = pctchg_pivot.abs()
            amt_safe = amount_pivot.replace(0, np.nan)
            illiq = (abs_ret / amt_safe).mean()
            features["illiq_20"] = illiq.replace([np.inf, -np.inf], np.nan)
        else:
            features["illiq_20"] = pd.Series(np.nan, index=valid_codes)
    else:
        features["illiq_20"] = pd.Series(np.nan, index=valid_codes)

    # Combine all features
    feat_df = pd.DataFrame(features)
    feat_df.index.name = "ts_code"
    feat_df = feat_df.reset_index()

    # Attach sw_l1 and circ_mv for later filtering
    meta = snap[["ts_code", "sw_l1", "circ_mv"]].copy()
    feat_df = feat_df.merge(meta, on="ts_code", how="inner")

    # Drop rows with too many NaN features (allow up to 5 missing)
    feat_df = feat_df.dropna(subset=FEATURE_NAMES, thresh=len(FEATURE_NAMES) - 5)

    return feat_df if len(feat_df) >= 50 else None


def compute_forward_return_from_memory(
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    bulk_data: pd.DataFrame,
) -> Optional[pd.Series]:
    """
    Compute forward return from in-memory bulk_data instead of DB queries.
    Uses cached date index for O(1) lookups.
    """
    _, date_to_rows = _get_bulk_date_index(bulk_data)

    # Normalize to pd.Timestamp (must match dict key type)
    dt_now = pd.Timestamp(date)
    dt_next = pd.Timestamp(next_date)

    if dt_now not in date_to_rows or dt_next not in date_to_rows:
        return None

    snap_now = bulk_data.iloc[date_to_rows[dt_now]]
    snap_next = bulk_data.iloc[date_to_rows[dt_next]]

    if snap_now.empty or snap_next.empty:
        return None

    p0 = snap_now[["ts_code", "close"]].dropna(subset=["close"])
    p0 = p0[p0["close"] > 0].set_index("ts_code")["close"]

    p1 = snap_next[["ts_code", "close"]].dropna(subset=["close"])
    p1 = p1[p1["close"] > 0].set_index("ts_code")["close"]

    common = p0.index.intersection(p1.index)
    if len(common) < 50:
        return None

    ret = p1.loc[common] / p0.loc[common] - 1
    ret.index.name = "ts_code"
    return ret


def compute_features_for_date(
    date: pd.Timestamp,
    accessor: DataAccessor,
    lookback: int = 260,
    st_codes: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute all 15 features for a single date (DB-based fallback).
    Prefer compute_features_from_memory() when bulk data is available.

    Parameters
    ----------
    date : pd.Timestamp
        The date to compute features for.
    accessor : DataAccessor
        Data accessor for DB queries.
    lookback : int
        Number of trading days to look back (260 ≈ 1 year).

    Returns
    -------
    pd.DataFrame with columns [ts_code, sw_l1, circ_mv] + FEATURE_NAMES,
    or None if insufficient data.
    """
    # Get the window of historical data
    window = accessor.get_window(
        date, lookback=lookback,
        columns=FEATURE_COLUMNS,
    )
    if window.empty or len(window["trade_date"].unique()) < 60:
        return None

    # Get the snapshot for the current date (for cross-sectional fields)
    snap = accessor.get_date(
        date,
        columns=["ts_code", "close", "pb", "pe_ttm", "circ_mv",
                  "roe", "roa", "grossprofit_margin", "debt_to_assets",
                  "dv_ttm", "tr_yoy", "op_yoy",
                  "sw_l1", "is_suspended", "turnover_rate_f"],
    )
    if snap.empty:
        return None

    # Filter universe: not suspended, valid close, SH/SZ main board, not ST
    snap = snap[
        (snap["is_suspended"] != 1)
        & (snap["close"].notna())
        & (snap["close"] > 0)
    ].copy()

    # Filter to SH/SZ main board only (60xxxx.SH, 000xxx.SZ, 001xxx.SZ, 003xxx.SZ)
    snap = snap[
        snap["ts_code"].str.match(r"^(6\d{5}\.SH|00[013]\d{3}\.SZ)$")
    ].copy()

    # Filter out ST / *ST stocks (based on stock_info.name)
    if st_codes:
        snap = snap[~snap["ts_code"].isin(st_codes)].copy()

    if len(snap) < 100:
        return None

    universe_codes = set(snap["ts_code"].tolist())

    # Build pivots from window data
    w = window[window["ts_code"].isin(universe_codes)].copy()
    w.sort_values(["trade_date", "ts_code"], inplace=True)

    close_pivot = w.pivot(index="trade_date", columns="ts_code", values="close")
    close_pivot.sort_index(inplace=True)
    n_dates = len(close_pivot)

    features = {}
    valid_codes = close_pivot.columns.tolist()

    # --- 1. mom_12_1: 12-month momentum, skip most recent month (~20 days) ---
    # approximately 252 trading days back, skip last ~20
    if n_dates >= 240:
        features["mom_12_1"] = _compute_momentum(close_pivot, 240, 20)
    else:
        features["mom_12_1"] = pd.Series(np.nan, index=valid_codes)

    # --- 2. rev_10: 10-day reversal (negative of 10-day return) ---
    if n_dates >= 10:
        features["rev_10"] = _compute_momentum(close_pivot, 10, 0)
    else:
        features["rev_10"] = pd.Series(np.nan, index=valid_codes)

    # --- 3. rvol_20: 20-day realized volatility ---
    if n_dates >= 21:
        daily_ret = close_pivot.iloc[-21:].pct_change(fill_method=None).iloc[1:]
        features["rvol_20"] = daily_ret.std()
    else:
        features["rvol_20"] = pd.Series(np.nan, index=valid_codes)

    # --- 4. vol_confirm: volume-price correlation (20d) ---
    if n_dates >= 20:
        ret_20 = close_pivot.iloc[-20:].pct_change(fill_method=None).iloc[1:]
        vol_pivot = w.pivot(
            index="trade_date", columns="ts_code", values="vol"
        ).sort_index().iloc[-20:]
        # Align indices
        common_dates = ret_20.index.intersection(vol_pivot.index)
        if len(common_dates) >= 10:
            r = ret_20.loc[common_dates]
            v = vol_pivot.loc[common_dates]
            corr_vals = {}
            for code in valid_codes:
                if code in r.columns and code in v.columns:
                    rc = r[code].dropna()
                    vc = v[code].dropna()
                    common = rc.index.intersection(vc.index)
                    if len(common) >= 10:
                        corr_vals[code] = rc.loc[common].corr(vc.loc[common])
            features["vol_confirm"] = pd.Series(corr_vals)
        else:
            features["vol_confirm"] = pd.Series(np.nan, index=valid_codes)
    else:
        features["vol_confirm"] = pd.Series(np.nan, index=valid_codes)

    # --- 5. inv_pb: 1/PB ---
    snap_indexed = snap.set_index("ts_code")
    pb = snap_indexed["pb"]
    features["inv_pb"] = pd.Series(
        np.where(pb > 0, 1.0 / pb, np.nan), index=snap_indexed.index
    )

    # --- 6. log_cap: log(circulating market value) ---
    mv = snap_indexed["circ_mv"]
    features["log_cap"] = pd.Series(
        np.where(mv > 0, np.log(mv), np.nan), index=snap_indexed.index
    )

    # --- 7. pe_ttm ---
    features["pe_ttm"] = snap_indexed["pe_ttm"]

    # --- 8. roe_ttm ---
    features["roe_ttm"] = snap_indexed["roe"]

    # --- 9. turnover_20: 20-day average turnover rate ---
    if n_dates >= 20:
        turn_pivot = w.pivot(
            index="trade_date", columns="ts_code", values="turnover_rate_f"
        ).sort_index().iloc[-20:]
        features["turnover_20"] = turn_pivot.mean()
    else:
        features["turnover_20"] = pd.Series(np.nan, index=valid_codes)

    # --- 10. mom_3_1: 3-month momentum, skip 1 month ---
    if n_dates >= 60:
        features["mom_3_1"] = _compute_momentum(close_pivot, 60, 20)
    else:
        features["mom_3_1"] = pd.Series(np.nan, index=valid_codes)

    # --- 11. mom_6_1: 6-month momentum, skip 1 month ---
    if n_dates >= 120:
        features["mom_6_1"] = _compute_momentum(close_pivot, 120, 20)
    else:
        features["mom_6_1"] = pd.Series(np.nan, index=valid_codes)

    # --- 12. ret_5d_std: 5-day return standard deviation ---
    if n_dates >= 6:
        daily_ret_5 = close_pivot.iloc[-6:].pct_change(fill_method=None).iloc[1:]
        features["ret_5d_std"] = daily_ret_5.std()
    else:
        features["ret_5d_std"] = pd.Series(np.nan, index=valid_codes)

    # --- 13. volume_chg: volume change ratio (20d avg / 60d avg) ---
    if n_dates >= 60:
        vol_pivot_full = w.pivot(
            index="trade_date", columns="ts_code", values="vol"
        ).sort_index()
        vol_20 = vol_pivot_full.iloc[-20:].mean()
        vol_60 = vol_pivot_full.iloc[-60:].mean()
        features["volume_chg"] = (vol_20 / vol_60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["volume_chg"] = pd.Series(np.nan, index=valid_codes)

    # --- 14. high_low_20: 20-day (high - low) range / close ---
    if n_dates >= 20:
        high_pivot = w.pivot(
            index="trade_date", columns="ts_code", values="high"
        ).sort_index().iloc[-20:]
        low_pivot = w.pivot(
            index="trade_date", columns="ts_code", values="low"
        ).sort_index().iloc[-20:]
        h20 = high_pivot.max()
        l20 = low_pivot.min()
        last_close = close_pivot.iloc[-1]
        features["high_low_20"] = ((h20 - l20) / last_close.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["high_low_20"] = pd.Series(np.nan, index=valid_codes)

    # --- 15. close_to_high_60: close / 60-day high ---
    if n_dates >= 60:
        high_pivot_60 = w.pivot(
            index="trade_date", columns="ts_code", values="high"
        ).sort_index().iloc[-60:]
        h60 = high_pivot_60.max()
        last_close = close_pivot.iloc[-1]
        features["close_to_high_60"] = (last_close / h60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["close_to_high_60"] = pd.Series(np.nan, index=valid_codes)

    # --- 16. dv_ttm: dividend yield TTM ---
    snap_indexed = snap.set_index("ts_code") if "ts_code" in snap.columns else snap
    if "ts_code" in snap_indexed.columns:
        snap_indexed = snap_indexed  # already indexed
    features["dv_ttm"] = snap_indexed["dv_ttm"] if "dv_ttm" in snap_indexed.columns else pd.Series(np.nan, index=valid_codes)

    # --- 17. roa_ttm: return on assets ---
    features["roa_ttm"] = snap_indexed["roa"] if "roa" in snap_indexed.columns else pd.Series(np.nan, index=valid_codes)

    # --- 18. gross_margin: gross profit margin ---
    features["gross_margin"] = snap_indexed["grossprofit_margin"] if "grossprofit_margin" in snap_indexed.columns else pd.Series(np.nan, index=valid_codes)

    # --- 19. low_leverage: negative debt-to-assets (lower leverage is better) ---
    if "debt_to_assets" in snap_indexed.columns:
        dta = snap_indexed["debt_to_assets"]
        features["low_leverage"] = pd.Series(
            np.where(dta.notna(), -dta, np.nan), index=snap_indexed.index
        )
    else:
        features["low_leverage"] = pd.Series(np.nan, index=valid_codes)

    # --- 20. growth_revenue: revenue YoY growth ---
    features["growth_revenue"] = snap_indexed["tr_yoy"] if "tr_yoy" in snap_indexed.columns else pd.Series(np.nan, index=valid_codes)

    # --- 21. growth_profit: operating profit YoY growth ---
    features["growth_profit"] = snap_indexed["op_yoy"] if "op_yoy" in snap_indexed.columns else pd.Series(np.nan, index=valid_codes)

    # --- 22. illiq_20: Amihud illiquidity (20d) ---
    if n_dates >= 20:
        pctchg_pivot = w.pivot(
            index="trade_date", columns="ts_code", values="pct_chg"
        ).sort_index().iloc[-20:]
        amount_pivot = w.pivot(
            index="trade_date", columns="ts_code", values="amount"
        ).sort_index().iloc[-20:]
        abs_ret = pctchg_pivot.abs()
        amt_safe = amount_pivot.replace(0, np.nan)
        illiq = (abs_ret / amt_safe).mean()
        features["illiq_20"] = illiq.replace([np.inf, -np.inf], np.nan)
    else:
        features["illiq_20"] = pd.Series(np.nan, index=valid_codes)

    # Combine all features
    feat_df = pd.DataFrame(features)
    feat_df.index.name = "ts_code"
    feat_df = feat_df.reset_index()

    # Attach sw_l1 and circ_mv for later filtering
    meta = snap[["ts_code", "sw_l1", "circ_mv"]].copy()
    feat_df = feat_df.merge(meta, on="ts_code", how="inner")

    # Drop rows with too many NaN features (allow up to 5 missing)
    feat_df = feat_df.dropna(subset=FEATURE_NAMES, thresh=len(FEATURE_NAMES) - 5)

    return feat_df if len(feat_df) >= 50 else None


def rank_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Rank-normalize specified columns to [0, 1] percentile within the
    cross-section.  NaN values remain NaN.
    """
    df = df.copy()
    for col in columns:
        s = df[col]
        # rank: average method, pct=True maps to [0,1]
        df[col] = s.rank(pct=True, method="average")
    return df


# ===================================================================
# Label construction
# ===================================================================

def compute_forward_return(
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    accessor: DataAccessor,
) -> Optional[pd.Series]:
    """
    Compute forward return (close-to-close) for label construction.

    Returns pd.Series indexed by ts_code with the return value,
    or None if data is insufficient.
    """
    prices_now = accessor.get_prices(date)
    prices_next = accessor.get_prices(next_date)

    if not prices_now or not prices_next:
        return None

    result = {}
    for code, p0 in prices_now.items():
        p1 = prices_next.get(code)
        if p1 is not None and p0 > 0:
            result[code] = p1 / p0 - 1

    if len(result) < 50:
        return None

    s = pd.Series(result, dtype=float)
    s.index.name = "ts_code"
    return s


# ===================================================================
# LightGBM model training
# ===================================================================

def train_lgbm_model(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: Optional[pd.DataFrame] = None,
    val_labels: Optional[pd.Series] = None,
) -> lgb.Booster:
    """
    Train a LightGBM model for cross-sectional ranking.

    Parameters
    ----------
    train_features : pd.DataFrame
        Feature matrix (columns = FEATURE_NAMES).
    train_labels : pd.Series
        Cross-sectional rank percentile [0, 1] as labels.
    val_features, val_labels : optional
        Validation set for early stopping.

    Returns
    -------
    lgb.Booster  —  trained model.
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "lambda_l1": 0.5,
        "lambda_l2": 5.0,
        "max_depth": 4,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Fill NaN with column median for training
    X_train = train_features[FEATURE_NAMES].copy()
    for col in FEATURE_NAMES:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)

    dtrain = lgb.Dataset(X_train, label=train_labels)

    callbacks = [lgb.log_evaluation(period=0)]  # suppress logging
    valid_sets = [dtrain]
    valid_names = ["train"]

    if val_features is not None and val_labels is not None and len(val_labels) > 0:
        X_val = val_features[FEATURE_NAMES].copy()
        for col in FEATURE_NAMES:
            med = X_train[col].median()
            X_val[col] = X_val[col].fillna(med)
        dval = lgb.Dataset(X_val, label=val_labels, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


# ===================================================================
# Strategy class
# ===================================================================

class LGBMCrossSectional(StrategyBase):
    """
    LightGBM cross-sectional stock selection strategy.

    Uses a 3-year rolling training window with purge to train a
    LightGBM model that predicts cross-sectional return rankings.
    Selects top 5% by ML score, constrained to max 5 per industry,
    with a holding buffer of +0.3σ for current positions.
    """

    def __init__(
        self,
        train_window_years: int = 3,
        top_pct: float = 0.05,
        max_per_industry: int = 5,
        buffer_sigma: float = 0.3,
        mv_pct_lower: float = 0.0,
        mv_pct_upper: float = 0.70,
        feature_lookback: int = 260,
        backtest_end_date: Optional[str] = None,
    ):
        super().__init__("lgbm_cross_sectional")
        self.train_window_years = train_window_years
        self.top_pct = top_pct
        self.max_per_industry = max_per_industry
        self.buffer_sigma = buffer_sigma
        self.mv_pct_lower = mv_pct_lower
        self.mv_pct_upper = mv_pct_upper
        self.feature_lookback = feature_lookback
        self._backtest_end_date = (
            pd.Timestamp(backtest_end_date) if backtest_end_date else None
        )

        # State
        self._model: Optional[lgb.Booster] = None
        self._train_data_cache: List[Tuple[str, pd.DataFrame, pd.Series]] = []
        self._last_train_date: Optional[pd.Timestamp] = None
        self._retrain_interval = 2  # retrain every N rebalance periods
        self._call_count = 0
        self._warmup_done = False
        self._bulk_data: Optional[pd.DataFrame] = None  # pre-fetched data cache
        self._st_codes: Optional[set] = None  # ST stock codes cache (from stock_info.name)

    def describe(self) -> str:
        return (
            f"### 策略思路\n\n"
            f"基于 LightGBM 的截面选股策略（V4），用机器学习取代传统线性多因子的固定权重，"
            f"自动学习因子间的非线性交互效应。\n\n"
            f"### 核心设计\n\n"
            f"1. **截面排序标签**：预测相对排名（百分位），消除收益分布的非平稳性\n"
            f"2. **3 年滚动训练窗口**：训练集按标签结束日 purge，避免前视偏差\n"
            f"3. **行业约束选股**：ML 分数前 {self.top_pct*100:.0f}%，"
            f"每行业最多 {self.max_per_industry} 只\n"
            f"4. **持仓缓冲带**：在持股票 ML 分数加 {self.buffer_sigma}σ，降低换手\n"
            f"5. **选股范围**：沪深主板，自由流通市值前 {self.mv_pct_upper*100:.0f}%\n\n"
            f"### 22 个特征\n\n"
            f"| 编号 | 特征名 | 大类 | 说明 |\n"
            f"|------|--------|------|------|\n"
            f"| 1 | mom_12_1 | 动量 | 12 个月动量（跳过最近 1 个月） |\n"
            f"| 2 | rev_10 | 反转 | 10 日短期反转 |\n"
            f"| 3 | rvol_20 | 风险 | 20 日已实现波动率 |\n"
            f"| 4 | vol_confirm | 量价 | 20 日量价相关性 |\n"
            f"| 5 | inv_pb | 价值 | 1/PB |\n"
            f"| 6 | log_cap | 规模 | log(流通市值) |\n"
            f"| 7 | pe_ttm | 价值 | 市盈率 TTM |\n"
            f"| 8 | roe_ttm | 质量 | 净资产收益率 |\n"
            f"| 9 | turnover_20 | 量价 | 20 日平均换手率 |\n"
            f"| 10 | mom_3_1 | 动量 | 3 个月动量（跳过 1 个月） |\n"
            f"| 11 | mom_6_1 | 动量 | 6 个月动量（跳过 1 个月） |\n"
            f"| 12 | ret_5d_std | 风险 | 5 日收益率标准差 |\n"
            f"| 13 | volume_chg | 量价 | 成交量变化比（20d/60d） |\n"
            f"| 14 | high_low_20 | 风险 | 20 日振幅 / 收盘价 |\n"
            f"| 15 | close_to_high_60 | 技术 | 收盘价 / 60 日最高价 |\n"
            f"| 16 | dv_ttm | 分红 | 股息率 TTM |\n"
            f"| 17 | roa_ttm | 质量 | 总资产收益率 |\n"
            f"| 18 | gross_margin | 质量 | 毛利率 |\n"
            f"| 19 | low_leverage | 质量 | 低杠杆（-资产负债率） |\n"
            f"| 20 | growth_revenue | 成长 | 营收同比增速 |\n"
            f"| 21 | growth_profit | 成长 | 营业利润同比增速 |\n"
            f"| 22 | illiq_20 | 流动性 | 20 日 Amihud 非流动性 |\n\n"
            f"### LightGBM 超参数\n\n"
            f"| 参数 | 值 |\n"
            f"|------|----|\n"
            f"| num_leaves | 31 |\n"
            f"| learning_rate | 0.05 |\n"
            f"| feature_fraction | 0.7 |\n"
            f"| bagging_fraction | 0.8 |\n"
            f"| max_depth | 4 |\n"
            f"| min_child_samples | 100 |\n"
            f"| lambda_l1 | 0.5 |\n"
            f"| lambda_l2 | 5.0 |\n"
            f"| num_boost_round | 300 (early stop 20) |\n\n"
            f"### 已知局限\n\n"
f"- 已过滤 ST / *ST 股票（基于 stock_info.name 静态名称匹配）\n"
            f"- 等权分配，未按 ML 分数做信号强度加权"
        )

    def _should_retrain(self) -> bool:
        """Decide whether to retrain the model on this call."""
        if self._model is None:
            return True
        return self._call_count % self._retrain_interval == 0

    def _warmup_training_cache(
        self,
        current_date: pd.Timestamp,
        accessor: DataAccessor,
    ):
        """
        Pre-compute features and labels for historical biweekly dates
        covering the 3-year training window BEFORE the backtest starts.

        Optimized: loads ALL data in ONE SQL query, then computes
        features/labels from memory — no per-date DB hits.
        """
        warmup_start = current_date - pd.DateOffset(
            years=self.train_window_years, months=2
        )
        warmup_end = current_date - pd.DateOffset(days=1)

        print(f"      [预热] 加载历史训练数据 "
              f"{warmup_start.strftime('%Y-%m-%d')} ~ "
              f"{warmup_end.strftime('%Y-%m-%d')} ...")

        # ── Step 0: Cache ST stock codes from stock_info table (one-time query)
        if self._st_codes is None:
            try:
                st_df = pd.read_sql_query(
                    "SELECT ts_code, name FROM stock_info WHERE name LIKE '%ST%'",
                    accessor.conn,
                )
                self._st_codes = set(st_df["ts_code"].tolist())
                print(f"      [ST过滤] ✓ 从 stock_info 加载 {len(self._st_codes)} 只 ST/*ST 股票")
            except Exception as e:
                print(f"      [ST过滤] ✗ 查询失败: {e}，跳过 ST 过滤")
                self._st_codes = set()

        # ── Step 1: Bulk-load ALL data for (warmup_start - lookback) ~ warmup_end
        # We need `lookback` extra days before warmup_start for feature computation
        data_start = warmup_start - pd.DateOffset(days=int(self.feature_lookback * 1.8))
        bulk = prefetch_bulk_data(accessor, data_start, warmup_end)

        if bulk.empty:
            print(f"      [预热] ✗ 未找到历史数据")
            return

        # Also prefetch data covering the backtest period (for future use)
        # Use actual backtest end date if provided, otherwise use a generous buffer
        if self._backtest_end_date is not None:
            backtest_end = self._backtest_end_date + pd.DateOffset(months=1)
        else:
            backtest_end = current_date + pd.DateOffset(years=4)
        bulk_backtest = prefetch_bulk_data(accessor, warmup_end, backtest_end)
        if not bulk_backtest.empty:
            self._bulk_data = pd.concat([bulk, bulk_backtest], ignore_index=True)
            self._bulk_data.drop_duplicates(
                subset=["ts_code", "trade_date"], keep="last", inplace=True
            )
            self._bulk_data.sort_values(["trade_date", "ts_code"], inplace=True)
            self._bulk_data.reset_index(drop=True, inplace=True)
        else:
            self._bulk_data = bulk

        # Clear date index cache — will be rebuilt lazily on first use
        _bulk_date_index_cache.clear()

        # ── Step 2: Determine biweekly rebalance dates within warmup window
        ws = warmup_start.strftime("%Y%m%d")
        we = warmup_end.strftime("%Y%m%d")
        hist_trade_dates_raw = bulk["trade_date"].unique()
        hist_trade_dates_raw = np.sort(hist_trade_dates_raw)
        # Use pd.Timestamp for comparison to avoid type mismatch
        mask = (hist_trade_dates_raw >= pd.Timestamp(warmup_start)) & \
               (hist_trade_dates_raw <= pd.Timestamp(warmup_end))
        hist_trade_dates = pd.DatetimeIndex(hist_trade_dates_raw[mask])

        if len(hist_trade_dates) == 0:
            print(f"      [预热] ✗ 历史交易日为空")
            return

        origin = hist_trade_dates[0]
        day_offsets = (hist_trade_dates - origin).days
        block_ids = day_offsets // 14
        s = pd.Series(hist_trade_dates, index=hist_trade_dates)
        groups = s.groupby(block_ids)
        hist_rebal_dates = pd.DatetimeIndex(groups.last().values)

        print(f"      [预热] 历史调仓日 {len(hist_rebal_dates)} 个，"
              f"开始计算特征和标签 (全内存模式) ...")

        # ── Step 3: Compute features and labels from memory
        import time
        t0 = time.time()
        n_success = 0
        n_skip = 0
        for i in range(len(hist_rebal_dates) - 1):
            d = hist_rebal_dates[i]
            d_next = hist_rebal_dates[i + 1]

            feat_df = compute_features_from_memory(
                d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            if feat_df is None or feat_df.empty:
                n_skip += 1
                continue

            feat_df = self._apply_market_cap_filter(feat_df)
            if len(feat_df) < 50:
                n_skip += 1
                continue

            feat_ranked = rank_normalize(feat_df, FEATURE_NAMES)

            fwd_ret = compute_forward_return_from_memory(d, d_next, self._bulk_data)
            if fwd_ret is None:
                n_skip += 1
                continue

            d_str = d.strftime("%Y-%m-%d")
            self._train_data_cache.append((d_str, feat_ranked.copy(), fwd_ret))
            n_success += 1

        # Also cache the LAST date
        if len(hist_rebal_dates) > 0:
            last_d = hist_rebal_dates[-1]
            feat_df = compute_features_from_memory(
                last_d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            if feat_df is not None and not feat_df.empty:
                feat_df = self._apply_market_cap_filter(feat_df)
                if len(feat_df) >= 50:
                    feat_ranked = rank_normalize(feat_df, FEATURE_NAMES)
                    fwd_ret = compute_forward_return_from_memory(
                        last_d, current_date, self._bulk_data
                    )
                    if fwd_ret is not None:
                        d_str = last_d.strftime("%Y-%m-%d")
                        self._train_data_cache.append(
                            (d_str, feat_ranked.copy(), fwd_ret)
                        )
                        n_success += 1

        # Prune old data beyond training window
        cutoff = current_date - pd.DateOffset(
            years=self.train_window_years, months=6
        )
        self._train_data_cache = [
            (d, f, l) for d, f, l in self._train_data_cache
            if pd.Timestamp(d) >= cutoff
        ]

        elapsed = time.time() - t0
        print(f"      [预热] ✓ 成功缓存 {n_success} 期 "
              f"(跳过 {n_skip} 期)，"
              f"耗时 {elapsed:.1f} 秒，"
              f"训练窗口已就绪")

    def _collect_training_data(
        self,
        current_date: pd.Timestamp,
        accessor: DataAccessor,
        rebal_dates_history: List[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Collect rolling training data from historical rebalance dates.

        Returns (train_X, train_y, val_X, val_y).
        """
        # Determine training window: 3 years back from current date
        train_start = current_date - pd.DateOffset(years=self.train_window_years)

        all_X = []
        all_y = []

        # Use cached data
        for date_str, feat_df, labels in self._train_data_cache:
            dt = pd.Timestamp(date_str)
            if dt >= train_start and dt < current_date:
                # Merge features with labels
                merged = feat_df.set_index("ts_code").join(
                    labels.rename("label"), how="inner"
                )
                if len(merged) >= 50:
                    # Rank-normalize labels to [0, 1]
                    merged["label"] = merged["label"].rank(pct=True, method="average")
                    X = merged[FEATURE_NAMES]
                    y = merged["label"]
                    all_X.append(X)
                    all_y.append(y)

        if not all_X:
            return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float)

        # Split: last 20% as validation, rest as training
        n = len(all_X)
        split_idx = max(1, int(n * 0.8))

        train_X = pd.concat(all_X[:split_idx], axis=0)
        train_y = pd.concat(all_y[:split_idx], axis=0)
        val_X = pd.concat(all_X[split_idx:], axis=0) if split_idx < n else pd.DataFrame()
        val_y = pd.concat(all_y[split_idx:], axis=0) if split_idx < n else pd.Series(dtype=float)

        return train_X, train_y, val_X, val_y

    def _update_training_cache(
        self,
        date: pd.Timestamp,
        next_date: pd.Timestamp,
        accessor: DataAccessor,
        feat_df: pd.DataFrame,
    ):
        """
        Compute forward returns for the given date and cache
        (features, labels) for future training.
        """
        fwd_ret = compute_forward_return(date, next_date, accessor)
        if fwd_ret is not None:
            date_str = date.strftime("%Y-%m-%d")
            self._train_data_cache.append((date_str, feat_df.copy(), fwd_ret))

            # Prune old data beyond training window + 6 months buffer
            cutoff = date - pd.DateOffset(years=self.train_window_years, months=6)
            self._train_data_cache = [
                (d, f, l) for d, f, l in self._train_data_cache
                if pd.Timestamp(d) >= cutoff
            ]

    def _apply_market_cap_filter(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to top mv_pct_upper by circulating market value."""
        if "circ_mv" not in feat_df.columns:
            return feat_df
        mv = feat_df["circ_mv"].dropna()
        if len(mv) == 0:
            return feat_df
        lower_bound = mv.quantile(1.0 - self.mv_pct_upper)
        return feat_df[feat_df["circ_mv"] >= lower_bound].copy()

    def _select_stocks(
        self,
        scores_df: pd.DataFrame,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Select stocks: top 5% by ML score, max per industry, with buffer.

        Parameters
        ----------
        scores_df : pd.DataFrame
            Must contain columns: ts_code, sw_l1, ml_score.
        current_holdings : dict
            Current holdings {ts_code: shares}.

        Returns
        -------
        dict  —  {ts_code: weight} equal-weight portfolio.
        """
        df = scores_df.copy()

        # Apply holding buffer: boost ML score for currently held stocks
        if current_holdings and self.buffer_sigma > 0:
            score_std = df["ml_score"].std()
            if score_std > 0:
                held_codes = set(current_holdings.keys())
                boost = self.buffer_sigma * score_std
                df.loc[df["ts_code"].isin(held_codes), "ml_score"] += boost

        # Sort by ML score descending
        df = df.sort_values("ml_score", ascending=False).reset_index(drop=True)

        # Top 5%
        n_select = max(1, int(len(df) * self.top_pct))
        candidates = df.head(n_select).copy()

        # Industry constraint: max N per industry
        selected = []
        industry_count: Dict[str, int] = {}
        for _, row in candidates.iterrows():
            ind = row.get("sw_l1", "unknown")
            if pd.isna(ind):
                ind = "unknown"
            cnt = industry_count.get(ind, 0)
            if cnt < self.max_per_industry:
                selected.append(row["ts_code"])
                industry_count[ind] = cnt + 1

        if not selected:
            return {}

        # Equal weight
        w = 1.0 / len(selected)
        return {code: w for code in selected}

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        """Main entry point called by the backtest engine."""
        self._call_count += 1
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n      ── 策略第 {self._call_count} 期  {date_str} ──")

        # Step 0: Warmup — pre-fill training cache from historical data
        if not self._warmup_done:
            self._warmup_done = True
            self._warmup_training_cache(date, accessor)

        # Step 1: Compute features for the current date
        # Prefer in-memory computation if bulk data is available
        print(f"      [特征] 计算 {len(FEATURE_NAMES)} 个特征 (回看 {self.feature_lookback} 天) ...")
        if self._bulk_data is not None:
            feat_df = compute_features_from_memory(
                date, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
        else:
            feat_df = compute_features_for_date(
                date, accessor, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
        if feat_df is None or feat_df.empty:
            print(f"      [特征] ✗ 数据不足，本期跳过")
            return {}

        # Step 2: Apply market cap filter (top 70%)
        n_before_mv = len(feat_df)
        feat_df = self._apply_market_cap_filter(feat_df)
        print(f"      [特征] ✓ 有效股票 {n_before_mv} 只 → 市值过滤后 {len(feat_df)} 只")
        if len(feat_df) < 50:
            print(f"      [特征] ✗ 过滤后不足 50 只，本期跳过")
            return {}

        # Step 3: Rank-normalize features
        feat_ranked = rank_normalize(feat_df, FEATURE_NAMES)
        print(f"      [特征] ✓ 排序归一化完成")

        # Step 4: Try to update training cache with PREVIOUS period's label
        # We use the current date's prices to compute forward return for
        # the PREVIOUS period (which is now realised, no look-ahead)
        if self._train_data_cache:
            last_cached = self._train_data_cache[-1]
            last_date = pd.Timestamp(last_cached[0])
            # Check if this is a new period (not the same date)
            if last_date < date:
                # The forward return from last_date to current date is now known
                if self._bulk_data is not None:
                    fwd_ret = compute_forward_return_from_memory(
                        last_date, date, self._bulk_data
                    )
                else:
                    fwd_ret = compute_forward_return(last_date, date, accessor)
                if fwd_ret is not None:
                    # Replace the label (was None before)
                    self._train_data_cache[-1] = (
                        last_cached[0], last_cached[1], fwd_ret
                    )

        # Step 5: Cache current features for future training
        # Label will be filled next period
        date_str = date.strftime("%Y-%m-%d")
        # Check if already cached
        cached_dates = {d for d, _, _ in self._train_data_cache}
        if date_str not in cached_dates:
            # Store with empty label (will be filled next period)
            self._train_data_cache.append(
                (date_str, feat_ranked.copy(), pd.Series(dtype=float))
            )

        # Step 6: Train or reuse model
        n_cached = len([1 for d, _, l in self._train_data_cache if not l.empty and pd.Timestamp(d) < date])
        min_train_periods = 4
        if self._should_retrain() and n_cached >= min_train_periods:
            print(f"      [训练] 触发重训练 (缓存 {n_cached} 期有标签数据) ...")
        elif not self._should_retrain() and self._model is not None:
            print(f"      [训练] 沿用现有模型 (上次训练: {self._last_train_date.strftime('%Y-%m-%d') if self._last_train_date else '无'})")
        elif n_cached < min_train_periods:
            print(f"      [训练] 有标签期数不足 ({n_cached} 期, 需要 ≥{min_train_periods})，跳过训练")
        if self._should_retrain() and n_cached >= min_train_periods:
            # Only use entries with actual labels
            valid_cache = [
                (d, f, l) for d, f, l in self._train_data_cache
                if not l.empty and pd.Timestamp(d) < date
            ]
            if len(valid_cache) < 4:
                print(f"      [训练] ✗ 有效训练期不足 (需要 ≥4, 当前 {len(valid_cache)})")
            if len(valid_cache) >= 4:
                # Prepare training data
                all_X = []
                all_y = []
                train_start = date - pd.DateOffset(years=self.train_window_years)
                for d_str, f_df, labels in valid_cache:
                    dt = pd.Timestamp(d_str)
                    if dt >= train_start:
                        merged = f_df.set_index("ts_code").join(
                            labels.rename("label"), how="inner"
                        )
                        if len(merged) >= 30:
                            merged["label"] = merged["label"].rank(
                                pct=True, method="average"
                            )
                            all_X.append(merged[FEATURE_NAMES])
                            all_y.append(merged["label"])

                if len(all_X) < 4:
                    print(f"      [训练] ✗ 合并后训练期不足 (需要 ≥4, 当前 {len(all_X)})")
                elif len(all_X) >= 4:
                    n = len(all_X)
                    split = max(1, int(n * 0.8))

                    train_X = pd.concat(all_X[:split])
                    train_y = pd.concat(all_y[:split])
                    val_X = pd.concat(all_X[split:]) if split < n else None
                    val_y = pd.concat(all_y[split:]) if split < n else None

                    try:
                        self._model = train_lgbm_model(
                            train_X, train_y, val_X, val_y
                        )
                        self._last_train_date = date
                        n_train = len(train_y)
                        n_val = len(val_y) if val_y is not None else 0
                        best_iter = self._model.best_iteration if hasattr(self._model, 'best_iteration') else '?'
                        print(
                            f"      [训练] ✓ 模型训练完成  "
                            f"训练={n_train:,} 样本  验证={n_val:,} 样本  "
                            f"使用 {len(all_X)} 期历史  "
                            f"最优轮次={best_iter}"
                        )
                    except Exception as e:
                        print(f"      [训练] ✗ 训练失败: {e}")

        # Step 7: Predict
        if self._model is None:
            # Cold start: use simple composite of rank-normalized features
            # (fallback before enough training data)
            scores = feat_ranked[FEATURE_NAMES].mean(axis=1)
            result_df = feat_ranked[["ts_code", "sw_l1"]].copy()
            result_df["ml_score"] = scores.values
            print(f"      [预测] 冷启动模式 (等权因子均值)，候选 {len(result_df)} 只")
        else:
            # Predict using LightGBM
            X_pred = feat_ranked[FEATURE_NAMES].copy()
            for col in FEATURE_NAMES:
                med = X_pred[col].median()
                X_pred[col] = X_pred[col].fillna(med)

            preds = self._model.predict(X_pred)
            result_df = feat_ranked[["ts_code", "sw_l1"]].copy()
            result_df["ml_score"] = preds
            print(
                f"      [预测] LightGBM 预测完成  "
                f"候选={len(result_df)} 只  "
                f"分数区间=[{preds.min():.4f}, {preds.max():.4f}]"
            )

        # Step 8: Select stocks with industry constraint and buffer
        weights = self._select_stocks(result_df, current_holdings)
        n_held = len(current_holdings) if current_holdings else 0
        n_industries = result_df[result_df["ts_code"].isin(weights.keys())]["sw_l1"].nunique() if weights else 0
        overlap = len(set(weights.keys()) & set(current_holdings.keys())) if current_holdings and weights else 0
        print(
            f"      [选股] 入选 {len(weights)} 只 / {n_industries} 个行业  "
            f"(上期持仓 {n_held} 只，留存 {overlap} 只，换手 {n_held + len(weights) - 2 * overlap} 只)"
        )
        return weights


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=1.5e-4,
        slippage=0.0015,
        start_date="2022-01-01",
        end_date="2025-12-31",
        rebalance_freq="BW",
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir="data/quant/backtest",
    )

    strategy = LGBMCrossSectional(
        train_window_years=3,
        top_pct=0.05,
        max_per_industry=5,
        buffer_sigma=0.3,
        mv_pct_upper=0.85,
        feature_lookback=260,
        backtest_end_date=cfg.end_date,
    )
    result = run_backtest(strategy, cfg)
