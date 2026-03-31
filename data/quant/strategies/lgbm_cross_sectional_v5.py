"""
LightGBM Cross-Sectional Stock Selection Strategy (V5)
======================================================

Upgrade from V4 — Feature Engineering Optimization
---------------------------------------------------
Key changes vs V4:
1. **Removed redundant features** (22 → 19 effective features):
   - Dropped `mom_3_1`, `mom_6_1` (highly correlated with `mom_12_1`)
   - Dropped `ret_5d_std`, `high_low_20` (redundant with `rvol_20`)
   - Dropped `pe_ttm` (redundant with `inv_pb`)
2. **Added industry-relative momentum** (`ind_rel_mom`):
   stock_mom_12_1 - industry_avg_mom_12_1, isolates stock-specific alpha
3. **Added volatility-adjusted momentum** (`mom_sharpe`):
   mom_12_1 / rvol_20, higher signal-to-noise ratio
4. **Added market beta** (`market_beta`):
   60-day rolling regression of stock returns on market returns
5. **Added market state features**:
   - `mkt_trend`: market 20d MA / 60d MA ratio (trend indicator)
   - `mkt_vol`: market 20d realized volatility (regime indicator)

Net feature count: 22 - 5 removed + 5 added = 22 features
But with higher information content and lower redundancy.

Usage
-----
cd <project_root>
python -m data.quant.strategies.lgbm_cross_sectional_v5
"""
import sys
import os
import warnings
from pathlib import Path

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
from strategies.utils import prefetch_bulk_data

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


# ===================================================================
# Feature engineering (V5 — optimized)
# ===================================================================

# All DB columns needed by features
FEATURE_COLUMNS = [
    "ts_code", "trade_date",
    "open", "high", "low", "close", "pre_close",
    "pct_chg", "vol", "amount",
    "turnover_rate_f",
    "pb", "circ_mv",
    "roe", "roa", "grossprofit_margin", "debt_to_assets",
    "dv_ttm",
    "tr_yoy", "op_yoy",
    "sw_l1", "is_suspended",
]

# V5 feature names (22 features, optimized from V4)
FEATURE_NAMES = [
    # ── Momentum (refined) ──
    "mom_12_1",          # 12-month momentum, skip most recent month
    "ind_rel_mom",       # [NEW] industry-relative momentum (stock - industry avg)
    "mom_sharpe",        # [NEW] volatility-adjusted momentum (mom / rvol)
    "rev_10",            # 10-day reversal
    # ── Risk ──
    "rvol_20",           # 20-day realized volatility
    "market_beta",       # [NEW] 60-day market beta
    # ── Volume / Price ──
    "vol_confirm",       # volume-price correlation (20d)
    "turnover_20",       # 20-day average turnover rate
    "volume_chg",        # volume change ratio (20d vs 60d)
    # ── Value ──
    "inv_pb",            # 1/PB (value)
    # ── Size ──
    "log_cap",           # log(circulating market value)
    # ── Quality ──
    "roe_ttm",           # ROE
    "roa_ttm",           # return on assets
    "gross_margin",      # gross profit margin
    "low_leverage",      # -debt_to_assets (lower is better)
    # ── Growth ──
    "growth_revenue",    # revenue YoY growth
    "growth_profit",     # operating profit YoY growth
    # ── Dividend ──
    "dv_ttm",            # dividend yield TTM
    # ── Technical ──
    "close_to_high_60",  # close / 60-day high
    "illiq_20",          # Amihud illiquidity (20d)
    # ── Market State [NEW] ──
    "mkt_trend",         # [NEW] market 20d MA / 60d MA (trend)
    "mkt_vol",           # [NEW] market 20d realized volatility (regime)
]


def _compute_momentum(close_pivot: pd.DataFrame, start_offset: int,
                       end_offset: int) -> pd.Series:
    """
    Compute momentum: close[end_offset] / close[start_offset] - 1.
    Offsets are in number of rows from the end (0 = last row).
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


# Pre-computed date index cache
_bulk_date_index_cache: Dict[int, Tuple[np.ndarray, Dict]] = {}


def _get_bulk_date_index(bulk_data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """Build or retrieve a cached date index for bulk_data."""
    cache_key = id(bulk_data)
    if cache_key in _bulk_date_index_cache:
        return _bulk_date_index_cache[cache_key]

    dates = bulk_data["trade_date"].values
    unique_dates_raw = np.sort(np.unique(dates))
    unique_dates = pd.DatetimeIndex(unique_dates_raw)
    date_to_rows = {}
    for d_ts in unique_dates:
        date_to_rows[d_ts] = np.where(dates == d_ts)[0]

    _bulk_date_index_cache[cache_key] = (unique_dates, date_to_rows)
    return unique_dates, date_to_rows


def _compute_market_returns(close_pivot: pd.DataFrame) -> pd.Series:
    """
    Compute equal-weighted market daily returns from close_pivot.
    Returns a Series indexed by trade_date.
    """
    daily_ret = close_pivot.pct_change(fill_method=None)
    # Equal-weighted market return = cross-sectional mean each day
    mkt_ret = daily_ret.mean(axis=1)
    return mkt_ret


def compute_features_from_memory(
    date: pd.Timestamp,
    bulk_data: pd.DataFrame,
    lookback: int = 260,
    st_codes: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute all V5 features using in-memory bulk_data.

    V5 changes vs V4:
    - Removed: mom_3_1, mom_6_1, ret_5d_std, high_low_20, pe_ttm
    - Added: ind_rel_mom, mom_sharpe, market_beta, mkt_trend, mkt_vol
    """
    all_dates, date_to_rows = _get_bulk_date_index(bulk_data)

    date_ts = pd.Timestamp(date)

    valid_dates = all_dates[all_dates <= date_ts]
    if len(valid_dates) < 60:
        return None
    window_dates = valid_dates[-lookback:] if len(valid_dates) >= lookback else valid_dates

    # Gather rows for all window dates
    row_indices = []
    for wd in window_dates:
        if wd in date_to_rows:
            row_indices.append(date_to_rows[wd])
    if not row_indices:
        return None
    all_row_idx = np.concatenate(row_indices)
    window = bulk_data.iloc[all_row_idx]

    # Get snapshot for target date
    if date_ts in date_to_rows:
        snap = bulk_data.iloc[date_to_rows[date_ts]].copy()
    else:
        return None
    if snap.empty:
        return None

    snap = snap[
        ["ts_code", "close", "pb", "circ_mv",
         "roe", "roa", "grossprofit_margin", "debt_to_assets",
         "dv_ttm", "tr_yoy", "op_yoy",
         "sw_l1", "is_suspended", "turnover_rate_f"]
    ].copy()

    # Filter universe
    snap = snap[
        (snap["is_suspended"] != 1)
        & (snap["close"].notna())
        & (snap["close"] > 0)
    ].copy()

    snap = snap[
        snap["ts_code"].str.match(r"^(6\d{5}\.SH|00[013]\d{3}\.SZ)$")
    ].copy()

    if st_codes:
        snap = snap[~snap["ts_code"].isin(st_codes)].copy()

    if len(snap) < 100:
        return None

    universe_codes = set(snap["ts_code"].tolist())

    # Build pivots
    w = window[window["ts_code"].isin(universe_codes)].copy()
    w.sort_values(["trade_date", "ts_code"], inplace=True)

    w_indexed = w.set_index(["trade_date", "ts_code"])
    pivot_cols = ["close", "vol", "high", "low", "turnover_rate_f", "pct_chg", "amount"]
    pivots = {}
    for col in pivot_cols:
        if col in w_indexed.columns:
            pivots[col] = w_indexed[col].unstack("ts_code").sort_index()

    close_pivot = pivots["close"]
    vol_pivot = pivots.get("vol")
    high_pivot = pivots.get("high")
    turn_pivot = pivots.get("turnover_rate_f")
    pctchg_pivot = pivots.get("pct_chg")
    amount_pivot = pivots.get("amount")

    n_dates = len(close_pivot)
    features = {}
    valid_codes = close_pivot.columns.tolist()
    snap_indexed = snap.set_index("ts_code")

    # ── 1. mom_12_1: 12-month momentum, skip most recent month ──
    if n_dates >= 240:
        features["mom_12_1"] = _compute_momentum(close_pivot, 240, 20)
    else:
        features["mom_12_1"] = pd.Series(np.nan, index=valid_codes)

    # ── 2. ind_rel_mom [NEW]: industry-relative momentum ──
    # stock_mom_12_1 - industry_avg_mom_12_1
    mom_12_1 = features["mom_12_1"]
    if not mom_12_1.empty and "sw_l1" in snap_indexed.columns:
        # Build a mapping: ts_code -> sw_l1
        ind_map = snap_indexed["sw_l1"]
        # Compute industry average momentum
        mom_with_ind = pd.DataFrame({
            "mom": mom_12_1,
            "sw_l1": ind_map
        }).dropna(subset=["mom", "sw_l1"])
        ind_avg = mom_with_ind.groupby("sw_l1")["mom"].transform("mean")
        ind_rel = mom_with_ind["mom"] - ind_avg
        features["ind_rel_mom"] = ind_rel
    else:
        features["ind_rel_mom"] = pd.Series(np.nan, index=valid_codes)

    # ── 3. mom_sharpe [NEW]: volatility-adjusted momentum ──
    # Computed after rvol_20 (feature 5), placeholder here
    # Will be filled below

    # ── 4. rev_10: 10-day reversal ──
    if n_dates >= 10:
        features["rev_10"] = _compute_momentum(close_pivot, 10, 0)
    else:
        features["rev_10"] = pd.Series(np.nan, index=valid_codes)

    # ── 5. rvol_20: 20-day realized volatility ──
    if n_dates >= 21:
        daily_ret = close_pivot.iloc[-21:].pct_change(fill_method=None).iloc[1:]
        features["rvol_20"] = daily_ret.std()
    else:
        features["rvol_20"] = pd.Series(np.nan, index=valid_codes)

    # Now compute mom_sharpe = mom_12_1 / rvol_20
    rvol = features["rvol_20"]
    if not mom_12_1.empty and not rvol.empty:
        rvol_safe = rvol.replace(0, np.nan)
        features["mom_sharpe"] = (mom_12_1 / rvol_safe).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["mom_sharpe"] = pd.Series(np.nan, index=valid_codes)

    # ── 6. market_beta [NEW]: 60-day rolling beta ──
    if n_dates >= 60:
        ret_60 = close_pivot.iloc[-61:].pct_change(fill_method=None).iloc[1:]
        mkt_ret = ret_60.mean(axis=1)  # equal-weighted market return

        # Vectorized beta computation: cov(stock, market) / var(market)
        mkt_arr = mkt_ret.values
        mkt_dm = mkt_arr - np.nanmean(mkt_arr)
        mkt_var = np.nansum(mkt_dm ** 2)

        if mkt_var > 0:
            stock_arr = ret_60.values  # shape: (60, n_stocks)
            # Demean each stock
            stock_dm = stock_arr - np.nanmean(stock_arr, axis=0, keepdims=True)
            # Handle NaN: set NaN positions to 0 for dot product, adjust count
            mask = ~(np.isnan(stock_dm) | np.isnan(mkt_dm[:, None]))
            stock_dm_clean = np.where(mask, stock_dm, 0)
            mkt_dm_clean = np.where(mask, mkt_dm[:, None], 0)

            cov = np.sum(stock_dm_clean * mkt_dm_clean, axis=0)
            # Adjust variance for valid observations
            mkt_var_adj = np.sum(mkt_dm_clean ** 2, axis=0)
            mkt_var_adj = np.where(mkt_var_adj > 0, mkt_var_adj, np.nan)

            beta_vals = cov / mkt_var_adj
            features["market_beta"] = pd.Series(
                beta_vals, index=ret_60.columns
            ).replace([np.inf, -np.inf], np.nan)
        else:
            features["market_beta"] = pd.Series(np.nan, index=valid_codes)
    else:
        features["market_beta"] = pd.Series(np.nan, index=valid_codes)

    # ── 7. vol_confirm: volume-price correlation (20d) ──
    if n_dates >= 20 and vol_pivot is not None:
        ret_20 = close_pivot.iloc[-20:].pct_change(fill_method=None).iloc[1:]
        vp_20 = vol_pivot.iloc[-20:]
        common_dates = ret_20.index.intersection(vp_20.index)
        if len(common_dates) >= 10:
            r = ret_20.loc[common_dates]
            v = vp_20.loc[common_dates]
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

    # ── 8. turnover_20: 20-day average turnover rate ──
    if n_dates >= 20 and turn_pivot is not None:
        features["turnover_20"] = turn_pivot.iloc[-20:].mean()
    else:
        features["turnover_20"] = pd.Series(np.nan, index=valid_codes)

    # ── 9. volume_chg: volume change ratio (20d avg / 60d avg) ──
    if n_dates >= 60 and vol_pivot is not None:
        vol_20 = vol_pivot.iloc[-20:].mean()
        vol_60 = vol_pivot.iloc[-60:].mean()
        features["volume_chg"] = (vol_20 / vol_60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["volume_chg"] = pd.Series(np.nan, index=valid_codes)

    # ── 10. inv_pb: 1/PB ──
    pb = snap_indexed["pb"]
    features["inv_pb"] = pd.Series(
        np.where(pb > 0, 1.0 / pb, np.nan), index=snap_indexed.index
    )

    # ── 11. log_cap: log(circulating market value) ──
    mv = snap_indexed["circ_mv"]
    features["log_cap"] = pd.Series(
        np.where(mv > 0, np.log(mv), np.nan), index=snap_indexed.index
    )

    # ── 12. roe_ttm ──
    features["roe_ttm"] = snap_indexed["roe"]

    # ── 13. roa_ttm ──
    features["roa_ttm"] = snap_indexed["roa"]

    # ── 14. gross_margin ──
    features["gross_margin"] = snap_indexed["grossprofit_margin"]

    # ── 15. low_leverage: -debt_to_assets ──
    dta = snap_indexed["debt_to_assets"]
    features["low_leverage"] = pd.Series(
        np.where(dta.notna(), -dta, np.nan), index=snap_indexed.index
    )

    # ── 16. growth_revenue ──
    features["growth_revenue"] = snap_indexed["tr_yoy"]

    # ── 17. growth_profit ──
    features["growth_profit"] = snap_indexed["op_yoy"]

    # ── 18. dv_ttm: dividend yield TTM ──
    features["dv_ttm"] = snap_indexed["dv_ttm"]

    # ── 19. close_to_high_60 ──
    if n_dates >= 60 and high_pivot is not None:
        h60 = high_pivot.iloc[-60:].max()
        last_close = close_pivot.iloc[-1]
        features["close_to_high_60"] = (last_close / h60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["close_to_high_60"] = pd.Series(np.nan, index=valid_codes)

    # ── 20. illiq_20: Amihud illiquidity (20d) ──
    if n_dates >= 20 and pctchg_pivot is not None and amount_pivot is not None:
        pc20 = pctchg_pivot.iloc[-20:]
        am20 = amount_pivot.iloc[-20:]
        abs_ret = pc20.abs()
        amt_safe = am20.replace(0, np.nan)
        illiq = (abs_ret / amt_safe).mean()
        features["illiq_20"] = illiq.replace([np.inf, -np.inf], np.nan)
    else:
        features["illiq_20"] = pd.Series(np.nan, index=valid_codes)

    # ── 21. mkt_trend [NEW]: market 20d MA / 60d MA ──
    # This is a market-level feature, same value for all stocks on a given date
    if n_dates >= 60:
        # Use equal-weighted market close as proxy
        mkt_close = close_pivot.mean(axis=1)
        ma20 = mkt_close.iloc[-20:].mean()
        ma60 = mkt_close.iloc[-60:].mean()
        mkt_trend_val = ma20 / ma60 if ma60 > 0 else np.nan
        features["mkt_trend"] = pd.Series(mkt_trend_val, index=valid_codes)
    else:
        features["mkt_trend"] = pd.Series(np.nan, index=valid_codes)

    # ── 22. mkt_vol [NEW]: market 20d realized volatility ──
    if n_dates >= 21:
        mkt_close = close_pivot.mean(axis=1)
        mkt_daily_ret = mkt_close.pct_change().iloc[-20:]
        mkt_vol_val = mkt_daily_ret.std()
        features["mkt_vol"] = pd.Series(mkt_vol_val, index=valid_codes)
    else:
        features["mkt_vol"] = pd.Series(np.nan, index=valid_codes)

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
    """Compute forward return from in-memory bulk_data."""
    _, date_to_rows = _get_bulk_date_index(bulk_data)

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


def compute_forward_return(
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    accessor: DataAccessor,
) -> Optional[pd.Series]:
    """Compute forward return (close-to-close) for label construction."""
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


def rank_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Rank-normalize specified columns to [0, 1] percentile within cross-section."""
    df = df.copy()
    for col in columns:
        s = df[col]
        df[col] = s.rank(pct=True, method="average")
    return df


# ===================================================================
# LightGBM model training
# ===================================================================

def train_lgbm_model(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: Optional[pd.DataFrame] = None,
    val_labels: Optional[pd.Series] = None,
) -> lgb.Booster:
    """Train a LightGBM model for cross-sectional ranking."""
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

    X_train = train_features[FEATURE_NAMES].copy()
    for col in FEATURE_NAMES:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)

    dtrain = lgb.Dataset(X_train, label=train_labels)

    callbacks = [lgb.log_evaluation(period=0)]
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

class LGBMCrossSectionalV5(StrategyBase):
    """
    LightGBM cross-sectional stock selection strategy V5.

    Feature engineering optimization over V4:
    - Removed 5 redundant features (mom_3_1, mom_6_1, ret_5d_std,
      high_low_20, pe_ttm)
    - Added 5 high-value features (ind_rel_mom, mom_sharpe,
      market_beta, mkt_trend, mkt_vol)
    """

    def __init__(
        self,
        train_window_years: int = 3,
        top_pct: float = 0.05,
        max_per_industry: int = 5,
        buffer_sigma: float = 0.3,
        mv_pct_lower: float = 0.0,
        mv_pct_upper: float = 0.85,
        feature_lookback: int = 260,
        backtest_end_date: Optional[str] = None,
    ):
        super().__init__("lgbm_cross_sectional_v5")
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
        self._retrain_interval = 2
        self._call_count = 0
        self._warmup_done = False
        self._bulk_data: Optional[pd.DataFrame] = None
        self._st_codes: Optional[set] = None

    def describe(self) -> str:
        return (
            f"### 策略思路\n\n"
            f"基于 LightGBM 的截面选股策略（V5），在 V4 基础上重点优化特征工程，"
            f"去除冗余因子、引入高信息量新因子。\n\n"
            f"### V5 vs V4 特征工程改进\n\n"
            f"#### 移除的冗余特征（5 个）\n\n"
            f"| 移除特征 | 原因 |\n"
            f"|---------|------|\n"
            f"| `mom_3_1` | 与 `mom_12_1` 高度相关（r>0.7），信息冗余 |\n"
            f"| `mom_6_1` | 与 `mom_12_1` 高度相关（r>0.7），信息冗余 |\n"
            f"| `ret_5d_std` | 与 `rvol_20` 度量同一维度（短期波动），冗余 |\n"
            f"| `high_low_20` | 与 `rvol_20` 度量同一维度（价格波动范围），冗余 |\n"
            f"| `pe_ttm` | 与 `inv_pb` 同为价值因子，信息重叠 |\n\n"
            f"#### 新增特征（5 个）\n\n"
            f"| 新特征 | 大类 | 计算方式 | 预期作用 |\n"
            f"|--------|------|---------|----------|\n"
            f"| `ind_rel_mom` | 动量 | stock_mom - industry_avg_mom | 剥离行业效应，提取个股 alpha |\n"
            f"| `mom_sharpe` | 动量 | mom_12_1 / rvol_20 | 波动率调整后动量，信噪比更高 |\n"
            f"| `market_beta` | 风险 | 60 日 stock vs market 回归斜率 | 捕捉系统性风险暴露 |\n"
            f"| `mkt_trend` | 市场状态 | 市场 20dMA / 60dMA | 趋势判断，牛熊市区分 |\n"
            f"| `mkt_vol` | 市场状态 | 市场 20 日波动率 | 波动率状态，高低波切换 |\n\n"
            f"### 完整 22 个特征\n\n"
            f"| 编号 | 特征名 | 大类 | 说明 |\n"
            f"|------|--------|------|------|\n"
            f"| 1 | mom_12_1 | 动量 | 12 个月动量（跳过最近 1 个月） |\n"
            f"| 2 | ind_rel_mom | 动量 | 🆕 行业相对动量 |\n"
            f"| 3 | mom_sharpe | 动量 | 🆕 波动率调整动量 |\n"
            f"| 4 | rev_10 | 反转 | 10 日短期反转 |\n"
            f"| 5 | rvol_20 | 风险 | 20 日已实现波动率 |\n"
            f"| 6 | market_beta | 风险 | 🆕 60 日市场 Beta |\n"
            f"| 7 | vol_confirm | 量价 | 20 日量价相关性 |\n"
            f"| 8 | turnover_20 | 量价 | 20 日平均换手率 |\n"
            f"| 9 | volume_chg | 量价 | 成交量变化比（20d/60d） |\n"
            f"| 10 | inv_pb | 价值 | 1/PB |\n"
            f"| 11 | log_cap | 规模 | log(流通市值) |\n"
            f"| 12 | roe_ttm | 质量 | 净资产收益率 |\n"
            f"| 13 | roa_ttm | 质量 | 总资产收益率 |\n"
            f"| 14 | gross_margin | 质量 | 毛利率 |\n"
            f"| 15 | low_leverage | 质量 | 低杠杆（-资产负债率） |\n"
            f"| 16 | growth_revenue | 成长 | 营收同比增速 |\n"
            f"| 17 | growth_profit | 成长 | 营业利润同比增速 |\n"
            f"| 18 | dv_ttm | 分红 | 股息率 TTM |\n"
            f"| 19 | close_to_high_60 | 技术 | 收盘价 / 60 日最高价 |\n"
            f"| 20 | illiq_20 | 流动性 | 20 日 Amihud 非流动性 |\n"
            f"| 21 | mkt_trend | 市场状态 | 🆕 市场趋势（20dMA/60dMA） |\n"
            f"| 22 | mkt_vol | 市场状态 | 🆕 市场波动率 |\n\n"
            f"### 其他参数（与 V4 一致）\n\n"
            f"| 参数 | 值 |\n"
            f"|------|----|\n"
            f"| 训练窗口 | {self.train_window_years} 年滚动 |\n"
            f"| 选股比例 | 前 {self.top_pct*100:.0f}% |\n"
            f"| 行业约束 | 每行业最多 {self.max_per_industry} 只 |\n"
            f"| 持仓缓冲 | +{self.buffer_sigma}σ |\n"
            f"| 市值范围 | 前 {self.mv_pct_upper*100:.0f}% |\n"
            f"| LightGBM | num_leaves=31, lr=0.05, depth=4 |\n\n"
            f"### 优化预期\n\n"
            f"- 去冗余后模型更不容易过拟合，泛化能力提升\n"
            f"- 行业相对动量剥离行业 beta，提取纯个股 alpha\n"
            f"- 波动率调整动量提高信噪比，减少高波动噪声干扰\n"
            f"- 市场 beta 帮助模型感知系统性风险暴露\n"
            f"- 市场状态因子让模型区分牛熊市，自适应调整选股偏好"
        )

    def _should_retrain(self) -> bool:
        if self._model is None:
            return True
        return self._call_count % self._retrain_interval == 0

    def _warmup_training_cache(
        self,
        current_date: pd.Timestamp,
        accessor: DataAccessor,
    ):
        """Pre-compute features and labels for historical dates."""
        warmup_start = current_date - pd.DateOffset(
            years=self.train_window_years, months=2
        )
        warmup_end = current_date - pd.DateOffset(days=1)

        print(f"      [预热] 加载历史训练数据 "
              f"{warmup_start.strftime('%Y-%m-%d')} ~ "
              f"{warmup_end.strftime('%Y-%m-%d')} ...")

        # Cache ST stock codes
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

        # Bulk-load ALL data
        data_start = warmup_start - pd.DateOffset(days=int(self.feature_lookback * 1.8))
        bulk = prefetch_bulk_data(accessor, data_start, warmup_end, FEATURE_COLUMNS)

        if bulk.empty:
            print(f"      [预热] ✗ 未找到历史数据")
            return

        # Also prefetch backtest period data
        if self._backtest_end_date is not None:
            backtest_end = self._backtest_end_date + pd.DateOffset(months=1)
        else:
            backtest_end = current_date + pd.DateOffset(years=4)
        bulk_backtest = prefetch_bulk_data(accessor, warmup_end, backtest_end, FEATURE_COLUMNS)
        if not bulk_backtest.empty:
            self._bulk_data = pd.concat([bulk, bulk_backtest], ignore_index=True)
            self._bulk_data.drop_duplicates(
                subset=["ts_code", "trade_date"], keep="last", inplace=True
            )
            self._bulk_data.sort_values(["trade_date", "ts_code"], inplace=True)
            self._bulk_data.reset_index(drop=True, inplace=True)
        else:
            self._bulk_data = bulk

        _bulk_date_index_cache.clear()

        # Determine biweekly rebalance dates
        hist_trade_dates_raw = bulk["trade_date"].unique()
        hist_trade_dates_raw = np.sort(hist_trade_dates_raw)
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

        # Cache the LAST date
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

        # Prune old data
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
        """Select stocks: top 5% by ML score, max per industry, with buffer."""
        df = scores_df.copy()

        if current_holdings and self.buffer_sigma > 0:
            score_std = df["ml_score"].std()
            if score_std > 0:
                held_codes = set(current_holdings.keys())
                boost = self.buffer_sigma * score_std
                df.loc[df["ts_code"].isin(held_codes), "ml_score"] += boost

        df = df.sort_values("ml_score", ascending=False).reset_index(drop=True)

        n_select = max(1, int(len(df) * self.top_pct))
        candidates = df.head(n_select).copy()

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
        print(f"\n      ── V5策略第 {self._call_count} 期  {date_str} ──")

        # Step 0: Warmup
        if not self._warmup_done:
            self._warmup_done = True
            self._warmup_training_cache(date, accessor)

        # Step 1: Compute features
        print(f"      [特征] 计算 {len(FEATURE_NAMES)} 个V5特征 (回看 {self.feature_lookback} 天) ...")
        if self._bulk_data is not None:
            feat_df = compute_features_from_memory(
                date, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
        else:
            # Fallback: should not happen in normal flow
            print(f"      [特征] ⚠ 无预取数据，本期跳过")
            return {}
        if feat_df is None or feat_df.empty:
            print(f"      [特征] ✗ 数据不足，本期跳过")
            return {}

        # Step 2: Market cap filter
        n_before_mv = len(feat_df)
        feat_df = self._apply_market_cap_filter(feat_df)
        print(f"      [特征] ✓ 有效股票 {n_before_mv} 只 → 市值过滤后 {len(feat_df)} 只")
        if len(feat_df) < 50:
            print(f"      [特征] ✗ 过滤后不足 50 只，本期跳过")
            return {}

        # Step 3: Rank-normalize
        feat_ranked = rank_normalize(feat_df, FEATURE_NAMES)
        print(f"      [特征] ✓ 排序归一化完成")

        # Step 4: Update training cache with previous period's label
        if self._train_data_cache:
            last_cached = self._train_data_cache[-1]
            last_date = pd.Timestamp(last_cached[0])
            if last_date < date:
                if self._bulk_data is not None:
                    fwd_ret = compute_forward_return_from_memory(
                        last_date, date, self._bulk_data
                    )
                else:
                    fwd_ret = compute_forward_return(last_date, date, accessor)
                if fwd_ret is not None:
                    self._train_data_cache[-1] = (
                        last_cached[0], last_cached[1], fwd_ret
                    )

        # Step 5: Cache current features
        date_str = date.strftime("%Y-%m-%d")
        cached_dates = {d for d, _, _ in self._train_data_cache}
        if date_str not in cached_dates:
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
            valid_cache = [
                (d, f, l) for d, f, l in self._train_data_cache
                if not l.empty and pd.Timestamp(d) < date
            ]
            if len(valid_cache) >= 4:
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

                if len(all_X) >= 4:
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
                else:
                    print(f"      [训练] ✗ 合并后训练期不足 (需要 ≥4, 当前 {len(all_X)})")

        # Step 7: Predict
        if self._model is None:
            scores = feat_ranked[FEATURE_NAMES].mean(axis=1)
            result_df = feat_ranked[["ts_code", "sw_l1"]].copy()
            result_df["ml_score"] = scores.values
            print(f"      [预测] 冷启动模式 (等权因子均值)，候选 {len(result_df)} 只")
        else:
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

        # Step 8: Select stocks
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
        start_date="2018-01-01",
        end_date="2025-12-31",
        rebalance_freq="BW",
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir="data/quant/backtest",
    )

    strategy = LGBMCrossSectionalV5(
        train_window_years=3,
        top_pct=0.05,
        max_per_industry=5,
        buffer_sigma=0.3,
        mv_pct_upper=0.85,
        feature_lookback=260,
        backtest_end_date=cfg.end_date,
    )
    result = run_backtest(strategy, cfg)
