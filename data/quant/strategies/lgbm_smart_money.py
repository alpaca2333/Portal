"""
LightGBM Smart Money Tracking Strategy
=======================================

Motivation
----------
Detect institutional / smart money accumulation patterns by analyzing
volume-price divergence, shrinking pullbacks, volatility compression,
and chip concentration signals. The strategy aims to enter AFTER the
accumulation phase and BEFORE the markup phase.

Core Design Choices
-------------------
1. **Smart money features**: 10 new features specifically designed to
   detect accumulation (bottom volume surge, OBV slope divergence,
   shrink pullback ratio, volatility compression, MA convergence,
   lower shadow ratio, money flow strength, bottom deviation,
   turnover concentration, Amihud illiquidity change).
2. **Retained base features**: 12 features from V4 for context
   (momentum, value, quality, growth, risk).
3. **Signal-strength filtering**: Instead of fixed top-N%, use absolute
   ML score threshold + multi-signal confirmation. "Rather miss than
   misjudge" — only trade stocks with clear signals.
4. **Weekly rebalancing**: Smart money operates on shorter cycles than
   monthly; weekly frequency captures the transition from accumulation
   to markup more precisely.
5. **Dynamic position sizing**: Fewer stocks when signals are weak,
   more when signals are strong. Portfolio size ranges from 0 to ~30.

Usage
-----
cd <project_root>
python -m data.quant.strategies.lgbm_smart_money
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, List, Optional, Tuple
from pathlib import Path
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


# ===================================================================
# Feature engineering
# ===================================================================

# All DB columns needed by features
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

# Feature names: 10 smart-money + 12 base context features = 22 total
FEATURE_NAMES = [
    # ── Smart money accumulation features (10) ──
    "vol_surge_ratio",       # bottom volume surge: vol_5d / vol_60d
    "shrink_pullback",       # shrink pullback: down-day vol / up-day vol
    "vol_compression",       # volatility compression: vol_10d / vol_60d
    "ma_convergence",        # MA convergence: std(MA5,MA10,MA20,MA60)/close
    "obv_slope",             # OBV 20d linear regression slope
    "lower_shadow_ratio",    # lower shadow ratio (20d avg)
    "money_flow_strength",   # up-day amount / down-day amount (20d)
    "bottom_deviation",      # close / 120d low - 1
    "turnover_concentration",# turnover rate CV (20d) — lower = more locked
    "illiq_change",          # Amihud illiq 10d / 60d — rising = chips locked

    # ── Base context features (12) ──
    "mom_12_1",              # 12-month momentum, skip 1 month
    "rev_10",                # 10-day reversal
    "rvol_20",               # 20-day realized volatility
    "vol_confirm",           # volume-price correlation (20d)
    "inv_pb",                # 1/PB (value)
    "log_cap",               # log(circulating market value)
    "roe_ttm",               # ROE
    "turnover_20",           # 20-day average turnover rate
    "volume_chg",            # volume change ratio (20d vs 60d)
    "close_to_high_60",      # close / 60-day high
    "growth_revenue",        # revenue YoY growth
    "growth_profit",         # operating profit YoY growth
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


# Pre-computed date index for bulk_data
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


def compute_features_from_memory(
    date: pd.Timestamp,
    bulk_data: pd.DataFrame,
    lookback: int = 260,
    st_codes: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute all 22 features (10 smart-money + 12 base) from in-memory data.
    """
    all_dates, date_to_rows = _get_bulk_date_index(bulk_data)
    date_ts = pd.Timestamp(date)

    valid_dates = all_dates[all_dates <= date_ts]
    if len(valid_dates) < 60:
        return None
    window_dates = valid_dates[-lookback:] if len(valid_dates) >= lookback else valid_dates

    # Gather rows for window dates
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
        ["ts_code", "close", "pb", "pe_ttm", "circ_mv",
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
    pivot_cols = ["close", "vol", "high", "low", "turnover_rate_f", "amount", "pct_chg"]
    pivots = {}
    for col in pivot_cols:
        if col in w_indexed.columns:
            pivots[col] = w_indexed[col].unstack("ts_code").sort_index()

    close_pivot = pivots["close"]
    vol_pivot = pivots.get("vol")
    high_pivot = pivots.get("high")
    low_pivot = pivots.get("low")
    turn_pivot = pivots.get("turnover_rate_f")
    amount_pivot = pivots.get("amount")
    pctchg_pivot = pivots.get("pct_chg")

    n_dates = len(close_pivot)
    features = {}
    valid_codes = close_pivot.columns.tolist()
    snap_indexed = snap.set_index("ts_code")

    # ================================================================
    # SMART MONEY FEATURES (10)
    # ================================================================

    # --- 1. vol_surge_ratio: bottom volume surge (vol_5d / vol_60d) ---
    if n_dates >= 60 and vol_pivot is not None:
        vol_5 = vol_pivot.iloc[-5:].mean()
        vol_60 = vol_pivot.iloc[-60:].mean()
        features["vol_surge_ratio"] = (vol_5 / vol_60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["vol_surge_ratio"] = pd.Series(np.nan, index=valid_codes)

    # --- 2. shrink_pullback: down-day vol / up-day vol (20d) ---
    if n_dates >= 20 and vol_pivot is not None and pctchg_pivot is not None:
        ret_20 = pctchg_pivot.iloc[-20:]
        v_20 = vol_pivot.iloc[-20:]
        # Align indices
        common_idx = ret_20.index.intersection(v_20.index)
        ret_20 = ret_20.loc[common_idx]
        v_20 = v_20.loc[common_idx]

        shrink_vals = {}
        for code in valid_codes:
            if code in ret_20.columns and code in v_20.columns:
                r = ret_20[code].dropna()
                v = v_20[code].dropna()
                ci = r.index.intersection(v.index)
                r, v = r.loc[ci], v.loc[ci]
                up_vol = v[r > 0].mean()
                down_vol = v[r < 0].mean()
                if pd.notna(up_vol) and up_vol > 0 and pd.notna(down_vol):
                    shrink_vals[code] = down_vol / up_vol
        features["shrink_pullback"] = pd.Series(shrink_vals)
    else:
        features["shrink_pullback"] = pd.Series(np.nan, index=valid_codes)

    # --- 3. vol_compression: realized vol 10d / 60d ---
    if n_dates >= 61:
        ret_10 = close_pivot.iloc[-11:].pct_change(fill_method=None).iloc[1:]
        ret_60 = close_pivot.iloc[-61:].pct_change(fill_method=None).iloc[1:]
        std_10 = ret_10.std()
        std_60 = ret_60.std()
        features["vol_compression"] = (std_10 / std_60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["vol_compression"] = pd.Series(np.nan, index=valid_codes)

    # --- 4. ma_convergence: std(MA5, MA10, MA20, MA60) / close ---
    if n_dates >= 60:
        ma5 = close_pivot.iloc[-5:].mean()
        ma10 = close_pivot.iloc[-10:].mean()
        ma20 = close_pivot.iloc[-20:].mean()
        ma60 = close_pivot.iloc[-60:].mean()
        ma_stack = pd.DataFrame({"ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60})
        ma_std = ma_stack.std(axis=1)
        last_close = close_pivot.iloc[-1]
        features["ma_convergence"] = (ma_std / last_close.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["ma_convergence"] = pd.Series(np.nan, index=valid_codes)

    # --- 5. obv_slope: OBV 20d linear regression slope ---
    if n_dates >= 20 and vol_pivot is not None and pctchg_pivot is not None:
        ret_20 = pctchg_pivot.iloc[-20:]
        v_20 = vol_pivot.iloc[-20:]
        common_idx = ret_20.index.intersection(v_20.index)
        ret_20 = ret_20.loc[common_idx]
        v_20 = v_20.loc[common_idx]

        # OBV: cumulative sum of signed volume
        sign_vol = v_20.copy()
        sign_vol[ret_20 > 0] = v_20[ret_20 > 0]
        sign_vol[ret_20 < 0] = -v_20[ret_20 < 0]
        sign_vol[ret_20 == 0] = 0
        obv = sign_vol.cumsum()

        # Linear regression slope for each stock
        x = np.arange(len(obv), dtype=float)
        x_dm = x - x.mean()
        x_var = (x_dm ** 2).sum()
        obv_vals = obv.values
        obv_mean = np.nanmean(obv_vals, axis=0)
        obv_dm = obv_vals - obv_mean
        slopes = {}
        for j, code in enumerate(obv.columns):
            y = obv_dm[:, j]
            mask = ~np.isnan(y)
            if mask.sum() >= 10:
                slopes[code] = (x_dm[mask] * y[mask]).sum() / (x_dm[mask] ** 2).sum()
        features["obv_slope"] = pd.Series(slopes)
    else:
        features["obv_slope"] = pd.Series(np.nan, index=valid_codes)

    # --- 6. lower_shadow_ratio: avg (close - low) / (high - low) over 20d ---
    if n_dates >= 20 and high_pivot is not None and low_pivot is not None:
        c_20 = close_pivot.iloc[-20:]
        h_20 = high_pivot.iloc[-20:]
        l_20 = low_pivot.iloc[-20:]
        hl_range = (h_20 - l_20).replace(0, np.nan)
        lower_shadow = (c_20 - l_20) / hl_range
        features["lower_shadow_ratio"] = lower_shadow.mean().replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["lower_shadow_ratio"] = pd.Series(np.nan, index=valid_codes)

    # --- 7. money_flow_strength: up-day amount / down-day amount (20d) ---
    if n_dates >= 20 and amount_pivot is not None and pctchg_pivot is not None:
        ret_20 = pctchg_pivot.iloc[-20:]
        amt_20 = amount_pivot.iloc[-20:]
        common_idx = ret_20.index.intersection(amt_20.index)
        ret_20 = ret_20.loc[common_idx]
        amt_20 = amt_20.loc[common_idx]

        mf_vals = {}
        for code in valid_codes:
            if code in ret_20.columns and code in amt_20.columns:
                r = ret_20[code].dropna()
                a = amt_20[code].dropna()
                ci = r.index.intersection(a.index)
                r, a = r.loc[ci], a.loc[ci]
                up_amt = a[r > 0].sum()
                down_amt = a[r < 0].sum()
                if down_amt > 0:
                    mf_vals[code] = up_amt / down_amt
        features["money_flow_strength"] = pd.Series(mf_vals)
    else:
        features["money_flow_strength"] = pd.Series(np.nan, index=valid_codes)

    # --- 8. bottom_deviation: close / 120d low - 1 ---
    if n_dates >= 120 and low_pivot is not None:
        low_120 = low_pivot.iloc[-120:].min()
        last_close = close_pivot.iloc[-1]
        features["bottom_deviation"] = (
            (last_close / low_120.replace(0, np.nan)) - 1
        ).replace([np.inf, -np.inf], np.nan)
    else:
        features["bottom_deviation"] = pd.Series(np.nan, index=valid_codes)

    # --- 9. turnover_concentration: CV of turnover_rate (20d) ---
    if n_dates >= 20 and turn_pivot is not None:
        t_20 = turn_pivot.iloc[-20:]
        t_mean = t_20.mean()
        t_std = t_20.std()
        features["turnover_concentration"] = (
            t_std / t_mean.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
    else:
        features["turnover_concentration"] = pd.Series(np.nan, index=valid_codes)

    # --- 10. illiq_change: Amihud illiq 10d / 60d ---
    if n_dates >= 60 and pctchg_pivot is not None and amount_pivot is not None:
        def _amihud(pct_p, amt_p, n):
            pct_n = pct_p.iloc[-n:]
            amt_n = amt_p.iloc[-n:]
            ci = pct_n.index.intersection(amt_n.index)
            pct_n = pct_n.loc[ci]
            amt_n = amt_n.loc[ci]
            return (pct_n.abs() / amt_n.replace(0, np.nan)).mean()

        illiq_10 = _amihud(pctchg_pivot, amount_pivot, 10)
        illiq_60 = _amihud(pctchg_pivot, amount_pivot, 60)
        features["illiq_change"] = (
            illiq_10 / illiq_60.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
    else:
        features["illiq_change"] = pd.Series(np.nan, index=valid_codes)

    # ================================================================
    # BASE CONTEXT FEATURES (12)
    # ================================================================

    # --- 11. mom_12_1 ---
    if n_dates >= 240:
        features["mom_12_1"] = _compute_momentum(close_pivot, 240, 20)
    else:
        features["mom_12_1"] = pd.Series(np.nan, index=valid_codes)

    # --- 12. rev_10 ---
    if n_dates >= 10:
        features["rev_10"] = _compute_momentum(close_pivot, 10, 0)
    else:
        features["rev_10"] = pd.Series(np.nan, index=valid_codes)

    # --- 13. rvol_20 ---
    if n_dates >= 21:
        daily_ret = close_pivot.iloc[-21:].pct_change(fill_method=None).iloc[1:]
        features["rvol_20"] = daily_ret.std()
    else:
        features["rvol_20"] = pd.Series(np.nan, index=valid_codes)

    # --- 14. vol_confirm ---
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

    # --- 15. inv_pb ---
    pb = snap_indexed["pb"]
    features["inv_pb"] = pd.Series(
        np.where(pb > 0, 1.0 / pb, np.nan), index=snap_indexed.index
    )

    # --- 16. log_cap ---
    mv = snap_indexed["circ_mv"]
    features["log_cap"] = pd.Series(
        np.where(mv > 0, np.log(mv), np.nan), index=snap_indexed.index
    )

    # --- 17. roe_ttm ---
    features["roe_ttm"] = snap_indexed["roe"]

    # --- 18. turnover_20 ---
    if n_dates >= 20 and turn_pivot is not None:
        features["turnover_20"] = turn_pivot.iloc[-20:].mean()
    else:
        features["turnover_20"] = pd.Series(np.nan, index=valid_codes)

    # --- 19. volume_chg ---
    if n_dates >= 60 and vol_pivot is not None:
        vol_20 = vol_pivot.iloc[-20:].mean()
        vol_60 = vol_pivot.iloc[-60:].mean()
        features["volume_chg"] = (vol_20 / vol_60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["volume_chg"] = pd.Series(np.nan, index=valid_codes)

    # --- 20. close_to_high_60 ---
    if n_dates >= 60 and high_pivot is not None:
        h60 = high_pivot.iloc[-60:].max()
        last_close = close_pivot.iloc[-1]
        features["close_to_high_60"] = (last_close / h60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        features["close_to_high_60"] = pd.Series(np.nan, index=valid_codes)

    # --- 21. growth_revenue ---
    features["growth_revenue"] = snap_indexed["tr_yoy"]

    # --- 22. growth_profit ---
    features["growth_profit"] = snap_indexed["op_yoy"]

    # ================================================================
    # Combine all features
    # ================================================================
    feat_df = pd.DataFrame(features)
    feat_df.index.name = "ts_code"
    feat_df = feat_df.reset_index()

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


def rank_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Rank-normalize specified columns to [0, 1] percentile."""
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
        "max_depth": 4,
        "min_child_samples": 100,
        "lambda_l1": 0.5,
        "lambda_l2": 5.0,
        "verbose": -1,
        "seed": 42,
    }

    X_train = train_features[FEATURE_NAMES].copy()
    for col in FEATURE_NAMES:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)

    dtrain = lgb.Dataset(X_train, label=train_labels)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=0)]

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

class LGBMSmartMoney(StrategyBase):
    """
    LightGBM smart money tracking strategy.

    Detects institutional accumulation patterns and enters positions
    only when multiple signals confirm. Uses absolute score thresholds
    instead of fixed top-N% to implement "rather miss than misjudge".
    """

    def __init__(
        self,
        train_window_years: int = 5,
        score_quantile: float = 0.85,
        min_signal_count: int = 3,
        max_per_industry: int = 3,
        max_positions: int = 30,
        buffer_sigma: float = 0.3,
        mv_pct_lower: float = 0.0,
        mv_pct_upper: float = 0.85,
        feature_lookback: int = 260,
        backtest_end_date: Optional[str] = None,
    ):
        super().__init__("lgbm_smart_money")
        self.train_window_years = train_window_years
        self.score_quantile = score_quantile  # ML score must be above this quantile
        self.min_signal_count = min_signal_count  # min smart-money signals to confirm
        self.max_per_industry = max_per_industry
        self.max_positions = max_positions
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
        self._retrain_interval = 2  # retrain every 2 bi-weeks (~monthly)
        self._call_count = 0
        self._warmup_done = False
        self._bulk_data: Optional[pd.DataFrame] = None
        self._st_codes: Optional[set] = None
        self._confidence_scale = 1.0  # position scale factor (reduced when data is thin)
        self._factor_ic_weights: Optional[pd.Series] = None  # IC-weighted fallback

    def describe(self) -> str:
        return (
            f"### 策略思路\n\n"
            f"基于 LightGBM 的大资金追踪选股策略，通过分析量价异动、筹码集中度、"
            f"洗盘特征等信号，识别机构/主力资金的吸筹行为，在拉升前介入。\n\n"
            f"### 核心设计\n\n"
            f"1. **大资金追踪特征**：10 个专门设计的吸筹/洗盘识别因子\n"
            f"2. **宁缺毋滥**：ML 分数必须超过 {self.score_quantile*100:.0f}% 分位 + "
            f"至少 {self.min_signal_count} 个大资金信号共振确认\n"
            f"3. **动态仓位**：信号弱时少持甚至空仓，信号强时满配（最多 {self.max_positions} 只）\n"
            f"4. **周频调仓**：捕捉吸筹→拉升的短周期转换\n"
            f"5. **行业分散**：每行业最多 {self.max_per_industry} 只\n\n"
            f"### 22 个特征\n\n"
            f"| 编号 | 特征名 | 大类 | 说明 |\n"
            f"|------|--------|------|------|\n"
            f"| 1 | vol_surge_ratio | 吸筹 | 底部放量比（5日/60日均量） |\n"
            f"| 2 | shrink_pullback | 洗盘 | 缩量回调比（跌日量/涨日量） |\n"
            f"| 3 | vol_compression | 蓄势 | 波动率压缩（10日/60日波动率） |\n"
            f"| 4 | ma_convergence | 蓄势 | 均线粘合度 |\n"
            f"| 5 | obv_slope | 吸筹 | OBV 20日斜率（量能趋势） |\n"
            f"| 6 | lower_shadow_ratio | 吸筹 | 下影线比例（20日均值） |\n"
            f"| 7 | money_flow_strength | 吸筹 | 资金流强度（涨日/跌日成交额） |\n"
            f"| 8 | bottom_deviation | 位置 | 底部偏离度（距120日低点） |\n"
            f"| 9 | turnover_concentration | 筹码 | 换手率集中度（CV越低=筹码越锁定） |\n"
            f"| 10 | illiq_change | 筹码 | 流动性变化（Amihud 10日/60日） |\n"
            f"| 11 | mom_12_1 | 动量 | 12个月动量（跳过1个月） |\n"
            f"| 12 | rev_10 | 反转 | 10日短期反转 |\n"
            f"| 13 | rvol_20 | 风险 | 20日已实现波动率 |\n"
            f"| 14 | vol_confirm | 量价 | 20日量价相关性 |\n"
            f"| 15 | inv_pb | 价值 | 1/PB |\n"
            f"| 16 | log_cap | 规模 | log(流通市值) |\n"
            f"| 17 | roe_ttm | 质量 | 净资产收益率 |\n"
            f"| 18 | turnover_20 | 量价 | 20日平均换手率 |\n"
            f"| 19 | volume_chg | 量价 | 成交量变化比（20d/60d） |\n"
            f"| 20 | close_to_high_60 | 技术 | 收盘价/60日最高价 |\n"
            f"| 21 | growth_revenue | 成长 | 营收同比增速 |\n"
            f"| 22 | growth_profit | 成长 | 营业利润同比增速 |\n\n"
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
            f"### 选股逻辑（宁缺毋滥）\n\n"
            f"1. ML 分数必须超过截面 {self.score_quantile*100:.0f}% 分位数\n"
            f"2. 10 个大资金因子中，至少 {self.min_signal_count} 个处于前 30% 分位\n"
            f"3. 每行业最多 {self.max_per_industry} 只，总持仓最多 {self.max_positions} 只\n"
            f"4. 不满足条件时宁可空仓\n\n"
            f"### 已知局限\n\n"
            f"- 已过滤 ST / *ST 股票（基于 stock_info.name 静态名称匹配）\n"
            f"- 无 Level2 数据，大单识别依赖量价间接推断\n"
            f"- 等权分配，未按信号强度加权"
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

        # Determine weekly rebalance dates within warmup window
        hist_trade_dates_raw = bulk["trade_date"].unique()
        hist_trade_dates_raw = np.sort(hist_trade_dates_raw)
        mask = (hist_trade_dates_raw >= pd.Timestamp(warmup_start)) & \
               (hist_trade_dates_raw <= pd.Timestamp(warmup_end))
        hist_trade_dates = pd.DatetimeIndex(hist_trade_dates_raw[mask])

        if len(hist_trade_dates) == 0:
            print(f"      [预热] ✗ 历史交易日为空")
            return

        # Weekly: group by ISO week, take last trading day of each week
        s = pd.Series(hist_trade_dates, index=hist_trade_dates)
        week_ids = s.index.isocalendar().year * 100 + s.index.isocalendar().week
        groups = s.groupby(week_ids.values)
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

        # ── Pre-train: build initial model so first backtest period uses ML ──
        self._pretrain_initial_model(current_date)

        # ── Compute factor IC weights for fallback scoring ──
        self._compute_factor_ic_weights(current_date)

    def _pretrain_initial_model(self, current_date: pd.Timestamp):
        """
        Train an initial LightGBM model at the end of warmup, so the very
        first backtest period uses ML predictions instead of naive averaging.
        """
        valid_cache = [
            (d, f, l) for d, f, l in self._train_data_cache
            if not l.empty and pd.Timestamp(d) < current_date
        ]
        n_valid = len(valid_cache)
        min_required = 20  # need at least 20 weekly periods (~5 months)

        if n_valid < 8:
            print(f"      [预训练] ✗ 有标签期数不足 ({n_valid} 期, 需要 ≥8)，"
                  f"首期将使用 IC 加权 fallback")
            self._confidence_scale = 0.5  # half position when no model
            return

        # Determine confidence based on data richness
        if n_valid < min_required:
            self._confidence_scale = 0.5 + 0.5 * (n_valid / min_required)
            print(f"      [预训练] ⚠ 训练数据偏少 ({n_valid} 期)，"
                  f"仓位系数={self._confidence_scale:.2f}")
        else:
            self._confidence_scale = 1.0

        # Build training set
        all_X, all_y = [], []
        train_start = current_date - pd.DateOffset(years=self.train_window_years)
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

        if len(all_X) < 8:
            print(f"      [预训练] ✗ 合并后训练期不足 ({len(all_X)} 期)")
            return

        n = len(all_X)
        split = max(1, int(n * 0.8))
        train_X = pd.concat(all_X[:split])
        train_y = pd.concat(all_y[:split])
        val_X = pd.concat(all_X[split:]) if split < n else None
        val_y = pd.concat(all_y[split:]) if split < n else None

        try:
            self._model = train_lgbm_model(train_X, train_y, val_X, val_y)
            self._last_train_date = current_date
            n_train = len(train_y)
            n_val = len(val_y) if val_y is not None else 0
            best_iter = self._model.best_iteration if hasattr(
                self._model, 'best_iteration') else '?'
            print(
                f"      [预训练] ✓ 初始模型训练完成  "
                f"训练={n_train:,} 样本  验证={n_val:,} 样本  "
                f"使用 {len(all_X)} 期历史  "
                f"最优轮次={best_iter}  "
                f"仓位系数={self._confidence_scale:.2f}"
            )
        except Exception as e:
            print(f"      [预训练] ✗ 训练失败: {e}")
            self._confidence_scale = 0.5

    def _compute_factor_ic_weights(self, current_date: pd.Timestamp):
        """
        Compute Information Coefficient (rank correlation) for each factor
        using historical data. Used as fallback scoring when model is unavailable.
        """
        valid_cache = [
            (d, f, l) for d, f, l in self._train_data_cache
            if not l.empty and pd.Timestamp(d) < current_date
        ]
        if len(valid_cache) < 5:
            print(f"      [IC权重] ✗ 数据不足，使用等权 fallback")
            self._factor_ic_weights = None
            return

        # Compute rank IC for each factor across all periods
        ic_records = {f: [] for f in FEATURE_NAMES}
        for d_str, f_df, labels in valid_cache[-52:]:  # use last ~1 year
            merged = f_df.set_index("ts_code").join(
                labels.rename("label"), how="inner"
            )
            if len(merged) < 50:
                continue
            label_rank = merged["label"].rank(pct=True)
            for feat in FEATURE_NAMES:
                if feat in merged.columns:
                    feat_rank = merged[feat].rank(pct=True)
                    valid_mask = feat_rank.notna() & label_rank.notna()
                    if valid_mask.sum() >= 30:
                        ic = feat_rank[valid_mask].corr(label_rank[valid_mask])
                        if pd.notna(ic):
                            ic_records[feat].append(ic)

        # Average IC per factor, clip negatives to 0
        avg_ic = {}
        for feat, ics in ic_records.items():
            if len(ics) >= 3:
                avg_ic[feat] = max(0.0, np.mean(ics))
            else:
                avg_ic[feat] = 0.0

        ic_series = pd.Series(avg_ic)
        ic_sum = ic_series.sum()
        if ic_sum > 0:
            self._factor_ic_weights = ic_series / ic_sum
            top3 = ic_series.nlargest(3)
            print(
                f"      [IC权重] ✓ 计算完成，"
                f"有效因子 {(ic_series > 0).sum()}/{len(FEATURE_NAMES)} 个  "
                f"Top3: {', '.join(f'{k}={v:.3f}' for k, v in top3.items())}"
            )
        else:
            self._factor_ic_weights = None
            print(f"      [IC权重] ⚠ 所有因子 IC ≤ 0，使用等权 fallback")

    def _apply_market_cap_filter(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to top mv_pct_upper by circulating market value."""
        if "circ_mv" not in feat_df.columns:
            return feat_df
        mv = feat_df["circ_mv"].dropna()
        if len(mv) == 0:
            return feat_df
        lower_bound = mv.quantile(1.0 - self.mv_pct_upper)
        return feat_df[feat_df["circ_mv"] >= lower_bound].copy()

    def _count_smart_money_signals(self, feat_ranked: pd.DataFrame) -> pd.Series:
        """
        Count how many of the 10 smart-money features are in the top 30%
        (i.e., rank-normalized value >= 0.70) for each stock.

        Signal definitions (after rank normalization, higher = stronger signal):
        - vol_surge_ratio >= 0.70: bottom volume surge
        - shrink_pullback <= 0.30: down-day volume much less than up-day (inverted)
        - vol_compression <= 0.30: volatility compressed (inverted)
        - ma_convergence <= 0.30: MAs converging (inverted)
        - obv_slope >= 0.70: OBV trending up
        - lower_shadow_ratio >= 0.70: strong lower shadows (buying support)
        - money_flow_strength >= 0.70: money flowing in
        - bottom_deviation <= 0.40: still near bottom (inverted, slightly relaxed)
        - turnover_concentration <= 0.30: turnover stable = chips locked (inverted)
        - illiq_change >= 0.70: illiquidity rising = chips being locked
        """
        signals = pd.DataFrame(index=feat_ranked.index)

        # Factors where HIGH rank = bullish signal
        high_is_good = ["vol_surge_ratio", "obv_slope", "lower_shadow_ratio",
                        "money_flow_strength", "illiq_change"]
        for col in high_is_good:
            if col in feat_ranked.columns:
                signals[col] = (feat_ranked[col] >= 0.70).astype(int)

        # Factors where LOW rank = bullish signal (inverted)
        low_is_good = ["shrink_pullback", "vol_compression",
                       "ma_convergence", "turnover_concentration"]
        for col in low_is_good:
            if col in feat_ranked.columns:
                signals[col] = (feat_ranked[col] <= 0.30).astype(int)

        # bottom_deviation: slightly relaxed threshold (near bottom but not AT bottom)
        if "bottom_deviation" in feat_ranked.columns:
            signals["bottom_deviation"] = (feat_ranked["bottom_deviation"] <= 0.40).astype(int)

        return signals.sum(axis=1)

    def _select_stocks(
        self,
        scores_df: pd.DataFrame,
        feat_ranked: pd.DataFrame,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Select stocks with signal-strength filtering (rather miss than misjudge).

        Criteria:
        1. ML score must be above score_quantile (e.g., top 15%)
        2. At least min_signal_count smart-money signals must fire
        3. Industry constraint: max N per industry
        4. Total positions capped at max_positions
        """
        df = scores_df.copy()

        # Step 1: ML score threshold
        score_threshold = df["ml_score"].quantile(self.score_quantile)
        df = df[df["ml_score"] >= score_threshold].copy()

        if df.empty:
            return {}

        # Step 2: Smart money signal count filter
        signal_counts = self._count_smart_money_signals(feat_ranked)
        # Map signal counts to scores_df by ts_code
        sc_map = pd.Series(
            signal_counts.values,
            index=feat_ranked["ts_code"].values,
        )
        df["signal_count"] = df["ts_code"].map(sc_map)
        df = df[df["signal_count"] >= self.min_signal_count].copy()

        if df.empty:
            return {}

        # Step 3: Apply holding buffer
        if current_holdings and self.buffer_sigma > 0:
            score_std = df["ml_score"].std()
            if score_std > 0:
                held_codes = set(current_holdings.keys())
                boost = self.buffer_sigma * score_std
                df.loc[df["ts_code"].isin(held_codes), "ml_score"] += boost

        # Sort by ML score descending
        df = df.sort_values("ml_score", ascending=False).reset_index(drop=True)

        # Step 4: Industry constraint + max positions
        selected = []
        industry_count: Dict[str, int] = {}
        for _, row in df.iterrows():
            if len(selected) >= self.max_positions:
                break
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

        # Step 0: Warmup
        if not self._warmup_done:
            self._warmup_done = True
            self._warmup_training_cache(date, accessor)

        # Step 1: Compute features
        print(f"      [特征] 计算 {len(FEATURE_NAMES)} 个特征 (回看 {self.feature_lookback} 天) ...")
        if self._bulk_data is not None:
            feat_df = compute_features_from_memory(
                date, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
        else:
            feat_df = None
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

        # Step 3: Rank-normalize features
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
                    fwd_ret = None
                if fwd_ret is not None:
                    self._train_data_cache[-1] = (
                        last_cached[0], last_cached[1], fwd_ret
                    )

        # Step 5: Cache current features
        date_str_cache = date.strftime("%Y-%m-%d")
        cached_dates = {d for d, _, _ in self._train_data_cache}
        if date_str_cache not in cached_dates:
            self._train_data_cache.append(
                (date_str_cache, feat_ranked.copy(), pd.Series(dtype=float))
            )

        # Step 6: Train or reuse model
        n_cached = len([1 for d, _, l in self._train_data_cache if not l.empty and pd.Timestamp(d) < date])
        min_train_periods = 8  # need more data for weekly frequency
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
            if len(valid_cache) >= 8:
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

                if len(all_X) >= 8:
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
                    print(f"      [训练] ✗ 合并后训练期不足 (需要 ≥8, 当前 {len(all_X)})")

        # Step 7: Predict
        if self._model is None:
            # IC-weighted fallback instead of naive equal-weight mean
            if self._factor_ic_weights is not None:
                weighted_scores = pd.Series(0.0, index=feat_ranked.index)
                for feat, w in self._factor_ic_weights.items():
                    if feat in feat_ranked.columns and w > 0:
                        weighted_scores += feat_ranked[feat].fillna(0.5) * w
                scores = weighted_scores
                fallback_mode = "IC加权"
            else:
                scores = feat_ranked[FEATURE_NAMES].mean(axis=1)
                fallback_mode = "等权均值"
            result_df = feat_ranked[["ts_code", "sw_l1"]].copy()
            result_df["ml_score"] = scores.values
            print(f"      [预测] 冷启动模式 ({fallback_mode})，候选 {len(result_df)} 只")
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

        # Step 8: Signal-strength filtering + select stocks
        signal_counts = self._count_smart_money_signals(feat_ranked)
        n_with_signals = (signal_counts >= self.min_signal_count).sum()
        score_threshold = result_df["ml_score"].quantile(self.score_quantile)
        n_above_score = (result_df["ml_score"] >= score_threshold).sum()
        print(
            f"      [筛选] ML分数阈值={score_threshold:.4f} (>{self.score_quantile*100:.0f}%分位)  "
            f"达标={n_above_score} 只  |  "
            f"大资金信号≥{self.min_signal_count}个: {n_with_signals} 只"
        )

        weights = self._select_stocks(result_df, feat_ranked, current_holdings)

        n_held = len(current_holdings) if current_holdings else 0
        n_industries = result_df[result_df["ts_code"].isin(weights.keys())]["sw_l1"].nunique() if weights else 0
        overlap = len(set(weights.keys()) & set(current_holdings.keys())) if current_holdings and weights else 0

        if len(weights) == 0:
            print(f"      [选股] ⚠ 本期无满足条件的股票，空仓")
        else:
            # Apply confidence scaling (reduced position when data is thin)
            if self._confidence_scale < 1.0:
                weights = {k: v * self._confidence_scale for k, v in weights.items()}
                # Gradually restore confidence as more data accumulates
                self._confidence_scale = min(1.0, self._confidence_scale + 0.02)

            # Show signal count distribution for selected stocks
            selected_signals = signal_counts[feat_ranked["ts_code"].isin(weights.keys())]
            avg_signals = selected_signals.mean()
            scale_info = f"  仓位系数={self._confidence_scale:.2f}" if self._confidence_scale < 1.0 else ""
            print(
                f"      [选股] 入选 {len(weights)} 只 / {n_industries} 个行业  "
                f"(上期持仓 {n_held} 只，留存 {overlap} 只，换手 {n_held + len(weights) - 2 * overlap} 只)  "
                f"平均信号数={avg_signals:.1f}{scale_info}"
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

    strategy = LGBMSmartMoney(
        train_window_years=5,
        score_quantile=0.85,
        min_signal_count=3,
        max_per_industry=3,
        max_positions=30,
        buffer_sigma=0.3,
        mv_pct_upper=0.85,
        feature_lookback=260,
        backtest_end_date=cfg.end_date,
    )
    result = run_backtest(strategy, cfg)
