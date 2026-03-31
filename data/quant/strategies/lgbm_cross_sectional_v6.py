# -*- coding: utf-8 -*-
"""
LightGBM Cross-Sectional Stock Selection Strategy (V6)
======================================================

Based on V4 with two optimization directions:
1. Increase value/fundamental factor weight via two-stage scoring
2. Bias towards mid-to-large cap stocks

Key Changes vs V4
------------------
A. Market cap filter tightened: top 85% → top 70% (exclude small caps)
B. Two-stage scoring: final = α × ML_score + (1-α) × fundamental_score
   - ML score: LightGBM prediction (captures non-linear interactions)
   - Fundamental score: rank-average of inv_pb, roe_ttm, roa_ttm, dv_ttm
   - α = 0.6 (ML 60%, fundamentals 40%)
C. Sqrt(market cap) weighting: replaces equal weight, naturally tilts
   towards larger caps while maintaining diversification

Usage
-----
cd <project_root>
python -m data.quant.strategies.lgbm_cross_sectional_v6
"""
import sys
import os
import warnings
from pathlib import Path

# Fix Windows GBK encoding issue for Unicode output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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


# ===================================================================
# Feature engineering (identical to V4)
# ===================================================================

# All DB columns needed by the 22 features
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

# ── V6: Fundamental factors used for two-stage scoring ──
FUNDAMENTAL_FACTORS = ["inv_pb", "roe_ttm", "roa_ttm", "dv_ttm"]


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


# Pre-computed date index for bulk_data to avoid repeated full-table scans
_bulk_date_index_cache: Dict[int, Tuple[np.ndarray, Dict]] = {}


def _get_bulk_date_index(bulk_data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """
    Build or retrieve a cached date index for bulk_data.
    Returns (sorted_unique_dates_as_Timestamps, {pd.Timestamp: row_indices}).
    """
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
    Compute all 22 features using in-memory bulk_data instead of DB queries.
    """
    all_dates, date_to_rows = _get_bulk_date_index(bulk_data)
    date_ts = pd.Timestamp(date)

    valid_dates = all_dates[all_dates <= date_ts]
    if len(valid_dates) < 60:
        return None
    window_dates = valid_dates[-lookback:] if len(valid_dates) >= lookback else valid_dates

    row_indices = []
    for wd in window_dates:
        if wd in date_to_rows:
            row_indices.append(date_to_rows[wd])
    if not row_indices:
        return None
    all_row_idx = np.concatenate(row_indices)
    window = bulk_data.iloc[all_row_idx]

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

    w = window[window["ts_code"].isin(universe_codes)].copy()
    w.sort_values(["trade_date", "ts_code"], inplace=True)

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

    # --- 16. dv_ttm ---
    features["dv_ttm"] = snap_indexed["dv_ttm"]

    # --- 17. roa_ttm ---
    features["roa_ttm"] = snap_indexed["roa"]

    # --- 18. gross_margin ---
    features["gross_margin"] = snap_indexed["grossprofit_margin"]

    # --- 19. low_leverage ---
    dta = snap_indexed["debt_to_assets"]
    features["low_leverage"] = pd.Series(
        np.where(dta.notna(), -dta, np.nan), index=snap_indexed.index
    )

    # --- 20. growth_revenue ---
    features["growth_revenue"] = snap_indexed["tr_yoy"]

    # --- 21. growth_profit ---
    features["growth_profit"] = snap_indexed["op_yoy"]

    # --- 22. illiq_20 ---
    if n_dates >= 20:
        w_recent = window[window["trade_date"].isin(window_dates[-20:])].copy()
        if "pct_chg" in w_recent.columns and "amount" in w_recent.columns:
            pctchg_pivot = w_recent.pivot_table(
                index="trade_date", columns="ts_code", values="pct_chg"
            ).sort_index()
            amount_pivot = w_recent.pivot_table(
                index="trade_date", columns="ts_code", values="amount"
            ).sort_index()
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
# Label construction
# ===================================================================

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


# ===================================================================
# LightGBM model training (identical to V4)
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
# V6: Fundamental score computation
# ===================================================================

def compute_fundamental_score(feat_ranked: pd.DataFrame) -> pd.Series:
    """
    Compute a composite fundamental score from rank-normalized features.

    Uses the average rank percentile of:
    - inv_pb   (value: higher = cheaper)
    - roe_ttm  (quality: higher = better)
    - roa_ttm  (quality: higher = better)
    - dv_ttm   (dividend: higher = better)

    Returns a Series indexed like feat_ranked with values in [0, 1].
    """
    fund_cols = [c for c in FUNDAMENTAL_FACTORS if c in feat_ranked.columns]
    if not fund_cols:
        return pd.Series(0.5, index=feat_ranked.index)

    # Each column is already rank-normalized to [0, 1]
    fund_score = feat_ranked[fund_cols].mean(axis=1)
    return fund_score


# ===================================================================
# Strategy class
# ===================================================================

class LGBMCrossSectionalV6(StrategyBase):
    """
    LightGBM cross-sectional stock selection strategy (V6).

    Optimizations vs V4:
    A. Market cap filter: top 85% → top 70% (exclude small caps)
    B. Two-stage scoring: α × ML + (1-α) × fundamental
    C. Sqrt(market cap) weighting instead of equal weight
    """

    def __init__(
        self,
        train_window_years: int = 3,
        top_pct: float = 0.05,
        max_per_industry: int = 5,
        buffer_sigma: float = 0.3,
        mv_pct_lower: float = 0.0,
        mv_pct_upper: float = 0.70,       # V6: tightened from 0.85
        feature_lookback: int = 260,
        alpha: float = 0.8,                # V6: ML weight in two-stage scoring
        backtest_end_date: Optional[str] = None,
    ):
        super().__init__("lgbm_cross_sectional_v6")
        self.train_window_years = train_window_years
        self.top_pct = top_pct
        self.max_per_industry = max_per_industry
        self.buffer_sigma = buffer_sigma
        self.mv_pct_lower = mv_pct_lower
        self.mv_pct_upper = mv_pct_upper
        self.feature_lookback = feature_lookback
        self.alpha = alpha                 # V6: ML vs fundamental blend ratio
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
            f"基于 LightGBM 的截面选股策略（V6），在 V4 基础上进行两个方向的优化：\n"
            f"**提高价值基本面因子比重** 和 **偏向中高市值股票**。\n\n"
            f"### V6 vs V4 改进对比\n\n"
            f"| 维度 | V4 原版 | V6 优化版 | 改进理由 |\n"
            f"|------|---------|-----------|----------|\n"
            f"| 选股池 | 流通市值前 85% | 流通市值前 70% | 剔除小盘股，降低波动 |\n"
            f"| 打分方式 | 纯 ML score | {self.alpha:.0%} ML + {1-self.alpha:.0%} 基本面 | 提高价值因子话语权 |\n"
            f"| 权重分配 | 等权 | √市值 加权 | 大盘股自然获得更高仓位 |\n"
            f"| 基本面分数 | 无 | inv_pb + roe_ttm + roa_ttm + dv_ttm 排名均值 | 价值+质量+分红综合 |\n\n"
            f"### 优化方向详解\n\n"
            f"#### 方向一：提高价值基本面因子比重\n\n"
            f"V4 中价值因子（inv_pb + pe_ttm）gain 占比仅 5.32%，质量因子（ROE/ROA/毛利率/杠杆）"
            f"合计仅 4.15%，模型被 turnover_20（31.94%）主导，基本面因子被淹没。\n\n"
            f"V6 采用**两阶段打分**方案：\n"
            f"- ML 分数（60%）：保留 LightGBM 学到的量价/风险非线性模式\n"
            f"- 基本面分数（40%）：inv_pb、roe_ttm、roa_ttm、dv_ttm 的排名均值\n"
            f"- 最终分数 = {self.alpha} × ML_score + {1-self.alpha} × fundamental_score\n\n"
            f"#### 方向二：偏向中高市值股票\n\n"
            f"V4 选股池为市值前 85%，且等权分配导致小盘股与大盘股仓位相同。\n\n"
            f"V6 双管齐下：\n"
            f"- 市值下限收紧至前 70%，直接剔除小盘股\n"
            f"- 权重改为 √(流通市值) 加权，大盘股自然获得更高仓位\n\n"
            f"### 22 个特征（与 V4 相同）\n\n"
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
            f"### LightGBM 超参数（与 V4 相同）\n\n"
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
            f"### V6 新增参数\n\n"
            f"| 参数 | 值 | 说明 |\n"
            f"|------|----|------|\n"
            f"| mv_pct_upper | {self.mv_pct_upper} | 市值过滤阈值（V4=0.85） |\n"
            f"| alpha | {self.alpha} | ML 分数权重（基本面权重 = 1-α） |\n"
            f"| 权重方式 | √市值加权 | 替代等权分配 |\n"
            f"| 基本面因子 | inv_pb, roe_ttm, roa_ttm, dv_ttm | 价值+质量+分红 |\n\n"
            f"### 已知局限\n\n"
            f"- 已过滤 ST / *ST 股票（基于 stock_info.name 静态名称匹配）\n"
            f"- α 参数为固定值，未做自适应调整"
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
        """Pre-compute features and labels for the 3-year training window."""
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

        # Determine biweekly rebalance dates within warmup window
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
        """
        Select stocks: top 5% by final score, max per industry, with buffer.

        V6 changes:
        - Uses 'final_score' (blended ML + fundamental) instead of pure ML
        - Returns sqrt(circ_mv) weighted portfolio instead of equal weight
        """
        df = scores_df.copy()

        # Apply holding buffer: boost final score for currently held stocks
        if current_holdings and self.buffer_sigma > 0:
            score_std = df["final_score"].std()
            if score_std > 0:
                held_codes = set(current_holdings.keys())
                boost = self.buffer_sigma * score_std
                df.loc[df["ts_code"].isin(held_codes), "final_score"] += boost

        # Sort by final score descending
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

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

        # V6: sqrt(market cap) weighting instead of equal weight
        selected_df = df[df["ts_code"].isin(selected)].copy()
        if "circ_mv" in selected_df.columns:
            mv_vals = selected_df.set_index("ts_code")["circ_mv"]
            mv_vals = mv_vals.reindex(selected).fillna(mv_vals.median())
            sqrt_mv = np.sqrt(mv_vals.clip(lower=1.0))
            weights = sqrt_mv / sqrt_mv.sum()
            return weights.to_dict()
        else:
            # Fallback to equal weight
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
        print(f"\n      ── V6 策略第 {self._call_count} 期  {date_str} ──")

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
            return {}
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

        # Step 4: Update training cache
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

        # Step 7: Predict & two-stage scoring
        if self._model is None:
            # Cold start: use fundamental score only
            fund_score = compute_fundamental_score(feat_ranked)
            result_df = feat_ranked[["ts_code", "sw_l1", "circ_mv"]].copy()
            result_df["ml_score"] = feat_ranked[FEATURE_NAMES].mean(axis=1).values
            result_df["fund_score"] = fund_score.values
            result_df["final_score"] = result_df["fund_score"]  # 100% fundamental in cold start
            print(f"      [预测] 冷启动模式 (纯基本面分数)，候选 {len(result_df)} 只")
        else:
            # Predict using LightGBM
            X_pred = feat_ranked[FEATURE_NAMES].copy()
            for col in FEATURE_NAMES:
                med = X_pred[col].median()
                X_pred[col] = X_pred[col].fillna(med)

            preds = self._model.predict(X_pred)

            # Normalize ML score to [0, 1] for blending
            ml_score_raw = pd.Series(preds, index=feat_ranked.index)
            ml_rank = ml_score_raw.rank(pct=True, method="average")

            # Compute fundamental score
            fund_score = compute_fundamental_score(feat_ranked)

            # V6: Two-stage blending
            final_score = self.alpha * ml_rank + (1 - self.alpha) * fund_score

            result_df = feat_ranked[["ts_code", "sw_l1", "circ_mv"]].copy()
            result_df["ml_score"] = ml_rank.values
            result_df["fund_score"] = fund_score.values
            result_df["final_score"] = final_score.values

            print(
                f"      [预测] 两阶段打分完成  "
                f"候选={len(result_df)} 只  "
                f"ML分数=[{ml_rank.min():.4f}, {ml_rank.max():.4f}]  "
                f"基本面=[{fund_score.min():.4f}, {fund_score.max():.4f}]  "
                f"混合比={self.alpha:.0%}ML+{1-self.alpha:.0%}基本面"
            )

        # Step 8: Select stocks with industry constraint, buffer, and sqrt(mv) weighting
        weights = self._select_stocks(result_df, current_holdings)
        n_held = len(current_holdings) if current_holdings else 0
        n_industries = result_df[result_df["ts_code"].isin(weights.keys())]["sw_l1"].nunique() if weights else 0
        overlap = len(set(weights.keys()) & set(current_holdings.keys())) if current_holdings and weights else 0

        # Show weight distribution info
        if weights:
            w_vals = list(weights.values())
            w_max = max(w_vals)
            w_min = min(w_vals)
            print(
                f"      [选股] 入选 {len(weights)} 只 / {n_industries} 个行业  "
                f"(上期持仓 {n_held} 只，留存 {overlap} 只，换手 {n_held + len(weights) - 2 * overlap} 只)  "
                f"权重范围=[{w_min:.4f}, {w_max:.4f}] (√市值加权)"
            )
        else:
            print(f"      [选股] 本期无入选股票")

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

    strategy = LGBMCrossSectionalV6(
        train_window_years=3,
        top_pct=0.05,
        max_per_industry=5,
        buffer_sigma=0.3,
        mv_pct_upper=0.70,        # V6: tightened from 0.85
        feature_lookback=260,
        alpha=0.8,                 # V6: 80% ML + 20% fundamental
        backtest_end_date=cfg.end_date,
    )
    result = run_backtest(strategy, cfg)
