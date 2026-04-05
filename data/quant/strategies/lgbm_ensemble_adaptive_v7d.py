"""
LightGBM Ensemble Adaptive Strategy V7D
===================================================

This variant keeps V7's stock selection pipeline, factor set, industry
constraints, turnover controls, and stop-loss logic, but replaces the
original five-level market exposure logic with a single small-cap regime
switch.

Three simple style timing modes are provided and selected by one parameter:
  1. csi500_trend:
     CSI500 / CSI300 ratio > MA20(ratio) and MA20(ratio) > MA60(ratio)
  2. csi1000_trend:
     CSI1000 / CSI300 ratio > MA20(ratio) and MA20(ratio) > MA60(ratio)
  3. csi1000_score:
     On CSI1000 / CSI300, satisfy at least 2 of 3 conditions:
       - ratio > MA20(ratio)
       - MA20(ratio) > MA60(ratio)
       - Ret20(CSI1000) - Ret20(CSI300) > 4%

Exposure rule is intentionally binary:
  - Small-cap regime detected  -> fully invested (100%)
  - Otherwise                  -> empty portfolio (0%)

Usage
-----
cd /data/Projects/Portal
python -m data.quant.strategies.lgbm_ensemble_adaptive_v7d
"""
import sys
import os
import argparse
import warnings
import time

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

# -- Import from Smart Money strategy --
from strategies.lgbm_smart_money import (
    FEATURE_COLUMNS as SM_FEATURE_COLUMNS,
    FEATURE_NAMES as SM_FEATURE_NAMES,
    compute_features_from_memory as sm_compute_features,
    rank_normalize,
    compute_forward_return_from_memory as sm_compute_forward_return,
    train_lgbm_model as sm_train_model,
)

# -- Import from Cross-Sectional strategy --
from strategies.lgbm_cross_sectional import (
    FEATURE_COLUMNS as CS_FEATURE_COLUMNS,
    FEATURE_NAMES as CS_FEATURE_NAMES,
    compute_features_from_memory as cs_compute_features,
    compute_forward_return_from_memory as cs_compute_forward_return,
    train_lgbm_model as cs_train_model,
)

# -- Module references for clearing date index caches --
import strategies.lgbm_smart_money as sm_module
import strategies.lgbm_cross_sectional as cs_module

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ===================================================================
# Merged FEATURE_COLUMNS (union of both strategies' DB columns)
# ===================================================================
ALL_FEATURE_COLUMNS = list(
    dict.fromkeys(SM_FEATURE_COLUMNS + CS_FEATURE_COLUMNS)
)

# Ensure industry momentum required columns are present
for _col in ["sw_l1", "circ_mv", "pct_chg"]:
    if _col not in ALL_FEATURE_COLUMNS:
        ALL_FEATURE_COLUMNS.append(_col)

# ===================================================================
# V5: Extended feature names (original + 3 new factors)
# ===================================================================
V5_NEW_FACTORS = ["ivol", "max_ret_20d", "skew_20d"]

SM_FEATURE_NAMES_V5 = list(SM_FEATURE_NAMES) + V5_NEW_FACTORS
CS_FEATURE_NAMES_V5 = list(CS_FEATURE_NAMES) + V5_NEW_FACTORS


def _compute_v5_factors(
    bulk_data: pd.DataFrame,
    date: pd.Timestamp,
    valid_codes: set,
    lookback: int = 260,
) -> dict:
    """
    Compute the 3 new V5 factors: ivol, max_ret_20d, skew_20d.

    Parameters
    ----------
    bulk_data : pd.DataFrame
        In-memory bulk data with trade_date, ts_code, close, pct_chg columns.
    date : pd.Timestamp
        Current date.
    valid_codes : set
        Set of ts_codes in the current universe.
    lookback : int
        Maximum lookback window.

    Returns
    -------
    dict of {factor_name: pd.Series}
    """
    from strategies.lgbm_smart_money import _get_bulk_date_index

    all_dates, date_to_rows = _get_bulk_date_index(bulk_data)
    date_ts = pd.Timestamp(date)
    valid_dates = all_dates[all_dates <= date_ts]

    if len(valid_dates) < 60:
        return {f: pd.Series(dtype=float) for f in V5_NEW_FACTORS}

    # Get last 60 dates for factor computation
    window_dates = valid_dates[-60:]
    row_indices = []
    for wd in window_dates:
        if wd in date_to_rows:
            row_indices.append(date_to_rows[wd])
    if not row_indices:
        return {f: pd.Series(dtype=float) for f in V5_NEW_FACTORS}

    all_row_idx = np.concatenate(row_indices)
    window = bulk_data.iloc[all_row_idx]
    window = window[window["ts_code"].isin(valid_codes)]

    # Build close pivot for the window
    close_pivot = window.pivot_table(
        index="trade_date", columns="ts_code", values="close"
    ).sort_index()

    n_dates = len(close_pivot)
    if n_dates < 20:
        return {f: pd.Series(dtype=float) for f in V5_NEW_FACTORS}

    daily_ret = close_pivot.pct_change(fill_method=None).iloc[1:]
    factors = {}

    # --- Factor 1: ivol (idiosyncratic volatility) ---
    # Residual volatility after removing market return
    # ivol = std(stock_ret - beta * market_ret) over 20 days
    if n_dates >= 21:
        ret_20 = daily_ret.iloc[-20:]
        market_ret = ret_20.mean(axis=1)  # equal-weight market return
        ivol_vals = {}
        for code in ret_20.columns:
            stock_ret = ret_20[code].dropna()
            if len(stock_ret) < 15:
                continue
            mkt = market_ret.reindex(stock_ret.index).dropna()
            common = stock_ret.index.intersection(mkt.index)
            if len(common) < 15:
                continue
            sr = stock_ret.loc[common].values
            mr = mkt.loc[common].values
            # OLS beta
            mr_dm = mr - mr.mean()
            denom = (mr_dm ** 2).sum()
            if denom > 0:
                beta = (mr_dm * (sr - sr.mean())).sum() / denom
                residual = sr - beta * mr
                ivol_vals[code] = np.std(residual)
            else:
                ivol_vals[code] = np.std(sr)
        factors["ivol"] = pd.Series(ivol_vals)
    else:
        factors["ivol"] = pd.Series(dtype=float)

    # --- Factor 2: max_ret_20d (lottery factor) ---
    # Maximum single-day return over the past 20 trading days
    if n_dates >= 21:
        ret_20 = daily_ret.iloc[-20:]
        factors["max_ret_20d"] = ret_20.max()
    else:
        factors["max_ret_20d"] = pd.Series(dtype=float)

    # --- Factor 3: skew_20d (return skewness) ---
    # Skewness of daily returns over the past 20 trading days
    if n_dates >= 21:
        ret_20 = daily_ret.iloc[-20:]
        factors["skew_20d"] = ret_20.skew()
    else:
        factors["skew_20d"] = pd.Series(dtype=float)

    return factors


def sm_compute_features_v5(
    date: pd.Timestamp,
    bulk_data: pd.DataFrame,
    lookback: int = 260,
    st_codes=None,
) -> pd.DataFrame:
    """Wrapper: compute original SM features + 3 V5 factors."""
    feat_df = sm_compute_features(date, bulk_data, lookback, st_codes)
    if feat_df is None:
        return None

    valid_codes = set(feat_df["ts_code"].tolist())
    v5_factors = _compute_v5_factors(bulk_data, date, valid_codes, lookback)

    for fname, fseries in v5_factors.items():
        feat_df[fname] = feat_df["ts_code"].map(fseries)

    return feat_df


def cs_compute_features_v5(
    date: pd.Timestamp,
    bulk_data: pd.DataFrame,
    lookback: int = 260,
    st_codes=None,
) -> pd.DataFrame:
    """Wrapper: compute original CS features + 3 V5 factors."""
    feat_df = cs_compute_features(date, bulk_data, lookback, st_codes)
    if feat_df is None:
        return None

    valid_codes = set(feat_df["ts_code"].tolist())
    v5_factors = _compute_v5_factors(bulk_data, date, valid_codes, lookback)

    for fname, fseries in v5_factors.items():
        feat_df[fname] = feat_df["ts_code"].map(fseries)

    return feat_df


def sm_train_model_v5(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features=None,
    val_labels=None,
) -> "lgb.Booster":
    """Train SM model with V5 extended features (25 factors)."""
    import lightgbm as lgb
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
    feature_names = SM_FEATURE_NAMES_V5
    X_train = train_features[feature_names].copy()
    for col in feature_names:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)

    dtrain = lgb.Dataset(X_train, label=train_labels)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=0)]

    if val_features is not None and val_labels is not None and len(val_labels) > 0:
        X_val = val_features[feature_names].copy()
        for col in feature_names:
            med = X_train[col].median()
            X_val[col] = X_val[col].fillna(med)
        dval = lgb.Dataset(X_val, label=val_labels, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

    model = lgb.train(
        params, dtrain, num_boost_round=300,
        valid_sets=valid_sets, valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def cs_train_model_v5(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features=None,
    val_labels=None,
) -> "lgb.Booster":
    """Train CS model with V5 extended features (25 factors)."""
    import lightgbm as lgb
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
    feature_names = CS_FEATURE_NAMES_V5
    X_train = train_features[feature_names].copy()
    for col in feature_names:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)

    dtrain = lgb.Dataset(X_train, label=train_labels)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=0)]

    if val_features is not None and val_labels is not None and len(val_labels) > 0:
        X_val = val_features[feature_names].copy()
        for col in feature_names:
            med = X_train[col].median()
            X_val[col] = X_val[col].fillna(med)
        dval = lgb.Dataset(X_val, label=val_labels, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

    model = lgb.train(
        params, dtrain, num_boost_round=300,
        valid_sets=valid_sets, valid_names=valid_names,
        callbacks=callbacks,
    )
    return model



# ===================================================================
# Stop-loss tracker
# ===================================================================

class StopLossTracker:
    """Track entry prices, trigger stop-loss, and manage cooldown periods."""

    def __init__(self, threshold: float = -0.12, cooldown_periods: int = 2):
        self.threshold = threshold
        self.cooldown = cooldown_periods
        self.entry_prices: Dict[str, float] = {}   # {ts_code: entry_price}
        self.cooldown_map: Dict[str, int] = {}     # {ts_code: remaining_cooldown}

    def update(
        self,
        current_holdings: Dict[str, int],
        prices: Dict[str, float],
    ) -> set:
        """
        Check stop-loss for current holdings. Returns set of stopped-out codes.
        Must be called at the beginning of each rebalance period.
        """
        stopped_out = set()

        # Check stop-loss
        for code in current_holdings:
            if code in self.entry_prices and code in prices:
                ret = prices[code] / self.entry_prices[code] - 1
                if ret < self.threshold:
                    stopped_out.add(code)
                    self.cooldown_map[code] = self.cooldown

        # Update cooldown periods
        expired = []
        for code, remaining in self.cooldown_map.items():
            if remaining <= 1:
                expired.append(code)
            else:
                self.cooldown_map[code] = remaining - 1
        for code in expired:
            del self.cooldown_map[code]

        # Clean up entry prices for stopped-out stocks
        for code in stopped_out:
            if code in self.entry_prices:
                del self.entry_prices[code]

        return stopped_out

    def is_in_cooldown(self, code: str) -> bool:
        return code in self.cooldown_map

    def record_entry(self, code: str, price: float):
        if code not in self.entry_prices:
            self.entry_prices[code] = price

    def record_exit(self, code: str):
        if code in self.entry_prices:
            del self.entry_prices[code]


# ===================================================================
# Strategy class
# ===================================================================

class LGBMEnsembleAdaptiveV7D(StrategyBase):
    """
    V4: Industry momentum scoring + dynamic industry constraint + factor attribution.

    Built on V1 base (dual LightGBM ensemble + market state timing).
    """

    def __init__(
        self,
        # Model parameters
        train_window_years: int = 3,
        retrain_interval: int = 4,
        # Ensemble parameters
        weight_model_a: float = 0.6,
        weight_model_b: float = 0.4,
        consensus_top_pct: float = 0.05,
        consensus_single_top_pct: float = 0.10,
        # Position parameters
        max_positions: int = 25,
        # Weight parameters
        softmax_temperature: float = 5.0,
        max_single_weight: float = 0.08,
        # Risk control parameters
        max_per_industry: int = 3,
        stop_loss_threshold: float = -0.12,
        stop_loss_cooldown: int = 2,
        buffer_sigma: float = 0.5,
        # V7: Minimum holding period (number of rebalance periods)
        min_holding_periods: int = 2,
        # V7: Portfolio drawdown circuit breaker
        drawdown_circuit_breaker: float = -0.15,
        drawdown_reduction_factor: float = 0.5,
        # Stock pool parameters
        mv_pct_upper: float = 0.85,
        small_cap_bonus: float = 0.02,
        small_cap_quantile: float = 0.70,
        feature_lookback: int = 260,
        backtest_end_date: Optional[str] = None,
        # V4: Industry momentum parameters
        industry_momentum_lookback_months: int = 6,
        industry_momentum_bonus: float = 0.02,
        industry_momentum_penalty: float = -0.01,
        industry_strong_top_n: int = 10,
        industry_weak_top_n: int = 10,
        industry_strong_max: int = 5,
        industry_weak_max: int = 2,
        # Style timing parameters
        baseline_dir: str = "data/quant/baseline",
        style_timing_mode: str = "csi1000_score",
        style_short_window: int = 20,
        style_long_window: int = 60,
        style_return_window: int = 20,
        style_return_threshold: float = 0.04,
    ):
        valid_modes = {"csi500_trend", "csi1000_trend", "csi1000_score"}
        if style_timing_mode not in valid_modes:
            raise ValueError(
                f"Unsupported style_timing_mode={style_timing_mode}. "
                f"Choose from {sorted(valid_modes)}"
            )

        super().__init__(f"lgbm_ensemble_adaptive_v7d_{style_timing_mode}")

        # Model parameters
        self.train_window_years = train_window_years
        self.retrain_interval = retrain_interval

        # Ensemble parameters
        self.weight_model_a = weight_model_a
        self.weight_model_b = weight_model_b
        self.consensus_top_pct = consensus_top_pct
        self.consensus_single_top_pct = consensus_single_top_pct

        # Position parameters
        self.max_positions = max_positions

        # Weight parameters
        self.softmax_temperature = softmax_temperature
        self.max_single_weight = max_single_weight

        # Risk control parameters
        self.max_per_industry = max_per_industry
        self.stop_loss_threshold = stop_loss_threshold
        self.buffer_sigma = buffer_sigma
        self.min_holding_periods = min_holding_periods
        self.drawdown_circuit_breaker = drawdown_circuit_breaker
        self.drawdown_reduction_factor = drawdown_reduction_factor

        # Stock pool parameters
        self.mv_pct_upper = mv_pct_upper
        self.small_cap_bonus = small_cap_bonus
        self.small_cap_quantile = small_cap_quantile
        self.feature_lookback = feature_lookback
        self._backtest_end_date = (
            pd.Timestamp(backtest_end_date) if backtest_end_date else None
        )

        # V4: Industry momentum parameters
        self.industry_momentum_lookback_months = industry_momentum_lookback_months
        self.industry_momentum_bonus = industry_momentum_bonus
        self.industry_momentum_penalty = industry_momentum_penalty
        self.industry_strong_top_n = industry_strong_top_n
        self.industry_weak_top_n = industry_weak_top_n
        self.industry_strong_max = industry_strong_max
        self.industry_weak_max = industry_weak_max

        # Style timing parameters
        self.baseline_dir = baseline_dir
        self.style_timing_mode = style_timing_mode
        self.style_short_window = style_short_window
        self.style_long_window = style_long_window
        self.style_return_window = style_return_window
        self.style_return_threshold = style_return_threshold
        self._style_data: Optional[pd.DataFrame] = None

        valid_modes = {"csi500_trend", "csi1000_trend", "csi1000_score"}
        if self.style_timing_mode not in valid_modes:
            raise ValueError(
                f"Unsupported style_timing_mode={self.style_timing_mode}. "
                f"Choose from {sorted(valid_modes)}"
            )

        # -- Sub-model states --
        self._model_a: Optional[lgb.Booster] = None  # Smart Money model
        self._model_b: Optional[lgb.Booster] = None  # Cross-Sectional model
        self._train_cache_a: List[Tuple[str, pd.DataFrame, pd.Series]] = []
        self._train_cache_b: List[Tuple[str, pd.DataFrame, pd.Series]] = []
        self._last_train_date: Optional[pd.Timestamp] = None

        # -- Strategy states --
        self._call_count = 0
        self._warmup_done = False
        self._bulk_data: Optional[pd.DataFrame] = None
        self._bulk_last_date: Optional[pd.Timestamp] = None
        self._st_codes: Optional[set] = None
        self._stop_loss = StopLossTracker(
            threshold=stop_loss_threshold,
            cooldown_periods=stop_loss_cooldown,
        )

        # V7: Minimum holding period tracker {ts_code: periods_held}
        self._holding_periods: Dict[str, int] = {}

        # V7: Drawdown circuit breaker state
        self._peak_nav: float = 0.0
        self._circuit_breaker_active: bool = False

        # Factor attribution cache
        self._last_factor_exposures: Optional[pd.DataFrame] = None

    def describe(self) -> str:
        rule_name, rule_desc = self._get_style_rule_meta()
        return (
            f"### 策略思路\n\n"
            f"自适应集成选股 V7（风格开关版）：保留 V7 的双模型选股、行业动量、换手控制、"
            f"个股止损等核心逻辑，但把市场暴露改成**单参数驱动的二元风格开关**。"
            f"当所选风格规则判断当前是小盘占优时，策略**全仓运行原 V7 选股逻辑**；"
            f"否则**直接空仓跳过**。\n\n"
            f"### 当前启用的风格判断策略\n\n"
            f"- `style_timing_mode = {self.style_timing_mode}`\n"
            f"- 规则名称：{rule_name}\n"
            f"- 规则说明：{rule_desc}\n"
            f"- 暴露方式：小盘占优=100%仓位；非小盘占优=0%仓位\n\n"
            f"### 可选的三种简单风格判断策略\n\n"
            f"1. `csi500_trend`：CSI500/CSI300 比值上穿短均线，且短均线在长均线上方\n"
            f"2. `csi1000_trend`：CSI1000/CSI300 比值上穿短均线，且短均线在长均线上方\n"
            f"3. `csi1000_score`：在 CSI1000/CSI300 上，三项条件满足至少两项\n"
            f"   （比值>MA{self.style_short_window}、MA{self.style_short_window}>MA{self.style_long_window}、"
            f"{self.style_return_window}日收益差>{self.style_return_threshold:.0%}）\n\n"
            f"### 继承自 V7 的核心逻辑\n\n"
            f"- 双模型融合：Smart Money ({self.weight_model_a}) + Cross-Sectional ({self.weight_model_b})\n"
            f"- 共识过滤（综合前{self.consensus_top_pct*100:.0f}% 且至少一个模型前{self.consensus_single_top_pct*100:.0f}%）\n"
            f"- 行业动量连续化打分（scale={self.industry_momentum_bonus}）\n"
            f"- 小盘加分（bonus={self.small_cap_bonus}，仅用于股内排序，不控制总仓位）\n"
            f"- 最小持仓期（{self.min_holding_periods}期/{self.min_holding_periods*2}周）+ 换手缓冲（sigma={self.buffer_sigma}）\n"
            f"- 个股止损（-{abs(self.stop_loss_threshold)*100:.0f}%）+ 冷却机制\n"
            f"- softmax 加权 (temperature={self.softmax_temperature})，单股上限{self.max_single_weight*100:.0f}%\n"
        )

    def _get_style_rule_meta(self) -> Tuple[str, str]:
        rule_map = {
            "csi500_trend": (
                "CSI500/CSI300 趋势比值",
                f"CSI500/CSI300 比值 > MA{self.style_short_window}，且 MA{self.style_short_window} > MA{self.style_long_window}",
            ),
            "csi1000_trend": (
                "CSI1000/CSI300 趋势比值",
                f"CSI1000/CSI300 比值 > MA{self.style_short_window}，且 MA{self.style_short_window} > MA{self.style_long_window}",
            ),
            "csi1000_score": (
                "CSI1000/CSI300 二选三评分",
                f"满足以下三项中的至少两项：比值 > MA{self.style_short_window}、MA{self.style_short_window} > MA{self.style_long_window}、{self.style_return_window}日收益差 > {self.style_return_threshold:.0%}",
            ),
        }
        return rule_map[self.style_timing_mode]

    def _load_style_data(self):
        """Load CSI300 / CSI500 / CSI1000 close prices from baseline CSV files."""
        if self._style_data is not None:
            return

        def _read_index_file(candidates: List[str], column_name: str) -> Optional[pd.DataFrame]:
            for filename in candidates:
                candidate_paths = []
                if os.path.isabs(filename):
                    candidate_paths.append(filename)
                else:
                    candidate_paths.append(os.path.join(self.baseline_dir, filename))
                    if os.sep in filename:
                        candidate_paths.append(filename)

                for path in candidate_paths:
                    if not os.path.exists(path):
                        continue
                    try:
                        df = pd.read_csv(path)
                        date_col = "date" if "date" in df.columns else "trade_date"
                        close_col = "close" if "close" in df.columns else None
                        if close_col is None or date_col not in df.columns:
                            continue
                        out = df[[date_col, close_col]].copy()
                        out[date_col] = pd.to_datetime(out[date_col])
                        out = out.rename(columns={date_col: "date", close_col: column_name})
                        return out.set_index("date").sort_index()
                    except Exception as e:
                        print(f"      [风格数据] [X] 读取 {path} 失败: {e}")
            return None

        index_map = {
            "csi300": ["000300.SH.csv", "000300.SZ.csv"],
            "csi500": ["000905.SH.csv", "000905.SZ.csv"],
            "csi1000": [
                "000852.SH.csv",
                "000852.SZ.csv",
                "data/quant/data/daily/000852.SZ.csv",
            ],
        }

        loaded = []
        for col, candidates in index_map.items():
            df = _read_index_file(candidates, col)
            if df is not None:
                loaded.append(df)

        if not loaded:
            print(f"      [风格数据] [X] baseline_dir={self.baseline_dir} 未加载到任何风格指数文件")
            self._style_data = pd.DataFrame()
            return

        style_data = loaded[0]
        for df in loaded[1:]:
            style_data = style_data.join(df, how="outer")

        self._style_data = style_data.sort_index()
        print(
            f"      [风格数据] [OK] 已加载列: {', '.join(self._style_data.columns.tolist())}  "
            f"共 {len(self._style_data)} 行"
        )

    def _compute_style_signal(self, date: pd.Timestamp) -> Tuple[bool, str]:
        """Return (is_small_cap_regime, detail_text)."""
        self._load_style_data()

        if self._style_data is None or self._style_data.empty:
            return False, "风格数据缺失"

        date_ts = pd.Timestamp(date)
        valid = self._style_data[self._style_data.index <= date_ts].copy()
        min_history = max(self.style_long_window, self.style_return_window + 1)
        if len(valid) < min_history:
            return False, f"历史不足 {min_history} 个交易日"

        def _ratio_detail(small_col: str) -> Tuple[Optional[pd.Series], str]:
            if small_col not in valid.columns or "csi300" not in valid.columns:
                return None, f"缺少列 {small_col} / csi300"
            ratio = (valid[small_col] / valid["csi300"]).dropna()
            if len(ratio) < self.style_long_window:
                return None, f"{small_col}/csi300 比值历史不足 {self.style_long_window} 个交易日"
            return ratio, ""

        if self.style_timing_mode == "csi500_trend":
            ratio, err = _ratio_detail("csi500")
            if ratio is None:
                return False, err
            ratio_now = ratio.iloc[-1]
            ma_short = ratio.rolling(self.style_short_window).mean().iloc[-1]
            ma_long = ratio.rolling(self.style_long_window).mean().iloc[-1]
            is_small = bool(ratio_now > ma_short and ma_short > ma_long)
            detail = (
                f"ratio={ratio_now:.4f}, ma{self.style_short_window}={ma_short:.4f}, "
                f"ma{self.style_long_window}={ma_long:.4f}"
            )
            return is_small, detail

        if self.style_timing_mode == "csi1000_trend":
            ratio, err = _ratio_detail("csi1000")
            if ratio is None:
                return False, err
            ratio_now = ratio.iloc[-1]
            ma_short = ratio.rolling(self.style_short_window).mean().iloc[-1]
            ma_long = ratio.rolling(self.style_long_window).mean().iloc[-1]
            is_small = bool(ratio_now > ma_short and ma_short > ma_long)
            detail = (
                f"ratio={ratio_now:.4f}, ma{self.style_short_window}={ma_short:.4f}, "
                f"ma{self.style_long_window}={ma_long:.4f}"
            )
            return is_small, detail

        ratio, err = _ratio_detail("csi1000")
        if ratio is None:
            return False, err
        ratio_now = ratio.iloc[-1]
        ma_short = ratio.rolling(self.style_short_window).mean().iloc[-1]
        ma_long = ratio.rolling(self.style_long_window).mean().iloc[-1]
        ret_small = valid["csi1000"].pct_change(self.style_return_window).iloc[-1]
        ret_large = valid["csi300"].pct_change(self.style_return_window).iloc[-1]
        ret_diff = ret_small - ret_large

        cond1 = bool(ratio_now > ma_short)
        cond2 = bool(ma_short > ma_long)
        cond3 = bool(ret_diff > self.style_return_threshold)
        score = int(cond1) + int(cond2) + int(cond3)
        is_small = score >= 2
        detail = (
            f"cond=({int(cond1)},{int(cond2)},{int(cond3)}) score={score}/3, "
            f"ratio={ratio_now:.4f}, ma{self.style_short_window}={ma_short:.4f}, "
            f"ma{self.style_long_window}={ma_long:.4f}, ret_diff={ret_diff:.2%}"
        )
        return is_small, detail

    # -- Internal helpers --

    def _should_retrain(self) -> bool:
        if self._model_a is None or self._model_b is None:
            return True
        return self._call_count % self.retrain_interval == 0

    def _apply_market_cap_filter(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to top mv_pct_upper by circulating market value."""
        if "circ_mv" not in feat_df.columns:
            return feat_df
        mv = feat_df["circ_mv"].dropna()
        if len(mv) == 0:
            return feat_df
        lower_bound = mv.quantile(1.0 - self.mv_pct_upper)
        return feat_df[feat_df["circ_mv"] >= lower_bound].copy()

    def _warmup_training_cache(
        self,
        current_date: pd.Timestamp,
        accessor: DataAccessor,
    ):
        """Pre-compute features and labels for historical dates (both models)."""
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
                print(f"      [ST过滤] [OK] 从 stock_info 加载 {len(self._st_codes)} 只 ST/*ST 股票")
            except Exception as e:
                print(f"      [ST过滤] [X] 查询失败: {e}，跳过 ST 过滤")
                self._st_codes = set()

        # Bulk-load ALL data (union of both models' columns)
        data_start = warmup_start - pd.DateOffset(days=int(self.feature_lookback * 1.8))
        bulk = prefetch_bulk_data(accessor, data_start, current_date, ALL_FEATURE_COLUMNS)

        if bulk.empty:
            print(f"      [预热] [X] 未找到历史数据")
            return

        self._bulk_data = bulk.copy()
        self._bulk_data.drop_duplicates(
            subset=["ts_code", "trade_date"], keep="last", inplace=True
        )
        self._bulk_data.sort_values(["trade_date", "ts_code"], inplace=True)
        self._bulk_data.reset_index(drop=True, inplace=True)
        self._bulk_last_date = pd.Timestamp(self._bulk_data["trade_date"].max())

        print(
            f"      [预热] [OK] 已缓存至首个交易日 {current_date.strftime('%Y-%m-%d')}，"
            f"后续将按实盘方式逐期 walk-forward 推进"
        )

        # Clear date index caches for both modules
        sm_module._bulk_date_index_cache.clear()
        cs_module._bulk_date_index_cache.clear()

        # Determine biweekly rebalance dates within warmup window
        hist_trade_dates_raw = self._bulk_data["trade_date"].unique()
        hist_trade_dates_raw = np.sort(hist_trade_dates_raw)
        mask = (hist_trade_dates_raw >= pd.Timestamp(warmup_start)) & \
               (hist_trade_dates_raw <= pd.Timestamp(warmup_end))
        hist_trade_dates = pd.DatetimeIndex(hist_trade_dates_raw[mask])

        if len(hist_trade_dates) == 0:
            print(f"      [预热] [X] 历史交易日为空")
            return

        # Biweekly: group by 14-day blocks, take last trading day of each
        origin = hist_trade_dates[0]
        day_offsets = (hist_trade_dates - origin).days
        block_ids = day_offsets // 14
        s = pd.Series(hist_trade_dates, index=hist_trade_dates)
        groups = s.groupby(block_ids)
        hist_rebal_dates = pd.DatetimeIndex(groups.last().values)

        print(f"      [预热] 历史调仓日 {len(hist_rebal_dates)} 个，"
              f"开始计算两组特征和标签 (全内存模式) ...")

        t0 = time.time()
        n_success = 0
        n_skip = 0

        for i in range(len(hist_rebal_dates) - 1):
            d = hist_rebal_dates[i]
            d_next = hist_rebal_dates[i + 1]

            # Model A features (Smart Money)
            feat_a = sm_compute_features_v5(
                d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            # Model B features (Cross-Sectional)
            feat_b = cs_compute_features_v5(
                d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )

            if feat_a is None or feat_b is None:
                n_skip += 1
                continue

            feat_a = self._apply_market_cap_filter(feat_a)
            feat_b = self._apply_market_cap_filter(feat_b)
            if len(feat_a) < 50 or len(feat_b) < 50:
                n_skip += 1
                continue

            feat_a_ranked = rank_normalize(feat_a, SM_FEATURE_NAMES_V5)
            feat_b_ranked = rank_normalize(feat_b, CS_FEATURE_NAMES_V5)

            # Forward return (shared)
            fwd_ret = sm_compute_forward_return(d, d_next, self._bulk_data)
            if fwd_ret is None:
                n_skip += 1
                continue

            d_str = d.strftime("%Y-%m-%d")
            self._train_cache_a.append((d_str, feat_a_ranked.copy(), fwd_ret))
            self._train_cache_b.append((d_str, feat_b_ranked.copy(), fwd_ret))
            n_success += 1

        # Cache the LAST date
        if len(hist_rebal_dates) > 0:
            last_d = hist_rebal_dates[-1]
            feat_a = sm_compute_features_v5(
                last_d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            feat_b = cs_compute_features_v5(
                last_d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            if feat_a is not None and feat_b is not None:
                feat_a = self._apply_market_cap_filter(feat_a)
                feat_b = self._apply_market_cap_filter(feat_b)
                if len(feat_a) >= 50 and len(feat_b) >= 50:
                    feat_a_ranked = rank_normalize(feat_a, SM_FEATURE_NAMES_V5)
                    feat_b_ranked = rank_normalize(feat_b, CS_FEATURE_NAMES_V5)
                    fwd_ret = sm_compute_forward_return(
                        last_d, current_date, self._bulk_data
                    )
                    if fwd_ret is not None:
                        d_str = last_d.strftime("%Y-%m-%d")
                        self._train_cache_a.append(
                            (d_str, feat_a_ranked.copy(), fwd_ret)
                        )
                        self._train_cache_b.append(
                            (d_str, feat_b_ranked.copy(), fwd_ret)
                        )
                        n_success += 1

        # Prune old data
        cutoff = current_date - pd.DateOffset(
            years=self.train_window_years, months=6
        )
        self._train_cache_a = [
            (d, f, l) for d, f, l in self._train_cache_a
            if pd.Timestamp(d) >= cutoff
        ]
        self._train_cache_b = [
            (d, f, l) for d, f, l in self._train_cache_b
            if pd.Timestamp(d) >= cutoff
        ]

        elapsed = time.time() - t0
        print(f"      [预热] [OK] 成功缓存 {n_success} 期 "
              f"(跳过 {n_skip} 期)，"
              f"耗时 {elapsed:.1f} 秒，"
              f"训练窗口已就绪")

        # Trim _bulk_data to only keep the lookback window needed for
        # feature computation at current_date. Historical data beyond
        # this window was only needed for warmup training and can be
        # discarded to keep compute_features_from_memory fast.
        trim_cutoff = current_date - pd.DateOffset(
            days=int(self.feature_lookback * 1.8)
        )
        rows_before = len(self._bulk_data)
        self._bulk_data = self._bulk_data[
            self._bulk_data["trade_date"] >= trim_cutoff
        ].reset_index(drop=True)
        sm_module._bulk_date_index_cache.clear()
        cs_module._bulk_date_index_cache.clear()
        print(f"      [预热] 裁剪缓存: {rows_before:,} → {len(self._bulk_data):,} 行 "
              f"(保留 {trim_cutoff.strftime('%Y-%m-%d')} 之后的数据)")

    def _append_walk_forward_data(
        self,
        current_date: pd.Timestamp,
        accessor: DataAccessor,
    ):
        """Append only data up to current_date after live trading starts."""
        if self._bulk_data is None or self._bulk_data.empty:
            data_start = current_date - pd.DateOffset(days=int(self.feature_lookback * 1.8))
            bulk = prefetch_bulk_data(accessor, data_start, current_date, ALL_FEATURE_COLUMNS)
            if bulk.empty:
                self._bulk_data = None
                self._bulk_last_date = None
                return
            self._bulk_data = bulk.copy()
            self._bulk_data.drop_duplicates(
                subset=["ts_code", "trade_date"], keep="last", inplace=True
            )
            self._bulk_data.sort_values(["trade_date", "ts_code"], inplace=True)
            self._bulk_data.reset_index(drop=True, inplace=True)
            self._bulk_last_date = pd.Timestamp(self._bulk_data["trade_date"].max())
            sm_module._bulk_date_index_cache.clear()
            cs_module._bulk_date_index_cache.clear()
            return

        if self._bulk_last_date is None:
            self._bulk_last_date = pd.Timestamp(self._bulk_data["trade_date"].max())

        if current_date <= self._bulk_last_date:
            return

        incremental = prefetch_bulk_data(
            accessor,
            self._bulk_last_date,
            current_date,
            ALL_FEATURE_COLUMNS,
        )
        if incremental.empty:
            print(
                f"      [推进] [X] 未获取到 {self._bulk_last_date.strftime('%Y-%m-%d')} ~ "
                f"{current_date.strftime('%Y-%m-%d')} 的增量数据"
            )
            return

        before_rows = len(self._bulk_data)
        self._bulk_data = pd.concat([self._bulk_data, incremental], ignore_index=True)
        self._bulk_data.drop_duplicates(
            subset=["ts_code", "trade_date"], keep="last", inplace=True
        )
        self._bulk_data.sort_values(["trade_date", "ts_code"], inplace=True)
        self._bulk_data.reset_index(drop=True, inplace=True)
        self._bulk_last_date = pd.Timestamp(self._bulk_data["trade_date"].max())

        # Trim old data beyond the lookback window to keep _bulk_data size stable.
        # This prevents compute_features_from_memory and _get_bulk_date_index
        # from slowing down as the backtest progresses.
        trim_cutoff = current_date - pd.DateOffset(
            days=int(self.feature_lookback * 1.8)
        )
        rows_before_trim = len(self._bulk_data)
        self._bulk_data = self._bulk_data[
            self._bulk_data["trade_date"] >= trim_cutoff
        ].reset_index(drop=True)
        trimmed = rows_before_trim - len(self._bulk_data)

        sm_module._bulk_date_index_cache.clear()
        cs_module._bulk_date_index_cache.clear()

        added_rows = len(self._bulk_data) - before_rows
        trim_info = f"  裁剪旧数据 {trimmed:,} 行" if trimmed > 0 else ""
        print(
            f"      [推进] [OK] 已追加至 {self._bulk_last_date.strftime('%Y-%m-%d')}，"
            f"净新增 {max(added_rows, 0):,} 行  "
            f"当前缓存 {len(self._bulk_data):,} 行{trim_info}"
        )

    def _train_sub_model(
        self,
        model_id: str,
        train_cache: List[Tuple[str, pd.DataFrame, pd.Series]],
        feature_names: List[str],
        train_fn,
        date: pd.Timestamp,
    ) -> Optional[lgb.Booster]:
        """Train a sub-model (A or B) from its training cache."""
        valid_cache = [
            (d, f, l) for d, f, l in train_cache
            if not l.empty and pd.Timestamp(d) < date
        ]
        if len(valid_cache) < 8:
            print(f"      [训练-{model_id}] [X] 有效训练期不足 ({len(valid_cache)})")
            return None

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
                    # V5: Industry-neutral labels
                    # Subtract industry median to learn intra-industry alpha
                    if "sw_l1" in f_df.columns and "ts_code" in f_df.columns:
                        ind_map = f_df.set_index("ts_code")["sw_l1"]
                        ind_for_merged = ind_map.reindex(merged.index)
                        if ind_for_merged.notna().sum() > 0:
                            ind_median = merged["label"].groupby(ind_for_merged).transform("median")
                            ind_median = ind_median.fillna(merged["label"].median())
                            merged["label"] = merged["label"] - ind_median
                    merged["label"] = merged["label"].rank(
                        pct=True, method="average"
                    )
                    all_X.append(merged[feature_names])
                    all_y.append(merged["label"])

        if len(all_X) < 8:
            print(f"      [训练-{model_id}] [X] 合并后训练期不足 ({len(all_X)})")
            return None

        n = len(all_X)
        split = max(1, int(n * 0.8))

        train_X = pd.concat(all_X[:split])
        train_y = pd.concat(all_y[:split])
        val_X = pd.concat(all_X[split:]) if split < n else None
        val_y = pd.concat(all_y[split:]) if split < n else None

        try:
            model = train_fn(train_X, train_y, val_X, val_y)
            n_train = len(train_y)
            n_val = len(val_y) if val_y is not None else 0
            best_iter = model.best_iteration if hasattr(model, 'best_iteration') else '?'
            print(
                f"      [训练-{model_id}] [OK] 完成  "
                f"训练={n_train:,}  验证={n_val:,}  "
                f"{len(all_X)}期  最优={best_iter}"
            )
            return model
        except Exception as e:
            print(f"      [训练-{model_id}] [X] 失败: {e}")
            return None

    def _compute_market_state(self, close_pivot: pd.DataFrame) -> float:
        """
        Compute market state from close_pivot and return position coefficient.

        Three dimensions:
        1. Trend: market median MA20 vs MA60
        2. Breadth: percentage of stocks above their MA20
        3. Volatility: median(20d vol / 60d vol)

        Returns coefficient in {0.1, 0.3, 0.5, 0.8, 1.0}.
        """
        n = len(close_pivot)
        if n < 60:
            return 0.5  # default to neutral

        # Trend dimension
        market_median = close_pivot.median(axis=1)
        ma20 = market_median.rolling(20).mean().iloc[-1]
        ma60 = market_median.rolling(60).mean().iloc[-1]
        is_uptrend = ma20 > ma60

        # Breadth dimension
        stock_ma20 = close_pivot.rolling(20).mean()
        above_ratio = (close_pivot.iloc[-1] > stock_ma20.iloc[-1]).mean()

        # Volatility dimension
        daily_ret = close_pivot.pct_change()
        vol_20 = daily_ret.iloc[-20:].std() * np.sqrt(252)
        vol_60 = daily_ret.iloc[-60:].std() * np.sqrt(252)
        vol_ratio_median = (vol_20 / vol_60).median()
        is_high_vol = vol_ratio_median > 1.5

        # Position matrix (V7: relaxed panic trigger - no vol requirement)
        if is_uptrend:
            if above_ratio > 0.60:
                coeff = 1.00  # strong bull
            else:
                coeff = 0.80  # mild bull
        else:
            if above_ratio >= 0.30:
                coeff = 0.50  # range-bound
            elif above_ratio >= 0.15:
                coeff = 0.30  # weak bear
            else:
                coeff = 0.10  # panic (V7: triggers when breadth < 15%)

        return coeff

    # ---------------------------------------------------------------
    # V4: Industry momentum computation
    # ---------------------------------------------------------------

    def _compute_industry_momentum(
        self,
        date: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """
        Compute 6-month circ_mv-weighted return for each SW-L1 industry.

        Uses self._bulk_data which contains daily pct_chg, circ_mv, sw_l1.

        Returns
        -------
        pd.DataFrame with columns: [industry, momentum_ret, rank]
            sorted by momentum_ret descending.
            rank=1 is the strongest industry.
        Returns None if data is insufficient.
        """
        if self._bulk_data is None or self._bulk_data.empty:
            return None

        # Determine lookback window: ~126 trading days for 6 months
        lookback_days = self.industry_momentum_lookback_months * 21
        date_ts = pd.Timestamp(date)

        # Get all unique dates up to current date
        all_dates = self._bulk_data["trade_date"].unique()
        all_dates = np.sort(all_dates)
        valid_dates = all_dates[all_dates <= date_ts]

        if len(valid_dates) < lookback_days:
            # Not enough history, use whatever is available
            if len(valid_dates) < 21:
                return None
            window_dates = valid_dates
        else:
            window_dates = valid_dates[-lookback_days:]

        start_date = window_dates[0]
        end_date = window_dates[-1]

        # Filter bulk data to the window
        mask = (
            (self._bulk_data["trade_date"] >= start_date)
            & (self._bulk_data["trade_date"] <= end_date)
            & (self._bulk_data["sw_l1"].notna())
            & (self._bulk_data["sw_l1"] != "")
            & (self._bulk_data["pct_chg"].notna())
            & (self._bulk_data["circ_mv"].notna())
            & (self._bulk_data["circ_mv"] > 0)
        )
        window_df = self._bulk_data.loc[mask].copy()

        if len(window_df) < 100:
            return None

        window_df["ret_daily"] = window_df["pct_chg"] / 100.0

        # Compute circ_mv-weighted daily return per industry, then compound
        records = []
        for ind, ind_grp in window_df.groupby("sw_l1"):
            daily_rets = []
            for day, day_grp in ind_grp.groupby("trade_date"):
                w = day_grp["circ_mv"].values
                r = day_grp["ret_daily"].values
                w_sum = w.sum()
                if w_sum > 0:
                    daily_rets.append(np.dot(w, r) / w_sum)

            if len(daily_rets) > 0:
                cum_ret = np.prod([1 + r for r in daily_rets]) - 1
            else:
                cum_ret = 0.0

            records.append({
                "industry": ind,
                "momentum_ret": cum_ret,
            })

        if len(records) < 5:
            return None

        result = pd.DataFrame(records)
        result = result.sort_values("momentum_ret", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        return result

    def _compute_ensemble_score(
        self,
        score_a: np.ndarray,
        score_b: np.ndarray,
        codes_a: pd.Index,
        codes_b: pd.Index,
        log_cap_series: pd.Series,
        industry_map: Optional[Dict[str, str]] = None,
        industry_momentum: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Fuse two model scores into an ensemble score.

        Steps:
        1. Rank-normalize each model's scores
        2. Weighted fusion: w_A * rank_A + w_B * rank_B
        3. Small-cap bonus: +0.02 for stocks below small_cap_quantile in log_cap
        4. V4: Industry momentum bonus/penalty
        """
        # Build Series from arrays
        s_a = pd.Series(score_a, index=codes_a)
        s_b = pd.Series(score_b, index=codes_b)

        # Find common universe
        common = s_a.index.intersection(s_b.index)
        if len(common) == 0:
            return pd.Series(dtype=float)

        s_a = s_a.loc[common]
        s_b = s_b.loc[common]

        # Rank normalize to [0, 1]
        rank_a = s_a.rank(pct=True, method="average")
        rank_b = s_b.rank(pct=True, method="average")

        # Weighted fusion
        ensemble = self.weight_model_a * rank_a + self.weight_model_b * rank_b

        # Small-cap bonus
        if log_cap_series is not None:
            cap_common = log_cap_series.reindex(common)
            cap_threshold = cap_common.quantile(self.small_cap_quantile)
            small_mask = cap_common <= cap_threshold
            ensemble[small_mask] += self.small_cap_bonus

        # V5: Continuous industry momentum scoring
        if industry_momentum is not None and industry_map is not None:
            n_industries = len(industry_momentum)
            # Build continuous rank-normalized bonus: rank 1 (strongest) -> +scale, last -> -scale
            ind_rank_map = {}
            for _, row in industry_momentum.iterrows():
                # Normalize rank to [-1, +1]: rank 1 -> +1, last rank -> -1
                normalized = 1.0 - 2.0 * (row["rank"] - 1) / max(n_industries - 1, 1)
                ind_rank_map[row["industry"]] = normalized

            scale = self.industry_momentum_bonus  # reuse as scale factor
            n_adjusted = 0
            for code in common:
                ind = industry_map.get(code)
                if ind is None or pd.isna(ind):
                    continue
                if ind in ind_rank_map:
                    bonus = ind_rank_map[ind] * scale
                    ensemble[code] += bonus
                    n_adjusted += 1

            if n_adjusted > 0:
                print(
                    f"      [V7行业动量] 连续化加分 {n_adjusted} 只 "
                    f"(scale={scale}, range=[{-scale:.3f}, +{scale:.3f}])"
                )

        return ensemble

    def _compute_weights(
        self, ensemble_scores: pd.Series
    ) -> pd.Series:
        """Softmax weighting with temperature and single-stock cap."""
        scaled = ensemble_scores * self.softmax_temperature
        scaled = scaled - scaled.max()  # numerical stability
        exp_scores = np.exp(scaled)
        weights = exp_scores / exp_scores.sum()

        # Apply single-stock cap
        weights = weights.clip(upper=self.max_single_weight)
        weights = weights / weights.sum()  # renormalize
        return weights

    def _select_and_weight(
        self,
        ensemble_scores: pd.Series,
        rank_a: pd.Series,
        rank_b: pd.Series,
        feat_df: pd.DataFrame,
        current_holdings: Dict[str, int],
        effective_max: int,
        stopped_out: set,
        industry_momentum: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Full pipeline: consensus filter → stop-loss/cooldown filter →
        turnover buffer → dynamic industry constraint → select → softmax weight.
        """
        if ensemble_scores.empty:
            return {}

        # Step 1: Consensus filter
        top_threshold = ensemble_scores.quantile(1.0 - self.consensus_top_pct)
        top_mask = ensemble_scores >= top_threshold

        single_threshold_a = 1.0 - self.consensus_single_top_pct
        single_threshold_b = 1.0 - self.consensus_single_top_pct
        model_a_top = rank_a >= single_threshold_a
        model_b_top = rank_b >= single_threshold_b
        consensus_mask = top_mask & (model_a_top | model_b_top)

        candidates = ensemble_scores[consensus_mask].copy()
        if candidates.empty:
            n_fallback = max(effective_max, 10)
            candidates = ensemble_scores.nlargest(n_fallback)

        # Step 2: Remove stopped-out and cooldown stocks
        to_remove = set()
        for code in candidates.index:
            if code in stopped_out or self._stop_loss.is_in_cooldown(code):
                to_remove.add(code)
        candidates = candidates.drop(to_remove, errors='ignore')

        if candidates.empty:
            return {}

        # Step 3: Turnover buffer + V7 minimum holding period
        if current_holdings and self.buffer_sigma > 0:
            score_std = candidates.std()
            if score_std > 0:
                held_codes = set(current_holdings.keys())
                boost = self.buffer_sigma * score_std
                for code in candidates.index:
                    if code in held_codes:
                        candidates[code] += boost
                        # V7: Extra boost for stocks below minimum holding period
                        periods_held = self._holding_periods.get(code, 0)
                        if periods_held < self.min_holding_periods:
                            candidates[code] += boost * 2.0  # strong retention boost

        candidates = candidates.sort_values(ascending=False)

        # Step 4: Dynamic industry constraint (V4) + position limit
        if "sw_l1" in feat_df.columns and "ts_code" in feat_df.columns:
            industry_map = feat_df.set_index("ts_code")["sw_l1"].to_dict()
        else:
            industry_map = {}

        # V4: Build per-industry max limit based on momentum rank
        industry_max_map: Dict[str, int] = {}
        if industry_momentum is not None and len(industry_momentum) > 0:
            strong_set = set(
                industry_momentum.head(self.industry_strong_top_n)["industry"]
            )
            weak_set = set(
                industry_momentum.tail(self.industry_weak_top_n)["industry"]
            )
            for ind in set(industry_map.values()):
                if pd.isna(ind):
                    continue
                if ind in strong_set:
                    industry_max_map[ind] = self.industry_strong_max
                elif ind in weak_set:
                    industry_max_map[ind] = self.industry_weak_max
                else:
                    industry_max_map[ind] = self.max_per_industry

        selected = []
        industry_count: Dict[str, int] = {}
        for code in candidates.index:
            if len(selected) >= effective_max:
                break
            ind = industry_map.get(code, "unknown")
            if pd.isna(ind):
                ind = "unknown"
            max_for_ind = industry_max_map.get(ind, self.max_per_industry)
            cnt = industry_count.get(ind, 0)
            if cnt < max_for_ind:
                selected.append(code)
                industry_count[ind] = cnt + 1

        if not selected:
            return {}

        # Step 5: Softmax weighting
        selected_scores = candidates.loc[selected]
        weights = self._compute_weights(selected_scores)

        return weights.to_dict()

    # ---------------------------------------------------------------
    # Factor attribution interface
    # ---------------------------------------------------------------

    def get_factor_exposures(
        self,
        date: pd.Timestamp,
        selected_codes: Dict[str, float],
    ) -> Optional[pd.DataFrame]:
        """
        Return per-stock factor exposures for factor attribution analysis.

        Factors reported:
        - model_a_score: Smart Money model prediction (rank-normalized)
        - model_b_score: Cross-Sectional model prediction (rank-normalized)
        - small_cap: small-cap bonus indicator (1 if small-cap, 0 otherwise)
        - industry_momentum: industry momentum rank (rank-normalized, higher=stronger)
        - market_state: market state coefficient (same for all stocks in a period)
        """
        if self._last_factor_exposures is not None:
            return self._last_factor_exposures
        return None

    # -- Main entry point --

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        """Main entry point called by the backtest engine."""
        self._call_count += 1
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n      -- V7风格开关策略第 {self._call_count} 期  {date_str} --")

        # Reset factor exposures
        self._last_factor_exposures = None

        # Step 0: Warmup
        if not self._warmup_done:
            self._warmup_done = True
            self._warmup_training_cache(date, accessor)

        self._append_walk_forward_data(date, accessor)
        if self._bulk_data is None or self._bulk_data.empty:
            print(f"      [数据] [X] 无可用行情缓存，本期跳过")
            return {}

        # Step 1: Check stop-loss for current holdings
        if current_holdings:
            prices = accessor.get_prices(date)
            stopped_out = self._stop_loss.update(current_holdings, prices)
            if stopped_out:
                print(f"      [止损] [!] 止损触发: {len(stopped_out)} 只 "
                      f"({', '.join(list(stopped_out)[:5])}...)")
        else:
            prices = accessor.get_prices(date)
            stopped_out = set()

        # Step 2: Compute features for both models
        print(f"      [特征] 计算 Model A ({len(SM_FEATURE_NAMES)}因子) + "
              f"Model B ({len(CS_FEATURE_NAMES_V5)}因子) ...")

        if self._bulk_data is not None:
            feat_a = sm_compute_features_v5(
                date, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            feat_b = cs_compute_features_v5(
                date, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
        else:
            feat_a = None
            feat_b = None

        if feat_a is None or feat_b is None:
            print(f"      [特征] [X] 数据不足，本期跳过")
            return {}

        # Step 3: Market cap filter
        n_a_before = len(feat_a)
        n_b_before = len(feat_b)
        feat_a = self._apply_market_cap_filter(feat_a)
        feat_b = self._apply_market_cap_filter(feat_b)
        print(f"      [特征] [OK] Model A: {n_a_before}→{len(feat_a)} | "
              f"Model B: {n_b_before}→{len(feat_b)}")

        if len(feat_a) < 50 or len(feat_b) < 50:
            print(f"      [特征] [X] 过滤后不足 50 只，本期跳过")
            return {}

        # Step 4: Rank normalize
        feat_a_ranked = rank_normalize(feat_a, SM_FEATURE_NAMES_V5)
        feat_b_ranked = rank_normalize(feat_b, CS_FEATURE_NAMES_V5)

        # Step 5: Update training caches with previous period's label
        for cache, feat_cache_list in [
            (self._train_cache_a, self._train_cache_a),
            (self._train_cache_b, self._train_cache_b),
        ]:
            if feat_cache_list:
                last_cached = feat_cache_list[-1]
                last_date = pd.Timestamp(last_cached[0])
                if last_date < date:
                    if self._bulk_data is not None:
                        fwd_ret = sm_compute_forward_return(
                            last_date, date, self._bulk_data
                        )
                    else:
                        fwd_ret = None
                    if fwd_ret is not None:
                        feat_cache_list[-1] = (
                            last_cached[0], last_cached[1], fwd_ret
                        )

        # Cache current features
        date_str_cache = date.strftime("%Y-%m-%d")
        cached_dates_a = {d for d, _, _ in self._train_cache_a}
        cached_dates_b = {d for d, _, _ in self._train_cache_b}
        if date_str_cache not in cached_dates_a:
            self._train_cache_a.append(
                (date_str_cache, feat_a_ranked.copy(), pd.Series(dtype=float))
            )
        if date_str_cache not in cached_dates_b:
            self._train_cache_b.append(
                (date_str_cache, feat_b_ranked.copy(), pd.Series(dtype=float))
            )

        # Step 6: Train or reuse models
        n_cached_a = len([1 for d, _, l in self._train_cache_a
                          if not l.empty and pd.Timestamp(d) < date])
        n_cached_b = len([1 for d, _, l in self._train_cache_b
                          if not l.empty and pd.Timestamp(d) < date])

        if self._should_retrain() and n_cached_a >= 8 and n_cached_b >= 8:
            print(f"      [训练] 触发重训练 "
                  f"(Model A: {n_cached_a}期, Model B: {n_cached_b}期) ...")
            new_a = self._train_sub_model(
                "A", self._train_cache_a, SM_FEATURE_NAMES_V5,
                sm_train_model_v5, date,
            )
            new_b = self._train_sub_model(
                "B", self._train_cache_b, CS_FEATURE_NAMES_V5,
                cs_train_model_v5, date,
            )
            if new_a is not None:
                self._model_a = new_a
            if new_b is not None:
                self._model_b = new_b
            self._last_train_date = date
        elif self._model_a is not None and self._model_b is not None:
            print(f"      [训练] 沿用现有模型")
        else:
            print(f"      [训练] 数据不足，等待积累 "
                  f"(A:{n_cached_a}, B:{n_cached_b})")

        # Step 7: Predict with both models
        def _predict_model(model, feat_ranked, feature_names, model_id):
            if model is None:
                scores = feat_ranked[feature_names].mean(axis=1).values
                codes = feat_ranked["ts_code"].values
                print(f"      [预测-{model_id}] 冷启动模式 (因子均值)")
                return scores, codes
            else:
                X_pred = feat_ranked[feature_names].copy()
                for col in feature_names:
                    med = X_pred[col].median()
                    X_pred[col] = X_pred[col].fillna(med)
                preds = model.predict(X_pred)
                codes = feat_ranked["ts_code"].values
                print(f"      [预测-{model_id}] 分数=[{preds.min():.4f}, {preds.max():.4f}]")
                return preds, codes

        score_a, codes_a = _predict_model(
            self._model_a, feat_a_ranked, SM_FEATURE_NAMES_V5, "A"
        )
        score_b, codes_b = _predict_model(
            self._model_b, feat_b_ranked, CS_FEATURE_NAMES_V5, "B"
        )

        codes_a_idx = pd.Index(codes_a)
        codes_b_idx = pd.Index(codes_b)

        # Step 7.5 (V4): Compute industry momentum
        industry_momentum = self._compute_industry_momentum(date)
        if industry_momentum is not None:
            top3 = industry_momentum.head(3)["industry"].tolist()
            bot3 = industry_momentum.tail(3)["industry"].tolist()
            print(
                f"      [V7行业动量] {len(industry_momentum)} 个行业  "
                f"强势: {', '.join(top3)}  弱势: {', '.join(bot3)}"
            )
        else:
            print(f"      [V7行业动量] 数据不足，本期跳过行业动量")

        # Build industry map for ensemble scoring
        if "sw_l1" in feat_a_ranked.columns and "ts_code" in feat_a_ranked.columns:
            industry_map = feat_a_ranked.set_index("ts_code")["sw_l1"].to_dict()
        else:
            industry_map = None

        # Step 8: Compute ensemble score (with V4 industry momentum)
        log_cap = feat_a_ranked.set_index("ts_code")["log_cap"] if "log_cap" in feat_a_ranked.columns else None

        ensemble_scores = self._compute_ensemble_score(
            score_a, score_b, codes_a_idx, codes_b_idx, log_cap,
            industry_map=industry_map,
            industry_momentum=industry_momentum,
        )

        if ensemble_scores.empty:
            print(f"      [融合] [X] 无交集股票")
            return {}

        # Also compute individual rank for consensus filtering
        s_a = pd.Series(score_a, index=codes_a_idx)
        s_b = pd.Series(score_b, index=codes_b_idx)
        common = ensemble_scores.index
        rank_a = s_a.reindex(common).rank(pct=True, method="average")
        rank_b = s_b.reindex(common).rank(pct=True, method="average")

        print(f"      [融合] [OK] 融合 {len(ensemble_scores)} 只，"
              f"score=[{ensemble_scores.min():.4f}, {ensemble_scores.max():.4f}]")

        # Step 9: Style regime gate (binary exposure)
        rule_name, _ = self._get_style_rule_meta()
        is_small_cap_regime, style_detail = self._compute_style_signal(date)
        position_coeff = 1.0 if is_small_cap_regime else 0.0
        final_coeff = position_coeff
        effective_max = self.max_positions if is_small_cap_regime else 0
        self._cash_ratio = 1.0 - final_coeff

        regime_label = "小盘占优" if is_small_cap_regime else "非小盘占优"
        print(
            f"      [风格开关] mode={self.style_timing_mode} ({rule_name})  "
            f"判断={regime_label}  细节: {style_detail}"
        )

        if not is_small_cap_regime:
            for code in list(current_holdings.keys() if current_holdings else []):
                self._stop_loss.record_exit(code)
            self._holding_periods.clear()
            self._last_factor_exposures = pd.DataFrame()
            print("      [风格开关] 当前非小盘占优，直接空仓跳过")
            return {}

        print(
            f"      [风格开关] 小盘占优，满仓运行原 V7 选股逻辑  "
            f"有效持仓上限={effective_max}  现金比例={self._cash_ratio:.0%}"
        )

        # Step 10: Select and weight (with V4 dynamic industry constraint)
        weights = self._select_and_weight(
            ensemble_scores, rank_a, rank_b,
            feat_a_ranked, current_holdings,
            effective_max, stopped_out,
            industry_momentum=industry_momentum,
        )

        # Step 10.5 (V4): Cache factor exposures for attribution
        if weights:
            exposure_rows = []
            # Prepare per-stock factor data
            score_a_series = pd.Series(score_a, index=codes_a_idx)
            score_b_series = pd.Series(score_b, index=codes_b_idx)
            # Rank-normalize model scores to [0, 1] for attribution
            rank_a_full = score_a_series.rank(pct=True, method="average")
            rank_b_full = score_b_series.rank(pct=True, method="average")

            # Industry momentum rank for each stock (normalized to [0, 1])
            ind_mom_rank_map = {}
            if industry_momentum is not None and industry_map is not None:
                n_ind = len(industry_momentum)
                for _, row in industry_momentum.iterrows():
                    # rank 1 = strongest → normalized to 1.0
                    ind_mom_rank_map[row["industry"]] = 1.0 - (row["rank"] - 1) / max(n_ind - 1, 1)

            # Small-cap indicator
            small_cap_map = {}
            if log_cap is not None:
                cap_threshold = log_cap.quantile(self.small_cap_quantile)
                for code in log_cap.index:
                    small_cap_map[code] = 1.0 if log_cap[code] <= cap_threshold else 0.0

            for code, w in weights.items():
                row = {
                    "ts_code": code,
                    "weight": w,
                    "model_a_score": rank_a_full.get(code, 0.5),
                    "model_b_score": rank_b_full.get(code, 0.5),
                    "small_cap": small_cap_map.get(code, 0.0),
                    "industry_momentum": ind_mom_rank_map.get(
                        industry_map.get(code, "") if industry_map else "", 0.5
                    ),
                    "market_state": position_coeff,
                }
                exposure_rows.append(row)

            self._last_factor_exposures = pd.DataFrame(exposure_rows)

        # Step 10.7 (V7): Cash-aware weight scaling
        # Scale all weights by final_coeff so sum(weights) = final_coeff, not 1.0
        # This means the engine will hold cash for the remainder
        if weights and final_coeff < 1.0:
            for code in weights:
                weights[code] *= final_coeff

        # Step 11: Update stop-loss tracker
        new_codes = set(weights.keys())
        old_codes = set(current_holdings.keys()) if current_holdings else set()
        for code in new_codes - old_codes:
            if code in prices:
                self._stop_loss.record_entry(code, prices[code])
        for code in old_codes - new_codes:
            self._stop_loss.record_exit(code)

        # Step 11.5 (V7): Update holding period tracker
        for code in new_codes:
            if code in old_codes:
                self._holding_periods[code] = self._holding_periods.get(code, 0) + 1
            else:
                self._holding_periods[code] = 1
        # Clean up exited stocks
        for code in list(self._holding_periods.keys()):
            if code not in new_codes:
                del self._holding_periods[code]

        # Print summary
        n_held = len(current_holdings) if current_holdings else 0
        overlap = len(new_codes & old_codes)
        ind_map_for_summary = (
            feat_a_ranked.set_index("ts_code")["sw_l1"].to_dict()
            if "sw_l1" in feat_a_ranked.columns else {}
        )
        n_industries = len(set(
            ind_map_for_summary.get(c, "unknown") for c in weights.keys()
        )) if weights else 0

        if len(weights) == 0:
            print(f"      [选股] [!] 本期无满足条件的股票，空仓")
        else:
            top_w = max(weights.values()) * 100
            avg_w = np.mean(list(weights.values())) * 100

            # V4: Log dynamic industry constraint info
            if industry_momentum is not None:
                strong_set = set(
                    industry_momentum.head(self.industry_strong_top_n)["industry"]
                )
                weak_set = set(
                    industry_momentum.tail(self.industry_weak_top_n)["industry"]
                )
                n_strong = sum(
                    1 for c in weights
                    if ind_map_for_summary.get(c) in strong_set
                )
                n_weak = sum(
                    1 for c in weights
                    if ind_map_for_summary.get(c) in weak_set
                )
                ind_info = f"  强势行业{n_strong}只/弱势行业{n_weak}只"
            else:
                ind_info = ""

            print(
                f"      [选股] 入选 {len(weights)} 只 / {n_industries} 个行业  "
                f"(上期 {n_held}，留存 {overlap}，"
                f"换手 {n_held + len(weights) - 2*overlap}){ind_info}  "
                f"权重: max={top_w:.1f}% avg={avg_w:.1f}%"
            )

        return weights


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V7 风格开关版：通过一个参数在三种小盘风格判断规则之间切换。"
    )
    parser.add_argument(
        "--style-timing-mode",
        default="csi1000_score",
        choices=["csi500_trend", "csi1000_trend", "csi1000_score"],
        help="风格判断模式：判断为小盘占优时满仓，否则空仓。",
    )
    args = parser.parse_args()

    cfg = BacktestConfig(
        initial_capital=300_000,
        commission_rate=1.5e-4,
        slippage=0.0015,
        start_date="2018-01-01",
        end_date="2025-12-31",
        rebalance_freq="BW",
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir="data/quant/backtest",
    )

    strategy = LGBMEnsembleAdaptiveV7D(
        # Model parameters
        train_window_years=3,
        retrain_interval=4,
        # Ensemble parameters
        weight_model_a=0.6,
        weight_model_b=0.4,
        consensus_top_pct=0.05,
        consensus_single_top_pct=0.10,
        # Position parameters
        max_positions=25,
        # Weight parameters
        softmax_temperature=5.0,
        max_single_weight=0.08,
        # Risk control parameters
        max_per_industry=3,
        stop_loss_threshold=-0.12,
        stop_loss_cooldown=2,
        buffer_sigma=1.0,
        min_holding_periods=2,
        drawdown_circuit_breaker=-0.15,
        drawdown_reduction_factor=0.5,
        # Stock pool parameters
        mv_pct_upper=0.85,
        small_cap_bonus=0.02,
        small_cap_quantile=0.70,
        feature_lookback=260,
        backtest_end_date=cfg.end_date,
        # V4: Industry momentum parameters
        industry_momentum_lookback_months=6,
        industry_momentum_bonus=0.02,
        industry_momentum_penalty=-0.01,
        industry_strong_top_n=10,
        industry_weak_top_n=10,
        industry_strong_max=5,
        industry_weak_max=2,
        # Style timing parameters
        baseline_dir=cfg.baseline_dir,
        style_timing_mode=args.style_timing_mode,
        style_short_window=20,
        style_long_window=60,
        style_return_window=20,
        style_return_threshold=0.04,
    )
    result = run_backtest(strategy, cfg)