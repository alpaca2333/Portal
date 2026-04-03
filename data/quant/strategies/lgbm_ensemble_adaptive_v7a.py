"""
LightGBM Ensemble Adaptive Strategy V7a
========================================

V7a is a **bugfix release** of V7. V7's Cash-Aware Position Sizing was
completely ineffective because the backtest engine's broker.rebalance()
re-normalizes weights to sum=1.0, undoing the cash ratio scaling.

**V7a Fix**: Instead of scaling weights (which gets undone by the broker),
inject a virtual "CASH" position into target_weights. The broker treats it
as a real position but since there's no matching price, it skips the buy,
leaving the capital as cash. After broker processes, CASH is removed.
This approach is broker-compatible without modifying the engine code.

--- Original V7 description below ---

V7 focuses on **risk control and turnover reduction**, addressing three key
weaknesses identified in V5's deep analysis:
  - V5 had a devastating -22.7% excess drawdown in 2024 H1 (小微盘流动性踩踏)
  - V5's annual turnover was ~1544%, with 41.5% stocks held only 1 period
  - V5's market state coefficient only controlled position count, not capital

Improvements over V5:

### P0: Capital Allocation & Drawdown Protection
1. **Cash-Aware Position Sizing**: Market state coefficient now controls
   ACTUAL capital allocation (cash ratio), not just number of positions.
   E.g. weak_bear coeff=0.3 means 30% invested + 70% cash, rather than
   just reducing from 25 to 7 positions while still being fully invested.
   This directly addresses the finding that V5's 择时 only managed position
   count but not capital exposure.

2. **Relaxed Panic Trigger**: Lowered panic threshold from
   (downtrend + breadth < 30% + high_vol > 1.5) to
   (downtrend + breadth < 15%), removing the vol requirement.
   Rationale: V5 missed the 2024-Feb crash because vol hadn't spiked yet
   when breadth was already collapsing. Now triggers earlier.

3. **Portfolio-Level Drawdown Circuit Breaker**: When portfolio NAV drops
   >15% from peak, force-reduce allocation to 50% of normal level.
   Recovery condition: drawdown < 5%. This is a direct response to the
   2024 H1 disaster where V5 lost -22.7% excess without any portfolio-
   level risk override.

### P1: Turnover Reduction
4. **Minimum Holding Period**: Stocks must be held for at least 2 periods
   (4 weeks) before being eligible for replacement. Implemented via a
   3x buffer boost for stocks below min holding period. This targets V5's
   41.5% single-period holding rate.

5. **Increased Turnover Buffer**: buffer_sigma raised from 0.5 → 1.0,
   doubling the score boost for existing holdings. Combined with #4,
   this aims to cut annual turnover from ~1544% to <1000%.

6. **Tighter Stop-Loss**: Individual stock stop-loss tightened from -20%
   to -12% (aligned with code default). Cuts losses faster on single
   positions, complementing the portfolio-level circuit breaker (#3).

### Design Philosophy
V7 is a "risk-first" iteration. V5 optimized for alpha generation (new
factors, industry-neutral labels); V7 optimizes for alpha PRESERVATION
by reducing the three biggest cost centers: excessive turnover (交易成本),
uncontrolled drawdowns (回撤), and binary position sizing (仓位管理).

Inherited from V5:
- 3 behavioral alpha factors: ivol, max_ret_20d, skew_20d
- Continuous rank-normalized industry momentum scoring
- Industry-neutral labels (subtract industry median before training)
- Optimized model weights: SM:CS = 0.6:0.4
- Dual LightGBM ensemble (Smart Money + Cross-Sectional)
- Industry momentum scoring + dynamic industry constraint
- Three-dimensional market state adaptive positioning
- Softmax weighting, stop-loss with cooldown, turnover buffer
- Factor attribution interface

Key Parameter Changes (V5 → V7):
- buffer_sigma:          0.5 → 1.0
- stop_loss_threshold:  -0.20 → -0.12
- min_holding_periods:   N/A → 2 (new)
- drawdown_circuit_breaker: N/A → -0.15 (new)
- drawdown_reduction_factor: N/A → 0.50 (new)
- panic trigger:        breadth<30%+highvol → breadth<15%

Usage
-----
cd /data/Projects/Portal
python -m data.quant.strategies.lgbm_ensemble_adaptive_v7a
"""
import sys
import os
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

class LGBMEnsembleAdaptiveV7a(StrategyBase):
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
    ):
        super().__init__("lgbm_ensemble_adaptive_v7")

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
        return (
            f"### 策略思路\n\n"
            f"自适应集成选股 V7a（仓位管理修复版）：修复V7中现金仓位管理未生效的严重Bug。"
            f"V7的Cash-Aware Sizing代码虽然将权重缩放到sum=final_coeff，但回测引擎"
            f"broker.rebalance()会将权重重新归一化回sum=1.0，导致现金比例完全失效。"
            f"V7a通过注入虚拟CASH持仓占位的方式绕过引擎归一化，真正实现现金仓位管理。\n\n"
            f"### V7a 修复项\n\n"
            f"- **Cash-Aware真正生效**：弱熊coeff=0.3时，策略只将30%资金分配给股票，"
            f"70%保持现金。V7中这个功能完全无效（被引擎归一化抵消）\n"
            f"- **修复方式**：在target_weights中注入虚拟CASH条目（权重=1-final_coeff），"
            f"broker因找不到CASH的价格而跳过买入，资金自然留为现金\n\n"
            f"1. **现金仓位管理 (Cash-Aware Sizing)**：仓位系数不仅控制持仓数量，"
            f"还控制实际资金分配比例。弱熊coeff=0.3意味着仅30%资金投入+70%现金，"
            f"而非V5的仅减少持仓数但仍满仓运作\n"
            f"2. **放宽恐慌触发**：恐慌模式从'下跌+广度<30%+高波动>1.5'放宽为"
            f"'下跌+广度<15%'，移除波动率要求。因为2024年2月崩盘时波动率尚未飙升，"
            f"但广度已经崩溃，V5未能及时触发恐慌\n"
            f"3. **组合级回撤熔断**：组合净值从峰值回撤>{abs(self.drawdown_circuit_breaker)*100:.0f}%时，"
            f"仓位强制缩减至{self.drawdown_reduction_factor*100:.0f}%。"
            f"恢复条件：回撤<5%。直接应对V5在2024H1无组合级风控覆盖的问题\n\n"
            f"#### P1: 换手率控制\n"
            f"4. **最小持仓期**：持仓不满{self.min_holding_periods}期（{self.min_holding_periods*2}周）"
            f"的股票获得3倍换手缓冲加分，大幅降低短期换手。"
            f"目标：将V5的41.5%单期持仓率显著降低\n"
            f"5. **增强换手缓冲**：buffer_sigma从0.5提高到{self.buffer_sigma}，"
            f"加倍惩罚不必要的换仓\n"
            f"6. **收紧个股止损**：从V5的-20%收紧至{abs(self.stop_loss_threshold)*100:.0f}%，"
            f"与组合熔断互补\n\n"
            f"### V5 继承项（因子层+模型层）\n\n"
            f"- 3个行为Alpha因子：ivol（特质波动率）、max_ret_20d（彩票因子）、skew_20d（偏度因子）\n"
            f"- 连续化行业动量（rank-normalized评分，scale={self.industry_momentum_bonus}）\n"
            f"- 行业中性标签（训练时减去行业中位数收益）\n"
            f"- 双模型融合：Smart Money ({self.weight_model_a}) + "
            f"Cross-Sectional ({self.weight_model_b})\n"
            f"- 共识过滤（综合前{self.consensus_top_pct*100:.0f}% 且至少一个模型前"
            f"{self.consensus_single_top_pct*100:.0f}%）\n"
            f"- 市场状态择时：五档仓位(1.0/0.8/0.5/0.3/0.1) + 现金比例控制\n"
            f"- softmax 加权 (temperature={self.softmax_temperature})，"
            f"单股上限{self.max_single_weight*100:.0f}%\n"
            f"- 个股止损(-{abs(self.stop_loss_threshold)*100:.0f}%) + 冷却 + 换手缓冲\n"
        )

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
        print(f"\n      -- V7a集成策略第 {self._call_count} 期  {date_str} --")

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

        # Step 9: Market state assessment
        all_dates_bulk, date_to_rows = sm_module._get_bulk_date_index(self._bulk_data)
        date_ts = pd.Timestamp(date)
        valid_dates = all_dates_bulk[all_dates_bulk <= date_ts]
        window_60_dates = valid_dates[-60:] if len(valid_dates) >= 60 else valid_dates

        row_indices = []
        for wd in window_60_dates:
            if wd in date_to_rows:
                row_indices.append(date_to_rows[wd])

        if row_indices:
            all_row_idx = np.concatenate(row_indices)
            window_data = self._bulk_data.iloc[all_row_idx]
            close_pivot = window_data.pivot_table(
                index="trade_date", columns="ts_code", values="close"
            ).sort_index()
            position_coeff = self._compute_market_state(close_pivot)
        else:
            position_coeff = 0.5

        # V7: Portfolio-level drawdown circuit breaker
        # Use accessor to get current portfolio NAV approximation
        current_nav = 0
        if current_holdings and prices:
            for code, shares in current_holdings.items():
                if code in prices and prices[code] > 0:
                    current_nav += shares * prices[code]
        if current_nav > 0:
            self._peak_nav = max(self._peak_nav, current_nav)
            if self._peak_nav > 0:
                dd_from_peak = current_nav / self._peak_nav - 1
                if dd_from_peak < self.drawdown_circuit_breaker:
                    if not self._circuit_breaker_active:
                        print(f"      [V7熔断] ⚠️ 组合回撤 {dd_from_peak:.1%} 触发熔断！"
                              f"仓位削减至 {self.drawdown_reduction_factor:.0%}")
                    self._circuit_breaker_active = True
                elif dd_from_peak >= -0.05:  # recover when drawdown < 5%
                    if self._circuit_breaker_active:
                        print(f"      [V7熔断] ✅ 组合回撤已恢复至 {dd_from_peak:.1%}，解除熔断")
                    self._circuit_breaker_active = False
        elif self._peak_nav == 0:
            # First period, initialize peak
            pass

        # V7: Apply circuit breaker reduction to position coefficient
        final_coeff = position_coeff
        if self._circuit_breaker_active:
            final_coeff = position_coeff * self.drawdown_reduction_factor

        # V7: Cash-aware position sizing
        # effective_max controls number of positions, but we also scale weights
        # to achieve actual cash ratio = final_coeff
        effective_max = max(2, int(self.max_positions * final_coeff))
        self._cash_ratio = 1.0 - final_coeff  # store for weight scaling

        state_labels = {
            1.0: "强牛", 0.8: "普牛", 0.5: "震荡", 0.3: "弱熊", 0.1: "恐慌"
        }
        state_name = state_labels.get(position_coeff, f"coeff={position_coeff}")
        breaker_str = " [熔断中]" if self._circuit_breaker_active else ""
        print(f"      [择时] 市场状态={state_name}  "
              f"仓位系数={position_coeff}→{final_coeff:.2f}{breaker_str}  "
              f"有效持仓上限={effective_max}  现金比例={self._cash_ratio:.0%}")

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

        # Step 10.7 (V7a): Cash-aware position sizing via virtual CASH entry
        # V7 BUG: Scaling weights to sum=final_coeff was undone by broker's
        # re-normalization (broker.rebalance normalises weights to sum=1.0).
        # FIX: Inject a virtual "CASH" entry so the broker sees sum=1.0 but
        # allocates (1-final_coeff) to CASH. Since there's no price for CASH,
        # the broker skips the buy, leaving that capital as real cash.
        if weights and final_coeff < 1.0:
            cash_weight = 1.0 - final_coeff
            # Scale stock weights so stocks + CASH = 1.0
            stock_total = sum(weights.values())
            if stock_total > 0:
                scale = final_coeff / stock_total
                for code in weights:
                    weights[code] *= scale
            weights["CASH"] = cash_weight
            print(f"      [V7a] Cash-Aware: 股票权重总和={final_coeff:.2f}, "
                  f"现金占位={cash_weight:.2f}, 合计={sum(weights.values()):.2f}")

        # Step 11: Update stop-loss tracker
        # (exclude virtual CASH from stop-loss / holding period tracking)
        new_codes = set(k for k in weights.keys() if k != "CASH")
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

        # Print summary (exclude virtual CASH from display)
        real_weights = {k: v for k, v in weights.items() if k != "CASH"}
        n_held = len(current_holdings) if current_holdings else 0
        overlap = len(new_codes & old_codes)
        ind_map_for_summary = (
            feat_a_ranked.set_index("ts_code")["sw_l1"].to_dict()
            if "sw_l1" in feat_a_ranked.columns else {}
        )
        n_industries = len(set(
            ind_map_for_summary.get(c, "unknown") for c in real_weights.keys()
        )) if real_weights else 0

        if len(real_weights) == 0:
            print(f"      [选股] [!] 本期无满足条件的股票，空仓")
        else:
            top_w = max(real_weights.values()) * 100
            avg_w = np.mean(list(real_weights.values())) * 100

            # V4: Log dynamic industry constraint info
            if industry_momentum is not None:
                strong_set = set(
                    industry_momentum.head(self.industry_strong_top_n)["industry"]
                )
                weak_set = set(
                    industry_momentum.tail(self.industry_weak_top_n)["industry"]
                )
                n_strong = sum(
                    1 for c in real_weights
                    if ind_map_for_summary.get(c) in strong_set
                )
                n_weak = sum(
                    1 for c in real_weights
                    if ind_map_for_summary.get(c) in weak_set
                )
                ind_info = f"  强势行业{n_strong}只/弱势行业{n_weak}只"
            else:
                ind_info = ""

            cash_info = f"  现金={weights.get('CASH', 0)*100:.0f}%" if "CASH" in weights else ""
            print(
                f"      [选股] 入选 {len(real_weights)} 只 / {n_industries} 个行业  "
                f"(上期 {n_held}，留存 {overlap}，"
                f"换手 {n_held + len(real_weights) - 2*overlap}){ind_info}  "
                f"权重: max={top_w:.1f}% avg={avg_w:.1f}%{cash_info}"
            )

        return weights


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
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

    strategy = LGBMEnsembleAdaptiveV7a(
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
        buffer_sigma=1.0,         # V7: increased from 0.5 to reduce turnover
        min_holding_periods=2,    # V7: minimum 2 periods (4 weeks) holding
        drawdown_circuit_breaker=-0.15,   # V7: trigger at -15% drawdown
        drawdown_reduction_factor=0.5,    # V7: reduce to 50% allocation
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
    )
    result = run_backtest(strategy, cfg)
