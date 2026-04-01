"""
LightGBM Ensemble Adaptive Strategy V2
=======================================

Improvements over V1:
1. Volatility targeting replaces discrete 5-level position sizing
2. Triple trend confirmation (short/mid/long) replaces single MA20/MA60
3. Adaptive stop-loss based on individual stock volatility
4. Dynamic style factor (large/small cap adaptation)
5. Training optimization: 4yr window, sample weighting, 6-period retrain
6. Portfolio construction: dynamic softmax temperature, relaxed industry constraint

Stock selection layer (dual-model 44 factors + consensus filter) is unchanged from V1.

Usage
-----
cd D:\\Projects\\Portal
python -m data.quant.strategies.lgbm_ensemble_adaptive_v2
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


# ===================================================================
# Adaptive stop-loss tracker (V2: volatility-based thresholds)
# ===================================================================

class AdaptiveStopLoss:
    """
    Adaptive stop-loss: threshold = max(fixed_floor, -k * stock_vol).
    High-vol stocks get wider stop bands; low-vol stocks get tighter ones.
    """

    def __init__(
        self,
        vol_multiplier: float = 2.5,
        min_threshold: float = -0.08,
        max_threshold: float = -0.20,
        cooldown_periods: int = 2,
    ):
        self.vol_multiplier = vol_multiplier
        self.min_threshold = min_threshold   # tightest (closest to 0)
        self.max_threshold = max_threshold   # widest (most negative)
        self.cooldown = cooldown_periods
        self.entry_prices: Dict[str, float] = {}
        self.entry_vols: Dict[str, float] = {}
        self.cooldown_map: Dict[str, int] = {}

    def compute_threshold(self, stock_vol_annual: float) -> float:
        """Compute stop-loss threshold from annualized vol."""
        holding_period_vol = stock_vol_annual * np.sqrt(10) / np.sqrt(252)
        threshold = -self.vol_multiplier * holding_period_vol
        return np.clip(threshold, self.max_threshold, self.min_threshold)

    def update(
        self,
        current_holdings: Dict[str, int],
        prices: Dict[str, float],
    ) -> set:
        """Check stop-loss for current holdings. Returns stopped-out codes."""
        stopped_out = set()
        for code in current_holdings:
            if code in self.entry_prices and code in prices:
                ret = prices[code] / self.entry_prices[code] - 1
                vol = self.entry_vols.get(code, 0.30)
                threshold = self.compute_threshold(vol)
                if ret < threshold:
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
            self.entry_prices.pop(code, None)
            self.entry_vols.pop(code, None)

        return stopped_out

    def is_in_cooldown(self, code: str) -> bool:
        return code in self.cooldown_map

    def record_entry(self, code: str, price: float, annual_vol: float = 0.30):
        if code not in self.entry_prices:
            self.entry_prices[code] = price
            self.entry_vols[code] = annual_vol

    def record_exit(self, code: str):
        self.entry_prices.pop(code, None)
        self.entry_vols.pop(code, None)


# ===================================================================
# Strategy class
# ===================================================================

class LGBMEnsembleAdaptiveV2(StrategyBase):
    """
    V2: Volatility targeting + multi-level timing + adaptive stop-loss.
    Stock selection layer (dual LightGBM + consensus) unchanged from V1.
    """

    def __init__(
        self,
        # Model parameters (V2 changes)
        train_window_years: int = 4,
        retrain_interval: int = 6,
        sample_weight_halflife: int = 12,
        # Ensemble parameters (unchanged)
        weight_model_a: float = 0.6,
        weight_model_b: float = 0.4,
        consensus_top_pct: float = 0.05,
        consensus_single_top_pct: float = 0.10,
        # Position parameters (V2 changes)
        max_positions: int = 25,
        target_vol: float = 0.18,
        vol_lookback: int = 40,
        vol_scale_min: float = 0.15,
        vol_scale_max: float = 1.20,
        vol_smoothing: float = 0.7,
        # Weight parameters (V2: dynamic temperature)
        softmax_temperature_bull: float = 6.0,
        softmax_temperature_neutral: float = 4.0,
        softmax_temperature_bear: float = 2.0,
        max_single_weight: float = 0.08,
        # Risk control parameters (V2 changes)
        max_per_industry: int = 4,
        min_industries: int = 4,
        stop_loss_vol_multiplier: float = 2.5,
        stop_loss_min: float = -0.08,
        stop_loss_max: float = -0.20,
        stop_loss_cooldown: int = 2,
        buffer_sigma: float = 0.5,
        # Stock pool parameters (V2: style adaptation)
        mv_pct_upper: float = 0.85,
        style_threshold: float = 0.05,
        feature_lookback: int = 260,
        backtest_end_date: Optional[str] = None,
    ):
        super().__init__("lgbm_ensemble_adaptive_v2")

        # Model parameters
        self.train_window_years = train_window_years
        self.retrain_interval = retrain_interval
        self.sample_weight_halflife = sample_weight_halflife

        # Ensemble parameters
        self.weight_model_a = weight_model_a
        self.weight_model_b = weight_model_b
        self.consensus_top_pct = consensus_top_pct
        self.consensus_single_top_pct = consensus_single_top_pct

        # Position parameters
        self.max_positions = max_positions
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.vol_scale_min = vol_scale_min
        self.vol_scale_max = vol_scale_max
        self.vol_smoothing = vol_smoothing

        # Weight parameters
        self.softmax_temperature_bull = softmax_temperature_bull
        self.softmax_temperature_neutral = softmax_temperature_neutral
        self.softmax_temperature_bear = softmax_temperature_bear
        self.max_single_weight = max_single_weight

        # Risk control parameters
        self.max_per_industry = max_per_industry
        self.min_industries = min_industries
        self.buffer_sigma = buffer_sigma

        # Stock pool parameters
        self.mv_pct_upper = mv_pct_upper
        self.style_threshold = style_threshold
        self.feature_lookback = feature_lookback
        self._backtest_end_date = (
            pd.Timestamp(backtest_end_date) if backtest_end_date else None
        )

        # -- Sub-model states --
        self._model_a: Optional[lgb.Booster] = None
        self._model_b: Optional[lgb.Booster] = None
        self._train_cache_a: List[Tuple[str, pd.DataFrame, pd.Series]] = []
        self._train_cache_b: List[Tuple[str, pd.DataFrame, pd.Series]] = []
        self._last_train_date: Optional[pd.Timestamp] = None

        # -- Strategy states --
        self._call_count = 0
        self._warmup_done = False
        self._bulk_data: Optional[pd.DataFrame] = None
        self._bulk_last_date: Optional[pd.Timestamp] = None
        self._st_codes: Optional[set] = None

        # V2: Adaptive stop-loss
        self._stop_loss = AdaptiveStopLoss(
            vol_multiplier=stop_loss_vol_multiplier,
            min_threshold=stop_loss_min,
            max_threshold=stop_loss_max,
            cooldown_periods=stop_loss_cooldown,
        )

        # V2: Volatility targeting state
        self._prev_vol_scale = 1.0
        self._portfolio_returns: List[float] = []
        self._prev_weights: Optional[Dict[str, float]] = None
        self._prev_prices: Optional[Dict[str, float]] = None

        # Factor attribution cache
        self._last_factor_exposures: Optional[pd.DataFrame] = None

    def describe(self) -> str:
        return (
            f"### 策略思路\n\n"
            f"自适应集成选股 V2：波动率目标制 + 多级择时 + 自适应止损。"
            f"基于 V1 改进，核心变化在仓位管理和风控层，选股层沿用 V1。\n\n"
            f"### 核心改进\n\n"
            f"1. **波动率目标制**：目标年化波动率 {self.target_vol*100:.0f}%，"
            f"连续平滑调仓，替代 V1 离散五档仓位\n"
            f"2. **三级趋势确认**：短期(MA5/MA20) + 中期(MA20/MA60) + "
            f"长期(新高新低比率)，2级以上确认才判定牛/熊\n"
            f"3. **自适应止损**：阈值 = f(个股波动率)，"
            f"范围 [{self._stop_loss.max_threshold*100:.0f}%, "
            f"{self._stop_loss.min_threshold*100:.0f}%]\n"
            f"4. **风格自适应**：动态 small_cap_bonus，根据大/小盘 60 日累计收益差切换\n"
            f"5. **训练优化**：{self.train_window_years}年窗口 + 指数衰减样本加权"
            f"(半衰期{self.sample_weight_halflife}月) + 每{self.retrain_interval}期重训练\n"
            f"6. **组合优化**：动态 softmax 温度(牛{self.softmax_temperature_bull}/"
            f"中{self.softmax_temperature_neutral}/熊{self.softmax_temperature_bear})，"
            f"行业上限{self.max_per_industry}只，最少{self.min_industries}个行业\n"
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

    # ── V2: New market analysis methods ──

    def _compute_market_regime(self, close_pivot: pd.DataFrame) -> str:
        """
        Triple trend confirmation:
        Level 1: Short-term - MA5 vs MA20
        Level 2: Mid-term - MA20 vs MA60
        Level 3: Long-term - 60d new high/low ratio
        Requires 2+ levels to confirm bull/bear.
        """
        n = len(close_pivot)
        if n < 60:
            return "neutral"

        market_median = close_pivot.median(axis=1)

        # Level 1: Short-term trend
        ma5 = market_median.rolling(5).mean().iloc[-1]
        ma20 = market_median.rolling(20).mean().iloc[-1]
        short_up = ma5 > ma20

        # Level 2: Mid-term trend
        ma60 = market_median.rolling(60).mean().iloc[-1]
        mid_up = ma20 > ma60

        # Level 3: Long-term trend - new high/low ratio
        high_60 = close_pivot.rolling(60).max()
        low_60 = close_pivot.rolling(60).min()
        last_close = close_pivot.iloc[-1]

        new_high_pct = (last_close >= high_60.iloc[-1] * 0.98).mean()
        new_low_pct = (last_close <= low_60.iloc[-1] * 1.02).mean()

        long_up = new_high_pct > new_low_pct + 0.05
        long_down = new_low_pct > new_high_pct + 0.05

        # Multi-level confirmation
        bull_score = sum([short_up, mid_up, long_up])
        bear_score = sum([not short_up, not mid_up, long_down])

        if bull_score >= 2:
            return "bull"
        elif bear_score >= 2:
            return "bear"
        else:
            return "neutral"

    def _compute_breadth_momentum(self, close_pivot: pd.DataFrame) -> float:
        """Breadth momentum: change in breadth over 5 days."""
        if len(close_pivot) < 25:
            return 0.0
        stock_ma20 = close_pivot.rolling(20).mean()
        breadth_current = (close_pivot.iloc[-1] > stock_ma20.iloc[-1]).mean()
        breadth_5d_ago = (close_pivot.iloc[-6] > stock_ma20.iloc[-6]).mean()
        return breadth_current - breadth_5d_ago

    def _compute_vol_target_scale(self) -> float:
        """Volatility targeting: compute position scale from recent portfolio returns."""
        if len(self._portfolio_returns) < 4:
            return 1.0

        returns = pd.Series(self._portfolio_returns)
        # Use EWM std; each return represents ~10 trading days
        period_vol = returns.ewm(span=self.vol_lookback // 10, min_periods=3).std().iloc[-1]
        if period_vol <= 0 or np.isnan(period_vol):
            return self._prev_vol_scale

        # Convert period vol to annualized: each period ~10 trading days
        daily_vol_approx = period_vol / np.sqrt(10)
        annualized_vol = daily_vol_approx * np.sqrt(252)

        if annualized_vol <= 0:
            return self._prev_vol_scale

        raw_scale = self.target_vol / annualized_vol
        smoothed_scale = self.vol_smoothing * self._prev_vol_scale + (1 - self.vol_smoothing) * raw_scale
        return np.clip(smoothed_scale, self.vol_scale_min, self.vol_scale_max)

    def _compute_style_factor(
        self, close_pivot: pd.DataFrame, circ_mv_series: pd.Series
    ) -> float:
        """
        Compute large/small cap style factor.
        Positive = large cap leading, negative = small cap leading.
        """
        if len(close_pivot) < 60:
            return 0.0

        mv_median = circ_mv_series.median()
        large_caps = circ_mv_series[circ_mv_series >= mv_median].index
        small_caps = circ_mv_series[circ_mv_series < mv_median].index

        ret_60 = close_pivot.iloc[-1] / close_pivot.iloc[-60] - 1

        large_ret = ret_60.reindex(large_caps).median()
        small_ret = ret_60.reindex(small_caps).median()

        if pd.isna(large_ret) or pd.isna(small_ret):
            return 0.0

        return large_ret - small_ret

    def _compute_sample_weights(self, dates: pd.Series) -> np.ndarray:
        """Exponential decay sample weights with configurable half-life."""
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.values.astype(float)
        half_life_days = self.sample_weight_halflife * 21
        weights = np.exp(-np.log(2) * days_ago / half_life_days)
        mean_w = weights.mean()
        if mean_w > 0:
            weights = weights / mean_w
        return weights

    def _compute_stock_vol(self, code: str, date: pd.Timestamp) -> float:
        """Compute annualized volatility for a single stock from bulk data."""
        if self._bulk_data is None:
            return 0.30
        stock_data = self._bulk_data[
            (self._bulk_data["ts_code"] == code) &
            (self._bulk_data["trade_date"] <= date)
        ].tail(30)
        if len(stock_data) < 10:
            return 0.30
        daily_ret = stock_data["close"].pct_change().dropna()
        if len(daily_ret) < 5:
            return 0.30
        return daily_ret.std() * np.sqrt(252)

    # ── Warmup and walk-forward (mostly from V1) ──

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

        # Bulk-load ALL data
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

        # Clear date index caches
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

        # Biweekly: group by 14-day blocks
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

            feat_a = sm_compute_features(
                d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            feat_b = cs_compute_features(
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

            feat_a_ranked = rank_normalize(feat_a, SM_FEATURE_NAMES)
            feat_b_ranked = rank_normalize(feat_b, CS_FEATURE_NAMES)

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
            feat_a = sm_compute_features(
                last_d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            feat_b = cs_compute_features(
                last_d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            if feat_a is not None and feat_b is not None:
                feat_a = self._apply_market_cap_filter(feat_a)
                feat_b = self._apply_market_cap_filter(feat_b)
                if len(feat_a) >= 50 and len(feat_b) >= 50:
                    feat_a_ranked = rank_normalize(feat_a, SM_FEATURE_NAMES)
                    feat_b_ranked = rank_normalize(feat_b, CS_FEATURE_NAMES)
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

        # Trim _bulk_data
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

        # Trim old data
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

    # ── V2: Modified training with sample weights ──

    def _train_sub_model(
        self,
        model_id: str,
        train_cache: List[Tuple[str, pd.DataFrame, pd.Series]],
        feature_names: List[str],
        train_fn,
        date: pd.Timestamp,
    ) -> Optional[lgb.Booster]:
        """Train a sub-model with V2 sample weighting."""
        valid_cache = [
            (d, f, l) for d, f, l in train_cache
            if not l.empty and pd.Timestamp(d) < date
        ]
        if len(valid_cache) < 8:
            print(f"      [训练-{model_id}] [X] 有效训练期不足 ({len(valid_cache)})")
            return None

        all_X = []
        all_y = []
        all_dates = []  # V2: track dates for sample weighting
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
                    all_X.append(merged[feature_names])
                    all_y.append(merged["label"])
                    all_dates.extend([dt] * len(merged))

        if len(all_X) < 8:
            print(f"      [训练-{model_id}] [X] 合并后训练期不足 ({len(all_X)})")
            return None

        n = len(all_X)
        split = max(1, int(n * 0.8))

        train_X = pd.concat(all_X[:split])
        train_y = pd.concat(all_y[:split])
        val_X = pd.concat(all_X[split:]) if split < n else None
        val_y = pd.concat(all_y[split:]) if split < n else None

        # V2: Compute sample weights for training set
        train_dates_count = sum(len(x) for x in all_X[:split])
        train_date_series = pd.Series(all_dates[:train_dates_count])
        sample_weights = self._compute_sample_weights(train_date_series)

        try:
            # Create weighted dataset
            train_data = lgb.Dataset(
                train_X, label=train_y,
                weight=sample_weights,
                feature_name=feature_names,
                free_raw_data=False,
            )
            if val_X is not None and val_y is not None:
                val_data = lgb.Dataset(
                    val_X, label=val_y,
                    reference=train_data,
                    feature_name=feature_names,
                    free_raw_data=False,
                )
            else:
                val_data = None

            params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
            }

            callbacks = [lgb.log_evaluation(period=0)]
            if val_data is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=20))

            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
                valid_sets=[val_data] if val_data is not None else None,
                valid_names=["valid"] if val_data is not None else None,
                callbacks=callbacks,
            )

            n_train = len(train_y)
            n_val = len(val_y) if val_y is not None else 0
            best_iter = model.best_iteration if hasattr(model, 'best_iteration') else '?'
            print(
                f"      [训练-{model_id}] [OK] 完成  "
                f"训练={n_train:,}  验证={n_val:,}  "
                f"{len(all_X)}期  最优={best_iter}  [V2:加权训练]"
            )
            return model
        except Exception as e:
            print(f"      [训练-{model_id}] [X] 失败: {e}")
            return None

    def _compute_ensemble_score(
        self,
        score_a: np.ndarray,
        score_b: np.ndarray,
        codes_a: pd.Index,
        codes_b: pd.Index,
        log_cap_series: pd.Series,
        small_cap_bonus: float,
        small_cap_quantile: float = 0.70,
    ) -> pd.Series:
        """Fuse two model scores with dynamic style bonus."""
        s_a = pd.Series(score_a, index=codes_a)
        s_b = pd.Series(score_b, index=codes_b)

        common = s_a.index.intersection(s_b.index)
        if len(common) == 0:
            return pd.Series(dtype=float)

        s_a = s_a.loc[common]
        s_b = s_b.loc[common]

        rank_a = s_a.rank(pct=True, method="average")
        rank_b = s_b.rank(pct=True, method="average")

        ensemble = self.weight_model_a * rank_a + self.weight_model_b * rank_b

        # V2: Dynamic small_cap_bonus based on style factor
        if log_cap_series is not None and small_cap_bonus != 0.0:
            cap_common = log_cap_series.reindex(common)
            cap_threshold = cap_common.quantile(small_cap_quantile)
            small_mask = cap_common <= cap_threshold
            ensemble[small_mask] += small_cap_bonus

        return ensemble

    def _compute_weights(
        self, ensemble_scores: pd.Series, temperature: float
    ) -> pd.Series:
        """Softmax weighting with dynamic temperature and single-stock cap."""
        scaled = ensemble_scores * temperature
        scaled = scaled - scaled.max()
        exp_scores = np.exp(scaled)
        weights = exp_scores / exp_scores.sum()

        weights = weights.clip(upper=self.max_single_weight)
        weights = weights / weights.sum()
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
        temperature: float,
    ) -> Dict[str, float]:
        """
        Full pipeline with V2 improvements:
        - Dynamic softmax temperature
        - Industry constraint: max 4 per industry, min 4 industries
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

        # Step 3: Turnover buffer
        if current_holdings and self.buffer_sigma > 0:
            score_std = candidates.std()
            if score_std > 0:
                held_codes = set(current_holdings.keys())
                boost = self.buffer_sigma * score_std
                for code in candidates.index:
                    if code in held_codes:
                        candidates[code] += boost

        candidates = candidates.sort_values(ascending=False)

        # Step 4: Industry constraint + position limit (V2: relaxed + diversity)
        if "sw_l1" in feat_df.columns and "ts_code" in feat_df.columns:
            industry_map = feat_df.set_index("ts_code")["sw_l1"].to_dict()
        else:
            industry_map = {}

        selected = []
        industry_count: Dict[str, int] = {}

        for code in candidates.index:
            if len(selected) >= effective_max:
                break
            ind = industry_map.get(code, "unknown")
            if pd.isna(ind):
                ind = "unknown"
            cnt = industry_count.get(ind, 0)
            if cnt < self.max_per_industry:
                selected.append(code)
                industry_count[ind] = cnt + 1

        # V2: If not enough industries, try to add stocks from different industries
        if len(set(industry_count.keys()) - {"unknown"}) < self.min_industries and len(selected) < effective_max:
            existing_industries = set(industry_count.keys())
            for code in candidates.index:
                if code in selected:
                    continue
                if len(selected) >= effective_max:
                    break
                ind = industry_map.get(code, "unknown")
                if pd.isna(ind):
                    ind = "unknown"
                if ind not in existing_industries and ind != "unknown":
                    cnt = industry_count.get(ind, 0)
                    if cnt < self.max_per_industry:
                        selected.append(code)
                        industry_count[ind] = cnt + 1
                        existing_industries.add(ind)

        if not selected:
            return {}

        # Step 5: Softmax weighting with dynamic temperature
        selected_scores = candidates.loc[selected]
        weights = self._compute_weights(selected_scores, temperature)

        return weights.to_dict()

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
        print(f"\n      -- V2集成策略第 {self._call_count} 期  {date_str} --")

        # V2: Track portfolio return from previous period
        if self._prev_weights and self._prev_prices:
            current_prices = accessor.get_prices(date)
            port_ret = 0.0
            for code, w in self._prev_weights.items():
                prev_price = self._prev_prices.get(code)
                cur_price = current_prices.get(code)
                if prev_price and cur_price and prev_price > 0:
                    port_ret += w * (cur_price / prev_price - 1)
            self._portfolio_returns.append(port_ret)

        # Step 0: Warmup
        if not self._warmup_done:
            self._warmup_done = True
            self._warmup_training_cache(date, accessor)

        self._append_walk_forward_data(date, accessor)
        if self._bulk_data is None or self._bulk_data.empty:
            print(f"      [数据] [X] 无可用行情缓存，本期跳过")
            return {}

        # Step 1: Check stop-loss (V2: adaptive)
        prices = accessor.get_prices(date)
        if current_holdings:
            stopped_out = self._stop_loss.update(current_holdings, prices)
            if stopped_out:
                print(f"      [止损] [!] 自适应止损触发: {len(stopped_out)} 只 "
                      f"({', '.join(list(stopped_out)[:5])}...)")
        else:
            stopped_out = set()

        # Step 2: Compute features for both models
        print(f"      [特征] 计算 Model A ({len(SM_FEATURE_NAMES)}因子) + "
              f"Model B ({len(CS_FEATURE_NAMES)}因子) ...")

        if self._bulk_data is not None:
            feat_a = sm_compute_features(
                date, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            feat_b = cs_compute_features(
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
        feat_a_ranked = rank_normalize(feat_a, SM_FEATURE_NAMES)
        feat_b_ranked = rank_normalize(feat_b, CS_FEATURE_NAMES)

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

        # Step 6: Train or reuse models (V2: sample weighting built into _train_sub_model)
        n_cached_a = len([1 for d, _, l in self._train_cache_a
                          if not l.empty and pd.Timestamp(d) < date])
        n_cached_b = len([1 for d, _, l in self._train_cache_b
                          if not l.empty and pd.Timestamp(d) < date])

        if self._should_retrain() and n_cached_a >= 8 and n_cached_b >= 8:
            print(f"      [训练] 触发重训练 "
                  f"(Model A: {n_cached_a}期, Model B: {n_cached_b}期) ...")
            new_a = self._train_sub_model(
                "A", self._train_cache_a, SM_FEATURE_NAMES,
                sm_train_model, date,
            )
            new_b = self._train_sub_model(
                "B", self._train_cache_b, CS_FEATURE_NAMES,
                cs_train_model, date,
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
            self._model_a, feat_a_ranked, SM_FEATURE_NAMES, "A"
        )
        score_b, codes_b = _predict_model(
            self._model_b, feat_b_ranked, CS_FEATURE_NAMES, "B"
        )

        codes_a_idx = pd.Index(codes_a)
        codes_b_idx = pd.Index(codes_b)

        # Step 8: Market state assessment (V2: triple trend confirmation)
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

            # V2: Triple trend confirmation
            market_regime = self._compute_market_regime(close_pivot)

            # V2: Regime modifier (simplified 3-level)
            regime_modifier_map = {
                "bull": 1.0,
                "neutral": 0.85,
                "bear": 0.60,
            }
            regime_modifier = regime_modifier_map[market_regime]

            # V2: Breadth momentum adjustment
            breadth_momentum = self._compute_breadth_momentum(close_pivot)
            if breadth_momentum < -0.10:
                regime_modifier *= 0.8
                print(f"      [择时] 广度动量恶化({breadth_momentum:.3f})，额外降仓20%")

            # V2: Style factor
            if "circ_mv" in feat_a.columns:
                circ_mv_series = feat_a.set_index("ts_code")["circ_mv"]
                # Reindex to match close_pivot columns
                circ_mv_for_style = circ_mv_series.reindex(close_pivot.columns).dropna()
            else:
                circ_mv_for_style = pd.Series(dtype=float)

            if len(circ_mv_for_style) > 50:
                style_factor = self._compute_style_factor(close_pivot, circ_mv_for_style)
            else:
                style_factor = 0.0

            # V2: Dynamic small_cap_bonus based on style factor
            if style_factor > self.style_threshold:
                small_cap_bonus = -0.01  # prefer large cap
            elif style_factor < -self.style_threshold:
                small_cap_bonus = 0.03  # prefer small cap
            else:
                small_cap_bonus = 0.0  # neutral

            # V2: Dynamic softmax temperature
            regime_temperature = {
                "bull": self.softmax_temperature_bull,
                "neutral": self.softmax_temperature_neutral,
                "bear": self.softmax_temperature_bear,
            }[market_regime]

        else:
            market_regime = "neutral"
            regime_modifier = 0.85
            style_factor = 0.0
            small_cap_bonus = 0.0
            regime_temperature = self.softmax_temperature_neutral

        # V2: Volatility targeting
        vol_scale = self._compute_vol_target_scale()
        self._prev_vol_scale = vol_scale

        # V2: Combined position sizing
        final_scale = min(vol_scale, regime_modifier)
        effective_max = max(3, int(self.max_positions * final_scale))

        print(f"      [择时] 市场状态={market_regime}  "
              f"regime_mod={regime_modifier:.2f}  "
              f"vol_scale={vol_scale:.2f}  "
              f"final_scale={final_scale:.2f}  "
              f"有效持仓上限={effective_max}  "
              f"风格因子={style_factor:.3f}  "
              f"小盘加分={small_cap_bonus:.3f}  "
              f"温度={regime_temperature:.1f}")

        # Step 9: Compute ensemble score (V2: dynamic style bonus)
        log_cap = feat_a_ranked.set_index("ts_code")["log_cap"] if "log_cap" in feat_a_ranked.columns else None

        ensemble_scores = self._compute_ensemble_score(
            score_a, score_b, codes_a_idx, codes_b_idx,
            log_cap, small_cap_bonus,
        )

        if ensemble_scores.empty:
            print(f"      [融合] [X] 无交集股票")
            return {}

        # Individual ranks for consensus filtering
        s_a = pd.Series(score_a, index=codes_a_idx)
        s_b = pd.Series(score_b, index=codes_b_idx)
        common = ensemble_scores.index
        rank_a = s_a.reindex(common).rank(pct=True, method="average")
        rank_b = s_b.reindex(common).rank(pct=True, method="average")

        print(f"      [融合] [OK] 融合 {len(ensemble_scores)} 只，"
              f"score=[{ensemble_scores.min():.4f}, {ensemble_scores.max():.4f}]")

        # Step 10: Select and weight (V2: dynamic temperature)
        weights = self._select_and_weight(
            ensemble_scores, rank_a, rank_b,
            feat_a_ranked, current_holdings,
            effective_max, stopped_out,
            regime_temperature,
        )

        # Step 10b: Cache factor exposures for Barra attribution
        self._last_factor_exposures = None
        if weights:
            # Merge both models' ranked features for selected stocks
            all_feature_names = list(SM_FEATURE_NAMES) + [
                f for f in CS_FEATURE_NAMES if f not in SM_FEATURE_NAMES
            ]
            selected_codes = set(weights.keys())
            exp_rows = []
            for code in selected_codes:
                row_dict = {"ts_code": code, "weight": weights[code]}
                # Model A features
                row_a = feat_a_ranked[feat_a_ranked["ts_code"] == code]
                if not row_a.empty:
                    for fn in SM_FEATURE_NAMES:
                        if fn in row_a.columns:
                            row_dict[fn] = row_a[fn].values[0]
                # Model B features (only those not already from A)
                row_b = feat_b_ranked[feat_b_ranked["ts_code"] == code]
                if not row_b.empty:
                    for fn in CS_FEATURE_NAMES:
                        if fn not in row_dict and fn in row_b.columns:
                            row_dict[fn] = row_b[fn].values[0]
                exp_rows.append(row_dict)
            if exp_rows:
                self._last_factor_exposures = pd.DataFrame(exp_rows)

        # Step 11: Update stop-loss tracker (V2: record stock volatility)
        new_codes = set(weights.keys())
        old_codes = set(current_holdings.keys()) if current_holdings else set()
        for code in new_codes - old_codes:
            if code in prices:
                stock_vol = self._compute_stock_vol(code, date)
                self._stop_loss.record_entry(code, prices[code], stock_vol)
        for code in old_codes - new_codes:
            self._stop_loss.record_exit(code)

        # V2: Save weights and prices for next period's return calculation
        self._prev_weights = dict(weights)
        self._prev_prices = dict(prices)

        # Print summary
        n_held = len(current_holdings) if current_holdings else 0
        overlap = len(new_codes & old_codes)
        n_industries = len(set(
            feat_a_ranked.set_index("ts_code")["sw_l1"].to_dict().get(c, "unknown")
            for c in weights.keys()
        )) if weights else 0

        if len(weights) == 0:
            print(f"      [选股] [!] 本期无满足条件的股票，空仓")
        else:
            top_w = max(weights.values()) * 100
            avg_w = np.mean(list(weights.values())) * 100
            print(
                f"      [选股] 入选 {len(weights)} 只 / {n_industries} 个行业  "
                f"(上期 {n_held}，留存 {overlap}，换手 {n_held + len(weights) - 2*overlap})  "
                f"权重: max={top_w:.1f}% avg={avg_w:.1f}%"
            )

        return weights

    def get_factor_exposures(
        self,
        date: pd.Timestamp,
        selected_codes: Dict[str, float],
    ) -> Optional[pd.DataFrame]:
        """
        Return per-stock factor exposures for the selected portfolio.

        Uses the cached factor exposure data from the most recent
        generate_target_weights() call.  Covers all 44 factors from
        both Model A (Smart Money) and Model B (Cross-Sectional).
        """
        if self._last_factor_exposures is not None and not self._last_factor_exposures.empty:
            return self._last_factor_exposures
        return None


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

    strategy = LGBMEnsembleAdaptiveV2(
        # Model parameters (V2)
        train_window_years=4,
        retrain_interval=6,
        sample_weight_halflife=12,
        # Ensemble parameters (unchanged)
        weight_model_a=0.6,
        weight_model_b=0.4,
        consensus_top_pct=0.05,
        consensus_single_top_pct=0.10,
        # Position parameters (V2)
        max_positions=25,
        target_vol=0.18,
        vol_lookback=40,
        vol_scale_min=0.15,
        vol_scale_max=1.20,
        vol_smoothing=0.7,
        # Weight parameters (V2)
        softmax_temperature_bull=6.0,
        softmax_temperature_neutral=4.0,
        softmax_temperature_bear=2.0,
        max_single_weight=0.08,
        # Risk control parameters (V2)
        max_per_industry=4,
        min_industries=4,
        stop_loss_vol_multiplier=2.5,
        stop_loss_min=-0.08,
        stop_loss_max=-0.20,
        stop_loss_cooldown=2,
        buffer_sigma=0.5,
        # Stock pool parameters (V2)
        mv_pct_upper=0.85,
        style_threshold=0.05,
        feature_lookback=260,
        backtest_end_date=cfg.end_date,
    )
    result = run_backtest(strategy, cfg)
