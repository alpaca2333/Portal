"""
LightGBM Ensemble Adaptive Strategy V4
=======================================

Improvements over V1 (base):
1. **Industry Momentum Scoring (P0)**: Compute 6-month circ_mv-weighted
   return for each SW-L1 industry. Stocks in top-ranked industries get
   an ensemble score bonus (+0.02); stocks in bottom-ranked industries
   get a penalty (-0.01). Based on research finding that 6-12M industry
   momentum is statistically significant (IC≈0.06, t>2).
2. **Dynamic Industry Constraint (P1)**: Adjust max_per_industry based
   on industry momentum rank. Strong industries get a higher limit (5),
   weak industries get a lower limit (2), others keep default (3).
3. **Factor Attribution**: Implements get_factor_exposures() for the
   backtest engine to generate factor contribution analysis.

Inherited from V1:
- Dual LightGBM ensemble (Smart Money + Cross-Sectional)
- Three-dimensional market state adaptive positioning
- Softmax weighting, stop-loss with cooldown, turnover buffer

Usage
-----
cd D:\\Projects\\Portal
python -m data.quant.strategies.lgbm_ensemble_adaptive_v4
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

class LGBMEnsembleAdaptiveV6(StrategyBase):
    """
    V4: Industry momentum scoring + dynamic industry constraint + factor attribution.

    Built on V1 base (dual LightGBM ensemble + market state timing).
    """

    def __init__(
        self,
        # Model parameters
        train_window_years: int = 3,
        train_window_years_b: int = 2,    # V6: CS model shorter window (§1.4)
        retrain_interval: int = 4,
        # Ensemble parameters
        weight_model_a: float = 0.4,
        weight_model_b: float = 0.6,
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
        super().__init__("lgbm_ensemble_adaptive_v6")

        # Model parameters
        self.train_window_years = train_window_years       # SM model: 3 years
        self.train_window_years_b = train_window_years_b   # CS model: 2 years (§1.4)
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
        self.buffer_sigma = buffer_sigma

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

        # Factor attribution cache
        self._last_factor_exposures: Optional[pd.DataFrame] = None

    def describe(self) -> str:
        return (
            f"### 策略思路\n\n"
            f"自适应集成选股 V6：V4基础 + 模型层改进（LambdaRank/超参差异化/行业中性标签/训练窗口差异化）。"
            f"融合两个 LightGBM 模型（Smart Money + Cross-Sectional），"
            f"并通过三维度市场状态判断动态调整仓位水平。\n\n"
            f"### V4 核心改进\n\n"
            f"1. **行业动量加分 (P0)**：计算过去{self.industry_momentum_lookback_months}个月"
            f"各申万一级行业的流通市值加权收益率，"
            f"强势行业(前{self.industry_strong_top_n}名)个股 ensemble_score "
            f"+{self.industry_momentum_bonus}，"
            f"弱势行业(后{self.industry_weak_top_n}名)个股 "
            f"{self.industry_momentum_penalty}\n"
            f"2. **动态行业约束 (P1)**：强势行业 max_per_industry="
            f"{self.industry_strong_max}，"
            f"弱势行业 max_per_industry={self.industry_weak_max}，"
            f"其余行业 max_per_industry={self.max_per_industry}\n"
            f"3. **因子归因**：实现 get_factor_exposures() 接口，"
            f"支持因子收益贡献分析\n\n"
            f"### 继承自原版\n\n"
            f"- 双模型融合：Smart Money ({self.weight_model_a}) + "
            f"Cross-Sectional ({self.weight_model_b})\n"
            f"- 共识过滤（综合前{self.consensus_top_pct*100:.0f}% 且至少一个模型前"
            f"{self.consensus_single_top_pct*100:.0f}%）\n"
            f"- 市场状态择时：五档仓位(1.0/0.8/0.5/0.3/0.1)\n"
            f"- softmax 加权 (temperature={self.softmax_temperature})，"
            f"单股上限{self.max_single_weight*100:.0f}%\n"
            f"- 个股止损 + 冷却 + 换手缓冲\n"
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
            feat_a = sm_compute_features(
                d, self._bulk_data, lookback=self.feature_lookback,
                st_codes=self._st_codes,
            )
            # Model B features (Cross-Sectional)
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
        """
        Train a sub-model (A or B) using LambdaRank with industry-neutral labels.

        V6 improvements:
        - §1.1: LambdaRank objective with NDCG metric (ranking-native)
        - §1.2: Differentiated hyperparameters per model
        - §1.3: Industry-neutral labels (subtract industry median before ranking)
        - §1.4: Differentiated training windows (A=3yr, B=2yr)
        """
        valid_cache = [
            (d, f, l) for d, f, l in train_cache
            if not l.empty and pd.Timestamp(d) < date
        ]
        if len(valid_cache) < 8:
            print(f"      [训练-{model_id}] [X] 有效训练期不足 ({len(valid_cache)})")
            return None

        # §1.4: Use different training windows for A and B
        if model_id == "B":
            train_start = date - pd.DateOffset(years=self.train_window_years_b)
        else:
            train_start = date - pd.DateOffset(years=self.train_window_years)

        all_X = []
        all_y = []
        group_sizes = []  # query group sizes for LambdaRank

        for d_str, f_df, labels in valid_cache:
            dt = pd.Timestamp(d_str)
            if dt >= train_start:
                merged = f_df.set_index("ts_code").join(
                    labels.rename("label"), how="inner"
                )
                if len(merged) >= 30:
                    # §1.3: Industry-neutral labels
                    raw_ret = merged["label"]
                    if "sw_l1" in f_df.columns:
                        ind_map = f_df.set_index("ts_code")["sw_l1"]
                        ind_for_merged = ind_map.reindex(merged.index)
                        ind_median = raw_ret.groupby(ind_for_merged).transform("median")
                        neutral_ret = raw_ret - ind_median
                    else:
                        neutral_ret = raw_ret
                    # Rank-normalize to [0, 1]
                    rank_pct = neutral_ret.rank(pct=True, method="average")
                    # §1.1: Quantize to 5 relevance grades for LambdaRank
                    # Drop NaN before quantization
                    valid_mask = rank_pct.notna()
                    merged_valid = merged.loc[valid_mask]
                    rank_pct_valid = rank_pct.loc[valid_mask]
                    if len(merged_valid) < 30:
                        continue
                    grades = pd.cut(
                        rank_pct_valid,
                        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
                        labels=[0, 1, 2, 3, 4],
                    ).astype(int)
                    all_X.append(merged_valid[feature_names])
                    all_y.append(grades)
                    group_sizes.append(len(merged_valid))

        if len(all_X) < 8:
            print(f"      [训练-{model_id}] [X] 合并后训练期不足 ({len(all_X)})")
            return None

        n = len(all_X)
        split = max(1, int(n * 0.8))

        train_X = pd.concat(all_X[:split])
        train_y = pd.concat(all_y[:split])
        train_groups = group_sizes[:split]
        val_X = pd.concat(all_X[split:]) if split < n else None
        val_y = pd.concat(all_y[split:]) if split < n else None
        val_groups = group_sizes[split:] if split < n else None

        # §1.2: Differentiated hyperparameters
        if model_id == "A":
            # SM model: conservative
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10, 20],
                "label_gain": [0, 1, 3, 7, 15],
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
            num_boost_round = 300
        else:
            # CS model: aggressive (more capacity)
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10, 20],
                "label_gain": [0, 1, 3, 7, 15],
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.03,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "max_depth": 6,
                "min_child_samples": 100,
                "lambda_l1": 0.1,
                "lambda_l2": 1.0,
                "verbose": -1,
                "seed": 42,
            }
            num_boost_round = 500

        # Fill NaN with column median
        for col in feature_names:
            med = train_X[col].median()
            train_X[col] = train_X[col].fillna(med)
            if val_X is not None:
                val_X[col] = val_X[col].fillna(med)

        dtrain = lgb.Dataset(train_X, label=train_y, group=train_groups)
        valid_sets = [dtrain]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=0)]

        if val_X is not None and val_y is not None and len(val_y) > 0:
            dval = lgb.Dataset(val_X, label=val_y, group=val_groups, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False, first_metric_only=True))

        try:
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            n_train = len(train_y)
            n_val = len(val_y) if val_y is not None else 0
            best_iter = model.best_iteration if hasattr(model, "best_iteration") else "?"
            print(
                f"      [训练-{model_id}] [OK] 完成  "
                f"训练={n_train:,}  验证={n_val:,}  "
                f"{len(all_X)}期  最优={best_iter}  "
                f"[LambdaRank, leaves={params['num_leaves']}, "
                f"lr={params['learning_rate']}, rounds={num_boost_round}]"
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

        # Position matrix
        if is_uptrend:
            if above_ratio > 0.60:
                coeff = 1.00  # strong bull
            else:
                coeff = 0.80  # mild bull
        else:
            if above_ratio >= 0.30:
                coeff = 0.50  # range-bound
            elif not is_high_vol:
                coeff = 0.30  # weak bear
            else:
                coeff = 0.10  # panic

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

        # V4: Industry momentum bonus/penalty
        if industry_momentum is not None and industry_map is not None:
            n_industries = len(industry_momentum)
            strong_set = set(
                industry_momentum.head(self.industry_strong_top_n)["industry"]
            )
            weak_set = set(
                industry_momentum.tail(self.industry_weak_top_n)["industry"]
            )

            n_bonus = 0
            n_penalty = 0
            for code in common:
                ind = industry_map.get(code)
                if ind is None or pd.isna(ind):
                    continue
                if ind in strong_set:
                    ensemble[code] += self.industry_momentum_bonus
                    n_bonus += 1
                elif ind in weak_set:
                    ensemble[code] += self.industry_momentum_penalty
                    n_penalty += 1

            if n_bonus > 0 or n_penalty > 0:
                print(
                    f"      [V4行业动量] 加分 {n_bonus} 只 "
                    f"(+{self.industry_momentum_bonus})  "
                    f"减分 {n_penalty} 只 "
                    f"({self.industry_momentum_penalty})"
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
        print(f"\n      -- V6集成策略第 {self._call_count} 期  {date_str} --")

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

        # Step 6: Train or reuse models
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

        # Step 7.5 (V4): Compute industry momentum
        industry_momentum = self._compute_industry_momentum(date)
        if industry_momentum is not None:
            top3 = industry_momentum.head(3)["industry"].tolist()
            bot3 = industry_momentum.tail(3)["industry"].tolist()
            print(
                f"      [V4行业动量] {len(industry_momentum)} 个行业  "
                f"强势: {', '.join(top3)}  弱势: {', '.join(bot3)}"
            )
        else:
            print(f"      [V4行业动量] 数据不足，本期跳过行业动量")

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

        effective_max = max(2, int(self.max_positions * position_coeff))
        state_labels = {
            1.0: "强牛", 0.8: "普牛", 0.5: "震荡", 0.3: "弱熊", 0.1: "恐慌"
        }
        state_name = state_labels.get(position_coeff, f"coeff={position_coeff}")
        print(f"      [择时] 市场状态={state_name}  "
              f"仓位系数={position_coeff}  有效持仓上限={effective_max}")

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

        # Step 11: Update stop-loss tracker
        new_codes = set(weights.keys())
        old_codes = set(current_holdings.keys()) if current_holdings else set()
        for code in new_codes - old_codes:
            if code in prices:
                self._stop_loss.record_entry(code, prices[code])
        for code in old_codes - new_codes:
            self._stop_loss.record_exit(code)

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

    strategy = LGBMEnsembleAdaptiveV6(
        # Model parameters
        train_window_years=3,
        train_window_years_b=2,    # V6: CS model shorter window
        retrain_interval=4,
        # Ensemble parameters
        weight_model_a=0.5,
        weight_model_b=0.5,
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
        buffer_sigma=0.5,
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
