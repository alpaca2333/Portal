"""
LightGBM Ensemble Adaptive Strategy
====================================

Motivation
----------
Combine two proven LightGBM models — Smart Money (accumulation tracking)
and Cross-Sectional (fundamental ranking) — into a single ensemble that
leverages both alpha sources.  Add a market-state-adaptive position sizing
module to reduce drawdowns in bear markets and maximize exposure in bull
markets.

Core Architecture
-----------------
1. **Multi-model ensemble stock selection**: Two independent LightGBM
   models (Smart Money + Cross-Sectional), weighted fusion with consensus
   filtering.
2. **Market state adaptive positioning**: Three-dimensional market state
   (trend, breadth, volatility) → five-level position coefficient that
   controls portfolio size.
3. **Portfolio construction & risk control**: Softmax weighting, industry
   constraint, individual stop-loss with cooldown, turnover buffer.

Usage
-----
cd D:\\Projects\\Portal
python -m data.quant.strategies.lgbm_ensemble_adaptive
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

class LGBMEnsembleAdaptive(StrategyBase):
    """
    Adaptive ensemble stock selection + macro timing strategy.

    Combines two LightGBM models (Smart Money + Cross-Sectional) with
    market-state-adaptive position sizing, softmax weighting, stop-loss,
    and industry constraints.
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
        # Stock pool parameters
        mv_pct_upper: float = 0.85,
        small_cap_bonus: float = 0.02,
        small_cap_quantile: float = 0.70,
        feature_lookback: int = 260,
        backtest_end_date: Optional[str] = None,
    ):
        super().__init__("lgbm_ensemble_adaptive")

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
        self.buffer_sigma = buffer_sigma

        # Stock pool parameters
        self.mv_pct_upper = mv_pct_upper
        self.small_cap_bonus = small_cap_bonus
        self.small_cap_quantile = small_cap_quantile
        self.feature_lookback = feature_lookback
        self._backtest_end_date = (
            pd.Timestamp(backtest_end_date) if backtest_end_date else None
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

    def describe(self) -> str:
        return (
            f"### 策略思路\n\n"
            f"自适应集成选股 + 宏观择时组合策略。融合两个 LightGBM 模型"
            f"（Smart Money 大资金追踪 + Cross-Sectional 截面排名），"
            f"并通过三维度市场状态判断动态调整仓位水平。\n\n"
            f"### 核心设计\n\n"
            f"1. **双模型融合**：Smart Money (权重{self.weight_model_a}) + "
            f"Cross-Sectional (权重{self.weight_model_b})，"
            f"共识过滤（综合前{self.consensus_top_pct*100:.0f}% 且至少一个模型前"
            f"{self.consensus_single_top_pct*100:.0f}%）\n"
            f"2. **市场状态择时**：趋势(MA20 vs MA60) + 广度(站上MA20比例) + "
            f"波动(20d/60d波动率比)，五档仓位(1.0/0.8/0.5/0.3/0.1)\n"
            f"3. **信号强度加权**：softmax(temperature={self.softmax_temperature})，"
            f"单股上限{self.max_single_weight*100:.0f}%\n"
            f"4. **个股止损**：跌超{abs(self._stop_loss.threshold)*100:.0f}%强制止损，"
            f"冷却{self._stop_loss.cooldown}期\n"
            f"5. **行业约束**：每行业最多{self.max_per_industry}只\n"
            f"6. **换手控制**：持仓股 buffer_sigma={self.buffer_sigma}\n"
            f"7. **首轮预热 + 逐期推进**：首轮交易前允许 warmup，"
            f"进入回测后仅按当前日期 walk-forward 追加数据\n"
            f"8. **双周频调仓**：平衡交易成本与 alpha 捕捉\n\n"
            f"### 仓位矩阵\n\n"
            f"| 状态 | 趋势 | 广度 | 波动 | 仓位 | 最大持仓 |\n"
            f"|------|------|------|------|------|--------|\n"
            f"| 强牛 | 多头 | >60% | 正常 | 100% | {self.max_positions} |\n"
            f"| 普牛 | 多头 | 40-60% | 正常 | 80% | {int(self.max_positions*0.8)} |\n"
            f"| 震荡 | 空头 | 30-60% | 正常 | 50% | {int(self.max_positions*0.5)} |\n"
            f"| 弱熊 | 空头 | <30% | 正常 | 30% | {int(self.max_positions*0.3)} |\n"
            f"| 恐慌 | 空头 | <30% | 高 | 10% | {max(2, int(self.max_positions*0.1))} |\n"
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

    def _compute_ensemble_score(
        self,
        score_a: np.ndarray,
        score_b: np.ndarray,
        codes_a: pd.Index,
        codes_b: pd.Index,
        log_cap_series: pd.Series,
    ) -> pd.Series:
        """
        Fuse two model scores into an ensemble score.

        Steps:
        1. Rank-normalize each model's scores
        2. Weighted fusion: w_A * rank_A + w_B * rank_B
        3. Small-cap bonus: +0.02 for stocks below small_cap_quantile in log_cap
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
    ) -> Dict[str, float]:
        """
        Full pipeline: consensus filter → stop-loss/cooldown filter →
        turnover buffer → industry constraint → select → softmax weight.
        """
        if ensemble_scores.empty:
            return {}

        # Step 1: Consensus filter
        # Top consensus_top_pct of ensemble score
        top_threshold = ensemble_scores.quantile(1.0 - self.consensus_top_pct)
        top_mask = ensemble_scores >= top_threshold

        # At least one model in top single_top_pct
        single_threshold_a = 1.0 - self.consensus_single_top_pct
        single_threshold_b = 1.0 - self.consensus_single_top_pct
        model_a_top = rank_a >= single_threshold_a
        model_b_top = rank_b >= single_threshold_b
        consensus_mask = top_mask & (model_a_top | model_b_top)

        candidates = ensemble_scores[consensus_mask].copy()
        if candidates.empty:
            # Fallback: just use top ensemble scores
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

        # Step 3: Turnover buffer — boost scores for currently held stocks
        if current_holdings and self.buffer_sigma > 0:
            score_std = candidates.std()
            if score_std > 0:
                held_codes = set(current_holdings.keys())
                boost = self.buffer_sigma * score_std
                for code in candidates.index:
                    if code in held_codes:
                        candidates[code] += boost

        # Sort by score descending
        candidates = candidates.sort_values(ascending=False)

        # Step 4: Industry constraint + position limit
        # Build sw_l1 lookup from feat_df
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

        if not selected:
            return {}

        # Step 5: Softmax weighting
        selected_scores = candidates.loc[selected]
        weights = self._compute_weights(selected_scores)

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
        print(f"\n      -- 集成策略第 {self._call_count} 期  {date_str} --")

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
                # Fallback: mean of rank-normalized features
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

        # Step 8: Compute ensemble score
        # Get log_cap from feat_a (both have it)
        log_cap = feat_a_ranked.set_index("ts_code")["log_cap"] if "log_cap" in feat_a_ranked.columns else None

        ensemble_scores = self._compute_ensemble_score(
            score_a, score_b, codes_a_idx, codes_b_idx, log_cap,
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
        # Build close_pivot for market state computation
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

        # Step 10: Select and weight
        # Use feat_a for industry lookup (both have sw_l1)
        weights = self._select_and_weight(
            ensemble_scores, rank_a, rank_b,
            feat_a_ranked, current_holdings,
            effective_max, stopped_out,
        )

        # Step 11: Update stop-loss tracker
        # Record new entries and exits
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

    strategy = LGBMEnsembleAdaptive(
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
        buffer_sigma=0.5,
        # Stock pool parameters
        mv_pct_upper=0.85,
        small_cap_bonus=0.02,
        small_cap_quantile=0.70,
        feature_lookback=260,
        backtest_end_date=cfg.end_date,
    )
    result = run_backtest(strategy, cfg)
