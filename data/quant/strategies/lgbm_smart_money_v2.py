"""
LightGBM Smart Money V2 (Round 1 implementation)
=================================================

Round 1 keeps the original smart-money feature engineering and LightGBM
training framework, but upgrades the portfolio construction layer with:
- regime-aware gross exposure / target holdings linkage
- stronger hold buffer (buy-vs-hold dual thresholds)
- tiered weighting
- lower ineffective turnover via turnover blending
- full-memory prefetch path reused from the existing baseline framework

Usage
-----
cd <project_root>
python -m data.quant.strategies.lgbm_smart_money_v2
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm")

from engine import BacktestConfig, StrategyBase, run_backtest
from engine.data_loader import DataAccessor
from strategies.utils import prefetch_bulk_data
from strategies.lgbm_smart_money import (
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    compute_features_from_memory,
    compute_forward_return_from_memory,
    rank_normalize,
)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

_bulk_date_index_cache: Dict[int, Tuple[np.ndarray, Dict]] = {}


def train_lgbm_model_fast(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: Optional[pd.DataFrame] = None,
    val_labels: Optional[pd.Series] = None,
) -> lgb.Booster:
    """Train LightGBM with conservative runtime settings for long weekly backtests."""
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
        "num_threads": 4,
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
        num_boost_round=160,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


class LGBMSmartMoneyV2(StrategyBase):
    def __init__(
        self,
        train_window_years: int = 3,
        score_quantile_buy: float = 0.93,
        score_quantile_hold: float = 0.88,
        min_signal_count_buy: int = 5,
        min_signal_count_hold: int = 4,
        max_per_industry: int = 2,
        max_positions: int = 16,
        buffer_sigma: float = 0.9,
        mv_pct_upper: float = 0.85,
        feature_lookback: int = 260,
        retrain_interval: int = 8,
        turnover_limit: float = 0.18,
        target_step_limit: int = 3,
        benchmark_code: str = "000300.SH",
        backtest_end_date: Optional[pd.Timestamp] = None,
    ):
        super().__init__("lgbm_smart_money_v2")
        self.train_window_years = train_window_years
        self.score_quantile_buy = score_quantile_buy
        self.score_quantile_hold = score_quantile_hold
        self.min_signal_count_buy = min_signal_count_buy
        self.min_signal_count_hold = min_signal_count_hold
        self.max_per_industry = max_per_industry
        self.max_positions = max_positions
        self.buffer_sigma = buffer_sigma
        self.mv_pct_upper = mv_pct_upper
        self.feature_lookback = feature_lookback
        self.retrain_interval = retrain_interval
        self.turnover_limit = turnover_limit
        self.target_step_limit = target_step_limit
        self.benchmark_code = benchmark_code
        self._backtest_end_date = pd.Timestamp(backtest_end_date) if backtest_end_date is not None else None

        self._model = None
        self._train_data_cache: List[Tuple[str, pd.DataFrame, pd.Series]] = []
        self._last_train_date: Optional[pd.Timestamp] = None
        self._call_count = 0
        self._st_codes: Optional[set] = None
        self._benchmark_cache: Optional[pd.DataFrame] = None
        self._prev_target_weights: Dict[str, float] = {}
        self._last_target_n: Optional[int] = None
        self._last_regime: Optional[str] = None
        self._bulk_data: Optional[pd.DataFrame] = None
        self._warmup_done = False

    def describe(self) -> str:
        return (
            "### 策略思路\n\n"
            "Round 1 沿用原版 smart-money 特征与 LightGBM 截面训练框架，"
            "重点升级组合构建层：市场状态过滤、双阈值持仓、分层权重、"
            "以及持仓缓冲来降低换手、改善回撤。当前参数优先级为：夏普 > 年化收益 > 累计收益。\n\n"
            "### Round 1 增强点\n\n"
            f"1. 新买入阈值更严格：ML 分数 ≥ {self.score_quantile_buy:.0%} 分位，且信号数 ≥ {self.min_signal_count_buy}\n"
            f"2. 老持仓阈值更宽松：ML 分数 ≥ {self.score_quantile_hold:.0%} 分位，且信号数 ≥ {self.min_signal_count_hold}\n"
            f"3. 持仓缓冲：已有持仓的 ML 分数加 {self.buffer_sigma:.1f}σ，减少无效替换\n"
            "4. Regime Filter：依据指数趋势 + 市场广度 + 市场波动，动态调整持股数与总仓位\n"
            "5. 分层权重：Top / Mid / Bottom 候选使用 1.6 / 1.0 / 0.55 权重系数，而非纯等权\n"
            f"6. 换手约束：单期单边换手超过 {self.turnover_limit:.0%} 时，对新旧目标权重做线性混合\n\n"
            "### 实现说明\n\n"
            "- 使用现有 engine / strategy / utility，不引入新框架\n"
            "- 复用基线的 bulk prefetch / 全内存特征与标签缓存，以保证周频全回测速度\n"
            "- risk_off 通过返回空仓直接持有现金，不引入额外伪资产键\n"
        )

    def _should_retrain(self) -> bool:
        if self._model is None:
            return True
        return self._call_count % self.retrain_interval == 0

    def _warmup_training_cache(self, current_date: pd.Timestamp, accessor: DataAccessor):
        warmup_start = current_date - pd.DateOffset(years=self.train_window_years, months=2)
        warmup_end = current_date - pd.DateOffset(days=1)

        print(
            f"      [预热] 加载历史训练数据 {warmup_start.strftime('%Y-%m-%d')} ~ {warmup_end.strftime('%Y-%m-%d')} ..."
        )

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

        data_start = warmup_start - pd.DateOffset(days=int(self.feature_lookback * 1.8))
        bulk = prefetch_bulk_data(accessor, data_start, warmup_end, FEATURE_COLUMNS)
        if bulk.empty:
            print("      [预热] ✗ 未找到历史数据")
            return

        if self._backtest_end_date is not None:
            backtest_end = self._backtest_end_date + pd.DateOffset(months=1)
        else:
            backtest_end = current_date + pd.DateOffset(years=4)
        bulk_backtest = prefetch_bulk_data(accessor, warmup_end, backtest_end, FEATURE_COLUMNS)
        if not bulk_backtest.empty:
            self._bulk_data = pd.concat([bulk, bulk_backtest], ignore_index=True)
            self._bulk_data.drop_duplicates(subset=["ts_code", "trade_date"], keep="last", inplace=True)
            self._bulk_data.sort_values(["trade_date", "ts_code"], inplace=True)
            self._bulk_data.reset_index(drop=True, inplace=True)
        else:
            self._bulk_data = bulk

        _bulk_date_index_cache.clear()

        hist_trade_dates_raw = np.sort(bulk["trade_date"].unique())
        mask = (hist_trade_dates_raw >= pd.Timestamp(warmup_start)) & (hist_trade_dates_raw <= pd.Timestamp(warmup_end))
        hist_trade_dates = pd.DatetimeIndex(hist_trade_dates_raw[mask])
        if len(hist_trade_dates) == 0:
            print("      [预热] ✗ 历史交易日为空")
            return

        s = pd.Series(hist_trade_dates, index=hist_trade_dates)
        week_ids = s.index.isocalendar().year * 100 + s.index.isocalendar().week
        hist_rebal_dates = pd.DatetimeIndex(s.groupby(week_ids.values).last().values)

        print(f"      [预热] 历史调仓日 {len(hist_rebal_dates)} 个，开始计算特征和标签 (全内存模式) ...")

        n_success = 0
        n_skip = 0
        for i in range(len(hist_rebal_dates) - 1):
            d = hist_rebal_dates[i]
            d_next = hist_rebal_dates[i + 1]
            feat_df = compute_features_from_memory(d, self._bulk_data, lookback=self.feature_lookback, st_codes=self._st_codes)
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
            self._train_data_cache.append((d.strftime("%Y-%m-%d"), feat_ranked.copy(), fwd_ret))
            n_success += 1

        if len(hist_rebal_dates) > 0:
            last_d = hist_rebal_dates[-1]
            feat_df = compute_features_from_memory(last_d, self._bulk_data, lookback=self.feature_lookback, st_codes=self._st_codes)
            if feat_df is not None and not feat_df.empty:
                feat_df = self._apply_market_cap_filter(feat_df)
                if len(feat_df) >= 50:
                    feat_ranked = rank_normalize(feat_df, FEATURE_NAMES)
                    fwd_ret = compute_forward_return_from_memory(last_d, current_date, self._bulk_data)
                    if fwd_ret is not None:
                        self._train_data_cache.append((last_d.strftime("%Y-%m-%d"), feat_ranked.copy(), fwd_ret))
                        n_success += 1

        cutoff = current_date - pd.DateOffset(years=self.train_window_years, months=6)
        self._train_data_cache = [(d, f, l) for d, f, l in self._train_data_cache if pd.Timestamp(d) >= cutoff]
        print(f"      [预热] ✓ 成功缓存 {n_success} 期 (跳过 {n_skip} 期)，训练窗口已就绪")

    def _load_benchmark_df(self) -> Optional[pd.DataFrame]:
        if self._benchmark_cache is not None:
            return self._benchmark_cache
        quant_root = Path(__file__).resolve().parents[1]
        primary = quant_root / "baseline" / f"{self.benchmark_code}.csv"
        fallback = quant_root / "baseline" / "000001.SH.csv"
        path = primary if primary.exists() else fallback if fallback.exists() else None
        if path is None:
            return None
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
        self._benchmark_cache = df
        return df

    def _apply_market_cap_filter(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        if "circ_mv" not in feat_df.columns:
            return feat_df
        mv = feat_df["circ_mv"].dropna()
        if len(mv) == 0:
            return feat_df
        lower_bound = mv.quantile(1.0 - self.mv_pct_upper)
        return feat_df[feat_df["circ_mv"] >= lower_bound].copy()

    def _count_smart_money_signals(self, feat_ranked: pd.DataFrame) -> pd.Series:
        signals = pd.DataFrame(index=feat_ranked.index)
        high_is_good = ["vol_surge_ratio", "obv_slope", "lower_shadow_ratio", "money_flow_strength", "illiq_change"]
        for col in high_is_good:
            if col in feat_ranked.columns:
                signals[col] = (feat_ranked[col] >= 0.70).astype(int)

        low_is_good = ["shrink_pullback", "vol_compression", "ma_convergence", "turnover_concentration"]
        for col in low_is_good:
            if col in feat_ranked.columns:
                signals[col] = (feat_ranked[col] <= 0.30).astype(int)

        if "bottom_deviation" in feat_ranked.columns:
            signals["bottom_deviation"] = (feat_ranked["bottom_deviation"] <= 0.40).astype(int)
        return signals.sum(axis=1)

    def _compute_market_regime(self, date: pd.Timestamp, window_df: pd.DataFrame) -> Dict[str, float]:
        breadth = np.nan
        breadth_pos = np.nan
        market_vol = np.nan
        vol_ratio = np.nan
        regime_score = 0
        trend_label = "mixed"

        w = window_df[
            window_df["ts_code"].astype(str).str.match(r"^(6\d{5}\.SH|00[013]\d{3}\.SZ)$")
            & (window_df["is_suspended"] != 1)
            & (window_df["close"].notna())
            & (window_df["close"] > 0)
        ][["trade_date", "ts_code", "close"]].copy()

        if not w.empty:
            close_pivot = w.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
            if len(close_pivot) >= 21 and close_pivot.shape[1] >= 100:
                ma20 = close_pivot.iloc[-20:].mean()
                current_close = close_pivot.iloc[-1]
                breadth = float((current_close > ma20).mean())
                ret20 = current_close / close_pivot.iloc[-21] - 1
                breadth_pos = float((ret20 > 0).mean())
                mkt_proxy = close_pivot.median(axis=1)
                market_rets = mkt_proxy.pct_change().dropna()
                if len(market_rets) >= 60:
                    market_vol = float(market_rets.iloc[-20:].std())
                    hist_vol = float(market_rets.iloc[-60:].std())
                    if hist_vol > 0:
                        vol_ratio = market_vol / hist_vol
                elif len(market_rets) >= 20:
                    market_vol = float(market_rets.iloc[-20:].std())

        bench = self._load_benchmark_df()
        trend_score = 0
        if bench is not None:
            hist = bench[bench["date"] <= pd.Timestamp(date)]
            if len(hist) >= 60:
                close = float(hist.iloc[-1]["close"])
                ma20 = float(hist.iloc[-20:]["close"].mean())
                ma60 = float(hist.iloc[-60:]["close"].mean())
                if close > ma20 > ma60:
                    trend_score = 1
                    trend_label = "up"
                elif close < ma20 < ma60:
                    trend_score = -1
                    trend_label = "down"

        breadth_score = 0
        if pd.notna(breadth) and pd.notna(breadth_pos):
            if breadth >= 0.58 and breadth_pos >= 0.55:
                breadth_score = 1
            elif breadth <= 0.42 and breadth_pos <= 0.45:
                breadth_score = -1

        vol_score = 0
        if pd.notna(vol_ratio):
            if vol_ratio >= 1.15:
                vol_score = -1
            elif vol_ratio <= 0.95:
                vol_score = 1

        regime_score = trend_score + breadth_score + vol_score

        if regime_score >= 2:
            regime = "risk_on"
            target_gross = 0.75
            target_n = 12
            turnover_limit = min(self.turnover_limit, 0.18)
            buy_q = self.score_quantile_buy
            hold_q = self.score_quantile_hold
            buy_sig = self.min_signal_count_buy
            hold_sig = self.min_signal_count_hold
        elif regime_score <= -1:
            regime = "risk_off"
            target_gross = 0.00
            target_n = 0
            turnover_limit = min(self.turnover_limit, 0.08)
            buy_q = min(0.97, self.score_quantile_buy + 0.02)
            hold_q = min(0.94, self.score_quantile_hold + 0.03)
            buy_sig = self.min_signal_count_buy + 1
            hold_sig = self.min_signal_count_hold + 1
        else:
            regime = "neutral"
            target_gross = 0.35
            target_n = 6
            turnover_limit = min(self.turnover_limit, 0.12)
            buy_q = min(0.95, self.score_quantile_buy + 0.01)
            hold_q = min(0.90, self.score_quantile_hold + 0.01)
            buy_sig = self.min_signal_count_buy
            hold_sig = self.min_signal_count_hold

        if self._last_target_n is not None and self._last_regime == regime:
            target_n = int(np.clip(
                target_n,
                max(0, self._last_target_n - self.target_step_limit),
                min(self.max_positions, self._last_target_n + self.target_step_limit),
            ))

        return {
            "regime": regime,
            "target_gross": target_gross,
            "target_n": int(min(target_n, self.max_positions)),
            "turnover_limit": turnover_limit,
            "score_quantile_buy": buy_q,
            "score_quantile_hold": hold_q,
            "min_signal_count_buy": buy_sig,
            "min_signal_count_hold": hold_sig,
            "breadth": breadth,
            "breadth_pos": breadth_pos,
            "market_vol": market_vol,
            "trend_label": trend_label,
            "regime_score": regime_score,
        }

    def _apply_candidate_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "rvol_20" in out.columns and len(out) >= 20:
            high_vol_cut = out["rvol_20"].quantile(0.90)
            out = out[~((out["rvol_20"] >= high_vol_cut) & (out["signal_count"] < 5))].copy()
        if all(c in out.columns for c in ["close_to_high_60", "vol_compression", "money_flow_strength"]):
            out = out[
                ~(
                    (out["close_to_high_60"] >= 0.97)
                    & (out["vol_compression"] > 0.75)
                    & (out["money_flow_strength"] < 0.55)
                )
            ].copy()
        return out

    def _build_tiered_weights(self, selected_df: pd.DataFrame, target_gross: float) -> Dict[str, float]:
        if selected_df.empty or target_gross <= 0:
            return {}
        df = selected_df.sort_values("blended_score", ascending=False).reset_index(drop=True)
        n = len(df)
        top_n = max(1, int(np.ceil(n * 0.30)))
        mid_n = max(1, int(np.ceil(n * 0.40)))
        coeffs = np.full(n, 0.55, dtype=float)
        coeffs[:top_n] = 1.6
        coeffs[top_n:min(n, top_n + mid_n)] = 1.0
        coeffs = coeffs * (0.85 + 0.30 * df["signal_density"].fillna(0.0).to_numpy())
        coeffs = np.maximum(coeffs, 1e-6)
        weights = coeffs / coeffs.sum() * target_gross
        weights = np.minimum(weights, 0.12)
        if weights.sum() > 0:
            weights = weights / weights.sum() * target_gross
        return {code: float(w) for code, w in zip(df["ts_code"].tolist(), weights) if w > 1e-8}

    def _blend_for_turnover(self, new_weights: Dict[str, float], turnover_limit: float, target_n: Optional[int] = None) -> Dict[str, float]:
        old_weights = self._prev_target_weights or {}
        if not old_weights or turnover_limit >= 0.999:
            blended = dict(new_weights)
        else:
            all_codes = set(old_weights.keys()) | set(new_weights.keys())
            one_way_turnover = sum(abs(new_weights.get(c, 0.0) - old_weights.get(c, 0.0)) for c in all_codes) / 2.0
            if one_way_turnover <= turnover_limit:
                blended = dict(new_weights)
            else:
                blend_ratio = turnover_limit / one_way_turnover if one_way_turnover > 0 else 1.0
                blended = {}
                for code in all_codes:
                    w_old = old_weights.get(code, 0.0)
                    w_new = new_weights.get(code, 0.0)
                    w = w_old + blend_ratio * (w_new - w_old)
                    if w > 1e-8:
                        blended[code] = float(w)
                total = sum(blended.values())
                target_total = sum(new_weights.values())
                if total > 0 and target_total > 0:
                    blended = {k: v / total * target_total for k, v in blended.items()}
                print(f"      [换手] 原始单边换手 {one_way_turnover:.1%} > 上限 {turnover_limit:.0%}, 混合比例 {blend_ratio:.2f}")

        if target_n is not None and target_n > 0 and len(blended) > target_n:
            keep = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)[:target_n]
            blended = {k: float(v) for k, v in keep if v > 1e-8}
            total = sum(blended.values())
            target_total = sum(new_weights.values())
            if total > 0 and target_total > 0:
                blended = {k: v / total * target_total for k, v in blended.items()}
        return blended

    def _select_stocks_with_hold_buffer(
        self,
        scores_df: pd.DataFrame,
        feat_ranked: pd.DataFrame,
        current_holdings: Dict[str, int],
        regime_info: Dict[str, float],
    ) -> Dict[str, float]:
        df = scores_df.copy()
        signal_counts = self._count_smart_money_signals(feat_ranked)
        sc_map = pd.Series(signal_counts.values, index=feat_ranked["ts_code"].values)
        df["signal_count"] = df["ts_code"].map(sc_map).fillna(0).astype(int)
        df["signal_density"] = df["signal_count"] / 10.0
        df["held"] = df["ts_code"].isin(set(current_holdings.keys()))
        df = self._apply_candidate_filters(df)
        if df.empty:
            return {}

        score_std = df["ml_score"].std()
        df["ml_score_adj"] = df["ml_score"]
        if current_holdings and pd.notna(score_std) and score_std > 0 and self.buffer_sigma > 0:
            df.loc[df["held"], "ml_score_adj"] += self.buffer_sigma * score_std

        df["score_rank"] = df["ml_score_adj"].rank(pct=True, method="average")
        df["blended_score"] = 0.85 * df["score_rank"] + 0.15 * df["signal_density"]

        buy_threshold = df["ml_score"].quantile(regime_info["score_quantile_buy"])
        hold_threshold = df["ml_score"].quantile(regime_info["score_quantile_hold"])

        held_df = df[
            df["held"]
            & (df["ml_score_adj"] >= hold_threshold)
            & (df["signal_count"] >= regime_info["min_signal_count_hold"])
        ].copy().sort_values("blended_score", ascending=False)

        new_df = df[
            (~df["held"])
            & (df["ml_score"] >= buy_threshold)
            & (df["signal_count"] >= regime_info["min_signal_count_buy"])
        ].copy().sort_values("blended_score", ascending=False)

        selected = []
        industry_count: Dict[str, int] = {}
        target_n = int(regime_info["target_n"])

        def try_add(rows: pd.DataFrame):
            for _, row in rows.iterrows():
                if len(selected) >= target_n:
                    break
                code = row["ts_code"]
                ind = row.get("sw_l1", "unknown")
                if pd.isna(ind):
                    ind = "unknown"
                if any(r["ts_code"] == code for r in selected):
                    continue
                if industry_count.get(ind, 0) >= self.max_per_industry:
                    continue
                selected.append(row)
                industry_count[ind] = industry_count.get(ind, 0) + 1

        try_add(held_df)
        try_add(new_df)

        if not selected:
            return {}
        selected_df = pd.DataFrame(selected).sort_values("blended_score", ascending=False).reset_index(drop=True)
        weights = self._build_tiered_weights(selected_df, float(regime_info["target_gross"]))
        return self._blend_for_turnover(weights, float(regime_info["turnover_limit"]), target_n=target_n)

    def generate_target_weights(self, date: pd.Timestamp, accessor: DataAccessor, current_holdings: Dict[str, int]) -> Dict[str, float]:
        self._call_count += 1
        print(f"\n      ── Round1/V2 第 {self._call_count} 期  {date.strftime('%Y-%m-%d')} ──")

        if not self._warmup_done:
            self._warmup_done = True
            self._warmup_training_cache(date, accessor)

        if self._bulk_data is not None:
            feat_df = compute_features_from_memory(date, self._bulk_data, lookback=self.feature_lookback, st_codes=self._st_codes)
            window_df = self._bulk_data[self._bulk_data["trade_date"] <= pd.Timestamp(date)].tail(self.feature_lookback * 200).copy()
        else:
            window_df = accessor.get_window(date, lookback=self.feature_lookback, columns=FEATURE_COLUMNS)
            feat_df = compute_features_from_memory(date, window_df, lookback=self.feature_lookback, st_codes=self._st_codes)

        if feat_df is None or feat_df.empty:
            print("      [特征] ✗ 特征计算失败，本期跳过")
            return {}

        n_before_mv = len(feat_df)
        feat_df = self._apply_market_cap_filter(feat_df)
        print(f"      [特征] ✓ 有效股票 {n_before_mv} 只 → 市值过滤后 {len(feat_df)} 只")
        if len(feat_df) < 50:
            print("      [特征] ✗ 过滤后不足 50 只，本期跳过")
            return {}

        feat_ranked = rank_normalize(feat_df, FEATURE_NAMES)
        print("      [特征] ✓ 排序归一化完成")

        if self._train_data_cache:
            last_cached = self._train_data_cache[-1]
            last_date = pd.Timestamp(last_cached[0])
            if last_date < date and last_cached[2].empty:
                fwd_ret = compute_forward_return_from_memory(last_date, date, self._bulk_data) if self._bulk_data is not None else None
                if fwd_ret is not None:
                    self._train_data_cache[-1] = (last_cached[0], last_cached[1], fwd_ret)

        date_str = date.strftime("%Y-%m-%d")
        cached_dates = {d for d, _, _ in self._train_data_cache}
        if date_str not in cached_dates:
            self._train_data_cache.append((date_str, feat_ranked.copy(), pd.Series(dtype=float)))

        train_start = date - pd.DateOffset(years=self.train_window_years)
        self._train_data_cache = [(d, f, l) for d, f, l in self._train_data_cache if pd.Timestamp(d) >= train_start]
        labeled_cache = [(d, f, l) for d, f, l in self._train_data_cache if not l.empty and pd.Timestamp(d) < date]

        if self._should_retrain() and len(labeled_cache) >= 8:
            print(f"      [训练] 触发重训练 (缓存 {len(labeled_cache)} 期有标签数据) ...")
            all_X, all_y = [], []
            for _, f_df, labels in labeled_cache:
                merged = f_df.set_index("ts_code").join(labels.rename("label"), how="inner")
                if len(merged) < 30:
                    continue
                merged["label"] = merged["label"].rank(pct=True, method="average")
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
                    self._model = train_lgbm_model_fast(train_X, train_y, val_X, val_y)
                    self._last_train_date = date
                    best_iter = self._model.best_iteration if hasattr(self._model, "best_iteration") else "?"
                    print(
                        f"      [训练] ✓ 模型训练完成  "
                        f"使用 {len(all_X)} 期历史  "
                        f"最优轮次={best_iter}"
                    )
                except Exception as e:
                    print(f"      [训练] ✗ 训练失败: {e}")
            else:
                print(f"      [训练] ✗ 合并后训练期不足 (需要 ≥8, 当前 {len(all_X)})")
        else:
            print(f"      [训练] 跳过 (缓存有标签期 {len(labeled_cache)}，需要 ≥8 且到达重训间隔)")

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

        # Step 8: 调 _select_stocks，含 regime filter / 双阈值 / 分层权重 / 换手约束
        signal_counts = self._count_smart_money_signals(feat_ranked)
        n_with_signals = (signal_counts >= self.min_signal_count_buy).sum()
        score_threshold = result_df["ml_score"].quantile(self.score_quantile_buy)
        n_above_score = (result_df["ml_score"] >= score_threshold).sum()
        print(
            f"      [筛选] ML分数阈值={score_threshold:.4f} (买入≥{self.score_quantile_buy*100:.0f}%分位)  "
            f"达标={n_above_score} 只  |  "
            f"大资金信号≥{self.min_signal_count_buy}个: {n_with_signals} 只"
        )

        weights = self._select_stocks(result_df, feat_ranked, current_holdings)

        n_held = len(current_holdings) if current_holdings else 0
        n_industries = result_df[result_df["ts_code"].isin(weights.keys())]["sw_l1"].nunique() if weights else 0
        overlap = len(set(weights.keys()) & set(current_holdings.keys())) if current_holdings and weights else 0

        if len(weights) == 0:
            print(f"      [选股] ⚠ 本期无满足条件的股票，空仓")
        else:
            selected_signals = signal_counts[feat_ranked["ts_code"].isin(weights.keys())]
            avg_signals = selected_signals.mean() if len(selected_signals) > 0 else 0
            print(
                f"      [选股] 入选 {len(weights)} 只 / {n_industries} 个行业  "
                f"(上期持仓 {n_held} 只，留存 {overlap} 只，换手 {n_held + len(weights) - 2 * overlap} 只)  "
                f"平均信号数={avg_signals:.1f}"
            )
        return weights


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    import os
    cfg = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=1.5e-4,
        slippage=0.0015,
        start_date="2018-01-01",
        end_date="2025-12-31",
        rebalance_freq="W",
        db_path=os.path.join(os.path.dirname(__file__), "../data/quant.db"),
        baseline_dir=os.path.join(os.path.dirname(__file__), "../baseline"),
        output_dir=os.path.join(os.path.dirname(__file__), "../backtest"),
    )

    strategy = LGBMSmartMoneyV2(
        train_window_years=3,
        score_quantile_buy=0.93,
        score_quantile_hold=0.88,
        min_signal_count_buy=5,
        min_signal_count_hold=4,
        max_per_industry=2,
        max_positions=16,
        buffer_sigma=0.9,
        mv_pct_upper=0.85,
        feature_lookback=260,
        retrain_interval=8,
        turnover_limit=0.18,
        target_step_limit=3,
        backtest_end_date=cfg.end_date,
    )
    result = run_backtest(strategy, cfg)
