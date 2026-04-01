"""
LightGBM Smart Money Round 2
=============================

Round 2 surgical repairs to Round 1 failures:
  A. Soft continuous regime scaling (no hard 3-bucket jumps)
  B. Narrow position-count range [14, 26], no starvation
  C. Remove portfolio-wide turnover blending; replace with name-level no-trade band + hard exits
  D. Enforce realized holdings discipline (min effective weight)
  E. Relaxed buy/hold thresholds vs Round 1
  F. Smooth conviction weighting (continuous rank-based, not coarse tiers)
  G. Candidate filters converted to score penalties, fewer hard exclusions

Evaluation target: Sharpe > 1.0, Annual > 15%, MaxDD better than -25%
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM required: pip install lightgbm")

from engine import BacktestConfig, StrategyBase, run_backtest
from engine.data_loader import DataAccessor
from strategies.lgbm_smart_money import (
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    compute_features_from_memory,
    compute_forward_return_from_memory,
    rank_normalize,
    train_lgbm_model,
)

CASH_PROXY = "__CASH__"


class LGBMSmartMoneyR2(StrategyBase):
    """Round 2: soft regime + no-trade band + conviction weights + realized-holdings discipline."""

    def __init__(
        self,
        train_window_years: int = 3,
        # --- Thresholds (relaxed vs R1) ---
        score_quantile_buy: float = 0.87,
        score_quantile_hold: float = 0.80,
        min_signal_count_buy: int = 3,
        min_signal_count_hold: int = 2,
        # --- Position discipline ---
        base_target_n: int = 22,
        min_target_n: int = 14,
        max_target_n: int = 26,
        max_per_industry: int = 3,
        # --- Gross exposure band ---
        base_gross: float = 0.90,
        min_gross: float = 0.55,
        max_gross: float = 0.95,
        # --- No-trade band (name-level friction) ---
        no_trade_band: float = 0.008,   # skip resize if |new_w - old_w| < this
        min_effective_weight: float = 0.025,  # drop sub-minimum names
        single_name_cap: float = 0.085,  # max per-name weight
        # --- Conviction weighting ---
        conviction_score_alpha: float = 1.3,   # convexity of weight curve
        signal_density_blend: float = 0.20,    # portion from signal density
        rvol_penalty_k: float = 0.5,           # risk penalty strength
        # --- Misc ---
        mv_pct_upper: float = 0.85,
        feature_lookback: int = 260,
        retrain_interval: int = 8,
        benchmark_code: str = "000300.SH",
        backtest_end_date: Optional[pd.Timestamp] = None,
    ):
        super().__init__("lgbm_smart_money_r2")
        self.train_window_years = train_window_years
        self.score_quantile_buy = score_quantile_buy
        self.score_quantile_hold = score_quantile_hold
        self.min_signal_count_buy = min_signal_count_buy
        self.min_signal_count_hold = min_signal_count_hold
        self.base_target_n = base_target_n
        self.min_target_n = min_target_n
        self.max_target_n = max_target_n
        self.max_per_industry = max_per_industry
        self.base_gross = base_gross
        self.min_gross = min_gross
        self.max_gross = max_gross
        self.no_trade_band = no_trade_band
        self.min_effective_weight = min_effective_weight
        self.single_name_cap = single_name_cap
        self.conviction_score_alpha = conviction_score_alpha
        self.signal_density_blend = signal_density_blend
        self.rvol_penalty_k = rvol_penalty_k
        self.mv_pct_upper = mv_pct_upper
        self.feature_lookback = feature_lookback
        self.retrain_interval = retrain_interval
        self.benchmark_code = benchmark_code
        self._backtest_end_date = pd.Timestamp(backtest_end_date) if backtest_end_date is not None else None

        self._model = None
        self._train_data_cache: List[Tuple[pd.Timestamp, pd.DataFrame, pd.Series]] = []
        self._last_train_date: Optional[pd.Timestamp] = None
        self._call_count = 0
        self._st_codes: Optional[set] = None
        self._benchmark_cache: Optional[pd.DataFrame] = None
        self._prev_target_weights: Dict[str, float] = {}
        self._bulk_data: Optional[pd.DataFrame] = None
        self._warmup_done = False

    def describe(self) -> str:
        return (
            "Round 2 智能资金策略：软状态连续仓位缩放 + 名单级无交易区间 + 平滑信念加权 + 实际持仓纪律。"
            "核心改进：去除组合级换手混合（Round 1主要失败点）、收窄持股数区间[14,26]、"
            "用连续信念权重替代粗粒度分层、最低有效权重过滤残留小仓。"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_retrain(self) -> bool:
        if self._model is None:
            return True
        return self._call_count % self.retrain_interval == 0

    def _load_st_codes(self, accessor: DataAccessor):
        if self._st_codes is not None:
            return
        try:
            df = pd.read_sql_query(
                "SELECT ts_code FROM stock_info WHERE name LIKE '%ST%'",
                accessor.conn,
            )
            self._st_codes = set(df["ts_code"].tolist())
        except Exception:
            self._st_codes = set()

    def _load_benchmark_df(self) -> Optional[pd.DataFrame]:
        if self._benchmark_cache is not None:
            return self._benchmark_cache
        quant_root = Path(__file__).resolve().parents[1]
        for code in [self.benchmark_code, "000001.SH"]:
            p = quant_root / "baseline" / f"{code}.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["date"] = pd.to_datetime(df["date"])
                df = df[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
                self._benchmark_cache = df
                return df
        return None

    def _apply_market_cap_filter(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        if "circ_mv" not in feat_df.columns:
            return feat_df
        mv = feat_df["circ_mv"].dropna()
        if len(mv) == 0:
            return feat_df
        lower = mv.quantile(1.0 - self.mv_pct_upper)
        return feat_df[feat_df["circ_mv"] >= lower].copy()

    def _count_smart_money_signals(self, feat_ranked: pd.DataFrame) -> pd.Series:
        signals = pd.DataFrame(index=feat_ranked.index)
        for col in ["vol_surge_ratio", "obv_slope", "lower_shadow_ratio", "money_flow_strength", "illiq_change"]:
            if col in feat_ranked.columns:
                signals[col] = (feat_ranked[col] >= 0.70).astype(int)
        for col in ["shrink_pullback", "vol_compression", "ma_convergence", "turnover_concentration"]:
            if col in feat_ranked.columns:
                signals[col] = (feat_ranked[col] <= 0.30).astype(int)
        if "bottom_deviation" in feat_ranked.columns:
            signals["bottom_deviation"] = (feat_ranked["bottom_deviation"] <= 0.40).astype(int)
        return signals.sum(axis=1)

    # ------------------------------------------------------------------
    # A. Soft continuous regime scaling
    # ------------------------------------------------------------------

    def _compute_market_regime_soft(
        self, date: pd.Timestamp, window_df: pd.DataFrame
    ) -> Dict:
        """
        Returns continuous gross_target, position_target, and extreme_risk_off flag.
        Penalties are additive; only all-bad condition triggers extreme defense.
        """
        trend_score = 0
        breadth_score = 0
        vol_score = 0
        breadth = np.nan
        breadth_pos = np.nan
        market_vol = np.nan
        vol_ratio = np.nan
        trend_label = "mixed"

        # --- Benchmark trend ---
        bench = self._load_benchmark_df()
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

        # --- Market breadth + vol from individual stocks ---
        w = window_df[
            window_df["ts_code"].astype(str).str.match(r"^(6\d{5}\.SH|00[013]\d{3}\.SZ)$")
            & (window_df.get("is_suspended", pd.Series(0, index=window_df.index)) != 1)
            & (window_df["close"].notna())
            & (window_df["close"] > 0)
        ][["trade_date", "ts_code", "close"]].copy()

        if not w.empty:
            cp = w.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
            if len(cp) >= 21 and cp.shape[1] >= 100:
                ma20_s = cp.iloc[-20:].mean()
                cur = cp.iloc[-1]
                breadth = float((cur > ma20_s).mean())
                ret20 = cur / cp.iloc[-21] - 1
                breadth_pos = float((ret20 > 0).mean())
                mkt = cp.median(axis=1)
                rets = mkt.pct_change().dropna()
                if len(rets) >= 60:
                    market_vol = float(rets.iloc[-20:].std())
                    hist_vol = float(rets.iloc[-60:].std())
                    vol_ratio = market_vol / hist_vol if hist_vol > 0 else np.nan
                elif len(rets) >= 20:
                    market_vol = float(rets.iloc[-20:].std())

        if pd.notna(breadth) and pd.notna(breadth_pos):
            if breadth >= 0.55 and breadth_pos >= 0.52:
                breadth_score = 1
            elif breadth <= 0.44 and breadth_pos <= 0.44:
                breadth_score = -1

        if pd.notna(vol_ratio):
            if vol_ratio >= 1.20:
                vol_score = -1
            elif vol_ratio <= 0.90:
                vol_score = 1

        combined = trend_score + breadth_score + vol_score  # range: [-3, +3]

        # --- Continuous gross exposure ---
        # Normal band: 0.80–0.95; soft defense: 0.65–0.80; extreme: 0.35–0.55
        if combined >= 2:
            gross_target = min(self.max_gross, self.base_gross + 0.05)
        elif combined == 1:
            gross_target = self.base_gross
        elif combined == 0:
            gross_target = self.base_gross - 0.08          # ~0.82
        elif combined == -1:
            gross_target = self.base_gross - 0.18          # ~0.72
        elif combined == -2:
            gross_target = self.base_gross - 0.30          # ~0.60
        else:  # -3: all bad
            gross_target = self.base_gross - 0.45          # ~0.45

        gross_target = float(np.clip(gross_target, self.min_gross, self.max_gross))
        extreme_risk_off = combined <= -3

        # --- Position count ---
        pos_adj = int(combined * 2)  # [-6, +6]
        position_target = int(np.clip(self.base_target_n + pos_adj, self.min_target_n, self.max_target_n))

        # --- Mild threshold adaptation ---
        if combined >= 1:
            buy_q = self.score_quantile_buy
            hold_q = self.score_quantile_hold
            buy_sig = self.min_signal_count_buy
            hold_sig = self.min_signal_count_hold
        else:
            buy_q = min(0.92, self.score_quantile_buy + 0.02)
            hold_q = min(0.86, self.score_quantile_hold + 0.02)
            buy_sig = min(4, self.min_signal_count_buy + 1)
            hold_sig = min(3, self.min_signal_count_hold + 1)

        return {
            "combined": combined,
            "trend_label": trend_label,
            "breadth": breadth,
            "breadth_pos": breadth_pos,
            "vol_ratio": vol_ratio,
            "gross_target": gross_target,
            "position_target": position_target,
            "extreme_risk_off": extreme_risk_off,
            "score_quantile_buy": buy_q,
            "score_quantile_hold": hold_q,
            "min_signal_count_buy": buy_sig,
            "min_signal_count_hold": hold_sig,
        }

    # ------------------------------------------------------------------
    # B/C/D/E. Stock selection with hold-asymmetry + no-trade band
    # ------------------------------------------------------------------

    def _select_candidates(
        self,
        scores_df: pd.DataFrame,
        feat_ranked: pd.DataFrame,
        current_holdings: Dict[str, float],
        regime: Dict,
    ) -> pd.DataFrame:
        """Score all candidates, apply score penalties (not hard filters), select held+new."""
        df = scores_df.copy()
        signal_counts = self._count_smart_money_signals(feat_ranked)
        sc_map = pd.Series(signal_counts.values, index=feat_ranked["ts_code"].values)
        df["signal_count"] = df["ts_code"].map(sc_map).fillna(0).astype(int)
        df["signal_density"] = df["signal_count"] / 10.0
        df["held"] = df["ts_code"].isin(set(current_holdings.keys()))

        # --- F. Score penalties instead of hard filters ---
        penalty = pd.Series(0.0, index=df.index)
        if "rvol_20" in df.columns:
            rvol_rank = df["rvol_20"].rank(pct=True)
            # High vol AND low signals → soft penalty on score
            high_vol = rvol_rank >= 0.90
            low_sig = df["signal_count"] < 4
            penalty[high_vol & low_sig] -= 0.06
        # Near 60d high with poor flow: penalty, not hard exclude
        if all(c in df.columns for c in ["close_to_high_60", "vol_compression", "money_flow_strength"]):
            near_high = df["close_to_high_60"] >= 0.97
            poor_flow = df["money_flow_strength"] < 0.55
            penalty[near_high & poor_flow] -= 0.03

        # --- Hold buffer: held names get +0.5σ on ml_score ---
        score_std = df["ml_score"].std()
        df["ml_score_adj"] = df["ml_score"].copy()
        if score_std > 0:
            df.loc[df["held"], "ml_score_adj"] += 0.5 * score_std
        df["ml_score_adj"] += penalty

        df["score_rank"] = df["ml_score_adj"].rank(pct=True, method="average")
        df["conviction_score"] = (
            (1.0 - self.signal_density_blend) * df["score_rank"]
            + self.signal_density_blend * df["signal_density"]
        )

        # Thresholds from unadjusted ml_score
        buy_thr = df["ml_score"].quantile(regime["score_quantile_buy"])
        hold_thr = df["ml_score"].quantile(regime["score_quantile_hold"])

        held_sel = df[
            df["held"]
            & (df["ml_score_adj"] >= hold_thr)
            & (df["signal_count"] >= regime["min_signal_count_hold"])
        ].sort_values("conviction_score", ascending=False)

        new_sel = df[
            (~df["held"])
            & (df["ml_score"] >= buy_thr)
            & (df["signal_count"] >= regime["min_signal_count_buy"])
        ].sort_values("conviction_score", ascending=False)

        selected = []
        industry_count: Dict[str, int] = {}
        target_n = regime["position_target"]

        def try_add(rows):
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
                selected.append(row.to_dict())
                industry_count[ind] = industry_count.get(ind, 0) + 1

        try_add(held_sel)
        try_add(new_sel)

        if not selected:
            return pd.DataFrame()
        return pd.DataFrame(selected).reset_index(drop=True)

    # ------------------------------------------------------------------
    # F. Smooth conviction weighting
    # ------------------------------------------------------------------

    def _build_conviction_weights(
        self, selected_df: pd.DataFrame, gross_target: float
    ) -> Dict[str, float]:
        """Smooth monotonic conviction weights with risk penalty and min-weight enforcement."""
        if selected_df.empty or gross_target <= 0:
            return {}

        df = selected_df.sort_values("conviction_score", ascending=False).reset_index(drop=True)
        n = len(df)

        # Smooth raw weights via rank-based convexity
        ranks = np.arange(1, n + 1)
        raw = (n + 1 - ranks) ** self.conviction_score_alpha  # higher rank → higher weight

        # Risk adjustment from rvol if available
        if "rvol_20" in df.columns:
            rvol_rank = df["rvol_20"].rank(pct=True).fillna(0.5).to_numpy()
            risk_adj = 1.0 / (1.0 + self.rvol_penalty_k * rvol_rank)
        else:
            risk_adj = np.ones(n)

        raw = raw * risk_adj
        raw = np.maximum(raw, 1e-8)

        # Normalize to gross target
        weights = raw / raw.sum() * gross_target

        # Single-name cap
        weights = np.minimum(weights, self.single_name_cap)

        # Re-normalize after cap
        total = weights.sum()
        if total > 0:
            weights = weights / total * gross_target

        # --- D. Realized holdings discipline: drop sub-minimum names ---
        codes = df["ts_code"].tolist()
        result = {c: float(w) for c, w in zip(codes, weights) if w >= self.min_effective_weight}

        if not result:
            # fallback: keep top names
            top_n = max(1, n // 2)
            result = {codes[i]: float(weights[i]) for i in range(top_n)}

        # Re-normalize after drops
        total = sum(result.values())
        if total > 0:
            result = {k: v / total * gross_target for k, v in result.items()}

        return result

    # ------------------------------------------------------------------
    # C. Name-level no-trade band + forced exits (replaces turnover blending)
    # ------------------------------------------------------------------

    def _apply_trade_band_and_cleanup(
        self,
        new_weights: Dict[str, float],
        old_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        - Skip tiny weight changes (no-trade band)
        - Force full exits on names that dropped out of new_weights
        - Remove sub-minimum tail positions
        """
        if not old_weights:
            return dict(new_weights)

        result = {}

        # Names in new target
        for code, w_new in new_weights.items():
            w_old = old_weights.get(code, 0.0)
            delta = abs(w_new - w_old)
            if delta < self.no_trade_band and code in old_weights:
                # Skip resize — keep old weight
                result[code] = w_old
            else:
                result[code] = w_new

        # Names that were held but are NOT in new target → force full exit (don't preserve)
        # (i.e., we do NOT carry over old names that are missing from new_weights)

        # Drop sub-minimum residuals
        result = {k: v for k, v in result.items() if v >= self.min_effective_weight}

        if not result:
            return dict(new_weights)

        # Renormalize to same gross as new_weights
        target_gross = sum(new_weights.values())
        total = sum(result.values())
        if total > 0 and target_gross > 0:
            result = {k: v / total * target_gross for k, v in result.items()}

        return result

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        self._call_count += 1
        print(f"\n      ── R2 第 {self._call_count} 期  {date.strftime('%Y-%m-%d')} ──")

        self._load_st_codes(accessor)

        # ---- Data ----
        window_df = accessor.get_window(date, lookback=self.feature_lookback, columns=FEATURE_COLUMNS)
        if window_df is None or window_df.empty:
            print("      [特征] ✗ 窗口数据为空，跳过")
            return {}

        feat_df = compute_features_from_memory(
            date, window_df, lookback=self.feature_lookback, st_codes=self._st_codes
        )
        if feat_df is None or feat_df.empty:
            print("      [特征] ✗ 特征为空，跳过")
            return {}

        feat_df = self._apply_market_cap_filter(feat_df)
        if len(feat_df) < 50:
            print(f"      [特征] ✗ 市值过滤后仅 {len(feat_df)} 只，跳过")
            return {}

        feat_ranked = rank_normalize(feat_df, FEATURE_NAMES)
        print(f"      [特征] ✓ {len(feat_ranked)} 只候选")

        # ---- Training cache ----
        if self._train_data_cache:
            prev_date, prev_feat, prev_label = self._train_data_cache[-1]
            if prev_label.empty and prev_date < date:
                fwd = self._compute_forward_return(prev_date, date, accessor)
                if fwd is not None:
                    self._train_data_cache[-1] = (prev_date, prev_feat, fwd)

        if not self._train_data_cache or self._train_data_cache[-1][0] != date:
            self._train_data_cache.append((date, feat_ranked.copy(), pd.Series(dtype=float)))

        cutoff = date - pd.DateOffset(years=self.train_window_years)
        self._train_data_cache = [x for x in self._train_data_cache if x[0] >= cutoff]
        labeled = [(d, f, l) for d, f, l in self._train_data_cache if not l.empty and d < date]

        # ---- Model training ----
        if self._should_retrain() and len(labeled) >= 8:
            print(f"      [训练] 触发重训练 ({len(labeled)} 期标签) ...")
            all_X, all_y = [], []
            for _, f_df, lbl in labeled:
                m = f_df.set_index("ts_code").join(lbl.rename("label"), how="inner")
                if len(m) < 30:
                    continue
                m["label"] = m["label"].rank(pct=True, method="average")
                all_X.append(m[FEATURE_NAMES])
                all_y.append(m["label"])
            if len(all_X) >= 8:
                n = len(all_X)
                sp = max(1, int(n * 0.8))
                try:
                    self._model = train_lgbm_model(
                        pd.concat(all_X[:sp]),
                        pd.concat(all_y[:sp]),
                        pd.concat(all_X[sp:]) if sp < n else None,
                        pd.concat(all_y[sp:]) if sp < n else None,
                    )
                    self._last_train_date = date
                    bi = getattr(self._model, "best_iteration", "?")
                    print(f"      [训练] ✓ 完成 期数={n} 最优轮次={bi}")
                except Exception as e:
                    print(f"      [训练] ✗ {e}")
        elif self._model is None:
            print(f"      [训练] 冷启动 标签期数={len(labeled)}")

        # ---- Prediction ----
        result_df = feat_ranked[["ts_code", "sw_l1"] + FEATURE_NAMES].copy()
        if self._model is None:
            result_df["ml_score"] = feat_ranked[FEATURE_NAMES].mean(axis=1).values
        else:
            X_pred = feat_ranked[FEATURE_NAMES].fillna(feat_ranked[FEATURE_NAMES].median())
            result_df["ml_score"] = self._model.predict(X_pred)

        # Add rvol for weighting if available
        if "rvol_20" in feat_df.columns:
            rvol_map = feat_df.set_index("ts_code")["rvol_20"]
            result_df["rvol_20"] = result_df["ts_code"].map(rvol_map)
        if "close_to_high_60" in feat_df.columns:
            result_df["close_to_high_60"] = result_df["ts_code"].map(feat_df.set_index("ts_code")["close_to_high_60"])
        if "vol_compression" in feat_df.columns:
            result_df["vol_compression"] = result_df["ts_code"].map(feat_df.set_index("ts_code")["vol_compression"])
        if "money_flow_strength" in feat_df.columns:
            result_df["money_flow_strength"] = result_df["ts_code"].map(feat_df.set_index("ts_code")["money_flow_strength"])

        # ---- Regime ----
        regime = self._compute_market_regime_soft(date, window_df)
        print(
            f"      [Regime] combined={regime['combined']:+d} trend={regime['trend_label']} "
            f"breadth={regime['breadth'] if pd.notna(regime['breadth']) else 'n/a':.2f} "
            f"gross={regime['gross_target']:.0%} pos_target={regime['position_target']} "
            f"extreme_off={regime['extreme_risk_off']}"
        )

        if regime["extreme_risk_off"]:
            print("      [Regime] ⚠ 极端风险关闭，全仓现金")
            self._prev_target_weights = {}
            return {CASH_PROXY: 1.0}

        # ---- Selection ----
        old_weights = dict(self._prev_target_weights)
        selected_df = self._select_candidates(result_df, feat_ranked, old_weights, regime)

        if selected_df.empty:
            print("      [选股] ⚠ 无满足条件股票，持现金")
            return {CASH_PROXY: 1.0}

        # ---- Weighting ----
        raw_weights = self._build_conviction_weights(selected_df, regime["gross_target"])
        if not raw_weights:
            print("      [权重] ✗ 权重生成失败")
            return {CASH_PROXY: 1.0}

        # ---- No-trade band + cleanup ----
        final_weights = self._apply_trade_band_and_cleanup(raw_weights, old_weights)

        # ---- Diagnostics ----
        invested = sum(final_weights.values())
        cash_w = max(0.0, 1.0 - invested)
        n_names = len(final_weights)
        overlap = len(set(final_weights.keys()) & set(old_weights.keys()))
        one_way_to = sum(abs(final_weights.get(c, 0.0) - old_weights.get(c, 0.0)) for c in set(final_weights) | set(old_weights)) / 2.0
        min_w = min(final_weights.values()) if final_weights else 0.0
        max_w = max(final_weights.values()) if final_weights else 0.0
        print(
            f"      [组合] 持股={n_names} 留存={overlap} 仓位={invested:.0%} 现金={cash_w:.0%} "
            f"单边换手={one_way_to:.1%} 权重[{min_w:.2%}~{max_w:.2%}]"
        )

        # ---- Cash ----
        if cash_w > 1e-8:
            final_weights[CASH_PROXY] = cash_w

        self._prev_target_weights = {k: v for k, v in final_weights.items() if k != CASH_PROXY}
        return final_weights

    def _compute_forward_return(
        self,
        date: pd.Timestamp,
        next_date: pd.Timestamp,
        accessor: DataAccessor,
    ) -> Optional[pd.Series]:
        try:
            window = accessor.get_window(next_date, lookback=5, columns=["ts_code", "trade_date", "close", "adj_factor"])
            if window is None or window.empty:
                return None
            d1 = window[window["trade_date"] == date][["ts_code", "close", "adj_factor"]].rename(columns={"close": "c1", "adj_factor": "af1"})
            d2 = window[window["trade_date"] == next_date][["ts_code", "close", "adj_factor"]].rename(columns={"close": "c2", "adj_factor": "af2"})
            merged = d1.merge(d2, on="ts_code")
            if merged.empty:
                return None
            merged["fwd"] = (merged["c2"] * merged["af2"]) / (merged["c1"] * merged["af1"]) - 1
            return merged.set_index("ts_code")["fwd"]
        except Exception:
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Round 2 backtest")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output-dir", default="data/quant/backtest")
    args = parser.parse_args()

    cfg = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=1.5e-4,
        slippage=0.0015,
        start_date=args.start,
        end_date=args.end,
        rebalance_freq="W",
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir=args.output_dir,
    )

    strategy = LGBMSmartMoneyR2(
        train_window_years=3,
        score_quantile_buy=0.87,
        score_quantile_hold=0.80,
        min_signal_count_buy=3,
        min_signal_count_hold=2,
        base_target_n=22,
        min_target_n=14,
        max_target_n=26,
        max_per_industry=3,
        base_gross=0.90,
        min_gross=0.55,
        max_gross=0.95,
        no_trade_band=0.008,
        min_effective_weight=0.025,
        single_name_cap=0.085,
        conviction_score_alpha=1.3,
        signal_density_blend=0.20,
        rvol_penalty_k=0.5,
        mv_pct_upper=0.85,
        feature_lookback=260,
        retrain_interval=8,
        benchmark_code="000300.SH",
        backtest_end_date=pd.Timestamp(args.end),
    )

    run_backtest(strategy, cfg)
