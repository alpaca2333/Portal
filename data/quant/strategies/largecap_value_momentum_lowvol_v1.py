"""
Large-Cap Value + Momentum + LowVol Strategy V1
===============================================

Design goals
------------
1. Simple and interpretable.
2. Avoid overfitting: no ML training, no regime model, no dynamic hyper-params.
3. Avoid look-ahead: only use current-date snapshot fields and historical prices
   up to the rebalance date via DataAccessor.
4. Use the existing event-driven backtest engine and OUTPUT.md-compliant outputs.

Important data constraint
-------------------------
The current `index_weight` table is empty in the local database, so this v1
cannot use true CSI300 constituent history. As a practical large-cap proxy,
it uses the top 300 stocks by current-date circulating market value (`circ_mv`).
"""
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import BacktestConfig, StrategyBase, run_backtest
from engine.data_loader import DataAccessor


ABS_PROJECT_ROOT = "/data/Projects/Portal/data/quant"
ABS_DB_PATH = f"{ABS_PROJECT_ROOT}/data/quant.db"
ABS_OUTPUT_DIR = f"{ABS_PROJECT_ROOT}/backtest"
ABS_BASELINE_DIR = f"{ABS_PROJECT_ROOT}/baseline_csi300_only"


class LargeCapValueMomentumLowVolV1(StrategyBase):
    def __init__(
        self,
        universe_size: int = 300,
        top_n: int = 40,
        max_per_industry: int = 8,
        lookback: int = 260,
        value_weight: float = 0.30,
        momentum_weight: float = 0.45,
        lowvol_weight: float = 0.25,
    ):
        super().__init__("largecap_value_momentum_lowvol_v1")
        self.universe_size = universe_size
        self.top_n = top_n
        self.max_per_industry = max_per_industry
        self.lookback = lookback
        self.value_weight = value_weight
        self.momentum_weight = momentum_weight
        self.lowvol_weight = lowvol_weight
        self._factor_cache: Dict[str, pd.DataFrame] = {}

    def describe(self) -> str:
        return (
            "### 策略定位\n\n"
            "这是一个面向 **大市值股票** 的规则型 v1 策略，优先追求 **可解释性、"
            "稳健性与低泄露风险**，而不是复杂模型堆料。\n\n"
            "### 为什么这样设计\n\n"
            "1. **避免过拟合**：不使用机器学习训练、不做超参搜索、不做市场状态切换。\n"
            "2. **避免未来数据泄露**：只使用调仓日当期可观测的截面字段（如 `circ_mv`、`pb`、`pe_ttm`）"
            "和调仓日前的历史收盘价窗口。\n"
            "3. **保证大市值风格纯度**：每期仅在 **当期流通市值前300** 的股票中选股。\n\n"
            "### 数据约束说明\n\n"
            "本地数据库中的 `index_weight` 表当前为空，因此无法严格复现历史沪深300成分股。"
            "本策略使用 **当期流通市值前300** 作为大市值代理股票池。这个近似虽然不等于"
            "指数增强，但在当前数据条件下是最稳妥、可运行、且不引入成分股未来泄露的实现方式。\n\n"
            "### 因子定义\n\n"
            "| 因子 | 构造方式 | 方向 | 权重 |\n"
            "|------|----------|------|------|\n"
            "| Value | 1/PB 与 1/PE_TTM 横截面均值 | 越高越好 | 30% |\n"
            "| Momentum | 6-1 月动量与 12-1 月动量横截面均值 | 越高越好 | 45% |\n"
            "| LowVol | 60日实现波动率取负后横截面排序 | 越高越好 | 25% |\n\n"
            f"### 组合构建\n\n"
            f"- 股票池：当期流通市值前 {self.universe_size} 股票\n"
            f"- 选股：综合得分前 {self.top_n} 名\n"
            f"- 权重：等权\n"
            f"- 行业约束：单一申万一级行业最多 {self.max_per_industry} 只\n"
            f"- 调仓频率：月频\n\n"
            "### 已知局限\n\n"
            "- 由于未使用真实指数成分权重，当前版本更接近 **大市值主动选股**，不是严格的指数增强。\n"
            "- 未使用财报质量因子（如 ROE、毛利率、负债率），是因为在未额外核验 PIT 口径前，"
            "v1 优先选择更安全的日频字段与价格衍生因子。\n"
            "- 未加入换手率惩罚和跟踪误差约束，后续 v2 可以逐步增强。"
        )

    @staticmethod
    def _rank01(series: pd.Series) -> pd.Series:
        ranked = series.rank(pct=True, method="average")
        return ranked.fillna(0.5)

    @staticmethod
    def _compute_momentum(close_pivot: pd.DataFrame, lookback_days: int, skip_days: int) -> pd.Series:
        n = len(close_pivot)
        start_idx = n - 1 - lookback_days - skip_days
        end_idx = n - 1 - skip_days
        if start_idx < 0 or end_idx < 0 or start_idx >= n or end_idx >= n:
            return pd.Series(np.nan, index=close_pivot.columns)
        start_price = close_pivot.iloc[start_idx]
        end_price = close_pivot.iloc[end_idx]
        mom = end_price / start_price - 1.0
        return mom.replace([np.inf, -np.inf], np.nan)

    def _build_factor_frame(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
    ) -> Optional[pd.DataFrame]:
        snap = accessor.get_date(
            date,
            columns=["close", "is_suspended", "circ_mv", "pb", "pe_ttm", "sw_l1"],
        )
        if snap.empty:
            return None

        snap = snap[
            (snap["is_suspended"] != 1)
            & snap["close"].notna()
            & (snap["close"] > 0)
            & snap["circ_mv"].notna()
            & (snap["circ_mv"] > 0)
        ].copy()

        snap = snap[
            snap["ts_code"].astype(str).str.match(r"^(6\d{5}\.SH|0\d{5}\.SZ|3\d{5}\.SZ)$")
        ].copy()
        if snap.empty:
            return None

        snap = snap.sort_values("circ_mv", ascending=False).head(self.universe_size).copy()
        if len(snap) < max(100, self.top_n):
            return None

        codes = snap["ts_code"].tolist()
        window = accessor.get_window(date, lookback=self.lookback, ts_codes=codes, columns=["close"])
        if window.empty:
            return None

        close_pivot = (
            window.pivot(index="trade_date", columns="ts_code", values="close")
            .sort_index()
            .ffill()
        )
        if close_pivot.empty or len(close_pivot) < 80:
            return None

        mom_6_1 = self._compute_momentum(close_pivot, lookback_days=120, skip_days=20)
        mom_12_1 = self._compute_momentum(close_pivot, lookback_days=240, skip_days=20)

        if len(close_pivot) >= 61:
            daily_ret = close_pivot.iloc[-61:].pct_change(fill_method=None).iloc[1:]
            rvol_60 = daily_ret.std().replace([np.inf, -np.inf], np.nan)
        else:
            rvol_60 = pd.Series(np.nan, index=close_pivot.columns)

        score_df = snap[["ts_code", "sw_l1", "circ_mv", "pb", "pe_ttm"]].copy()
        score_df = score_df.set_index("ts_code")
        score_df["inv_pb"] = np.where(score_df["pb"] > 0, 1.0 / score_df["pb"], np.nan)
        score_df["inv_pe_ttm"] = np.where(score_df["pe_ttm"] > 0, 1.0 / score_df["pe_ttm"], np.nan)
        score_df["mom_6_1"] = mom_6_1
        score_df["mom_12_1"] = mom_12_1
        score_df["rvol_60"] = rvol_60

        score_df["value_score"] = (
            self._rank01(score_df["inv_pb"]) + self._rank01(score_df["inv_pe_ttm"])
        ) / 2.0
        score_df["momentum_score"] = (
            self._rank01(score_df["mom_6_1"]) + self._rank01(score_df["mom_12_1"])
        ) / 2.0
        score_df["lowvol_score"] = self._rank01(-score_df["rvol_60"])

        score_df["composite_score"] = (
            self.value_weight * score_df["value_score"]
            + self.momentum_weight * score_df["momentum_score"]
            + self.lowvol_weight * score_df["lowvol_score"]
        )

        score_df = score_df.reset_index()
        score_df = score_df.sort_values(
            ["composite_score", "circ_mv"],
            ascending=[False, False],
        ).reset_index(drop=True)
        return score_df

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        score_df = self._build_factor_frame(date, accessor)
        if score_df is None or score_df.empty:
            return {}

        selected = []
        industry_counts: Dict[str, int] = {}
        for _, row in score_df.iterrows():
            industry = row.get("sw_l1")
            industry_key = industry if pd.notna(industry) and str(industry).strip() else "UNKNOWN"
            if industry_counts.get(industry_key, 0) >= self.max_per_industry:
                continue
            selected.append(row["ts_code"])
            industry_counts[industry_key] = industry_counts.get(industry_key, 0) + 1
            if len(selected) >= self.top_n:
                break

        if not selected:
            return {}

        date_key = pd.Timestamp(date).strftime("%Y%m%d")
        self._factor_cache[date_key] = score_df[
            ["ts_code", "value_score", "momentum_score", "lowvol_score", "composite_score"]
        ].copy()

        weight = 1.0 / len(selected)
        return {code: weight for code in selected}

    def get_factor_exposures(
        self,
        date: pd.Timestamp,
        selected_codes: Dict[str, float],
    ) -> pd.DataFrame | None:
        if not selected_codes:
            return None
        date_key = pd.Timestamp(date).strftime("%Y%m%d")
        factor_df = self._factor_cache.get(date_key)
        if factor_df is None or factor_df.empty:
            return None

        selected_df = factor_df[factor_df["ts_code"].isin(selected_codes.keys())].copy()
        if selected_df.empty:
            return None

        selected_df["weight"] = selected_df["ts_code"].map(selected_codes)
        return selected_df[[
            "ts_code",
            "weight",
            "value_score",
            "momentum_score",
            "lowvol_score",
            "composite_score",
        ]]


if __name__ == "__main__":
    os.makedirs(ABS_BASELINE_DIR, exist_ok=True)
    src = f"{ABS_PROJECT_ROOT}/baseline/000300.SH.csv"
    dst = f"{ABS_BASELINE_DIR}/000300.SH.csv"
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil
        shutil.copy2(src, dst)

    cfg = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=1.5e-4,
        slippage=0.001,
        start_date="2018-01-01",
        end_date="2025-12-31",
        rebalance_freq="M",
        db_path=ABS_DB_PATH,
        baseline_dir=ABS_BASELINE_DIR,
        output_dir=ABS_OUTPUT_DIR,
    )

    strategy = LargeCapValueMomentumLowVolV1(
        universe_size=300,
        top_n=40,
        max_per_industry=8,
        lookback=260,
        value_weight=0.30,
        momentum_weight=0.45,
        lowvol_weight=0.25,
    )
    run_backtest(strategy, cfg)
