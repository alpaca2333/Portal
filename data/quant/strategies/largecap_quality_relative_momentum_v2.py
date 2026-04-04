"""
Large-Cap Quality + Relative Momentum Strategy V2
=================================================

Design goals
------------
1. Keep the strategy simple, interpretable, and directly runnable.
2. Preserve large-cap style purity while reducing defensive value traps.
3. Avoid look-ahead bias by using point-in-time daily snapshot fields and
   historical prices only up to the rebalance date.
4. Improve v1 by adding quality, industry-relative momentum, stronger industry
   diversification, and a holding buffer to reduce ranking-boundary churn.

Data notes
----------
- The local `index_weight` table is currently unavailable for usable historical
  CSI300 constituent reconstruction, so we still use current-date top 300 by
  circulating market value as the large-cap proxy universe.
- Financial fields inside `stock_daily` are built from `fina_indicator` with
  effective date = next trading day after `ann_date`, which makes them safe to
  use in this backtest framework.
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


class LargeCapQualityRelativeMomentumV2(StrategyBase):
    def __init__(
        self,
        universe_size: int = 300,
        target_n: int = 40,
        buy_top_n: int = 30,
        hold_top_n: int = 60,
        max_per_industry: int = 6,
        lookback: int = 260,
        quality_weight: float = 0.35,
        rel_momentum_weight: float = 0.35,
        value_weight: float = 0.15,
        lowvol_weight: float = 0.15,
    ):
        super().__init__("largecap_quality_relative_momentum_v2")
        self.universe_size = universe_size
        self.target_n = target_n
        self.buy_top_n = buy_top_n
        self.hold_top_n = hold_top_n
        self.max_per_industry = max_per_industry
        self.lookback = lookback
        self.quality_weight = quality_weight
        self.rel_momentum_weight = rel_momentum_weight
        self.value_weight = value_weight
        self.lowvol_weight = lowvol_weight
        self._factor_cache: Dict[str, pd.DataFrame] = {}

    def describe(self) -> str:
        return (
            "### 策略定位\n\n"
            "这是一个面向 **大市值股票** 的规则型 v2 策略，核心目标是："
            "在保持可解释性与低泄露风险的前提下，修复 v1 过度偏向低估值防守行业的问题。\n\n"
            "### 相比 v1 的关键改进\n\n"
            "1. **加入质量因子**：不再只靠低估值与低波，减少‘便宜但不涨’的大票陷阱。\n"
            "2. **改用行业相对动量**：优先选择行业内部更强的大票，而不是只买全市场最便宜的老资产。\n"
            "3. **更严格的行业约束**：单一申万一级行业最多 6 只，降低银行/建筑/公用事业式抱团。\n"
            "4. **加入持仓缓冲区**：新买入要求进入前 30，原持仓只要没跌出前 60 就可保留，从而降低边界抖动与无效换手。\n\n"
            "### 数据约束说明\n\n"
            "本地数据库中的 `index_weight` 历史成分数据当前仍不可直接用于严格沪深300增强复现，"
            "因此继续使用 **当期流通市值前300** 作为大市值代理股票池。\n\n"
            "### 因子定义\n\n"
            "| 因子 | 构造方式 | 方向 | 权重 |\n"
            "|------|----------|------|------|\n"
            "| Quality | ROE、ROA、毛利率、低负债率的横截面均值 | 越高越好 | 35% |\n"
            "| Relative Momentum | 3-1月、6-1月、12-1月动量减去行业均值后的横截面均值 | 越高越好 | 35% |\n"
            "| Value | 1/PB 与 1/PE_TTM 横截面均值 | 越高越好 | 15% |\n"
            "| LowVol | 60日实现波动率取负后横截面排序 | 越高越好 | 15% |\n\n"
            "### 组合构建\n\n"
            f"- 股票池：当期流通市值前 {self.universe_size} 股票\n"
            f"- 目标持仓：{self.target_n} 只\n"
            f"- 新买入阈值：综合得分前 {self.buy_top_n} 名\n"
            f"- 继续持有阈值：综合得分前 {self.hold_top_n} 名\n"
            f"- 行业约束：单一申万一级行业最多 {self.max_per_industry} 只\n"
            "- 权重：等权\n"
            "- 调仓频率：月频\n\n"
            "### 未来数据安全性\n\n"
            "价格因子仅使用调仓日及之前历史窗口；财务字段来自本地构库脚本的 PIT 合并逻辑，"
            "按公告日次一交易日生效，因此可用于当前框架。\n\n"
            "### 已知局限\n\n"
            "- 仍然不是严格意义上的 CSI300 指数增强。\n"
            "- 未加入显式跟踪误差约束与行业相对基准偏离约束。\n"
            "- 仍采用等权，未做更复杂的容量或优化器约束。"
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

    @staticmethod
    def _industry_demean(df: pd.DataFrame, value_col: str, industry_col: str = "sw_l1") -> pd.Series:
        industry_mean = df.groupby(industry_col)[value_col].transform("mean")
        return df[value_col] - industry_mean

    def _build_factor_frame(self, date: pd.Timestamp, accessor: DataAccessor) -> Optional[pd.DataFrame]:
        snap = accessor.get_date(
            date,
            columns=[
                "close",
                "is_suspended",
                "circ_mv",
                "pb",
                "pe_ttm",
                "sw_l1",
                "roe",
                "roa",
                "grossprofit_margin",
                "debt_to_assets",
            ],
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
        if len(snap) < max(100, self.hold_top_n):
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

        mom_3_1 = self._compute_momentum(close_pivot, lookback_days=60, skip_days=20)
        mom_6_1 = self._compute_momentum(close_pivot, lookback_days=120, skip_days=20)
        mom_12_1 = self._compute_momentum(close_pivot, lookback_days=240, skip_days=20)

        if len(close_pivot) >= 61:
            daily_ret = close_pivot.iloc[-61:].pct_change(fill_method=None).iloc[1:]
            rvol_60 = daily_ret.std().replace([np.inf, -np.inf], np.nan)
        else:
            rvol_60 = pd.Series(np.nan, index=close_pivot.columns)

        score_df = snap[
            [
                "ts_code",
                "sw_l1",
                "circ_mv",
                "pb",
                "pe_ttm",
                "roe",
                "roa",
                "grossprofit_margin",
                "debt_to_assets",
            ]
        ].copy()
        score_df = score_df.set_index("ts_code")

        score_df["inv_pb"] = np.where(score_df["pb"] > 0, 1.0 / score_df["pb"], np.nan)
        score_df["inv_pe_ttm"] = np.where(score_df["pe_ttm"] > 0, 1.0 / score_df["pe_ttm"], np.nan)

        score_df["mom_3_1"] = mom_3_1
        score_df["mom_6_1"] = mom_6_1
        score_df["mom_12_1"] = mom_12_1
        score_df["rvol_60"] = rvol_60

        score_df["rel_mom_3_1"] = self._industry_demean(score_df, "mom_3_1")
        score_df["rel_mom_6_1"] = self._industry_demean(score_df, "mom_6_1")
        score_df["rel_mom_12_1"] = self._industry_demean(score_df, "mom_12_1")

        score_df["quality_score"] = (
            self._rank01(score_df["roe"])
            + self._rank01(score_df["roa"])
            + self._rank01(score_df["grossprofit_margin"])
            + self._rank01(-score_df["debt_to_assets"])
        ) / 4.0

        score_df["relative_momentum_score"] = (
            self._rank01(score_df["rel_mom_3_1"])
            + self._rank01(score_df["rel_mom_6_1"])
            + self._rank01(score_df["rel_mom_12_1"])
        ) / 3.0

        score_df["value_score"] = (
            self._rank01(score_df["inv_pb"]) + self._rank01(score_df["inv_pe_ttm"])
        ) / 2.0
        score_df["lowvol_score"] = self._rank01(-score_df["rvol_60"])

        score_df["composite_score"] = (
            self.quality_weight * score_df["quality_score"]
            + self.rel_momentum_weight * score_df["relative_momentum_score"]
            + self.value_weight * score_df["value_score"]
            + self.lowvol_weight * score_df["lowvol_score"]
        )

        score_df = score_df.reset_index()
        score_df = score_df.sort_values(
            ["composite_score", "circ_mv"],
            ascending=[False, False],
        ).reset_index(drop=True)
        score_df["rank"] = np.arange(1, len(score_df) + 1)
        return score_df

    def _select_codes(self, score_df: pd.DataFrame, current_holdings: Dict[str, int]) -> list[str]:
        current_set = set(current_holdings.keys())
        selected: list[str] = []
        selected_set = set()
        industry_counts: Dict[str, int] = {}

        def try_add_rows(df: pd.DataFrame):
            for _, row in df.iterrows():
                code = row["ts_code"]
                if code in selected_set:
                    continue
                industry = row.get("sw_l1")
                industry_key = industry if pd.notna(industry) and str(industry).strip() else "UNKNOWN"
                if industry_counts.get(industry_key, 0) >= self.max_per_industry:
                    continue
                selected.append(code)
                selected_set.add(code)
                industry_counts[industry_key] = industry_counts.get(industry_key, 0) + 1
                if len(selected) >= self.target_n:
                    break

        eligible = score_df[
            (score_df["rank"] <= self.buy_top_n)
            | ((score_df["rank"] <= self.hold_top_n) & score_df["ts_code"].isin(current_set))
        ].copy()

        try_add_rows(eligible)
        if len(selected) < self.target_n:
            try_add_rows(score_df)

        return selected[: self.target_n]

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        score_df = self._build_factor_frame(date, accessor)
        if score_df is None or score_df.empty:
            return {}

        selected = self._select_codes(score_df, current_holdings)
        if not selected:
            return {}

        date_key = pd.Timestamp(date).strftime("%Y%m%d")
        self._factor_cache[date_key] = score_df[
            [
                "ts_code",
                "quality_score",
                "relative_momentum_score",
                "value_score",
                "lowvol_score",
                "composite_score",
            ]
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
        return selected_df[
            [
                "ts_code",
                "weight",
                "quality_score",
                "relative_momentum_score",
                "value_score",
                "lowvol_score",
                "composite_score",
            ]
        ]


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

    strategy = LargeCapQualityRelativeMomentumV2(
        universe_size=300,
        target_n=40,
        buy_top_n=30,
        hold_top_n=60,
        max_per_industry=6,
        lookback=260,
        quality_weight=0.35,
        rel_momentum_weight=0.35,
        value_weight=0.15,
        lowvol_weight=0.15,
    )
    run_backtest(strategy, cfg)
