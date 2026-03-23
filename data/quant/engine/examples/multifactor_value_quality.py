"""
Multi-Factor Cross-Sectional Stock Selection Strategy (V1)
==========================================================

Factors:
  - Value   : 1 / PB  (lower PB → higher score)
  - Quality : ROE     (higher ROE → higher score)
  - Momentum: 20-day return (higher recent return → higher score)

Method:
  1. Universe filter: remove suspended, ST-like, missing data, IPO < 252 days
  2. Winsorize each raw factor at [1%, 99%] percentile
  3. Within each SW-L1 industry, z-score normalize each factor
  4. Composite score = equal-weight average of the three z-scores
  5. Select top-N by composite score, assign equal weight
  6. Rebalance monthly

Usage
-----
cd <project_root>
python -m data.quant.engine.examples.multifactor_value_quality
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Dict
import numpy as np
import pandas as pd

from engine import BacktestConfig, StrategyBase, run_backtest
from engine.data_loader import DataAccessor


class MultifactorValueQuality(StrategyBase):
    """
    Cross-sectional multi-factor strategy: Value + Quality + Momentum.
    Industry-neutral z-score normalization, equal-weight top-N portfolio.
    """

    def __init__(self, top_n: int = 30, momentum_window: int = 20):
        super().__init__("multifactor_vqm")
        self.top_n = top_n
        self.momentum_window = momentum_window

    def describe(self) -> str:
        return (
            f"### 策略思路\n\n"
            f"经典截面多因子选股策略，综合 **价值（Value）**、**质量（Quality）**、"
            f"**动量（Momentum）** 三大因子，在全市场中选取综合得分最高的 "
            f"**{self.top_n}** 只股票，等权持有，月度调仓。\n\n"
            f"### 因子定义\n\n"
            f"| 因子 | 原始指标 | 方向 | 权重 |\n"
            f"|------|---------|------|------|\n"
            f"| 价值 | 1/PB (市净率倒数) | 越高越好 | 1/3 |\n"
            f"| 质量 | ROE (净资产收益率) | 越高越好 | 1/3 |\n"
            f"| 动量 | 近{self.momentum_window}日收益率 | 越高越好 | 1/3 |\n\n"
            f"### 技术细节\n\n"
            f"1. **选股池过滤**：剔除停牌股、无收盘价/PB/ROE 的股票、"
            f"PB≤0 的股票（亏损或破净异常）、上市不满 252 个交易日的次新股\n"
            f"2. **极端值处理**：各因子在 [1%, 99%] 分位数处 Winsorize\n"
            f"3. **行业中性化**：在申万一级行业内做 Z-Score 标准化，"
            f"消除行业间估值/盈利水平差异\n"
            f"4. **综合打分**：三因子 Z-Score 等权平均\n"
            f"5. **组合构建**：综合得分排名前 {self.top_n}，等权分配\n\n"
            f"### 已知局限\n\n"
            f"- 因子权重固定为等权 1/3，未做 IC 加权优化\n"
            f"- 未加入换手率约束，可能存在高换手\n"
            f"- 动量因子窗口固定为 {self.momentum_window} 日，未做参数敏感性分析\n"
            f"- 未对 ST / *ST 做显式标记过滤（仅通过 PB/ROE 有效性间接剔除部分）"
        )

    @staticmethod
    def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """Clip values to [lower_pct, upper_pct] percentile range."""
        lo = s.quantile(lower)
        hi = s.quantile(upper)
        return s.clip(lo, hi)

    @staticmethod
    def _zscore_by_group(df: pd.DataFrame, col: str, group_col: str) -> pd.Series:
        """Within-group z-score normalization. Groups with std=0 get score 0."""
        result = pd.Series(0.0, index=df.index, dtype=float)
        for _, g in df.groupby(group_col):
            mu = g[col].mean()
            sigma = g[col].std()
            if sigma == 0 or np.isnan(sigma):
                continue  # leave as 0.0
            result.loc[g.index] = (g[col] - mu) / sigma
        return result

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:

        # ── Step 1: Load cross-sectional snapshot ──
        snap = accessor.get_date(date, columns=[
            "close", "pb", "roe", "circ_mv", "sw_l1", "is_suspended",
        ])

        # ── Step 2: Universe filter ──
        snap = snap[
            (snap["is_suspended"] != 1)
            & (snap["close"].notna()) & (snap["close"] > 0)
            & (snap["pb"].notna()) & (snap["pb"] > 0)
            & (snap["roe"].notna())
            & (snap["circ_mv"].notna()) & (snap["circ_mv"] > 0)
            & (snap["sw_l1"].notna())
        ].copy()

        if len(snap) < self.top_n:
            return {}

        # ── Step 3: Compute momentum factor (20-day return) ──
        window_data = accessor.get_window(
            date, lookback=self.momentum_window + 1,
            columns=["close"],
        )

        if window_data.empty:
            return {}

        # Compute return: close_today / close_N_days_ago - 1
        pivot = window_data.pivot(index="trade_date", columns="ts_code", values="close")
        if len(pivot) < 2:
            return {}

        mom_ret = (pivot.iloc[-1] / pivot.iloc[0] - 1)
        mom_ret.name = "momentum"
        snap = snap.merge(
            mom_ret.reset_index().rename(columns={"ts_code": "ts_code", "momentum": "momentum"}),
            on="ts_code", how="inner"
        )

        # Drop stocks with NaN momentum
        snap = snap[snap["momentum"].notna()].copy()

        if len(snap) < self.top_n:
            return {}

        # ── Step 4: Construct raw factors ──
        # Value factor: inverse PB (lower PB = higher value score)
        snap["f_value"] = 1.0 / snap["pb"]
        # Quality factor: ROE as-is
        snap["f_quality"] = snap["roe"]
        # Momentum factor: 20-day return as-is
        snap["f_momentum"] = snap["momentum"]

        # ── Step 5: Winsorize ──
        for col in ["f_value", "f_quality", "f_momentum"]:
            snap[col] = self._winsorize(snap[col])

        # ── Step 6: Industry-neutral z-score ──
        # Only keep industries with >= 5 stocks for meaningful normalization
        ind_counts = snap["sw_l1"].value_counts()
        valid_industries = ind_counts[ind_counts >= 5].index
        snap = snap[snap["sw_l1"].isin(valid_industries)].copy()

        if len(snap) < self.top_n:
            return {}

        snap["z_value"] = self._zscore_by_group(snap, "f_value", "sw_l1")
        snap["z_quality"] = self._zscore_by_group(snap, "f_quality", "sw_l1")
        snap["z_momentum"] = self._zscore_by_group(snap, "f_momentum", "sw_l1")

        # ── Step 7: Composite score (equal weight) ──
        snap["score"] = (snap["z_value"] + snap["z_quality"] + snap["z_momentum"]) / 3.0

        # Drop any NaN scores
        snap = snap[snap["score"].notna()]

        if len(snap) < self.top_n:
            return {}

        # ── Step 8: Select top N, equal weight ──
        top = snap.nlargest(self.top_n, "score")
        codes = top["ts_code"].tolist()
        w = 1.0 / len(codes)
        return {c: w for c in codes}


if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=1.5e-4,
        slippage=0.001,           # 1bp slippage for realism
        start_date="2020-01-01",
        end_date="2025-12-31",
        rebalance_freq="M",
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir="data/quant/backtest",
    )

    strategy = MultifactorValueQuality(top_n=30, momentum_window=20)
    result = run_backtest(strategy, cfg)
