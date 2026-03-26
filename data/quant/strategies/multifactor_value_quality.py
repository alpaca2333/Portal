"""
Multi-Factor Cross-Sectional Stock Selection Strategy (V2)
==========================================================

Uses the generic ``FactorEngine`` from ``engine.factors`` to compute:
  - Value   : 1/PB  (Book-to-Price)
  - Quality : ROE
  - Momentum: 20-day return

Method:
  1. FactorEngine handles: universe filter → winsorize → industry z-score
  2. Composite score = weighted average of the three z-scores
  3. Select top-N by composite score, assign equal weight
  4. Rebalance monthly

Usage
-----
cd <project_root>
python -m data.quant.strategies.multifactor_value_quality
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict
import pandas as pd

from engine import (
    BacktestConfig, StrategyBase, run_backtest,
    FactorEngine, ValueBP, QualityROE, Momentum,
)
from engine.data_loader import DataAccessor


class MultifactorValueQuality(StrategyBase):
    """
    Cross-sectional multi-factor strategy: Value + Quality + Momentum.
    Industry-neutral z-score normalization, equal-weight top-N portfolio.

    Now powered by the generic FactorEngine.
    """

    def __init__(self, top_n: int = 30, momentum_window: int = 20):
        super().__init__("multifactor_vqm")
        self.top_n = top_n
        self.momentum_window = momentum_window

        # Build factor engine with 3 equal-weight factors
        self.factor_engine = FactorEngine(
            factors=[
                ValueBP(weight=1.0),
                QualityROE(weight=1.0),
                Momentum(lookback=momentum_window, weight=1.0),
            ],
            winsorize_pct=(0.01, 0.99),
            neutralize_by="sw_l1",
            min_industry_size=5,
        )

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
            f"1. **选股池过滤**：剔除停牌股、无收盘价的股票（FactorEngine 自动处理）\n"
            f"2. **极端值处理**：各因子在 [1%, 99%] 分位数处 Winsorize\n"
            f"3. **行业中性化**：在申万一级行业内做 Z-Score 标准化\n"
            f"4. **综合打分**：三因子 Z-Score 等权平均\n"
            f"5. **组合构建**：综合得分排名前 {self.top_n}，等权分配\n\n"
            f"### 已知局限\n\n"
            f"- 因子权重固定为等权 1/3，未做 IC 加权优化\n"
            f"- 未加入换手率约束，可能存在高换手\n"
            f"- 动量因子窗口固定为 {self.momentum_window} 日，未做参数敏感性分析\n"
            f"- 未对 ST / *ST 做显式标记过滤（仅通过有效性间接剔除部分）"
        )

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:

        # One-line factor computation: winsorize + z-score + composite
        scores = self.factor_engine.run(date, accessor)

        if scores.empty or len(scores) < self.top_n:
            return {}

        # Select top N, equal weight
        top = scores.head(self.top_n)
        codes = top["ts_code"].tolist()
        w = 1.0 / len(codes)
        return {c: w for c in codes}


if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=1.5e-4,
        slippage=0.001,
        start_date="2020-01-01",
        end_date="2025-12-31",
        rebalance_freq="M",
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir="data/quant/backtest",
    )

    strategy = MultifactorValueQuality(top_n=30, momentum_window=20)
    result = run_backtest(strategy, cfg)
