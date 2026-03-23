"""
Example: equal-weight top-N by market cap strategy.

Usage
-----
cd data/quant
python -m engine.examples.equal_weight_demo
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Dict
import pandas as pd

from engine import BacktestConfig, StrategyBase, run_backtest
from engine.data_loader import DataAccessor


class EqualWeightTopN(StrategyBase):
    """
    Every rebalance date, pick the top N stocks by circulating market value
    (circ_mv) and assign equal weight to each.
    """

    def __init__(self, top_n: int = 30):
        super().__init__(f"equal_weight_top{top_n}")
        self.top_n = top_n

    def describe(self) -> str:
        return (
            f"### 策略思路\n\n"
            f"每月末从全市场中选取流通市值最大的 **{self.top_n}** 只股票，"
            f"按等权方式分配资金。\n\n"
            f"### 技术细节\n\n"
            f"- **选股池**：剔除当日停牌、无收盘价、无流通市值的股票\n"
            f"- **排序指标**：`circ_mv`（流通市值），降序取 Top {self.top_n}\n"
            f"- **权重分配**：等权 1/{self.top_n}\n\n"
            f"### 已知局限\n\n"
            f"- 等权配置未考虑行业集中度风险\n"
            f"- 大市值股票池变动较小，换手率低但超额来源有限\n"
            f"- 未对 ST / 风险警示股做特殊处理"
        )

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        # On-demand: only load data for this single date
        snap = accessor.get_date(date)

        # Filter: not suspended, has valid price and market cap
        snap = snap[
            (snap["is_suspended"] != 1)
            & (snap["close"].notna())
            & (snap["close"] > 0)
            & (snap["circ_mv"].notna())
            & (snap["circ_mv"] > 0)
        ]

        if snap.empty:
            return {}

        top = snap.nlargest(self.top_n, "circ_mv")
        codes = top["ts_code"].tolist()
        w = 1.0 / len(codes)
        return {c: w for c in codes}


if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=1.5e-4,
        slippage=0.0,
        start_date="2020-01-01",
        end_date="2025-12-31",
        rebalance_freq="M",
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir="data/quant/backtest",
    )

    strategy = EqualWeightTopN(top_n=30)
    result = run_backtest(strategy, cfg)
