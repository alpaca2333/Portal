"""
IC-guided strategy with buffer band sweep
==========================================
Tests buffer_sigma = 0.3 / 0.5 / 0.8 to find optimal turnover reduction.

Buffer logic: incumbent holdings get +buffer_sigma bonus on composite score,
making them harder to replace unless a new stock is clearly better.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import StrategyConfig, FactorDef, run_pipeline
from engine.types import SelectionMode, RebalanceFreq

BASE_RATIONALE = """
## 策略背景

在 `ic_reweighted` 基础上加入换仓缓冲带（buffer band）。

**问题**：`ic_reweighted` 平均单边换手率 57.8%，每双周约换掉一半持仓，
边界附近的股票频繁进出，产生不必要的交易成本（累计拖累 ~3%）。

**方案**：给当前持仓股的综合得分加一个 bonus（+buffer_sigma σ），
让它们在得分略低于新候选股时仍然保留，只有明显落后的才被替换。

**参数 buffer_sigma = {sigma}**：相当于持仓股享有 {sigma}σ 的"留任优势"，
理论上能把换手率从 ~58% 降至 ~40% 以下，同时减少摩擦成本。
"""

FACTORS = [
    FactorDef("inv_pb",   +0.25),
    FactorDef("rev_10",   +0.30),
    FactorDef("rvol_20",  -0.35),
    FactorDef("log_cap",  -0.10),
]

BASE_CONFIG = dict(
    warm_up_start="2018-01-01",
    backtest_start="2019-01-01",
    end="2026-02-28",
    freq=RebalanceFreq.BIWEEKLY,
    mcap_keep_pct=0.70,
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.05,
    max_per_industry=5,
    min_industry_count=5,
    min_holding=20,
    single_side_cost=0.00015,
)

if __name__ == "__main__":
    for sigma in [0.3, 0.5, 0.8]:
        print(f"\n{'='*64}")
        print(f"Buffer sigma = {sigma}")
        print('='*64)
        cfg = StrategyConfig(
            name=f"ic_reweighted_buf{int(sigma*10)}",
            description=f"IC重新定权 + 换仓缓冲带 buffer_sigma={sigma}",
            rationale=BASE_RATIONALE.format(sigma=sigma),
            buffer_sigma=sigma,
            **BASE_CONFIG,
        )
        run_pipeline(cfg, FACTORS)
