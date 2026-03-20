"""
Industry-neutral concentrated strategy (biweekly rebalance)
============================================================
Rewritten using the engine framework.

Original: 573 lines → now ~40 lines of strategy-specific code.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add engine to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import StrategyConfig, FactorDef, run_pipeline
from engine.types import SelectionMode, RebalanceFreq

# ─────────────────────────── Config ────────────────────────────

config = StrategyConfig(
    name="industry_neutral_concentrated_ng",
    description="Industry-neutral concentrated strategy (biweekly), "
                "top 5% + max 5/industry, engine rewrite",
    rationale="""
## 策略思路

本策略是行业中性多因子选股策略的集中化改进版，核心逻辑：

1. **行业中性化**：因子在申万一级行业内做 winsorized z-score，消除行业轮动带来的风格偏差
2. **集中选股**：选 top 5%（约90只）而非 top 20%（约500只），alpha 信号更集中、稀释更少
3. **每行业上限**：每个行业最多 5 只，防止单行业过度集中
4. **双周调仓**：每月15日和月末各调仓一次，信号更及时，动量衰减更少

## 因子组合设计

6个因子覆盖动量、价值、量价、波动率、规模、反转六个维度，权重参考学术文献初始设定。

**注意**：本版为原始权重版本，后续经 IC 分析发现 `mom_12_1` 和 `vol_confirm` 在 A 股双周频率下实际无效，
参见 `ic_reweighted.py` 的改进版本。
""",
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
    single_side_cost=0.00015,  # 1.5 bps
    buffer_sigma=0.0,          # no buffer in original
)

# ─────────────────────────── Factors ───────────────────────────

factors = [
    FactorDef("mom_12_1",    +0.25),   # 12-1 momentum
    FactorDef("inv_pb",      +0.25),   # value (1/PB)
    FactorDef("vol_confirm", +0.15),   # volume confirmation
    FactorDef("rvol_20",     -0.15),   # low volatility
    FactorDef("log_cap",     -0.10),   # small size
    FactorDef("rev_10",      +0.10),   # short-term reversal
]

# ─────────────────────────── Run ───────────────────────────────

if __name__ == "__main__":
    run_pipeline(config, factors)
