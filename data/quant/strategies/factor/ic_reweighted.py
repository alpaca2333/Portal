"""
IC-guided factor reweighting strategy (biweekly rebalance)
==========================================================
Based on industry_neutral_concentrated_ng.py.
Weights adjusted according to IC analysis (factor_ic_analysis.py):

Changes vs original:
  mom_12_1    +0.25 → removed  (全期 IC=-0.014, ICIR=-0.10, 全期反向)
  vol_confirm +0.15 → removed  (全期 IC=-0.064, ICIR=-0.54, 方向完全相反)
  rvol_20     -0.15 → -0.35    (ICIR=+0.34, 最稳定因子, 加大)
  rev_10      +0.10 → +0.30    (ICIR=+0.37, 2025年IC=+0.080, 加大)
  inv_pb      +0.25 → +0.25    (ICIR=+0.33, 稳定, 保持)
  log_cap     -0.10 → -0.10    (ICIR=+0.35, 稳定, 保持)

All other parameters (universe, freq, cost, top_pct) unchanged.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import StrategyConfig, FactorDef, run_pipeline
from engine.types import SelectionMode, RebalanceFreq

# ─────────────────────────── Config ────────────────────────────
# 完全不动，与 industry_neutral_concentrated_ng.py 一致

config = StrategyConfig(
    name="ic_reweighted",
    description="IC引导因子重新定权：移除动量和量价，加大反转和低波动；buffer_sigma=0.3 降低换手",
    rationale="""
## 为什么做这次改动

通过 `factor_ic_analysis.py` 对 2022-2026 年所有因子的 Rank IC 进行了系统测算，发现原版策略存在两个"方向错误"的因子：

- **`mom_12_1`（12-1月动量，权重 +25%）**：全期 IC 均值 = -0.014，ICIR = -0.10，每年都是负向或接近零。
  原本假设"过去涨的股票未来继续涨"，但在 A 股双周频率下，动量完全无效，反而略微拖累。
- **`vol_confirm`（量价确认，权重 +15%）**：全期 IC 均值 = -0.064，ICIR = -0.54，全年都是反向。
  这是最强的"负向因子"——选高量价比的股票反而亏钱，方向完全搞反了。

真正有效的因子（ICIR > 0.3）：`rev_10`（+0.37）、`log_cap`（+0.35）、`rvol_20`（+0.34）、`inv_pb`（+0.33）。

## 改动内容

仅修改因子权重，其他参数（股票池、调仓频率、选股比例、成本）与 `industry_neutral_concentrated_ng` 完全一致：

| 因子 | 原权重 | 新权重 | 原因 |
|------|--------|--------|------|
| `mom_12_1` | +0.25 | 移除 | 全期IC负向 |
| `vol_confirm` | +0.15 | 移除 | ICIR=-0.54，方向相反 |
| `rvol_20` | -0.15 | **-0.35** | 最稳定因子，加大 |
| `rev_10` | +0.10 | **+0.30** | 2025年IC=+0.08，加大 |
| `inv_pb` | +0.25 | +0.25 | 稳定，保持 |
| `log_cap` | -0.10 | -0.10 | 稳定，保持 |
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
    single_side_cost=0.00015,
    buffer_sigma=0.3,
)

# ─────────────────────────── Factors ───────────────────────────
# 仅修改权重，因子列名与框架内置完全一致

factors = [
    # 移除：mom_12_1（全期IC负向），vol_confirm（ICIR=-0.54反向）
    FactorDef("inv_pb",   +0.25),   # 估值，稳定有效 ICIR=+0.33
    FactorDef("rev_10",   +0.30),   # 短期反转，最有效 ICIR=+0.37
    FactorDef("rvol_20",  -0.35),   # 低波动，最稳定 ICIR=+0.34
    FactorDef("log_cap",  -0.10),   # 小市值，稳定 ICIR=+0.35
]

# ─────────────────────────── Run ───────────────────────────────

if __name__ == "__main__":
    run_pipeline(config, factors)
