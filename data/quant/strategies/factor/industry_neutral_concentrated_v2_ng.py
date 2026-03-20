"""
Industry-neutral concentrated strategy v2 (biweekly rebalance)
===============================================================
Rewritten using the engine framework.

Upgrades over v1:
1. ROE risk filter: exclude stocks with ROE_TTM < -20%
2. Rebalance buffer band: +0.3σ score bonus for incumbent holdings

Original: 626 lines → now ~70 lines of strategy-specific code.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# Add engine to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import StrategyConfig, FactorDef, run_pipeline
from engine.types import SelectionMode, RebalanceFreq

# ─────────────────────────── Hooks ─────────────────────────────

ROE_FLOOR = -20.0  # Exclude stocks with ROE_TTM < -20%


def roe_risk_filter(snap: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Pre-filter hook: remove stocks with extreme negative ROE.
    Stocks with missing ROE are kept (don't penalize data gaps).
    """
    before = len(snap)
    mask = (snap["roe_ttm"].isna()) | (snap["roe_ttm"] >= ROE_FLOOR)
    snap = snap[mask].reset_index(drop=True)
    dropped = before - len(snap)
    print(f"[roe_filter] Excluded {dropped:,} rows with ROE < {ROE_FLOOR}%")
    return snap


# ─────────────────────────── Config ────────────────────────────

config = StrategyConfig(
    name="industry_neutral_concentrated_v2_ng",
    description="Industry-neutral concentrated v2 (biweekly), "
                "ROE risk filter + 0.3σ buffer band, engine rewrite",
    warm_up_start="2018-01-01",
    backtest_start="2019-01-01",
    end="2026-02-28",
    freq=RebalanceFreq.BIWEEKLY,
    extra_columns=["roe_ttm"],       # request ROE from DB
    mcap_keep_pct=0.70,
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.05,
    max_per_industry=5,
    min_industry_count=5,
    min_holding=20,
    single_side_cost=0.00015,
    buffer_sigma=0.3,                # 0.3σ incumbent bonus
    pre_filter=roe_risk_filter,      # ROE risk hook
)

# ─────────────────────────── Factors ───────────────────────────

factors = [
    FactorDef("mom_12_1",    +0.25),
    FactorDef("inv_pb",      +0.25),
    FactorDef("vol_confirm", +0.15),
    FactorDef("rvol_20",     -0.15),
    FactorDef("log_cap",     -0.10),
    FactorDef("rev_10",      +0.10),
]

# ─────────────────────────── Report extras ─────────────────────

EXTRA_REPORT = """
## Strategy Upgrades (vs original)

### 1. ROE Risk Filter
- **Rule**: Exclude stocks with ROE_TTM < -20%
- **Purpose**: Remove extreme loss-making companies (near-bankruptcy, fraud suspects)
- **Design**: Pure risk screen, not a scoring factor (avoids ROE/PB hedge effect)

### 2. Rebalance Buffer Band
- **Rule**: Incumbent holdings get +0.3σ score bonus
- **Purpose**: Reduce marginal turnover for borderline stocks
- **Effect**: Lower turnover → lower costs, better for live trading
"""

# ─────────────────────────── Run ───────────────────────────────

if __name__ == "__main__":
    run_pipeline(config, factors, extra_report_sections=EXTRA_REPORT)
