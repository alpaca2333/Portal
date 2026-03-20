"""
Multi-factor backtest — engine framework rewrite
=================================================
Original: multifactor_backtest.py (330 lines, qlib-based)
Framework version: ~70 lines

Factors (same 3 as original):
  - mom_12_1   (+1/3)  12-1 month momentum
  - rvol_20    (-1/3)  20-day realized volatility (low vol preferred)
  - turnover   (-1/3)  turnover ratio = Mean(volume,20) / close  (low preferred)

Changes vs original:
  - Data source: SQLite (instead of qlib D.features)
  - Scoring: industry-neutral z-score (upgrade from global z-score)
  - Universe: SH+SZ with market-cap filter (top 90%)
  - Transaction cost: 1.5 bps single-side (original had none)
  - Selection: top 10%, no per-industry cap (same as original intent)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add engine to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import StrategyConfig, FactorDef, run_pipeline
from engine.types import SelectionMode, RebalanceFreq


# ─────────────── Custom factor: add turnover ───────────────────

def compute_factors_with_turnover(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Compute the original 3 factors: mom_12_1, rvol_20, turnover.
    Turnover = Mean(volume, 20) / close  (not in the default engine).
    """
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")

    # Daily return
    df["ret_1d"] = g["close"].pct_change()

    # 12-1 momentum (skip most recent month)
    df["close_lag20"] = g["close"].shift(20)
    df["close_lag250"] = g["close"].shift(250)
    df["mom_12_1"] = df["close_lag20"] / df["close_lag250"] - 1

    # 20-day realized volatility
    df["rvol_20"] = g["ret_1d"].transform(
        lambda x: x.rolling(20, min_periods=15).std())

    # Turnover: Mean(volume, 20) / close  — the original factor
    df["vol_ma20"] = g["volume"].transform(
        lambda x: x.rolling(20, min_periods=15).mean())
    df["turnover"] = df["vol_ma20"] / df["close"].replace(0, np.nan)

    # Still need these for universe filtering
    df["inv_pb"] = 1.0 / df["pb"].replace(0, np.nan)
    df.loc[df["pb"] < 0, "inv_pb"] = np.nan
    df["log_cap"] = np.log(df["free_market_cap"].replace(0, np.nan))

    print(f"[factors] Custom 3-factor (mom + ivol + turnover). Shape: {df.shape}")
    return df


# ─────────────────────────── Config ────────────────────────────

config = StrategyConfig(
    name="multifactor_ng",
    description="Multi-factor strategy (mom + low vol + low turnover), "
                "monthly rebalance, industry-neutral, engine rewrite",
    warm_up_start="2018-01-01",
    backtest_start="2019-01-01",
    end="2026-02-28",
    freq=RebalanceFreq.MONTHLY,
    # Universe: broader than concentrated — keep top 90% by market cap
    mcap_keep_pct=0.90,
    # Selection: top 10% (same as original), no per-industry cap
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.10,
    max_per_industry=0,         # no industry cap (same as original)
    min_industry_count=5,
    min_holding=20,
    # Transaction cost (original had none; adding for realism)
    single_side_cost=0.00015,   # 1.5 bps single-side
    # Custom factor computation
    compute_factors_fn=compute_factors_with_turnover,
)

# ─────────────────────────── Factors ───────────────────────────
# Equal weight 1/3 each, matching the original: (mom - ivol - turn) / 3
W = 1.0 / 3.0

factors = [
    FactorDef("mom_12_1",  +W),    # momentum: higher is better
    FactorDef("rvol_20",   -W),    # low volatility: lower is better
    FactorDef("turnover",  -W),    # low turnover: lower is better
]

# ─────────────────────────── Run ───────────────────────────────

if __name__ == "__main__":
    run_pipeline(config, factors)
