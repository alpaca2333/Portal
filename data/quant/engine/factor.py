"""
Factor utilities: z-scoring and industry-neutral composite scoring.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from engine.types import FactorDef, StrategyConfig


def winsorized_zscore(s: pd.Series) -> pd.Series:
    """
    Winsorized z-score: clip at 2nd/98th percentile, then standardize.
    Returns NaN series if fewer than 5 valid values.
    """
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=s.index)
    lo, hi = valid.quantile(0.02), valid.quantile(0.98)
    c = s.clip(lo, hi)
    std = c.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (c - c.mean()) / std


def score_within_industry(
    snap: pd.DataFrame,
    factors: List[FactorDef],
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """
    Industry-neutral scoring: compute weighted z-score composite
    within each (period, industry_code) group.

    Uses explicit iteration instead of groupby().apply() for
    Pandas 3.0 compatibility.
    """
    print("[score] Industry-neutral scoring ...")

    def score_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < cfg.min_industry_count:
            g["score"] = np.nan
            return g
        g = g.copy()
        composite = pd.Series(0.0, index=g.index)
        for f in factors:
            if f.column in g.columns:
                composite += f.weight * winsorized_zscore(g[f.column])
        g["score"] = composite
        return g

    parts = []
    for (_period, _ind), g in snap.groupby(["period", "industry_code"]):
        parts.append(score_group(g))

    scored = pd.concat(parts, ignore_index=True)
    n_valid = scored["score"].notna().sum()
    print(f"[score] Valid scored: {n_valid:,}")
    return scored
