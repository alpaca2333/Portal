"""
Quant Backtest Engine
=====================
A lightweight framework for cross-sectional factor strategy backtesting.

Designed for A-share (SH + SZ) biweekly/monthly rebalance strategies
with industry-neutral scoring, transaction cost modelling, and
dual-benchmark reporting.

Usage:
    from engine import StrategyConfig, FactorDef, run_pipeline

    config = StrategyConfig(name="my_strategy", ...)
    factors = [FactorDef("mom_12_1", 0.25), ...]
    run_pipeline(config, factors)
"""

from engine.types import StrategyConfig, FactorDef, SelectionMode
from engine.pipeline import run_pipeline

__all__ = [
    "StrategyConfig",
    "FactorDef",
    "SelectionMode",
    "run_pipeline",
]
