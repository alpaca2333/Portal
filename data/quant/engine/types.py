"""
Type definitions for the backtest engine.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, List


class SelectionMode(Enum):
    """How to select stocks from scored universe."""
    TOP_PCT = "top_pct"      # top X% by composite score
    TOP_N = "top_n"          # top N stocks globally


class RebalanceFreq(Enum):
    """Rebalance frequency."""
    MONTHLY = "monthly"
    BIWEEKLY = "biweekly"


@dataclass
class FactorDef:
    """
    A single factor definition.

    Parameters
    ----------
    column : str
        Column name in the DataFrame after compute_factors() runs.
        E.g. "mom_12_1", "inv_pb", "rvol_20".
    weight : float
        Weight in the composite score. Positive = higher is better,
        negative = lower is better (e.g. -0.15 for volatility).
    """
    column: str
    weight: float


@dataclass
class BenchmarkDef:
    """A benchmark definition."""
    code: str         # e.g. "sh000001"
    name: str         # e.g. "Shanghai Composite (sh000001)"
    csv_filename: str  # e.g. "000001.SH.csv"


@dataclass
class StrategyConfig:
    """
    All configuration for a backtest run.

    Strategies only need to set the fields relevant to them;
    everything has sensible defaults matching the 'concentrated' strategy.
    """
    # --- Identity ---
    name: str = "unnamed_strategy"
    description: str = ""
    rationale: str = ""     # 改动原因 / 策略思路，会写入 Markdown 报告

    # --- Time range ---
    warm_up_start: str = "2021-01-01"
    backtest_start: str = "2022-01-01"
    end: str = "2026-02-28"

    # --- Rebalance ---
    freq: RebalanceFreq = RebalanceFreq.BIWEEKLY

    # --- Universe filtering ---
    db_path: str = "/projects/portal/data/quant/processed/stocks.db"
    extra_columns: List[str] = field(default_factory=list)
    mcap_keep_pct: float = 0.70         # keep top X% by free_market_cap

    # --- Selection ---
    selection_mode: SelectionMode = SelectionMode.TOP_PCT
    top_pct: float = 0.05               # for TOP_PCT mode
    top_n: int = 100                     # for TOP_N mode
    max_per_industry: int = 5            # 0 = no cap
    min_industry_count: int = 5          # min stocks to score an industry
    min_holding: int = 20                # min portfolio size to accept a period

    # --- Transaction cost ---
    single_side_cost: float = 0.00015    # 1.5 bps single side

    # --- Buffer band (0 = disabled) ---
    buffer_sigma: float = 0.0            # score bonus for incumbent holdings

    # --- Benchmarks ---
    benchmarks: List[BenchmarkDef] = field(default_factory=lambda: [
        BenchmarkDef("sh000001", "Shanghai Composite (sh000001)", "000001.SH.csv"),
        BenchmarkDef("sz000905", "CSI 500 (sz000905)", "000905.SZ.csv"),
    ])
    csv_data_dir: str = "/root/qlib_data/daily"

    # --- Output ---
    output_dir: str = "/projects/portal/data/quant/backtest"

    # --- Hooks (strategy-specific logic) ---
    # pre_filter: called after standard universe filters, before scoring
    #   signature: (snap: DataFrame, config: StrategyConfig) -> DataFrame
    pre_filter: Optional[Callable] = field(default=None, repr=False)

    # post_score: called after scoring, returns selected DataFrame
    #   signature: (signal: DataFrame, prev_holdings: set, config: StrategyConfig) -> DataFrame
    post_select: Optional[Callable] = field(default=None, repr=False)

    # compute_factors: custom factor computation (replaces default)
    #   signature: (df: DataFrame, config: StrategyConfig) -> DataFrame
    compute_factors_fn: Optional[Callable] = field(default=None, repr=False)
