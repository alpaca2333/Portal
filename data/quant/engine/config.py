"""
Backtest configuration.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """All parameters needed by the backtest engine."""

    # ── Capital ──
    initial_capital: float = 1_000_000.0

    # ── Trading cost (one-way, applied to both buy and sell) ──
    commission_rate: float = 1.5e-4          # 1.5 bps per side

    # ── Slippage (fraction of price, applied to both buy and sell) ──
    slippage: float = 0.0

    # ── Round-lot size (A-share = 100) ──
    lot_size: int = 100

    # ── Date range ──
    start_date: str = "2010-01-01"
    end_date: str = "2026-12-31"

    # ── Rebalance frequency ──
    # "M" = month-end, "W" = week-end, "Q" = quarter-end, "D" = daily
    rebalance_freq: str = "M"

    # ── Benchmark ──
    # Automatically loads ALL .csv files under baseline_dir as benchmarks.
    # No single benchmark_code needed.

    # ── Database path ──
    db_path: str = "data/quant/data/quant.db"

    # ── Baseline directory (for benchmark index CSV) ──
    baseline_dir: str = "data/quant/baseline"

    # ── Output directory ──
    output_dir: str = "data/quant/backtest"

    # ── Strategy name (set by engine) ──
    strategy_name: str = "unnamed"

    # ── Run timestamp (set by engine) ──
    run_timestamp: str = ""
