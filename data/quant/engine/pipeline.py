"""
Pipeline: the top-level orchestrator that wires everything together.

    from engine import StrategyConfig, FactorDef, run_pipeline
    run_pipeline(config, factors)
"""
from __future__ import annotations
from typing import List
import pandas as pd
from engine.types import StrategyConfig, FactorDef
from engine.data import load_stock_data, compute_daily_factors, sample, filter_universe
from engine.factor import score_within_industry
from engine.backtest import run_backtest
from engine.benchmark import load_all_benchmarks
from engine.report import (
    compute_periods_per_year, print_summary, save_outputs, write_report,
)


def run_pipeline(
    cfg: StrategyConfig,
    factors: List[FactorDef],
    extra_report_sections: str = "",
) -> pd.DataFrame:
    """
    Full backtest pipeline: data → factors → sample → filter → score →
    backtest → benchmark → report.

    Parameters
    ----------
    cfg : StrategyConfig
    factors : list of FactorDef
    extra_report_sections : optional Markdown string appended to the report

    Returns
    -------
    combined : DataFrame with full results (portfolio + benchmark returns)
    """
    print("=" * 64)
    print(f"Strategy: {cfg.name}")
    if cfg.description:
        print(f"  {cfg.description}")
    print(f"Period  : {cfg.backtest_start} ~ {cfg.end}")
    print(f"Freq    : {cfg.freq.value}")
    factor_str = ", ".join(f"{f.column}={f.weight:+.2f}" for f in factors)
    print(f"Factors : {factor_str}")
    print("=" * 64)

    # 1. Load data
    print("\n[1] Loading stock data ...")
    daily = load_stock_data(cfg)

    # 2. Compute factors
    print("\n[2] Computing factors ...")
    daily = compute_daily_factors(daily, cfg)

    # 3. Sample (biweekly / monthly)
    print(f"\n[3] Sampling ({cfg.freq.value}) ...")
    snap = sample(daily, cfg)
    del daily  # free memory

    # 4. Filter universe
    print("\n[4] Filtering universe ...")
    snap = filter_universe(snap, cfg)

    # 5. Score within industry
    print("\n[5] Industry-neutral scoring ...")
    snap = score_within_industry(snap, factors, cfg)

    # 6. Run backtest
    print("\n[6] Running backtest ...")
    portfolio_df = run_backtest(snap, cfg)
    if portfolio_df.empty:
        raise RuntimeError("Backtest produced no valid observations.")

    # Compute periods per year
    ppy = compute_periods_per_year(portfolio_df)
    print(f"[info] {len(portfolio_df)} periods over "
          f"{len(portfolio_df)/ppy:.1f} years = {ppy:.1f} periods/year")

    # 7. Load benchmarks
    print("\n[7] Loading benchmarks ...")
    combined = load_all_benchmarks(portfolio_df, cfg)

    # 8. Output
    print("\n[8] Generating output ...")
    print_summary(combined.copy(), cfg, factors, ppy)
    save_outputs(combined, cfg)
    write_report(combined.copy(), cfg, factors, ppy, extra_report_sections)

    print(f"\n✅ {cfg.name} backtest complete.")
    return combined
