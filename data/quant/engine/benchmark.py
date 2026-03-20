"""
Benchmark loading: read index data from CSV and align to strategy dates.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from engine.types import StrategyConfig, BenchmarkDef


def _resolve_csv_path(cfg: StrategyConfig, csv_filename: str) -> Path:
    """
    Resolve the CSV path for a benchmark file.

    Search order:
      1. cfg.csv_data_dir (if explicitly set and exists)
      2. data/quant/baseline/  (relative to engine/)
      3. /root/qlib_data/daily (legacy server path)
    """
    candidates = []

    # 1) explicit config
    if cfg.csv_data_dir:
        candidates.append(Path(cfg.csv_data_dir) / csv_filename)

    # 2) baseline dir (relative to this file: engine/../baseline)
    baseline_dir = Path(__file__).resolve().parent.parent / "baseline"
    candidates.append(baseline_dir / csv_filename)

    # 3) legacy server path
    candidates.append(Path("/root/qlib_data/daily") / csv_filename)

    for p in candidates:
        if p.exists():
            return p

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Benchmark CSV '{csv_filename}' not found. Searched:\n  {tried}"
    )


def load_benchmark(
    bench: BenchmarkDef,
    snap_dates: list,
    cfg: StrategyConfig,
) -> pd.Series:
    """
    Load a benchmark's returns aligned to the strategy's rebalance dates.

    Parameters
    ----------
    bench : BenchmarkDef
    snap_dates : list of date strings from portfolio_df["date"]
    cfg : StrategyConfig

    Returns
    -------
    pd.Series indexed by datetime, name = descriptive label
    """
    csv_path = _resolve_csv_path(cfg, bench.csv_filename)
    # Read close and factor columns; restore real price = close * factor
    usecols = ["date", "close"]
    raw = pd.read_csv(csv_path, low_memory=False)
    has_factor = "factor" in raw.columns
    df = raw[["date", "close"] + (["factor"] if has_factor else [])].copy()
    df["date"] = pd.to_datetime(df["date"])
    if has_factor:
        df["close"] = df["close"] * df["factor"]
        df = df.drop(columns=["factor"])
    df = df.sort_values("date")
    df = df[(df["date"] >= cfg.warm_up_start) & (df["date"] <= cfg.end)]

    snap_dates_dt = pd.to_datetime(snap_dates)
    vals = []
    for dt in snap_dates_dt:
        mask = df["date"] <= dt
        if mask.any():
            vals.append(df.loc[mask, "close"].iloc[-1])
        else:
            vals.append(np.nan)

    bdf = pd.DataFrame({"date": snap_dates_dt, "close": vals})
    bdf = bdf.sort_values("date")
    bdf["ret"] = bdf["close"].pct_change()
    ret = bdf.dropna(subset=["ret"]).set_index("date")["ret"]
    ret.name = bench.name
    print(f"[基准] {bench.name}: {len(ret)} 期")
    return ret


def load_all_benchmarks(
    portfolio_df: pd.DataFrame,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """
    Load all configured benchmarks and join with portfolio returns.

    Returns combined DataFrame with columns:
        date, port_ret, port_ret_gross, n_stocks, n_industries,
        turnover_sell, turnover_buy, tc,
        bench_ret, excess, [bench2_ret, excess2, ...]
    """
    portfolio_df = portfolio_df.copy()
    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
    combined = portfolio_df.set_index("date")

    for i, bench in enumerate(cfg.benchmarks):
        ret_series = load_benchmark(
            bench, portfolio_df["date"].tolist(), cfg
        )
        if i == 0:
            ret_series.name = "bench_ret"
        else:
            ret_series.name = f"bench{i + 1}_ret"
        combined = combined.join(ret_series, how="left")

    combined = combined.reset_index()

    # Compute excess returns
    if "bench_ret" in combined.columns:
        combined["excess"] = combined["port_ret"] - combined["bench_ret"]
    if "bench2_ret" in combined.columns:
        combined["excess2"] = combined["port_ret"] - combined["bench2_ret"]

    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    return combined
