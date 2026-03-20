"""
Reporting: terminal summary + Markdown report generation.

Follows the OutputFormat.md specification for file naming and content.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from engine.types import StrategyConfig, FactorDef


# ─────────────────────────── metrics ───────────────────────────

def calc_return_metrics(rets: pd.Series, freq: float = None) -> dict:
    """
    Compute standard return metrics.

    Parameters
    ----------
    rets : Series of periodic returns (decimal)
    freq : periods per year. If None, auto-detect from length (assume ~24 for biweekly).
    """
    r = rets.dropna()
    nans = {k: float("nan") for k in
            ["ann_ret", "ann_vol", "sharpe", "max_dd", "cum_ret", "win_rate"]}
    if r.empty:
        return nans

    nav = (1 + r).cumprod()
    if freq is None:
        n_years = len(r) / 24.0
    else:
        n_years = len(r) / freq

    if n_years <= 0:
        return nans

    ann_ret = nav.iloc[-1] ** (1.0 / n_years) - 1
    periods_per_year = len(r) / n_years
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    dd = nav / nav.cummax() - 1

    return dict(
        ann_ret=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_dd=dd.min(),
        cum_ret=nav.iloc[-1] - 1,
        win_rate=(r > 0).mean(),
    )


def compute_periods_per_year(df: pd.DataFrame) -> float:
    """Compute actual periods per year from the date column."""
    n = len(df)
    date_range_years = (
        pd.to_datetime(df["date"].iloc[-1])
        - pd.to_datetime(df["date"].iloc[0])
    ).days / 365.25
    return n / date_range_years if date_range_years > 0 else 24


# ─────────────────────────── terminal ──────────────────────────

def print_summary(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    factors: List[FactorDef],
    ppy: float,
) -> None:
    """Print formatted performance summary to stdout."""
    port = calc_return_metrics(df["port_ret"], freq=ppy)
    bench_names = []
    bench_metrics = []
    excess_metrics = []
    ir_vals = []

    # Up to 2 benchmarks
    for i, bench in enumerate(cfg.benchmarks[:2]):
        col = "bench_ret" if i == 0 else f"bench{i+1}_ret"
        ex_col = "excess" if i == 0 else f"excess{i+1}"
        if col in df.columns:
            bm = calc_return_metrics(df[col], freq=ppy)
            em = calc_return_metrics(df[ex_col], freq=ppy) if ex_col in df.columns else None
            ir = (em["ann_ret"] / em["ann_vol"]) if em and em["ann_vol"] > 0 else float("nan")
            bench_names.append(bench.name)
            bench_metrics.append(bm)
            excess_metrics.append(em)
            ir_vals.append(ir)

    print("=" * 64)
    print(f"Strategy : {cfg.name}")
    print(f"Period   : {cfg.backtest_start} ~ {cfg.end}")
    print(f"Frequency: {cfg.freq.value}")
    print(f"Selection: top {cfg.top_pct:.0%}, max {cfg.max_per_industry}/industry")
    if cfg.buffer_sigma > 0:
        print(f"Buffer   : +{cfg.buffer_sigma}σ for incumbents")
    factor_str = ", ".join(f"{f.column}={f.weight:+.2f}" for f in factors)
    print(f"Factors  : {factor_str}")
    bench_str = " / ".join(b.name for b in cfg.benchmarks[:2])
    print(f"Benchmark: {bench_str}")
    print("=" * 64)

    # Header
    headers = ["Strategy"] + [b.name.split("(")[0].strip() for b in cfg.benchmarks[:2]]
    hdr = f"{'Metric':<20}" + "".join(f"{h:>14}" for h in headers)
    print(hdr)
    print("-" * len(hdr))

    # Rows
    def _fmt(m, key, fmt_str="+.1%"):
        v = m.get(key, float("nan"))
        if np.isnan(v):
            return "—"
        return f"{v:{fmt_str}}"

    for label, key, fmt in [
        ("Ann. Return", "ann_ret", "+.1%"),
        ("Ann. Volatility", "ann_vol", ".1%"),
        ("Sharpe Ratio", "sharpe", ".2f"),
        ("Max Drawdown", "max_dd", ".1%"),
    ]:
        row = f"{label:<20}{_fmt(port, key, fmt):>14}"
        for bm in bench_metrics:
            row += f"{_fmt(bm, key, fmt):>14}"
        print(row)

    print(f"{'Win Rate':<20}{port['win_rate']:>14.1%}" + "".join(f"{'—':>14}" for _ in bench_metrics))
    print(f"{'Cum. Return':<20}{port['cum_ret']:>+14.1%}" + "".join(f"{bm['cum_ret']:>+14.1%}" for bm in bench_metrics))

    for i, (em, ir) in enumerate(zip(excess_metrics, ir_vals)):
        bname = bench_names[i].split("(")[0].strip()
        if em:
            print(f"{'Excess vs ' + bname:<20}{em['ann_ret']:>+14.1%}")
            print(f"{'IR vs ' + bname:<20}{ir:>14.2f}")

    print("=" * 64)

    # Turnover stats
    if "turnover_sell" in df.columns:
        avg_to = (df["turnover_sell"] + df["turnover_buy"]).mean() / 2
        total_tc = df["tc"].sum()
        print(f"\nTurnover : avg single-side {avg_to:.1%}")
        print(f"Cost drag: {total_tc:.2%} cumulative")

    # Year-by-year
    print("\nYear-by-year:")
    tmp = df.copy()
    tmp["_date"] = pd.to_datetime(tmp["date"])
    for year in sorted(tmp["_date"].dt.year.unique()):
        ys = tmp[tmp["_date"].dt.year == year]
        pr = (1 + ys["port_ret"].dropna()).prod() - 1
        parts = [f"  {year}: Strategy {pr:+.1%}"]
        for i, bench in enumerate(cfg.benchmarks[:2]):
            col = "bench_ret" if i == 0 else f"bench{i+1}_ret"
            if col in ys.columns:
                br = (1 + ys[col].dropna()).prod() - 1
                bname = bench.name.split("(")[0].strip()
                parts.append(f"{bname} {br:+.1%}")
                parts.append(f"Excess {pr - br:+.1%}")
        if "n_stocks" in ys.columns:
            n = ys["n_stocks"].mean()
            ni = ys["n_industries"].mean() if "n_industries" in ys.columns else 0
            parts.append(f"({n:.0f} stocks/{ni:.0f} ind)")
        print("  ".join(parts))


# ─────────────────────────── file output ───────────────────────

def save_outputs(
    combined: pd.DataFrame,
    cfg: StrategyConfig,
) -> None:
    """Save NAV and returns CSV files per OutputFormat.md spec."""
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    nav_df = pd.DataFrame({"date": combined["date"]})
    nav_df["strategy"] = (1 + combined["port_ret"]).cumprod()
    if "port_ret_gross" in combined.columns:
        nav_df["strategy_gross"] = (1 + combined["port_ret_gross"]).cumprod()
    for i, bench in enumerate(cfg.benchmarks[:2]):
        col = "bench_ret" if i == 0 else f"bench{i+1}_ret"
        nav_col = "benchmark" if i == 0 else f"benchmark{i+1}"
        if col in combined.columns:
            nav_df[nav_col] = (1 + combined[col].fillna(0)).cumprod()

    # Returns file
    ret_cols = ["date", "port_ret"]
    if "port_ret_gross" in combined.columns:
        ret_cols.append("port_ret_gross")
    ret_cols.extend(["n_stocks"])
    if "n_industries" in combined.columns:
        ret_cols.append("n_industries")
    if "turnover_sell" in combined.columns:
        ret_cols.extend(["turnover_sell", "turnover_buy", "tc"])
    if "bench_ret" in combined.columns:
        ret_cols.extend(["bench_ret", "excess"])
    if "bench2_ret" in combined.columns:
        ret_cols.extend(["bench2_ret", "excess2"])

    returns_df = combined[[c for c in ret_cols if c in combined.columns]].copy()
    if "n_stocks" in returns_df.columns:
        returns_df["n_stocks"] = returns_df["n_stocks"].astype(int)

    nav_path = outdir / f"{cfg.name}_nav.csv"
    ret_path = outdir / f"{cfg.name}_monthly_returns.csv"
    nav_df.to_csv(nav_path, index=False, float_format="%.7f")
    returns_df.to_csv(ret_path, index=False, float_format="%.8f")
    print(f"[save] {nav_path.name}")
    print(f"[save] {ret_path.name}")


# ─────────────────────────── MD report ─────────────────────────

def write_report(
    combined: pd.DataFrame,
    cfg: StrategyConfig,
    factors: List[FactorDef],
    ppy: float,
    extra_sections: str = "",
) -> None:
    """Generate a Markdown report file."""
    outdir = Path(cfg.output_dir)
    path = outdir / f"{cfg.name}_report.md"

    port = calc_return_metrics(combined["port_ret"], freq=ppy)
    gross = calc_return_metrics(
        combined.get("port_ret_gross", pd.Series(dtype=float)), freq=ppy
    )

    bench_data = []
    for i, bench in enumerate(cfg.benchmarks[:2]):
        col = "bench_ret" if i == 0 else f"bench{i+1}_ret"
        ex_col = "excess" if i == 0 else f"excess{i+1}"
        if col in combined.columns:
            bm = calc_return_metrics(combined[col], freq=ppy)
            em = calc_return_metrics(combined[ex_col], freq=ppy) if ex_col in combined.columns else None
            ir = (em["ann_ret"] / em["ann_vol"]) if em and em["ann_vol"] > 0 else float("nan")
            bench_data.append((bench, bm, em, ir))

    avg_n = combined["n_stocks"].mean() if "n_stocks" in combined.columns else 0
    avg_ind = combined["n_industries"].mean() if "n_industries" in combined.columns else 0
    avg_to = (
        (combined["turnover_sell"] + combined["turnover_buy"]).mean() / 2
        if "turnover_sell" in combined.columns else 0
    )
    total_tc = combined["tc"].sum() if "tc" in combined.columns else 0

    with open(path, "w", encoding="utf-8") as f:
        # Title
        f.write(f"# {cfg.name} Backtest Report\n\n")
        f.write(f"**Description**: {cfg.description}  \n")
        if cfg.rationale:
            f.write("\n## 策略背景与改动原因\n\n")
            f.write(cfg.rationale.strip())
            f.write("\n\n")
        f.write(f"**Period**: {cfg.backtest_start} ~ {cfg.end}  \n")
        f.write(f"**Frequency**: {cfg.freq.value}  \n")
        bench_str = " / ".join(b.name for b in cfg.benchmarks[:2])
        f.write(f"**Benchmarks**: {bench_str}  \n")
        f.write(f"**Universe**: SH+SZ, free_market_cap top {cfg.mcap_keep_pct:.0%}  \n")
        f.write(f"**Selection**: top {cfg.top_pct:.0%}, max {cfg.max_per_industry}/industry, equal weight  \n")
        f.write(f"**Avg holdings**: {avg_n:.0f} stocks / {avg_ind:.0f} industries  \n")
        if cfg.buffer_sigma > 0:
            f.write(f"**Buffer band**: +{cfg.buffer_sigma}σ for incumbents  \n")
        f.write(f"**Cost**: {cfg.single_side_cost:.4%} single-side  \n\n")

        # Factors
        f.write("## Factors\n\n")
        f.write("| Factor | Weight | Direction |\n")
        f.write("|--------|--------|-----------|\n")
        for fac in factors:
            direction = "positive" if fac.weight > 0 else "negative"
            f.write(f"| {fac.column} | {fac.weight:+.2f} | {direction} |\n")
        f.write("\n")

        # Performance summary
        f.write("## Performance Summary\n\n")
        f.write("| Metric | Strategy |")
        for bd, _, _, _ in bench_data:
            f.write(f" {bd.name.split('(')[0].strip()} |")
        f.write("\n|--------|----------|")
        for _ in bench_data:
            f.write("----------|")
        f.write("\n")

        rows = [
            ("Ann. Return", "ann_ret", "+.2%"),
            ("Ann. Volatility", "ann_vol", ".2%"),
            ("Sharpe Ratio", "sharpe", ".2f"),
            ("Max Drawdown", "max_dd", ".2%"),
            ("Cum. Return", "cum_ret", "+.2%"),
        ]
        for label, key, fmt in rows:
            f.write(f"| {label} | {port[key]:{fmt}} |")
            for _, bm, _, _ in bench_data:
                f.write(f" {bm[key]:{fmt}} |")
            f.write("\n")
        f.write(f"| Win Rate | {port['win_rate']:.2%} |")
        for _ in bench_data:
            f.write(" — |")
        f.write("\n")

        for bd, _, em, ir in bench_data:
            bname = bd.name.split("(")[0].strip()
            if em:
                f.write(f"| Excess vs {bname} | {em['ann_ret']:+.2%} | — | — |\n")
                f.write(f"| IR vs {bname} | {ir:.2f} | — | — |\n")
        f.write("\n")

        # Year-by-year
        f.write("## Year-by-Year\n\n")
        hdr = "| Year | Strategy |"
        for bd, _, _, _ in bench_data:
            bname = bd.name.split("(")[0].strip()
            hdr += f" {bname} | Excess |"
        hdr += " Holdings |\n"
        f.write(hdr)
        sep = "|------|----------|"
        for _ in bench_data:
            sep += "----------|--------|"
        sep += "----------|\n"
        f.write(sep)

        tmp = combined.copy()
        tmp["_date"] = pd.to_datetime(tmp["date"])
        for year in sorted(tmp["_date"].dt.year.unique()):
            ys = tmp[tmp["_date"].dt.year == year]
            pr = (1 + ys["port_ret"].dropna()).prod() - 1
            row = f"| {year} | {pr:+.1%} |"
            for i, (bd, _, _, _) in enumerate(bench_data):
                col = "bench_ret" if i == 0 else f"bench{i+1}_ret"
                br = (1 + ys[col].dropna()).prod() - 1 if col in ys.columns else 0
                row += f" {br:+.1%} | {pr - br:+.1%} |"
            n = ys["n_stocks"].mean() if "n_stocks" in ys.columns else 0
            ni = ys["n_industries"].mean() if "n_industries" in ys.columns else 0
            row += f" {n:.0f}/{ni:.0f} |\n"
            f.write(row)
        f.write("\n")

        # Transaction cost
        f.write("## Transaction Cost\n\n")
        f.write(f"- **Single-side cost**: {cfg.single_side_cost:.4%}\n")
        f.write(f"- **Avg single-side turnover**: {avg_to:.1%}\n")
        f.write(f"- **Total cost drag**: {total_tc:.2%}\n\n")
        if not np.isnan(gross.get("ann_ret", float("nan"))):
            f.write("| Metric | Gross | Net | Diff |\n")
            f.write("|--------|-------|-----|------|\n")
            f.write(f"| Ann. Return | {gross['ann_ret']:+.2%} | {port['ann_ret']:+.2%} | {gross['ann_ret']-port['ann_ret']:.2%} |\n")
            f.write(f"| Sharpe | {gross['sharpe']:.2f} | {port['sharpe']:.2f} | {gross['sharpe']-port['sharpe']:.2f} |\n")
            f.write(f"| Cum. Return | {gross['cum_ret']:+.2%} | {port['cum_ret']:+.2%} | {gross['cum_ret']-port['cum_ret']:.2%} |\n")
        f.write("\n")

        # Extra sections (strategy-specific notes)
        if extra_sections:
            f.write(extra_sections)
            f.write("\n")

    print(f"[report] Saved: {path}")
