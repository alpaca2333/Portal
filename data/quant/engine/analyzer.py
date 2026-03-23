"""
Performance analyzer — NAV computation, metrics, and output formatting.
Conforms to OUTPUT.md specification.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig


# ---------------------------------------------------------------------------
# NAV & returns construction
# ---------------------------------------------------------------------------

def build_nav_df(snapshots: List[dict],
                 benchmarks: Dict[str, pd.DataFrame],
                 rebal_dates: pd.DatetimeIndex,
                 cfg: BacktestConfig) -> pd.DataFrame:
    """
    Build cumulative NAV DataFrame.

    Columns: date, strategy, bench_{name1}, bench_{name2}, ...
    """
    if not snapshots:
        return pd.DataFrame(columns=["date", "strategy"])

    rows = []
    init_cap = cfg.initial_capital
    for snap in snapshots:
        rows.append({
            "date": snap["date"],
            "strategy": snap["total_value"] / init_cap,
        })
    nav = pd.DataFrame(rows)

    # Benchmark NAV — one column per benchmark
    nav_dates = set(nav["date"].tolist())
    for bname, bench_df in benchmarks.items():
        if bench_df is None or bench_df.empty:
            continue
        bdf = bench_df.copy().sort_values("date")
        col = f"bench_{bname}"
        bench_nav_map = {}
        first_close = None
        for _, row in bdf.iterrows():
            ds = row["date"].strftime("%Y%m%d")
            if first_close is None and ds in nav_dates:
                first_close = row["close"]
            if first_close is not None:
                bench_nav_map[ds] = row["close"] / first_close
        nav[col] = nav["date"].map(bench_nav_map)

    # Format date
    nav["date"] = nav["date"].apply(
        lambda x: pd.Timestamp(x).strftime("%Y-%m-%d") if not isinstance(x, str)
        else pd.Timestamp(x).strftime("%Y-%m-%d")
    )

    return nav


def _bench_columns(nav_df: pd.DataFrame) -> List[str]:
    """Return list of bench_* columns present in nav_df."""
    return [c for c in nav_df.columns if c.startswith("bench_")]


def _bench_name(col: str) -> str:
    """Extract human-readable name from column name, e.g. 'bench_000905.SZ' -> '000905.SZ'."""
    return col[len("bench_"):]


def build_returns_df(nav_df: pd.DataFrame,
                     snapshots: List[dict],
                     benchmarks: Dict[str, pd.DataFrame],
                     cfg: BacktestConfig) -> pd.DataFrame:
    """
    Build period returns DataFrame.

    Columns: date, port_ret, n_stocks,
             bench_ret_{name1}, excess_{name1}, bench_ret_{name2}, excess_{name2}, ...
    """
    if len(nav_df) < 2:
        return pd.DataFrame(columns=["date", "port_ret", "n_stocks"])

    nav = nav_df.copy()
    nav["port_ret"] = nav["strategy"].pct_change()

    # n_stocks from snapshots
    n_map = {snap["date"]: snap["n_stocks"] for snap in snapshots}
    nav["n_stocks"] = nav["date"].apply(
        lambda d: n_map.get(pd.Timestamp(d).strftime("%Y%m%d"), 0)
    )

    # Per-benchmark return & excess columns
    bench_cols = _bench_columns(nav)
    for bcol in bench_cols:
        bname = _bench_name(bcol)
        ret_col = f"bench_ret_{bname}"
        exc_col = f"excess_{bname}"
        nav[ret_col] = nav[bcol].pct_change()
        nav[exc_col] = nav["port_ret"] - nav[ret_col]

    # Drop first row (NaN return)
    nav = nav.iloc[1:].copy()

    # Select columns
    cols = ["date", "port_ret", "n_stocks"]
    for bcol in bench_cols:
        bname = _bench_name(bcol)
        cols += [f"bench_ret_{bname}", f"excess_{bname}"]

    return nav[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary printing (Chinese output per user preference)
# ---------------------------------------------------------------------------

def _compute_metrics(rets: np.ndarray, periods_per_year: float):
    """Compute standard performance metrics from a return series."""
    n = len(rets)
    if n == 0:
        return {}
    cum_ret = (1 + rets).prod() - 1
    ann_ret = (1 + cum_ret) ** (periods_per_year / n) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    win_rate = (rets > 0).mean()

    nav = (1 + rets).cumprod()
    drawdown = nav / np.maximum.accumulate(nav) - 1
    max_dd = drawdown.min()

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "cum_ret": cum_ret,
    }


def _periods_per_year(freq: str) -> float:
    return {"D": 252, "W": 52, "M": 12, "Q": 4}.get(freq, 12)


def _discover_bench_names(returns_df: pd.DataFrame) -> List[str]:
    """Extract benchmark names from returns_df columns like bench_ret_{name}."""
    prefix = "bench_ret_"
    return [c[len(prefix):] for c in returns_df.columns if c.startswith(prefix)]


def print_summary(strategy_name: str,
                  returns_df: pd.DataFrame,
                  nav_df: pd.DataFrame,
                  cfg: BacktestConfig):
    """Print standardised performance summary to stdout (Chinese)."""
    if returns_df.empty:
        print("[警告] 无收益数据，跳过总结输出")
        return

    ppy = _periods_per_year(cfg.rebalance_freq)
    rets = returns_df["port_ret"].values
    sm = _compute_metrics(rets, ppy)

    # Discover all benchmarks from returns_df columns
    bench_names = _discover_bench_names(returns_df)
    bench_metrics = {}  # {name: metrics_dict}
    for bname in bench_names:
        brets = returns_df[f"bench_ret_{bname}"].dropna().values
        bench_metrics[bname] = _compute_metrics(brets, ppy)

    # ── Header ──
    print()
    print("=" * 60)
    print(f"  策略名称 : {strategy_name}")
    print(f"  回测区间 : {returns_df['date'].iloc[0]} ~ "
          f"{returns_df['date'].iloc[-1]}")
    if bench_names:
        print(f"  基    准 : {', '.join(bench_names)}")
    else:
        print(f"  基    准 : 无")
    print("=" * 60)

    # ── Table header ──
    header = f"  {'指标':<16s}  {'策略':>12s}"
    for bname in bench_names:
        header += f"  {bname:>12s}"
    print(header)
    print("-" * 60)

    def _row(label, strat_val, bench_vals=None, fmt="+.1%"):
        sv = format(strat_val, fmt) if strat_val is not None else "—"
        line = f"  {label:<16s}  {sv:>12s}"
        if bench_vals:
            for bv in bench_vals:
                line += f"  {format(bv, fmt) if bv is not None else '—':>12s}"
        else:
            for _ in bench_names:
                line += f"  {'—':>12s}"
        print(line)

    _row("年化收益率", sm["ann_ret"],
         [bench_metrics[b].get("ann_ret") for b in bench_names])
    _row("年化波动率", sm["ann_vol"],
         [bench_metrics[b].get("ann_vol") for b in bench_names], ".1%")
    _row("夏普比率", sm["sharpe"],
         [bench_metrics[b].get("sharpe") for b in bench_names], ".2f")
    _row("最大回撤", sm["max_dd"],
         [bench_metrics[b].get("max_dd") for b in bench_names])
    _row("胜率", sm["win_rate"], None, ".1%")
    _row("累计收益", sm["cum_ret"],
         [bench_metrics[b].get("cum_ret") for b in bench_names])

    # Alpha & IR per benchmark
    for bname in bench_names:
        bm = bench_metrics[bname]
        alpha = sm["ann_ret"] - bm.get("ann_ret", 0)
        excess_col = f"excess_{bname}"
        if excess_col in returns_df.columns:
            excess = returns_df[excess_col].dropna().values
            te = excess.std() * np.sqrt(ppy) if len(excess) > 0 else 0
            ir = (alpha / te) if te > 0 else float("nan")
        else:
            ir = None
        label_suffix = f"({bname})" if len(bench_names) > 1 else ""
        _row(f"超额收益{label_suffix}", alpha, fmt="+.1%")
        if ir is not None:
            _row(f"信息比率{label_suffix}", ir, fmt=".2f")

    print("=" * 60)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def save_results(strategy_name: str,
                 nav_df: pd.DataFrame,
                 returns_df: pd.DataFrame,
                 cfg: BacktestConfig):
    """Save NAV and returns CSV to the output directory."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    base = os.path.join(cfg.output_dir, strategy_name)
    nav_path = f"{base}_nav.csv"
    ret_path = f"{base}_monthly_returns.csv"
    nav_df.to_csv(nav_path, index=False, float_format="%.7f")
    returns_df.to_csv(ret_path, index=False, float_format="%.8f")
    print(f"\n  [输出] {nav_path}")
    print(f"  [输出] {ret_path}")


# ---------------------------------------------------------------------------
# Report generation — {strategy_name}_report.md
# ---------------------------------------------------------------------------

def save_report(strategy_name: str,
                strategy_description: str,
                nav_df: pd.DataFrame,
                returns_df: pd.DataFrame,
                cfg: BacktestConfig):
    """
    Generate ``{strategy_name}_report.md`` per OUTPUT.md specification.

    Sections:
    1. 策略概述 — from ``strategy.describe()``
    2. 回测配置
    3. 绩效汇总 (all benchmarks)
    4. 分年度表现 (all benchmarks)
    5. 总结与改进空间
    """
    if returns_df.empty:
        print("  [警告] 无收益数据，跳过报告生成")
        return

    ppy = _periods_per_year(cfg.rebalance_freq)
    rets = returns_df["port_ret"].values
    sm = _compute_metrics(rets, ppy)

    bench_names = _discover_bench_names(returns_df)
    bench_metrics = {}
    for bname in bench_names:
        brets = returns_df[f"bench_ret_{bname}"].dropna().values
        bench_metrics[bname] = _compute_metrics(brets, ppy)

    has_bench = len(bench_names) > 0

    lines = []

    def _fmt(val, fmt="+.2%"):
        return format(val, fmt) if val is not None else "—"

    # ── Title ──
    lines.append(f"# {strategy_name} 回测报告\n")

    # ── 1. Strategy description ──
    lines.append("## 1. 策略概述\n")
    lines.append(strategy_description.strip())
    lines.append("")

    # ── 2. Backtest config ──
    lines.append("## 2. 回测配置\n")
    lines.append("| 参数 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 回测区间 | {cfg.start_date} ~ {cfg.end_date} |")
    lines.append(f"| 初始资金 | {cfg.initial_capital:,.0f} |")
    lines.append(f"| 调仓频率 | {cfg.rebalance_freq} |")
    lines.append(f"| 单边佣金 | {cfg.commission_rate * 10000:.1f} bps |")
    lines.append(f"| 滑点 | {cfg.slippage * 10000:.1f} bps |")
    lines.append(f"| 整手约束 | {cfg.lot_size} 股/手 |")
    lines.append(f"| 基准 | {', '.join(bench_names) if bench_names else '无'} |")
    lines.append("")

    # ── 3. Performance summary ──
    lines.append("## 3. 绩效汇总\n")

    if has_bench:
        # Header: | 指标 | 策略 | bench1 | bench2 | ...
        hdr = "| 指标 | 策略 |" + " | ".join(f" {b} " for b in bench_names) + " |"
        sep = "|------|------" + "|------" * len(bench_names) + "|"
        lines.append(hdr)
        lines.append(sep)

        def _report_row(label, key, fmt="+.2%"):
            sv = _fmt(sm.get(key), fmt)
            bvs = " | ".join(_fmt(bench_metrics[b].get(key), fmt) for b in bench_names)
            lines.append(f"| {label} | {sv} | {bvs} |")

        _report_row("年化收益率", "ann_ret")
        _report_row("年化波动率", "ann_vol", ".2%")
        _report_row("夏普比率", "sharpe", ".2f")
        _report_row("最大回撤", "max_dd")
        win_bvs = " | ".join("—" for _ in bench_names)
        lines.append(f"| 胜率 | {_fmt(sm['win_rate'], '.1%')} | {win_bvs} |")
        _report_row("累计收益", "cum_ret")

        # Alpha & IR per benchmark
        for bname in bench_names:
            bm = bench_metrics[bname]
            alpha = sm["ann_ret"] - bm.get("ann_ret", 0)
            excess_col = f"excess_{bname}"
            if excess_col in returns_df.columns:
                excess = returns_df[excess_col].dropna().values
                te = excess.std() * np.sqrt(ppy) if len(excess) > 0 else 0
                ir = (alpha / te) if te > 0 else float("nan")
            else:
                ir = None
            dash_others = " | ".join("—" for _ in bench_names)
            lines.append(f"| 超额收益 vs {bname} | {_fmt(alpha)} | {dash_others} |")
            if ir is not None:
                lines.append(f"| 信息比率 vs {bname} | {_fmt(ir, '.2f')} | {dash_others} |")
    else:
        lines.append("| 指标 | 策略 |")
        lines.append("|------|------|")
        lines.append(f"| 年化收益率 | {_fmt(sm['ann_ret'])} |")
        lines.append(f"| 年化波动率 | {_fmt(sm['ann_vol'], '.2%')} |")
        lines.append(f"| 夏普比率 | {_fmt(sm['sharpe'], '.2f')} |")
        lines.append(f"| 最大回撤 | {_fmt(sm['max_dd'])} |")
        lines.append(f"| 胜率 | {_fmt(sm['win_rate'], '.1%')} |")
        lines.append(f"| 累计收益 | {_fmt(sm['cum_ret'])} |")
    lines.append("")

    # ── 4. Yearly breakdown ──
    lines.append("## 4. 分年度表现\n")
    rdf = returns_df.copy()
    rdf["year"] = pd.to_datetime(rdf["date"]).dt.year

    if has_bench:
        # Header: | 年份 | 策略收益 | bench1 收益 | excess1 | bench2 收益 | excess2 | ... | 期数 |
        hdr_parts = ["| 年份 | 策略收益"]
        sep_parts = ["|------|--------"]
        for bname in bench_names:
            hdr_parts.append(f"{bname}收益")
            hdr_parts.append(f"超额vs{bname}")
            sep_parts.append("--------")
            sep_parts.append("--------")
        hdr_parts.append("期数 |")
        sep_parts.append("------|")
        lines.append(" | ".join(hdr_parts))
        lines.append(" | ".join(sep_parts))
    else:
        lines.append("| 年份 | 策略收益 | 期数 |")
        lines.append("|------|----------|------|")

    for year, grp in rdf.groupby("year"):
        yr_ret = (1 + grp["port_ret"]).prod() - 1
        n_periods = len(grp)
        if has_bench:
            parts = [f"| {year} | {yr_ret:+.2%}"]
            for bname in bench_names:
                yr_bench = (1 + grp[f"bench_ret_{bname}"].fillna(0)).prod() - 1
                yr_excess = yr_ret - yr_bench
                parts.append(f"{yr_bench:+.2%}")
                parts.append(f"{yr_excess:+.2%}")
            parts.append(f"{n_periods} |")
            lines.append(" | ".join(parts))
        else:
            lines.append(f"| {year} | {yr_ret:+.2%} | {n_periods} |")
    lines.append("")

    # ── 5. Summary & improvements (placeholder) ──
    lines.append("## 5. 总结与改进空间\n")
    lines.append("> 以下内容由策略作者补充，运行回测后请编辑此节。\n")
    lines.append("- **表现总结**：（待补充）")
    lines.append("- **风险分析**：（待补充）")
    lines.append("- **改进方向**：（待补充）")
    lines.append("")

    # ── Write file ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    report_path = os.path.join(cfg.output_dir, f"{strategy_name}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [输出] {report_path}")
