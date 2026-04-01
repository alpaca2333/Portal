"""
Performance analyzer — NAV computation, metrics, and output formatting.
Conforms to OUTPUT.md specification.
"""
import os
from typing import Dict, List, Optional, Tuple

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
    Build cumulative NAV DataFrame — absolute capital values.

    Columns: date, strategy, bench_{name1}, bench_{name2}, ...
    strategy column = actual portfolio value (not normalised).
    bench columns = scaled to initial_capital for comparability.
    """
    if not snapshots:
        return pd.DataFrame(columns=["date", "strategy"])

    rows = []
    init_cap = cfg.initial_capital
    for snap in snapshots:
        rows.append({
            "date": snap["date"],
            "strategy": snap["total_value"],       # absolute capital
        })
    nav = pd.DataFrame(rows)

    # Benchmark — scale to init_cap so the lines are directly comparable
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
                bench_nav_map[ds] = row["close"] / first_close * init_cap
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
    nav["port_ret"] = nav["strategy"].pct_change(fill_method=None)

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
        nav[ret_col] = nav[bcol].pct_change(fill_method=None)
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
    return {"D": 252, "W": 52, "BW": 26, "M": 12, "Q": 4}.get(freq, 12)


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
# Factor contribution analysis
# ---------------------------------------------------------------------------

def build_factor_contribution(
    factor_exposures: List[Tuple[str, pd.DataFrame]],
    returns_df: pd.DataFrame,
    cfg: "BacktestConfig",
) -> pd.DataFrame:
    """
    Compute per-period factor contribution to portfolio return.

    For each period t:
      factor_contribution_j(t) = Σ_i [ weight_i(t) × exposure_i_j(t) ] × port_ret(t)

    This decomposes the portfolio return into contributions from each factor,
    proportional to the portfolio's weighted average exposure to that factor.

    Parameters
    ----------
    factor_exposures : list of (date_str, DataFrame)
        Each DataFrame has columns: ts_code, weight, factor_1, factor_2, ...
    returns_df : pd.DataFrame
        Must contain columns: date, port_ret.
    cfg : BacktestConfig
        For output directory.

    Returns
    -------
    pd.DataFrame
        Columns: date, factor_1, factor_2, ...
        Each value = that factor's contribution to portfolio return in that period.
    """
    if not factor_exposures or returns_df.empty:
        return pd.DataFrame()

    # Build a date → port_ret lookup
    ret_map = {}
    for _, row in returns_df.iterrows():
        d = pd.Timestamp(row["date"]).strftime("%Y%m%d")
        ret_map[d] = row["port_ret"]

    # Discover factor columns from the first exposure DataFrame
    sample_df = factor_exposures[0][1]
    factor_cols = [c for c in sample_df.columns if c not in ("ts_code", "weight")]

    rows = []
    for date_str, exp_df in factor_exposures:
        port_ret = ret_map.get(date_str)
        if port_ret is None or pd.isna(port_ret):
            continue

        row = {"date": pd.Timestamp(date_str).strftime("%Y-%m-%d")}

        # Weighted average exposure per factor (raw, no demeaning).
        # Factor values are rank-normalized to [0, 1].  The contribution
        # directly reflects how much each factor's weighted exposure
        # contributed to the portfolio return.
        weights = exp_df["weight"].values
        for fc in factor_cols:
            if fc in exp_df.columns:
                raw_exposure = exp_df[fc].fillna(0.0).values
                weighted_exposure = (weights * raw_exposure).sum()
                contrib = weighted_exposure * port_ret
                row[fc] = contrib
            else:
                row[fc] = 0.0

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    contrib_df = pd.DataFrame(rows)

    # Save to CSV
    out_dir = os.path.join(cfg.output_dir, cfg.strategy_name)
    os.makedirs(out_dir, exist_ok=True)
    
    ts = f"-{cfg.run_timestamp}" if cfg.run_timestamp else ""
    path = os.path.join(out_dir, f"{cfg.strategy_name}{ts}_factor_contrib.csv")
    contrib_df.to_csv(path, index=False, float_format="%.8f")
    print(f"  [输出] {path}")

    # Print summary table
    _print_factor_contribution_summary(contrib_df, factor_cols, cfg)

    return contrib_df


def _print_factor_contribution_summary(
    contrib_df: pd.DataFrame,
    factor_cols: List[str],
    cfg: "BacktestConfig",
):
    """Print a summary table of cumulative factor contributions."""
    if contrib_df.empty:
        return

    print()
    print("=" * 60)
    print("  因子收益贡献分析")
    print("=" * 60)
    print(f"  {'因子':<20s}  {'累计贡献':>10s}  {'平均贡献/期':>12s}  {'贡献占比':>10s}")
    print("-" * 60)

    cum_contribs = {}
    for fc in factor_cols:
        if fc in contrib_df.columns:
            cum = contrib_df[fc].sum()
            cum_contribs[fc] = cum

    grand_total = sum(cum_contribs.values())

    # Sort by contribution value descending
    sorted_contribs = sorted(cum_contribs.items(), key=lambda x: x[1], reverse=True)
    for name, cum in sorted_contribs:
        avg = cum / len(contrib_df)
        pct = cum / grand_total * 100 if grand_total != 0 else 0
        print(f"  {name:<20s}  {cum:>+10.4f}  {avg:>+12.6f}  {pct:>9.1f}%")

    print("-" * 60)
    print(f"  {'合计':<20s}  {grand_total:>+10.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def save_results(strategy_name: str,
                 nav_df: pd.DataFrame,
                 returns_df: pd.DataFrame,
                 cfg: BacktestConfig):
    """Save NAV and returns CSV to the output directory."""
    out_dir = os.path.join(cfg.output_dir, strategy_name)
    os.makedirs(out_dir, exist_ok=True)
    
    ts = f"-{cfg.run_timestamp}" if cfg.run_timestamp else ""
    nav_path = os.path.join(out_dir, f"{strategy_name}{ts}_nav.csv")
    ret_path = os.path.join(out_dir, f"{strategy_name}{ts}_monthly_returns.csv")
    
    nav_df.to_csv(nav_path, index=False, float_format="%.2f")
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
                cfg: BacktestConfig,
                factor_contrib_df: pd.DataFrame = None):
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

    # ── 5. Factor contribution (if available) ──
    if factor_contrib_df is not None and not factor_contrib_df.empty:
        lines.append("## 5. 因子收益贡献\n")
        factor_cols_report = [c for c in factor_contrib_df.columns if c != "date"]

        # Cumulative contribution table
        lines.append("### 5.1 累计因子贡献\n")
        lines.append("| 因子 | 累计贡献 | 平均贡献/期 | 贡献占比 |")
        lines.append("|------|----------|-------------|----------|")

        cum_map = {}
        for fc in factor_cols_report:
            cum_map[fc] = factor_contrib_df[fc].sum()

        grand_total = sum(cum_map.values())
        n_periods = len(factor_contrib_df)
        # Sort by contribution value descending
        sorted_cum = sorted(cum_map.items(), key=lambda x: x[1], reverse=True)
        for name, cum in sorted_cum:
            avg = cum / n_periods
            pct = cum / grand_total * 100 if grand_total != 0 else 0
            lines.append(f"| {name} | {cum:+.4f} | {avg:+.6f} | {pct:.1f}% |")
        lines.append(f"| **合计** | **{grand_total:+.4f}** | | |")
        lines.append("")

        # Yearly factor contribution
        lines.append("### 5.2 分年度因子贡献\n")
        cdf = factor_contrib_df.copy()
        cdf["year"] = pd.to_datetime(cdf["date"]).dt.year
        hdr = "| 年份 |" + " | ".join(f" {fc} " for fc in factor_cols_report) + " |"
        sep = "|------" + "|------" * len(factor_cols_report) + "|"
        lines.append(hdr)
        lines.append(sep)
        for year, grp in cdf.groupby("year"):
            parts = [f"| {year}"]
            for fc in factor_cols_report:
                parts.append(f"{grp[fc].sum():+.4f}")
            parts[-1] = parts[-1] + " |"
            lines.append(" | ".join(parts))
        lines.append("")

    # ── 6. Summary & improvements (placeholder) ──
    section_num = 6 if (factor_contrib_df is not None and not factor_contrib_df.empty) else 5
    lines.append(f"## {section_num}. 总结与改进空间\n")
    lines.append("> 以下内容由策略作者补充，运行回测后请编辑此节。\n")
    lines.append("- **表现总结**：（待补充）")
    lines.append("- **风险分析**：（待补充）")
    lines.append("- **改进方向**：（待补充）")
    lines.append("")

    # ── Write file ──
    out_dir = os.path.join(cfg.output_dir, strategy_name)
    os.makedirs(out_dir, exist_ok=True)
    
    ts = f"-{cfg.run_timestamp}" if cfg.run_timestamp else ""
    report_path = os.path.join(out_dir, f"{strategy_name}{ts}_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"      报告已保存: {report_path}")


# ---------------------------------------------------------------------------
# Windows desktop notification
# ---------------------------------------------------------------------------

def notify_backtest_complete(strategy_name: str,
                             returns_df: pd.DataFrame,
                             cfg: BacktestConfig):
    """
    Send a Windows toast notification when backtest finishes.
    Uses PowerShell + BurntToast (if available) or native .NET toast API,
    with ctypes MessageBox as fallback.  Errors are silently swallowed.
    """
    import sys
    if sys.platform != "win32":
        return

    # Compute metrics for the notification
    try:
        ppy = _periods_per_year(cfg.rebalance_freq)
        rets = returns_df["port_ret"].values
        sm = _compute_metrics(rets, ppy)
        cum_ret = sm.get("cum_ret", 0)
        ann_ret = sm.get("ann_ret", 0)
        sharpe = sm.get("sharpe", 0)
        max_dd = sm.get("max_dd", 0)

        title = f"回测完成 — {strategy_name}"
        body = (
            f"累计收益: {cum_ret:+.1%}  |  年化收益: {ann_ret:+.1%}\n"
            f"夏普比率: {sharpe:.2f}  |  最大回撤: {max_dd:+.1%}"
        )
    except Exception:
        title = f"回测完成 — {strategy_name}"
        body = "回测已结束，请查看报告。"

    # ── Attempt 1: PowerShell toast (non-blocking) ──
    try:
        import subprocess
        ps_script = (
            "[Windows.UI.Notifications.ToastNotificationManager, "
            "Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; "
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
            "ContentType = WindowsRuntime] > $null; "
            f"$template = '<toast><visual><binding template=\"ToastGeneric\">"
            f"<text>{title}</text>"
            f"<text>{body}</text>"
            f"</binding></visual></toast>'; "
            "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument; "
            "$xml.LoadXml($template); "
            "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); "
            "[Windows.UI.Notifications.ToastNotificationManager]::"
            "CreateToastNotifier('QuantBacktest').Show($toast)"
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", ps_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return  # success — no need for fallback
    except Exception:
        pass

    # ── Attempt 2: ctypes MessageBox (blocking but always works) ──
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            f"{body}",
            title,
            0x00000040,  # MB_ICONINFORMATION
        )
    except Exception:
        pass