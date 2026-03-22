"""
Backtest engine: the main rebalance loop.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from engine.types import StrategyConfig, SelectionMode


def _default_select(
    signal: pd.DataFrame,
    prev_holdings: set,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """
    Default concentrated selection:
    1. (Optional) Apply buffer band to incumbent holdings
    2. Select by TOP_PCT or TOP_N
    3. (Optional) Cap per industry
    """
    signal = signal.copy()

    # Buffer band: give incumbents a score bonus
    if cfg.buffer_sigma > 0 and len(prev_holdings) > 0:
        is_incumbent = signal["code"].isin(prev_holdings)
        signal.loc[is_incumbent, "score"] += cfg.buffer_sigma

    # Selection
    if cfg.selection_mode == SelectionMode.TOP_PCT:
        cutoff = signal["score"].quantile(1 - cfg.top_pct)
        selected = signal[signal["score"] >= cutoff].copy()
    else:  # TOP_N
        selected = signal.nlargest(cfg.top_n, "score").copy()

    # Cap per industry
    if cfg.max_per_industry > 0:
        selected = (
            selected.sort_values("score", ascending=False)
            .groupby("industry_code", group_keys=False)
            .head(cfg.max_per_industry)
        )

    return selected


def run_backtest(snap: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Run the rebalance backtest loop.

    Parameters
    ----------
    snap : DataFrame
        Scored universe with 'period', 'period_sort', 'code', 'score',
        'industry_code', 'date' columns, and optionally 'next_open'/'next_date'
        for next-trade execution.
    cfg : StrategyConfig

    Returns
    -------
    DataFrame with one row per holding period:
        date, period, port_ret_gross, port_ret, n_stocks, n_industries,
        turnover_sell, turnover_buy, tc
    """
    cost_label = f"{cfg.single_side_cost:.4%}"
    print(f"[回测] 开始调仓 "
          f"(模式={cfg.selection_mode.value}, "
          f"成本={cost_label}) ...")

    entry_price_col = "next_open" if "next_open" in snap.columns else "close"
    entry_date_col = "next_date" if "next_date" in snap.columns else "date"
    exit_price_col = entry_price_col
    exit_date_col = entry_date_col

    # Build ordered period list
    period_info = (
        snap.groupby("period")
        .agg(period_sort=("period_sort", "first"))
        .reset_index()
        .sort_values("period_sort")
    )
    ordered_periods = period_info["period"].tolist()

    # Determine backtest start sort key
    # Parse backtest_start -> sort key
    bt_year = int(cfg.backtest_start[:4])
    bt_month = int(cfg.backtest_start[5:7])
    if cfg.freq.value == "biweekly":
        bt_start_sort = bt_year * 100 * 10 + bt_month * 10 + 1
    elif cfg.freq.value == "weekly":
        # For weekly: convert backtest_start date to ISO week number
        bt_date = pd.Timestamp(cfg.backtest_start)
        bt_iso_year = bt_date.isocalendar()[0]
        bt_iso_week = bt_date.isocalendar()[1]
        bt_start_sort = bt_iso_year * 100 + bt_iso_week
    else:
        bt_start_sort = bt_year * 100 + bt_month

    select_fn = cfg.post_select if cfg.post_select is not None else _default_select

    results = []
    prev_holdings: set = set()

    for i in range(len(ordered_periods) - 1):
        sig_period = ordered_periods[i]
        hold_period = ordered_periods[i + 1]

        sig_sort = period_info.loc[
            period_info["period"] == sig_period, "period_sort"
        ].values[0]
        if sig_sort < bt_start_sort:
            continue

        signal = snap[
            (snap["period"] == sig_period) & snap["score"].notna()
        ].copy()
        if len(signal) < cfg.min_holding * 2:
            continue

        # Selection (default or custom)
        selected = select_fn(signal, prev_holdings, cfg)[["code", entry_price_col, entry_date_col]].copy()
        selected = selected.rename(columns={
            entry_price_col: "entry_open",
            entry_date_col: "entry_date",
        })

        hold = snap[snap["period"] == hold_period][
            ["code", exit_price_col, exit_date_col]
        ].copy()
        hold = hold.rename(columns={
            exit_price_col: "exit_open",
            exit_date_col: "exit_date",
        })

        merged = selected.merge(hold, on="code", how="inner").dropna(
            subset=["entry_open", "exit_open", "entry_date", "exit_date"]
        )
        merged = merged[(merged["entry_open"] > 0) & (merged["exit_open"] > 0)]
        if len(merged) < cfg.min_holding:
            continue

        curr_holdings = set(merged["code"].tolist())
        n_curr = len(curr_holdings)

        # Turnover
        if len(prev_holdings) == 0:
            turnover_buy = 1.0
            turnover_sell = 0.0
        else:
            sold = prev_holdings - curr_holdings
            bought = curr_holdings - prev_holdings
            n_prev = len(prev_holdings)
            turnover_sell = len(sold) / n_prev if n_prev > 0 else 0.0
            turnover_buy = len(bought) / n_curr if n_curr > 0 else 0.0

        tc = cfg.single_side_cost * (turnover_sell + turnover_buy)

        # Equal-weight portfolio return
        merged["ret"] = merged["exit_open"] / merged["entry_open"] - 1
        gross_ret = merged["ret"].mean()
        net_ret = gross_ret - tc
        hold_date = pd.to_datetime(merged["exit_date"]).max()

        # Industry distribution
        n_industries = (
            signal[signal["code"].isin(curr_holdings)]["industry_code"].nunique()
        )

        results.append({
            "date": hold_date.strftime("%Y-%m-%d"),
            "period": hold_period,
            "port_ret_gross": gross_ret,
            "port_ret": net_ret,
            "n_stocks": int(n_curr),
            "n_industries": int(n_industries),
            "turnover_sell": turnover_sell,
            "turnover_buy": turnover_buy,
            "tc": tc,
        })
        prev_holdings = curr_holdings

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        avg_to = (result_df["turnover_sell"] + result_df["turnover_buy"]).mean() / 2
        avg_tc = result_df["tc"].mean()
        total_tc = result_df["tc"].sum()
        avg_n = result_df["n_stocks"].mean()
        avg_ind = result_df["n_industries"].mean()
        print(f"[回测] 有效调仓期数: {len(result_df)}")
        print(f"[回测] 平均持仓: {avg_n:.0f} 只 / {avg_ind:.0f} 个行业")
        print(f"[回测] 平均单边换手: {avg_to:.1%}")
        print(f"[回测] 平均每期成本: {avg_tc:.4%}，累计拖累: {total_tc:.4%}")
    return result_df
