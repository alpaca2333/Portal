"""
Backtest engine — main loop.

Flow
----
1. Load trade calendar (lightweight)
2. Determine rebalance dates
3. Open DataAccessor (holds SQLite connection)
4. On each rebalance date:
   a. Query prices on demand via DataAccessor
   b. Call strategy.generate_target_weights(date, accessor, holdings)
   c. Execute rebalance via SimBroker
   d. Record NAV snapshot, save trades to disk
5. Close DataAccessor, save results & print summary

Memory strategy: no full-table load.  Each rebalance date only fetches
the rows it needs via SQL queries.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .strategy_base import StrategyBase
from .broker import SimBroker
from .data_loader import (
    DataAccessor,
    load_trade_calendar,
    get_rebalance_dates,
    load_all_benchmarks,
)
from .analyzer import build_nav_df, build_returns_df, print_summary, save_results, save_report, build_factor_contribution


def _save_all_trades(all_trades, strategy_name: str, output_dir: str):
    """Save all trade records to a single CSV: {strategy_name}-trade.csv."""
    if not all_trades:
        return
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for r in all_trades:
        rows.append({
            "date": r.date,
            "ts_code": r.ts_code,
            "direction": r.direction,
            "price": r.price,
            "shares": r.shares,
            "amount": r.amount,
            "commission": r.commission,
            "status": r.status,
            "reason": r.reason,
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"{strategy_name}-trade.csv")
    df.to_csv(path, index=False)
    print(f"      成交记录已保存: {path}  ({len(df)} 笔)")


def run_backtest(
    strategy: StrategyBase,
    cfg: Optional[BacktestConfig] = None,
) -> dict:
    """
    Run a full backtest.

    Parameters
    ----------
    strategy : StrategyBase
        A concrete strategy instance.
    cfg : BacktestConfig, optional
        If not provided, default config is used.

    Returns
    -------
    dict with keys: nav_df, returns_df, trade_log, snapshots
    """
    if cfg is None:
        cfg = BacktestConfig()
    cfg.strategy_name = strategy.name

    print("=" * 50)
    print(f"  回测引擎启动")
    print(f"  策略名称 : {strategy.name}")
    print(f"  回测区间 : {cfg.start_date} ~ {cfg.end_date}")
    print(f"  初始资金 : {cfg.initial_capital:,.0f}")
    print(f"  单边佣金 : {cfg.commission_rate * 10000:.1f} bps")
    print(f"  滑点     : {cfg.slippage * 10000:.1f} bps")
    print(f"  整手约束 : {cfg.lot_size} 股/手")
    print(f"  调仓频率 : {cfg.rebalance_freq}")
    print(f"  基准目录 : {cfg.baseline_dir}")
    print(f"  数据模式 : 懒加载（低内存）")
    print("=" * 50)

    # ── 1. Load trade calendar (lightweight: only date strings) ──
    print("\n[1/5] 加载交易日历 ...")
    trade_dates = load_trade_calendar(cfg)
    print(f"      交易日共 {len(trade_dates)} 天  "
          f"({trade_dates[0].strftime('%Y-%m-%d')} ~ "
          f"{trade_dates[-1].strftime('%Y-%m-%d')})")

    # ── 2. Rebalance dates ──
    rebal_dates = get_rebalance_dates(trade_dates, cfg.rebalance_freq)
    print(f"      调仓日共 {len(rebal_dates)} 个")

    # ── 3. Open DataAccessor ──
    print("\n[2/5] 初始化数据访问器（懒加载模式） ...")
    accessor = DataAccessor(cfg)
    accessor.open()
    n_stocks = accessor.count_stocks()
    n_rows = accessor.count_rows()
    print(f"      数据库覆盖 {n_stocks} 只股票, {n_rows:,} 条记录")
    print(f"      数据将在每期调仓时按需加载，不预先读入内存")

    # ── 4. Load ALL benchmarks from baseline dir ──
    print("\n[3/5] 加载基准数据 ...")
    benchmarks = load_all_benchmarks(cfg)
    if benchmarks:
        for bname, bdf in benchmarks.items():
            print(f"      基准 {bname}  ({len(bdf)} 条记录)")
    else:
        print("      无基准数据")

    # ── 5. Run main loop ──
    print(f"\n[4/5] 运行回测主循环 ({len(rebal_dates)} 期) ...")
    broker = SimBroker(cfg)
    snapshots = []
    all_trades = []
    factor_exposures = []  # List of (date_str, DataFrame) for factor attribution

    prev_total = cfg.initial_capital

    try:
        for i, rebal_date in enumerate(rebal_dates):
            date_str = rebal_date.strftime("%Y%m%d")

            # Arm the look-ahead guard for this rebalance date
            accessor.set_current_date(rebal_date)

            # On-demand: fetch prices for this date only
            prices = accessor.get_prices(rebal_date)

            if not prices:
                # No market data on this date, skip
                continue

            # Call strategy — pass accessor, NOT a big DataFrame
            target_weights = strategy.generate_target_weights(
                rebal_date, accessor, dict(broker.holdings)
            )

            # Collect factor exposures (optional, strategy may return None)
            factor_exp = strategy.get_factor_exposures(rebal_date, target_weights)
            if factor_exp is not None:
                factor_exposures.append((date_str, factor_exp))

            # Execute rebalance
            records = broker.rebalance(date_str, target_weights, prices)
            all_trades.extend(records)

            # Record snapshot
            snap = broker.snapshot(date_str, prices)
            snapshots.append(snap)

            # ── Progress logging (every period) ──
            filled_recs = [r for r in records if r.status == "FILLED"]
            rejected_cnt = sum(1 for r in records if r.status == "REJECTED")
            buy_recs = [r for r in filled_recs if r.direction == "BUY"]
            sell_recs = [r for r in filled_recs if r.direction == "SELL"]
            buy_amt = sum(r.amount for r in buy_recs)
            sell_amt = sum(r.amount for r in sell_recs)
            turnover = (buy_amt + sell_amt) / 2 / snap["total_value"] * 100 if snap["total_value"] > 0 else 0
            period_ret = (snap["total_value"] / prev_total - 1) * 100 if prev_total > 0 else 0
            cum_ret = (snap["total_value"] / cfg.initial_capital - 1) * 100
            cash_pct = snap["cash"] / snap["total_value"] * 100 if snap["total_value"] > 0 else 0

            ret_sign = "+" if period_ret >= 0 else ""
            cum_sign = "+" if cum_ret >= 0 else ""

            print(f"      [{i+1:>{len(str(len(rebal_dates)))}}/{len(rebal_dates)}] "
                  f"{date_str}  "
                  f"资金={snap['total_value']:>12,.0f}  "
                  f"本期={ret_sign}{period_ret:.2f}%  "
                  f"累计={cum_sign}{cum_ret:.2f}%  "
                  f"持仓={snap['n_stocks']:>3}只  "
                  f"买入={len(buy_recs)}笔 {buy_amt:>10,.0f}元  "
                  f"卖出={len(sell_recs)}笔 {sell_amt:>10,.0f}元  "
                  f"换手={turnover:.1f}%  "
                  f"现金={cash_pct:.1f}%"
                  + (f"  拒绝={rejected_cnt}笔" if rejected_cnt > 0 else ""))

            prev_total = snap["total_value"]
    finally:
        # Always close the accessor
        accessor.close()

    # ── 6. Build result DataFrames ──
    print(f"\n[5/5] 生成回测报告 ...")
    nav_df = build_nav_df(snapshots, benchmarks, rebal_dates, cfg)
    returns_df = build_returns_df(nav_df, snapshots, benchmarks, cfg)

    # ── 7. Save trades & results ──
    _save_all_trades(all_trades, cfg.strategy_name, cfg.output_dir)
    save_results(cfg.strategy_name, nav_df, returns_df, cfg)

    # ── 7b. Factor contribution analysis (if exposures available) ──
    factor_contrib_df = None
    if factor_exposures:
        factor_contrib_df = build_factor_contribution(
            factor_exposures, returns_df, cfg
        )

    save_report(cfg.strategy_name, strategy.describe(), nav_df, returns_df, cfg,
                factor_contrib_df=factor_contrib_df)
    print_summary(cfg.strategy_name, returns_df, nav_df, cfg)

    return {
        "nav_df": nav_df,
        "returns_df": returns_df,
        "trade_log": all_trades,
        "snapshots": snapshots,
        "factor_contrib_df": factor_contrib_df,
    }
