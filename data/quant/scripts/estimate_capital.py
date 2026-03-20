"""
LGB策略实盘资金测算脚本
======================
从已有的预测结果和原始数据中，模拟最近一期的选股，
计算最低资金需求（按A股100股/手的最小交易单位）。

用法:
    cd /projects/portal/data/quant
    python scripts/estimate_capital.py
"""
from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────── Config (mirror lgb_strategy.py) ──────────────

PREDICTIONS_PATH = PROJECT_ROOT / "backtest" / "lgb_stock_selection_predictions.csv"
DB_PATH = PROJECT_ROOT / "processed" / "stocks.db"

TOP_PCT = 0.05
MAX_PER_INDUSTRY = 5
MIN_INDUSTRY_COUNT = 5
MIN_HOLDING = 20
BUFFER_SIGMA = 0.3


def load_last_n_periods(n: int = 3) -> pd.DataFrame:
    """Load last N periods from predictions CSV (only read necessary columns)."""
    print(f"[资金测算] 加载预测数据（仅最后{n}期）...")

    # Read only needed columns to save memory
    usecols = ["code", "date", "period", "period_sort", "pred_score",
               "close", "next_open", "industry_code", "industry_name",
               "free_market_cap"]
    df = pd.read_csv(PREDICTIONS_PATH, usecols=usecols)
    df["date"] = pd.to_datetime(df["date"])

    # Get last N periods
    periods_sorted = df.sort_values("period_sort")["period"].unique()
    last_periods = periods_sorted[-n:]
    df = df[df["period"].isin(last_periods)].copy()

    print(f"    加载了 {len(df):,} 行, 覆盖 {len(last_periods)} 期: {list(last_periods)}")
    return df


def simulate_selection(snap: pd.DataFrame, period: str,
                       prev_holdings: set) -> pd.DataFrame:
    """
    Simulate lgb_strategy's ml_select logic for a given period.
    Mirrors the selection in lgb_strategy.py:
    1. Buffer band for incumbents
    2. Top 5% by score
    3. Max 5 per industry
    """
    signal = snap[snap["period"] == period].copy()
    signal = signal.rename(columns={"pred_score": "score"})
    signal = signal.dropna(subset=["score"])

    # Buffer band
    if BUFFER_SIGMA > 0 and len(prev_holdings) > 0:
        score_std = signal["score"].std()
        if score_std > 0:
            is_incumbent = signal["code"].isin(prev_holdings)
            signal.loc[is_incumbent, "score"] += BUFFER_SIGMA * score_std

    # Top 5%
    cutoff = signal["score"].quantile(1 - TOP_PCT)
    selected = signal[signal["score"] >= cutoff].copy()

    # Max per industry
    if MAX_PER_INDUSTRY > 0:
        selected = (
            selected.sort_values("score", ascending=False)
            .groupby("industry_code", group_keys=False)
            .head(MAX_PER_INDUSTRY)
        )

    return selected


def get_latest_prices(codes: list) -> pd.DataFrame:
    """Get the latest available prices from SQLite for each stock."""
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join([f"'{c}'" for c in codes])
    query = f"""
    SELECT code, date, close, open
    FROM kline
    WHERE code IN ({placeholders})
    ORDER BY code, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Take the latest date per stock
    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").groupby("code").last().reset_index()
    return latest[["code", "date", "close", "open"]]


def main():
    print("=" * 64)
    print("LGB 策略实盘资金测算")
    print("=" * 64)

    # Load last few periods
    snap = load_last_n_periods(n=3)

    # Get ordered periods
    periods = snap.sort_values("period_sort")["period"].unique()
    print(f"\n最后 {len(periods)} 期: {list(periods)}")

    # Simulate selection for last 2 periods to build prev_holdings
    prev_holdings = set()
    for i, period in enumerate(periods):
        selected = simulate_selection(snap, period, prev_holdings)
        n_stocks = len(selected)
        n_industries = selected["industry_code"].nunique()

        if i < len(periods) - 1:
            print(f"\n[模拟] 期 {period}: 选中 {n_stocks} 只 / {n_industries} 行业 (用于构建上期持仓)")
            prev_holdings = set(selected["code"].tolist())
        else:
            # This is the last period - the one we care about
            print(f"\n{'='*64}")
            print(f"[最终选股] 期 {period}: 选中 {n_stocks} 只 / {n_industries} 行业")
            print(f"{'='*64}")

    # Get latest real prices for selected stocks
    selected_codes = selected["code"].tolist()
    latest_prices = get_latest_prices(selected_codes)

    # Merge
    holdings = selected.merge(latest_prices, on="code", how="left", suffixes=("_signal", "_latest"))
    holdings = holdings.rename(columns={
        "close_latest": "latest_close",
        "date_latest": "latest_date",
        "close_signal": "signal_close",
        "open": "latest_open",
    })

    # Use the latest close price for capital estimation
    # A-share minimum lot: 100 shares
    LOT_SIZE = 100
    holdings["min_shares"] = LOT_SIZE
    holdings["min_cost_per_stock"] = holdings["latest_close"] * LOT_SIZE
    holdings = holdings.sort_values("min_cost_per_stock", ascending=False)

    # ── Summary Statistics ──
    n_total = len(holdings)
    total_min_capital = holdings["min_cost_per_stock"].sum()
    max_single = holdings["min_cost_per_stock"].max()
    min_single = holdings["min_cost_per_stock"].min()
    median_single = holdings["min_cost_per_stock"].median()
    mean_single = holdings["min_cost_per_stock"].mean()

    print(f"\n{'='*64}")
    print(f"📊 资金测算结果（最低门槛，每只买1手=100股）")
    print(f"{'='*64}")
    print(f"  持仓股票数   : {n_total} 只")
    print(f"  覆盖行业数   : {holdings['industry_code'].nunique()} 个")
    print(f"  最贵一手      : ¥{max_single:>12,.2f}")
    print(f"  最便宜一手    : ¥{min_single:>12,.2f}")
    print(f"  中位数一手    : ¥{median_single:>12,.2f}")
    print(f"  平均一手      : ¥{mean_single:>12,.2f}")
    print(f"  ────────────────────────────────")
    print(f"  最低总资金    : ¥{total_min_capital:>12,.2f}")
    print(f"  约合          : ¥{total_min_capital/10000:>12,.2f} 万元")

    # ── Equal-weight capital estimation ──
    # For true equal-weight, we need each position to have the same notional value.
    # The binding constraint is the stock with the highest price per lot.
    # Equal weight target = total / n_stocks, must be >= max(100 * price_i)
    # So: total_capital >= n_stocks * max(100 * price_i)
    equal_weight_min = n_total * max_single
    print(f"\n{'='*64}")
    print(f"📊 等权配置资金测算（每只股票分配相同金额）")
    print(f"{'='*64}")
    print(f"  每只股票最低分配 : ¥{max_single:>12,.2f}（受最贵股票约束）")
    print(f"  等权最低总资金   : ¥{equal_weight_min:>12,.2f}")
    print(f"  约合             : ¥{equal_weight_min/10000:>12,.2f} 万元")

    # ── More realistic: allow ±10% weight deviation ──
    # If we allow each position to deviate ±10% from target weight,
    # the binding constraint relaxes. Use ceiling to nearest lot.
    print(f"\n{'='*64}")
    print(f"📊 实际可操作资金测算（允许权重偏差，向上取整到100股）")
    print(f"{'='*64}")
    for target_capital in [500000, 1000000, 1500000, 2000000, 3000000, 5000000]:
        per_stock = target_capital / n_total
        holdings["target_shares"] = (per_stock / holdings["latest_close"]).apply(
            lambda x: max(LOT_SIZE, int(x // LOT_SIZE) * LOT_SIZE)
        )
        holdings["actual_cost"] = holdings["target_shares"] * holdings["latest_close"]
        actual_total = holdings["actual_cost"].sum()
        max_weight = holdings["actual_cost"].max() / actual_total
        min_weight = holdings["actual_cost"].min() / actual_total
        target_weight = 1.0 / n_total
        max_dev = max(abs(max_weight - target_weight), abs(min_weight - target_weight))

        feasible = "✅" if max_dev < 0.02 else ("⚠️" if max_dev < 0.05 else "❌")
        print(f"  目标资金 ¥{target_capital/10000:>6.0f}万 → "
              f"实际 ¥{actual_total/10000:>7.1f}万 | "
              f"权重偏差 ±{max_dev:.2%} | "
              f"每股 ¥{per_stock:>8.0f} {feasible}")

    # ── Print detailed holdings ──
    print(f"\n{'='*64}")
    print(f"📋 持仓明细（按一手金额降序）")
    print(f"{'='*64}")
    print(f"{'序号':>4} {'代码':<12} {'行业':<10} {'最新价':>10} {'一手金额':>12}")
    print(f"{'-'*4} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")
    for i, (_, row) in enumerate(holdings.iterrows(), 1):
        ind_name = str(row.get("industry_name", ""))[:8]
        print(f"{i:>4} {row['code']:<12} {ind_name:<10} "
              f"¥{row['latest_close']:>9.2f} ¥{row['min_cost_per_stock']:>11,.2f}")

    # ── Price distribution by range ──
    print(f"\n{'='*64}")
    print(f"📊 股价分布")
    print(f"{'='*64}")
    bins = [0, 5, 10, 20, 30, 50, 100, 200, 500, float("inf")]
    labels = ["<5", "5-10", "10-20", "20-30", "30-50", "50-100", "100-200", "200-500", ">500"]
    holdings["price_range"] = pd.cut(holdings["latest_close"], bins=bins, labels=labels)
    dist = holdings.groupby("price_range", observed=True).agg(
        count=("code", "count"),
        total_min=("min_cost_per_stock", "sum"),
    )
    for price_range, row in dist.iterrows():
        print(f"  {price_range:>8} 元: {int(row['count']):>3} 只, "
              f"合计一手 ¥{row['total_min']:>12,.2f}")

    print(f"\n✅ 测算完成")


if __name__ == "__main__":
    main()
