"""Temp perf check script — profile one rebalance period."""
import time
import sqlite3
import pandas as pd
import numpy as np

DB = "data/quant/data/quant.db"
DATE = "20210331"

conn = sqlite3.connect(DB)

# 1. Check indexes
print("=== Indexes on stock_daily ===")
rows = conn.execute(
    "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='stock_daily'"
).fetchall()
for name, sql in rows:
    print(f"  {name}: {sql}")
if not rows:
    print("  (none)")

# 2. get_prices
t0 = time.perf_counter()
cur = conn.execute("SELECT ts_code, close FROM stock_daily WHERE trade_date=?", (DATE,))
prices = {r[0]: r[1] for r in cur.fetchall() if r[1] is not None}
t1 = time.perf_counter()
print(f"\n[get_prices]          {(t1-t0)*1000:>8.1f} ms  ({len(prices)} stocks)")

# 3. get_date (CrossSectionalFactor needs this)
t2 = time.perf_counter()
df_snap = pd.read_sql_query(
    "SELECT ts_code, trade_date, close, pb, roe, sw_l1, is_suspended, circ_mv "
    "FROM stock_daily WHERE trade_date=?",
    conn, params=(DATE,)
)
t3 = time.perf_counter()
print(f"[get_date(snap)]      {(t3-t2)*1000:>8.1f} ms  ({len(df_snap)} rows)")

# 4. get_window for Momentum (21 dates)
t4 = time.perf_counter()
date_rows = conn.execute(
    "SELECT DISTINCT trade_date FROM stock_daily WHERE trade_date<=? ORDER BY trade_date DESC LIMIT ?",
    (DATE, 21)
).fetchall()
date_list = [r[0] for r in date_rows]
ph = ",".join(["?"] * len(date_list))
df_win = pd.read_sql_query(
    f"SELECT ts_code, trade_date, close FROM stock_daily WHERE trade_date IN ({ph})",
    conn, params=date_list
)
t5 = time.perf_counter()
print(f"[get_window(21d)]     {(t5-t4)*1000:>8.1f} ms  ({len(df_win)} rows)")

# 5. pivot for momentum
t6 = time.perf_counter()
pivot = df_win.pivot(index="trade_date", columns="ts_code", values="close")
ret = pivot.iloc[-1] / pivot.iloc[0] - 1
t7 = time.perf_counter()
print(f"[momentum pivot+calc] {(t7-t6)*1000:>8.1f} ms")

# 6. FactorEngine.run overhead: get_date called AGAIN for universe filter
t8 = time.perf_counter()
df_snap2 = pd.read_sql_query(
    "SELECT ts_code, trade_date, close, pb, roe, sw_l1, is_suspended, circ_mv, "
    "pe_ttm, ps_ttm, dv_ttm, cfps, roa, grossprofit_margin, current_ratio, "
    "debt_to_assets, assets_turn, tr_yoy, op_yoy, equity_yoy, ebt_yoy, "
    "turnover_rate_f, pct_chg, amount, vol "
    "FROM stock_daily WHERE trade_date=?",
    conn, params=(DATE,)
)
t9 = time.perf_counter()
print(f"[get_date(all cols)]  {(t9-t8)*1000:>8.1f} ms  ({len(df_snap2)} rows)")

# 7. Total for a typical FactorEngine.run (3 factors: BP, ROE, Momentum)
# CrossSectional factors each call get_date individually
# Then FactorEngine.run calls get_date AGAIN for universe filter
print("\n=== Estimated total per period ===")
cs_time = (t3 - t2) * 2  # BP + ROE each do get_date
ts_time = (t5 - t4)       # Momentum does get_window
engine_time = (t9 - t8)   # FactorEngine does another get_date
prices_time = (t1 - t0)   # get_prices in main loop
calc_time = (t7 - t6)     # pivot + calc
total = cs_time + ts_time + engine_time + prices_time + calc_time
print(f"  get_prices:          {prices_time*1000:>8.1f} ms")
print(f"  CS factors (x2):     {cs_time*1000:>8.1f} ms  <-- BP, ROE each query DB")
print(f"  TS factor (window):  {ts_time*1000:>8.1f} ms  <-- Momentum queries 21 dates")
print(f"  Engine get_date:     {engine_time*1000:>8.1f} ms  <-- universe filter")
print(f"  Pivot + calc:        {calc_time*1000:>8.1f} ms")
print(f"  ──────────────────────────────")
print(f"  TOTAL estimate:      {total*1000:>8.1f} ms")

# Count how many get_date calls happen
print("\n=== DB query count per period ===")
print("  1x get_prices (main loop)")
print("  1x get_date per CrossSectionalFactor (BP=1, ROE=1)")
print("  1x get_window per TimeSeriesFactor (Momentum=1)")
print("  1x get_date in FactorEngine.run (universe filter)")
print("  = 4-5 SQL queries per period!")

conn.close()
