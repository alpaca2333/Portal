"""
Build a single SQLite database from all downloaded CSV data.

Tables:
  1. stock_daily   — wide table: one row per (ts_code, trade_date)
                     price + daily_basic + adj_factor + fina_indicator + industry
  2. industry_info — industry_code -> industry_name (L1 & L2)
  3. stock_info    — ts_code -> name / list_date / delist_date / area / market

Point-in-time rules (avoid look-ahead bias):
  - Financial indicators: effective from the NEXT trading day after ann_date.
    (announcement is published after market close on ann_date)
  - Industry membership: effective from in_date, expires on out_date.

Usage:
  python scripts/build_db.py                 # full rebuild
  python scripts/build_db.py --since 20260301  # incremental from date
"""

import os
import sys
import glob
import argparse
import sqlite3

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DB_PATH = os.path.join(DATA_DIR, "quant.db")

DAILY_DIR = os.path.join(DATA_DIR, "daily")
DAILY_BASIC_DIR = os.path.join(DATA_DIR, "daily_basic")
ADJ_FACTOR_DIR = os.path.join(DATA_DIR, "adj_factor")
FINA_DIR = os.path.join(DATA_DIR, "fina_indicator")
INDUSTRY_DIR = os.path.join(DATA_DIR, "industry")
BASIC_PATH = os.path.join(DATA_DIR, "stock_basic", "stock_basic.csv")
CALENDAR_PATH = os.path.join(DATA_DIR, "calendar", "trade_cal.csv")
SUSPEND_PATH = os.path.join(DATA_DIR, "suspend", "suspend_d.csv")

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="将CSV原始数据导入SQLite")
parser.add_argument("--since", type=str, default=None,
                    help="只导入该日期(YYYYMMDD)之后的交易日数据 (增量)")
args = parser.parse_args()

SINCE = args.since  # None means full rebuild


# ─── helpers ─────────────────────────────────────────────────────────
def load_trade_calendar() -> pd.Series:
    """Return sorted Series of trading dates (str, YYYYMMDD)."""
    cal = pd.read_csv(CALENDAR_PATH, dtype=str)
    cal = cal[cal["is_open"] == "1"]
    return cal["cal_date"].sort_values().reset_index(drop=True)


def next_trade_date_map(trade_dates: pd.Series) -> dict:
    """Build a dict: date_str -> next_trading_date_str.
    For any date (including non-trading days), map it to the next trading day.
    """
    td_set = set(trade_dates.tolist())
    td_sorted = sorted(td_set)
    # Build mapping for all calendar days from min to max trade date
    from datetime import datetime, timedelta
    min_d = datetime.strptime(td_sorted[0], "%Y%m%d")
    max_d = datetime.strptime(td_sorted[-1], "%Y%m%d")

    # For efficiency, build a reverse index: for each trade date, all preceding
    # non-trade dates also map to this trade date's next.
    # Simpler: iterate trade dates and build next-trade-date lookup.
    next_td = {}
    for i in range(len(td_sorted) - 1):
        next_td[td_sorted[i]] = td_sorted[i + 1]
    # Last trade date has no next
    next_td[td_sorted[-1]] = None

    # For non-trading days, find the next trading day >= day+1
    result = {}
    td_list = td_sorted  # sorted trading dates
    ptr = 0
    d = min_d
    while d <= max_d:
        ds = d.strftime("%Y%m%d")
        # Advance ptr to first trade date > ds
        while ptr < len(td_list) and td_list[ptr] <= ds:
            ptr += 1
        if ptr < len(td_list):
            result[ds] = td_list[ptr]
        else:
            result[ds] = None
        d += timedelta(days=1)

    return result


def load_industry_membership():
    """Load L1 and L2 industry membership with time ranges.
    Returns DataFrame: con_code, industry_code, level, in_date, out_date
    """
    frames = []
    for fname, level in [("sw_industry_member.csv", "L1"),
                          ("sw_industry_member_l2.csv", "L2")]:
        path = os.path.join(INDUSTRY_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str)
        df = df.rename(columns={"con_code": "ts_code"})
        df["level"] = level
        # Use index_code as the industry identifier (not industry_code which is a different column)
        df = df[["ts_code", "index_code", "level", "in_date", "out_date"]]
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_industry_on_date(ind_df: pd.DataFrame, ts_code: str, trade_dates):
    """For a given stock, return a DataFrame with (trade_date, sw_l1, sw_l2).
    Respects in_date/out_date for point-in-time correctness.
    """
    sub = ind_df[ind_df["ts_code"] == ts_code]
    if sub.empty:
        return pd.DataFrame({"trade_date": trade_dates, "sw_l1": None, "sw_l2": None})

    result = pd.DataFrame({"trade_date": trade_dates})
    for level, col in [("L1", "sw_l1"), ("L2", "sw_l2")]:
        level_rows = sub[sub["level"] == level]
        result[col] = None
        for _, row in level_rows.iterrows():
            in_d = row["in_date"] if pd.notna(row["in_date"]) else "00000000"
            out_d = row["out_date"] if pd.notna(row["out_date"]) else "99991231"
            mask = (result["trade_date"] >= in_d) & (result["trade_date"] < out_d)
            result.loc[mask, col] = row["index_code"]
    return result


def align_fina_pit(fina_df: pd.DataFrame, trade_dates, ntd_map: dict):
    """Point-in-time align financial indicators.
    Each record becomes effective from the NEXT trading day after ann_date.
    Forward-fill until the next announcement.

    Args:
        fina_df: fina_indicator data for one stock (sorted by ann_date, end_date)
        trade_dates: sorted list of trade dates for this stock
        ntd_map: date -> next_trade_date mapping

    Returns:
        DataFrame indexed by trade_date with fina columns.
    """
    if fina_df.empty:
        return pd.DataFrame(index=trade_dates)

    fina_cols = [c for c in fina_df.columns
                 if c not in ("ts_code", "ann_date", "end_date", "update_flag")]

    # Determine effective date = next trading day after ann_date
    fina_df = fina_df.copy()
    fina_df["eff_date"] = fina_df["ann_date"].map(ntd_map)
    fina_df = fina_df.dropna(subset=["eff_date"])

    if fina_df.empty:
        return pd.DataFrame(index=trade_dates)

    # If multiple records on same eff_date, keep the one with the latest end_date
    fina_df = fina_df.sort_values(["eff_date", "end_date"])
    fina_df = fina_df.drop_duplicates(subset=["eff_date"], keep="last")

    # Build a time series indexed by trade_date, forward-fill
    fina_df = fina_df.set_index("eff_date")[fina_cols]

    # For incremental mode: if all eff_dates are before trade_dates[0],
    # we still need the latest record before trade_dates[0] as seed for ffill.
    first_td = trade_dates[0]
    prior = fina_df[fina_df.index < first_td]
    if not prior.empty:
        # Insert the last known record at a synthetic key just before first_td
        # so that ffill can propagate it forward.
        seed = prior.iloc[[-1]].copy()
        seed.index = [first_td]  # place it at first_td so ffill picks it up
        fina_df = pd.concat([seed, fina_df[fina_df.index >= first_td]])

    # Ensure no duplicate index values (keep last for most recent data)
    fina_df = fina_df[~fina_df.index.duplicated(keep="last")]

    td_df = pd.DataFrame(index=trade_dates)
    td_df = td_df.join(fina_df, how="left")
    td_df = td_df.ffill()

    return td_df


# ─── main ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CSV → SQLite 数据导入")
    print("=" * 60)

    if SINCE:
        print(f"  增量模式: 只处理 trade_date >= {SINCE}")
    else:
        print(f"  全量重建模式")
    print(f"  数据库路径: {DB_PATH}")
    print()

    # ── 1. Load trade calendar & build next-trade-date map ──
    print("加载交易日历 ...")
    trade_dates_all = load_trade_calendar()
    ntd_map = next_trade_date_map(trade_dates_all)
    trade_dates_set = set(trade_dates_all.tolist())

    # ── 2. Load industry membership ──
    print("加载行业分类 ...")
    ind_df = load_industry_membership()

    # ── 3. Load stock list ──
    print("加载股票列表 ...")
    stock_basic = pd.read_csv(BASIC_PATH, dtype=str)
    ts_codes = sorted(stock_basic["ts_code"].tolist())

    # ── 4. Load suspend data ──
    print("加载停牌数据 ...")
    suspend_set = set()
    if os.path.exists(SUSPEND_PATH):
        susp = pd.read_csv(SUSPEND_PATH, dtype=str)
        if "ts_code" in susp.columns and "trade_date" in susp.columns:
            suspend_set = set(zip(susp["ts_code"], susp["trade_date"]))

    # ── 5. Open DB connection ──
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-200000")  # 200MB cache

    # ── 6. Create tables ──
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stock_daily (
            ts_code       TEXT    NOT NULL,
            trade_date    TEXT    NOT NULL,
            -- price
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            pre_close     REAL,
            change        REAL,
            pct_chg       REAL,
            vol           REAL,
            amount        REAL,
            -- daily basic
            turnover_rate   REAL,
            turnover_rate_f REAL,
            volume_ratio    REAL,
            pe              REAL,
            pe_ttm          REAL,
            pb              REAL,
            ps              REAL,
            ps_ttm          REAL,
            dv_ratio        REAL,
            dv_ttm          REAL,
            free_share      REAL,
            total_share     REAL,
            float_share     REAL,
            total_mv        REAL,
            circ_mv         REAL,
            -- adj factor
            adj_factor    REAL,
            -- financial indicators (point-in-time)
            eps           REAL,
            bps           REAL,
            cfps          REAL,
            revenue_ps    REAL,
            roe           REAL,
            roe_dt        REAL,
            roe_waa       REAL,
            roe_yearly    REAL,
            roa           REAL,
            roa_yearly    REAL,
            grossprofit_margin  REAL,
            netprofit_margin    REAL,
            profit_to_gr  REAL,
            debt_to_assets REAL,
            current_ratio REAL,
            quick_ratio   REAL,
            inv_turn      REAL,
            ar_turn       REAL,
            ca_turn       REAL,
            fa_turn       REAL,
            assets_turn   REAL,
            op_yoy        REAL,
            ebt_yoy       REAL,
            tr_yoy        REAL,
            or_yoy        REAL,
            equity_yoy    REAL,
            -- industry (point-in-time)
            sw_l1         TEXT,
            sw_l2         TEXT,
            -- suspend flag
            is_suspended  INTEGER DEFAULT 0,
            PRIMARY KEY (ts_code, trade_date)
        );

        CREATE TABLE IF NOT EXISTS industry_info (
            industry_code TEXT PRIMARY KEY,
            industry_name TEXT,
            level         TEXT,
            parent_code   TEXT
        );

        CREATE TABLE IF NOT EXISTS stock_info (
            ts_code     TEXT PRIMARY KEY,
            name        TEXT,
            area        TEXT,
            industry    TEXT,
            market      TEXT,
            list_date   TEXT,
            list_status TEXT,
            delist_date TEXT
        );
    """)

    # ── 7. Populate industry_info ──
    print("写入 industry_info ...")
    ind_index_path = os.path.join(INDUSTRY_DIR, "sw_industry_index.csv")
    if os.path.exists(ind_index_path):
        ind_index = pd.read_csv(ind_index_path, dtype=str)
        conn.execute("DELETE FROM industry_info")
        for _, row in ind_index.iterrows():
            conn.execute(
                "INSERT OR REPLACE INTO industry_info VALUES (?,?,?,?)",
                (row.get("index_code"), row.get("industry_name"),
                 row.get("level"), row.get("parent_code"))
            )
    conn.commit()

    # ── 8. Populate stock_info ──
    print("写入 stock_info ...")
    conn.execute("DELETE FROM stock_info")
    for _, row in stock_basic.iterrows():
        conn.execute(
            "INSERT OR REPLACE INTO stock_info VALUES (?,?,?,?,?,?,?,?)",
            (row.get("ts_code"), row.get("name"), row.get("area"),
             row.get("industry"), row.get("market"), row.get("list_date"),
             row.get("list_status"),
             row.get("delist_date") if pd.notna(row.get("delist_date")) else None)
        )
    conn.commit()

    # ── 9. Populate stock_daily (per stock) ──
    print(f"\n写入 stock_daily ({len(ts_codes)} 只股票) ...")

    # If incremental, only delete rows >= since for each stock
    if SINCE:
        conn.execute("DELETE FROM stock_daily WHERE trade_date >= ?", (SINCE,))
        conn.commit()

    failed = []
    for ts_code in tqdm(ts_codes, desc="构建宽表", unit="只", ncols=80):
        try:
            _build_one_stock(conn, ts_code, ind_df, ntd_map, suspend_set)
        except Exception as e:
            import traceback
            tqdm.write(f"  [错误] {ts_code}: {e}")
            tqdm.write(traceback.format_exc())
            failed.append((ts_code, str(e)))
            if len(failed) == 1:
                # Only print full traceback for first failure
                pass

    conn.commit()

    # ── 10. Create indexes ──
    print("\n创建索引 ...")
    conn.executescript("""
CREATE INDEX IF NOT EXISTS idx_sd_date      ON stock_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_sd_code      ON stock_daily(ts_code);
CREATE INDEX IF NOT EXISTS idx_sd_l1        ON stock_daily(sw_l1);
CREATE INDEX IF NOT EXISTS idx_sd_date_code ON stock_daily(trade_date, ts_code);
-- Covering index for window queries (e.g. Momentum factor): enables
-- index-only scan on (trade_date, ts_code, close) without table lookback.
CREATE INDEX IF NOT EXISTS idx_sd_date_close ON stock_daily(trade_date, ts_code, close);
    """)
    conn.commit()
    conn.close()

    print()
    if failed:
        print(f"失败 {len(failed)} 只: {[f[0] for f in failed[:20]]}")
    print(f"数据库已生成 -> {DB_PATH}")
    # Show file size
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"文件大小: {size_mb:.1f} MB")


def _build_one_stock(conn, ts_code, ind_df, ntd_map, suspend_set):
    """Build and insert stock_daily rows for one stock."""

    # ── Load daily (price) ──
    daily_path = os.path.join(DAILY_DIR, f"{ts_code}.csv")
    if not os.path.exists(daily_path):
        return
    df_daily = pd.read_csv(daily_path, dtype={"trade_date": str, "ts_code": str})
    if df_daily.empty:
        return

    # Filter by --since
    if SINCE:
        df_daily = df_daily[df_daily["trade_date"] >= SINCE]
    if df_daily.empty:
        return

    df_daily = df_daily.sort_values("trade_date").reset_index(drop=True)
    trade_dates = df_daily["trade_date"].tolist()
    df_daily = df_daily.set_index("trade_date")

    # ── Load daily_basic ──
    db_path = os.path.join(DAILY_BASIC_DIR, f"{ts_code}.csv")
    df_db = pd.DataFrame(index=trade_dates)
    if os.path.exists(db_path):
        tmp = pd.read_csv(db_path, dtype={"trade_date": str, "ts_code": str})
        tmp = tmp.set_index("trade_date")
        tmp = tmp.drop(columns=["ts_code", "close"], errors="ignore")
        df_db = df_db.join(tmp, how="left")

    # ── Load adj_factor ──
    adj_path = os.path.join(ADJ_FACTOR_DIR, f"{ts_code}.csv")
    df_adj = pd.DataFrame(index=trade_dates)
    if os.path.exists(adj_path):
        tmp = pd.read_csv(adj_path, dtype={"trade_date": str, "ts_code": str})
        tmp = tmp.set_index("trade_date")
        tmp = tmp.drop(columns=["ts_code"], errors="ignore")
        df_adj = df_adj.join(tmp, how="left")

    # ── Load fina_indicator (point-in-time) ──
    fina_path = os.path.join(FINA_DIR, f"{ts_code}.csv")
    df_fina_pit = pd.DataFrame(index=trade_dates)
    if os.path.exists(fina_path):
        fina_raw = pd.read_csv(fina_path, dtype={"ann_date": str, "end_date": str,
                                                   "ts_code": str, "update_flag": str})
        # Keep only records with update_flag == '1' (latest revision) when available
        if "update_flag" in fina_raw.columns:
            has_flag = fina_raw[fina_raw["update_flag"] == "1"]
            no_flag = fina_raw[~fina_raw["end_date"].isin(has_flag["end_date"])]
            fina_raw = pd.concat([has_flag, no_flag], ignore_index=True)

        fina_raw = fina_raw.dropna(subset=["ann_date"])
        fina_raw = fina_raw.sort_values(["ann_date", "end_date"])
        df_fina_pit = align_fina_pit(fina_raw, trade_dates, ntd_map)

    # ── Industry (point-in-time) ──
    df_ind = get_industry_on_date(ind_df, ts_code, trade_dates)
    df_ind = df_ind.set_index("trade_date")

    # ── Suspend flag ──
    is_suspended = [1 if (ts_code, td) in suspend_set else 0 for td in trade_dates]

    # ── Merge all ──
    result = df_daily.drop(columns=["ts_code"], errors="ignore").copy()

    # Join daily_basic columns
    for col in df_db.columns:
        if col not in result.columns:
            result[col] = df_db[col]

    # Join adj_factor
    if "adj_factor" in df_adj.columns:
        result["adj_factor"] = df_adj["adj_factor"]

    # Join fina indicators
    for col in df_fina_pit.columns:
        result[col] = df_fina_pit[col]

    # Join industry
    for col in ["sw_l1", "sw_l2"]:
        if col in df_ind.columns:
            result[col] = df_ind[col]

    result["is_suspended"] = is_suspended
    result["ts_code"] = ts_code
    result = result.reset_index().rename(columns={"index": "trade_date"})

    # ── Ensure column order matches table schema ──
    schema_cols = [
        "ts_code", "trade_date",
        "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount",
        "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm",
        "free_share", "total_share", "float_share", "total_mv", "circ_mv",
        "adj_factor",
        "eps", "bps", "cfps", "revenue_ps",
        "roe", "roe_dt", "roe_waa", "roe_yearly", "roa", "roa_yearly",
        "grossprofit_margin", "netprofit_margin", "profit_to_gr",
        "debt_to_assets", "current_ratio", "quick_ratio",
        "inv_turn", "ar_turn", "ca_turn", "fa_turn", "assets_turn",
        "op_yoy", "ebt_yoy", "tr_yoy", "or_yoy", "equity_yoy",
        "sw_l1", "sw_l2",
        "is_suspended",
    ]

    # Add missing columns as NaN
    for col in schema_cols:
        if col not in result.columns:
            result[col] = None

    result = result[schema_cols]

    # Replace NaN with None for SQLite
    result = result.where(result.notna(), other=None)

    # ── Insert into DB ──
    placeholders = ",".join(["?"] * len(schema_cols))
    sql = f"INSERT OR REPLACE INTO stock_daily ({','.join(schema_cols)}) VALUES ({placeholders})"
    rows = result.values.tolist()
    conn.executemany(sql, rows)


if __name__ == "__main__":
    main()
