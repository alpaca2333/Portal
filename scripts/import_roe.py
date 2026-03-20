#!/usr/bin/env python3
"""
import_roe.py
=============
Merge ROE data into all_stocks_daily chunk files and stocks.db.

ROE source: /root/qlib_data/roe/{TICKER}.{EXCHANGE}.csv
  - Quarterly frequency, keyed by announcement date (date column)
  - Fields: date, end_date, symbol, roe, roe_ttm, roe_deducted
  - symbol format: sz000001 (lowercase)

Target:
  1. Chunk files: /projects/portal/data/quant/processed/all_stocks_daily_chunk{0,1,2}.csv
     -> Add roe_ttm column (forward-filled from announcement date)
  2. SQLite: /projects/portal/data/quant/processed/stocks.db  kline table
     -> Add roe_ttm column

Strategy: use announcement date (date) as the point-in-time marker,
forward-fill roe_ttm to each trading day to avoid look-ahead bias.

Usage:
    python scripts/import_roe.py
"""
from __future__ import annotations
import os, sys, glob, sqlite3, time
import pandas as pd
import numpy as np
from pathlib import Path

ROE_DIR   = "/root/qlib_data/roe"
CHUNK_DIR = "/projects/portal/data/quant/processed"
DB_PATH   = os.path.join(CHUNK_DIR, "stocks.db")
CHUNK_PATTERN = os.path.join(CHUNK_DIR, "all_stocks_daily_chunk*.csv")


def load_all_roe() -> pd.DataFrame:
    """Load all ROE CSVs and build a unified DataFrame."""
    files = sorted(glob.glob(os.path.join(ROE_DIR, "*.csv")))
    print(f"[roe] Found {len(files)} ROE files")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str)
            if df.empty or "roe_ttm" not in df.columns:
                continue
            frames.append(df[["date", "symbol", "roe_ttm"]])
        except Exception as e:
            print(f"  [warn] Skipping {f}: {e}")
            continue

    roe = pd.concat(frames, ignore_index=True)
    roe["date"] = pd.to_datetime(roe["date"])
    roe["roe_ttm"] = pd.to_numeric(roe["roe_ttm"], errors="coerce")
    # symbol is lowercase like "sz000001"
    # chunk files use lowercase, SQLite uses uppercase
    # Keep lowercase as canonical, convert when needed
    roe["code"] = roe["symbol"].str.lower()
    roe = roe.drop(columns=["symbol"])
    roe = roe.dropna(subset=["roe_ttm"])
    roe = roe.sort_values(["code", "date"]).drop_duplicates(subset=["code", "date"], keep="last")
    print(f"[roe] Loaded {len(roe):,} valid ROE observations for {roe['code'].nunique()} stocks")
    return roe


def forward_fill_roe_for_chunk(chunk_df: pd.DataFrame, roe_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stock in chunk_df, forward-fill roe_ttm from announcement dates.
    This is a point-in-time merge: on any trading day, the roe_ttm value is
    the most recent ROE that was publicly announced on or before that date.
    """
    chunk_df = chunk_df.copy()
    chunk_df["_date"] = pd.to_datetime(chunk_df["date"])

    # Build a lookup: for each code, sorted list of (announce_date, roe_ttm)
    roe_lookup = {}
    for code, grp in roe_df.groupby("code"):
        roe_lookup[code] = grp[["date", "roe_ttm"]].sort_values("date").values

    # Vectorized approach: merge_asof per code group
    # But with 22M rows and 5K stocks, we use merge_asof which is efficient
    chunk_df = chunk_df.sort_values(["code", "_date"])
    roe_df_sorted = roe_df.sort_values(["code", "date"])

    # merge_asof needs both sorted by the key
    results = []
    codes = chunk_df["code"].unique()
    total = len(codes)
    t0 = time.time()

    for i, code in enumerate(codes):
        stock_daily = chunk_df[chunk_df["code"] == code][["_date"]].copy()
        if code in roe_lookup:
            roe_vals = roe_lookup[code]
            # For each daily date, find the latest ROE announcement <= that date
            roe_series = pd.DataFrame(roe_vals, columns=["ann_date", "roe_ttm"])
            roe_series["ann_date"] = pd.to_datetime(roe_series["ann_date"])
            merged = pd.merge_asof(
                stock_daily.sort_values("_date"),
                roe_series.sort_values("ann_date"),
                left_on="_date",
                right_on="ann_date",
                direction="backward"
            )
            results.append(merged.set_index(stock_daily.index)["roe_ttm"])
        else:
            results.append(pd.Series(np.nan, index=stock_daily.index, name="roe_ttm"))

        if (i + 1) % 500 == 0 or i + 1 == total:
            elapsed = time.time() - t0
            print(f"\r  [ffill] {i+1}/{total} stocks ({elapsed:.0f}s)", end="", flush=True)

    print()
    chunk_df["roe_ttm"] = pd.concat(results)
    chunk_df = chunk_df.drop(columns=["_date"])
    return chunk_df


def process_chunks(roe_df: pd.DataFrame):
    """Add roe_ttm column to each chunk file."""
    chunk_files = sorted(glob.glob(CHUNK_PATTERN))
    print(f"\n[chunks] Found {len(chunk_files)} chunk files")

    for cfile in chunk_files:
        print(f"\n[chunk] Processing {os.path.basename(cfile)} ...")
        t0 = time.time()
        chunk = pd.read_csv(cfile, dtype=str)
        nrows = len(chunk)
        print(f"  Loaded {nrows:,} rows")

        # Remove existing roe_ttm column if present (idempotent)
        if "roe_ttm" in chunk.columns:
            chunk = chunk.drop(columns=["roe_ttm"])
            print("  Dropped existing roe_ttm column")

        # Ensure lowercase for matching (chunk files already lowercase)
        chunk["code"] = chunk["code"].str.lower()

        # Forward-fill ROE
        chunk = forward_fill_roe_for_chunk(chunk, roe_df)

        # Save
        chunk.to_csv(cfile, index=False)
        elapsed = time.time() - t0
        filled = chunk["roe_ttm"].notna().sum() if "roe_ttm" in chunk.columns else 0
        fill_pct = filled / nrows * 100 if nrows > 0 else 0
        print(f"  Saved. ROE coverage: {filled:,}/{nrows:,} ({fill_pct:.1f}%) in {elapsed:.0f}s")


def update_sqlite(roe_df: pd.DataFrame):
    """Add roe_ttm column to SQLite kline table."""
    print(f"\n[sqlite] Updating {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check if roe_ttm column already exists
    cols = [row[1] for row in cur.execute("PRAGMA table_info(kline)").fetchall()]
    if "roe_ttm" not in cols:
        print("  Adding roe_ttm column ...")
        cur.execute("ALTER TABLE kline ADD COLUMN roe_ttm REAL")
        conn.commit()
    else:
        print("  roe_ttm column already exists, will update values")

    # Strategy: for each stock, get all trading dates from kline,
    # then forward-fill ROE and batch update.
    # This is more efficient than row-by-row.

    # SQLite stores codes in UPPERCASE
    roe_upper = roe_df.copy()
    roe_upper["code"] = roe_upper["code"].str.upper()
    codes = roe_upper["code"].unique().tolist()
    print(f"  Updating ROE for {len(codes)} stocks ...")

    # Prepare batch update statement
    update_stmt = "UPDATE kline SET roe_ttm = ? WHERE code = ? AND date = ?"

    t0 = time.time()
    total_updated = 0

    for i, code in enumerate(codes):
        # Get trading dates for this stock
        rows = cur.execute(
            "SELECT date FROM kline WHERE code = ? ORDER BY date", (code,)
        ).fetchall()
        if not rows:
            continue

        dates = pd.DataFrame(rows, columns=["date"])
        dates["_date"] = pd.to_datetime(dates["date"])

        # Get ROE announcements for this stock (uppercase)
        stock_roe = roe_upper[roe_upper["code"] == code][["date", "roe_ttm"]].copy()
        stock_roe = stock_roe.rename(columns={"date": "ann_date"})
        stock_roe["ann_date"] = pd.to_datetime(stock_roe["ann_date"])
        stock_roe = stock_roe.sort_values("ann_date")

        if stock_roe.empty:
            continue

        # merge_asof
        merged = pd.merge_asof(
            dates.sort_values("_date"),
            stock_roe,
            left_on="_date",
            right_on="ann_date",
            direction="backward"
        )

        # Only update rows with non-null roe_ttm
        valid = merged.dropna(subset=["roe_ttm"])
        if valid.empty:
            continue

        updates = [(float(row["roe_ttm"]), code, row["date"]) for _, row in valid.iterrows()]

        cur.executemany(update_stmt, updates)
        total_updated += len(updates)

        if (i + 1) % 500 == 0 or (i + 1) == len(codes):
            conn.commit()
            elapsed = time.time() - t0
            print(f"\r  [sqlite] {i+1}/{len(codes)} stocks, {total_updated:,} rows updated ({elapsed:.0f}s)", end="", flush=True)

    conn.commit()
    print(f"\n  Total rows updated: {total_updated:,}")

    # Verify
    sample = cur.execute(
        "SELECT code, date, roe_ttm FROM kline WHERE roe_ttm IS NOT NULL ORDER BY date DESC LIMIT 5"
    ).fetchall()
    total_with_roe = cur.execute("SELECT COUNT(*) FROM kline WHERE roe_ttm IS NOT NULL").fetchone()[0]
    total_rows = cur.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
    pct = total_with_roe / total_rows * 100 if total_rows > 0 else 0

    print(f"  Verification: {total_with_roe:,}/{total_rows:,} rows have ROE ({pct:.1f}%)")
    print(f"  Sample rows:")
    for row in sample:
        print(f"    {row[0]} | {row[1]} | roe_ttm={row[2]}")

    conn.close()
    print("[sqlite] Done.")


if __name__ == "__main__":
    print("=" * 64)
    print("ROE Data Import Script")
    print("=" * 64)

    print("\n[1/3] Loading all ROE data ...")
    roe = load_all_roe()

    print("\n[2/3] Merging into chunk files ...")
    process_chunks(roe)

    print("\n[3/3] Updating SQLite database ...")
    update_sqlite(roe)

    print("\n✅ All done!")
