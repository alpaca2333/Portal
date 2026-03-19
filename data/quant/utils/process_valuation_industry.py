"""
Merge valuation & industry data INTO all_stocks_daily.csv
=========================================================

Adds columns: pb, pe_ttm, free_market_cap, industry_code, industry_name

Input:
  - /root/qlib_data/valuation/*.csv        (per-stock: date,symbol,pb,pe_ttm,free_market_cap)
  - /root/qlib_data/sw_industry.csv         (in_date,out_date,symbol,industry_code,industry_name)
  - data/quant/processed/all_stocks_daily.csv (existing: code,date,open,high,low,close,volume,factor)

Output:
  - data/quant/processed/all_stocks_daily.csv  (original + 5 new columns)

Strategy:
  1. Pre-load valuation data into a pandas DataFrame keyed by (symbol, date)
  2. Pre-build industry interval table for bisect lookup
  3. Stream all_stocks_daily.csv in chunks, merge, write to tmp file
  4. Replace original
"""
from __future__ import annotations

import bisect
import glob
import os
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

VALUATION_DIR = Path("/root/qlib_data/valuation")
INDUSTRY_FILE = Path("/root/qlib_data/sw_industry.csv")
ALL_STOCKS_CSV = Path("/projects/portal/data/quant/processed/all_stocks_daily.csv")
TMP_OUTPUT = ALL_STOCKS_CSV.with_suffix(".csv.tmp")

CHUNK_SIZE = 500_000  # rows per chunk when reading all_stocks_daily


# ────────────────────────────────────────────────────────
# Step 1: Load all valuation data into a single DataFrame
# ────────────────────────────────────────────────────────
def load_valuation() -> pd.DataFrame:
    print("=" * 60)
    print("[Step 1] Loading valuation data ...")
    files = sorted(glob.glob(str(VALUATION_DIR / "*.csv")))
    # Only SH/SZ by filename
    sh_sz_files = [f for f in files if ".SH.csv" in f or ".SZ.csv" in f]
    print(f"  Found {len(sh_sz_files)} SH/SZ valuation files")

    chunks: list[pd.DataFrame] = []
    batch = 500
    for i in range(0, len(sh_sz_files), batch):
        batch_chunks = []
        for f in sh_sz_files[i : i + batch]:
            try:
                df = pd.read_csv(
                    f,
                    dtype={"date": str, "symbol": str},
                    usecols=["symbol", "date", "pb", "pe_ttm", "free_market_cap"],
                )
                if not df.empty:
                    batch_chunks.append(df)
            except Exception:
                pass
        if batch_chunks:
            chunks.append(pd.concat(batch_chunks, ignore_index=True))
        done = min(i + batch, len(sh_sz_files))
        print(f"    ... loaded {done}/{len(sh_sz_files)} files")

    val = pd.concat(chunks, ignore_index=True)
    # Normalize symbol: all_stocks_daily uses lowercase like "sz000001"
    # valuation already uses lowercase
    print(f"  Total valuation rows: {len(val):,}")
    print(f"  Unique symbols: {val['symbol'].nunique()}")
    return val


# ────────────────────────────────────────────────────────
# Step 2: Build industry interval lookup
# ────────────────────────────────────────────────────────
def build_industry_lookup() -> dict:
    """
    Returns: dict[symbol] -> list of (in_date_int, out_date_int, industry_code, industry_name)
    sorted by in_date_int, for bisect lookup.
    in_date_int / out_date_int are int like 20060320 for fast comparison.
    """
    print("\n" + "=" * 60)
    print("[Step 2] Building industry lookup ...")
    raw = pd.read_csv(INDUSTRY_FILE, dtype=str)
    raw = raw[raw["symbol"].str.startswith(("sh", "sz"))].copy()
    print(f"  Rows after SH/SZ filter: {len(raw):,}")
    print(f"  Unique symbols: {raw['symbol'].nunique()}")

    def to_int(s):
        try:
            return int(s.replace("-", ""))
        except Exception:
            return None

    lookup: dict[str, list[tuple]] = defaultdict(list)
    for _, row in raw.iterrows():
        sym = row["symbol"]
        in_d = to_int(row["in_date"])
        out_d = to_int(row["out_date"]) if pd.notna(row["out_date"]) and row["out_date"] != "" else 20261231
        if in_d is None:
            continue
        lookup[sym].append((in_d, out_d, row["industry_code"], row["industry_name"]))

    # Sort each symbol's intervals by in_date
    for sym in lookup:
        lookup[sym].sort(key=lambda x: x[0])

    print(f"  Industry lookup built for {len(lookup)} symbols")
    return dict(lookup)


def find_industry(lookup: dict, symbol: str, date_int: int):
    """Find industry for a given symbol and date (as int like 20240101)."""
    intervals = lookup.get(symbol)
    if not intervals:
        return None, None

    # Binary search: find the last interval where in_date <= date_int
    in_dates = [iv[0] for iv in intervals]
    idx = bisect.bisect_right(in_dates, date_int) - 1
    if idx < 0:
        return None, None

    iv = intervals[idx]
    if iv[0] <= date_int <= iv[1]:
        return iv[2], iv[3]  # industry_code, industry_name
    return None, None


# ────────────────────────────────────────────────────────
# Step 3: Stream-merge into all_stocks_daily.csv
# ────────────────────────────────────────────────────────
def merge_into_all_stocks(val_df: pd.DataFrame, ind_lookup: dict) -> None:
    print("\n" + "=" * 60)
    print("[Step 3] Merging into all_stocks_daily.csv ...")

    # Index valuation for fast merge: set (symbol, date) as index
    val_df = val_df.set_index(["symbol", "date"])
    # Remove duplicates (keep last)
    val_df = val_df[~val_df.index.duplicated(keep="last")]
    print(f"  Valuation index built: {len(val_df):,} unique (symbol,date) pairs")

    total_rows = 0
    val_matched = 0
    ind_matched = 0
    first_chunk = True

    reader = pd.read_csv(
        ALL_STOCKS_CSV,
        chunksize=CHUNK_SIZE,
        dtype={"code": str, "date": str},
    )

    for chunk_idx, chunk in enumerate(reader):
        n = len(chunk)
        total_rows += n

        # all_stocks_daily uses "code" column (lowercase like "sz000001" or "bj920000")
        # Valuation uses "symbol" (same format)
        # Create lookup key
        merge_key = chunk["code"].values
        merge_date = chunk["date"].values

        # ── Valuation merge ──
        # Build a MultiIndex for this chunk to do fast reindex
        chunk_idx_pairs = pd.MultiIndex.from_arrays([merge_key, merge_date], names=["symbol", "date"])
        val_slice = val_df.reindex(chunk_idx_pairs)

        chunk["pb"] = val_slice["pb"].values
        chunk["pe_ttm"] = val_slice["pe_ttm"].values
        chunk["free_market_cap"] = val_slice["free_market_cap"].values
        val_matched += chunk["pb"].notna().sum()

        # ── Industry merge (vectorized via apply) ──
        date_ints = np.array([int(d.replace("-", "")) if isinstance(d, str) else 0 for d in merge_date])
        ind_codes = []
        ind_names = []
        for sym, di in zip(merge_key, date_ints):
            ic, iname = find_industry(ind_lookup, sym, di)
            ind_codes.append(ic)
            ind_names.append(iname)

        chunk["industry_code"] = ind_codes
        chunk["industry_name"] = ind_names
        ind_matched += sum(1 for x in ind_codes if x is not None)

        # Write
        chunk.to_csv(
            TMP_OUTPUT,
            index=False,
            float_format="%.6f",
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        print(f"    ... chunk {chunk_idx}: {total_rows:,} rows processed")

    print(f"\n  Total rows: {total_rows:,}")
    print(f"  Valuation matched: {val_matched:,} ({val_matched / total_rows:.1%})")
    print(f"  Industry matched: {ind_matched:,} ({ind_matched / total_rows:.1%})")

    # Replace original
    backup = ALL_STOCKS_CSV.with_suffix(".csv.bak")
    print(f"\n  Backing up original to {backup.name} ...")
    shutil.move(str(ALL_STOCKS_CSV), str(backup))
    shutil.move(str(TMP_OUTPUT), str(ALL_STOCKS_CSV))
    print(f"  Original replaced. Backup at: {backup}")

    size_mb = ALL_STOCKS_CSV.stat().st_size / 1024 / 1024
    print(f"  New file size: {size_mb:.1f} MB")


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    val_df = load_valuation()
    ind_lookup = build_industry_lookup()
    merge_into_all_stocks(val_df, ind_lookup)

    # Verify
    print("\n" + "=" * 60)
    print("[Verify] Checking output ...")
    sample = pd.read_csv(ALL_STOCKS_CSV, nrows=5)
    print(f"  Columns: {list(sample.columns)}")
    print(sample.to_string(index=False))

    # Also save standalone files for reference
    print("\n  Also saving standalone valuation & industry files ...")
    val_df.reset_index().to_csv(
        ALL_STOCKS_CSV.parent / "valuation_daily.csv", index=False, float_format="%.6f"
    )
    # Save industry range table
    raw_ind = pd.read_csv(INDUSTRY_FILE, dtype=str)
    raw_ind = raw_ind[raw_ind["symbol"].str.startswith(("sh", "sz"))]
    raw_ind.to_csv(ALL_STOCKS_CSV.parent / "sw_industry_range.csv", index=False)

    print("\n" + "=" * 60)
    print("All done!")
    print(f"  Updated: {ALL_STOCKS_CSV}")
    print(f"  Standalone: valuation_daily.csv, sw_industry_range.csv")
    print("=" * 60)
