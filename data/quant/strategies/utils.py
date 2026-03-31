"""
Shared utilities for LightGBM cross-sectional strategies.
"""
import hashlib
import re
from pathlib import Path
from typing import List

import pandas as pd

from engine.data_loader import DataAccessor


def prefetch_bulk_data(
    accessor: DataAccessor,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Load all stock_daily data for [start_date, end_date] in a single SQL query.

    Results are cached as Parquet files under <db_dir>/.cache/ so that
    subsequent runs with the same date range skip the SQL entirely.
    The cache auto-invalidates when:
      - the DB file is modified (db_mtime changes)
      - feature_columns change (col_hash changes)
      - date range changes (no exact or superset match)

    Parameters
    ----------
    accessor : DataAccessor
        Opened data accessor with a live DB connection.
    start_date, end_date : pd.Timestamp
        Inclusive date range to load.
    feature_columns : list[str]
        DB columns to SELECT (e.g. each strategy's FEATURE_COLUMNS).

    Returns
    -------
    pd.DataFrame
        Sorted by (trade_date, ts_code), trade_date as datetime64.
    """
    s = start_date.strftime("%Y%m%d")
    e = end_date.strftime("%Y%m%d")

    # ── Build cache path ──
    db_path = Path(accessor.cfg.db_path)
    cache_dir = db_path.parent / ".cache"
    db_mtime = int(db_path.stat().st_mtime)
    col_hash = hashlib.md5(",".join(feature_columns).encode()).hexdigest()[:8]
    cache_file = cache_dir / f"bulk_{s}_{e}_{col_hash}_{db_mtime}.parquet"

    # ── 1) Exact match ──
    if cache_file.exists():
        print(f"      [预取] 命中缓存 {cache_file.name}，直接加载 ...")
        df = pd.read_parquet(cache_file)
        print(f"      [预取] ✓ 缓存加载完成: {len(df):,} 行, "
              f"{df['ts_code'].nunique()} 只股票, "
              f"{df['trade_date'].nunique()} 个交易日")
        return df

    # ── 2) Superset match: find a cached file whose range covers [s, e] ──
    if cache_dir.exists():
        pattern = re.compile(
            rf"^bulk_(\d{{8}})_(\d{{8}})_{re.escape(col_hash)}_{db_mtime}\.parquet$"
        )
        for f in cache_dir.iterdir():
            m = pattern.match(f.name)
            if m and m.group(1) <= s and m.group(2) >= e:
                print(f"      [预取] 命中超集缓存 {f.name}，切片 {s}~{e} ...")
                df = pd.read_parquet(f)
                df = df[
                    (df["trade_date"] >= start_date)
                    & (df["trade_date"] <= end_date)
                ].reset_index(drop=True)
                print(f"      [预取] ✓ 缓存切片完成: {len(df):,} 行, "
                      f"{df['ts_code'].nunique()} 只股票, "
                      f"{df['trade_date'].nunique()} 个交易日")
                return df

    # ── Cache miss: query DB ──
    col_sql = ", ".join(feature_columns)
    sql = f"""
        SELECT {col_sql} FROM stock_daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date, ts_code
    """
    print(f"      [预取] 执行批量 SQL 加载 {s} ~ {e} ...")
    df = pd.read_sql_query(sql, accessor.conn, params=(s, e))
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    print(f"      [预取] ✓ 加载完成: {len(df):,} 行, "
          f"{df['ts_code'].nunique()} 只股票, "
          f"{df['trade_date'].nunique()} 个交易日")

    # ── Write cache ──
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False, engine="pyarrow")
        size_mb = cache_file.stat().st_size / 1024 / 1024
        print(f"      [预取] ✓ 已写入缓存 {cache_file.name} ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"      [预取] ⚠ 缓存写入失败 ({exc})，不影响运行")

    return df
