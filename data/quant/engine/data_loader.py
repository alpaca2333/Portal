"""
Data loading utilities — lazy, on-demand access from SQLite and CSV.

Design: no full-table load. Instead, ``DataAccessor`` keeps a single
SQLite connection open and serves per-date / per-window queries so that
the backtest engine runs comfortably on low-memory machines.
"""
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig


# ---------------------------------------------------------------------------
# Look-ahead bias guard
# ---------------------------------------------------------------------------

class LookAheadError(Exception):
    """Raised when a query attempts to access data beyond the current date."""
    pass


# ---------------------------------------------------------------------------
# Trade calendar (lightweight — only date strings, always safe to load)
# ---------------------------------------------------------------------------

def load_trade_calendar(cfg: BacktestConfig) -> pd.DatetimeIndex:
    """Return sorted DatetimeIndex of all trade dates within [start, end]."""
    conn = sqlite3.connect(cfg.db_path)
    sql = """
        SELECT DISTINCT trade_date FROM stock_daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """
    df = pd.read_sql_query(sql, conn,
                           params=(cfg.start_date.replace("-", ""),
                                   cfg.end_date.replace("-", "")))
    conn.close()
    dates = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return pd.DatetimeIndex(dates)


def get_rebalance_dates(trade_dates: pd.DatetimeIndex,
                        freq: str) -> pd.DatetimeIndex:
    """
    Pick rebalance dates from the full calendar.

    freq: "M" month-end, "W" week-end (Fri), "Q" quarter-end, "D" daily,
          "BW" bi-weekly (every 2 weeks, last trade date of each 2-week block).
    """
    if freq == "D":
        return trade_dates

    # Bi-weekly: every 2 weeks, pick the last trade date of each 2-week block
    if freq == "BW":
        if len(trade_dates) == 0:
            return trade_dates
        origin = trade_dates[0]
        day_offsets = (trade_dates - origin).days
        block_ids = day_offsets // 14  # 14-day blocks
        s = pd.Series(trade_dates, index=trade_dates)
        groups = s.groupby(block_ids)
        last_dates = groups.last()
        return pd.DatetimeIndex(last_dates.values)

    s = pd.Series(trade_dates, index=trade_dates)

    if freq == "M":
        groups = s.groupby([s.index.year, s.index.month])
    elif freq == "W":
        groups = s.groupby([s.index.year, s.index.isocalendar().week])
    elif freq == "Q":
        groups = s.groupby([s.index.year, s.index.quarter])
    else:
        raise ValueError(f"Unknown rebalance freq: {freq}")

    last_dates = groups.last()
    return pd.DatetimeIndex(last_dates.values)


# ---------------------------------------------------------------------------
# Benchmark (small CSV — safe to load once)
# ---------------------------------------------------------------------------

def load_all_benchmarks(cfg: BacktestConfig) -> Dict[str, pd.DataFrame]:
    """
    Scan ``baseline_dir`` and load **every** ``.csv`` file as a benchmark.

    Returns
    -------
    dict  —  {benchmark_name: DataFrame(date, close, ret)}
             benchmark_name is the file stem, e.g. ``"000905.SZ"``.
             Returns empty dict if directory is missing or contains no CSV.
    """
    result: Dict[str, pd.DataFrame] = {}
    baseline_dir = cfg.baseline_dir
    if not os.path.isdir(baseline_dir):
        print(f"[警告] 基准目录不存在: {baseline_dir}")
        return result

    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)

    for fname in sorted(os.listdir(baseline_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        name = fname[:-4]  # strip .csv → e.g. "000905.SZ"
        csv_path = os.path.join(baseline_dir, fname)
        try:
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
            df["ret"] = df["close"].pct_change().fillna(0.0)
            result[name] = df
        except Exception as e:
            print(f"[警告] 加载基准 {fname} 失败: {e}")

    return result


# ---------------------------------------------------------------------------
# DataAccessor — lazy, on-demand queries
# ---------------------------------------------------------------------------

class DataAccessor:
    """
    Lazy data accessor that queries SQLite on demand, avoiding full-table
    loads.  Designed for low-memory environments.

    Lifecycle:  create → use in backtest loop → close().
    Supports context-manager (``with`` statement).

    Example
    -------
    with DataAccessor(cfg) as da:
        snap = da.get_date(some_date)
        window = da.get_window(some_date, lookback=60)
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self._conn: Optional[sqlite3.Connection] = None
        self._current_date: Optional[pd.Timestamp] = None

    # ── look-ahead guard ──

    def set_current_date(self, date: pd.Timestamp):
        """
        Set the current simulation date.  All subsequent queries will be
        validated against this date — any attempt to access data *after*
        it will raise ``LookAheadError``.

        Called automatically by the backtest engine at each rebalance date.
        """
        self._current_date = pd.Timestamp(date)

    @property
    def current_date(self) -> Optional[pd.Timestamp]:
        return self._current_date

    def _check_look_ahead(self, query_date: pd.Timestamp, method: str):
        """
        Validate that *query_date* does not exceed *_current_date*.
        Raises LookAheadError if it does.
        """
        if self._current_date is None:
            return  # Guard not armed (e.g. used outside backtest)
        qd = pd.Timestamp(query_date)
        if qd > self._current_date:
            raise LookAheadError(
                f"前视偏差检测: {method}() 试图访问 {qd.strftime('%Y-%m-%d')} 的数据, "
                f"但当前模拟日期为 {self._current_date.strftime('%Y-%m-%d')}。"
                f"策略不允许访问未来数据。"
            )

    # ── context manager ──

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.cfg.db_path)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        return self._conn

    # ── single-date queries ──

    def get_date(self, date: pd.Timestamp,
                 columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load all stocks' data for a *single* trade date.

        Parameters
        ----------
        date : pd.Timestamp
            The trade date to query.
        columns : list of str, optional
            If given, only these columns are selected (always includes
            ts_code and trade_date).  Pass None for ``SELECT *``.

        Returns
        -------
        pd.DataFrame  — one row per stock on that date.
        """
        self._check_look_ahead(date, "get_date")
        date_str = date.strftime("%Y%m%d")
        if columns:
            cols_set = list(dict.fromkeys(["ts_code", "trade_date"] + columns))
            col_sql = ", ".join(cols_set)
        else:
            col_sql = "*"

        sql = f"""
            SELECT {col_sql} FROM stock_daily
            WHERE trade_date = ?
        """
        df = pd.read_sql_query(sql, self.conn, params=(date_str,))
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return df

    def get_prices(self, date: pd.Timestamp) -> Dict[str, float]:
        """Return {ts_code: close} for a given date.  Lightweight query."""
        self._check_look_ahead(date, "get_prices")
        date_str = date.strftime("%Y%m%d")
        sql = """
            SELECT ts_code, close FROM stock_daily
            WHERE trade_date = ?
        """
        cur = self.conn.execute(sql, (date_str,))
        return {row[0]: row[1] for row in cur.fetchall() if row[1] is not None}

    # ── window / lookback queries ──

    def get_window(self, end_date: pd.Timestamp,
                   lookback: int = 1,
                   ts_codes: Optional[List[str]] = None,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data for the most recent *lookback* trade dates up to
        and including *end_date*.

        Parameters
        ----------
        end_date : pd.Timestamp
            The latest date in the window (inclusive).
        lookback : int
            Number of trade dates to include (default 1 = single date).
        ts_codes : list of str, optional
            If given, restrict to these stocks.
        columns : list of str, optional
            If given, only select these columns.

        Returns
        -------
        pd.DataFrame sorted by (trade_date, ts_code).
        """
        self._check_look_ahead(end_date, "get_window")
        end_str = end_date.strftime("%Y%m%d")

        # Step 1: find the N most recent trade dates <= end_date
        date_sql = """
            SELECT DISTINCT trade_date FROM stock_daily
            WHERE trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT ?
        """
        date_rows = self.conn.execute(date_sql, (end_str, lookback)).fetchall()
        if not date_rows:
            return pd.DataFrame()

        date_list = [r[0] for r in date_rows]
        placeholders = ",".join(["?"] * len(date_list))

        # Step 2: query data for those dates
        if columns:
            cols_set = list(dict.fromkeys(["ts_code", "trade_date"] + columns))
            col_sql = ", ".join(cols_set)
        else:
            col_sql = "*"

        sql = f"SELECT {col_sql} FROM stock_daily WHERE trade_date IN ({placeholders})"
        params: list = list(date_list)

        if ts_codes:
            code_ph = ",".join(["?"] * len(ts_codes))
            sql += f" AND ts_code IN ({code_ph})"
            params.extend(ts_codes)

        sql += " ORDER BY trade_date, ts_code"
        df = pd.read_sql_query(sql, self.conn, params=params)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return df

    def get_stocks_on_date(self, date: pd.Timestamp,
                           ts_codes: List[str],
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data for *specific* stocks on a *single* date.
        Useful when the strategy already knows which stocks it cares about.
        """
        self._check_look_ahead(date, "get_stocks_on_date")
        if not ts_codes:
            return pd.DataFrame()

        date_str = date.strftime("%Y%m%d")
        if columns:
            cols_set = list(dict.fromkeys(["ts_code", "trade_date"] + columns))
            col_sql = ", ".join(cols_set)
        else:
            col_sql = "*"

        code_ph = ",".join(["?"] * len(ts_codes))
        sql = f"""
            SELECT {col_sql} FROM stock_daily
            WHERE trade_date = ? AND ts_code IN ({code_ph})
        """
        params = [date_str] + list(ts_codes)
        df = pd.read_sql_query(sql, self.conn, params=params)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return df

    def count_stocks(self) -> int:
        """Return total distinct stock count within the date range (cheap)."""
        sql = """
            SELECT COUNT(DISTINCT ts_code) FROM stock_daily
            WHERE trade_date >= ? AND trade_date <= ?
        """
        params = (self.cfg.start_date.replace("-", ""),
                  self.cfg.end_date.replace("-", ""))
        cur = self.conn.execute(sql, params)
        return cur.fetchone()[0]

    def count_rows(self) -> int:
        """Return total row count within the date range (cheap)."""
        sql = """
            SELECT COUNT(*) FROM stock_daily
            WHERE trade_date >= ? AND trade_date <= ?
        """
        params = (self.cfg.start_date.replace("-", ""),
                  self.cfg.end_date.replace("-", ""))
        cur = self.conn.execute(sql, params)
        return cur.fetchone()[0]
