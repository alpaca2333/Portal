"""
Generic Factor Calculation Framework
=====================================

Provides reusable, composable factor building blocks that work with
the backtest engine's ``DataAccessor``.

Architecture
------------
- ``Factor`` : abstract base — defines ``compute(date, accessor) -> Series``
- ``CrossSectionalFactor`` : factors needing only a single-date snapshot
- ``TimeSeriesFactor``     : factors needing a lookback window
- ``FactorEngine``         : orchestrates multi-factor calculation with
                             winsorize, industry-neutral z-score, and
                             composite scoring

Built-in factors cover:
  - Momentum  (short / medium / long-term return, reversal)
  - Value     (EP, BP, SP, DP, CFTP)
  - Quality   (ROE, ROA, gross margin, current ratio)
  - Growth    (revenue / profit / equity YoY growth)
  - Risk      (volatility, beta-proxy, turnover stability)
  - Technical (ILLIQ illiquidity, volume-price divergence)
  - Size      (log market cap — typically used as control)

Usage
-----
::

    from engine.factors import FactorEngine, Momentum, ValueBP, QualityROE

    engine = FactorEngine(
        factors=[Momentum(20), ValueBP(), QualityROE()],
        winsorize_pct=(0.01, 0.99),
        neutralize_by="sw_l1",   # industry-neutral z-score
        min_industry_size=5,
    )
    scores = engine.run(date, accessor)
    # scores: pd.DataFrame  with columns [ts_code, <factor_z>, ..., composite]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .data_loader import DataAccessor


# ===================================================================
# Base classes
# ===================================================================

class Factor(ABC):
    """Abstract factor. Subclasses must implement ``compute``."""

    def __init__(self, name: str, ascending: bool = False, weight: float = 1.0):
        """
        Parameters
        ----------
        name : str
            Human-readable factor name (used as column name).
        ascending : bool
            If True, *lower* raw values are better (e.g. volatility).
            Default False → higher raw values are better.
        weight : float
            Weight in the composite score (default 1.0).
        """
        self.name = name
        self.ascending = ascending
        self.weight = weight

    @abstractmethod
    def compute(self, date: pd.Timestamp, accessor: "DataAccessor") -> pd.Series:
        """
        Return a ``pd.Series`` indexed by ``ts_code`` with the raw
        factor values for the given date.

        May contain NaN for stocks where the factor cannot be computed.
        """
        ...

    def __repr__(self):
        direction = "↓" if self.ascending else "↑"
        return f"{self.__class__.__name__}(name={self.name!r}, {direction}, w={self.weight})"


class CrossSectionalFactor(Factor):
    """
    Factor that only needs a *single-date* snapshot.

    Subclasses implement ``compute_from_snapshot`` which receives
    the full cross-sectional DataFrame for that date.

    When called from ``FactorEngine``, the pre-loaded snapshot is
    passed directly via ``compute_from_preloaded`` to avoid redundant
    DB queries.
    """

    def __init__(self, name: str, columns: List[str],
                 ascending: bool = False, weight: float = 1.0):
        super().__init__(name, ascending, weight)
        self.columns = columns

    def compute(self, date: pd.Timestamp, accessor: "DataAccessor") -> pd.Series:
        snap = accessor.get_date(date, columns=self.columns)
        if snap.empty:
            return pd.Series(dtype=float)
        snap = snap.set_index("ts_code")
        return self.compute_from_snapshot(snap)

    def compute_from_preloaded(self, snap: pd.DataFrame) -> pd.Series:
        """
        Compute from a pre-loaded snapshot (already indexed by ts_code).
        Used by FactorEngine to avoid per-factor DB queries.
        """
        return self.compute_from_snapshot(snap)

    @abstractmethod
    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        """Return Series indexed by ts_code."""
        ...


class TimeSeriesFactor(Factor):
    """
    Factor that needs a *lookback window* of historical data.

    Subclasses implement ``compute_from_window`` which receives
    the full panel DataFrame (multi-date, multi-stock).
    """

    def __init__(self, name: str, lookback: int, columns: List[str],
                 ascending: bool = False, weight: float = 1.0):
        super().__init__(name, ascending, weight)
        self.lookback = lookback
        self.columns = columns

    def compute(self, date: pd.Timestamp, accessor: "DataAccessor") -> pd.Series:
        window = accessor.get_window(date, lookback=self.lookback, columns=self.columns)
        if window.empty:
            return pd.Series(dtype=float)
        return self.compute_from_window(window, date)

    @abstractmethod
    def compute_from_window(self, window: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """Return Series indexed by ts_code."""
        ...


# ===================================================================
# Built-in: Momentum factors
# ===================================================================

class Momentum(TimeSeriesFactor):
    """
    Price momentum: ``close_t / close_{t-N} - 1``.

    Higher return → higher score (default).

    Performance note: uses a fast-path that queries only (ts_code, close)
    via a covering index, bypassing the generic ``get_window`` path.
    """

    def __init__(self, lookback: int = 20, weight: float = 1.0):
        super().__init__(
            name=f"momentum_{lookback}d",
            lookback=lookback + 1,  # need N+1 rows to compute N-day return
            columns=["close"],
            ascending=False,
            weight=weight,
        )
        self._period = lookback

    def compute(self, date: pd.Timestamp, accessor: "DataAccessor") -> pd.Series:
        """
        Fast-path: query only the first and last dates in the window
        to compute N-day return, using cursor for minimal overhead.
        """
        accessor._check_look_ahead(date, "Momentum.compute")
        end_str = date.strftime("%Y%m%d")
        conn = accessor.conn

        # Find the N+1 most recent trade dates (we need first & last)
        date_rows = conn.execute(
            "SELECT DISTINCT trade_date FROM stock_daily "
            "WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?",
            (end_str, self.lookback)
        ).fetchall()
        if len(date_rows) < 2:
            return pd.Series(dtype=float)

        end_date = date_rows[0][0]
        start_date = date_rows[-1][0]

        # Query only the two boundary dates — covering index scan
        cur = conn.execute(
            "SELECT ts_code, trade_date, close FROM stock_daily "
            "WHERE trade_date IN (?, ?) AND close IS NOT NULL",
            (start_date, end_date)
        )
        # Build {ts_code: {date: close}} in one pass
        price_map: dict = {}
        for code, td, close in cur:
            if code not in price_map:
                price_map[code] = {}
            price_map[code][td] = close

        # Compute return for stocks that have both dates
        result = {}
        for code, prices in price_map.items():
            p0 = prices.get(start_date)
            p1 = prices.get(end_date)
            if p0 and p1 and p0 > 0:
                result[code] = p1 / p0 - 1

        s = pd.Series(result, dtype=float, name=self.name)
        s.index.name = "ts_code"
        return s

    def compute_from_window(self, window: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        # Fallback (not used in fast-path, kept for compatibility)
        pivot = window.pivot(index="trade_date", columns="ts_code", values="close")
        if len(pivot) < 2:
            return pd.Series(dtype=float)
        ret = pivot.iloc[-1] / pivot.iloc[0] - 1
        ret.name = self.name
        return ret


class Reversal(TimeSeriesFactor):
    """
    Short-term reversal: negative of 5-day return.

    Lower recent return → higher score (contrarian).

    Uses same fast-path as Momentum (only queries boundary dates).
    """

    def __init__(self, lookback: int = 5, weight: float = 1.0):
        super().__init__(
            name=f"reversal_{lookback}d",
            lookback=lookback + 1,
            columns=["close"],
            ascending=True,  # lower return is "better" for reversal
            weight=weight,
        )

    def compute(self, date: pd.Timestamp, accessor: "DataAccessor") -> pd.Series:
        """Fast-path: only query boundary dates."""
        accessor._check_look_ahead(date, "Reversal.compute")
        end_str = date.strftime("%Y%m%d")
        conn = accessor.conn

        date_rows = conn.execute(
            "SELECT DISTINCT trade_date FROM stock_daily "
            "WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?",
            (end_str, self.lookback)
        ).fetchall()
        if len(date_rows) < 2:
            return pd.Series(dtype=float)

        end_date = date_rows[0][0]
        start_date = date_rows[-1][0]

        cur = conn.execute(
            "SELECT ts_code, trade_date, close FROM stock_daily "
            "WHERE trade_date IN (?, ?) AND close IS NOT NULL",
            (start_date, end_date)
        )
        price_map: dict = {}
        for code, td, close in cur:
            if code not in price_map:
                price_map[code] = {}
            price_map[code][td] = close

        result = {}
        for code, prices in price_map.items():
            p0 = prices.get(start_date)
            p1 = prices.get(end_date)
            if p0 and p1 and p0 > 0:
                result[code] = p1 / p0 - 1

        s = pd.Series(result, dtype=float, name=self.name)
        s.index.name = "ts_code"
        return s

    def compute_from_window(self, window: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        pivot = window.pivot(index="trade_date", columns="ts_code", values="close")
        if len(pivot) < 2:
            return pd.Series(dtype=float)
        ret = pivot.iloc[-1] / pivot.iloc[0] - 1
        ret.name = self.name
        return ret


# ===================================================================
# Built-in: Value factors
# ===================================================================

class ValueBP(CrossSectionalFactor):
    """Book-to-Price = 1 / PB. Higher → cheaper."""

    def __init__(self, weight: float = 1.0):
        super().__init__("value_bp", columns=["pb"], ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        pb = snap["pb"]
        bp = pd.Series(np.where(pb > 0, 1.0 / pb, np.nan), index=snap.index, name=self.name)
        return bp


class ValueEP(CrossSectionalFactor):
    """Earnings-to-Price = 1 / PE_TTM. Higher → cheaper."""

    def __init__(self, weight: float = 1.0):
        super().__init__("value_ep", columns=["pe_ttm"], ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        pe = snap["pe_ttm"]
        ep = pd.Series(np.where(pe > 0, 1.0 / pe, np.nan), index=snap.index, name=self.name)
        return ep


class ValueSP(CrossSectionalFactor):
    """Sales-to-Price = 1 / PS_TTM. Higher → cheaper."""

    def __init__(self, weight: float = 1.0):
        super().__init__("value_sp", columns=["ps_ttm"], ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        ps = snap["ps_ttm"]
        sp = pd.Series(np.where(ps > 0, 1.0 / ps, np.nan), index=snap.index, name=self.name)
        return sp


class ValueDP(CrossSectionalFactor):
    """Dividend yield (TTM). Higher → more income."""

    def __init__(self, weight: float = 1.0):
        super().__init__("value_dp", columns=["dv_ttm"], ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["dv_ttm"].rename(self.name)


class ValueCFTP(CrossSectionalFactor):
    """
    Cash-flow-to-Price = CFPS / close.
    Higher → more cash-flow per unit price.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__("value_cftp", columns=["cfps", "close"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        close = snap["close"]
        cfps = snap["cfps"]
        cftp = pd.Series(
            np.where(close > 0, cfps / close, np.nan),
            index=snap.index, name=self.name,
        )
        return cftp


# ===================================================================
# Built-in: Quality factors
# ===================================================================

class QualityROE(CrossSectionalFactor):
    """ROE (TTM or latest annual). Higher → better profitability."""

    def __init__(self, weight: float = 1.0):
        super().__init__("quality_roe", columns=["roe"], ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["roe"].rename(self.name)


class QualityROA(CrossSectionalFactor):
    """ROA. Higher → better asset efficiency."""

    def __init__(self, weight: float = 1.0):
        super().__init__("quality_roa", columns=["roa"], ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["roa"].rename(self.name)


class QualityGrossMargin(CrossSectionalFactor):
    """Gross profit margin. Higher → stronger pricing power."""

    def __init__(self, weight: float = 1.0):
        super().__init__("quality_gross_margin", columns=["grossprofit_margin"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["grossprofit_margin"].rename(self.name)


class QualityNetMargin(CrossSectionalFactor):
    """Net profit margin. Higher → better cost control."""

    def __init__(self, weight: float = 1.0):
        super().__init__("quality_net_margin", columns=["netprofit_margin"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["netprofit_margin"].rename(self.name)


class QualityCurrentRatio(CrossSectionalFactor):
    """Current ratio. Higher → better short-term solvency."""

    def __init__(self, weight: float = 1.0):
        super().__init__("quality_current_ratio", columns=["current_ratio"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["current_ratio"].rename(self.name)


class QualityDebtToAssets(CrossSectionalFactor):
    """Debt-to-assets ratio. Lower → safer balance sheet."""

    def __init__(self, weight: float = 1.0):
        super().__init__("quality_low_leverage", columns=["debt_to_assets"],
                         ascending=True, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["debt_to_assets"].rename(self.name)


class QualityAssetTurnover(CrossSectionalFactor):
    """Total asset turnover. Higher → more efficient use of assets."""

    def __init__(self, weight: float = 1.0):
        super().__init__("quality_asset_turnover", columns=["assets_turn"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["assets_turn"].rename(self.name)


# ===================================================================
# Built-in: Growth factors
# ===================================================================

class GrowthRevenue(CrossSectionalFactor):
    """Revenue year-over-year growth. Higher → faster revenue growth."""

    def __init__(self, weight: float = 1.0):
        super().__init__("growth_revenue", columns=["tr_yoy"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["tr_yoy"].rename(self.name)


class GrowthProfit(CrossSectionalFactor):
    """Operating profit year-over-year growth."""

    def __init__(self, weight: float = 1.0):
        super().__init__("growth_profit", columns=["op_yoy"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["op_yoy"].rename(self.name)


class GrowthEquity(CrossSectionalFactor):
    """Net equity year-over-year growth. Higher → stronger retention."""

    def __init__(self, weight: float = 1.0):
        super().__init__("growth_equity", columns=["equity_yoy"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["equity_yoy"].rename(self.name)


class GrowthEarnings(CrossSectionalFactor):
    """EBT year-over-year growth."""

    def __init__(self, weight: float = 1.0):
        super().__init__("growth_earnings", columns=["ebt_yoy"],
                         ascending=False, weight=weight)

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        return snap["ebt_yoy"].rename(self.name)


# ===================================================================
# Built-in: Risk / Volatility factors
# ===================================================================

class Volatility(TimeSeriesFactor):
    """
    Realised volatility: std of daily returns over lookback window.

    Lower volatility → higher score (ascending=True by default).
    """

    def __init__(self, lookback: int = 20, weight: float = 1.0):
        super().__init__(
            name=f"volatility_{lookback}d",
            lookback=lookback + 1,
            columns=["close"],
            ascending=True,  # lower vol is "better"
            weight=weight,
        )

    def compute_from_window(self, window: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        pivot = window.pivot(index="trade_date", columns="ts_code", values="close")
        daily_ret = pivot.pct_change().dropna(how="all")
        vol = daily_ret.std()
        vol.name = self.name
        return vol


class TurnoverStability(TimeSeriesFactor):
    """
    Turnover coefficient of variation over lookback window.

    Lower CV → more stable trading pattern → higher score.
    """

    def __init__(self, lookback: int = 20, weight: float = 1.0):
        super().__init__(
            name=f"turnover_stability_{lookback}d",
            lookback=lookback,
            columns=["turnover_rate_f"],
            ascending=True,  # lower CV is "better"
            weight=weight,
        )

    def compute_from_window(self, window: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        pivot = window.pivot(index="trade_date", columns="ts_code",
                             values="turnover_rate_f")
        mean = pivot.mean()
        std = pivot.std()
        cv = pd.Series(
            np.where(mean > 0, std / mean, np.nan),
            index=mean.index, name=self.name,
        )
        return cv


# ===================================================================
# Built-in: Technical / Liquidity factors
# ===================================================================

class Illiquidity(TimeSeriesFactor):
    """
    Amihud ILLIQ = mean(|return| / amount) over lookback window.

    Higher ILLIQ → less liquid → higher score if you want *illiquidity premium*,
    but by default ascending=True (prefer more liquid stocks).
    Set ascending=False if you want to tilt towards illiquidity premium.
    """

    def __init__(self, lookback: int = 20, ascending: bool = True,
                 weight: float = 1.0):
        super().__init__(
            name=f"illiquidity_{lookback}d",
            lookback=lookback,
            columns=["pct_chg", "amount"],
            ascending=ascending,
            weight=weight,
        )

    def compute_from_window(self, window: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        df = window.copy()
        df["abs_ret"] = df["pct_chg"].abs()
        df["illiq"] = df["abs_ret"] / df["amount"].replace(0, np.nan)
        result = df.groupby("ts_code")["illiq"].mean()
        result.name = self.name
        return result


class VolumePriceDivergence(TimeSeriesFactor):
    """
    Volume-price divergence: correlation between daily return and volume
    over lookback window. Negative correlation may signal distribution.

    Higher (positive) correlation → momentum confirmation → higher score.
    """

    def __init__(self, lookback: int = 20, weight: float = 1.0):
        super().__init__(
            name=f"vol_price_corr_{lookback}d",
            lookback=lookback,
            columns=["pct_chg", "vol"],
            ascending=False,
            weight=weight,
        )

    def compute_from_window(self, window: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        result = {}
        for code, g in window.groupby("ts_code"):
            if len(g) < 5:
                continue
            corr = g["pct_chg"].corr(g["vol"])
            result[code] = corr
        s = pd.Series(result, dtype=float, name=self.name)
        return s


# ===================================================================
# Built-in: Size factor
# ===================================================================

class SizeLogMV(CrossSectionalFactor):
    """
    Log of circulating market value. Typically used as *control* factor.

    ascending=True → prefer *smaller* stocks (small-cap premium).
    ascending=False → prefer *larger* stocks (large-cap tilt).
    """

    def __init__(self, ascending: bool = True, weight: float = 1.0):
        super().__init__(
            name="size_log_mv",
            columns=["circ_mv"],
            ascending=ascending,
            weight=weight,
        )

    def compute_from_snapshot(self, snap: pd.DataFrame) -> pd.Series:
        mv = snap["circ_mv"]
        log_mv = pd.Series(
            np.where(mv > 0, np.log(mv), np.nan),
            index=snap.index, name=self.name,
        )
        return log_mv


# ===================================================================
# FactorEngine — orchestration
# ===================================================================

@dataclass
class FactorEngine:
    """
    Multi-factor scoring engine.

    Computes all factors, applies winsorize & industry-neutral z-score,
    and produces a composite score.

    Parameters
    ----------
    factors : list of Factor
        Factors to compute. Each contributes ``factor.weight`` to the
        composite score.
    winsorize_pct : tuple of (lower, upper)
        Percentile bounds for winsorization (default 1%–99%).
        Set to None to skip winsorization.
    neutralize_by : str or None
        Column name for industry-neutral z-score (e.g. ``"sw_l1"``).
        If None, global z-score is applied instead.
    min_industry_size : int
        Minimum number of stocks in an industry group. Groups smaller
        than this are dropped when ``neutralize_by`` is set.
    universe_filter : dict or None
        Additional filter conditions applied to the snapshot before
        factor computation. Keys are column names; values are callables
        ``(Series) -> bool_mask``.
    """

    factors: List[Factor]
    winsorize_pct: Optional[Tuple[float, float]] = (0.01, 0.99)
    neutralize_by: Optional[str] = "sw_l1"
    min_industry_size: int = 5
    universe_filter: Optional[Dict] = None

    def _get_all_columns(self) -> List[str]:
        """Collect all DB columns required by all factors + neutralize column."""
        cols = set()
        for f in self.factors:
            if isinstance(f, CrossSectionalFactor):
                cols.update(f.columns)
        # Always need these for filtering
        cols.update(["close", "is_suspended", "circ_mv"])
        if self.neutralize_by:
            cols.add(self.neutralize_by)
        return list(cols)

    def _apply_universe_filter(self, snap: pd.DataFrame) -> pd.DataFrame:
        """Apply basic + custom universe filters."""
        mask = (
            (snap["is_suspended"] != 1)
            & (snap["close"].notna())
            & (snap["close"] > 0)
        )
        if self.universe_filter:
            for col, func in self.universe_filter.items():
                if col in snap.columns:
                    mask = mask & func(snap[col])
        return snap[mask].copy()

    @staticmethod
    def _winsorize(s: pd.Series, lower: float, upper: float) -> pd.Series:
        """Clip values to [lower_pct, upper_pct] percentile range."""
        lo = s.quantile(lower)
        hi = s.quantile(upper)
        return s.clip(lo, hi)

    def _zscore_global(self, s: pd.Series) -> pd.Series:
        """Global z-score normalization."""
        mu = s.mean()
        sigma = s.std()
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    def _zscore_by_group(self, df: pd.DataFrame,
                         col: str, group_col: str) -> pd.Series:
        """Within-group z-score. Groups with std=0 get score 0."""
        result = pd.Series(0.0, index=df.index, dtype=float)
        for _, g in df.groupby(group_col):
            mu = g[col].mean()
            sigma = g[col].std()
            if sigma == 0 or np.isnan(sigma):
                continue
            result.loc[g.index] = (g[col] - mu) / sigma
        return result

    def run(self, date: pd.Timestamp, accessor: "DataAccessor") -> pd.DataFrame:
        """
        Compute all factors and produce a composite score.

        Optimised: loads the full snapshot once and passes it to all
        CrossSectionalFactors, avoiding repeated DB queries.

        Returns
        -------
        pd.DataFrame
            Columns: ``ts_code``, one column per factor (z-scored),
            and ``composite`` (weighted sum of z-scores).
            Sorted by ``composite`` descending.
            Only contains stocks that passed the universe filter and
            have valid values for *all* factors.
        """
        # ── Step 1: Load snapshot ONCE for all cross-sectional factors + universe filter ──
        all_cs_cols = self._get_all_columns()
        snap_full = accessor.get_date(date, columns=all_cs_cols)
        if snap_full.empty:
            return pd.DataFrame(columns=["ts_code", "composite"])

        snap_filtered = self._apply_universe_filter(snap_full)
        snap_indexed = snap_filtered.set_index("ts_code")

        # ── Step 2: Compute raw factor values ──
        raw: Dict[str, pd.Series] = {}
        for f in self.factors:
            if isinstance(f, CrossSectionalFactor):
                # Use pre-loaded snapshot — no extra DB query
                raw[f.name] = f.compute_from_preloaded(snap_indexed)
            else:
                # TimeSeriesFactor still needs its own window query
                raw[f.name] = f.compute(date, accessor)

        # Merge all into one DataFrame
        frames = []
        for name, s in raw.items():
            if s.empty:
                continue
            s = s.rename(name)
            if s.index.name != "ts_code":
                s.index.name = "ts_code"
            frames.append(s)

        if not frames:
            return pd.DataFrame(columns=["ts_code", "composite"])

        combined = pd.concat(frames, axis=1)
        combined.index.name = "ts_code"
        combined = combined.reset_index()

        # ── Step 3: Inner join with filtered universe ──
        snap_keys = snap_filtered[["ts_code"] + ([self.neutralize_by] if self.neutralize_by else [])]
        df = snap_keys.merge(combined, on="ts_code", how="inner")

        factor_names = [f.name for f in self.factors]

        # Drop rows with any NaN factor value
        df = df.dropna(subset=factor_names)

        if df.empty:
            return pd.DataFrame(columns=["ts_code", "composite"])

        # ── Step 3: Industry filter (if neutralizing) ──
        if self.neutralize_by and self.neutralize_by in df.columns:
            ind_counts = df[self.neutralize_by].value_counts()
            valid_ind = ind_counts[ind_counts >= self.min_industry_size].index
            df = df[df[self.neutralize_by].isin(valid_ind)].copy()

        if df.empty:
            return pd.DataFrame(columns=["ts_code", "composite"])

        df = df.reset_index(drop=True)

        # ── Step 4: Winsorize ──
        if self.winsorize_pct is not None:
            lo, hi = self.winsorize_pct
            for name in factor_names:
                df[name] = self._winsorize(df[name], lo, hi)

        # ── Step 5: Z-score (industry-neutral or global) ──
        for f in self.factors:
            col = f.name
            z_col = f"z_{col}"
            if self.neutralize_by and self.neutralize_by in df.columns:
                df[z_col] = self._zscore_by_group(df, col, self.neutralize_by)
            else:
                df[z_col] = self._zscore_global(df[col])

            # Flip sign for ascending factors (lower raw → higher z-score)
            if f.ascending:
                df[z_col] = -df[z_col]

        # ── Step 6: Composite score (weighted sum) ──
        total_weight = sum(f.weight for f in self.factors)
        df["composite"] = 0.0
        for f in self.factors:
            df["composite"] += f.weight * df[f"z_{f.name}"]
        if total_weight > 0:
            df["composite"] /= total_weight

        # ── Clean up and sort ──
        keep_cols = ["ts_code"] + [f"z_{f.name}" for f in self.factors] + ["composite"]
        if self.neutralize_by and self.neutralize_by in df.columns:
            keep_cols.insert(1, self.neutralize_by)
        result = df[keep_cols].sort_values("composite", ascending=False).reset_index(drop=True)
        return result


# ===================================================================
# Convenience: pre-built factor collections
# ===================================================================

def value_factors(weight: float = 1.0) -> List[Factor]:
    """Return a list of common value factors with equal weight."""
    w = weight
    return [ValueBP(w), ValueEP(w), ValueSP(w), ValueDP(w)]


def quality_factors(weight: float = 1.0) -> List[Factor]:
    """Return a list of common quality factors with equal weight."""
    w = weight
    return [QualityROE(w), QualityROA(w), QualityGrossMargin(w),
            QualityDebtToAssets(w)]


def growth_factors(weight: float = 1.0) -> List[Factor]:
    """Return a list of common growth factors with equal weight."""
    w = weight
    return [GrowthRevenue(w), GrowthProfit(w), GrowthEarnings(w)]


def momentum_factors(lookback: int = 20, weight: float = 1.0) -> List[Factor]:
    """Return momentum + reversal factors."""
    return [Momentum(lookback, weight), Reversal(5, weight * 0.5)]


# ===================================================================
# ALL_FACTORS registry — for discovery / documentation
# ===================================================================

ALL_FACTORS = {
    # Momentum
    "momentum":           Momentum,
    "reversal":           Reversal,
    # Value
    "value_bp":           ValueBP,
    "value_ep":           ValueEP,
    "value_sp":           ValueSP,
    "value_dp":           ValueDP,
    "value_cftp":         ValueCFTP,
    # Quality
    "quality_roe":        QualityROE,
    "quality_roa":        QualityROA,
    "quality_gross_margin": QualityGrossMargin,
    "quality_net_margin": QualityNetMargin,
    "quality_current_ratio": QualityCurrentRatio,
    "quality_low_leverage": QualityDebtToAssets,
    "quality_asset_turnover": QualityAssetTurnover,
    # Growth
    "growth_revenue":     GrowthRevenue,
    "growth_profit":      GrowthProfit,
    "growth_equity":      GrowthEquity,
    "growth_earnings":    GrowthEarnings,
    # Risk
    "volatility":         Volatility,
    "turnover_stability": TurnoverStability,
    # Technical
    "illiquidity":        Illiquidity,
    "vol_price_corr":     VolumePriceDivergence,
    # Size
    "size_log_mv":        SizeLogMV,
}
