"""
Strategy base class — users should inherit and implement ``generate_target_weights``.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

import pandas as pd

if TYPE_CHECKING:
    from .data_loader import DataAccessor


class StrategyBase(ABC):
    """
    Abstract base for all strategies.

    The engine calls ``generate_target_weights`` on each rebalance date,
    passing a ``DataAccessor`` for on-demand data queries.

    The strategy should use the accessor's methods to fetch only the data
    it needs (single-date snapshot, lookback window, etc.) — this keeps
    memory usage minimal.

    The method should return a dict ``{ts_code: weight}`` where weights
    are target *portfolio fractions* (they will be normalised to sum to 1
    by the engine if they don't).  A stock not in the dict is assumed
    to have weight 0 (i.e. should be sold).

    Example
    -------
    class EqualWeight(StrategyBase):
        def __init__(self):
            super().__init__("equal_weight")

        def generate_target_weights(self, date, accessor, current_holdings):
            snap = accessor.get_date(date)
            top = snap.nlargest(30, "circ_mv")
            codes = top["ts_code"].tolist()
            w = 1.0 / len(codes) if codes else 0
            return {c: w for c in codes}
    """

    def __init__(self, name: str):
        self.name = name

    def describe(self) -> str:
        """
        Return a free-form text description of the strategy.

        This will be embedded in ``{strategy_name}_report.md``.
        Override to provide:
        - Strategy thesis / rationale
        - Technical details (signal construction, filtering, etc.)
        - Known limitations & potential improvements

        If not overridden, a default placeholder is used.
        """
        return "（策略描述未提供，请在子类中重写 `describe()` 方法。）"

    @abstractmethod
    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: "DataAccessor",
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Return target portfolio weights for the given rebalance date.

        Parameters
        ----------
        date : pd.Timestamp
            The current rebalance date.
        accessor : DataAccessor
            Lazy data accessor — call ``accessor.get_date(date)``,
            ``accessor.get_window(date, lookback=N)``, etc. to fetch
            only the data you need.  See ``DataAccessor`` for full API.
        current_holdings : dict
            ``{ts_code: shares}`` of the portfolio *before* rebalancing.

        Returns
        -------
        dict
            ``{ts_code: target_weight}`` — weights should be non-negative
            and ideally sum to 1 (they will be normalised otherwise).
            Return an empty dict to go fully to cash.
        """
        ...
