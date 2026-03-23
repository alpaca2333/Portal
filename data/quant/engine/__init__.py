"""
Quant Backtest Engine
=====================
A rigorous event-driven backtesting framework with:
- Initial capital + round-lot constraint (100 shares per lot)
- One-way commission (default 1.5 bps) and slippage support
- Per-period trade logging to disk
- Insufficient-funds guard on buy orders
- Lazy data loading via DataAccessor (low-memory friendly)
"""

from .config import BacktestConfig
from .strategy_base import StrategyBase
from .data_loader import DataAccessor, LookAheadError
from .backtest import run_backtest

__all__ = ["BacktestConfig", "StrategyBase", "DataAccessor", "LookAheadError", "run_backtest"]
