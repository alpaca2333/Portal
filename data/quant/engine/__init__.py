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
from .factors import (
    Factor, CrossSectionalFactor, TimeSeriesFactor, FactorEngine,
    # Momentum
    Momentum, Reversal,
    # Value
    ValueBP, ValueEP, ValueSP, ValueDP, ValueCFTP,
    # Quality
    QualityROE, QualityROA, QualityGrossMargin, QualityNetMargin,
    QualityCurrentRatio, QualityDebtToAssets, QualityAssetTurnover,
    # Growth
    GrowthRevenue, GrowthProfit, GrowthEquity, GrowthEarnings,
    # Risk
    Volatility, TurnoverStability,
    # Technical
    Illiquidity, VolumePriceDivergence,
    # Size
    SizeLogMV,
    # Convenience
    value_factors, quality_factors, growth_factors, momentum_factors,
    ALL_FACTORS,
)

__all__ = [
    "BacktestConfig", "StrategyBase", "DataAccessor", "LookAheadError", "run_backtest",
    # Factor framework
    "Factor", "CrossSectionalFactor", "TimeSeriesFactor", "FactorEngine",
    # Built-in factors
    "Momentum", "Reversal",
    "ValueBP", "ValueEP", "ValueSP", "ValueDP", "ValueCFTP",
    "QualityROE", "QualityROA", "QualityGrossMargin", "QualityNetMargin",
    "QualityCurrentRatio", "QualityDebtToAssets", "QualityAssetTurnover",
    "GrowthRevenue", "GrowthProfit", "GrowthEquity", "GrowthEarnings",
    "Volatility", "TurnoverStability",
    "Illiquidity", "VolumePriceDivergence",
    "SizeLogMV",
    "value_factors", "quality_factors", "growth_factors", "momentum_factors",
    "ALL_FACTORS",
]
