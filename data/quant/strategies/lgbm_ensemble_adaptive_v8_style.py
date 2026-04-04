"""
LightGBM Ensemble Adaptive Strategy V8 (Style-Aware)
=====================================================

V8 inherits ALL logic from V7 and adds a single **style gate**:
a small-cap dominance detector that decides whether to run the V7
stock-selection engine (full position) or skip entirely (cash).

Three gate rules are available, selected via ``style_rule``:

  Rule 5 — "rolling_excess"  (RECOMMENDED, medium sensitivity)
  ─────────────────────────
  12-week rolling excess return: CSI1000 – CSI300.
  - Small-cap dominant  : excess > +threshold_enter  (default +3%)
  - Large-cap dominant  : excess < -threshold_exit   (default -5%)
  - Asymmetric thresholds reduce unnecessary switching.
  - In-between zone: hold previous state (inertia).

  Rule 3 — "vol_ratio"  (HIGH sensitivity, good for crash detection)
  ────────────────────
  Ratio of small-cap vs large-cap 20-day realised volatility.
  - Small-cap dominant  : vol_ratio < vol_upper  (default 1.2)
  - Crash / stampede    : vol_ratio > vol_crash  (default 1.5)
  - Designed to catch the 2024-H1 small-cap liquidity crush early,
    because volatility spikes *before* breadth collapses.

  Rule 7 — "long_trend"  (LOW sensitivity, quarterly trend)
  ─────────────────────
  26-week (≈half-year) cumulative excess return: CSI1000 – CSI300.
  - Small-cap dominant  : excess > 0
  - Large-cap dominant  : excess ≤ 0
  - Switches at most a few times per year; very stable but slow.

Gate behaviour
──────────────
  is_smallcap_dominant() == True  →  run V7 stock selection, full position
  is_smallcap_dominant() == False →  return {} (cash, skip period entirely)

Both CSI1000 (000905.SZ) and CSI300 (000300.SH) daily close series are
loaded once from the baseline CSV files at warmup and reused each period.

Usage
-----
cd /data/Projects/Portal
python -m data.quant.strategies.lgbm_ensemble_adaptive_v8_style
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from typing import Dict, Literal, Optional

from engine import BacktestConfig, run_backtest
from engine.data_loader import DataAccessor

# Re-use everything from V7
from strategies.lgbm_ensemble_adaptive_v7 import LGBMEnsembleAdaptiveV7

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ---------------------------------------------------------------------------
# Style gate logic (pure, stateless helpers)
# ---------------------------------------------------------------------------

StyleRule = Literal["rolling_excess", "vol_ratio", "long_trend"]


def _load_index_series(baseline_dir: str, code: str) -> pd.Series:
    """
    Load daily close price series for an index from baseline CSV.
    Returns pd.Series indexed by pd.Timestamp, sorted ascending.
    """
    path = os.path.join(baseline_dir, f"{code}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline file not found: {path}. "
            f"Expected CSI1000=000905.SZ.csv, CSI300=000300.SH.csv"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")["close"]
    return df


def _compute_rolling_excess(
    smallcap: pd.Series,
    largecap: pd.Series,
    date: pd.Timestamp,
    window_weeks: int,
) -> Optional[float]:
    """
    Compute cumulative excess return of smallcap vs largecap
    over the past ``window_weeks`` calendar weeks ending on ``date``.

    Returns None if insufficient data.
    """
    start = date - pd.DateOffset(weeks=window_weeks)
    sc = smallcap[(smallcap.index >= start) & (smallcap.index <= date)]
    lc = largecap[(largecap.index >= start) & (largecap.index <= date)]

    common = sc.index.intersection(lc.index)
    if len(common) < 5:
        return None

    sc_ret = sc.loc[common].iloc[-1] / sc.loc[common].iloc[0] - 1
    lc_ret = lc.loc[common].iloc[-1] / lc.loc[common].iloc[0] - 1
    return float(sc_ret - lc_ret)


def _compute_vol_ratio(
    smallcap: pd.Series,
    largecap: pd.Series,
    date: pd.Timestamp,
    window_days: int = 20,
) -> Optional[float]:
    """
    Ratio of small-cap 20-day realised vol to large-cap 20-day realised vol.
    Returns None if insufficient data.
    """
    start = date - pd.DateOffset(days=window_days * 2)
    sc = smallcap[(smallcap.index >= start) & (smallcap.index <= date)]
    lc = largecap[(largecap.index >= start) & (largecap.index <= date)]

    common = sc.index.intersection(lc.index)
    if len(common) < window_days:
        return None

    sc_tail = sc.loc[common].iloc[-window_days:]
    lc_tail = lc.loc[common].iloc[-window_days:]

    sc_vol = sc_tail.pct_change().dropna().std()
    lc_vol = lc_tail.pct_change().dropna().std()

    if lc_vol == 0:
        return None
    return float(sc_vol / lc_vol)


# ---------------------------------------------------------------------------
# V8 Strategy
# ---------------------------------------------------------------------------

class LGBMEnsembleAdaptiveV8Style(LGBMEnsembleAdaptiveV7):
    """
    V7 + style gate.

    Parameters
    ----------
    style_rule : str
        One of "rolling_excess", "vol_ratio", "long_trend".
        Controls which small-cap dominance rule is used.

    baseline_dir : str
        Path to the directory that contains 000905.SZ.csv and 000300.SH.csv.

    # rolling_excess (Rule 5) parameters
    excess_enter_threshold : float
        12-week excess return above which small-cap is considered dominant.
        Default +0.03 (+3%).
    excess_exit_threshold : float
        12-week excess return below which large-cap is considered dominant.
        Default -0.05 (-5%). Asymmetric: harder to declare large-cap dominant.

    # vol_ratio (Rule 3) parameters
    vol_upper : float
        Vol ratio below which small-cap volatility is "normal" (dominant).
        Default 1.2.
    vol_crash : float
        Vol ratio above which a small-cap stampede is detected (exit).
        Default 1.5.

    # long_trend (Rule 7) parameters
    long_trend_weeks : int
        Number of weeks for the long-term trend window. Default 26.

    All other kwargs are forwarded to LGBMEnsembleAdaptiveV7.
    """

    def __init__(
        self,
        # Style gate
        style_rule: StyleRule = "rolling_excess",
        baseline_dir: str = "data/quant/baseline",
        # Rule 5 params
        excess_enter_threshold: float = 0.03,
        excess_exit_threshold: float = -0.05,
        # Rule 3 params
        vol_upper: float = 1.2,
        vol_crash: float = 1.5,
        # Rule 7 params
        long_trend_weeks: int = 26,
        # V7 passthrough
        **kwargs,
    ):
        # Provide V7 defaults so describe() works even without explicit kwargs
        kwargs.setdefault("stop_loss_threshold", -0.12)
        kwargs.setdefault("buffer_sigma", 1.0)
        kwargs.setdefault("min_holding_periods", 2)
        kwargs.setdefault("drawdown_circuit_breaker", -0.15)
        kwargs.setdefault("drawdown_reduction_factor", 0.5)
        kwargs.setdefault("max_positions", 25)
        kwargs.setdefault("max_single_weight", 0.08)
        kwargs.setdefault("small_cap_quantile", 0.70)
        kwargs.setdefault("softmax_temperature", 5.0)
        kwargs.setdefault("weight_model_a", 0.6)
        kwargs.setdefault("weight_model_b", 0.4)
        kwargs.setdefault("consensus_top_pct", 0.05)
        kwargs.setdefault("consensus_single_top_pct", 0.10)
        kwargs.setdefault("industry_momentum_bonus", 0.02)

        # Force strategy name to v8_style so output dirs are separate
        super().__init__(**kwargs)
        self.name = "lgbm_ensemble_adaptive_v8_style"

        # V7's __init__ passes stop_loss_threshold only to StopLossTracker
        # but never stores it as self.stop_loss_threshold, so describe() breaks.
        # Patch it here from kwargs defaults we already set.
        if not hasattr(self, "stop_loss_threshold"):
            self.stop_loss_threshold = kwargs.get("stop_loss_threshold", -0.12)

        # Style gate config
        self.style_rule: StyleRule = style_rule
        self._baseline_dir = baseline_dir
        self.excess_enter_threshold = excess_enter_threshold
        self.excess_exit_threshold = excess_exit_threshold
        self.vol_upper = vol_upper
        self.vol_crash = vol_crash
        self.long_trend_weeks = long_trend_weeks

        # Runtime state
        self._smallcap_series: Optional[pd.Series] = None  # CSI1000 daily close
        self._largecap_series: Optional[pd.Series] = None  # CSI300 daily close
        self._style_state: bool = True   # last known style state (True = smallcap)
        self._style_loaded: bool = False

    # ------------------------------------------------------------------
    # describe() — switches based on active rule
    # ------------------------------------------------------------------

    def describe(self) -> str:
        base = super().describe()

        if self.style_rule == "rolling_excess":
            gate_desc = (
                f"\n\n### V8 风格门控 (Rule 5 · rolling_excess · 中灵敏)\n\n"
                f"**逻辑**：每期计算过去12周 CSI1000 – CSI300 累计超额收益。\n"
                f"- 超额 > **{self.excess_enter_threshold*100:+.1f}%** → 小盘占优，"
                f"执行 V7 全仓选股\n"
                f"- 超额 < **{self.excess_exit_threshold*100:+.1f}%** → 大盘占优，"
                f"直接空仓跳过\n"
                f"- 介于两者之间 → 沿用上一期状态（惰性，减少频繁切换）\n"
                f"- **非对称阈值**：进入小盘容易 ({self.excess_enter_threshold*100:+.1f}%)，"
                f"退出小盘需要更强信号 ({self.excess_exit_threshold*100:+.1f}%)，"
                f"避免市场震荡时来回切换\n"
                f"- 数据来源：baseline/ 下 CSI1000 (000905.SZ) 与 CSI300 (000300.SH)"
            )
        elif self.style_rule == "vol_ratio":
            gate_desc = (
                f"\n\n### V8 风格门控 (Rule 3 · vol_ratio · 高灵敏)\n\n"
                f"**逻辑**：每期计算 CSI1000 近20日波动率 / CSI300 近20日波动率。\n"
                f"- 比值 < **{self.vol_upper}** → 小盘波动正常，执行 V7 全仓选股\n"
                f"- 比值 > **{self.vol_crash}** → 小盘踩踏信号，直接空仓跳过\n"
                f"- 介于两者之间 → 沿用上一期状态（惰性）\n"
                f"- **设计初衷**：2024-H1 小微盘流动性踩踏时，波动率在广度指标崩溃"
                f"之前就已飙升，本规则可提前逃顶。对突发系统性风险最灵敏。\n"
                f"- 数据来源：baseline/ 下 CSI1000 (000905.SZ) 与 CSI300 (000300.SH)"
            )
        else:  # long_trend
            gate_desc = (
                f"\n\n### V8 风格门控 (Rule 7 · long_trend · 低灵敏)\n\n"
                f"**逻辑**：每期计算过去 **{self.long_trend_weeks} 周** "
                f"CSI1000 – CSI300 累计超额收益。\n"
                f"- 超额 > 0 → 半年趋势偏小盘，执行 V7 全仓选股\n"
                f"- 超额 ≤ 0 → 半年趋势偏大盘，直接空仓跳过\n"
                f"- **切换频率极低**（一年 2-3 次），对交易成本友好\n"
                f"- 缺点：滞后严重，极端行情时可能要晚 4-6 周才反应\n"
                f"- 数据来源：baseline/ 下 CSI1000 (000905.SZ) 与 CSI300 (000300.SH)"
            )

        return base + gate_desc

    # ------------------------------------------------------------------
    # Style gate helpers
    # ------------------------------------------------------------------

    def _ensure_index_data_loaded(self):
        """Load CSI1000 and CSI300 once from baseline CSVs."""
        if self._style_loaded:
            return
        self._smallcap_series = _load_index_series(self._baseline_dir, "000905.SZ")
        self._largecap_series = _load_index_series(self._baseline_dir, "000300.SH")
        self._style_loaded = True
        print(
            f"      [V8风格] 加载指数数据 "
            f"CSI1000={self._smallcap_series.index[0].date()}~{self._smallcap_series.index[-1].date()} "
            f"CSI300={self._largecap_series.index[0].date()}~{self._largecap_series.index[-1].date()}"
        )

    def _is_smallcap_dominant(self, date: pd.Timestamp) -> bool:
        """
        Evaluate the active style rule and return True if small-cap is dominant.
        Falls back to previous state when data is insufficient.
        """
        self._ensure_index_data_loaded()
        sc = self._smallcap_series
        lc = self._largecap_series

        if self.style_rule == "rolling_excess":
            excess = _compute_rolling_excess(sc, lc, date, window_weeks=12)
            if excess is None:
                print(f"      [V8风格] 数据不足，沿用上期状态: {self._style_state}")
                return self._style_state
            if excess > self.excess_enter_threshold:
                result = True
            elif excess < self.excess_exit_threshold:
                result = False
            else:
                result = self._style_state  # inertia zone
            print(
                f"      [V8风格|rolling_excess] 12周超额={excess*100:+.2f}%  "
                f"阈值=[{self.excess_enter_threshold*100:+.1f}%, {self.excess_exit_threshold*100:+.1f}%]  "
                f"→ {'小盘占优✅' if result else '大盘占优⛔ 空仓'}"
            )
            return result

        elif self.style_rule == "vol_ratio":
            ratio = _compute_vol_ratio(sc, lc, date, window_days=20)
            if ratio is None:
                print(f"      [V8风格] 数据不足，沿用上期状态: {self._style_state}")
                return self._style_state
            if ratio < self.vol_upper:
                result = True
            elif ratio > self.vol_crash:
                result = False
            else:
                result = self._style_state  # inertia zone
            print(
                f"      [V8风格|vol_ratio] 波动率比={ratio:.3f}  "
                f"阈值=[<{self.vol_upper}, >{self.vol_crash}]  "
                f"→ {'小盘占优✅' if result else '大盘占优⛔ 空仓'}"
            )
            return result

        else:  # long_trend
            excess = _compute_rolling_excess(
                sc, lc, date, window_weeks=self.long_trend_weeks
            )
            if excess is None:
                print(f"      [V8风格] 数据不足，沿用上期状态: {self._style_state}")
                return self._style_state
            result = excess > 0
            print(
                f"      [V8风格|long_trend] {self.long_trend_weeks}周超额={excess*100:+.2f}%  "
                f"→ {'小盘占优✅' if result else '大盘占优⛔ 空仓'}"
            )
            return result

    # ------------------------------------------------------------------
    # Override generate_target_weights — add style gate
    # ------------------------------------------------------------------

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Style gate → V7 logic.

        If small-cap is NOT dominant: return {} (full cash, skip period).
        If small-cap IS dominant: delegate to V7's full logic.
        """
        dominant = self._is_smallcap_dominant(date)
        self._style_state = dominant  # persist for inertia zone

        if not dominant:
            # Log context so the backtest record is informative
            held = len(current_holdings) if current_holdings else 0
            if held > 0:
                print(
                    f"      [V8风格] 大盘占优，清仓空仓（上期持有 {held} 只）"
                )
            else:
                print(f"      [V8风格] 大盘占优，继续空仓")
            return {}

        # Small-cap dominant → run the full V7 engine
        return super().generate_target_weights(date, accessor, current_holdings)


# ===================================================================
# Entry point — change style_rule to test each of the three variants
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V8 Style-Aware Strategy Backtest")
    parser.add_argument(
        "--style-rule",
        choices=["rolling_excess", "vol_ratio", "long_trend"],
        default="rolling_excess",
        help=(
            "Style gate rule: "
            "rolling_excess (Rule5, medium), "
            "vol_ratio (Rule3, high sensitivity), "
            "long_trend (Rule7, low sensitivity)"
        ),
    )
    args = parser.parse_args()

    cfg = BacktestConfig(
        initial_capital=300_000,
        commission_rate=1.5e-4,
        slippage=0.0015,
        start_date="2018-01-01",
        end_date="2025-12-31",
        rebalance_freq="BW",
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir="data/quant/backtest",
    )

    strategy = LGBMEnsembleAdaptiveV8Style(
        # ── Style gate ──────────────────────────────────────
        style_rule=args.style_rule,          # "rolling_excess" | "vol_ratio" | "long_trend"
        baseline_dir=cfg.baseline_dir,
        # Rule 5 params (rolling_excess)
        excess_enter_threshold=0.03,         # +3%: enter small-cap regime
        excess_exit_threshold=-0.05,         # -5%: exit to large-cap regime (asymmetric)
        # Rule 3 params (vol_ratio)
        vol_upper=1.2,                       # below → small-cap calm, enter
        vol_crash=1.5,                       # above → stampede detected, exit
        # Rule 7 params (long_trend)
        long_trend_weeks=26,                 # 26-week half-year trend
        # ── V7 passthrough ──────────────────────────────────
        train_window_years=3,
        retrain_interval=4,
        weight_model_a=0.6,
        weight_model_b=0.4,
        consensus_top_pct=0.05,
        consensus_single_top_pct=0.10,
        max_positions=25,
        softmax_temperature=5.0,
        max_single_weight=0.08,
        max_per_industry=3,
        stop_loss_threshold=-0.12,
        stop_loss_cooldown=2,
        buffer_sigma=1.0,
        min_holding_periods=2,
        drawdown_circuit_breaker=-0.15,
        drawdown_reduction_factor=0.5,
        mv_pct_upper=0.85,
        small_cap_bonus=0.02,
        small_cap_quantile=0.70,
        feature_lookback=260,
        backtest_end_date=cfg.end_date,
        industry_momentum_lookback_months=6,
        industry_momentum_bonus=0.02,
        industry_momentum_penalty=-0.01,
        industry_strong_top_n=10,
        industry_weak_top_n=10,
        industry_strong_max=5,
        industry_weak_max=2,
    )

    result = run_backtest(strategy, cfg)
