"""
Regime-switching factor strategy
=================================
基于 ic_reweighted (buf=0.3) 框架，增加市场状态判别层：
每个调仓期先识别市场状态，再动态调整因子权重。

市场状态 (4种):
  BULL       = 市场动量正 且 市场宽度高（普涨趋势）
  BEAR       = 市场动量负 且 市场宽度低（普跌趋势）
  HIGH_VOL   = 波动率 > 历史75分位（高波动震荡）
  NEUTRAL    = 其余（低波动震荡，默认状态）

因子权重按状态切换:
  BULL    → 价值+小市值加大，反转降低（趋势延续，选价值小盘）
  BEAR    → 低波动权重最大（防御第一）
  HIGH_VOL→ 反转权重最大（震荡市反转最有效）
  NEUTRAL → 默认权重（与 ic_reweighted 一致）

实现方式: compute_factors_fn 注入状态信号，post_select 动态修改打分权重
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import StrategyConfig, FactorDef, run_pipeline
from engine.types import SelectionMode, RebalanceFreq
from engine.factor import winsorized_zscore

# ── 状态定义 ──────────────────────────────────────────────────────────────────
BULL     = "BULL"
BEAR     = "BEAR"
HIGH_VOL = "HIGH_VOL"
NEUTRAL  = "NEUTRAL"

# 每种状态下的因子权重
# 设计原则:
#   BULL     → 趋势市，价值+小市值有超额，反转信号弱化
#   BEAR     → 熊市防御，低波动权重最大，其余降低
#   HIGH_VOL → 震荡市，反转最有效，低波动保底
#   NEUTRAL  → IC分析的最优默认权重
REGIME_WEIGHTS = {
    BULL:     {"inv_pb": +0.30, "rev_10": +0.15, "rvol_20": -0.25, "log_cap": -0.30},
    BEAR:     {"inv_pb": +0.20, "rev_10": +0.15, "rvol_20": -0.50, "log_cap": -0.15},
    HIGH_VOL: {"inv_pb": +0.20, "rev_10": +0.45, "rvol_20": -0.25, "log_cap": -0.10},
    NEUTRAL:  {"inv_pb": +0.25, "rev_10": +0.30, "rvol_20": -0.35, "log_cap": -0.10},
}

# 状态切换平滑：连续 N 期相同信号才切换（防止噪音频繁切换）
SMOOTH_PERIODS = 2

# ── 市场状态计算 ───────────────────────────────────────────────────────────────
def compute_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于全市场等权日收益率，计算每个交易日的市场状态信号。
    只用 SH+SZ 股票（排除 BJ），与回测股票池一致。

    输出列:
      mkt_ret_1d   : 全市场等权日收益
      mkt_mom_20   : 20日滚动均值（趋势信号）
      mkt_vol_20   : 20日滚动标准差（波动率信号）
      mkt_breadth  : 当日上涨股占比（宽度信号）
      mkt_vol_q75  : mkt_vol_20 的滚动历史75分位（用252日窗口）
      regime_raw   : 当日原始状态
      regime       : 平滑后状态（连续 SMOOTH_PERIODS 期确认才切换）
    """
    print("[regime] Computing market state signals ...")

    # 只用 SH+SZ
    mkt = df[df["code"].str.startswith(("SH", "SZ"))].copy()
    mkt = mkt.sort_values(["date", "code"])
    # 只保留有效 ret_1d（排除 NaN，避免第一行 pct_change 污染统计）
    mkt = mkt[mkt["ret_1d"].notna()]

    # 日截面等权统计
    daily = mkt.groupby("date").agg(
        mkt_ret_1d=("ret_1d", "mean"),
        mkt_breadth=("ret_1d", lambda x: (x > 0).sum() / max(len(x), 1)),
        n_stocks=("ret_1d", "count"),
    ).reset_index().sort_values("date")

    # 20日滚动趋势和波动
    daily["mkt_mom_20"]  = daily["mkt_ret_1d"].rolling(20, min_periods=15).mean()
    daily["mkt_vol_20"]  = daily["mkt_ret_1d"].rolling(20, min_periods=15).std()

    # 波动率的历史75分位（252日窗口，约1年）
    daily["mkt_vol_q75"] = daily["mkt_vol_20"].rolling(252, min_periods=120).quantile(0.75)

    # 原始状态判别（当日）
    def classify(row):
        mom   = row["mkt_mom_20"]
        vol   = row["mkt_vol_20"]
        q75   = row["mkt_vol_q75"]
        bread = row["mkt_breadth"]
        if pd.isna(mom) or pd.isna(vol) or pd.isna(q75):
            return NEUTRAL
        if vol > q75:
            return HIGH_VOL
        if mom > 0 and bread > 0.55:
            return BULL
        if mom < 0 and bread < 0.45:
            return BEAR
        return NEUTRAL

    daily["regime_raw"] = daily.apply(classify, axis=1)

    # 不做额外平滑：mkt_mom_20 和 mkt_vol_20 本身已经是20日滚动均值/标准差，
    # 信号天然平滑，不需要再做额外的状态平滑处理。
    # 每个调仓期取该期最后一个交易日的状态作为决策依据（在 merge 时自然取到）。
    daily["regime"] = daily["regime_raw"]

    # 统计分布
    dist = daily["regime"].value_counts()
    print(f"[regime] State distribution (daily):")
    for state, cnt in dist.items():
        print(f"         {state:10s}: {cnt:4d} days ({cnt/len(daily):.1%})")

    return daily[["date", "mkt_mom_20", "mkt_vol_20", "mkt_breadth",
                  "mkt_vol_q75", "regime_raw", "regime"]]


# ── 自定义因子计算（注入状态信号）────────────────────────────────────────────
def compute_factors_with_regime(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    标准因子计算 + 注入市场状态。
    状态信号存入 df["regime"] 列，供 post_select 使用。
    """
    print("[factors] Computing standard factors ...")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")

    df["ret_1d"]       = g["close"].pct_change()
    df["close_lag20"]  = g["close"].shift(20)
    df["close_lag250"] = g["close"].shift(250)
    df["mom_12_1"]     = df["close_lag20"] / df["close_lag250"] - 1
    df["close_lag10"]  = g["close"].shift(10)
    df["rev_10"]       = df["close_lag10"] / df["close"] - 1
    df["rvol_20"]      = g["ret_1d"].transform(
        lambda x: x.rolling(20, min_periods=15).std())
    df["vol_ma20"]     = g["volume"].transform(
        lambda x: x.rolling(20, min_periods=15).mean())
    df["vol_ma120"]    = g["volume"].transform(
        lambda x: x.rolling(120, min_periods=80).mean())
    df["vol_confirm"]  = df["vol_ma20"] / df["vol_ma120"]
    pb = df["pb"].replace(0, np.nan)
    df["inv_pb"]       = np.where(pb > 0, 1.0 / pb, np.nan)
    df["log_cap"]      = np.log(df["free_market_cap"].replace(0, np.nan))

    # 释放不再需要的中间列，节省内存
    df.drop(columns=["close_lag20", "close_lag250", "close_lag10",
                      "vol_ma20", "vol_ma120"], inplace=True)

    # 计算市场状态并 merge 进来
    regime_df = compute_market_regime(df)
    df = df.merge(regime_df[["date", "regime"]], on="date", how="left")
    df["regime"] = df["regime"].fillna(NEUTRAL)

    print(f"[factors] Done. Shape: {df.shape}")
    return df


# ── 自定义选股（动态因子权重）────────────────────────────────────────────────
def regime_post_select(signal: pd.DataFrame,
                       prev_holdings: set,
                       cfg) -> pd.DataFrame:
    """
    根据当期市场状态动态计算综合得分，替代引擎默认的固定权重打分。

    signal: 已过滤好的截面（含所有因子列），需要我们打分并选股
    返回: 选出的股票 DataFrame（含 code + score）
    """
    factor_cols = ["inv_pb", "rev_10", "rvol_20", "log_cap"]

    # 确定当期状态（取截面中的众数，同一期所有行应该相同）
    if "regime" in signal.columns:
        state = signal["regime"].mode().iloc[0] if len(signal) > 0 else NEUTRAL
    else:
        state = NEUTRAL

    weights = REGIME_WEIGHTS[state]

    # 行业内重新打分
    from engine.factor import winsorized_zscore
    from engine.types import SelectionMode

    MIN_INDUSTRY = 5
    parts = []
    for (ind,), grp in signal.groupby(["industry_code"]):
        if len(grp) < MIN_INDUSTRY:
            grp = grp.copy()
            grp["score"] = np.nan
            parts.append(grp)
            continue
        grp = grp.copy()
        composite = pd.Series(0.0, index=grp.index)
        for col, w in weights.items():
            if col in grp.columns:
                composite += w * winsorized_zscore(grp[col])
        grp["score"] = composite
        parts.append(grp)

    scored = pd.concat(parts, ignore_index=True)
    scored = scored[scored["score"].notna()]

    if len(scored) < cfg.min_holding:
        return scored.nlargest(cfg.min_holding, "score")

    # 应用 buffer band
    if cfg.buffer_sigma > 0 and len(prev_holdings) > 0:
        is_incumbent = scored["code"].isin(prev_holdings)
        scored.loc[is_incumbent, "score"] += cfg.buffer_sigma

    # 选 top X%
    cutoff = scored["score"].quantile(1 - cfg.top_pct)
    selected = scored[scored["score"] >= cutoff].copy()

    # 每行业上限
    if cfg.max_per_industry > 0:
        selected = (selected
                    .sort_values("score", ascending=False)
                    .groupby("industry_code", group_keys=False)
                    .head(cfg.max_per_industry))

    return selected


# ── Config ─────────────────────────────────────────────────────────────────────
config = StrategyConfig(
    name="regime_switching_2015",
    description="市场状态切换策略：4状态(BULL/BEAR/HIGH_VOL/NEUTRAL) + 动态因子权重",
    rationale="""
## 策略背景

`ic_reweighted` 策略使用固定因子权重，在不同市场环境下表现差异明显：
- 2022熊市：低波动权重够，防守好（+1%）
- 2025政策急涨：固定权重无法适应反转行情

本策略在 `ic_reweighted` 基础上增加**市场状态判别层**，根据当期市场状态动态调整因子权重。

## 市场状态定义

用3个纯价格信号，无前视偏差：

| 信号 | 计算方式 |
|------|---------|
| `mkt_mom_20` | 全市场等权日收益的20日滚动均值 |
| `mkt_vol_20` | 全市场等权日收益的20日滚动标准差 |
| `mkt_breadth` | 当日上涨股占比 |

状态判别（优先级从高到低）：
1. `HIGH_VOL`：mkt_vol_20 > 历史75分位
2. `BULL`：mkt_mom_20 > 0 且 mkt_breadth > 0.55
3. `BEAR`：mkt_mom_20 < 0 且 mkt_breadth < 0.45
4. `NEUTRAL`：其余

加平滑：连续2期同状态才切换，防止噪音频繁切换。

## 因子权重切换逻辑

| 状态 | inv_pb | rev_10 | rvol_20 | log_cap | 设计逻辑 |
|------|--------|--------|---------|---------|---------|
| BULL | +0.30 | +0.15 | -0.25 | -0.30 | 趋势市，价值+小盘，反转信号弱化 |
| BEAR | +0.20 | +0.15 | -0.50 | -0.15 | 熊市防御，低波动最大 |
| HIGH_VOL | +0.20 | +0.45 | -0.25 | -0.10 | 震荡市，反转最有效 |
| NEUTRAL | +0.25 | +0.30 | -0.35 | -0.10 | IC分析默认最优权重 |

## 实现方式

- `compute_factors_fn`：标准因子计算 + 注入 `regime` 列
- `post_select`：根据 `regime` 选对应权重重新打分选股
- 其他参数（股票池、调仓频率、成本、buffer）与 `ic_reweighted` 完全一致
""",
    warm_up_start="2014-01-01",
    backtest_start="2015-01-01",
    end="2026-02-28",
    freq=RebalanceFreq.BIWEEKLY,
    mcap_keep_pct=0.70,
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.05,
    max_per_industry=5,
    min_industry_count=5,
    min_holding=20,
    single_side_cost=0.00015,
    buffer_sigma=0.3,
    compute_factors_fn=compute_factors_with_regime,
    post_select=regime_post_select,
)

# factors 列表仅用于报告展示（实际打分在 post_select 里动态计算）
factors = [
    FactorDef("inv_pb",   +0.25),  # NEUTRAL权重（代表值）
    FactorDef("rev_10",   +0.30),
    FactorDef("rvol_20",  -0.35),
    FactorDef("log_cap",  -0.10),
]

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(config, factors)
