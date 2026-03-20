"""
data_prep_v2.py — 独立实现的数据管道
=====================================
完全独立实现，不参考任何原始 data_prep 代码或字节码。
仅知道：特征名列表 + 标准量化因子定义。

特征列表（15个，来自原策略 ALL_FEATURES）:
  mom_12_1, rev_10, rvol_20, vol_confirm, inv_pb, log_cap,
  pe_ttm, roe_ttm, turnover_20, mom_3_1, mom_6_1,
  ret_5d_std, volume_chg, high_low_20, close_to_high_60
"""
from __future__ import annotations

import sqlite3
import gc
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = "/projects/portal/data/quant/processed/stocks.db"

ALL_FEATURES = [
    "mom_12_1", "rev_10", "rvol_20", "vol_confirm", "inv_pb", "log_cap",
    "pe_ttm", "roe_ttm", "turnover_20", "mom_3_1", "mom_6_1",
    "ret_5d_std", "volume_chg", "high_low_20", "close_to_high_60",
]


# ──────────────────────────────────────────────
# 1. 数据加载
# ──────────────────────────────────────────────

def load_raw(start: str, end: str) -> pd.DataFrame:
    """从 SQLite 加载 SH/SZ 日线数据，全程 float32 + category 节省内存。"""
    print(f"[V2-data] 加载 {start} ~ {end} ...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT code, date, open, high, low, close, volume,
               pb, pe_ttm, roe_ttm, free_market_cap,
               industry_code, industry_name
        FROM kline
        WHERE (code LIKE 'SH%' OR code LIKE 'SZ%')
          AND date >= '{start}' AND date <= '{end}'
        ORDER BY code, date
    """, conn)
    conn.close()

    # 类型压缩：这是内存控制的关键
    df["date"] = pd.to_datetime(df["date"])
    df["code"]          = df["code"].astype("category")
    df["industry_code"] = df["industry_code"].astype("category")
    df["industry_name"] = df["industry_name"].astype("category")

    for c in ["open", "high", "low", "close", "volume",
              "pb", "pe_ttm", "roe_ttm", "free_market_cap"]:
        df[c] = df[c].astype("float32")

    mb = df.memory_usage(deep=True).sum() / 1e9
    print(f"[V2-data] {len(df):,} 行, {df['code'].nunique()} 只股票, {mb:.2f} GB")
    return df


# ──────────────────────────────────────────────
# 2. 特征计算
# ──────────────────────────────────────────────

def _roll(series: pd.Series, by: pd.Series,
          window: int, min_p: int, func: str) -> pd.Series:
    """
    按股票分组的滚动计算，内存最优版。
    使用 groupby().rolling().agg()，计算完即转 float32 压缩内存。
    """
    result = (
        series.groupby(by, sort=False)
              .rolling(window, min_periods=min_p)
              .agg(func)
              .droplevel(0)
              .sort_index()
              .astype("float32")
    )
    return result


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 15 个特征。
    每个因子都按照标准定义独立实现，不参考任何原始代码。
    """
    print("[V2-data] 计算特征 ...")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    code = df["code"]

    # ── 日收益率（供波动率因子使用）──
    ret_1d = df.groupby("code", sort=False)["close"].pct_change()

    # ── 下一交易日开盘价（用于构建标签）──
    df["next_open"] = df.groupby("code", sort=False)["open"].shift(-1)
    df["next_date"] = df.groupby("code", sort=False)["date"].shift(-1)

    # ────────── 动量 & 反转因子（逐步计算，用完即释放）──────────
    c = df.groupby("code", sort=False)["close"]

    # mom_12_1 = close.shift(20) / close.shift(250) - 1
    c20  = c.shift(20).astype("float32")
    c250 = c.shift(250).astype("float32")
    df["mom_12_1"] = (c20 / c250 - 1).astype("float32")
    del c250

    # rev_10 = close.shift(10) / close - 1
    c10 = c.shift(10).astype("float32")
    df["rev_10"] = (c10 / df["close"] - 1).astype("float32")
    del c10

    # mom_3_1 = close.shift(20) / close.shift(60) - 1
    c60 = c.shift(60).astype("float32")
    df["mom_3_1"] = (c20 / c60 - 1).astype("float32")
    del c60

    # mom_6_1 = close.shift(20) / close.shift(125) - 1
    c125 = c.shift(125).astype("float32")
    df["mom_6_1"] = (c20 / c125 - 1).astype("float32")
    del c125, c20
    gc.collect()

    # ────────── 波动率因子 ──────────
    df["rvol_20"]    = _roll(ret_1d, code, 20, 15, "std")
    df["ret_5d_std"] = _roll(ret_1d, code,  5,  4, "std")
    del ret_1d
    gc.collect()

    # ────────── 成交量因子 ──────────
    vol = df["volume"].astype("float32")
    vol_ma20 = _roll(vol, code, 20, 15, "mean")
    vol_ma60 = _roll(vol, code, 60, 40, "mean")
    df["vol_confirm"] = (vol_ma20 / vol_ma60).astype("float32")
    del vol_ma60

    vol_ma5 = _roll(vol, code, 5, 4, "mean")
    df["volume_chg"] = (vol_ma5 / vol_ma20).astype("float32")
    del vol_ma5, vol_ma20
    gc.collect()

    # ────────── 价值因子 ──────────
    pb = df["pb"].replace(0, np.nan).astype("float32")
    df["inv_pb"] = (1.0 / pb).astype("float32")
    df.loc[df["pb"] < 0, "inv_pb"] = np.nan
    del pb

    # ────────── 规模因子 ──────────
    df["log_cap"] = np.log(df["free_market_cap"].replace(0, np.nan)).astype("float32")

    # ────────── 换手率因子 ──────────
    turnover_raw = (vol / df["free_market_cap"].replace(0, np.nan)).astype("float32")
    df["turnover_20"] = _roll(turnover_raw, code, 20, 15, "mean")
    del turnover_raw, vol
    gc.collect()

    # ────────── 财务因子 ──────────
    df["pe_ttm"]  = df["pe_ttm"].where(df["pe_ttm"] > 0).astype("float32")
    df["roe_ttm"] = df["roe_ttm"].astype("float32")

    # ────────── 技术形态因子 ──────────
    hi20 = _roll(df["high"], code, 20, 15, "max")
    lo20 = _roll(df["low"],  code, 20, 15, "min")
    df["high_low_20"] = ((hi20 - lo20) / df["close"]).astype("float32")
    del hi20, lo20

    hi60 = _roll(df["high"], code, 60, 40, "max")
    df["close_to_high_60"] = (df["close"] / hi60 - 1).astype("float32")
    del hi60
    gc.collect()

    print(f"[V2-data] 特征计算完成，形状: {df.shape}, "
          f"{df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return df


# ──────────────────────────────────────────────
# 3. 双周采样
# ──────────────────────────────────────────────

def sample_biweekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    双周采样：每半个月（上半=1~15日，下半=16日~月末）取最后一个交易日。
    period_sort: YYYYMM * 10 + half（1 或 2），用于排序。
    """
    print("[V2-data] 双周采样 ...")
    df = df.copy()
    y = df["date"].dt.year
    m = df["date"].dt.month
    h = (df["date"].dt.day > 15).astype(int) + 1  # 1=上半月, 2=下半月

    df["period_sort"] = (y * 100 + m) * 10 + h
    df["period"] = (
        y.astype(str) + "-"
        + m.astype(str).str.zfill(2) + "-H"
        + h.astype(str)
    )

    # 每个 (code, period) 取最后一个交易日
    snap = (
        df.sort_values("date")
          .groupby(["code", "period"], sort=False)
          .last()
          .reset_index()
    )

    print(f"[V2-data] {len(snap):,} 条记录，{snap['period'].nunique()} 个调仓期，"
          f"{snap.memory_usage(deep=True).sum()/1e6:.0f} MB")
    return snap


# ──────────────────────────────────────────────
# 4. 股票池过滤
# ──────────────────────────────────────────────

def filter_universe(snap: pd.DataFrame,
                    mcap_keep_pct: float = 0.70,
                    roe_floor: float = -20.0,
                    feature_cols: list | None = None) -> pd.DataFrame:
    """
    过滤规则（顺序）：
    1. 删除特征/必要字段的 NaN 行
    2. 去除市值后 (1 - mcap_keep_pct) 的小市值股
    3. ROE >= roe_floor（去除持续亏损）
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    required = feature_cols + ["close", "free_market_cap", "industry_code", "period"]
    before = len(snap)
    snap = snap.dropna(subset=required)
    print(f"[V2-data] 删除缺失值后: {len(snap):,}（删除 {before - len(snap):,}）")

    # 市值分位数过滤：每期内保留前 mcap_keep_pct
    cutoffs = snap.groupby("period")["free_market_cap"].quantile(1 - mcap_keep_pct)
    snap = snap.join(cutoffs.rename("_cap_cut"), on="period")
    snap = snap[snap["free_market_cap"] >= snap["_cap_cut"]].drop(columns=["_cap_cut"])
    print(f"[V2-data] 市值过滤后 (前{mcap_keep_pct:.0%}): {len(snap):,}")

    # ROE 过滤
    before_roe = len(snap)
    snap = snap[snap["roe_ttm"] >= roe_floor]
    print(f"[V2-data] ROE 过滤后 (>= {roe_floor}%): {len(snap):,}（删除 {before_roe - len(snap):,}）")

    return snap.reset_index(drop=True)


# ──────────────────────────────────────────────
# 5. 前瞻收益标签
# ──────────────────────────────────────────────

def build_forward_returns(snap: pd.DataFrame) -> pd.DataFrame:
    """
    构建前瞻收益标签。
    策略：持有一个双周调仓期。
    - entry_open：本期 open（当期调仓日开盘买入）
    - exit_open：下一调仓期的 open（下期调仓日开盘卖出）
    - fwd_ret = exit_open / entry_open - 1
    - label_end_date = 下期的 date（标签实现日，用于 purge）
    - label = fwd_ret 的截面百分位排序 → [0, 1]
    """
    print("[V2-data] 构建前瞻收益标签 ...")
    snap = snap.sort_values(["code", "period_sort"]).copy()

    # 下一期的 open 和 date（按股票排列）
    snap["exit_open"]       = snap.groupby("code", sort=False)["open"].shift(-1)
    snap["label_end_date"]  = snap.groupby("code", sort=False)["date"].shift(-1)

    snap["fwd_ret"] = (snap["exit_open"] / snap["open"] - 1).astype("float32")

    # 截面百分位排序 → label
    snap["label"] = snap.groupby("period")["fwd_ret"].rank(pct=True, na_option="keep")

    n_valid = snap["label"].notna().sum()
    print(f"[V2-data] 前瞻收益: {n_valid:,} 有效")
    return snap


# ──────────────────────────────────────────────
# 6. 特征截面归一化
# ──────────────────────────────────────────────

def cross_sectional_rank_norm(snap: pd.DataFrame,
                               feature_cols: list | None = None) -> pd.DataFrame:
    """每期内对所有特征做截面百分位排序归一化 → [0, 1]。"""
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    print("[V2-data] 截面排序归一化 ...")
    for col in feature_cols:
        if col not in snap.columns:
            print(f"  [警告] 特征 '{col}' 不存在，跳过")
            continue
        snap[col] = snap.groupby("period")[col].rank(pct=True, na_option="keep")

    print(f"[V2-data] 已归一化 {len(feature_cols)} 个特征")
    return snap


# ──────────────────────────────────────────────
# 7. 主入口
# ──────────────────────────────────────────────

def build_ml_dataset(
    warm_up_start: str,
    backtest_end: str,
    feature_cols: list | None = None,
    mcap_keep_pct: float = 0.70,
    rank_normalize: bool = True,
) -> pd.DataFrame:
    """
    完整数据管道：加载 → 特征 → 采样 → 过滤 → 标签 → 归一化。
    返回可直接用于模型训练/回测的 DataFrame。
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    # 加载到数据截止日的下一个月（确保 next_open 有数据）
    load_end = (pd.Timestamp(backtest_end) + pd.DateOffset(months=2)).strftime("%Y-%m-%d")

    raw   = load_raw(warm_up_start, load_end)
    raw   = compute_features(raw)
    snap  = sample_biweekly(raw)
    del raw  # 释放原始日线内存

    snap  = filter_universe(snap, mcap_keep_pct=mcap_keep_pct, feature_cols=feature_cols)
    snap  = build_forward_returns(snap)

    if rank_normalize:
        snap = cross_sectional_rank_norm(snap, feature_cols=feature_cols)

    # 只保留截止日前的数据
    snap = snap[snap["date"] <= pd.Timestamp(backtest_end)].reset_index(drop=True)

    n_stocks  = snap["code"].nunique()
    n_periods = snap["period"].nunique()
    n_valid   = snap["label"].notna().sum()
    mb = snap.memory_usage(deep=True).sum() / 1e6
    print(f"\n[V2-data] 最终数据集: {len(snap):,} 行, "
          f"{n_stocks} stocks, {n_periods} periods, "
          f"{n_valid:,} labeled ({mb:.0f} MB)")
    return snap
