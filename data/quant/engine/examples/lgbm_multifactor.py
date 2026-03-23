"""
LightGBM Multi-Factor ML Stock Selection Strategy (V3)
======================================================

Upgrades over linear multi-factor V2:
  - Non-linear factor interaction via LightGBM (gradient-boosted trees)
  - 15 features (6 base + 9 extended) with cross-sectional rank normalization
  - Rolling 3-year training window with purge gap
  - Cross-sectional ranking label (percentile of forward return)
  - Industry-constrained selection: top 5% by ML score, max 5 per industry
  - Buffer band: incumbents get +0.3σ bonus to reduce turnover
  - Universe: Main Board (沪深主板) + top 70% free-float market cap

All data access goes through DataAccessor to preserve look-ahead bias
protection provided by the backtest engine.

Usage
-----
cd <project_root>
python -m data.quant.engine.examples.lgbm_multifactor
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError(
        "LightGBM is required. Install via: pip install lightgbm"
    )

from engine import BacktestConfig, StrategyBase, run_backtest
from engine.data_loader import DataAccessor, get_rebalance_dates


# =========================================================================
# Constants
# =========================================================================

FEATURE_COLS = [
    "inv_pb", "roe_ttm", "pe_ttm", "log_cap",
    "mom_12_1", "turnover_20",
    "mom_3_1", "mom_6_1", "rev_10", "rvol_20",
    "ret_5d_std", "volume_chg", "high_low_20",
    "close_to_high_60", "vol_confirm",
]

# Columns needed from stock_daily for feature computation
_SNAP_COLS = [
    "close", "open", "high", "low",
    "pb", "pe_ttm", "roe", "circ_mv", "turnover_rate_f",
    "vol", "sw_l1", "is_suspended",
]

_WINDOW_COLS = ["close", "high", "low", "vol"]


# =========================================================================
# Feature Engineering Helpers
# =========================================================================

def _rank_normalize(s: pd.Series) -> pd.Series:
    """Cross-sectional rank normalization to [0, 1]."""
    return s.rank(pct=True, method="average")


def _compute_features_for_date(
    accessor: DataAccessor,
    date: pd.Timestamp,
    market_filter: Optional[set] = None,
) -> pd.DataFrame:
    """
    Compute all 15 features for a single date using DataAccessor.
    All queries go through accessor for look-ahead protection.

    Returns DataFrame with ts_code, sw_l1, circ_mv + 15 feature columns.
    """
    # ── Single-date snapshot ──
    snap = accessor.get_date(date, columns=_SNAP_COLS)
    if snap.empty:
        return pd.DataFrame()

    # Apply market filter (main board only)
    if market_filter is not None:
        snap = snap[snap["ts_code"].isin(market_filter)].copy()

    # Basic filters: not suspended, valid close/pb/circ_mv/industry
    snap = snap[
        (snap["is_suspended"] != 1)
        & (snap["close"].notna()) & (snap["close"] > 0)
        & (snap["pb"].notna()) & (snap["pb"] > 0)
        & (snap["circ_mv"].notna()) & (snap["circ_mv"] > 0)
        & (snap["sw_l1"].notna())
    ].copy()

    if len(snap) < 50:
        return pd.DataFrame()

    # ── Base features (from snapshot) ──
    snap["inv_pb"] = 1.0 / snap["pb"]
    snap["roe_ttm"] = snap["roe"].fillna(0)
    # pe_ttm already exists in the data
    snap["log_cap"] = np.log(snap["circ_mv"].clip(lower=1))
    snap["turnover_20"] = snap["turnover_rate_f"].fillna(0)

    # ── Window-based features (via accessor.get_window) ──
    # Load 253 trade days for 12-month momentum calculations
    window_data = accessor.get_window(date, lookback=253, columns=_WINDOW_COLS)
    if window_data.empty or len(window_data["trade_date"].unique()) < 22:
        return pd.DataFrame()

    # Build pivot tables for vectorised computation
    close_pivot = window_data.pivot_table(
        index="trade_date", columns="ts_code", values="close"
    ).sort_index()
    n_dates = len(close_pivot)

    # Feature container — one row per ts_code that appears in the window
    features = pd.DataFrame({"ts_code": close_pivot.columns})

    # mom_12_1: return from t-253 to t-22 (12mo, skip most recent 1mo)
    if n_dates >= 253:
        features["mom_12_1"] = (
            close_pivot.iloc[-22] / close_pivot.iloc[0] - 1
        ).values
    else:
        features["mom_12_1"] = np.nan

    # mom_6_1: return from t-127 to t-22
    if n_dates >= 127:
        idx_start = max(0, n_dates - 127)
        features["mom_6_1"] = (
            close_pivot.iloc[-22] / close_pivot.iloc[idx_start] - 1
        ).values
    else:
        features["mom_6_1"] = np.nan

    # mom_3_1: return from t-64 to t-22
    if n_dates >= 64:
        idx_start = max(0, n_dates - 64)
        features["mom_3_1"] = (
            close_pivot.iloc[-22] / close_pivot.iloc[idx_start] - 1
        ).values
    else:
        features["mom_3_1"] = np.nan

    # rev_10: short-term reversal (10-day return)
    if n_dates >= 11:
        features["rev_10"] = (
            close_pivot.iloc[-1] / close_pivot.iloc[-11] - 1
        ).values
    else:
        features["rev_10"] = np.nan

    # rvol_20: 20-day realised volatility
    if n_dates >= 21:
        _w = close_pivot.iloc[-21:]
        daily_ret = (_w / _w.shift(1) - 1).iloc[1:]
        features["rvol_20"] = daily_ret.std().values
    else:
        features["rvol_20"] = np.nan

    # ret_5d_std: 5-day return std (micro volatility)
    if n_dates >= 6:
        _w5 = close_pivot.iloc[-6:]
        daily_ret_5 = (_w5 / _w5.shift(1) - 1).iloc[1:]
        features["ret_5d_std"] = daily_ret_5.std().values
    else:
        features["ret_5d_std"] = np.nan

    # volume_chg: volume change ratio (recent 5d avg / prior 20d avg)
    vol_pivot = window_data.pivot_table(
        index="trade_date", columns="ts_code", values="vol"
    ).sort_index()
    if len(vol_pivot) >= 25:
        vol_recent = vol_pivot.iloc[-5:].mean()
        vol_prior = vol_pivot.iloc[-25:-5].mean()
        features["volume_chg"] = (
            vol_recent / vol_prior.clip(lower=1e-8) - 1
        ).values
    else:
        features["volume_chg"] = np.nan

    # high_low_20: 20-day high-low range / close
    high_pivot = window_data.pivot_table(
        index="trade_date", columns="ts_code", values="high"
    ).sort_index()
    low_pivot = window_data.pivot_table(
        index="trade_date", columns="ts_code", values="low"
    ).sort_index()
    if len(high_pivot) >= 20:
        high_20 = high_pivot.iloc[-20:].max()
        low_20 = low_pivot.iloc[-20:].min()
        last_close = close_pivot.iloc[-1]
        features["high_low_20"] = (
            (high_20 - low_20) / last_close.clip(lower=1e-8)
        ).values
    else:
        features["high_low_20"] = np.nan

    # close_to_high_60: distance from 60-day high
    if n_dates >= 60:
        high_60 = close_pivot.iloc[-60:].max()
        last_close = close_pivot.iloc[-1]
        features["close_to_high_60"] = (
            last_close / high_60.clip(lower=1e-8) - 1
        ).values
    else:
        features["close_to_high_60"] = np.nan

    # vol_confirm: momentum × volume_chg interaction
    if "rev_10" in features.columns and "volume_chg" in features.columns:
        features["vol_confirm"] = (
            features["rev_10"].fillna(0) * features["volume_chg"].fillna(0)
        )
    else:
        features["vol_confirm"] = np.nan

    # ── Merge window features back to snapshot ──
    snap = snap.merge(features, on="ts_code", how="inner", suffixes=("_snap", "_feat"))

    # Resolve column conflicts (snapshot vs features)
    for col in FEATURE_COLS:
        if col + "_feat" in snap.columns:
            snap[col] = snap[col + "_feat"]
        elif col not in snap.columns:
            snap[col] = np.nan

    # ── Universe filter: top 70% by free-float market cap ──
    cap_threshold = snap["circ_mv"].quantile(0.30)
    snap = snap[snap["circ_mv"] >= cap_threshold].copy()

    if len(snap) < 30:
        return pd.DataFrame()

    # ── Rank normalise all features to [0, 1] ──
    for col in FEATURE_COLS:
        if col in snap.columns:
            snap[col] = _rank_normalize(snap[col].fillna(snap[col].median()))

    result_cols = ["ts_code", "sw_l1", "circ_mv"] + FEATURE_COLS
    result_cols = [c for c in result_cols if c in snap.columns]
    return snap[result_cols].copy()


# =========================================================================
# Training via DataAccessor
# =========================================================================

def _make_training_accessor(cfg: BacktestConfig,
                            train_start: pd.Timestamp,
                            current_date: pd.Timestamp) -> DataAccessor:
    """
    Create a DataAccessor for the training window.

    The accessor's date range covers [train_start - 1y buffer, current_date],
    and its current_date guard is set to current_date so that it can never
    peek into the future.
    """
    # Extend start backward by 1 year to ensure enough lookback for features
    buf_start = train_start - pd.DateOffset(years=1)
    train_cfg = BacktestConfig(
        initial_capital=cfg.initial_capital,
        commission_rate=cfg.commission_rate,
        slippage=cfg.slippage,
        start_date=buf_start.strftime("%Y-%m-%d"),
        end_date=current_date.strftime("%Y-%m-%d"),
        rebalance_freq=cfg.rebalance_freq,
        db_path=cfg.db_path,
        baseline_dir=cfg.baseline_dir,
        output_dir=cfg.output_dir,
    )
    acc = DataAccessor(train_cfg)
    acc.open()
    acc.set_current_date(current_date)
    return acc


def _get_biweekly_dates_from_accessor(
    accessor: DataAccessor,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[pd.Timestamp]:
    """
    Derive biweekly rebalance dates within [start, end] by querying
    trade dates via the accessor's SQL connection (no look-ahead issue
    since end <= current_date).
    """
    # Query trade dates in range through accessor's guarded connection
    accessor._check_look_ahead(end, "_get_biweekly_dates")
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    sql = """
        SELECT DISTINCT trade_date FROM stock_daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """
    df = pd.read_sql_query(sql, accessor.conn, params=(start_str, end_str))
    if df.empty:
        return []

    dates = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    dti = pd.DatetimeIndex(dates)
    return list(get_rebalance_dates(dti, "BW"))


def _build_forward_returns(
    accessor: DataAccessor,
    rebal_dates: List[pd.Timestamp],
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Compute forward returns for each rebalance date using accessor.
    forward_return = close[next_rebal] / close[current_rebal] - 1
    """
    fwd_map: Dict[pd.Timestamp, Dict[str, float]] = {}
    for i in range(len(rebal_dates) - 1):
        d_curr = rebal_dates[i]
        d_next = rebal_dates[i + 1]

        prices_curr = accessor.get_prices(d_curr)
        prices_next = accessor.get_prices(d_next)

        if not prices_curr or not prices_next:
            continue

        rets = {}
        for code, p0 in prices_curr.items():
            if code in prices_next and p0 > 0:
                p1 = prices_next[code]
                if p1 > 0:
                    rets[code] = p1 / p0 - 1
        if rets:
            fwd_map[d_curr] = rets

    return fwd_map


def _build_training_data(
    accessor: DataAccessor,
    forward_ret_map: Dict[pd.Timestamp, Dict[str, float]],
    market_filter: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build training X, y for the training window.

    All data access goes through accessor, which has its current_date
    guard set to the current simulation date.
    """
    all_X = []
    all_y = []

    for date, fwd_rets in forward_ret_map.items():
        features_df = _compute_features_for_date(accessor, date, market_filter)
        if features_df.empty:
            continue

        # Only keep stocks that have forward returns
        fwd_series = pd.Series(fwd_rets, name="fwd_ret")
        features_df = features_df.merge(
            fwd_series.reset_index().rename(columns={"index": "ts_code"}),
            on="ts_code", how="inner",
        )

        if len(features_df) < 30:
            continue

        # Label: rank percentile of forward return
        features_df["label"] = features_df["fwd_ret"].rank(
            pct=True, method="average"
        )

        feat_cols = [c for c in FEATURE_COLS if c in features_df.columns]
        all_X.append(features_df[feat_cols])
        all_y.append(features_df["label"])

    if not all_X:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    return X, y


# =========================================================================
# LightGBM Multi-Factor Strategy
# =========================================================================

class LGBMMultifactorStrategy(StrategyBase):
    """
    LightGBM-based multi-factor stock selection strategy.

    All data access uses DataAccessor for look-ahead bias protection.

    - 3-year rolling training window
    - 15 features (rank-normalised)
    - Cross-sectional ranking label
    - Top 5% selection, max 5 per industry, equal weight
    - Buffer band for incumbents
    """

    def __init__(
        self,
        train_years: int = 3,
        top_pct: float = 0.05,
        max_per_industry: int = 5,
        buffer_sigma: float = 0.3,
    ):
        super().__init__("lgbm_multifactor")
        self.train_years = train_years
        self.top_pct = top_pct
        self.max_per_industry = max_per_industry
        self.buffer_sigma = buffer_sigma
        self._model: Optional[lgb.Booster] = None
        self._last_train_date: Optional[str] = None
        self._retrain_interval = 4  # retrain every N rebalance periods
        self._rebal_count = 0
        self._market_stocks: Optional[set] = None

    def describe(self) -> str:
        return (
            "### 策略思路\n\n"
            "基于 LightGBM 梯度提升树的截面多因子 ML 选股策略。"
            "使用 15 个特征（6 基础 + 9 扩展），通过 3 年滚动窗口训练，"
            "预测截面排名分位数（而非绝对收益），消除收益分布非平稳性。"
            "在全市场沪深主板中选取自由流通市值前 70% 的股票，"
            "按 ML 预测分数取前 5%，每行业最多 5 只，等权持有，双周调仓。\n\n"
            "### 因子列表（15个特征）\n\n"
            "| 因子 | 描述 | 类别 |\n"
            "|------|------|------|\n"
            "| inv_pb | 市净率倒数 | 价值 |\n"
            "| roe_ttm | ROE（TTM） | 质量 |\n"
            "| pe_ttm | 市盈率（TTM） | 价值 |\n"
            "| log_cap | 对数自由流通市值 | 规模 |\n"
            "| mom_12_1 | 12月动量（跳过最近1月） | 动量 |\n"
            "| turnover_20 | 20日换手率 | 流动性 |\n"
            "| mom_3_1 | 3月动量（跳过最近1月） | 短期动量 |\n"
            "| mom_6_1 | 6月动量（跳过最近1月） | 中期动量 |\n"
            "| rev_10 | 10日反转 | 反转 |\n"
            "| rvol_20 | 20日已实现波动率 | 风险 |\n"
            "| ret_5d_std | 5日收益标准差 | 微观波动 |\n"
            "| volume_chg | 量比变化（5日/20日） | 量价 |\n"
            "| high_low_20 | 20日振幅比 | 风险 |\n"
            "| close_to_high_60 | 收盘价距60日高点 | 价格距离 |\n"
            "| vol_confirm | 动量×量变交互项 | 量价确认 |\n\n"
            "### 技术细节\n\n"
            "1. **特征预处理**：所有特征在每期截面内做排序归一化映射到 [0,1]\n"
            "2. **标签构建**：截面排序标签（forward return 百分位排名）\n"
            "3. **训练窗口**：3 年滚动窗口，每 4 个调仓周期重训一次\n"
            "4. **Purge**：训练数据 forward return 计算不超出训练窗口末尾\n"
            "5. **模型**：LightGBM 回归模式，预测排名分位\n"
            "6. **选股**：ML 分数前 5%，每行业最多 5 只\n"
            "7. **缓冲带**：持仓股 ML 分数 +0.3σ 加分，降低换手\n"
            "8. **选股范围**：沪深主板，自由流通市值前 70%\n"
            "9. **数据访问**：全部通过 DataAccessor，受前视偏差保护\n\n"
            "### 已知局限\n\n"
            "- 训练窗口固定 3 年，未做窗口长度敏感性测试\n"
            "- 未使用交叉验证选超参\n"
            "- 未做因子 IC/IR 分析\n"
            "- 未考虑涨跌停限制\n"
        )

    def _ensure_market_filter(self, accessor: DataAccessor):
        """Load main-board stock codes once via accessor's DB connection."""
        if self._market_stocks is not None:
            return
        # stock_info is a static reference table — no look-ahead concern
        sql = "SELECT ts_code FROM stock_info WHERE market = '主板'"
        df = pd.read_sql_query(sql, accessor.conn)
        self._market_stocks = set(df["ts_code"].tolist())
        print(f"      [ML] 主板股票池: {len(self._market_stocks)} 只")

    def _train_model(self, accessor: DataAccessor, current_date: pd.Timestamp):
        """Train LightGBM on a rolling 3-year window using DataAccessor."""
        cfg = accessor.cfg

        # Training window: [current - 3y, current - 1month purge gap]
        train_start_dt = current_date - pd.DateOffset(years=self.train_years)
        train_end_dt = current_date - pd.DateOffset(months=1)

        print(f"      [ML] 训练窗口: "
              f"{train_start_dt.strftime('%Y%m%d')} ~ "
              f"{train_end_dt.strftime('%Y%m%d')}")

        # Create a dedicated training accessor with wider date range
        train_acc = _make_training_accessor(cfg, train_start_dt, current_date)

        try:
            # Get biweekly rebalance dates in training window
            rebal_dates = _get_biweekly_dates_from_accessor(
                train_acc, train_start_dt, train_end_dt
            )
            if len(rebal_dates) < 10:
                print(f"      [ML] 训练数据不足 ({len(rebal_dates)} 期), 跳过训练")
                return

            # Build forward returns via accessor
            fwd_map = _build_forward_returns(train_acc, rebal_dates)
            if len(fwd_map) < 8:
                print(f"      [ML] forward return 数据不足 ({len(fwd_map)} 期)")
                return

            # Build training data via accessor
            X, y = _build_training_data(train_acc, fwd_map, self._market_stocks)
        finally:
            train_acc.close()

        if len(X) < 100:
            print(f"      [ML] 训练样本不足: {len(X)}")
            return

        # Drop NaN rows
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

        if len(X) < 100:
            print(f"      [ML] 有效训练样本不足: {len(X)}")
            return

        print(f"      [ML] 训练样本: {len(X)} 行, {X.shape[1]} 特征")

        # Split: last 20% as validation (temporal split)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
            "seed": 42,
        }

        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        self._last_train_date = current_date.strftime("%Y%m%d")
        print(f"      [ML] 模型训练完成, best_iteration={self._model.best_iteration}")

    def generate_target_weights(
        self,
        date: pd.Timestamp,
        accessor: DataAccessor,
        current_holdings: Dict[str, int],
    ) -> Dict[str, float]:

        self._ensure_market_filter(accessor)
        self._rebal_count += 1

        # ── Retrain model periodically ──
        need_train = (
            self._model is None
            or (self._rebal_count % self._retrain_interval == 1)
        )
        if need_train:
            self._train_model(accessor, date)

        if self._model is None:
            print(f"      [ML] 无可用模型, 返回空权重")
            return {}

        # ── Compute features for current date (via accessor) ──
        features_df = _compute_features_for_date(
            accessor, date, self._market_stocks
        )

        if features_df.empty or len(features_df) < 30:
            return {}

        # ── Predict ML scores ──
        feat_cols = [c for c in FEATURE_COLS if c in features_df.columns]
        X_pred = features_df[feat_cols].fillna(0.5)  # median rank for NaN

        scores = self._model.predict(X_pred)
        features_df["ml_score"] = scores

        # ── Buffer band: incumbents get +0.3σ bonus ──
        if current_holdings and self.buffer_sigma > 0:
            score_std = features_df["ml_score"].std()
            bonus = self.buffer_sigma * score_std
            held_codes = set(current_holdings.keys())
            features_df.loc[
                features_df["ts_code"].isin(held_codes), "ml_score"
            ] += bonus

        # ── Selection: top 5% ──
        n_select = max(1, int(len(features_df) * self.top_pct))
        top_candidates = features_df.nlargest(n_select * 2, "ml_score").copy()

        # ── Industry constraint: max 5 per industry ──
        if "sw_l1" in top_candidates.columns:
            top_candidates = top_candidates.sort_values(
                "ml_score", ascending=False
            )
            selected = []
            industry_count: Dict[str, int] = {}
            for _, row in top_candidates.iterrows():
                ind = row["sw_l1"]
                cnt = industry_count.get(ind, 0)
                if cnt < self.max_per_industry:
                    selected.append(row)
                    industry_count[ind] = cnt + 1
                if len(selected) >= n_select:
                    break
            if not selected:
                return {}
            selected_df = pd.DataFrame(selected)
        else:
            selected_df = top_candidates.head(n_select)

        # ── Equal weight ──
        codes = selected_df["ts_code"].tolist()
        if not codes:
            return {}

        w = 1.0 / len(codes)
        return {c: w for c in codes}


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=1_000_000,
        commission_rate=1.5e-4,      # 1.5 bps single-side commission
        slippage=0.0,
        start_date="2025-02-28",
        end_date="2026-02-28",
        rebalance_freq="BW",         # biweekly
        db_path="data/quant/data/quant.db",
        baseline_dir="data/quant/baseline",
        output_dir="data/quant/backtest",
    )

    strategy = LGBMMultifactorStrategy(
        train_years=3,
        top_pct=0.05,
        max_per_industry=5,
        buffer_sigma=0.3,
    )
    result = run_backtest(strategy, cfg)
