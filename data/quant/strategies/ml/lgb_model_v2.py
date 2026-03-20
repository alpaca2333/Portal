"""
LightGBM v2: Expanding-Window Dual-Ensemble Model
===================================================
A completely new implementation — shares NO code with lgb_model.py.

Key design differences from v1:
1. **Expanding window** (not rolling): training set grows over time,
   with exponential decay weighting so recent data matters more.
2. **Dual ensemble**: a regression head predicting excess returns +
   a LambdaRank head optimizing pairwise ranking; final score =
   weighted blend of both.
3. **Industry-neutral labels**: forward returns are neutralized against
   industry mean before training, so the model learns stock-specific alpha.
4. **Quantile portfolio analysis**: top/bottom quintile spread, monotonicity.
5. **Turnover-adjusted IC**: penalizes signal instability across periods.
6. **Purged + embargoed CV**: configurable embargo gap between train/test.

Usage:
    from strategies.ml.lgb_model_v2 import ExpandingLGBEnsemble
    ens = ExpandingLGBEnsemble()
    preds = ens.expanding_train_predict(dataset, feature_cols)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm")


# ───────────────────── Hyperparameter Presets ───────────────────

REGRESSOR_PARAMS = {
    "objective": "huber",           # robust to outliers in return residuals
    "alpha": 0.9,                   # huber delta
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 63,               # slightly more complex than v1's 31
    "max_depth": 7,
    "learning_rate": 0.03,          # slower learning, relies on early stopping
    "n_estimators": 2000,
    "min_child_samples": 200,       # stricter than v1 to combat overfitting
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 2024,
    "verbose": -1,
    "n_jobs": -1,
}

RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [50, 100],
    "boosting_type": "gbdt",
    "num_leaves": 48,
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 1500,
    "min_child_samples": 150,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "random_state": 2024,
    "verbose": -1,
    "n_jobs": -1,
    "label_gain": None,  # auto
}


# ───────────────────── Evaluation Toolkit ───────────────────────

def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (Rank IC)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 20:
        return np.nan
    return stats.spearmanr(y_true[mask], y_pred[mask]).statistic


def pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation (normal IC)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 20:
        return np.nan
    return np.corrcoef(y_true[mask], y_pred[mask])[0, 1]


def quantile_returns(
    pred: np.ndarray,
    fwd_ret: np.ndarray,
    n_quantiles: int = 5,
) -> Dict[str, float]:
    """
    Compute mean forward return for each quantile of predicted scores.

    Returns dict with keys Q1..Q5, long_short_spread, monotonicity_score.
    """
    mask = np.isfinite(pred) & np.isfinite(fwd_ret)
    pred_clean = pred[mask]
    ret_clean = fwd_ret[mask]
    if len(pred_clean) < n_quantiles * 10:
        return {}

    # pd.qcut for equal-frequency bins
    try:
        bins = pd.qcut(pred_clean, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return {}

    result = {}
    means = []
    for q in range(n_quantiles):
        q_mask = bins == q
        mean_ret = ret_clean[q_mask].mean() if q_mask.sum() > 0 else np.nan
        result[f"Q{q + 1}"] = mean_ret
        means.append(mean_ret)

    # Long-short spread: top quantile - bottom quantile
    result["long_short_spread"] = means[-1] - means[0] if len(means) >= 2 else np.nan

    # Monotonicity: Spearman corr between quantile index and mean return
    if len(means) >= 3:
        valid = [(i, m) for i, m in enumerate(means) if np.isfinite(m)]
        if len(valid) >= 3:
            idx, vals = zip(*valid)
            result["monotonicity"] = stats.spearmanr(idx, vals).statistic
        else:
            result["monotonicity"] = np.nan
    else:
        result["monotonicity"] = np.nan

    return result


def turnover_adjusted_ic(
    ic_series: pd.Series,
    turnover_series: pd.Series,
    decay: float = 0.5,
) -> float:
    """
    IC adjusted for signal turnover.
    Higher turnover → more transaction cost → penalize IC.

    adjusted_IC = IC * (1 - decay * turnover)
    """
    mask = ic_series.notna() & turnover_series.notna()
    if mask.sum() == 0:
        return np.nan
    ic = ic_series[mask].values
    to = turnover_series[mask].values
    adjusted = ic * (1 - decay * np.clip(to, 0, 1))
    return float(np.nanmean(adjusted))


def summarize_ic(ic_series: pd.Series) -> Dict[str, float]:
    """Comprehensive IC summary statistics."""
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {k: np.nan for k in
                ["mean_ic", "median_ic", "std_ic", "ir",
                 "ic_gt0_rate", "ic_gt005_rate", "n_periods"]}
    mean = ic.mean()
    std = ic.std(ddof=1) if len(ic) > 1 else np.nan
    return {
        "mean_ic": mean,
        "median_ic": ic.median(),
        "std_ic": std,
        "ir": mean / std if std > 0 else np.nan,
        "ic_gt0_rate": (ic > 0).mean(),
        "ic_gt005_rate": (ic > 0.05).mean(),
        "n_periods": len(ic),
    }


# ───────────────────── Sample Weighting ─────────────────────────

def exponential_decay_weights(
    dates: pd.Series,
    reference_date: pd.Timestamp,
    half_life_days: int = 365,
) -> np.ndarray:
    """
    Compute exponential decay weights based on distance from reference_date.

    Samples closer to reference_date get higher weights.
    Half-life = number of days for weight to halve.

    Returns array of weights in [0, 1], same length as dates.
    """
    days_ago = (reference_date - pd.to_datetime(dates)).dt.days.values.astype(float)
    days_ago = np.maximum(days_ago, 0)
    lam = np.log(2) / half_life_days
    weights = np.exp(-lam * days_ago)
    return weights.astype(np.float32)


# ───────────────────── Industry Neutralization ──────────────────

def neutralize_by_industry(
    df: pd.DataFrame,
    target_col: str,
    industry_col: str = "industry_code",
    period_col: str = "period",
) -> pd.Series:
    """
    Neutralize a column by subtracting industry mean within each period.

    This removes industry beta, leaving only stock-specific residual.
    """
    group_mean = df.groupby([period_col, industry_col])[target_col].transform("mean")
    return (df[target_col] - group_mean).astype(np.float32)


def zscore_cross_section(
    df: pd.DataFrame,
    cols: List[str],
    period_col: str = "period",
    clip: float = 3.0,
) -> pd.DataFrame:
    """
    Cross-sectional z-score normalization with winsorization.

    For each period, subtract mean and divide by std, then clip to [-clip, +clip].
    This preserves distance information (unlike rank normalization).
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        grp = df.groupby(period_col)[col]
        mu = grp.transform("mean")
        sigma = grp.transform("std")
        sigma = sigma.replace(0, np.nan)
        z = ((df[col] - mu) / sigma).clip(-clip, clip)
        df[col] = z.astype(np.float32)
    return df


# ───────────────────── Ranker Data Helpers ──────────────────────

def _build_group_sizes(period_series: pd.Series) -> np.ndarray:
    """Build LightGBM ranker group sizes from period column."""
    return period_series.value_counts(sort=False).reindex(
        period_series.unique()
    ).values


def _relevance_labels(fwd_ret: pd.Series, n_bins: int = 5) -> np.ndarray:
    """
    Convert continuous forward returns into ordinal relevance labels
    for LambdaRank (0 = worst, n_bins-1 = best) within each period.
    """
    labels = np.zeros(len(fwd_ret), dtype=np.int32)
    try:
        labels[:] = pd.qcut(fwd_ret, n_bins, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        # fallback: simple rank-based bins
        ranks = fwd_ret.rank(pct=True, na_option="keep")
        labels[:] = (ranks * (n_bins - 1)).fillna(0).astype(np.int32)
    return labels


# ───────────────────── Model Class ──────────────────────────────

@dataclass
class ExpandingLGBEnsemble:
    """
    LightGBM v2: Expanding-window dual-ensemble model for stock selection.

    Architecture:
    - Head A (Regressor): Huber loss on industry-neutralized excess returns.
    - Head B (Ranker): LambdaRank on intra-period relevance labels.
    - Final score = alpha * rank(regressor_pred) + (1-alpha) * rank(ranker_pred)

    Training:
    - Expanding window: each step adds new data without discarding old data.
    - Exponential decay weighting: recent samples weighted more heavily.
    - Purge + embargo: configurable gap between train and test.

    Parameters
    ----------
    reg_params : dict
        Regressor LightGBM parameters.
    rank_params : dict
        Ranker LightGBM parameters.
    initial_train_years : int
        Minimum training history before first prediction.
    step_months : int
        How many months of new data to add each step.
    embargo_months : int
        Gap between end of training and start of test (prevents label leakage).
    early_stopping_rounds : int
        Early stopping patience.
    ensemble_alpha : float
        Weight of regressor in final blend (0 = pure ranker, 1 = pure regressor).
    decay_half_life_days : int
        Exponential decay half-life for sample weighting.
    use_ranker : bool
        Whether to train the LambdaRank head (set False for faster iteration).
    """
    reg_params: Dict = field(default_factory=lambda: REGRESSOR_PARAMS.copy())
    rank_params: Dict = field(default_factory=lambda: RANKER_PARAMS.copy())
    initial_train_years: int = 4
    step_months: int = 6
    embargo_months: int = 1
    early_stopping_rounds: int = 80
    ensemble_alpha: float = 0.6
    decay_half_life_days: int = 540   # ~1.5 years
    use_ranker: bool = True

    # State
    feature_cols: List[str] = field(default_factory=list)
    reg_models: List[Any] = field(default_factory=list)
    rank_models: List[Any] = field(default_factory=list)
    eval_history: List[Dict] = field(default_factory=list)
    importance_history: List[Dict[str, float]] = field(default_factory=list)
    quantile_history: List[Dict] = field(default_factory=list)

    # ──────────── Training ────────────

    def _fit_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        w_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> lgb.LGBMRegressor:
        """Fit the regression head with sample weights and early stopping."""
        model = lgb.LGBMRegressor(**self.reg_params)

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        return model

    def _fit_ranker(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        group_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        group_val: np.ndarray,
    ) -> lgb.LGBMRanker:
        """Fit the LambdaRank head."""
        params = {k: v for k, v in self.rank_params.items()
                  if k not in ("label_gain", "eval_at")}
        model = lgb.LGBMRanker(**params)

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model.fit(
            X_train, y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            callbacks=callbacks,
        )
        return model

    def _blend_predictions(
        self,
        reg_pred: np.ndarray,
        rank_pred: Optional[np.ndarray],
        period_labels: pd.Series,
    ) -> np.ndarray:
        """
        Blend regressor and ranker predictions via rank-space averaging.

        Within each period, convert both predictions to percentile ranks,
        then blend with ensemble_alpha.
        """
        reg_ranks = np.full_like(reg_pred, np.nan)
        rank_ranks = np.full_like(reg_pred, np.nan) if rank_pred is not None else None

        for period in period_labels.unique():
            mask = (period_labels == period).values
            n = mask.sum()
            if n < 2:
                continue
            # Percentile rank within period
            reg_ranks[mask] = stats.rankdata(reg_pred[mask]) / n
            if rank_pred is not None and rank_ranks is not None:
                rank_ranks[mask] = stats.rankdata(rank_pred[mask]) / n

        if rank_pred is not None and rank_ranks is not None:
            alpha = self.ensemble_alpha
            blended = alpha * reg_ranks + (1 - alpha) * rank_ranks
        else:
            blended = reg_ranks

        return blended

    # ──────────── Main Pipeline ────────────

    def expanding_train_predict(
        self,
        dataset: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        fwd_ret_col: str = "fwd_ret",
        min_date: str = "2013-01-01",
        max_date: str = "2026-02-28",
    ) -> pd.DataFrame:
        """
        Full expanding-window training and prediction pipeline.

        For each step t:
        1. Train on [min_date, t) with exponential decay weights
        2. Embargo gap of `embargo_months`
        3. Predict on [t + embargo, t + embargo + step_months)
        4. Evaluate: Rank IC, quantile returns, feature importance

        Parameters
        ----------
        dataset : DataFrame from data_prep.build_ml_dataset()
            Must contain: code, date, period, period_sort, industry_code,
            fwd_ret, label, and all feature columns.
        feature_cols : list of feature column names
        label_col : column name for the training label
        fwd_ret_col : column name for raw forward returns (for evaluation)
        min_date : earliest date for data
        max_date : latest date

        Returns
        -------
        DataFrame with all out-of-sample predictions, including:
            code, date, period, pred_score, fwd_ret, label,
            industry_code, free_market_cap, close, reg_pred, rank_pred
        """
        self.feature_cols = feature_cols
        self.reg_models = []
        self.rank_models = []
        self.eval_history = []
        self.importance_history = []
        self.quantile_history = []

        ds = dataset.copy()
        ds["_date"] = pd.to_datetime(ds["date"])
        if "label_end_date" in ds.columns:
            ds["_label_end_date"] = pd.to_datetime(ds["label_end_date"])

        min_dt = pd.Timestamp(min_date)
        max_dt = pd.Timestamp(max_date)

        # Industry-neutralize the label for the regressor
        ds["_neutral_label"] = neutralize_by_industry(
            ds, label_col, "industry_code", "period"
        )

        # Build relevance labels for the ranker (per-period binning)
        ds["_relevance"] = np.nan
        for period, g in ds.groupby("period"):
            valid = g[fwd_ret_col].notna()
            if valid.sum() < 20:
                continue
            ds.loc[g.index[valid], "_relevance"] = _relevance_labels(
                g.loc[valid, fwd_ret_col], n_bins=5
            )

        # First possible test start: initial_train_years after min_date
        first_test = min_dt + pd.DateOffset(years=self.initial_train_years)

        all_preds = []
        step_idx = 0
        t = first_test

        while t <= max_dt:
            step_idx += 1
            train_end = t - pd.DateOffset(days=1)
            embargo_start = t
            test_start = t + pd.DateOffset(months=self.embargo_months)
            test_end = min(
                test_start + pd.DateOffset(months=self.step_months) - pd.DateOffset(days=1),
                max_dt,
            )

            if test_start > max_dt:
                break

            # ── Slice data ──
            # Training: all data from min_date to train_end (expanding!)
            train_mask = (ds["_date"] >= min_dt) & (ds["_date"] <= train_end)
            if "label_end_date" in ds.columns:
                # Purge: label must be fully realized before embargo start
                train_mask &= ds["_label_end_date"].notna()
                train_mask &= ds["_label_end_date"] < embargo_start

            train_df = ds.loc[train_mask].copy()

            # Test set
            test_mask = (ds["_date"] >= test_start) & (ds["_date"] <= test_end)
            test_df = ds.loc[test_mask].copy()

            if len(train_df) < 1000 or len(test_df) == 0:
                t += pd.DateOffset(months=self.step_months)
                continue

            # ── Validation split: last 15% of training data chronologically ──
            train_sorted = train_df.sort_values("_date")
            split_idx = int(len(train_sorted) * 0.85)
            val_df = train_sorted.iloc[split_idx:]
            train_core = train_sorted.iloc[:split_idx]

            print(f"\n[v2-model] 步骤 {step_idx}: "
                  f"训练 {min_dt.date()}~{train_end.date()} ({len(train_core):,} + 验证{len(val_df):,}), "
                  f"测试 {test_start.date()}~{test_end.date()} ({len(test_df):,})")

            # ── Sample weights (exponential decay) ──
            weights = exponential_decay_weights(
                train_core["_date"], train_end,
                half_life_days=self.decay_half_life_days,
            )

            # ── Features ──
            X_train = train_core[feature_cols]
            X_val = val_df[feature_cols]
            X_test = test_df[feature_cols]

            # ── HEAD A: Regressor on neutralized label ──
            y_train_reg = train_core["_neutral_label"].values
            y_val_reg = val_df["_neutral_label"].values

            valid_train = np.isfinite(y_train_reg)
            valid_val = np.isfinite(y_val_reg)

            reg_model = self._fit_regressor(
                X_train.loc[valid_train], y_train_reg[valid_train],
                weights[valid_train],
                X_val.loc[valid_val], y_val_reg[valid_val],
            )
            self.reg_models.append(reg_model)

            best_reg = reg_model.best_iteration_ if reg_model.best_iteration_ > 0 else reg_model.n_estimators
            print(f"  [回归头] 最佳迭代: {best_reg}")

            # Record feature importance (regressor)
            imp = dict(zip(feature_cols, reg_model.feature_importances_))
            self.importance_history.append(imp)

            # Regressor prediction on test
            reg_pred = reg_model.predict(X_test)

            # ── HEAD B: Ranker (optional) ──
            rank_pred = None
            if self.use_ranker:
                try:
                    # Prepare group-based data for ranker
                    # Train: sort by period, build groups
                    train_rank = train_core.sort_values("period").copy()
                    y_rank_train = train_rank["_relevance"].values
                    valid_rank = np.isfinite(y_rank_train)
                    train_rank = train_rank[valid_rank]
                    y_rank_train = y_rank_train[valid_rank].astype(np.int32)
                    group_train = _build_group_sizes(train_rank["period"])

                    val_rank = val_df.sort_values("period").copy()
                    y_rank_val = val_rank["_relevance"].values
                    valid_rank_v = np.isfinite(y_rank_val)
                    val_rank = val_rank[valid_rank_v]
                    y_rank_val = y_rank_val[valid_rank_v].astype(np.int32)
                    group_val = _build_group_sizes(val_rank["period"])

                    if len(group_train) > 0 and len(group_val) > 0:
                        rank_model = self._fit_ranker(
                            train_rank[feature_cols], y_rank_train, group_train,
                            val_rank[feature_cols], y_rank_val, group_val,
                        )
                        self.rank_models.append(rank_model)
                        best_rank = rank_model.best_iteration_ if rank_model.best_iteration_ > 0 else rank_model.n_estimators
                        print(f"  [排序头] 最佳迭代: {best_rank}")
                        rank_pred = rank_model.predict(X_test)
                except Exception as e:
                    warnings.warn(f"Ranker training failed at step {step_idx}: {e}")

            # ── Blend predictions ──
            blended = self._blend_predictions(
                reg_pred, rank_pred, test_df["period"]
            )

            # ── Store predictions ──
            test_out = test_df.copy()
            test_out["pred_score"] = blended
            test_out["reg_pred"] = reg_pred
            if rank_pred is not None:
                test_out["rank_pred"] = rank_pred

            # ── Per-period evaluation ──
            for period, g in test_out.groupby("period"):
                y_true = g[fwd_ret_col].values
                y_hat = g["pred_score"].values

                ric = spearman_ic(y_true, y_hat)
                pic = pearson_ic(y_true, y_hat)
                qr = quantile_returns(y_hat, y_true, n_quantiles=5)

                eval_entry = {
                    "step": step_idx,
                    "period": period,
                    "rank_ic": ric,
                    "pearson_ic": pic,
                    "n_stocks": len(g),
                    "reg_rank_ic": spearman_ic(y_true, g["reg_pred"].values),
                }
                if rank_pred is not None and "rank_pred" in g.columns:
                    eval_entry["ranker_rank_ic"] = spearman_ic(
                        y_true, g["rank_pred"].values
                    )
                eval_entry.update(qr)
                self.eval_history.append(eval_entry)

            # Keep useful columns
            keep = [
                "code", "date", "period", "period_sort",
                "close", "next_open", "next_date", "label_end_date",
                "free_market_cap", "industry_code", "industry_name",
                "pred_score", "reg_pred", "fwd_ret", "label",
            ]
            if rank_pred is not None:
                keep.append("rank_pred")
            keep = [c for c in keep if c in test_out.columns]
            all_preds.append(test_out[keep])

            t += pd.DateOffset(months=self.step_months)

        if not all_preds:
            raise RuntimeError(
                f"No valid prediction windows produced!\n"
                f"  min_date={min_date}, max_date={max_date}\n"
                f"  initial_train_years={self.initial_train_years}, "
                f"embargo_months={self.embargo_months}"
            )

        predictions = pd.concat(all_preds, ignore_index=True)

        # Deduplicate overlapping windows
        predictions = (
            predictions.sort_values(["period", "code"])
            .drop_duplicates(subset=["period", "code"], keep="last")
            .reset_index(drop=True)
        )

        # Drop internal columns
        for tmp_col in ["_date", "_label_end_date", "_neutral_label", "_relevance"]:
            if tmp_col in predictions.columns:
                predictions = predictions.drop(columns=[tmp_col])

        self._report_summary()
        self._report_importance()
        self._report_quantiles()

        return predictions

    # ──────────── Reporting ────────────

    def _report_summary(self) -> None:
        """Print comprehensive IC summary."""
        if not self.eval_history:
            return
        edf = pd.DataFrame(self.eval_history)
        summary = summarize_ic(edf["rank_ic"])

        print("\n" + "=" * 60)
        print("V2 模型质量报告 — Rank IC（样本外）")
        print("=" * 60)
        print(f"  平均 Rank IC     : {summary['mean_ic']:.4f}")
        print(f"  中位数 Rank IC   : {summary['median_ic']:.4f}")
        print(f"  IC 标准差        : {summary['std_ic']:.4f}")
        print(f"  IC IR            : {summary['ir']:.2f}")
        print(f"  IC > 0 占比      : {summary['ic_gt0_rate']:.1%}")
        print(f"  IC > 0.05 占比   : {summary['ic_gt005_rate']:.1%}")
        print(f"  调仓期数         : {int(summary['n_periods'])}")

        # Regressor-only IC
        reg_summary = summarize_ic(edf["reg_rank_ic"])
        print(f"\n  [回归头单独] 平均IC={reg_summary['mean_ic']:.4f}, "
              f"IR={reg_summary['ir']:.2f}")

        if "ranker_rank_ic" in edf.columns:
            rank_summary = summarize_ic(edf["ranker_rank_ic"])
            print(f"  [排序头单独] 平均IC={rank_summary['mean_ic']:.4f}, "
                  f"IR={rank_summary['ir']:.2f}")

        print("=" * 60)

    def _report_importance(self, top_n: int = 15) -> None:
        """Print feature importance summary across expanding steps."""
        if not self.importance_history:
            return
        imp_df = pd.DataFrame(self.importance_history)
        mean_imp = imp_df.mean().sort_values(ascending=False)
        std_imp = imp_df.std()

        print("\n特征重要性（跨步骤平均，回归头）:")
        print(f"  {'特征':<25} {'重要性':>10}  {'标准差':>8}  {'CV':>8}")
        print("  " + "-" * 55)
        for feat in mean_imp.head(top_n).index:
            m = mean_imp[feat]
            s = std_imp.get(feat, 0)
            cv = s / m if m > 0 else float("inf")
            print(f"  {feat:<25} {m:>10.1f}  {s:>8.1f}  {cv:>8.2f}")

    def _report_quantiles(self) -> None:
        """Print quantile portfolio spread summary."""
        if not self.eval_history:
            return
        edf = pd.DataFrame(self.eval_history)
        if "long_short_spread" not in edf.columns:
            return

        spreads = edf["long_short_spread"].dropna()
        mono = edf["monotonicity"].dropna() if "monotonicity" in edf.columns else pd.Series()

        print(f"\n分位组合分析（5分位）:")
        print(f"  平均多空价差 (Q5-Q1) : {spreads.mean():.4f} "
              f"(std={spreads.std():.4f})")
        print(f"  多空价差 > 0 占比    : {(spreads > 0).mean():.1%}")
        if len(mono) > 0:
            print(f"  平均单调性分数       : {mono.mean():.2f}")

        # Per-quantile average
        q_cols = [f"Q{i}" for i in range(1, 6)]
        available = [c for c in q_cols if c in edf.columns]
        if available:
            print(f"\n  各分位平均收益:")
            for qc in available:
                vals = edf[qc].dropna()
                if len(vals) > 0:
                    print(f"    {qc}: {vals.mean():.4f} (std={vals.std():.4f})")

    # ──────────── Accessors ────────────

    def get_eval_dataframe(self) -> pd.DataFrame:
        """Return full evaluation history as DataFrame."""
        return pd.DataFrame(self.eval_history)

    def get_importance_dataframe(self) -> pd.DataFrame:
        """Return feature importance history as DataFrame."""
        return pd.DataFrame(self.importance_history)

    def predict_with_last_model(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Predict using the most recently trained models (for holdout inference).

        Returns blended prediction scores.
        """
        if not self.reg_models:
            raise RuntimeError("No trained models available.")

        cols = feature_cols or self.feature_cols
        X = df[cols]

        reg_pred = self.reg_models[-1].predict(X)

        rank_pred = None
        if self.use_ranker and self.rank_models:
            rank_pred = self.rank_models[-1].predict(X)

        return self._blend_predictions(reg_pred, rank_pred, df["period"])
