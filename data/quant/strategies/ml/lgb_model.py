"""
LightGBM model for cross-sectional stock ranking.
===================================================
Responsibilities:
1. Rolling-window training with early stopping on validation set
2. Prediction (output: cross-sectional scores per period)
3. Rank IC evaluation (Spearman correlation of predicted vs actual ranks)
4. Feature importance tracking & stability analysis

Usage:
    from strategies.ml.lgb_model import RollingLGBModel
    model = RollingLGBModel()
    predictions = model.rolling_train_predict(dataset, feature_cols)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm")


# ─────────────────────── Default Hyperparameters ────────────────

# Tuned for cross-sectional stock ranking with ~2000 stocks per period.
# Key design choices:
#   - Low num_leaves (31) + high min_child_samples (100): prevent overfitting
#     on noisy financial data where signal-to-noise ratio is ~1%.
#   - subsample + colsample: add randomness, reduce memorization.
#   - learning_rate 0.05 with 1000 rounds + early stopping: find optimal depth.
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


# ─────────────────────── Evaluation Metrics ─────────────────────

def rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Rank IC: Spearman rank correlation between predicted scores
    and actual forward returns.

    This is THE standard metric for cross-sectional stock models.
    - IC > 0.05 is decent for daily/biweekly models.
    - IC > 0.08 is very good.
    - IC > 0.10 is exceptional (probably overfitting).
    """
    mask = y_true.notna() & pd.Series(y_pred).notna()
    if mask.sum() < 10:
        return np.nan
    corr, _ = stats.spearmanr(y_true[mask], pd.Series(y_pred)[mask])
    return corr


def rank_ic_by_period(
    df: pd.DataFrame,
    pred_col: str = "pred_score",
    label_col: str = "fwd_ret",
    period_col: str = "period",
) -> pd.DataFrame:
    """
    Compute Rank IC for each rebalance period.

    Returns DataFrame with columns: period, rank_ic, n_stocks
    """
    results = []
    for period, g in df.groupby(period_col):
        ic = rank_ic(g[label_col], g[pred_col])
        results.append({
            "period": period,
            "rank_ic": ic,
            "n_stocks": len(g),
        })
    return pd.DataFrame(results)


def ic_summary(ic_series: pd.Series) -> Dict[str, float]:
    """
    Summarize Rank IC time series.

    Returns dict with: mean_ic, std_ic, ir (information coefficient ratio),
    ic_positive_rate (% of periods with IC > 0).
    """
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {"mean_ic": np.nan, "std_ic": np.nan,
                "ir": np.nan, "ic_positive_rate": np.nan}
    mean = ic.mean()
    std = ic.std(ddof=0)
    return {
        "mean_ic": mean,
        "std_ic": std,
        "ir": mean / std if std > 0 else np.nan,
        "ic_positive_rate": (ic > 0).mean(),
    }


# ─────────────────────── Model Class ────────────────────────────

@dataclass
class RollingLGBModel:
    """
    LightGBM model with rolling-window training for stock selection.

    The model is retrained periodically on a sliding window of historical
    data, then used to predict scores for the next period(s).

    Parameters
    ----------
    params : dict
        LightGBM parameters. Defaults to DEFAULT_PARAMS.
    train_years : int
        Length of training window in years.
    val_months : int
        Length of validation window in months (for early stopping).
    step_months : int
        How far the window slides forward each iteration.
    early_stopping_rounds : int
        Stop training if val metric doesn't improve for this many rounds.
    feature_cols : list of str
        Feature column names. Set during training.
    """
    params: Dict = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    train_years: int = 3
    val_months: int = 6
    step_months: int = 6
    early_stopping_rounds: int = 50

    # State (populated during training)
    feature_cols: List[str] = field(default_factory=list)
    models: List = field(default_factory=list)
    importance_history: List[Dict[str, float]] = field(default_factory=list)
    ic_history: List[Dict] = field(default_factory=list)

    def _train_single(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
    ) -> lgb.LGBMRegressor:
        """
        Train a single LightGBM model with early stopping.

        Returns trained model.
        """
        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[label_col].values

        # Drop rows with NaN label
        train_mask = ~np.isnan(y_train)
        val_mask = ~np.isnan(y_val)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_val[val_mask], y_val[val_mask]

        model = lgb.LGBMRegressor(**self.params)

        # Handle NaN in features: LightGBM handles NaN natively,
        # but we need to make sure the arrays are float
        X_train = np.nan_to_num(X_train, nan=np.nan)  # keep NaN for LGB
        X_val = np.nan_to_num(X_val, nan=np.nan)

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),  # suppress per-round logging
        ]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        # Record feature importance
        imp = dict(zip(feature_cols, model.feature_importances_))
        self.importance_history.append(imp)

        return model

    def _predict(
        self,
        model: lgb.LGBMRegressor,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Predict scores for a DataFrame."""
        X = df[feature_cols].values
        return model.predict(X)

    def rolling_train_predict(
        self,
        dataset: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        min_date: str = "2019-01-01",
        max_date: str = "2026-02-28",
    ) -> pd.DataFrame:
        """
        Full rolling-window training and prediction pipeline.

        For each window:
        1. Train on [T - train_years, T)
        2. Validate on [T, T + val_months) for early stopping
        3. Predict on [T + val_months, T + val_months + step_months)

        Parameters
        ----------
        dataset : DataFrame from data_prep.build_ml_dataset()
        feature_cols : list of feature column names
        label_col : label column name
        min_date : earliest date for training start
        max_date : latest date

        Returns
        -------
        DataFrame with all test predictions concatenated, including
        columns: code, date, period, pred_score, fwd_ret, label,
        industry_code, free_market_cap, close
        """
        self.feature_cols = feature_cols
        self.models = []
        self.importance_history = []
        self.ic_history = []

        dataset = dataset.copy()
        dataset["_date"] = pd.to_datetime(dataset["date"])

        min_dt = pd.Timestamp(min_date)
        max_dt = pd.Timestamp(max_date)

        # First possible train end: after train_years from min_date
        first_train_end = min_dt + pd.DateOffset(years=self.train_years)

        all_predictions = []
        window_idx = 0
        t = first_train_end

        while t + pd.DateOffset(months=self.val_months) <= max_dt:
            window_idx += 1
            train_start = t - pd.DateOffset(years=self.train_years)
            train_end = t - pd.DateOffset(days=1)
            val_start = t
            val_end = t + pd.DateOffset(months=self.val_months) - pd.DateOffset(days=1)
            test_start = t + pd.DateOffset(months=self.val_months)
            test_end = min(
                t + pd.DateOffset(months=self.val_months + self.step_months) - pd.DateOffset(days=1),
                max_dt,
            )

            train_df = dataset[
                (dataset["_date"] >= train_start) & (dataset["_date"] <= train_end)
            ]
            val_df = dataset[
                (dataset["_date"] >= val_start) & (dataset["_date"] <= val_end)
            ]
            test_df = dataset[
                (dataset["_date"] >= test_start) & (dataset["_date"] <= test_end)
            ]

            if len(train_df) == 0 or len(test_df) == 0:
                t += pd.DateOffset(months=self.step_months)
                continue

            # Handle empty validation set: use last 20% of training data
            if len(val_df) == 0:
                split_idx = int(len(train_df) * 0.8)
                sorted_train = train_df.sort_values("_date")
                val_df = sorted_train.iloc[split_idx:]
                train_df = sorted_train.iloc[:split_idx]

            print(f"\n[model] Window {window_idx}: "
                  f"train {train_start.date()}~{train_end.date()} ({len(train_df):,}), "
                  f"val ({len(val_df):,}), "
                  f"test {test_start.date()}~{test_end.date()} ({len(test_df):,})")

            # Train
            model = self._train_single(train_df, val_df, feature_cols, label_col)
            self.models.append(model)
            best_iter = model.best_iteration_ if model.best_iteration_ > 0 else model.n_estimators
            print(f"  Best iteration: {best_iter}")

            # Predict on test set
            test_copy = test_df.copy()
            test_copy["pred_score"] = self._predict(model, test_df, feature_cols)

            # Compute per-period Rank IC on test set
            for period, g in test_copy.groupby("period"):
                ic = rank_ic(g["fwd_ret"], g["pred_score"])
                self.ic_history.append({
                    "window": window_idx,
                    "period": period,
                    "rank_ic": ic,
                    "n_stocks": len(g),
                })

            # Keep useful columns
            keep = ["code", "date", "period", "period_sort",
                    "close", "free_market_cap", "industry_code", "industry_name",
                    "pred_score", "fwd_ret", "label"]
            keep = [c for c in keep if c in test_copy.columns]
            all_predictions.append(test_copy[keep])

            t += pd.DateOffset(months=self.step_months)

        if not all_predictions:
            raise RuntimeError("No valid prediction windows generated!")

        predictions = pd.concat(all_predictions, ignore_index=True)

        # Remove duplicate period-code rows (from overlapping windows)
        predictions = (
            predictions.sort_values(["period", "code"])
            .drop_duplicates(subset=["period", "code"], keep="last")
            .reset_index(drop=True)
        )

        # Drop the temp _date column if present
        if "_date" in predictions.columns:
            predictions = predictions.drop(columns=["_date"])

        self._print_ic_summary()
        self._print_feature_importance()

        return predictions

    def _print_ic_summary(self) -> None:
        """Print Rank IC summary to stdout."""
        if not self.ic_history:
            return
        ic_df = pd.DataFrame(self.ic_history)
        summary = ic_summary(ic_df["rank_ic"])
        print("\n" + "=" * 50)
        print("Rank IC Summary (out-of-sample)")
        print("=" * 50)
        print(f"  Mean IC      : {summary['mean_ic']:.4f}")
        print(f"  Std IC       : {summary['std_ic']:.4f}")
        print(f"  IC IR        : {summary['ir']:.2f}")
        print(f"  IC > 0 rate  : {summary['ic_positive_rate']:.1%}")
        print(f"  Num periods  : {len(ic_df)}")
        print("=" * 50)

    def _print_feature_importance(self, top_n: int = 15) -> None:
        """Print average feature importance across all windows."""
        if not self.importance_history:
            return
        imp_df = pd.DataFrame(self.importance_history)
        mean_imp = imp_df.mean().sort_values(ascending=False)
        std_imp = imp_df.std()

        print("\nFeature Importance (avg across windows):")
        print(f"  {'Feature':<25} {'Importance':>10}  {'Std':>8}  {'Stability':>10}")
        print("  " + "-" * 58)
        for feat in mean_imp.head(top_n).index:
            m = mean_imp[feat]
            s = std_imp.get(feat, 0)
            # Stability: coefficient of variation (lower = more stable)
            stability = s / m if m > 0 else np.inf
            print(f"  {feat:<25} {m:>10.1f}  {s:>8.1f}  {stability:>10.2f}")

    def get_ic_dataframe(self) -> pd.DataFrame:
        """Return IC history as DataFrame for further analysis."""
        return pd.DataFrame(self.ic_history)

    def get_importance_dataframe(self) -> pd.DataFrame:
        """Return feature importance history as DataFrame."""
        return pd.DataFrame(self.importance_history)


# ─────────────────────── Standalone Helper ──────────────────────

def quick_train_predict(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    train_end: str = "2023-12-31",
    val_end: str = "2024-12-31",
    label_col: str = "label",
    params: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, lgb.LGBMRegressor, Dict]:
    """
    Quick one-shot train/predict for rapid experimentation.

    Splits data into train (up to train_end), val (train_end ~ val_end),
    test (after val_end).

    Returns (predictions_df, model, ic_summary_dict)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    dataset = dataset.copy()
    dataset["_date"] = pd.to_datetime(dataset["date"])

    train = dataset[dataset["_date"] <= train_end]
    val = dataset[(dataset["_date"] > train_end) & (dataset["_date"] <= val_end)]
    test = dataset[dataset["_date"] > val_end]

    print(f"[quick] train={len(train):,}, val={len(val):,}, test={len(test):,}")

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Train or test set is empty!")

    # Handle empty val: split train
    if len(val) == 0:
        split_idx = int(len(train) * 0.8)
        sorted_train = train.sort_values("_date")
        val = sorted_train.iloc[split_idx:]
        train = sorted_train.iloc[:split_idx]

    # Train
    X_train = train[feature_cols].values
    y_train = train[label_col].values
    X_val = val[feature_cols].values
    y_val = val[label_col].values

    # Drop NaN labels
    tr_mask = ~np.isnan(y_train)
    va_mask = ~np.isnan(y_val)
    X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    X_val, y_val = X_val[va_mask], y_val[va_mask]

    model = lgb.LGBMRegressor(**params)
    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration_ if model.best_iteration_ > 0 else model.n_estimators
    print(f"[quick] Best iteration: {best_iter}")

    # Predict on test
    test = test.copy()
    test["pred_score"] = model.predict(test[feature_cols].values)

    # Compute IC
    ic_df = rank_ic_by_period(test, pred_col="pred_score", label_col="fwd_ret")
    summary = ic_summary(ic_df["rank_ic"])

    print(f"[quick] Test Rank IC: mean={summary['mean_ic']:.4f}, "
          f"IR={summary['ir']:.2f}, IC>0={summary['ic_positive_rate']:.1%}")

    # Feature importance
    imp = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n[quick] Top features:")
    for feat, score in imp[:10]:
        print(f"  {feat:<25} {score:>6}")

    return test, model, summary
