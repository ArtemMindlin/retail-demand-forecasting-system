from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

# Normalizing denominator for 16-hour operative window (6:00-22:00).
OPERATIVE_WINDOW_HOURS = 16.0

# Single source of truth for the latent-demand imputation strategies.
ImputationStrategy = Literal["supervised", "historical_mean", "clipped_scaling", "none"]

# Untuned default LGBM hyperparameters for supervised imputation.
DEFAULT_SUPERVISED_LGBM_PARAMS: dict[str, int | float] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
}

# Shared filename convention for persisted tuned imputation hyperparameters.
IMPUTATION_LGBM_PARAMS_FILENAME = "imputation_lgbm_params.json"

SYNTHETIC_CENSORING_EVAL_FRACTION = 0.30


def _synthetic_censor_holdout(
    panel: pd.DataFrame, seed: int, eval_fraction: float = SYNTHETIC_CENSORING_EVAL_FRACTION
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Hold out a sample of CLEAN days and synthetically censor them.

    Latent demand on real stockouts is an unobserved counterfactual, so imputation quality
    can only be scored on days where the true demand is actually known: clean days. Each
    held-out clean day is assigned a stockout ratio sampled from the empirical distribution
    of real stockouts, and its sale is reduced proportionally -- giving a synthetic stand-in
    for a censored day with a known ground truth, reusable by any strategy-scoring function.

    Returns (censored_panel, eval_idx, true_demand); eval_idx/true_demand are empty when the
    panel has no clean or no censored rows to draw the synthetic ratio from.
    """
    rng = np.random.default_rng(seed)
    clean_mask = panel["stockout_hours"] == 0
    real_ratios = panel.loc[panel["stockout_hours"] > 0, "stockout_hours"] / OPERATIVE_WINDOW_HOURS
    clean_idx = panel.index[clean_mask]
    if len(clean_idx) == 0 or len(real_ratios) == 0:
        return panel.copy(), np.array([], dtype=int), np.array([], dtype=float)

    n_eval = max(1, int(len(clean_idx) * eval_fraction))
    eval_idx = rng.choice(clean_idx, size=n_eval, replace=False)
    sampled_ratios = rng.choice(real_ratios, size=n_eval, replace=True)

    true_demand = panel.loc[eval_idx, "observed_demand"].to_numpy()

    censored = panel.copy()
    censored.loc[eval_idx, "stockout_hours"] = sampled_ratios * OPERATIVE_WINDOW_HOURS
    censored.loc[eval_idx, "observed_demand"] = true_demand * (1.0 - sampled_ratios)
    return censored, eval_idx, true_demand


class LatentDemandImputer:
    """Imputes latent demand for periods with stockouts using various strategies.

    This class supports comparing different ways to recover 'hidden' demand when
    sales are censored by zero or low stock.
    """

    def __init__(
        self,
        strategy: ImputationStrategy = "supervised",
        stockout_col: str = "stockout_hours",
        target_col: str = "observed_demand",
        scaling_factor: float = 1.2,
        lgbm_params: dict[str, int | float] | None = None,
        model_path: Path | None = None,
    ):
        self.strategy = strategy
        self.stockout_col = stockout_col
        self.target_col = target_col
        self.scaling_factor = scaling_factor
        self.lgbm_params = lgbm_params
        self.model_path = model_path
        self.model: lgb.LGBMRegressor | None = None

    def impute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Correct censored demand in the input panel based on selected strategy.

        Args:
            panel: Daily panel containing date, series_id, observed_demand, and stockout_hours.

        Returns:
            A panel with 'latent_demand_est', 'is_imputed' and updated 'observed_demand'.
        """
        if self.strategy == "none":
            return self._passthrough(panel)

        df = panel.copy()
        is_clean = df[self.stockout_col] == 0
        is_censored = ~is_clean

        if not is_censored.any():
            return self._passthrough(panel)

        if self.strategy == "supervised":
            df = self._impute_supervised(df, is_clean, is_censored)
        elif self.strategy == "historical_mean":
            df = self._impute_historical_mean(df, is_clean, is_censored)
        elif self.strategy == "clipped_scaling":
            df = self._impute_clipped_scaling(df, is_clean, is_censored)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.strategy!r}")

        # Ensure we don't have NaNs in the estimate
        df["latent_demand_est"] = df["latent_demand_est"].fillna(df[self.target_col])
        df["is_imputed"] = is_censored

        # Backup original and swap
        df["original_observed_demand"] = df[self.target_col]
        df[self.target_col] = df["latent_demand_est"]

        return df

    def _resolve_lgbm_params(self) -> dict[str, int | float]:
        """Pick the supervised teacher model's hyperparameters.

        Precedence: a tuned params file on disk (written by imputation_tuning.py) beats
        an explicit override, which beats the never-tuned defaults. Only the recipe is
        persisted, not fitted weights, so the model is always re-fit on the current panel's
        clean days -- this keeps leakage/feature-space properties identical to before tuning
        existed.
        """
        if self.model_path is not None and self.model_path.exists():
            loaded: dict[str, int | float] = json.loads(self.model_path.read_text(encoding="utf-8"))
            return loaded
        if self.lgbm_params is not None:
            return self.lgbm_params
        return dict(DEFAULT_SUPERVISED_LGBM_PARAMS)

    def _passthrough(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return the panel unchanged, marking demand as not imputed.

        Used when no correction applies (strategy ``none`` or no censored rows).
        """
        df = panel.copy()
        df["latent_demand_est"] = df[self.target_col]
        df["is_imputed"] = False
        return df

    def _impute_supervised(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """LGBM teacher-model with rich covariates, reconciled by stockout severity.

        The teacher is fitted on clean days only, so it predicts FULL-DAY demand -- a
        quantity that does not depend on how many hours the day was out of stock. Severity
        therefore has no place among its features (it was one until it was measured at
        exactly 0 importance: every training row is a clean day, where the ratio is
        identically 0, and LightGBM cannot split on a constant). It belongs in the
        reconciliation instead, see ``_reconcile_with_severity``.
        """
        df_feat = df.copy()
        df_feat["month"] = df_feat["date"].dt.month
        df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
        df_feat["day_of_month"] = df_feat["date"].dt.day
        df_feat["series_cat"] = df_feat["series_id"].astype("category")

        # Series-level clean-day mean as a prior (mirrors historical_mean but as a feature)
        series_means = df_feat[is_clean].groupby("series_id")[self.target_col].mean()
        df_feat["series_mean_demand"] = df_feat["series_id"].map(series_means)
        global_mean = float(df_feat[is_clean][self.target_col].mean())
        df_feat["series_mean_demand"] = df_feat["series_mean_demand"].fillna(global_mean)

        optional_cols = [
            "discount",
            "holiday_flag",
            "avg_temperature",
            "precpt",
            "avg_humidity",
            "avg_wind_level",
        ]
        extra_features = [c for c in optional_cols if c in df_feat.columns]

        feature_cols = [
            "month",
            "day_of_week",
            "day_of_month",
            "series_cat",
            "series_mean_demand",
        ] + extra_features

        train_df = df_feat[is_clean].copy()
        X_train = train_df[feature_cols]
        y_train = train_df[self.target_col]

        params = self._resolve_lgbm_params()
        # Single-threaded on purpose: the teacher trains on one panel's clean days, and at
        # that width LightGBM spends more time synchronising threads than splitting nodes.
        # Measured 18x faster at 1.5k train rows and still 4.4x at 46k (the 500-series
        # panel), with bit-identical predictions. Revisit above ~200k rows, where the two
        # converge.
        self.model = lgb.LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        self.model.fit(X_train, y_train)

        censored_df = df_feat[is_censored].copy()
        X_censored = censored_df[feature_cols]

        predicted_latent = np.asarray(self.model.predict(X_censored), dtype=float)

        df.loc[is_censored, "latent_demand_est"] = self._reconcile_with_severity(
            observed=df.loc[is_censored, self.target_col].to_numpy(),
            predicted_full_day=predicted_latent,
            stockout_hours=df.loc[is_censored, self.stockout_col].to_numpy(),
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, self.target_col]
        return df

    @staticmethod
    def _reconcile_with_severity(
        observed: np.ndarray, predicted_full_day: np.ndarray, stockout_hours: np.ndarray
    ) -> np.ndarray:
        """Keep the sales that did happen, and estimate only the unstocked slice of the day.

        With ``r`` the fraction of the operative window without stock, the day was sellable
        for ``1 - r`` of it: the observed sale already covers that part, and only ``r`` of a
        full day of demand is missing. So the estimate is ``observed + r * predicted``.

        This replaces a ``max(observed, predicted)`` that treated the two as rival estimates
        of the same quantity and discarded whichever was smaller -- throwing away real sales
        on lightly-censored days, where the observed value is by far the better evidence.

        NOTE: this rule cannot be ranked against the old one on the synthetic-censoring
        holdout. That holdout is built as ``observed = truth * (1 - r)``, which is the very
        assumption this formula encodes, so it scores near-zero error there by construction
        (measured MAE 0.02 vs 0.74) and the comparison is circular. It is adopted on the
        modelling argument above, not on that number. See invariant 42.
        """
        ratio = np.clip(stockout_hours / OPERATIVE_WINDOW_HOURS, 0.0, 1.0)
        estimate: np.ndarray = observed + ratio * np.clip(predicted_full_day, 0.0, None)
        return estimate

    def _impute_historical_mean(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """Baseline: Impute using the historical mean of clean days for each series."""
        series_means = df[is_clean].groupby("series_id")[self.target_col].mean()
        global_mean = float(df[is_clean][self.target_col].mean())

        fallback_means = df.loc[is_censored, "series_id"].map(series_means).fillna(global_mean)

        df.loc[is_censored, "latent_demand_est"] = np.maximum(
            df.loc[is_censored, self.target_col],
            fallback_means,
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, self.target_col]
        return df

    def _impute_clipped_scaling(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """Baseline: Simply scale up the observed demand by a fixed factor during stockouts."""
        df.loc[is_censored, "latent_demand_est"] = np.maximum(
            df.loc[is_censored, self.target_col],
            df.loc[is_censored, self.target_col] * self.scaling_factor,
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, self.target_col]
        return df
