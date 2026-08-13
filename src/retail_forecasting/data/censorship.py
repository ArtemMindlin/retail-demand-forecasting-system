from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

# The dataset records stock availability over the 6:00–22:00 operative window (16 hours).
# stockout_hours is counted within that window, so this is its normalizing denominator.
OPERATIVE_WINDOW_HOURS = 16.0

# Single source of truth for the latent-demand imputation strategies.
ImputationStrategy = Literal["supervised", "historical_mean", "clipped_scaling", "none"]

# Defaults used by the supervised strategy's LGBM teacher model when no tuned
# hyperparameters are supplied (never empirically tuned; see imputation_tuning.py).
DEFAULT_SUPERVISED_LGBM_PARAMS: dict[str, int | float] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
}

# Filename convention for the tuned hyperparameters written by imputation_tuning.py under
# settings.models.models_dir; shared so producer and consumers never drift apart.
IMPUTATION_LGBM_PARAMS_FILENAME = "imputation_lgbm_params.json"


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
        """LGBM teacher-model with rich covariates."""
        df_feat = df.copy()
        df_feat["month"] = df_feat["date"].dt.month
        df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
        df_feat["day_of_month"] = df_feat["date"].dt.day
        df_feat["series_cat"] = df_feat["series_id"].astype("category")

        # Stockout severity: fraction of operative window without stock
        df_feat["stockout_ratio"] = df_feat[self.stockout_col] / OPERATIVE_WINDOW_HOURS

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
            "stockout_ratio",
            "series_mean_demand",
        ] + extra_features

        train_df = df_feat[is_clean].copy()
        X_train = train_df[feature_cols]
        y_train = train_df[self.target_col]

        params = self._resolve_lgbm_params()
        self.model = lgb.LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            random_state=42,
            verbosity=-1,
        )
        self.model.fit(X_train, y_train)

        censored_df = df_feat[is_censored].copy()
        X_censored = censored_df[feature_cols]

        predicted_latent = np.asarray(self.model.predict(X_censored), dtype=float)

        df.loc[is_censored, "latent_demand_est"] = np.maximum(
            df.loc[is_censored, self.target_col], predicted_latent
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, self.target_col]
        return df

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
