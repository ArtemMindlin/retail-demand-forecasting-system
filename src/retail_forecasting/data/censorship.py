from __future__ import annotations

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Literal, Protocol, cast


class RegressorProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> object: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class LatentDemandImputer:
    """Imputes latent demand for periods with stockouts using various strategies.

    This class supports comparing different ways to recover 'hidden' demand when
    sales are censored by zero or low stock.
    """

    def __init__(
        self,
        strategy: Literal[
            "supervised", "historical_mean", "clipped_scaling", "none"
        ] = "supervised",
        stockout_col: str = "stockout_hours",
        target_col: str = "observed_demand",
        scaling_factor: float = 1.2,
    ):
        self.strategy = strategy
        self.stockout_col = stockout_col
        self.target_col = target_col
        self.scaling_factor = scaling_factor
        self.model: RegressorProtocol | None = None

    def impute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Correct censored demand in the input panel based on selected strategy.

        Args:
            panel: Daily panel containing date, series_id, observed_demand, and stockout_hours.

        Returns:
            A panel with 'latent_demand_est', 'is_imputed' and updated 'observed_demand'.
        """
        if self.strategy == "none":
            df = panel.copy()
            df["latent_demand_est"] = df[self.target_col]
            df["is_imputed"] = False
            return df

        df = panel.copy()
        is_clean = df[self.stockout_col] == 0
        is_censored = ~is_clean

        if not is_censored.any():
            df["latent_demand_est"] = df[self.target_col]
            df["is_imputed"] = False
            return df

        if self.strategy == "supervised":
            df = self._impute_supervised(df, is_clean, is_censored)
        elif self.strategy == "historical_mean":
            df = self._impute_historical_mean(df, is_clean, is_censored)
        elif self.strategy == "clipped_scaling":
            df = self._impute_clipped_scaling(df, is_clean, is_censored)

        # Ensure we don't have NaNs in the estimate
        df["latent_demand_est"] = df["latent_demand_est"].fillna(df[self.target_col])
        df["is_imputed"] = is_censored

        # Backup original and swap
        df["original_observed_demand"] = df[self.target_col]
        df[self.target_col] = df["latent_demand_est"]

        return df

    def _impute_supervised(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """Original teacher-model (LGBM) approach."""
        # Simple features
        df_feat = df.copy()
        df_feat["month"] = df_feat["date"].dt.month
        df_feat["day_of_week"] = df_feat["date"].dt.dayofweek

        train_df = df_feat[is_clean].copy()
        train_df["series_cat"] = train_df["series_id"].astype("category")

        X_train = train_df[["month", "day_of_week", "series_cat"]]
        y_train = train_df[self.target_col]

        self.model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42, verbosity=-1
        )
        self.model.fit(X_train, y_train)

        censored_df = df_feat[is_censored].copy()
        censored_df["series_cat"] = censored_df["series_id"].astype("category")
        X_censored = censored_df[["month", "day_of_week", "series_cat"]]

        predicted_latent = self.model.predict(X_censored)

        df.loc[is_censored, "latent_demand_est"] = np.maximum(
            df.loc[is_censored, self.target_col], predicted_latent
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, self.target_col]
        return df

    def _impute_historical_mean(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """Baseline: Impute using the historical mean of clean days for each series."""
        means = cast(
            dict[object, float],
            df[is_clean].groupby("series_id")[self.target_col].mean().to_dict(),
        )

        # Default to global mean if a series has NO clean days
        global_mean = float(df[is_clean][self.target_col].mean())

        def get_mean(series_id: object) -> float:
            return means.get(series_id, global_mean)

        df.loc[is_censored, "latent_demand_est"] = np.maximum(
            df.loc[is_censored, self.target_col],
            df.loc[is_censored, "series_id"].map(get_mean),
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, self.target_col]
        return df

    def _impute_clipped_scaling(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """Baseline: Simply scale up the observed demand by a fixed factor during stockouts."""
        df.loc[is_censored, "latent_demand_est"] = (
            df.loc[is_censored, self.target_col] * self.scaling_factor
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, self.target_col]
        return df
