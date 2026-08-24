from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_forecasting.contracts.contracts_config import ImputationStrategy
from retail_forecasting.contracts.contracts_tuning import ImputationBoostingParams

# Normalizing denominator for 16-hour operative window (6:00-22:00).
OPERATIVE_WINDOW_HOURS = 16.0

# Untuned default LGBM hyperparameters for supervised imputation.
DEFAULT_SUPERVISED_LGBM_PARAMS: dict[str, int | float] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "colsample_bytree": 1.0,
    "subsample": 1.0,
    "subsample_freq": 0,
    "reg_alpha": 1e-8,
    "reg_lambda": 1e-8,
    "min_data_per_group": 100,
    "cat_smooth": 10.0,
    "max_bin": 255,
}

# Shared filename convention for persisted tuned imputation hyperparameters.

SYNTHETIC_CENSORING_EVAL_FRACTION = 0.30

SUPERVISED_CANDIDATE_FEATURES = [
    "month",
    "day_of_week",
    "day_of_month",
    "series_cat",
    "series_mean_demand",
    "discount",
    "holiday_flag",
    "avg_temperature",
    "precpt",
    "avg_humidity",
    "avg_wind_level",
]


def synthetic_censor_holdout(
    panel: pd.DataFrame,
    seed: int,
    eval_fraction: float = SYNTHETIC_CENSORING_EVAL_FRACTION,
    censorable_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Hold out a sample of CLEAN days and synthetically censor them.

    Latent demand on real stockouts is an unobserved counterfactual, so imputation quality
    can only be scored on days where the true demand is actually known: clean days. Each
    held-out clean day is assigned a stockout ratio sampled from the empirical distribution
    of real stockouts, and its sale is reduced proportionally -- giving a synthetic stand-in
    for a censored day with a known ground truth, reusable by any strategy-scoring function.

    Args:
        panel: The panel to censor. The imputer is later fitted on THIS panel's remaining
            clean days, so its width sets the teacher's training size.
        seed: Draw seed.
        eval_fraction: Share of eligible clean rows to censor.
        censorable_mask: Restricts which rows participate, without shrinking the panel. This is
            the difference between "score a smaller panel" and "score one window of a full
            panel": the teacher keeps its whole training set either way, and only the evaluation
            targets move. The imputation search uses it to hold out a time window while the
            teacher still sees the deployment-sized panel. Note it restricts BOTH the rows that
            may be censored AND the real stockouts the severity ratios are drawn from -- see the
            comment at the intersection below for why the second one is not optional. Defaults
            to all rows.

    Returns:
        ``(censored_panel, eval_idx, true_demand)``, always non-empty.
    """
    rng = np.random.default_rng(seed)
    clean_mask = panel["stockout_hours"] == 0
    stockout_mask = panel["stockout_hours"] > 0
    if censorable_mask is not None:
        clean_mask = clean_mask & censorable_mask
        stockout_mask = stockout_mask & censorable_mask
    real_ratios = panel.loc[stockout_mask, "stockout_hours"] / OPERATIVE_WINDOW_HOURS
    clean_idx = panel.index[clean_mask]
    if len(clean_idx) == 0 or len(real_ratios) == 0:
        scope = "the requested window of the panel" if censorable_mask is not None else "the panel"
        raise ValueError(
            f"Cannot build a synthetic-censoring holdout: {scope} holds {len(clean_idx)} clean "
            f"rows to censor and {len(real_ratios)} real stockouts to draw a severity from, and "
            "both are needed. Reconstruction can only be scored where the true demand is known, "
            "so a clean day must be faked -- and it must be faked at a realistic severity."
        )

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
        scaling_factor: float = 1.2,
        lgbm_params: dict[str, int | float] | None = None,
        model_path: Path | None = None,
    ):
        self.strategy = strategy
        self.scaling_factor = scaling_factor
        if lgbm_params is not None and model_path is not None:
            raise ValueError(
                "Pass lgbm_params or model_path, not both: they are two sources for the same "
                "13 numbers and there is no reading of 'both' that is not a caller mistake. "
                "The old precedence let the file win, which would have made an imputation "
                "search score the tuned file 300 times instead of each trial's own params, "
                "with a flat best_value and nothing to explain it."
            )
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
        is_clean = df["stockout_hours"] == 0
        is_censored = ~is_clean

        if not is_censored.any():
            return self._passthrough(panel)

        if self.strategy == "supervised":
            df = self._impute_supervised(df, is_clean, is_censored)
        elif self.strategy == "historical_mean":
            df = self._impute_historical_mean(df, is_clean, is_censored)
        elif self.strategy == "clipped_scaling":
            df = self._impute_clipped_scaling(df, is_clean, is_censored)

        # Ensure we don't have NaNs in the estimate
        df["latent_demand_est"] = df["latent_demand_est"].fillna(df["observed_demand"])
        df["is_imputed"] = is_censored

        # Backup original and swap
        df["original_observed_demand"] = df["observed_demand"]
        df["observed_demand"] = df["latent_demand_est"]

        return df

    def _passthrough(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return demand uncorrected, marked as not imputed."""
        df = panel.copy()
        df["latent_demand_est"] = df["observed_demand"]
        df["original_observed_demand"] = df["observed_demand"]
        df["is_imputed"] = False
        return df

    def _impute_supervised(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """Train a supervised model on clean days and predict latent demand for censored days."""
        df_feat = df.copy()
        df_feat["month"] = df_feat["date"].dt.month
        df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
        df_feat["day_of_month"] = df_feat["date"].dt.day
        df_feat["series_cat"] = df_feat["series_id"].astype("category")

        # Series-level clean-day mean as a prior (mirrors historical_mean but as a feature)
        series_means = df_feat[is_clean].groupby("series_id")["observed_demand"].mean()
        df_feat["series_mean_demand"] = df_feat["series_id"].map(series_means)
        global_mean = float(df_feat[is_clean]["observed_demand"].mean())
        df_feat["series_mean_demand"] = df_feat["series_mean_demand"].fillna(global_mean)

        feature_cols = [c for c in SUPERVISED_CANDIDATE_FEATURES if c in df_feat.columns]

        train_df = df_feat[is_clean].copy()
        X_train = train_df[feature_cols]
        y_train = train_df["observed_demand"]

        params: dict[str, int | float] = DEFAULT_SUPERVISED_LGBM_PARAMS
        if self.lgbm_params is not None:
            params = self.lgbm_params
        elif self.model_path is not None and self.model_path.exists():
            params = ImputationBoostingParams.model_validate_json(
                self.model_path.read_text(encoding="utf-8")
            ).model_dump()

        lgbm_kwargs: dict[str, Any] = dict(params)
        # `n_jobs=1` measured 16x FASTER than -1 here (1.6s vs 25.5s): ~17k clean rows over 11
        # features is too little work per tree to pay for thread coordination, times 1827 trees.
        # It is also what the tuning needs, calling this inside a 10-thread `Parallel`.
        lgbm_kwargs.update({"random_state": 42, "verbosity": -1, "n_jobs": 1})
        self.model = lgb.LGBMRegressor(**lgbm_kwargs)
        self.model.fit(X_train, y_train)

        censored_df = df_feat[is_censored].copy()
        X_censored = censored_df[feature_cols]

        predicted_latent = np.asarray(self.model.predict(X_censored), dtype=float)

        ratio = np.clip(
            df.loc[is_censored, "stockout_hours"].to_numpy() / OPERATIVE_WINDOW_HOURS, 0.0, 1.0
        )
        df.loc[is_censored, "latent_demand_est"] = df.loc[
            is_censored, "observed_demand"
        ].to_numpy() + ratio * np.clip(predicted_latent, 0.0, None)
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, "observed_demand"]
        return df

    def _impute_historical_mean(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """Baseline: Impute using the historical mean of clean days for each series."""
        series_means = df[is_clean].groupby("series_id")["observed_demand"].mean()
        global_mean = float(df[is_clean]["observed_demand"].mean())

        fallback_means = df.loc[is_censored, "series_id"].map(series_means).fillna(global_mean)

        df.loc[is_censored, "latent_demand_est"] = np.maximum(
            df.loc[is_censored, "observed_demand"],
            fallback_means,
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, "observed_demand"]
        return df

    def _impute_clipped_scaling(
        self, df: pd.DataFrame, is_clean: pd.Series, is_censored: pd.Series
    ) -> pd.DataFrame:
        """Baseline: Simply scale up the observed demand by a fixed factor during stockouts."""
        df.loc[is_censored, "latent_demand_est"] = np.maximum(
            df.loc[is_censored, "observed_demand"],
            df.loc[is_censored, "observed_demand"] * self.scaling_factor,
        )
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, "observed_demand"]
        return df
