from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from retail_forecasting.utils.io import quantile_column_name


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class AutoBoostingModel:
    quantiles: list[float]
    random_seed: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    model_name: str = "auto_boosting"
    backend_name: str = field(init=False, default="unknown")

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "AutoBoostingModel":
        # Build the main point model. This model is used to produce a single prediction per row.
        self.point_model_ = self._build_point_model()

        # Fit the point model to the data.
        self.point_model_.fit(features, target)

        # Create a dictionary to store one fitted model per quantile level.
        self.quantile_models_: dict[float, object] = {}

        # Loop over the requested quantiles.
        for quantile in sorted(set(self.quantiles)):
            quantile_model = self._build_quantile_model(quantile)
            quantile_model.fit(features, target)
            self.quantile_models_[quantile] = quantile_model

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.point_model_.predict(features)
        return np.maximum(np.asarray(predictions, dtype=float), 0.0)

    def predict_quantiles(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        quantile_predictions = {}
        ordered_quantiles = sorted(self.quantile_models_.keys())
        raw_predictions = [
            np.maximum(
                np.asarray(self.quantile_models_[quantile].predict(features), dtype=float),
                0.0,
            )
            for quantile in ordered_quantiles
        ]
        monotonic = np.maximum.accumulate(np.column_stack(raw_predictions), axis=1)
        for index, quantile in enumerate(ordered_quantiles):
            quantile_predictions[quantile_column_name(quantile)] = monotonic[:, index]
        return quantile_predictions

    def _build_point_model(self) -> object:
        if _lightgbm_available():
            import lightgbm as lgb

            self.backend_name = "lightgbm"
            return lgb.LGBMRegressor(
                objective="regression",
                random_state=self.random_seed,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=31,
                max_depth=self.max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
            )

        if _xgboost_available():
            from xgboost import XGBRegressor

            self.backend_name = "xgboost"
            return XGBRegressor(
                objective="reg:squarederror",
                random_state=self.random_seed,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
            )

        self.backend_name = "sklearn_hist_gradient_boosting"
        return HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_iter=self.n_estimators,
            random_state=self.random_seed,
        )

    def _build_quantile_model(self, quantile: float) -> object:
        if _lightgbm_available():
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                random_state=self.random_seed,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=31,
                max_depth=self.max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
            )

        return GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=min(self.n_estimators, 300),
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_seed,
        )
