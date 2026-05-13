from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from retail_forecasting.utils.io import quantile_column_name


@runtime_checkable
class BaseRegressor(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> np.ndarray: ...


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
    overstock_cost: float = 1.0
    stockout_cost: float = 0.0  # 0.0 means standard regression
    model_name: str = "auto_boosting"
    backend_name: str = field(init=False, default="unknown")
    point_model_: BaseRegressor | None = field(init=False, default=None)
    quantile_models_: dict[float, BaseRegressor] = field(
        init=False, default_factory=dict
    )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "AutoBoostingModel":
        # Build the main point model.
        # If stockout_cost > 0, we use a cost-aware objective (pinball loss at critical fractil)
        self.point_model_ = self._build_point_model()

        # Fit the point model to the data.
        self.point_model_.fit(features, target)

        # Loop over the requested quantiles.
        for quantile in sorted(set(self.quantiles)):
            quantile_model = self._build_quantile_model(quantile)
            quantile_model.fit(features, target)
            self.quantile_models_[quantile] = quantile_model

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.point_model_ is None:
            raise ValueError("Model has not been fitted yet.")
        predictions = self.point_model_.predict(features)
        return cast(np.ndarray, np.maximum(np.asarray(predictions, dtype=float), 0.0))

    def predict_quantiles(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        quantile_predictions: dict[str, np.ndarray] = {}
        ordered_quantiles = sorted(self.quantile_models_.keys())
        raw_predictions = [
            np.maximum(
                np.asarray(
                    self.quantile_models_[quantile].predict(features), dtype=float
                ),
                0.0,
            )
            for quantile in ordered_quantiles
        ]
        monotonic = np.maximum.accumulate(np.column_stack(raw_predictions), axis=1)
        for index, quantile in enumerate(ordered_quantiles):
            quantile_predictions[quantile_column_name(quantile)] = monotonic[:, index]
        return quantile_predictions

    def _build_point_model(self) -> BaseRegressor:
        # Calculate critical fractil if costs are provided
        if self.stockout_cost > 0:
            critical_fractil = self.stockout_cost / (
                self.stockout_cost + self.overstock_cost
            )
            print(
                f"🎯 Cost-Aware Training: Optimizing point model for critical fractil τ = {critical_fractil:.4f}"
            )
            return self._build_quantile_model(critical_fractil)

        # Default to standard regression (MSE-like)
        if _lightgbm_available():
            import lightgbm as lgb

            self.backend_name = "lightgbm"
            model: BaseRegressor = lgb.LGBMRegressor(
                objective="regression",
                random_state=self.random_seed,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=31,
                max_depth=self.max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                verbosity=-1,
            )
            return model

        if _xgboost_available():
            from xgboost import XGBRegressor

            self.backend_name = "xgboost"
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=self.random_seed,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            return cast(BaseRegressor, model)

        self.backend_name = "sklearn_hist_gradient_boosting"
        return cast(
            BaseRegressor,
            HistGradientBoostingRegressor(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_iter=self.n_estimators,
                random_state=self.random_seed,
            ),
        )

    def _build_quantile_model(self, quantile: float) -> BaseRegressor:
        if _lightgbm_available():
            import lightgbm as lgb

            self.backend_name = "lightgbm"
            return cast(
                BaseRegressor,
                lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=quantile,
                    random_state=self.random_seed,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    num_leaves=31,
                    max_depth=self.max_depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )

        return cast(
            BaseRegressor,
            GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=min(self.n_estimators, 300),
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_seed,
            ),
        )
