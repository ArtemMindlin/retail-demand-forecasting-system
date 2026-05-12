from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from catboost import CatBoostRegressor

from retail_forecasting.utils.io import quantile_column_name


@dataclass
class CatBoostingModel:
    """A wrapper for CatBoost that supports point and quantile forecasts.

    CatBoost is known for its excellent handling of categorical features
    and its symmetric tree structure which often leads to better
    generalization in tabular datasets.
    """

    quantiles: list[float]
    random_seed: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    model_name: str = "catboost"
    backend_name: str = field(init=False, default="catboost_official")

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "CatBoostingModel":
        # Identify categorical features
        cat_features = features.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()

        # 1. Fit Point Model (Root Mean Square Error)
        self.point_model_ = CatBoostRegressor(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=min(self.max_depth, 16),  # CatBoost max depth is usually 16
            random_seed=self.random_seed,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        self.point_model_.fit(features, target, cat_features=cat_features)

        # 2. Fit Quantile Models (One per quantile for consistency)
        self.quantile_models_: dict[float, CatBoostRegressor] = {}
        for q in sorted(set(self.quantiles)):
            q_model = CatBoostRegressor(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=min(self.max_depth, 16),
                random_seed=self.random_seed,
                loss_function=f"Quantile:alpha={q}",
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
            q_model.fit(features, target, cat_features=cat_features)
            self.quantile_models_[q] = q_model

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.point_model_.predict(features)
        return np.maximum(np.asarray(predictions, dtype=float), 0.0)

    def predict_quantiles(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        quantile_predictions = {}
        ordered_quantiles = sorted(self.quantile_models_.keys())

        raw_predictions = [
            np.maximum(
                np.asarray(self.quantile_models_[q].predict(features), dtype=float),
                0.0,
            )
            for q in ordered_quantiles
        ]

        # Ensure monotonicity
        monotonic = np.maximum.accumulate(np.column_stack(raw_predictions), axis=1)
        for index, q in enumerate(ordered_quantiles):
            quantile_predictions[quantile_column_name(q)] = monotonic[:, index]

        return quantile_predictions
