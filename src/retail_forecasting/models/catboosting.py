from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from catboost import CatBoostRegressor

from retail_forecasting.models._quantile_forecaster import QuantileForecasterMixin

logger = logging.getLogger(__name__)


@dataclass
class CatBoostingModel(QuantileForecasterMixin):
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
    overstock_cost: float = 1.0
    stockout_cost: float = 4.0  # must be > 0; drives the critical fractile q* = cu/(cu+co)
    model_name: str = "catboost"
    backend_name: str = field(init=False, default="catboost_official")
    point_model_: CatBoostRegressor | None = field(init=False, default=None)
    quantile_models_: dict[float, CatBoostRegressor] = field(init=False, default_factory=dict)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> CatBoostingModel:
        critical_fractile = self._critical_fractile()
        cat_features = features.select_dtypes(include=["category", "object"]).columns.tolist()

        # 1. Fit Point Model at critical fractile q* (cost-aware, pinball loss)
        self.point_model_ = CatBoostRegressor(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=min(self.max_depth, 16),
            random_seed=self.random_seed,
            loss_function=f"Quantile:alpha={critical_fractile}",
            task_type="CPU",
            l2_leaf_reg=3.0,
            nan_mode="Min",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        logger.info(
            "Cost-aware training: optimizing point model for critical fractile tau = %.4f",
            critical_fractile,
        )
        self.point_model_.fit(features, target, cat_features=cat_features)

        # 2. Fit Quantile Models (One per quantile for consistency)
        for q in sorted(set(self.quantiles)):
            q_model = CatBoostRegressor(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=min(self.max_depth, 16),
                random_seed=self.random_seed,
                loss_function=f"Quantile:alpha={q}",
                task_type="CPU",
                l2_leaf_reg=3.0,
                nan_mode="Min",
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
            q_model.fit(features, target, cat_features=cat_features)
            self.quantile_models_[q] = q_model

        return self
