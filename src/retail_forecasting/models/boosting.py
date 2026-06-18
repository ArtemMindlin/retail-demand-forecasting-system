from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from retail_forecasting.models._quantile_forecaster import QuantileForecasterMixin

logger = logging.getLogger(__name__)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class AutoBoostingModel(QuantileForecasterMixin):
    quantiles: list[float]
    random_seed: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    overstock_cost: float = 1.0
    stockout_cost: float = 4.0  # must be > 0; drives the critical fractile q* = cu/(cu+co)
    model_name: str = "auto_boosting"
    backend_name: str = field(init=False, default="unknown")
    point_model_: Any = field(init=False, default=None)
    quantile_models_: dict[float, Any] = field(init=False, default_factory=dict)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> AutoBoostingModel:
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

    def _build_point_model(self) -> Any:
        # The point model always optimizes the critical fractile (pinball at q*).
        # Training against RMSE/MAE is inconsistent with the system's objective of
        # minimizing logistical cost, so it is not supported.
        critical_fractile = self._critical_fractile()
        logger.info(
            "Cost-aware training: optimizing point model for critical fractile tau = %.4f",
            critical_fractile,
        )
        self.backend_name = "lightgbm" if _lightgbm_available() else "sklearn_gradient_boosting"
        return self._build_quantile_model(critical_fractile)

    def _build_quantile_model(self, quantile: float) -> Any:
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
                n_jobs=-1,
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
