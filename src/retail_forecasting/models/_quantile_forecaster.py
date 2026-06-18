from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.utils.io import quantile_column_name, rearrange_quantiles


class QuantileForecasterMixin:
    """Shared point/quantile prediction logic for the boosting forecasters.

    Concrete classes (``AutoBoostingModel``, ``CatBoostingModel``) provide the
    attributes declared below; the mixin only depends on those, so the concrete
    classes keep full ownership of how the underlying models are built and fitted.
    """

    # Provided by the concrete dataclasses (declared here for the type checker).
    point_model_: Any
    quantile_models_: dict[float, Any]
    overstock_cost: float
    stockout_cost: float

    def _critical_fractile(self) -> float:
        """Critical fractile q* = cu / (cu + co); validates that cu > 0."""
        if self.stockout_cost <= 0:
            raise ValueError(
                "stockout_cost must be > 0. Provide valid inventory costs in the configuration."
            )
        return self.stockout_cost / (self.stockout_cost + self.overstock_cost)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.point_model_ is None:
            raise ValueError("Model has not been fitted yet.")
        predictions = self.point_model_.predict(features)
        return np.maximum(np.asarray(predictions, dtype=np.float64), 0.0)

    def predict_quantiles(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        ordered_quantiles = sorted(self.quantile_models_.keys())
        raw_predictions = [
            np.maximum(
                np.asarray(self.quantile_models_[quantile].predict(features), dtype=np.float64),
                0.0,
            )
            for quantile in ordered_quantiles
        ]
        # Enforce monotonicity via Chernozhukov rearrangement (Econometrica, 2010).
        monotonic = rearrange_quantiles(raw_predictions)
        return {
            quantile_column_name(quantile): monotonic[:, index]
            for index, quantile in enumerate(ordered_quantiles)
        }
