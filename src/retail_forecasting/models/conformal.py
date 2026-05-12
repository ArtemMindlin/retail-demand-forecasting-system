from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Any, Protocol, runtime_checkable

from retail_forecasting.utils.io import quantile_column_name


@runtime_checkable
class Forecaster(Protocol):
    """Protocol for models that can be wrapped by ConformalForecaster."""

    backend_name: str
    model_name: str

    def predict(self, features: Any) -> np.ndarray: ...
    def predict_quantiles(self, features: Any) -> dict[str, np.ndarray]: ...


class ConformalForecaster:
    """A universal wrapper that provides conformal prediction guarantees.

    This model implements the Split Conformal Prediction method. It supports
    Mondrian Conformal Prediction, allowing for separate calibration factors
    (q_hat) based on SKU groups (taxonomies, intermittency, etc.).
    """

    def __init__(self, base_model: Any):
        self.base_model = base_model
        self.q_hat: Optional[float] = None
        self.mondrian_q_hat: dict[Any, float] = {}
        self.confidence_level: Optional[float] = None
        self.alpha: Optional[float] = None

    def fit(self, features: Any, target: pd.Series) -> "ConformalForecaster":
        """Fit the underlying base model."""
        self.base_model.fit(features, target)
        return self

    def calibrate(
        self,
        features: Any,
        target: pd.Series,
        alpha: float = 0.2,
        group_ids: Optional[pd.Series] = None,
    ) -> "ConformalForecaster":
        """Calculate the conformal correction factor (q_hat) using a calibration set.

        Args:
            features: Features/Panel from a set the model hasn't seen during fit.
            target: True values for the calibration set.
            alpha: Significance level (1 - alpha = desired coverage).
            group_ids: Optional series for Mondrian Conformal Prediction.
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha

        y_true = target.values
        scores = self._calculate_conformity_scores(features, y_true)

        # Global q_hat
        self.q_hat = self._compute_q_hat(scores, alpha)

        # Mondrian (Group-specific) q_hat
        if group_ids is not None:
            group_ids_arr = group_ids.values
            unique_groups = np.unique(group_ids_arr)
            for group in unique_groups:
                group_mask = group_ids_arr == group
                if np.any(group_mask):
                    group_scores = scores[group_mask]
                    self.mondrian_q_hat[group] = self._compute_q_hat(
                        group_scores, alpha
                    )

        return self

    def _calculate_conformity_scores(
        self, features: Any, y_true: np.ndarray
    ) -> np.ndarray:
        # Check if base model supports quantiles
        if hasattr(self.base_model, "predict_quantiles"):
            alpha = self.alpha if self.alpha is not None else 0.2
            q_low_level = alpha / 2
            q_high_level = 1 - (alpha / 2)

            preds = self.base_model.predict_quantiles(features)
            low_col = quantile_column_name(q_low_level)
            high_col = quantile_column_name(q_high_level)

            if low_col in preds and high_col in preds:
                y_low = preds[low_col]
                y_high = preds[high_col]
                return np.maximum(y_low - y_true, y_true - y_high)

        # Fallback to absolute residual
        y_pred = self.base_model.predict(features)
        return np.abs(y_true - y_pred)

    def _compute_q_hat(self, scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return 0.0
        q_level = (1 - alpha) * (1 + 1 / n)
        q_level = min(max(q_level, 0.0), 1.0)
        q_hat = np.quantile(scores, q_level, method="higher")
        return float(max(q_hat, 0.0))

    def predict(self, features: Any) -> np.ndarray:
        """Standard point prediction from base model."""
        return self.base_model.predict(features)

    def predict_quantiles(
        self, features: Any, group_ids: Optional[pd.Series] = None
    ) -> dict[str, np.ndarray]:
        """Predict adjusted (conformalized) quantiles."""
        y_pred = self.base_model.predict(features)

        alpha = self.alpha if self.alpha is not None else 0.2
        q_low_level = alpha / 2
        q_high_level = 1 - (alpha / 2)

        low_col = quantile_column_name(q_low_level)
        mid_col = quantile_column_name(0.5)
        high_col = quantile_column_name(q_high_level)

        if self.q_hat is None:
            if hasattr(self.base_model, "predict_quantiles"):
                return self.base_model.predict_quantiles(features)
            return {mid_col: y_pred}

        # Select q_hat values
        if group_ids is not None and self.mondrian_q_hat:
            # Vectorized selection of q_hat
            # Fallback to global q_hat for unknown groups
            q_hat_vec = group_ids.map(self.mondrian_q_hat).fillna(self.q_hat).values
        else:
            q_hat_vec = np.full(len(y_pred), self.q_hat)

        if hasattr(self.base_model, "predict_quantiles"):
            raw_preds = self.base_model.predict_quantiles(features)
            adjusted_preds = {}
            for col, values in raw_preds.items():
                quantile_val = float(col.replace("q_", "").replace("_", "."))
                if quantile_val < 0.5:
                    adjusted_preds[col] = np.maximum(values - q_hat_vec, 0.0)
                elif quantile_val > 0.5:
                    adjusted_preds[col] = np.maximum(values + q_hat_vec, 0.0)
                else:
                    adjusted_preds[col] = values
            return adjusted_preds
        else:
            return {
                low_col: np.maximum(y_pred - q_hat_vec, 0.0),
                mid_col: y_pred,
                high_col: np.maximum(y_pred + q_hat_vec, 0.0),
            }

    @property
    def backend_name(self) -> str:
        return f"conformal_{self.base_model.backend_name}"

    @property
    def model_name(self) -> str:
        return self.base_model.model_name
