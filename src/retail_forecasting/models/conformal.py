from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import joblib
import numpy as np
import pandas as pd

from retail_forecasting.utils.io import quantile_column_name


@runtime_checkable
class Forecaster(Protocol):
    """Protocol for models that can be wrapped by ConformalForecaster."""

    backend_name: str
    model_name: str

    def predict(self, features: Any) -> np.ndarray: ...
    def predict_quantiles(self, features: Any) -> dict[str, np.ndarray]: ...


@dataclass
class ConformalForecaster:
    """A universal wrapper that provides conformal prediction guarantees.

    This model implements the Split Conformal Prediction method. It supports
    Mondrian Conformal Prediction, allowing for separate calibration factors
    (q_hat) based on SKU groups (taxonomies, intermittency, etc.).
    """

    base_model: Any
    q_hat: float | None = field(default=None, init=False)
    mondrian_q_hat: dict[Any, float] = field(default_factory=dict, init=False)
    confidence_level: float | None = field(default=None, init=False)
    alpha: float | None = field(default=None, init=False)

    def fit(self, features: Any, target: pd.Series) -> ConformalForecaster:
        """Fit the underlying base model."""
        self.base_model.fit(features, target)
        return self

    def calibrate(
        self,
        features: Any,
        target: pd.Series,
        alpha: float = 0.2,
        group_ids: pd.Series | None = None,
    ) -> ConformalForecaster:
        """Calculate the conformal correction factor (q_hat) using a calibration set."""
        self.alpha = alpha
        self.confidence_level = 1 - alpha

        y_true = target.to_numpy()
        scores = self._calculate_conformity_scores(features, y_true)

        # Global q_hat
        self.q_hat = self._compute_q_hat(scores, alpha)

        # Mondrian (Group-specific) q_hat
        if group_ids is not None:
            group_ids_arr = group_ids.to_numpy()
            unique_groups = np.unique(group_ids_arr)
            for group in unique_groups:
                group_mask = group_ids_arr == group
                if np.any(group_mask):
                    group_scores = scores[group_mask]
                    self.mondrian_q_hat[group] = self._compute_q_hat(group_scores, alpha)

        return self

    def _calculate_conformity_scores(self, features: Any, y_true: np.ndarray) -> np.ndarray:
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
                return cast(np.ndarray, np.maximum(y_low - y_true, y_true - y_high))

        # Fallback to absolute residual
        y_pred = np.asarray(self.base_model.predict(features))
        return cast(np.ndarray, np.abs(y_true - y_pred))

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
        return np.asarray(self.base_model.predict(features))

    def predict_quantiles(
        self, features: Any, group_ids: pd.Series | None = None
    ) -> dict[str, np.ndarray]:
        """Predict adjusted (conformalized) quantiles."""
        y_pred = np.asarray(self.base_model.predict(features))

        alpha = self.alpha if self.alpha is not None else 0.2
        q_low_level = alpha / 2
        q_high_level = 1 - (alpha / 2)

        mid_col = quantile_column_name(0.5)

        if self.q_hat is None:
            if hasattr(self.base_model, "predict_quantiles"):
                return {
                    str(k): np.asarray(v)
                    for k, v in self.base_model.predict_quantiles(features).items()
                }
            return {mid_col: y_pred}

        # Select q_hat values
        if group_ids is not None and self.mondrian_q_hat:
            q_hat_vec = cast(
                np.ndarray,
                group_ids.map(self.mondrian_q_hat).fillna(self.q_hat).to_numpy(),
            )
        else:
            q_hat_vec = np.full(len(y_pred), self.q_hat)

        if hasattr(self.base_model, "predict_quantiles"):
            raw_preds = self.base_model.predict_quantiles(features)
            adjusted_preds: dict[str, np.ndarray] = {}
            for col, values in raw_preds.items():
                quantile_val = float(col.replace("q_", "").replace("_", "."))
                vals = np.asarray(values)
                if quantile_val < 0.5:
                    adjusted_preds[col] = np.maximum(vals - q_hat_vec, 0.0)
                elif quantile_val > 0.5:
                    adjusted_preds[col] = np.maximum(vals + q_hat_vec, 0.0)
                else:
                    adjusted_preds[col] = vals
            return adjusted_preds
        else:
            low_col = quantile_column_name(q_low_level)
            high_col = quantile_column_name(q_high_level)
            return {
                low_col: cast(np.ndarray, np.maximum(y_pred - q_hat_vec, 0.0)),
                mid_col: y_pred,
                high_col: cast(np.ndarray, np.maximum(y_pred + q_hat_vec, 0.0)),
            }

    def save(self, path: Path) -> None:
        """Persist the full forecaster state (base model + conformal calibration) to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> ConformalForecaster:
        """Load a previously saved ConformalForecaster from disk."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ConformalForecaster, got {type(obj)}")
        return obj

    @property
    def backend_name(self) -> str:
        return str(f"conformal_{self.base_model.backend_name}")

    @property
    def model_name(self) -> str:
        return str(self.base_model.model_name)
