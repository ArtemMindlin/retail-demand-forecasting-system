from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import joblib
import numpy as np
import pandas as pd

from retail_forecasting.utils.io import quantile_column_name, quantile_level_from_column

# Default miscoverage level (alpha=0.2 -> 80% nominal interval) used when the
# detector has not been calibrated with an explicit alpha.
DEFAULT_ALPHA = 0.2


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
        alpha: float = DEFAULT_ALPHA,
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
            for group in np.unique(group_ids_arr):
                group_mask = group_ids_arr == group
                self.mondrian_q_hat[group] = self._compute_q_hat(scores[group_mask], alpha)

        return self

    def _calculate_conformity_scores(self, features: Any, y_true: np.ndarray) -> np.ndarray:
        if hasattr(self.base_model, "predict_quantiles"):
            alpha = self.alpha if self.alpha is not None else DEFAULT_ALPHA
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
        has_base_quantiles = hasattr(self.base_model, "predict_quantiles")

        # Not yet calibrated: pass through the base quantiles (or just the point forecast).
        if self.q_hat is None:
            if has_base_quantiles:
                return {
                    str(k): np.asarray(v)
                    for k, v in self.base_model.predict_quantiles(features).items()
                }
            return {quantile_column_name(0.5): y_pred}

        q_hat_vec = self._resolve_q_hat_vector(y_pred, group_ids)

        if has_base_quantiles:
            return self._adjust_existing_quantiles(
                self.base_model.predict_quantiles(features), q_hat_vec
            )
        return self._synthesize_quantiles(y_pred, q_hat_vec)

    def _resolve_q_hat_vector(self, y_pred: np.ndarray, group_ids: pd.Series | None) -> np.ndarray:
        """Per-row conformal radius: Mondrian group value when available, else the global q_hat."""
        if group_ids is not None and self.mondrian_q_hat:
            return cast(
                np.ndarray,
                group_ids.map(self.mondrian_q_hat).fillna(self.q_hat).to_numpy(),
            )
        return np.full(len(y_pred), self.q_hat)

    def _adjust_existing_quantiles(
        self, raw_preds: dict[str, np.ndarray], q_hat_vec: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Shift each base quantile by ±q_hat (lower down, upper up; median unchanged)."""
        adjusted: dict[str, np.ndarray] = {}
        for col, values in raw_preds.items():
            quantile_val = quantile_level_from_column(col)
            vals = np.asarray(values)
            if quantile_val < 0.5:
                adjusted[col] = np.maximum(vals - q_hat_vec, 0.0)
            elif quantile_val > 0.5:
                adjusted[col] = np.maximum(vals + q_hat_vec, 0.0)
            else:
                adjusted[col] = vals
        return adjusted

    def _synthesize_quantiles(
        self, y_pred: np.ndarray, q_hat_vec: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Build a symmetric [low, mid, high] interval from the point forecast ± q_hat."""
        alpha = self.alpha if self.alpha is not None else DEFAULT_ALPHA
        return {
            quantile_column_name(alpha / 2): cast(np.ndarray, np.maximum(y_pred - q_hat_vec, 0.0)),
            quantile_column_name(0.5): y_pred,
            quantile_column_name(1 - (alpha / 2)): cast(
                np.ndarray, np.maximum(y_pred + q_hat_vec, 0.0)
            ),
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
