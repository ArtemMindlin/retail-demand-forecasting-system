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
    
    This model implements the Split Conformal Prediction method. If the base model
    already provides quantiles, it adjusts them (widens/narrows). If the base model
    only provides point forecasts, it builds intervals around them using the 
    distribution of absolute residuals.
    """
    
    def __init__(self, base_model: Any):
        self.base_model = base_model
        self.q_hat: Optional[float] = None
        self.confidence_level: Optional[float] = None
        self.alpha: Optional[float] = None
        
    def fit(self, features: Any, target: pd.Series) -> "ConformalForecaster":
        """Fit the underlying base model."""
        self.base_model.fit(features, target)
        return self
        
    def calibrate(self, features: Any, target: pd.Series, alpha: float = 0.2) -> "ConformalForecaster":
        """Calculate the conformal correction factor (q_hat) using a calibration set.
        
        Args:
            features: Features/Panel from a set the model hasn't seen during fit.
            target: True values for the calibration set.
            alpha: Significance level (1 - alpha = desired coverage).
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        
        y_true = target.values
        
        # Check if base model supports quantiles
        if hasattr(self.base_model, "predict_quantiles"):
            q_low_level = alpha / 2
            q_high_level = 1 - (alpha / 2)
            
            preds = self.base_model.predict_quantiles(features)
            # Ensure the required quantiles exist
            low_col = quantile_column_name(q_low_level)
            high_col = quantile_column_name(q_high_level)
            
            if low_col in preds and high_col in preds:
                y_low = preds[low_col]
                y_high = preds[high_col]
                # Conformity score for interval-based conformal (CQR)
                scores = np.maximum(y_low - y_true, y_true - y_high)
            else:
                # Fallback to absolute residual if specific quantiles are missing
                y_pred = self.base_model.predict(features)
                scores = np.abs(y_true - y_pred)
        else:
            # Base model only has point forecasts
            y_pred = self.base_model.predict(features)
            scores = np.abs(y_true - y_pred)
        
        # Calculate q_hat (1-alpha) quantile of scores
        n = len(scores)
        q_level = (1 - alpha) * (1 + 1/n)
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_hat = np.quantile(scores, q_level, method='higher')
        self.q_hat = max(self.q_hat, 0.0)

        return self

    def predict(self, features: Any) -> np.ndarray:
        """Standard point prediction from base model."""
        return self.base_model.predict(features)
        
    def predict_quantiles(self, features: Any) -> dict[str, np.ndarray]:
        """Predict adjusted (conformalized) quantiles."""
        y_pred = self.base_model.predict(features)
        
        # Determine target quantiles from alpha if set, otherwise use defaults
        alpha = self.alpha if self.alpha is not None else 0.2
        q_low_level = alpha / 2
        q_high_level = 1 - (alpha / 2)
        
        low_col = quantile_column_name(q_low_level)
        mid_col = quantile_column_name(0.5)
        high_col = quantile_column_name(q_high_level)

        if self.q_hat is None:
            # Uncalibrated fallback: try base model quantiles or return dummy
            if hasattr(self.base_model, "predict_quantiles"):
                return self.base_model.predict_quantiles(features)
            return {mid_col: y_pred}

        # Check if we should adjust existing quantiles or build from scratch
        if hasattr(self.base_model, "predict_quantiles"):
            raw_preds = self.base_model.predict_quantiles(features)
            adjusted_preds = {}
            for col, values in raw_preds.items():
                quantile_val = float(col.replace("q_", "").replace("_", "."))
                if quantile_val < 0.5:
                    adjusted_preds[col] = np.maximum(values - self.q_hat, 0.0)
                elif quantile_val > 0.5:
                    adjusted_preds[col] = np.maximum(values + self.q_hat, 0.0)
                else:
                    adjusted_preds[col] = values
            return adjusted_preds
        else:
            # Build intervals around point forecast
            return {
                low_col: np.maximum(y_pred - self.q_hat, 0.0),
                mid_col: y_pred,
                high_col: np.maximum(y_pred + self.q_hat, 0.0)
            }

    @property
    def backend_name(self) -> str:
        return f"conformal_{self.base_model.backend_name}"

    @property
    def model_name(self) -> str:
        return self.base_model.model_name
