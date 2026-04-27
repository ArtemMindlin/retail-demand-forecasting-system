from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from retail_forecasting.models.boosting import AutoBoostingModel
from retail_forecasting.utils.io import quantile_column_name


class ConformalBoostingModel:
    """A wrapper for AutoBoostingModel that provides conformal prediction guarantees.
    
    This model implements the Split Conformal Prediction method to ensure that 
    quantile intervals have the requested coverage probability.
    """
    
    def __init__(self, base_model: AutoBoostingModel):
        self.base_model = base_model
        self.q_hat: Optional[float] = None
        self.confidence_level: Optional[float] = None
        
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ConformalBoostingModel":
        """Fit the underlying base model."""
        self.base_model.fit(features, target)
        return self
        
    def calibrate(self, features: pd.DataFrame, target: pd.Series, alpha: float = 0.2) -> "ConformalBoostingModel":
        """Calculate the conformal correction factor (q_hat) using a calibration set.
        
        Args:
            features: Features from a set the model hasn't seen during fit.
            target: True values for the calibration set.
            alpha: Significance level (1 - alpha = desired coverage). 
                   Default 0.2 means 80% coverage (between q_0.1 and q_0.9).
        """
        self.confidence_level = 1 - alpha
        
        # 1. Get base quantile predictions (e.g., q0.1 and q0.9)
        # We assume the base model was configured with these quantiles.
        q_low_level = alpha / 2
        q_high_level = 1 - (alpha / 2)
        
        preds = self.base_model.predict_quantiles(features)
        y_low = preds[quantile_column_name(q_low_level)]
        y_high = preds[quantile_column_name(q_high_level)]
        y_true = target.values
        
        # 2. Calculate conformity scores (S)
        # S_i measures the distance to the nearest boundary. 
        # Positive means the true value is OUTSIDE the interval.
        # Negative means it is INSIDE.
        scores = np.maximum(y_low - y_true, y_true - y_high)
        
        # 3. Calculate q_hat
        # We want the quantile (1-alpha) of the scores.
        n = len(scores)
        # Small sample correction: (n+1)*(1-alpha) / n
        q_level = (1 - alpha) * (1 + 1/n)
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_hat = np.quantile(scores, q_level, method='higher')
        
        # Guard: If q_hat is negative, the model is already "too wide" (over-conservative).
        # In conformal prediction we usually allow it to be negative to narrow the interval,
        # but in this retail context, to avoid the previous coverage drop, 
        # we will ensure we only widen if we are under-covering.
        self.q_hat = max(self.q_hat, 0.0)

        return self

        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Standard point prediction from base model."""
        return self.base_model.predict(features)
        
    def predict_quantiles(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict adjusted (conformalized) quantiles."""
        raw_preds = self.base_model.predict_quantiles(features)
        
        if self.q_hat is None:
            # If not calibrated, return raw predictions
            return raw_preds
            
        adjusted_preds = {}
        for col, values in raw_preds.items():
            # Parse the quantile value from the column name. e.g., 'q_0_1' -> 0.1
            quantile_str = col.replace("q_", "").replace("_", ".")
            quantile_val = float(quantile_str)
            
            if quantile_val < 0.5:
                adjusted_preds[col] = np.maximum(values - self.q_hat, 0.0)
            elif quantile_val > 0.5:
                adjusted_preds[col] = np.maximum(values + self.q_hat, 0.0)
            else:
                adjusted_preds[col] = values

                
        return adjusted_preds

    @property
    def backend_name(self) -> str:
        return f"conformal_{self.base_model.backend_name}"
