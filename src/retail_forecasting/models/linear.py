from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RidgeBaselineModel:
    """A linear regression baseline model using Ridge.
    
    This model serves as an intermediate baseline between the simple 
    seasonal heuristic and complex boosting models.
    """
    random_seed: int = 42
    alpha: float = 1.0
    model_name: str = "ridge_regression"
    backend_name: str = field(init=False, default="sklearn_ridge")

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RidgeBaselineModel":
        # Linear models are sensitive to missing values and scale, 
        # so we impute missing features (e.g., empty lags) and standard scale them.
        self.pipeline_ = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=self.alpha, random_state=self.random_seed)
        )
        self.pipeline_.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.pipeline_.predict(features)
        # Demand cannot be negative
        return np.maximum(np.asarray(predictions, dtype=float), 0.0)
