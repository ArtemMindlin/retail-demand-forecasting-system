from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from retail_forecasting.data.censorship import SupervisedImputer
from retail_forecasting.drift.detectors import PageHinkleyDetector
from retail_forecasting.models.boosting import AutoBoostingModel
from retail_forecasting.models.conformal import ConformalBoostingModel


def test_supervised_imputer_logic():
    """Verify that the imputer corrects demand when stockouts are present."""
    # Create a synthetic panel where demand is 10 but sales are 0 due to stockout
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "series_id": "test_1",
        "observed_demand": 10.0,
        "stockout_hours": 0.0
    })
    
    # Force a stockout on the last day with 0 sales
    df.loc[99, "observed_demand"] = 0.0
    df.loc[99, "stockout_hours"] = 24.0
    
    imputer = SupervisedImputer()
    imputed_df = imputer.impute(df)
    
    # The last day should be imputed with a value close to 10 (since history was 10)
    assert bool(imputed_df.loc[99, "is_imputed"]) is True
    assert imputed_df.loc[99, "observed_demand"] > 0.0
    assert imputed_df.loc[99, "original_observed_demand"] == 0.0


def test_page_hinkley_drift_detection():
    """Verify that Page-Hinkley detects a sudden increase in error."""
    detector = PageHinkleyDetector(threshold=5.0, min_instances=10)
    
    # 1. Stable low error
    for _ in range(20):
        detector.update(1.0)
        
    # 2. Sudden jump in error
    drift_detected = False
    for _ in range(10):
        result = detector.update(10.0) # Error jumps from 1 to 10
        if result.is_drift:
            drift_detected = True
            break
            
    assert drift_detected is True


def test_conformal_interval_widening():
    """Verify that Conformal wrapper widens intervals to ensure coverage."""
    # Dummy base model that predicts narrow intervals [4.5, 5.5]
    class DummyModel:
        def fit(self, X, y): pass
        def predict(self, X): return np.array([5.0] * len(X))
        def predict_quantiles(self, X):
            return {"q_0_1": np.array([4.5] * len(X)), "q_0_9": np.array([5.5] * len(X))}
        @property
        def backend_name(self): return "dummy"

    # Calibration data where the actual value is 10 (far outside [4.5, 5.5])
    X_cal = pd.DataFrame({"feat": [1, 2, 3]})
    y_cal = pd.Series([10.0, 10.0, 10.0])
    
    conformal = ConformalBoostingModel(DummyModel())
    conformal.calibrate(X_cal, y_cal, alpha=0.2)
    
    # q_hat should be large because the error (10 - 5.5) is large
    assert conformal.q_hat > 0
    
    # Adjusted intervals should be wider than [4.5, 5.5]
    preds = conformal.predict_quantiles(X_cal)
    assert preds["q_0_1"][0] < 4.5
    assert preds["q_0_9"][0] > 5.5
