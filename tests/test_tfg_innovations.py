from __future__ import annotations

import numpy as np
import pandas as pd
from retail_forecasting.data.censorship import LatentDemandImputer
from retail_forecasting.drift.detectors import PageHinkleyDetector
from retail_forecasting.models.conformal import ConformalForecaster


def test_supervised_imputer_logic():
    """Verify that the imputer corrects demand when stockouts are present."""
    # Create a synthetic panel where demand is 10 but sales are 0 due to stockout
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )

    # Force a stockout on the last day with 0 sales
    df.loc[99, "observed_demand"] = 0.0
    df.loc[99, "stockout_hours"] = 24.0

    imputer = LatentDemandImputer(strategy="supervised")
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
        result = detector.update(10.0)  # Error jumps from 1 to 10
        if result.is_drift:
            drift_detected = True
            break

    assert drift_detected is True
    assert 21 <= detector.observations_seen <= 30
    assert detector.current_mean_error > 0
    assert detector.cumulative_deviation >= detector.min_cumulative_deviation


def test_conformal_interval_widening():
    """Verify that Conformal wrapper widens intervals to ensure coverage."""

    # Dummy base model that predicts narrow intervals [4.5, 5.5]
    class DummyModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.array([5.0] * len(X))

        def predict_quantiles(self, X):
            return {
                "q_0_1": np.array([4.5] * len(X)),
                "q_0_9": np.array([5.5] * len(X)),
            }

        @property
        def backend_name(self):
            return "dummy"

    # Calibration data where the actual value is 10 (far outside [4.5, 5.5])
    X_cal = pd.DataFrame({"feat": [1, 2, 3]})
    y_cal = pd.Series([10.0, 10.0, 10.0])

    conformal = ConformalForecaster(DummyModel())
    conformal.calibrate(X_cal, y_cal, alpha=0.2)

    # q_hat should be large because the error (10 - 5.5) is large
    assert conformal.q_hat > 0

    # Adjusted intervals should be wider than [4.5, 5.5]
    preds = conformal.predict_quantiles(X_cal)
    assert preds["q_0_1"][0] < 4.5
    assert preds["q_0_9"][0] > 5.5


def test_mondrian_conformal_calibration():
    class DummyModel:
        def predict(self, X):
            return np.array([10.0] * len(X))

        def predict_quantiles(self, X):
            return {
                "q_0_1": np.array([8.0] * len(X)),
                "q_0_9": np.array([12.0] * len(X)),
            }

        @property
        def backend_name(self):
            return "dummy"

        @property
        def model_name(self):
            return "dummy"

    # Calibration data:
    # Group A: Target is 10 (perfect fit, score = 0)
    # Group B: Target is 20 (large error, score = 8)
    X_cal = pd.DataFrame({"feat": range(10)})
    y_cal = pd.Series([10.0] * 5 + [20.0] * 5)
    groups = pd.Series(["A"] * 5 + ["B"] * 5)

    conformal = ConformalForecaster(DummyModel())
    conformal.calibrate(X_cal, y_cal, alpha=0.2, group_ids=groups)

    # Check that Mondrian q_hats are different
    assert "A" in conformal.mondrian_q_hat
    assert "B" in conformal.mondrian_q_hat
    assert conformal.mondrian_q_hat["A"] < conformal.mondrian_q_hat["B"]

    # Check predictions with groups
    X_test = pd.DataFrame({"feat": [1, 2]})
    test_groups = pd.Series(["A", "B"])
    preds = conformal.predict_quantiles(X_test, group_ids=test_groups)

    # Group A should have narrower intervals than Group B
    width_A = preds["q_0_9"][0] - preds["q_0_1"][0]
    width_B = preds["q_0_9"][1] - preds["q_0_1"][1]
    assert width_A < width_B
