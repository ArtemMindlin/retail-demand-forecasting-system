from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.evaluation.metrics import summarize_predictions
from retail_forecasting.models.conformal import ConformalForecaster


def test_coverage_uses_available_quantile_bounds_instead_of_hardcoded_defaults() -> None:
    predictions = _prediction_frame().assign(
        q_0_2=[0.0, 2.0, 8.0],
        q_0_5=[1.0, 3.0, 10.0],
        q_0_8=[2.0, 4.0, 9.0],
    )

    metrics_summary, fold_metrics = summarize_predictions(predictions)
    record = metrics_summary.iloc[0]

    assert "coverage_q_0_2_q_0_8" in metrics_summary.columns
    assert "coverage_q_0_1_q_0_9" not in metrics_summary.columns
    assert np.isclose(record["coverage_q_0_2_q_0_8"], 2.0 / 3.0)
    assert "pinball_q_0_2" in metrics_summary.columns
    assert "pinball_q_0_5" in metrics_summary.columns
    assert "pinball_q_0_8" in metrics_summary.columns
    assert "coverage_q_0_2_q_0_8" in fold_metrics.columns


def test_default_quantile_bounds_keep_existing_coverage_column_name() -> None:
    predictions = _prediction_frame().assign(
        q_0_1=[0.0, 2.0, 8.0],
        q_0_5=[1.0, 3.0, 10.0],
        q_0_9=[2.0, 4.0, 9.0],
    )

    metrics_summary, _ = summarize_predictions(predictions)

    assert "coverage_q_0_1_q_0_9" in metrics_summary.columns
    assert np.isclose(metrics_summary.loc[0, "coverage_q_0_1_q_0_9"], 2.0 / 3.0)


def test_single_quantile_does_not_create_interval_coverage() -> None:
    predictions = _prediction_frame().assign(q_0_5=[1.0, 3.0, 10.0])

    metrics_summary, _ = summarize_predictions(predictions)

    coverage_columns = [
        column for column in metrics_summary.columns if column.startswith("coverage_")
    ]
    assert coverage_columns == []
    assert "pinball_q_0_5" in metrics_summary.columns


def test_conformal_forecaster_widens_intervals_to_ensure_coverage() -> None:
    class DummyModel:
        def fit(self, x: Any, y: Any) -> None:
            pass

        def predict(self, x: Any) -> np.ndarray:
            return np.array([5.0] * len(x))

        def predict_quantiles(self, x: Any) -> dict[str, np.ndarray]:
            return {
                "q_0_1": np.array([4.5] * len(x)),
                "q_0_9": np.array([5.5] * len(x)),
            }

        @property
        def backend_name(self) -> str:
            return "dummy"

    x_cal = pd.DataFrame({"feat": [1, 2, 3]})
    y_cal = pd.Series([10.0, 10.0, 10.0])

    conformal = ConformalForecaster(DummyModel())  # type: ignore[arg-type]
    conformal.calibrate(x_cal, y_cal, alpha=0.2)

    assert conformal.q_hat > 0
    preds = conformal.predict_quantiles(x_cal)
    assert preds["q_0_1"][0] < 4.5
    assert preds["q_0_9"][0] > 5.5


def test_mondrian_conformal_uses_group_specific_q_hat() -> None:
    class DummyModel:
        def predict(self, x: Any) -> np.ndarray:
            return np.array([10.0] * len(x))

        def predict_quantiles(self, x: Any) -> dict[str, np.ndarray]:
            return {
                "q_0_1": np.array([8.0] * len(x)),
                "q_0_9": np.array([12.0] * len(x)),
            }

        @property
        def backend_name(self) -> str:
            return "dummy"

        @property
        def model_name(self) -> str:
            return "dummy"

    x_cal = pd.DataFrame({"feat": range(10)})
    y_cal = pd.Series([10.0] * 5 + [20.0] * 5)
    groups = pd.Series(["A"] * 5 + ["B"] * 5)

    conformal = ConformalForecaster(DummyModel())  # type: ignore[arg-type]
    conformal.calibrate(x_cal, y_cal, alpha=0.2, group_ids=groups)

    assert "A" in conformal.mondrian_q_hat
    assert "B" in conformal.mondrian_q_hat
    assert conformal.mondrian_q_hat["A"] < conformal.mondrian_q_hat["B"]

    x_test = pd.DataFrame({"feat": [1, 2]})
    test_groups = pd.Series(["A", "B"])
    preds = conformal.predict_quantiles(x_test, group_ids=test_groups)

    width_a = preds["q_0_9"][0] - preds["q_0_1"][0]
    width_b = preds["q_0_9"][1] - preds["q_0_1"][1]
    assert width_a < width_b


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["model"] * 3,
            "backend_name": ["backend"] * 3,
            "fold_id": [0, 0, 0],
            "y_true": [1.0, 3.0, 10.0],
            "y_pred": [1.0, 2.5, 9.5],
        }
    )
