from __future__ import annotations

import numpy as np
import pandas as pd

from retail_forecasting.evaluation.metrics import summarize_predictions


def test_coverage_uses_available_quantile_bounds_instead_of_hardcoded_defaults() -> (
    None
):
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
