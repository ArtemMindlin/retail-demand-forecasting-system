from __future__ import annotations

import pandas as pd
import pytest

from retail_forecasting.evaluation.metrics import _build_metric_record, winkler_score


def test_winkler_score_basic():
    actual = pd.Series([10.0, 10.0, 10.0])
    lower = pd.Series([8.0, 8.0, 8.0])
    upper = pd.Series([12.0, 12.0, 12.0])
    alpha = 0.2  # 80% interval

    # All actuals are inside the interval. Winkler = Width = 4.0
    score = winkler_score(actual, lower, upper, alpha)
    assert score == 4.0


def test_winkler_score_penalties():
    actual = pd.Series([15.0])  # Outside upper (12)
    lower = pd.Series([8.0])
    upper = pd.Series([12.0])
    alpha = 0.2

    # Width = 4
    # Over penalty = (2/0.2) * (15 - 12) = 10 * 3 = 30
    # Total = 34
    score = winkler_score(actual, lower, upper, alpha)
    assert score == 34.0

    actual_under = pd.Series([5.0])  # Outside lower (8)
    # Under penalty = (2/0.2) * (8 - 5) = 10 * 3 = 30
    # Total = 34
    score_under = winkler_score(actual_under, lower, upper, alpha)
    assert score_under == 34.0


def test_build_metric_record_includes_calibration():
    df = pd.DataFrame(
        {
            "y_true": [10.0, 20.0, 30.0],
            "y_pred": [11.0, 19.0, 31.0],
            "q_0_1": [8.0, 18.0, 28.0],
            "q_0_9": [12.0, 22.0, 32.0],
        }
    )

    record = _build_metric_record(df, "test_model", "test_backend")

    assert "interval_coverage" in record
    assert "interval_width" in record
    assert "winkler_score" in record
    assert record["interval_coverage"] == 1.0
    assert record["interval_width"] == 4.0
    assert record["winkler_score"] == 4.0


def test_build_metric_record_with_mismatched_coverage():
    df = pd.DataFrame(
        {
            "y_true": [15.0, 20.0, 30.0],  # 15 is outside [8, 12]
            "y_pred": [11.0, 19.0, 31.0],
            "q_0_1": [8.0, 18.0, 28.0],
            "q_0_9": [12.0, 22.0, 32.0],
        }
    )

    record = _build_metric_record(df, "test_model", "test_backend")

    # Coverage should be 2/3
    assert record["interval_coverage"] == pytest.approx(0.666, rel=1e-2)
    # Winkler for first row: 4 + (2/0.2)*(15-12) = 34
    # Winkler for others: 4
    # Mean: (34 + 4 + 4) / 3 = 42 / 3 = 14
    assert record["winkler_score"] == 14.0
