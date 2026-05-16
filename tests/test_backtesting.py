from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from retail_forecasting.config import ValidationConfig
from retail_forecasting.drift.detectors import PageHinkleyDetector
from retail_forecasting.forecasting.backtesting import (
    FoldSpec,
    build_walk_forward_folds,
)
from tests import make_synthetic_panel


def test_walk_forward_folds_respect_horizon_cutoff() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=90)
    validation = ValidationConfig(initial_train_days=56, n_folds=3, fold_size_days=7)

    folds = build_walk_forward_folds(panel, validation, horizon=7)

    assert len(folds) == 3
    first_fold = folds[0]
    assert first_fold.horizon == 7
    assert (first_fold.validation_start_date - first_fold.train_end_date).days == 7


def test_walk_forward_folds_allow_last_complete_target_origin() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=14)
    validation = ValidationConfig(initial_train_days=5, n_folds=2, fold_size_days=3)

    folds = build_walk_forward_folds(panel, validation, horizon=4)

    assert len(folds) == 2
    assert folds[0].horizon == 4
    assert (folds[0].validation_start_date - folds[0].train_end_date).days == 4
    assert (folds[1].train_end_date - folds[0].train_end_date).days == 3
    assert folds[-1].validation_end_date == panel["date"].max() - pd.Timedelta(days=3)


def test_walk_forward_folds_require_enough_dates_for_last_target() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=13)
    validation = ValidationConfig(initial_train_days=5, n_folds=2, fold_size_days=3)

    with pytest.raises(ValueError, match="Need at least 14, found 13"):
        build_walk_forward_folds(panel, validation, horizon=4)


def test_page_hinkley_detects_sudden_error_increase() -> None:
    detector = PageHinkleyDetector(threshold=5.0, min_instances=10)

    for _ in range(20):
        detector.update(1.0)

    drift_detected = False
    for _ in range(10):
        result = detector.update(10.0)
        if result.is_drift:
            drift_detected = True
            break

    assert drift_detected is True
    assert 21 <= detector.observations_seen <= 30
    assert detector.current_mean_error > 0
    assert detector.cumulative_deviation >= detector.min_cumulative_deviation


def test_fold_spec_validates_temporal_contract() -> None:
    with pytest.raises(
        ValidationError,
        match="train_end_date must equal validation_start_date - horizon",
    ):
        FoldSpec(
            fold_id=0,
            horizon=7,
            train_end_date=pd.Timestamp("2024-01-05"),
            validation_start_date=pd.Timestamp("2024-01-10"),
            validation_end_date=pd.Timestamp("2024-01-16"),
        )
