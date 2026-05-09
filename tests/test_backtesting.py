from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from retail_forecasting.config import ValidationConfig
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
