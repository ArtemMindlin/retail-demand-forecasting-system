from __future__ import annotations

from retail_forecasting.config import ValidationConfig
from retail_forecasting.forecasting.backtesting import build_walk_forward_folds
from tests import make_synthetic_panel


def test_walk_forward_folds_respect_horizon_cutoff() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=90)
    validation = ValidationConfig(initial_train_days=56, n_folds=3, fold_size_days=7)

    folds = build_walk_forward_folds(panel, validation, horizon=7)

    assert len(folds) == 3
    first_fold = folds[0]
    assert (first_fold.validation_start_date - first_fold.train_end_date).days == 7
