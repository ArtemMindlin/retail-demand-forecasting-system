from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retail_forecasting.config import ValidationConfig


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp


def build_walk_forward_folds(
    panel: pd.DataFrame,
    validation_config: ValidationConfig,
    horizon: int,
) -> list[FoldSpec]:
    unique_dates = sorted(pd.to_datetime(panel["date"]).drop_duplicates())
    minimum_dates_required = (
        validation_config.initial_train_days
        + validation_config.n_folds * validation_config.fold_size_days
        + horizon
    )
    if len(unique_dates) < minimum_dates_required:
        raise ValueError(
            "Not enough dates to create the requested walk-forward folds. "
            f"Need at least {minimum_dates_required}, found {len(unique_dates)}."
        )

    folds: list[FoldSpec] = []
    last_valid_index = len(unique_dates) - horizon - 1

    for fold_id in range(validation_config.n_folds):
        validation_start_index = (
            validation_config.initial_train_days
            + fold_id * validation_config.fold_size_days
        )
        validation_end_index = validation_start_index + validation_config.fold_size_days - 1
        if validation_end_index > last_valid_index:
            break

        validation_start_date = unique_dates[validation_start_index]
        validation_end_date = unique_dates[validation_end_index]
        train_end_date = validation_start_date - pd.Timedelta(days=horizon)

        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_end_date=train_end_date,
                validation_start_date=validation_start_date,
                validation_end_date=validation_end_date,
            )
        )

    if not folds:
        raise ValueError("No valid fold could be created with the current configuration.")

    return folds
