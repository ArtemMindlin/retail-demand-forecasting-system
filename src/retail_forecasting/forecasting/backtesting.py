from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retail_forecasting.config import ValidationConfig


class FoldSpec(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    fold_id: int = Field(ge=0)
    horizon: int = Field(gt=0)
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp

    @field_validator(
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        mode="before",
    )
    @classmethod
    def _coerce_timestamp(cls, value: object) -> pd.Timestamp:
        return pd.Timestamp(value)

    @model_validator(mode="after")
    def _validate_temporal_contract(self) -> FoldSpec:
        if self.validation_end_date < self.validation_start_date:
            raise ValueError(
                "validation_end_date must be greater than or equal to "
                "validation_start_date."
            )

        expected_train_end = self.validation_start_date - pd.Timedelta(
            days=self.horizon
        )
        if self.train_end_date != expected_train_end:
            raise ValueError(
                "train_end_date must equal validation_start_date - horizon."
            )

        return self


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
        validation_end_index = (
            validation_start_index + validation_config.fold_size_days - 1
        )
        if validation_end_index > last_valid_index:
            break

        validation_start_date = unique_dates[validation_start_index]
        validation_end_date = unique_dates[validation_end_index]
        train_end_date = validation_start_date - pd.Timedelta(days=horizon)

        folds.append(
            FoldSpec(
                fold_id=fold_id,
                horizon=horizon,
                train_end_date=train_end_date,
                validation_start_date=validation_start_date,
                validation_end_date=validation_end_date,
            )
        )

    if not folds:
        raise ValueError(
            "No valid fold could be created with the current configuration."
        )

    return folds
