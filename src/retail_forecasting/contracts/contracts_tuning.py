from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BoostingParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_depth: int = Field(gt=0)


class TuningMetadata(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    strategy: Literal["optuna_temporal_holdout", "optuna_multiobjective_pareto", "default_fallback"]
    n_trials_requested: int = Field(gt=0)
    best_score: float | None = Field(default=None, ge=0)
    train_rows: int = Field(ge=0)
    validation_rows: int = Field(ge=0)
    validation_cutoff: pd.Timestamp
    feature_count: int = Field(ge=0)
    target_col: str
    best_params: BoostingParams

    @field_validator("validation_cutoff", mode="before")
    @classmethod
    def _coerce_timestamp(cls, value: Any) -> pd.Timestamp:
        return pd.Timestamp(value)


class ParetoTrial(BaseModel):
    """A single Optuna trial on the multi-objective (Pinball vs Winkler) plane."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_number: int = Field(ge=0)
    pinball: float = Field(ge=0)
    winkler: float
    is_on_front: bool
    is_selected: bool
    n_estimators: int | None = Field(default=None, ge=0)
    learning_rate: float | None = Field(default=None, ge=0)
    max_depth: int | None = Field(default=None, ge=0)


class TuningResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    best_params: BoostingParams
    metadata: TuningMetadata
    pareto_front: list[ParetoTrial] = Field(default_factory=list)


class ImputationTuningMetadata(BaseModel):
    """Metadata for a single-objective Optuna search over the supervised imputer's LGBM.

    Scores are split in two because they answer different questions: ``best_mae_selection`` is
    the objective the search minimized (averaged over the selection holdouts, so it is in-sample
    for the search and cannot evidence a gain), while ``best_mae_validation`` and
    ``default_mae_validation`` come from holdouts the search never saw and are what
    ``improvement_pct`` and ``persisted`` are decided on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["optuna_imputation_lgbm"] = "optuna_imputation_lgbm"
    n_trials_requested: int = Field(gt=0)
    best_mae_selection: float = Field(ge=0)
    best_mae_validation: float = Field(ge=0)
    default_mae_validation: float = Field(ge=0)
    improvement_pct: float
    persisted: bool
    n_selection_holdouts: int = Field(gt=0)
    n_validation_holdouts: int = Field(gt=0)
    selection_seeds: list[int]
    validation_seeds: list[int]
    train_rows: int = Field(ge=0)
    eval_rows: int = Field(ge=0)
    seed: int
    created_at: str
    git_commit: str | None
    best_params: BoostingParams
