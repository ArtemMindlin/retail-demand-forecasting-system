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


class TuningResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    best_params: BoostingParams
    metadata: TuningMetadata
    pareto_front: list[ParetoTrial] = Field(default_factory=list)
