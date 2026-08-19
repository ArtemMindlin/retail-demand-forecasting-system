from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BoostingParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_depth: int = Field(gt=0)


class ImputationBoostingParams(BaseModel):
    """LGBM hyperparameters for the supervised imputer's teacher, kept separate from
    ``BoostingParams`` (the forecasting model's own tuning contract) since the two searches
    tune different hyperparameter sets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_depth: int = Field(gt=0)
    # LightGBM's own hard floor is num_leaves > 1 (config_auto.cpp), so 2 is the true minimum --
    # confirmed empirically against the installed lightgbm==4.6.0, not assumed from docs.
    num_leaves: int = Field(gt=1)
    min_child_samples: int = Field(gt=0)
    colsample_bytree: float = Field(gt=0.0, le=1.0)
    subsample: float = Field(gt=0.0, le=1.0)
    subsample_freq: int = Field(ge=0)
    reg_alpha: float = Field(ge=0.0)
    reg_lambda: float = Field(ge=0.0)
    min_data_per_group: int = Field(gt=0)
    # LightGBM's own floor is cat_smooth >= 0.0 (docs + empirical check), not > 0 -- an earlier
    # search bound of 1.0 was an unverified assumption, not a real LightGBM constraint, and it
    # pinned the winner to that edge until this was checked.
    cat_smooth: float = Field(ge=0.0)
    max_bin: int = Field(gt=0)


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
    ``default_mae_validation`` come from holdouts the search never saw. ``persisted`` is
    decided on ``improvement_ci95`` -- the Student-t interval of the per-draw difference --
    rather than on ``improvement_pct``, whose sign alone does not distinguish a real gain
    from a coin flip.

    ``persisted`` needs BOTH gates to pass: ``improvement_ci95`` below zero (tuning beats the
    untuned defaults, the number the thesis cites) AND ``beats_incumbent`` (this search beats
    the winner already on disk). The second exists because the first cannot see the incumbent
    at all, so a worse search silently replaced a better one -- observed twice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["optuna_imputation_lgbm"] = "optuna_imputation_lgbm"
    n_trials_requested: int = Field(gt=0)
    best_mae_selection: float = Field(ge=0)
    best_mae_validation: float = Field(ge=0)
    default_mae_validation: float = Field(ge=0)
    improvement_pct: float
    improvement_ci95: list[float] = Field(min_length=2, max_length=2)
    # Second gate, guarding the OVERWRITE rather than the claim: the run above answers "does
    # tuning beat no tuning", which says nothing about whether this particular search beat the
    # winner already on disk. Both None on the first run, when there is no incumbent to beat.
    incumbent_mae_validation: float | None = Field(default=None, ge=0)
    beats_incumbent: bool | None = None
    persisted: bool
    n_selection_holdouts: int = Field(gt=0)
    n_validation_holdouts: int = Field(gt=0)
    # The split is TEMPORAL: both windows cover every series of one full panel, and the
    # teacher is fitted on that whole panel while only one window's rows are censored. So the
    # split is described by the cut date and by how many rows each window can score, not by
    # series counts -- an earlier design partitioned by series and recorded those instead.
    n_series: int = Field(gt=0)
    selection_window_end: str
    validation_window_start: str
    n_selection_eval_rows: int = Field(gt=0)
    n_validation_eval_rows: int = Field(gt=0)
    selection_seeds: list[int]
    validation_seeds: list[int]
    # Clean rows the LGBM teacher actually fits on. Recorded because it turned out to be the
    # variable that decides whether tuning helps at all: measured against the untuned defaults,
    # the gain shrinks monotonically as this grows (-5.3% at 467 rows, -1.7% at 1930) and
    # REVERSES at scale (+12.4% worse at 14243). A tuned params file is therefore only valid
    # near the teacher size it was tuned at, which the file itself cannot express.
    teacher_fit_rows: int = Field(ge=0)
    seed: int
    created_at: str
    git_commit: str | None
    best_params: ImputationBoostingParams
