from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FoldRunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fold_id: int = Field(ge=0)
    horizon: int = Field(gt=0)
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    train_rows: int = Field(ge=0)
    validation_rows: int = Field(ge=0)
    train_series: int = Field(ge=0)
    validation_series: int = Field(ge=0)


class FairCostMetadata(BaseModel):
    """Identity of one fair-cost backtest: what was compared, on what, and how it decided.

    The comparison trains no model. Every strategy reconstructs the same synthetically
    censored days and pays for the same order policy, so the only thing separating their
    costs is the demand signal.

    Read ``best_cost_delta_pct`` and ``best_ci95`` as different quantities. The percentage is
    a ratio of two mean costs; the interval is over the PAIRED per-draw differences and is
    therefore in cost units. ``best_beats_baseline`` is decided on the interval, not on the
    percentage, whose sign alone does not distinguish a real gap from a coin flip.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["fair_cost_synthetic_censoring"] = "fair_cost_synthetic_censoring"
    baseline_strategy: str
    # The ranking flips with the panel the sample was drawn from, so both counts are recorded:
    # sampling the 50-series subset instead of the 500 inverts the order.
    source_panel_series: int = Field(gt=0)
    sampled_series: int = Field(gt=0)
    panel_rows: int = Field(gt=0)
    # Clean rows the supervised imputer's teacher actually fits on. Recorded for the same
    # reason the imputation search records it: a tuned params file is only valid near the
    # teacher size it was tuned at (invariant 41), and neither the file nor the panel says so.
    # It must reflect the WHOLE panel, not the evaluated sample -- shrinking it cost the
    # supervised arm 27% of its reconstruction accuracy while the heuristics were untouched.
    teacher_fit_rows: int = Field(gt=0)
    panel_start: str
    panel_end: str
    n_draws: int = Field(gt=1)
    n_eval_rows: int = Field(gt=0)
    eval_fraction: float = Field(gt=0.0, le=1.0)
    seeds: list[int] = Field(min_length=2)
    # The cost pair the signals are priced with. There is no safety-stock term to record: the
    # order IS the signal, so these two numbers are the whole policy (invariant 44).
    overstock_cost: float = Field(gt=0.0)
    stockout_cost: float = Field(gt=0.0)
    best_strategy: str
    best_cost_delta_pct: float
    best_ci95: list[float] = Field(min_length=2, max_length=2)
    best_beats_baseline: bool
    seed: int
    created_at: str
    git_commit: str | None
