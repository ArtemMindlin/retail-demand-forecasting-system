from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DataQualityIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: str
    code: str
    message: str


class DataQualityError(ValueError):
    """Raised when the prepared panel fails blocking runtime quality checks."""


class DataQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_mode: str
    checked_rows: int = Field(ge=0)
    checked_series: int = Field(ge=0)
    date_min: str | None = None
    date_max: str | None = None
    warning_count: int = Field(ge=0)
    blocking_error_count: int = Field(ge=0)
    warnings: list[DataQualityIssue] = Field(default_factory=list)
    blocking_errors: list[DataQualityIssue] = Field(default_factory=list)
    passed: bool


class EdaRunMetadata(BaseModel):
    """What one EDA run analysed, so its figures can be cited under the rule in docs/runs.md.

    Every other run mode leaves a record of itself -- the imputation search, the fair-cost
    backtest, the experiment's promotion decision -- and the EDA left none, so a directory of
    figures under `reports/` could not be traced to a commit or a config. Chapter 3 cites
    numbers out of those figures.

    `panel_source` is the field that earns its place hardest. The panel cache is keyed on four
    `dataset` fields only, NOT on the preprocessing config nor on the code version, so a panel
    built before a change to `prepare_daily_panel` is served unchanged afterwards under a run
    whose `git_commit` says otherwise. Recording the parquet it actually read is what makes
    that detectable.

    The `configured_*` fields sit beside the measured ones on purpose: the pair is what says
    whether the panel on disk is the panel the config asked for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    panel_source: str
    n_series: int = Field(ge=0)
    rows: int = Field(ge=0)
    date_min: str
    date_max: str
    configured_top_n_series: int | None = Field(default=None, gt=0)
    configured_min_history_days: int = Field(ge=0)
    configured_max_rows: int | None = Field(default=None, gt=0)
    imputation_strategy: str
    drop_negative_sales: bool
    fill_missing_values: bool
    config_hash: str
    config_path: str | None
    created_at: str
    git_commit: str | None
