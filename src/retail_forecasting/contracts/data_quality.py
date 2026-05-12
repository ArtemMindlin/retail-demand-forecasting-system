from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DataQualityIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: str
    code: str
    message: str


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
