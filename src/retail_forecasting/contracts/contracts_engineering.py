from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureMetadata(BaseModel):
    """Auditable metadata for feature frame construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["supervised", "inference"]
    feature_columns: list[str] = Field(min_length=1)
    target_column: str | None = None
    horizon: int | None = Field(default=None, gt=0)
    lags: list[int] = Field(min_length=1)
    rolling_windows: list[int] = Field(min_length=1)
    input_rows: int = Field(ge=0)
    output_rows: int = Field(ge=0)
    dropped_rows_missing_target: int = Field(default=0, ge=0)
    dropped_rows_missing_features: int = Field(default=0, ge=0)
    rows_not_latest_origin: int = Field(default=0, ge=0)


class InferenceFallbackMetadata(BaseModel):
    """Auditable metadata for inference-time cold-start fallback planning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_columns: list[str] = Field(min_length=1)
    horizon: int = Field(gt=0)
    lags: list[int] = Field(min_length=1)
    rolling_windows: list[int] = Field(min_length=1)
    input_rows: int = Field(ge=0)
    output_rows: int = Field(ge=0)
    model_rows: int = Field(ge=0)
    cold_start_rows: int = Field(ge=0)
    fallback_rows_series: int = Field(default=0, ge=0)
    fallback_rows_product: int = Field(default=0, ge=0)
    fallback_rows_third_category: int = Field(default=0, ge=0)
    fallback_rows_global: int = Field(default=0, ge=0)
