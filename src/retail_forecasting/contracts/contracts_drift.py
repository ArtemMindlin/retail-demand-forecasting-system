from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DriftResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_drift: bool
    score: float = Field(ge=0)
    threshold: float = Field(gt=0)
    detected_at_index: int | None = Field(default=None, ge=0)


class DriftEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    date: str
    score: float = Field(ge=0)
    threshold: float = Field(gt=0)
    fold_id: int = Field(ge=0)


class DriftDetectorMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    detector_name: str
    threshold: float = Field(gt=0)
    delta: float = Field(ge=0)
    min_instances: int = Field(gt=0)
    monitored_metric: str
    observations_seen: int = Field(ge=0)
    alerts_detected: int = Field(ge=0)
    max_score: float = Field(ge=0)
    last_score: float = Field(ge=0)
