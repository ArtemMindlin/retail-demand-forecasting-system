from __future__ import annotations

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
