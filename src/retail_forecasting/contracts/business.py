from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ChampionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_strategy: str | None
    model_name: str
    backend_name: str
    promoted_at: str
    run_name: str
    git_commit: str | None
    config_hash: str
    reason: str
    model_path: str | None = None


class ChampionRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    updated_at: str
    current_champion: ChampionRecord
