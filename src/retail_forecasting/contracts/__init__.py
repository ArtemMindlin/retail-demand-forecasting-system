"""Shared runtime contracts used across pipeline modules."""

from retail_forecasting.contracts.backtesting import FoldRunMetadata
from retail_forecasting.contracts.business import ChampionRecord, ChampionRegistry
from retail_forecasting.contracts.drift import (
    DriftDetectorMetadata,
    DriftEvent,
    DriftResult,
)
from retail_forecasting.contracts.tuning import (
    BoostingParams,
    TuningMetadata,
    TuningResult,
)

__all__ = [
    "FoldRunMetadata",
    "BoostingParams",
    "ChampionRecord",
    "ChampionRegistry",
    "DriftDetectorMetadata",
    "DriftEvent",
    "DriftResult",
    "TuningMetadata",
    "TuningResult",
]
