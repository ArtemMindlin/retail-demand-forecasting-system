"""Shared runtime contracts used across pipeline modules."""

from retail_forecasting.contracts.contracts_backtesting import FoldRunMetadata
from retail_forecasting.contracts.contracts_business import ChampionRecord, ChampionRegistry
from retail_forecasting.contracts.contracts_drift import (
    DriftDetectorMetadata,
    DriftEvent,
    DriftResult,
)
from retail_forecasting.contracts.contracts_quality import (
    DataQualityError,
    DataQualityIssue,
    DataQualityReport,
)
from retail_forecasting.contracts.contracts_tuning import (
    BoostingParams,
    TuningMetadata,
    TuningResult,
)

__all__ = [
    "FoldRunMetadata",
    "BoostingParams",
    "ChampionRecord",
    "ChampionRegistry",
    "DataQualityError",
    "DataQualityIssue",
    "DataQualityReport",
    "DriftDetectorMetadata",
    "DriftEvent",
    "DriftResult",
    "TuningMetadata",
    "TuningResult",
]
