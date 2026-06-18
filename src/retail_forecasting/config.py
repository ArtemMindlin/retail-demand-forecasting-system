from __future__ import annotations

from pathlib import Path

import yaml

from retail_forecasting.contracts.contracts_config import (
    BusinessConfig,
    DataQualityConfig,
    DatasetConfig,
    DriftConfig,
    FeatureConfig,
    InventoryConfig,
    ModelConfig,
    PreprocessingConfig,
    ProjectConfig,
    ReportingConfig,
    Settings,
    SimulationConfig,
    ValidationConfig,
)

__all__ = [
    "BusinessConfig",
    "DataQualityConfig",
    "DatasetConfig",
    "DriftConfig",
    "FeatureConfig",
    "InventoryConfig",
    "ModelConfig",
    "PreprocessingConfig",
    "ProjectConfig",
    "ReportingConfig",
    "Settings",
    "SimulationConfig",
    "ValidationConfig",
    "load_config",
]


def load_config(path: str | Path) -> Settings:
    """Load and validate the project settings from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated Settings object populated with the YAML values and environment overrides.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    return Settings(**raw_config)
