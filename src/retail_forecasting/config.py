from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    random_seed: int = 42


@dataclass
class DatasetConfig:
    source: str = "fresh_retailnet"
    hf_dataset_id: str = "Dingdong-Inc/FreshRetailNet-50K"
    splits: dict[str, str] = field(
        default_factory=lambda: {
            "train": "data/train.parquet",
            "eval": "data/eval.parquet",
        }
    )
    local_cache_dir: Path = Path("data/raw/fresh_retailnet")
    processed_panel_path: Path = Path("data/processed/fresh_retailnet_train.parquet")
    use_local_cache: bool = True
    refresh_processed_cache: bool = False
    top_n_series: int = 100
    min_history_days: int = 70
    max_rows: int | None = None
    horizon: int = 7
    use_eval_as_holdout: bool = False


@dataclass
class PreprocessingConfig:
    drop_negative_sales: bool = True
    fill_missing_values: bool = True
    imputation_strategy: str = "supervised"


@dataclass
class FeatureConfig:
    lags: list[int] = field(default_factory=lambda: [1, 7, 14, 28])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 28])
    include_static_ids: bool = True
    include_weather_lags: bool = True
    include_discount_lags: bool = True
    include_stockout_lags: bool = True


@dataclass
class ValidationConfig:
    initial_train_days: int = 56
    n_folds: int = 3
    fold_size_days: int = 7
    retrain_each_fold: bool = True
    drift_triggered_retrain: bool = False


@dataclass
class ModelConfig:
    point_model: str = "auto_boosting"
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    seasonal_period: int = 7
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6
    use_tuning: bool = False
    tuning_trials: int = 20


@dataclass
class InventoryConfig:
    overstock_cost: float = 1.0
    stockout_cost: float = 4.0
    clip_negative_orders: bool = True


@dataclass
class ReportingConfig:
    output_dir: Path = Path("reports")
    run_name: str = "fresh_retailnet_v1"
    make_plots: bool = True


@dataclass
class Settings:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def _with_path(section: dict[str, Any], key: str, default: Path) -> Path:
    value = section.get(key, default)
    return value if isinstance(value, Path) else Path(value)


def load_config(path: str | Path) -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    default_dataset = DatasetConfig()
    default_reporting = ReportingConfig()
    project = ProjectConfig(**raw_config.get("project", {}))

    dataset_section = raw_config.get("dataset", {})
    dataset = DatasetConfig(
        source=dataset_section.get("source", default_dataset.source),
        hf_dataset_id=dataset_section.get("hf_dataset_id", default_dataset.hf_dataset_id),
        splits=dataset_section.get("splits", default_dataset.splits),
        local_cache_dir=_with_path(
            dataset_section,
            "local_cache_dir",
            default_dataset.local_cache_dir,
        ),
        processed_panel_path=_with_path(
            dataset_section,
            "processed_panel_path",
            default_dataset.processed_panel_path,
        ),
        use_local_cache=dataset_section.get("use_local_cache", default_dataset.use_local_cache),
        refresh_processed_cache=dataset_section.get(
            "refresh_processed_cache",
            default_dataset.refresh_processed_cache,
        ),
        top_n_series=dataset_section.get("top_n_series", default_dataset.top_n_series),
        min_history_days=dataset_section.get(
            "min_history_days",
            default_dataset.min_history_days,
        ),
        max_rows=dataset_section.get("max_rows", default_dataset.max_rows),
        horizon=dataset_section.get("horizon", default_dataset.horizon),
        use_eval_as_holdout=dataset_section.get(
            "use_eval_as_holdout",
            default_dataset.use_eval_as_holdout,
        ),
    )

    preprocessing = PreprocessingConfig(**raw_config.get("preprocessing", {}))
    features = FeatureConfig(**raw_config.get("features", {}))
    validation = ValidationConfig(**raw_config.get("validation", {}))
    models = ModelConfig(**raw_config.get("models", {}))
    inventory = InventoryConfig(**raw_config.get("inventory", {}))

    reporting_section = raw_config.get("reporting", {})
    reporting = ReportingConfig(
        output_dir=_with_path(reporting_section, "output_dir", default_reporting.output_dir),
        run_name=reporting_section.get("run_name", default_reporting.run_name),
        make_plots=reporting_section.get("make_plots", default_reporting.make_plots),
    )

    return Settings(
        project=project,
        dataset=dataset,
        preprocessing=preprocessing,
        features=features,
        validation=validation,
        models=models,
        inventory=inventory,
        reporting=reporting,
    )


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(asdict(settings))
