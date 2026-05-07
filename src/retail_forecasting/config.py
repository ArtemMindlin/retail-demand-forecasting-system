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
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    seasonal_period: int = 7
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6
    use_tuning: bool = False
    tuning_trials: int = 20
    optimize_for_cost: bool = False


@dataclass
class InventoryConfig:
    overstock_cost: float = 1.0
    stockout_cost: float = 4.0
    use_series_costs: bool = False
    series_cost_strategy: str = "synthetic_series"
    clip_negative_orders: bool = True
    pareto_order_scales: list[float] = field(
        default_factory=lambda: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    )


@dataclass
class ReportingConfig:
    output_dir: Path = Path("reports")
    run_name: str = "fresh_retailnet_v2"
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


def validate_settings(settings: Settings) -> None:
    """Validate semantic experiment guardrails after loading configuration."""
    errors: list[str] = []

    if settings.dataset.source != "fresh_retailnet":
        errors.append("dataset.source must be 'fresh_retailnet'.")

    if settings.dataset.use_eval_as_holdout:
        errors.append(
            "dataset.use_eval_as_holdout must remain false until eval split temporal semantics are verified."
        )

    if settings.dataset.horizon <= 0:
        errors.append("dataset.horizon must be greater than 0.")

    if settings.dataset.min_history_days < settings.dataset.horizon:
        errors.append("dataset.min_history_days must be at least dataset.horizon.")

    if settings.dataset.top_n_series is not None and settings.dataset.top_n_series <= 0:
        errors.append("dataset.top_n_series must be greater than 0 when set.")

    if settings.dataset.max_rows is not None and settings.dataset.max_rows <= 0:
        errors.append("dataset.max_rows must be greater than 0 when set.")

    if settings.validation.initial_train_days < settings.dataset.horizon:
        errors.append("validation.initial_train_days must be at least dataset.horizon.")

    if settings.validation.n_folds <= 0:
        errors.append("validation.n_folds must be greater than 0.")

    if settings.validation.fold_size_days <= 0:
        errors.append("validation.fold_size_days must be greater than 0.")

    if not settings.features.lags:
        errors.append("features.lags must not be empty.")

    if any(lag <= 0 for lag in settings.features.lags):
        errors.append("features.lags must contain only positive lags.")

    if not settings.features.rolling_windows:
        errors.append("features.rolling_windows must not be empty.")

    if any(window <= 0 for window in settings.features.rolling_windows):
        errors.append("features.rolling_windows must contain only positive windows.")

    quantiles = settings.models.quantiles
    if not quantiles:
        errors.append("models.quantiles must not be empty.")
    else:
        if any(quantile <= 0.0 or quantile >= 1.0 for quantile in quantiles):
            errors.append("models.quantiles must be strictly between 0 and 1.")
        if len(set(quantiles)) != len(quantiles) or quantiles != sorted(quantiles):
            errors.append(
                "models.quantiles must be unique and sorted in ascending order."
            )

    if settings.models.seasonal_period <= 0:
        errors.append("models.seasonal_period must be greater than 0.")

    if settings.models.n_estimators <= 0:
        errors.append("models.n_estimators must be greater than 0.")

    if settings.models.learning_rate <= 0.0:
        errors.append("models.learning_rate must be greater than 0.")

    if settings.models.max_depth <= 0:
        errors.append("models.max_depth must be greater than 0.")

    if settings.models.tuning_trials <= 0:
        errors.append("models.tuning_trials must be greater than 0.")

    if settings.inventory.overstock_cost <= 0.0:
        errors.append("inventory.overstock_cost must be greater than 0.")

    if settings.inventory.stockout_cost <= 0.0:
        errors.append("inventory.stockout_cost must be greater than 0.")

    if settings.inventory.series_cost_strategy not in {"synthetic_series"}:
        errors.append(
            "inventory.series_cost_strategy must be one of: synthetic_series."
        )

    if not settings.inventory.pareto_order_scales:
        errors.append("inventory.pareto_order_scales must not be empty.")

    if any(scale < 0.0 for scale in settings.inventory.pareto_order_scales):
        errors.append(
            "inventory.pareto_order_scales must contain only non-negative scales."
        )

    if settings.reporting.output_dir.parts[:1] == ("data",):
        errors.append("reporting.output_dir must not write into the data cache.")

    if errors:
        formatted_errors = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid configuration:\n{formatted_errors}")


def load_config(path: str | Path) -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    project = ProjectConfig(**raw_config.get("project", {}))

    dataset = DatasetConfig(**raw_config.get("dataset", {}))
    dataset.local_cache_dir = Path(dataset.local_cache_dir)
    dataset.processed_panel_path = Path(dataset.processed_panel_path)

    preprocessing = PreprocessingConfig(**raw_config.get("preprocessing", {}))

    features = FeatureConfig(**raw_config.get("features", {}))

    validation = ValidationConfig(**raw_config.get("validation", {}))

    models = ModelConfig(**raw_config.get("models", {}))

    inventory = InventoryConfig(**raw_config.get("inventory", {}))

    reporting = ReportingConfig(**raw_config.get("reporting", {}))
    reporting.output_dir = Path(reporting.output_dir)

    settings = Settings(
        project=project,
        dataset=dataset,
        preprocessing=preprocessing,
        features=features,
        validation=validation,
        models=models,
        inventory=inventory,
        reporting=reporting,
    )
    validate_settings(settings)
    return settings


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
