from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    random_seed: int = 42
    run_mode: Literal["backtest", "retrain", "score_daily"] = "backtest"


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source: Literal["fresh_retailnet"] = "fresh_retailnet"
    hf_dataset_id: str = "Dingdong-Inc/FreshRetailNet-50K"
    splits: dict[str, str] = Field(
        default_factory=lambda: {
            "train": "data/train.parquet",
            "eval": "data/eval.parquet",
        }
    )
    local_cache_dir: Path = Path("data/raw/fresh_retailnet")
    processed_panel_path: Path = Path("data/processed/fresh_retailnet_train.parquet")
    use_local_cache: bool = True
    refresh_processed_cache: bool = False
    top_n_series: int | None = Field(default=100, gt=0)
    min_history_days: int = Field(default=70, ge=0)
    max_rows: int | None = Field(default=None, gt=0)
    horizon: int = Field(default=7, gt=0)
    use_eval_as_holdout: bool = False

    @field_validator("use_eval_as_holdout")
    @classmethod
    def validate_holdout_semantics(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "use_eval_as_holdout must remain false until eval split temporal semantics are verified."
            )
        return v

    @model_validator(mode="after")
    def validate_temporal_consistency(self) -> DatasetConfig:
        if self.min_history_days < self.horizon:
            raise ValueError("min_history_days must be at least dataset.horizon.")
        return self


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    drop_negative_sales: bool = True
    fill_missing_values: bool = True
    imputation_strategy: Literal[
        "supervised", "historical_mean", "clipped_scaling", "none"
    ] = "supervised"


class FeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lags: list[int] = Field(default_factory=lambda: [1, 7, 14, 28], min_length=1)
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28], min_length=1)
    include_static_ids: bool = True
    include_weather_lags: bool = True
    include_discount_lags: bool = True
    include_stockout_lags: bool = True

    @field_validator("lags", "rolling_windows")
    @classmethod
    def validate_positive_values(cls, v: list[int]) -> list[int]:
        if any(x <= 0 for x in v):
            raise ValueError("All values must be strictly positive.")
        return v


class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    initial_train_days: int = Field(default=56, gt=0)
    n_folds: int = Field(default=3, gt=0)
    fold_size_days: int = Field(default=7, gt=0)
    retrain_each_fold: bool = True
    drift_triggered_retrain: bool = False


class DriftConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    threshold: float = Field(default=15.0, gt=0)
    delta: float = Field(default=0.005, ge=0)
    min_instances: int = Field(default=2, gt=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    quantiles: list[float] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9], min_length=1
    )
    seasonal_period: int = Field(default=7, gt=0)
    n_estimators: int = Field(default=200, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    max_depth: int = Field(default=6, gt=0)
    use_tuning: bool = False
    tuning_trials: int = Field(default=20, gt=0)
    optimize_for_cost: bool = False

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, v: list[float]) -> list[float]:
        if any(q <= 0.0 or q >= 1.0 for q in v):
            raise ValueError("models.quantiles must be strictly between 0 and 1.")
        if len(set(v)) != len(v):
            raise ValueError("models.quantiles must be unique.")
        if v != sorted(v):
            raise ValueError("models.quantiles must be sorted in ascending order.")
        return v


class SyntheticCostConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    perishability_weights: list[float] = Field(
        default_factory=lambda: [0.5, 0.3, 0.2], min_length=3, max_length=3
    )
    slow_moving_weights: list[float] = Field(
        default_factory=lambda: [0.6, 0.4], min_length=2, max_length=2
    )
    criticality_weights: list[float] = Field(
        default_factory=lambda: [0.7, 0.3], min_length=2, max_length=2
    )
    perishability_base: float = Field(default=0.8, gt=0)
    perishability_multiplier: float = Field(default=0.8, ge=0)
    slow_moving_base: float = Field(default=0.9, gt=0)
    slow_moving_multiplier: float = Field(default=0.5, ge=0)
    service_criticality_base: float = Field(default=0.9, gt=0)
    service_criticality_multiplier: float = Field(default=0.5, ge=0)

    @field_validator(
        "perishability_weights", "slow_moving_weights", "criticality_weights"
    )
    @classmethod
    def validate_weights_sum(cls, v: list[float]) -> list[float]:
        if any(weight < 0.0 for weight in v):
            raise ValueError("Synthetic cost weights must be non-negative.")
        if not (0.99 <= sum(v) <= 1.01):
            raise ValueError("Synthetic cost weights must sum to approximately 1.0.")
        return v


class InventoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    overstock_cost: float = Field(default=1.0, gt=0)
    stockout_cost: float = Field(default=4.0, gt=0)
    use_series_costs: bool = False
    series_cost_strategy: Literal["synthetic_series"] = "synthetic_series"
    clip_negative_orders: bool = True
    pareto_order_scales: list[float] = Field(
        default_factory=lambda: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], min_length=1
    )
    synthetic_cost_config: SyntheticCostConfig = Field(
        default_factory=SyntheticCostConfig
    )

    @field_validator("pareto_order_scales")
    @classmethod
    def validate_scales(cls, v: list[float]) -> list[float]:
        if any(scale < 0.0 for scale in v):
            raise ValueError(
                "inventory.pareto_order_scales must contain only non-negative scales."
            )
        return v


class BusinessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    flag_cold_start: bool = True
    flag_drift_watch: bool = True
    flag_high_uncertainty: bool = True
    high_uncertainty_interval_quantile: float = Field(default=0.95, gt=0.0, lt=1.0)
    flag_extreme_order_quantity: bool = True
    extreme_order_quantity_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)
    champion_data_strategy: str | None = "Observed"
    champion_model_name: str = "catboost"
    champion_backend_name: str = "conformal_catboost_official"
    champion_min_cost_improvement_pct: float = Field(default=0.0, ge=0.0)
    champion_max_service_level_degradation: float = Field(default=0.02, ge=0.0, le=1.0)


class ReportingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    output_dir: Path = Path("reports")
    run_name: str = "fresh_retailnet_v2"
    make_plots: bool = True

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        if v.parts[:1] == ("data",):
            raise ValueError("reporting.output_dir must not write into the data cache.")
        return v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_nested_delimiter="__",
        env_prefix="RETAIL_",
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    business: BusinessConfig = Field(default_factory=BusinessConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @model_validator(mode="after")
    def validate_cross_module_consistency(self) -> Settings:
        if self.validation.initial_train_days < self.dataset.horizon:
            raise ValueError(
                "validation.initial_train_days must be at least dataset.horizon."
            )
        return self


def load_config(path: str | Path) -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    return Settings(**raw_config)
