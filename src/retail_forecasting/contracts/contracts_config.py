from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for the execution modes; reused by the CLI parser
# (run.py) and the run metadata schema (evaluation.reporting).
RunMode = Literal["experiment", "retrain", "score_daily", "simulate_ops", "fair_cost_backtest"]


class ProjectConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    random_seed: int = 42
    run_mode: RunMode = "experiment"


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    hf_dataset_id: str = "Dingdong-Inc/FreshRetailNet-50K"
    splits: dict[str, str] = Field(
        default_factory=lambda: {
            "train": "data/train.parquet",
            "eval": "data/eval.parquet",
        }
    )
    local_cache_dir: Path = Path("data/raw/fresh_retailnet")
    processed_panel_dir: Path = Path("data/processed")
    use_cache: bool = True
    top_n_series: int | None = Field(default=100, gt=0)
    min_history_days: int = Field(default=70, ge=0)
    max_rows: int | None = Field(default=None, gt=0)
    horizon: int = Field(default=7, gt=0)

    @model_validator(mode="after")
    def validate_temporal_consistency(self) -> DatasetConfig:
        if self.min_history_days < self.horizon:
            raise ValueError("min_history_days must be at least dataset.horizon.")
        return self


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    drop_negative_sales: bool = True
    fill_missing_values: bool = True
    imputation_strategy: Literal["supervised", "historical_mean", "clipped_scaling", "none"] = (
        "supervised"
    )
    # When True, the experiment run_mode skips forecasting and runs only the latent-demand
    # imputation strategies side by side, writing a lightweight comparison artifact for the
    # dashboard (no models, no folds).
    compare_imputation: bool = False


class FeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lags: list[int] = Field(default_factory=lambda: [1, 7, 14, 28], min_length=1)
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28], min_length=1)
    include_static_ids: bool = True
    include_weather_lags: bool = True
    include_discount_lags: bool = True
    include_stockout_lags: bool = True


class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    initial_train_days: int = Field(default=56, gt=0)
    n_folds: int = Field(default=3, gt=0)
    fold_size_days: int = Field(default=7, gt=0)
    calibration_days: int = Field(default=21, gt=0)
    retrain_each_fold: bool = True
    drift_triggered_retrain: bool = False


class DriftConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    threshold: float = Field(default=15.0, gt=0)
    delta: float = Field(default=0.005, ge=0)
    min_instances: int = Field(default=2, gt=0)


class DataQualityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_missing_fraction_warning: float = Field(default=0.05, ge=0.0, le=1.0)
    max_data_age_days: int | None = Field(default=None, ge=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    models_dir: Path = Field(default=Path("models"))
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9], min_length=1)
    seasonal_period: int = Field(default=7, gt=0)
    n_estimators: int = Field(default=200, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    max_depth: int = Field(default=6, gt=0)
    use_tuning: bool = True
    tuning_trials: int = Field(default=20, gt=0)

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


class InventoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    overstock_cost: float = Field(default=1.0, gt=0)
    stockout_cost: float = Field(default=4.0, gt=0)
    use_series_costs: bool = False
    clip_negative_orders: bool = True
    global_capacity_units: int | None = Field(default=None, gt=0)


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


class SimulationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    retrain_cadences: list[int | None] = Field(
        default_factory=lambda: [None, 7, 1],
        min_length=1,
        description="Days between retrains per cadence; None means never retrain (baseline).",
    )
    simulation_days: int | None = Field(default=None, gt=0)
    make_plots: bool = True

    @field_validator("retrain_cadences")
    @classmethod
    def validate_cadences(cls, v: list[int | None]) -> list[int | None]:
        if any(item is not None and item <= 0 for item in v):
            raise ValueError("simulation.retrain_cadences must contain positive ints or None.")
        seen: set[int | None] = set()
        for item in v:
            if item in seen:
                raise ValueError("simulation.retrain_cadences must be unique.")
            seen.add(item)
        return v


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
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    business: BusinessConfig = Field(default_factory=BusinessConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    @model_validator(mode="after")
    def validate_cross_module_consistency(self) -> Settings:
        if self.validation.initial_train_days < self.dataset.horizon:
            raise ValueError("validation.initial_train_days must be at least dataset.horizon.")
        return self
