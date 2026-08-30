from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for the execution modes; reused by the CLI parser
# (run.py) and the run metadata schema (evaluation.reporting).
RunMode = Literal[
    "experiment",
    "retrain",
    "score_daily",
    "simulate_ops",
    "fair_cost_backtest",
    "tune_imputation",
    "tune_forecasting",
    "eda",
]

# Which `Settings` sections each run mode actually reads, traced from each entry point by
# transitive closure over the call graph. A config under `configs/<mode>/` may declare a
# SUBSET of these (every section has a default) but never a section outside the set: a
# declared section the mode never reads is a knob that silently does nothing, which is
# worse than a missing one. Pinned by `tests/test_config_layout.py`.
MODE_SECTIONS: dict[RunMode, frozenset[str]] = {
    "experiment": frozenset(
        {
            "project",
            "dataset",
            "preprocessing",
            "features",
            "validation",
            "drift",
            "data_quality",
            "models",
            "inventory",
            "business",
            "reporting",
        }
    ),
    "retrain": frozenset(
        {
            "project",
            "dataset",
            "preprocessing",
            "features",
            "validation",
            "data_quality",
            "models",
            "inventory",
            "business",
        }
    ),
    "score_daily": frozenset(
        {
            "project",
            "dataset",
            "preprocessing",
            "features",
            "data_quality",
            "models",
            "inventory",
            "business",
            "reporting",
        }
    ),
    "simulate_ops": frozenset(
        {
            "project",
            "dataset",
            "preprocessing",
            "features",
            "validation",
            "data_quality",
            "models",
            "inventory",
            "business",
            "reporting",
            "simulation",
        }
    ),
    "fair_cost_backtest": frozenset(
        {"project", "dataset", "preprocessing", "models", "inventory", "reporting"}
    ),
    "tune_imputation": frozenset({"project", "dataset", "preprocessing", "models"}),
    "tune_forecasting": frozenset(
        {"project", "dataset", "preprocessing", "features", "validation", "models", "inventory"}
    ),
    "eda": frozenset({"project", "dataset", "preprocessing", "reporting"}),
}


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


# Single source of truth for the latent-demand imputation strategies. It lives in `contracts`
# because both consumers can reach it from here: `PreprocessingConfig` below validates the
# configured value, and `data/censorship.py` types the imputer against it. The reverse is
# impossible -- `contracts` imports no first-party layer -- which is why this was previously
# declared twice, once in each place, with nothing keeping the two in step.
ImputationStrategy = Literal["supervised", "historical_mean", "clipped_scaling", "none"]

# Single source of truth for the boosting backends a tuning run can target. It lives in
# `contracts` so `forecasting_tuning` can type against it without importing `models`.
BoostingBackend = Literal["lightgbm", "catboost"]


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    drop_negative_sales: bool = True
    fill_missing_values: bool = True
    imputation_strategy: ImputationStrategy = "supervised"


class FeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lags: list[int] = Field(default_factory=lambda: [1, 7, 14, 28], min_length=1)
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28], min_length=1)
    include_weather_lags: bool = True
    include_discount_lags: bool = True
    include_stockout_lags: bool = True

    @field_validator("lags", "rolling_windows")
    @classmethod
    def validate_positive_unique_sorted(cls, v: list[int], info: ValidationInfo) -> list[int]:
        """Let feature engineering consume these as declared, without re-normalising."""
        field = f"features.{info.field_name}"
        if any(item <= 0 for item in v):
            raise ValueError(f"{field} must be strictly positive: a zero lag or window leaks.")
        if len(set(v)) != len(v):
            raise ValueError(f"{field} must be unique.")
        if v != sorted(v):
            raise ValueError(f"{field} must be sorted in ascending order.")
        return v


class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    initial_train_days: int = Field(default=56, gt=0)
    n_folds: int = Field(default=3, gt=0)
    fold_size_days: int = Field(default=7, gt=0)
    calibration_days: int = Field(default=21, gt=0)
    retrain_each_fold: bool = True
    drift_triggered_retrain: bool = False

    def minimum_dates_required(self, horizon: int) -> int:
        """Unique dates a panel needs for this fold layout to fit.

        `horizon` is a parameter rather than a field because it lives in `DatasetConfig`,
        and `build_walk_forward_folds` receives the two separately.
        """
        return self.initial_train_days + self.n_folds * self.fold_size_days + horizon - 1


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
    # Where `tune_imputation` persists its winner and the three modes that use the supervised
    # imputer read it from. Pinned by `SHARED_FIELDS`: writer and readers each load their own
    # config, and a value that differed between them would send the readers to a file that is
    # not there, which degrades to the untuned defaults without an error.
    imputation_params_filename: str = "imputation_lgbm_params.json"
    # Written by `tune_forecasting` and read by every mode that trains a boosting model, so it
    # is the same kind of shared name as the line above, and drifts the same way.
    forecasting_params_filename: str = "forecasting_params.json"
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9], min_length=1)
    seasonal_period: int = Field(default=7, gt=0)
    n_estimators: int = Field(default=200, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    max_depth: int = Field(default=6, gt=0)
    use_tuning: bool = True
    tuning_trials: int = Field(default=20, gt=0)
    # Which backend `run_mode = tune_forecasting` searches. ONE per run: two backends mean
    # two independent searches, two gates and two verdicts, so they are two executions.
    tuning_backend: BoostingBackend = "catboost"

    @field_validator("imputation_params_filename", "forecasting_params_filename")
    @classmethod
    def validate_bare_params_filename(cls, v: str, info: ValidationInfo) -> str:
        if Path(v).name != v:
            raise ValueError(f"models.{info.field_name} must be a bare file name.")
        return v

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
    clip_negative_orders: bool = True


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
    # How many points of relative COST improvement one point of relative WINKLER improvement is
    # worth when promoting. A declared business preference, like the c_u/c_o pair: 0.0 ranks on
    # cost alone (the behaviour every figure in the thesis was produced under), higher values buy
    # interval quality with cost. Both terms are relative improvements over the incumbent, so the
    # weight is unit-free and reads directly as an exchange rate.
    champion_winkler_weight: float = Field(default=0.0, ge=0.0)


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
    run_name: str = "fresh_retailnet_v2"
    make_plots: bool = True


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
