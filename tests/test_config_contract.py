from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from retail_forecasting.config import (
    BusinessConfig,
    DataQualityConfig,
    DatasetConfig,
    InventoryConfig,
    ModelConfig,
    PreprocessingConfig,
    ReportingConfig,
    Settings,
    SyntheticCostConfig,
    ValidationConfig,
    load_config,
)

CONFIG_PATH = Path("configs/default.yaml")


def test_default_config_preserves_experimental_guardrails() -> None:
    settings = load_config(CONFIG_PATH)

    assert settings.dataset.source == "fresh_retailnet"
    assert settings.project.run_mode in {"backtest", "retrain", "score_daily"}
    assert settings.dataset.use_eval_as_holdout is False
    assert settings.dataset.horizon > 0
    assert settings.dataset.min_history_days >= settings.dataset.horizon
    assert settings.validation.initial_train_days >= settings.dataset.horizon
    assert settings.validation.n_folds > 0
    assert settings.validation.fold_size_days > 0
    assert settings.drift.threshold > 0
    assert settings.drift.delta >= 0
    assert settings.drift.min_instances > 0
    assert 0.0 <= settings.data_quality.max_missing_fraction_warning <= 1.0
    assert 0.0 < settings.business.high_uncertainty_interval_quantile < 1.0
    assert 0.0 < settings.business.extreme_order_quantity_quantile < 1.0
    assert settings.business.champion_model_name
    assert settings.business.champion_backend_name
    assert settings.business.champion_min_cost_improvement_pct >= 0.0
    assert 0.0 <= settings.business.champion_max_service_level_degradation <= 1.0


def test_default_model_quantiles_are_valid_and_orderable() -> None:
    settings = load_config(CONFIG_PATH)

    quantiles = settings.models.quantiles

    assert quantiles
    assert all(0.0 < quantile < 1.0 for quantile in quantiles)
    assert sorted(set(quantiles)) == sorted(quantiles)


def test_default_inventory_costs_are_positive() -> None:
    settings = load_config(CONFIG_PATH)
    synthetic_cost_config = settings.inventory.synthetic_cost_config

    assert settings.inventory.overstock_cost > 0
    assert settings.inventory.stockout_cost > 0
    assert isinstance(settings.inventory.use_series_costs, bool)
    assert settings.inventory.series_cost_strategy == "synthetic_series"
    assert settings.inventory.pareto_order_scales
    assert all(scale >= 0.0 for scale in settings.inventory.pareto_order_scales)
    assert sum(synthetic_cost_config.perishability_weights) == pytest.approx(1.0)
    assert sum(synthetic_cost_config.slow_moving_weights) == pytest.approx(1.0)
    assert sum(synthetic_cost_config.criticality_weights) == pytest.approx(1.0)
    assert synthetic_cost_config.perishability_base > 0.0
    assert synthetic_cost_config.slow_moving_base > 0.0
    assert synthetic_cost_config.service_criticality_base > 0.0


def test_default_reporting_does_not_write_into_data_cache() -> None:
    settings = load_config(CONFIG_PATH)

    output_dir = settings.reporting.output_dir
    assert output_dir.parts[0] != "data"


def test_settings_instantiation_rejects_unsupported_eval_holdout() -> None:
    with pytest.raises(ValidationError, match="use_eval_as_holdout"):
        Settings(dataset=DatasetConfig(use_eval_as_holdout=True))


def test_settings_instantiation_rejects_invalid_temporal_guardrails() -> None:
    with pytest.raises(ValidationError, match="min_history_days"):
        Settings(
            dataset=DatasetConfig(horizon=7, min_history_days=6),
            validation=ValidationConfig(initial_train_days=6),
        )


def test_settings_instantiation_rejects_invalid_quantiles() -> None:
    with pytest.raises(ValidationError, match="quantiles"):
        Settings(models=ModelConfig(quantiles=[0.5, 0.1, 0.5]))


def test_settings_instantiation_rejects_unsorted_quantiles() -> None:
    with pytest.raises(ValidationError, match="ascending order"):
        Settings(models=ModelConfig(quantiles=[0.5, 0.1, 0.9]))


def test_settings_instantiation_rejects_invalid_inventory_costs() -> None:
    with pytest.raises(ValidationError, match="overstock_cost"):
        Settings(inventory=InventoryConfig(overstock_cost=0.0))


def test_settings_instantiation_rejects_invalid_business_thresholds() -> None:
    with pytest.raises(ValidationError, match="high_uncertainty_interval_quantile"):
        Settings(business=BusinessConfig(high_uncertainty_interval_quantile=1.0))

    with pytest.raises(ValidationError, match="extreme_order_quantity_quantile"):
        Settings(business=BusinessConfig(extreme_order_quantity_quantile=0.0))

    with pytest.raises(ValidationError, match="champion_min_cost_improvement_pct"):
        Settings(business=BusinessConfig(champion_min_cost_improvement_pct=-0.1))

    with pytest.raises(ValidationError, match="champion_max_service_level_degradation"):
        Settings(business=BusinessConfig(champion_max_service_level_degradation=1.1))

    with pytest.raises(ValidationError, match="max_missing_fraction_warning"):
        Settings(data_quality=DataQualityConfig(max_missing_fraction_warning=1.1))


def test_settings_instantiation_rejects_invalid_synthetic_cost_weights() -> None:
    with pytest.raises(ValidationError, match="Synthetic cost weights"):
        Settings(
            inventory=InventoryConfig(
                synthetic_cost_config=SyntheticCostConfig(
                    perishability_weights=[0.8, 0.3, -0.1]
                )
            )
        )

    with pytest.raises(ValidationError, match="Synthetic cost weights"):
        Settings(
            inventory=InventoryConfig(
                synthetic_cost_config=SyntheticCostConfig(
                    criticality_weights=[0.7, 0.4]
                )
            )
        )


def test_settings_instantiation_rejects_invalid_synthetic_cost_scaling() -> None:
    with pytest.raises(ValidationError, match="perishability_base"):
        Settings(
            inventory=InventoryConfig(
                synthetic_cost_config=SyntheticCostConfig(perishability_base=0.0)
            )
        )


def test_settings_instantiation_rejects_data_cache_reporting_output() -> None:
    with pytest.raises(ValidationError, match="output_dir"):
        Settings(reporting=ReportingConfig(output_dir=Path("data/reports")))


def test_settings_instantiation_rejects_invalid_imputation_strategy() -> None:
    with pytest.raises(ValidationError, match="imputation_strategy"):
        Settings(preprocessing=PreprocessingConfig(imputation_strategy="random_forest"))


def test_config_is_immutable() -> None:
    settings = load_config(CONFIG_PATH)
    with pytest.raises(ValidationError):
        # Pydantic raises ValidationError (specifically frozen error) on assignment
        settings.project.random_seed = 100


def test_environment_variable_overrides() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "RETAIL_DATASET__HORIZON": "99",
            "RETAIL_DATASET__MIN_HISTORY_DAYS": "100",
            "RETAIL_VALIDATION__INITIAL_TRAIN_DAYS": "100",
        },
    ):
        settings = Settings()
        assert settings.dataset.horizon == 99
        assert settings.dataset.min_history_days == 100
        assert settings.validation.initial_train_days == 100


def test_validation_cross_module_consistency() -> None:
    # Test validation in Settings model_validator
    with pytest.raises(ValidationError, match="initial_train_days"):
        Settings(
            dataset=DatasetConfig(horizon=14, min_history_days=14),
            validation=ValidationConfig(initial_train_days=7),
        )
