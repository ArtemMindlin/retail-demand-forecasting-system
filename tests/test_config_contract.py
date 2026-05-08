from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from retail_forecasting.config import (
    DatasetConfig,
    InventoryConfig,
    ModelConfig,
    ReportingConfig,
    Settings,
    ValidationConfig,
    load_config,
)

CONFIG_PATH = Path("configs/default.yaml")


def test_default_config_preserves_experimental_guardrails() -> None:
    settings = load_config(CONFIG_PATH)

    assert settings.dataset.source == "fresh_retailnet"
    assert settings.dataset.use_eval_as_holdout is False
    assert settings.dataset.horizon > 0
    assert settings.dataset.min_history_days >= settings.dataset.horizon
    assert settings.validation.initial_train_days >= settings.dataset.horizon
    assert settings.validation.n_folds > 0
    assert settings.validation.fold_size_days > 0


def test_default_model_quantiles_are_valid_and_orderable() -> None:
    settings = load_config(CONFIG_PATH)

    quantiles = settings.models.quantiles

    assert quantiles
    assert all(0.0 < quantile < 1.0 for quantile in quantiles)
    assert sorted(set(quantiles)) == sorted(quantiles)


def test_default_inventory_costs_are_positive() -> None:
    settings = load_config(CONFIG_PATH)

    assert settings.inventory.overstock_cost > 0
    assert settings.inventory.stockout_cost > 0
    assert isinstance(settings.inventory.use_series_costs, bool)
    assert settings.inventory.series_cost_strategy == "synthetic_series"
    assert settings.inventory.pareto_order_scales
    assert all(scale >= 0.0 for scale in settings.inventory.pareto_order_scales)


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


def test_settings_instantiation_rejects_data_cache_reporting_output() -> None:
    with pytest.raises(ValidationError, match="output_dir"):
        Settings(reporting=ReportingConfig(output_dir=Path("data/reports")))


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
