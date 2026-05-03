from __future__ import annotations

from pathlib import Path

from retail_forecasting.config import load_config


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
    assert settings.inventory.series_cost_strategy in {"synthetic_series"}
    assert settings.inventory.pareto_order_scales
    assert all(scale >= 0.0 for scale in settings.inventory.pareto_order_scales)


def test_default_reporting_does_not_write_into_data_cache() -> None:
    settings = load_config(CONFIG_PATH)

    output_dir = settings.reporting.output_dir
    assert output_dir.parts[0] != "data"
