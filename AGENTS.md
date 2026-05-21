# AGENTS.md

This is a research/prototype Python project for retail demand forecasting and inventory decisions under uncertainty, stockouts, and temporal drift.

The priority of this repo is experimental validity: avoid temporal leakage, preserve dataframe contracts, and keep forecasting, inventory decisions, and evaluation separated.

## Repo Map

- `configs/`: experiment configuration. `configs/default.yaml` is the canonical v1 config.
- `data/`: local raw/interim/processed caches. Do not commit generated datasets.
- `docs/`: system of record for architecture, contracts, invariants, and decisions.
- `notebooks/`: lightweight exploration only. Production pipeline logic belongs in `src/`.
- `reports/`: generated experiment outputs. Do not edit manually unless documenting a final result.
- `src/retail_forecasting/config.py`: typed settings loaded from YAML.
- `src/retail_forecasting/data/`: raw dataset loading and raw-to-panel preparation.
- `src/retail_forecasting/features/`: supervised frame creation, temporal features, and target construction.
- `src/retail_forecasting/forecasting/`: walk-forward validation and experiment orchestration.
- `src/retail_forecasting/models/`: forecast models only.
- `src/retail_forecasting/inventory/`: newsvendor order quantity and cost logic.
- `src/retail_forecasting/evaluation/`: metrics and report generation.
- `src/retail_forecasting/drift/`: regime/drift analysis hooks.
- `tests/`: contract tests, smoke tests, and synthetic-panel tests.

## Current Scope

The v1 pipeline supports only `FreshRetailNet-50K` through `dataset.source = fresh_retailnet`.

The v1 target is observed demand, not latent/censored demand. Stockout information is currently used as context and regime analysis, not to reconstruct lost demand.

The official `eval` split is intentionally not wired as a holdout until its temporal semantics are verified.

## Core Pipeline

```text
run.py
 -> load_config()
 -> run_experiment()
 -> load_prepared_panel()
 -> label_stockout_regime()
 -> build_supervised_frame()
 -> build_walk_forward_folds()
 -> SeasonalNaiveModel / AutoBoostingModel
 -> choose_order_quantity()
 -> attach_inventory_costs()
 -> summarize_predictions() / summarize_costs()
 -> write_run_artifacts()
```

## Hard Rules

- Raw dataset names such as `dt`, `sale_amount`, and `stock_hour6_22_cnt` must not leak beyond `data/dataset.py`.
- The canonical prepared panel uses `date`, `series_id`, `observed_demand`, and `stockout_hours`.
- `target_lead_time_demand` is built only in `features/engineering.py`.
- Temporal features must use only information available at the forecast origin.
- Lagged non-ex-ante variables such as weather, discount, and stockout must use positive lags.
- Walk-forward training rows must end at least `horizon` days before validation starts.
- Models must not compute inventory costs.
- Inventory code must not train models or build features.
- Evaluation code must summarize predictions and costs, not change forecasts or decisions.
- Quantile columns must use `quantile_column_name()`.

## Before Changing Code

Read these first:

- `docs/invariants.md`
- `docs/contracts/dataframes.md`
- `docs/conventions.md`
- `docs/experimental_design.md`
- `docs/system_design.md`

Prefer small changes that preserve the pipeline contract. If a change modifies target semantics, fold semantics, dataframe schemas, or inventory policy, update docs and tests in the same change.

For large changes, inspect the unstaged worktree and propose a split into two or more commits before committing. Ask for confirmation after each large project change before staging or committing.

## Commands

Install dependencies:

```bash
uv sync --extra dev
```

Install optional ML backends:

```bash
uv sync --extra dev --extra ml
```

Run tests:

```bash
uv run pytest
```

Run fast harness checks:

```bash
uv run pytest tests/test_architecture_imports.py tests/test_temporal_leakage_contract.py tests/test_quantile_contract.py tests/test_dataframe_contracts.py tests/test_raw_column_boundaries.py tests/test_config_contract.py tests/test_generated_artifact_boundaries.py
```

Run the default experiment:

```bash
uv run python -m retail_forecasting.run --config configs/default.yaml
```
