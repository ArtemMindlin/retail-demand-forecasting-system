# CLAUDE.md

This is a research/prototype Python project for retail demand forecasting and inventory decisions under uncertainty, stockouts, and temporal drift.

The priority of this repo is experimental validity: avoid temporal leakage, preserve dataframe contracts, and keep forecasting, inventory decisions, and evaluation separated.

## Repo Map

- `configs/`: experiment configuration. `configs/experiment.yaml` is the canonical v1 config; `configs/experiment_large.yaml`/`experiment_daily.yaml` cover the scale/daily variants, `configs/imputation_compare.yaml` the imputation study, `configs/simulation.yaml` the OPS-plane rolling-origin backtest.
- `data/`: local raw/interim/processed caches. Do not commit generated datasets.
- `docs/`: system of record for architecture, contracts, invariants, and decisions.
- `notebooks/`: lightweight exploration only. Production pipeline logic belongs in `src/`.
- `reports/`: generated experiment outputs. Do not edit manually unless documenting a final result.
- `scripts/build_ops_sim_split.py`: carves the dedicated train/eval split the OPS plane streams, into `data/processed/ops_sim/`. Invoked by `make simulate` when the split is missing; not something to run by hand.
- `manage.py`: Django management entrypoint for the dashboard (`src/retail_forecasting/api/`).
- `src/retail_forecasting/config.py`: typed settings loaded from YAML.
- `src/retail_forecasting/data/`: raw dataset loading, raw-to-panel preparation, and `censorship.py` (`LatentDemandImputer` — stockout/censored-demand reconstruction strategies).
- `src/retail_forecasting/features/`: supervised frame creation, temporal features, and target construction.
- `src/retail_forecasting/forecasting/`: walk-forward validation, conformal calibration, imputation comparison, fair-cost backtesting, and experiment/retrain/scoring orchestration (`pipeline.py`); `imputation_tuning.py` runs a separate Optuna search over the supervised imputer's LGBM hyperparameters and persists the winner (never fitted weights) to `models.models_dir/imputation_lgbm_params.json`. It is the only entry point that logs to MLflow (`mlflow.db`, browsable via `make mlflow-ui`); every other run mode still writes its artifacts under `reports/`, which the dashboard and `utils/latex_exporter.py` read.
- `src/retail_forecasting/models/`: forecast models only (`naive.py`, `boosting.py` for LightGBM, `catboosting.py` for CatBoost — the current champion).
- `src/retail_forecasting/inventory/`: newsvendor order quantity, cost profiles, optimization, and dynamic simulation logic.
- `src/retail_forecasting/simulation/`: OPS-plane rolling-origin production backtest, reused by the dashboard's `/ops/` view. Independent single-period Newsvendor decisions — no inventory state, no lead time — so its costs rank policies rather than reproduce a replenishment ledger.
- `src/retail_forecasting/evaluation/`: metrics, run reporting, post-mortem analysis, and XAI/explainability.
- `src/retail_forecasting/drift/`: regime/drift analysis hooks.
- `src/retail_forecasting/eda/`: exploratory analysis and figure generation, surfaced in the dashboard's `/eda/` tab.
- `src/retail_forecasting/contracts/`: pydantic-style contracts for backtesting, business rules, config, drift, feature engineering, quality, and tuning — enforced by the `tests/test_*_contract.py` / `tests/test_*_boundaries.py` suite.
- `src/retail_forecasting/api/`: Django dashboard and JSON API (`views/`, `templates/`, `static/`, `services/`) — visualizes experiments, EDA, drift, latent-demand imputation, the OPS plane, and exposes a documented JSON surface (`/api/forecast`, `/api/skus`, `/predict_orders`, etc.). Served via `asgi.py`/`wsgi.py`.
- `tests/`: contract tests, smoke tests, synthetic-panel tests, and API tests.

## Current Scope

The v1 pipeline supports only `FreshRetailNet-50K` through `dataset.source = fresh_retailnet`.

Stockout-censored demand can be reconstructed via `LatentDemandImputer` (`data/censorship.py`), compared against the raw observed-demand baseline. Model selection and champion evaluation run on a base subset of the 50 highest-rotation series; a separate 500-series run validates conformal-calibration stability at scale, and a 30-series fair-cost backtest isolates the imputation signal from the rest of the pipeline under a common ground truth. Each subset answers a different question, so do not average metrics across them as if they were the same experiment.

CatBoost is the current champion model, selected by simulated logistic cost (Order-Up-To), not by MAE or Winkler Score alone — point-error and logistic-cost rankings can (and do) invert.

The official `eval` split is wired as an external holdout and its temporal semantics are verified: the 7 days immediately after the train split. A non-train split inherits train's series universe rather than recomputing one — applying `min_history_days`/`top_n_series` to a 7-day split emptied it entirely (and, had it not, would have selected 50 series the model never trained on). A holdout that prepares to zero rows now raises instead of being skipped. Note the scope: horizon 7 over a 7-day split leaves ONE origin per series (50 rows, all 2024-06-26), so it corroborates the walk-forward and cannot rank models by itself. Every number currently in the thesis predates this and comes from the walk-forward. See invariant 14.

## Core Pipeline

```text
run.py
 -> load_config()
 -> run_experiment()
 -> load_prepared_panel()
 -> label_stockout_regime()
 -> [optional] LatentDemandImputer reconstruction
 -> build_supervised_frame()
 -> build_walk_forward_folds()
 -> SeasonalNaiveModel / LightGBMModel / CatBoostingModel
 -> conformal calibration (train/calibration split)
 -> choose_order_quantity()
 -> attach_inventory_costs()
 -> summarize_predictions() / summarize_costs()
 -> write_run_artifacts()
```

Related entry points in `forecasting/pipeline.py`: `run_imputation_comparison()`, `run_fair_cost_backtest()`, `train_and_save_champion()` / `run_retrain()`, `run_scoring()`. `tune_imputation_lgbm()` (in `forecasting/imputation_tuning.py`, `run_mode = tune_imputation`) is a separate, upstream entry point: it tunes the supervised imputer's LGBM hyperparameters, not the forecasting model.

The dashboard's what-if simulator does not go through the pipeline: it recomputes a
single-period Newsvendor quantity in `api/services/forecast.py` from the run's
predictions. There is no `run_whatif_simulation()` — a pipeline-side version existed,
was never wired to the CLI or the web layer, and has been removed.

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
- `src/retail_forecasting/api/` (Django) must not duplicate forecasting/inventory/evaluation logic — it renders and orchestrates calls into `forecasting/`, `inventory/`, `simulation/`, and `eda/`.

## Before Changing Code

Read these first:

- `docs/invariants.md`
- `docs/contracts/dataframes.md`
- `docs/conventions.md`
- `docs/system_design.md`
- `docs/web_layer.md` (only when touching `src/retail_forecasting/api/`)
- `docs/runs.md` (only when a number is going into the thesis, or when reading one out of `reports/`)

Prefer small changes that preserve the pipeline contract. If a change modifies target semantics, fold semantics, dataframe schemas, or inventory policy, update docs and tests in the same change.

For large changes, inspect the unstaged worktree and propose a split into two or more commits before committing. Ask for confirmation after each large project change before staging or committing.

## Commands

Install dependencies:

```bash
uv sync --extra dev
```

Install optional ML backends (required by `run_mode = tune_imputation`, whose Optuna GPSampler
needs the `torch` backend and whose experiment tracking needs `mlflow`):

```bash
uv sync --extra dev --extra ml
```

Browse past imputation tuning searches (MLflow UI at `http://localhost:5000`):

```bash
make mlflow-ui
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
uv run python -m retail_forecasting.run --config configs/experiment.yaml
```

Run the dashboard:

```bash
DJANGO_DEBUG=true uv run --env-file .env python manage.py runserver 127.0.0.1:8000
```
