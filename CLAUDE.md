# CLAUDE.md

This is a research/prototype Python project for retail demand forecasting and inventory decisions under uncertainty, stockouts, and temporal drift.

The priority of this repo is experimental validity: avoid temporal leakage, preserve dataframe contracts, and keep forecasting, inventory decisions, and evaluation separated.

## Repo Map

- `configs/`: one folder per run mode, so a config file declares only the `Settings` sections
  its own mode reads. `configs/experiment/default.yaml` is the canonical v1 config
  and `daily.yaml` the daily variant, both `run_mode = experiment`. A `large.yaml` held the
  500-series scale variant until every experiment config moved to 500 series, which left it
  differing from `default.yaml` only in `run_name` and `make_plots`.
  The other folders (`retrain/`, `score_daily/`, `simulate_ops/`, `fair_cost_backtest/`,
  `tune_imputation/`, `tune_forecasting/`) hold a `default.yaml` each, `eda/` included: exploratory analysis is a run mode like the
  rest, not a second CLI. `project.run_mode` always matches the folder name, so `--run-mode`
  is only for ad-hoc overrides. The map of which sections each mode reads is
  `MODE_SECTIONS` in `contracts/contracts_config.py`, enforced by `tests/test_config_layout.py`:
  declaring a section the mode never reads is a knob that silently does nothing, and fails the
  suite. The price of the split is that `dataset` and `preprocessing` are duplicated across
  files; the same test pins the identity fields that must not drift apart.
- `data/`: local raw/interim/processed caches. Do not commit generated datasets.
- `mlruns/` + `mlflow.db`: the run store. Every mode opens an MLflow run and writes its artifacts straight into that run's directory, so there is one copy and no mirroring step. Both are gitignored, which is why each artifact directory carries an `mlflow_run.json` naming its run: with a database-backed store, `mlruns/` holds artifacts and nothing that identifies them. The artifact location baked into each experiment row is RELATIVE, so the same `mlflow.db` works in the repo and mounted at `/app` in the container — at the price of a run launched from a subdirectory not finding its artifacts. MLflow cannot store it that way itself; see `docs/web_layer.md`.
- `docs/`: system of record for architecture, contracts, invariants, and decisions.
- `notebooks/`: lightweight exploration only. Production pipeline logic belongs in `src/`.
- `var/`: the dashboard's scratch state, created on demand and gitignored. Holds `active_run.log`, which a triggered run writes while it is in flight and the console tails — not the artifact of a finished run, which goes to the run store. Relocatable with `RETAIL_STATE_DIR`. A `reports/` directory used to sit here holding run output; it is gone, and so are the pre-migration copies of the runs `docs/runs.md` cites — those runs live in the store now and resolve from there, which is what `mlflow_run.json` is for.
- `scripts/build_ops_sim_split.py`: carves the dedicated train/eval split the OPS plane streams, into `data/processed/ops_sim/`. Invoked by `make simulate` when the split is missing; not something to run by hand.
- `manage.py`: Django management entrypoint for the dashboard (`src/retail_forecasting/api/`).
- `src/retail_forecasting/config.py`: typed settings loaded from YAML.
- `src/retail_forecasting/data/`: raw dataset loading, raw-to-panel preparation, and `censorship.py` (`LatentDemandImputer` — stockout/censored-demand reconstruction strategies).
- `src/retail_forecasting/features/`: supervised frame creation, temporal features, and target construction.
- `src/retail_forecasting/forecasting/`: walk-forward validation, conformal calibration, and experiment/retrain/scoring orchestration (`pipeline.py`); `fair_cost.py` is the fair-cost backtest, its own module because it trains no model and shares nothing with the walk-forward but the synthetic-censoring holdout; `imputation_tuning.py` runs a separate Optuna search over the supervised imputer's LGBM hyperparameters and persists the winner (never fitted weights) to `models.models_dir/imputation_lgbm_params.json` — its flow, draws and persist gates are mapped in `src/retail_forecasting/forecasting/imputation_tuning.md`. It logs to MLflow (`mlflow.db`, browsable via `make mlflow-ui`), as does every other mode through `tracking.py`. `open_run_directory` opens the run and hands back the directory its artifacts live in, which the pipeline writes into directly: for a local store, writing into that directory IS logging, so there is no upload step and one copy. On top of the files, `log_run_metadata` records what a directory cannot answer — the config, the metrics keyed by model, the decisions — which is what lets `mlflow.search_runs` compare runs.
- `src/retail_forecasting/models/`: forecast models only (`naive.py`, `boosting.py` for LightGBM, `catboosting.py` for CatBoost — the current champion).
- `src/retail_forecasting/inventory/`: newsvendor order quantity and the catalogue-wide cost coefficients. A synthetic per-series cost profile lived here and was removed; invariant 43 records why. The multi-period Order-Up-To simulation and the LP capacity allocator lived here and were removed; the decision is single-period, one per forecast origin.
- `src/retail_forecasting/simulation/`: OPS-plane rolling-origin production backtest, reused by the dashboard's `/ops/` view. Independent single-period Newsvendor decisions — no inventory state, no lead time — so its costs rank policies rather than reproduce a replenishment ledger.
- `src/retail_forecasting/evaluation/`: metrics, run reporting, XAI/explainability, and `latex_exporter.py`, which renders the generated tables of chapter 6. It sits here rather than in `utils` because it resolves a run by the name `docs/runs.md` cites, and reaching the run index needs `tracking` — which `utils`, importing no first-party layer, cannot have without closing a cycle.
- `src/retail_forecasting/drift/`: regime/drift analysis hooks.
- `src/retail_forecasting/eda/`: exploratory analysis and figure generation, surfaced in the dashboard's `/eda/` tab. Entered through `run_mode = eda` (`--split` picks the split); the dataset config is honoured as written, so `top_n_series: null` in `configs/eda/default.yaml` is what makes the analysis describe the whole panel rather than the subset a model trains on. Set it to explore an experiment's exact panel instead.
- `src/retail_forecasting/tracking.py`: MLflow tracking for every run mode — its own layer, not part of `evaluation`, because recording a run is something every mode does and `eda` may not import `evaluation`. Owns the tracking-store URI and derives the artifact root from it.
- `src/retail_forecasting/contracts/`: pydantic-style contracts for backtesting, business rules, config, drift, feature engineering, quality, and tuning — enforced by the `tests/test_*_contract.py` / `tests/test_*_boundaries.py` suite.
- `src/retail_forecasting/api/`: Django dashboard and JSON API (`views/`, `templates/`, `static/`, `services/`) — visualizes experiments, EDA, drift, the OPS plane, and exposes a documented JSON surface (`/api/forecast`, `/api/skus`, `/predict_orders`, etc.). Served via `asgi.py`/`wsgi.py`.
- `tests/`: contract tests, smoke tests, synthetic-panel tests, and API tests.

## Current Scope

The v1 pipeline supports only `FreshRetailNet-50K` through `dataset.source = fresh_retailnet`.

Stockout-censored demand can be reconstructed via `LatentDemandImputer` (`data/censorship.py`), compared against the raw observed-demand baseline by `run_fair_cost_backtest` -- the only comparison that ranks strategies, since invariant 42 rules out the reconstruction-MAE route and an experiment scores one arm per run against its own target. Every forecasting mode runs on the 500 highest-rotation series: model selection, champion evaluation, retraining and scoring. That is the scale the hyperparameters are tuned at, and invariant 41 measured what happens when the two diverge. The fair-cost backtest still samples 30 series, now drawn from those 500, to isolate the imputation signal from the rest of the pipeline under a common ground truth. The sample is a censoring MASK, not a subset: only those series are scored, while the supervised imputer's teacher keeps the whole 500-series panel it gets in deployment (invariant 44). It scores them on 20 censoring draws and reports the cost gap against the observed signal as a PAIRED difference with a 95% interval: a single draw ranked four strategies with nothing around the ranking, and the ranking does invert with the panel. The OPS plane is the exception at 100 series, because it streams its own re-carved split.

CatBoost is the current champion model, selected by simulated logistic cost, not by MAE or Winkler Score alone — point-error and logistic-cost rankings can (and do) invert.

The official `eval` split is wired as an external holdout and its temporal semantics are verified: the 7 days immediately after the train split. A non-train split inherits train's series universe rather than recomputing one — applying `min_history_days`/`top_n_series` to a 7-day split emptied it entirely (and, had it not, would have selected 50 series the model never trained on). A holdout that prepares to zero rows now raises instead of being skipped. Note the scope: horizon 7 over a 7-day split leaves ONE origin per series (50 rows, all 2024-06-26), so it corroborates the walk-forward and cannot rank models by itself. Every number currently in the thesis predates this and comes from the walk-forward. See invariant 14.

## Core Pipeline

```text
run.py
 -> load_config()
 -> run_experiment()
 -> load_prepared_panel()            (train, then the eval holdout)
 -> validate_prepared_panel()
 -> [optional] LatentDemandImputer reconstruction (one arm per run: `imputation_strategy`
    picks it, `none` = Observed). It REWRITES `observed_demand`, so every step below --
    regime labels included -- sees the reconstruction, not the raw sale.
 -> label_all_regimes()
 -> build_supervised_frame()
 -> build_walk_forward_folds()
 -> SeasonalNaiveModel / LightGBMModel / CatBoostingModel
 -> conformal calibration (train/calibration split)
 -> choose_order_quantity()
 -> attach_inventory_costs()
 -> summarize_predictions() / summarize_costs()
 -> write_run_artifacts()
```

Related entry points in `forecasting/pipeline.py`: `train_and_save_champion()` / `run_retrain()`, `run_scoring()`. `run_fair_cost_backtest()` lives in `forecasting/fair_cost.py` (`run_mode = fair_cost_backtest`). `tune_imputation_lgbm()` (in `forecasting/imputation_tuning.py`, `run_mode = tune_imputation`) is a separate, upstream entry point: it tunes the supervised imputer's LGBM hyperparameters, not the forecasting model.

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
- `docs/runs.md` (only when a number is going into the thesis, or when reading one out of the run store)

Prefer small changes that preserve the pipeline contract. If a change modifies target semantics, fold semantics, dataframe schemas, or inventory policy, update docs and tests in the same change.

For large changes, inspect the unstaged worktree and propose a split into two or more commits before committing. Ask for confirmation after each large project change before staging or committing.

## Commands

Install dependencies:

```bash
uv sync --extra dev
```

Wire the git hooks. Two stages: `pre-commit` runs ruff, ruff-format and mypy, and `pre-push`
runs the suite. Needed once per clone, because pre-commit only installs the hook types it is
told to, and a hook that is not installed fails silently by never running:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

The suite is on push and not on commit because it takes about three minutes. Note that a green
commit therefore says nothing about whether the tests pass: ruff and mypy accept plenty of code
the suite rejects.

Install optional ML backends (required by `run_mode = tune_imputation`, whose Optuna GPSampler
needs the `torch` backend). `mlflow` is NOT among them: it moved to the core dependencies when
the run store became where the pipeline writes, so without it there is no run and no dashboard.

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
uv run python -m retail_forecasting.run --config configs/experiment/default.yaml
```

Every other mode has a `make` target named after it (`make help` lists them).

Run the dashboard:

```bash
DJANGO_DEBUG=true uv run --env-file .env python manage.py runserver 127.0.0.1:8000
```
