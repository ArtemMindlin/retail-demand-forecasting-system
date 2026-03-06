# Retail Demand Forecasting For Inventory Decisions

This repository contains the first research-grade implementation of a TFG on probabilistic demand forecasting for retail inventory decisions under uncertainty, stockouts, and drift. The project uses `FreshRetailNet-50K` as the default dataset and evaluates models with predictive, probabilistic, and economic metrics.

## Focus

This is not a generic model benchmark. The pipeline is designed around four questions:

1. How much uncertainty does the forecast capture?
2. How does that forecast translate into an inventory decision?
3. What is the operational cost of that decision?
4. How robust is the system when the environment changes?

The v1 implementation models observed demand, not latent censored demand. The code and docs keep explicit hooks for future work on conformal prediction, adaptive retraining, drift detection, and inventory policies beyond single-period newsvendor decisions.

## Project Structure

```text
configs/        Central experiment configuration
data/           Raw, interim, and processed data caches
docs/           TFG proposal and methodological documents
notebooks/      Lightweight exploratory notebooks only
reports/        Generated reports, metrics, and plots
src/            Python package
tests/          Unit tests and smoke tests
```

Main package modules:

- `data`: dataset loading and panel preparation
- `features`: temporal features and supervised frame creation
- `models`: seasonal naive and boosted tree forecasting
- `forecasting`: walk-forward backtesting and orchestration
- `drift`: regime summaries and extension points for drift analysis
- `inventory`: newsvendor cost logic and order quantity selection
- `evaluation`: metrics and report generation
- `visualization`: plots for reports
- `utils`: reusable helpers

## Dataset

Default dataset:

- `Dingdong-Inc/FreshRetailNet-50K`
- access pattern: `pd.read_parquet("hf://datasets/Dingdong-Inc/FreshRetailNet-50K/data/train.parquet")`

The loader reads only the columns required by the v1 pipeline, caches the split locally, and materializes a processed daily panel under `data/processed/`.

## Quickstart

Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional accelerated backends:

```bash
python -m pip install -e ".[dev,ml]"
```

Run the default experiment:

```bash
python -m retail_forecasting.run --config configs/default.yaml
```

The command writes a timestamped run directory inside `reports/` with:

- `report.md`
- `metrics_summary.csv`
- `fold_metrics.csv`
- `cost_summary.csv`
- `predictions.csv`
- plots if enabled

## Configuration

Main configuration file:

- [configs/default.yaml](/Users/artemmindlin/code/sandbox/retail-demand-forecasting-system/configs/default.yaml)

Important knobs:

- dataset source and Hugging Face dataset id
- `dataset.horizon`
- `dataset.top_n_series`
- `validation.n_folds`
- `validation.initial_train_days`
- `models.point_model`
- `models.quantiles`
- `inventory.overstock_cost`
- `inventory.stockout_cost`

To change the forecast horizon:

```yaml
dataset:
  horizon: 14
```

To change the model backend:

```yaml
models:
  point_model: auto_boosting
```

The current implementation tries `LightGBM`, then `XGBoost`, then a `scikit-learn` fallback. Quantile models also fall back to `scikit-learn` when needed.

## Reproducibility

The pipeline is designed to be deterministic enough for a TFG baseline:

- fixed random seed in config
- no random train/validation split
- walk-forward temporal validation only
- no future information in feature engineering
- all experiment settings serialized into the report output

## Tests

Run the test suite with:

```bash
pytest
```

The tests cover:

- walk-forward split boundaries
- temporal feature construction without future leakage
- newsvendor cost logic
- end-to-end smoke run on a synthetic panel

## Current Scope

Implemented in v1:

- FreshRetailNet-50K ingestion with local cache
- panel filtering and preparation
- lag and rolling temporal features
- seasonal naive baseline
- global boosted tree model with quantile support
- walk-forward backtesting
- predictive, probabilistic, and economic evaluation
- Markdown report generation

Planned next:

- latent demand recovery
- conformal prediction
- drift detectors
- adaptive retraining policies
- base-stock and reorder-point policies
- Streamlit dashboard
