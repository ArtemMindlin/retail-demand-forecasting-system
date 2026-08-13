# Variables
PYTHON = uv run python
PYTEST = uv run pytest
CONFIG = configs/experiment.yaml
SIM_CONFIG = configs/simulation.yaml
OPS_SPLIT = data/processed/ops_sim/.built
.PHONY: help install run retrain score simulate backtest-fair-cost tune-imputation eda api dev collectstatic up test test-harness lint format clean pdf

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and create virtual environment with uv
	uv sync --extra ml --extra dev

run: ## Run the full experiment with default configuration
	$(PYTHON) -m retail_forecasting.run --config $(CONFIG) --run-mode experiment

retrain: ## Retrain champion model on all available data
	$(PYTHON) -m retail_forecasting.run --config $(CONFIG) --run-mode retrain

score: ## Generate daily reorder recommendations (production mode)
	$(PYTHON) -m retail_forecasting.run --config $(CONFIG) --run-mode score_daily

simulate: $(OPS_SPLIT) ## Run rolling-origin production backtest comparing retrain cadences
	$(PYTHON) -m retail_forecasting.run --config $(SIM_CONFIG) --run-mode simulate_ops

# The OPS plane streams a dedicated train/eval split carved out of the prepared
# panel. Built once, on demand: `simulate` depends on the file, not the script.
$(OPS_SPLIT):
	$(PYTHON) scripts/build_ops_sim_split.py --train-days 49 --n-series 100

backtest-fair-cost: ## Backtest: inventory cost of each strategy vs a common ground truth (no training)
	$(PYTHON) -m retail_forecasting.run --config $(CONFIG) --run-mode fair_cost_backtest

tune-imputation: ## Tune LGBM hyperparameters for the supervised imputer, persist to disk
	$(PYTHON) -m retail_forecasting.run --config $(CONFIG) --run-mode tune_imputation

eda: ## Run the reproducible EDA module on the prepared panel
	$(PYTHON) -m retail_forecasting.eda.run --config $(CONFIG)

api: ## Start the dashboard over ASGI (production-style, no autoreload)
	uv run uvicorn retail_forecasting.api.asgi:application --host 0.0.0.0 --port 8000

dev: ## Run the Django dev server (dashboard + API) at http://localhost:8000
	DJANGO_DEBUG=true uv run --env-file .env python manage.py runserver 127.0.0.1:8000

collectstatic: ## Collect static assets into staticfiles/ (needed when DEBUG=false)
	uv run python manage.py collectstatic --noinput

up: ## Start the entire ecosystem with Docker Compose
	docker compose up --build

test: ## Run the full test suite
	$(PYTEST)

test-harness: ## Run only contract and architecture tests (fast)
	$(PYTEST) tests/test_architecture_imports.py tests/test_temporal_leakage_contract.py tests/test_quantile_contract.py tests/test_dataframe_contracts.py tests/test_raw_column_boundaries.py tests/test_config_contract.py tests/test_generated_artifact_boundaries.py

lint: ## Run the linter (ruff)
	uv run ruff check .

format: ## Format the code (ruff)
	uv run ruff format .

pdf: ## Compile memoria/main.tex with tectonic
	cd memoria && tectonic main.tex

clean: ## Clean temporary files and Python caches
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .uv-cache
