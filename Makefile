# Retail Demand Forecasting System - TFG Makefile

# Variables
PYTHON = uv run python
PYTEST = uv run pytest
CONFIG = configs/default.yaml
DASHBOARD = src/retail_forecasting/visualization/dashboard.py

.PHONY: help install run retrain score simulate eda dashboard api mlflow up test test-harness lint format clean

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

simulate: ## Run operational simulation comparing retrain cadences
	$(PYTHON) -m retail_forecasting.run --config $(CONFIG) --run-mode simulate_ops

eda: ## Run the reproducible EDA module on the prepared panel
	$(PYTHON) -m retail_forecasting.eda.run --config $(CONFIG)

dashboard: ## Launch the interactive Streamlit dashboard
	uv run streamlit run $(DASHBOARD)

api: ## Start the FastAPI microservice
	uv run uvicorn retail_forecasting.api.main:app --reload

mlflow: ## Start the MLflow UI
	uv run mlflow ui

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

clean: ## Clean temporary files and Python caches
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .uv-cache
