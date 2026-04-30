# Retail Demand Forecasting System - TFG Makefile

# Variables
PYTHON = uv run python
PYTEST = uv run pytest
CONFIG = configs/default.yaml
DASHBOARD = src/retail_forecasting/visualization/dashboard.py

.PHONY: help install run eda dashboard test test-harness lint format clean

help: ## Muestra este mensaje de ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependencias y crea el entorno virtual con uv
	uv sync --extra ml --extra dev

run: ## Ejecuta el experimento completo con la configuración por defecto
	$(PYTHON) -m retail_forecasting.run --config $(CONFIG)

eda: ## Ejecuta el modulo de EDA reproducible sobre el panel preparado
	$(PYTHON) -m retail_forecasting.eda.run --config $(CONFIG)

dashboard: ## Lanza el dashboard interactivo de Streamlit
	uv run streamlit run $(DASHBOARD)

test: ## Ejecuta la suite completa de tests
	$(PYTEST)

test-harness: ## Ejecuta solo los tests de contratos y arquitectura (rápidos)
	$(PYTEST) tests/test_architecture_imports.py tests/test_temporal_leakage_contract.py tests/test_quantile_contract.py tests/test_dataframe_contracts.py tests/test_raw_column_boundaries.py tests/test_config_contract.py tests/test_generated_artifact_boundaries.py

lint: ## Ejecuta el linter (ruff)
	uv run ruff check .

format: ## Formatea el código (ruff)
	uv run ruff format .

clean: ## Limpia archivos temporales y cachés de Python
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .uv-cache
