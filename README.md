# Forecasting de Demanda Retail para Decisiones de Inventario

Este repositorio contiene una primera implementación con nivel de investigación para un TFG sobre forecasting probabilístico de demanda aplicado a decisiones de inventario en retail bajo incertidumbre, stockouts y drift. El proyecto usa `FreshRetailNet-50K` como dataset por defecto y evalúa los modelos con métricas predictivas, probabilísticas y económicas.

## Enfoque

Esto no es un benchmark genérico de modelos. El pipeline está diseñado alrededor de cuatro preguntas:

1. ¿Cuánta incertidumbre captura el forecast?
2. ¿Cómo se traduce ese forecast en una decisión de inventario?
3. ¿Cuál es el coste operativo de esa decisión?
4. ¿Qué tan robusto es el sistema cuando cambia el entorno?

La implementación v1 modela demanda observada, no demanda latente censurada. El código y la documentación dejan puntos de extensión explícitos para trabajo futuro en conformal prediction, reentrenamiento adaptativo, detección de drift y políticas de inventario más allá de decisiones `newsvendor` de un solo periodo.

## Estructura del Proyecto

```text
configs/        Configuración central de experimentos
data/           Cachés de datos raw, interim y processed
docs/           Propuesta del TFG y documentos metodológicos
notebooks/      Solo notebooks ligeros de exploración
reports/        Reportes, métricas y gráficos generados
src/            Paquete Python
tests/          Tests unitarios y smoke tests
```

Módulos principales del paquete:

- `data`: carga del dataset y preparación del panel
- `features`: variables temporales y construcción del frame supervisado
- `models`: forecasting con seasonal naive y árboles boosting
- `forecasting`: backtesting walk-forward y orquestación
- `drift`: resúmenes por regímenes y puntos de extensión para análisis de drift
- `inventory`: lógica de costes tipo newsvendor y selección de cantidad a pedir
- `evaluation`: métricas y generación de reportes
- `visualization`: gráficos para reportes
- `utils`: utilidades reutilizables

## Dataset

Dataset por defecto:

- `Dingdong-Inc/FreshRetailNet-50K`
- patrón de acceso: `pd.read_parquet("hf://datasets/Dingdong-Inc/FreshRetailNet-50K/data/train.parquet")`

El loader lee solo las columnas necesarias para el pipeline v1, cachea el split en local y materializa un panel diario procesado bajo `data/processed/`.

## Inicio Rápido

Crea un entorno virtual con `uv` e instala las dependencias del proyecto:

```bash
brew install uv
uv venv
source .venv/bin/activate
uv sync --extra dev
```

Backends acelerados opcionales:

```bash
uv sync --extra dev --extra ml
```

Ejecuta el experimento por defecto:

```bash
uv run python -m retail_forecasting.run --config configs/default.yaml
```

El comando escribe un directorio de ejecución con timestamp dentro de `reports/` con:

- `report.md`
- `metrics_summary.csv`
- `fold_metrics.csv`
- `cost_summary.csv`
- `predictions.csv`
- gráficos si están habilitados

## Configuración

Archivo principal de configuración:

- [configs/default.yaml](/Users/artemmindlin/code/sandbox/retail-demand-forecasting-system/configs/default.yaml)

Parámetros importantes:

- fuente del dataset e id del dataset en Hugging Face
- `dataset.horizon`
- `dataset.top_n_series`
- `validation.n_folds`
- `validation.initial_train_days`
- `models.point_model`
- `models.quantiles`
- `inventory.overstock_cost`
- `inventory.stockout_cost`

Para cambiar el horizonte de forecast:

```yaml
dataset:
  horizon: 14
```

Para cambiar el backend del modelo:

```yaml
models:
  point_model: auto_boosting
```

La implementación actual intenta usar `LightGBM`, luego `XGBoost` y después un fallback a `scikit-learn`. Los modelos de cuantiles también usan fallback a `scikit-learn` cuando hace falta.

## Reproducibilidad

El pipeline está diseñado para ser suficientemente determinista como baseline de TFG:

- semilla aleatoria fija en configuración
- sin split aleatorio de train/validation
- solo validación temporal walk-forward
- sin uso de información futura en el feature engineering
- toda la configuración del experimento se serializa en la salida del reporte
- dependencias bloqueables mediante `uv.lock`

Para generar o refrescar el lockfile:

```bash
uv lock
```

## Tests

Ejecuta la suite de tests con:

```bash
uv run pytest
```

Los tests cubren:

- límites de los splits walk-forward
- construcción de variables temporales sin leakage futuro
- lógica de costes newsvendor
- ejecución end-to-end mínima con un panel sintético

## Alcance Actual

Implementado en la v1:

- ingesta de FreshRetailNet-50K con caché local
- filtrado y preparación del panel
- variables temporales con lags y rolling features
- baseline seasonal naive
- modelo global de boosting con soporte para cuantiles
- backtesting walk-forward
- evaluación predictiva, probabilística y económica
- generación de reportes en Markdown

Planificado a continuación:

- recuperación de demanda latente
- conformal prediction
- detectores de drift
- políticas de reentrenamiento adaptativo
- políticas base-stock y reorder-point
- dashboard en Streamlit
