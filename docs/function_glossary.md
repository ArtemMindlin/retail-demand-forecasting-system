# Function Glossary

Glosario practico de las funciones principales del proyecto. El objetivo no es documentar cada helper minimo, sino tener un mapa rapido para recordar:

- que hace una funcion;
- donde esta;
- que recibe;
- que devuelve;
- en que parte del flujo aparece.

## Flujo principal

```text
main()
 -> load_config()
 -> run_experiment()
    -> load_prepared_panel()
       -> load_raw_split()
       -> prepare_daily_panel()
    -> run_experiment_from_frame()
       -> label_stockout_regime()
       -> build_supervised_frame()
       -> build_walk_forward_folds()
       -> SeasonalNaiveModel.fit()
       -> SeasonalNaiveModel.predict()
       -> LightGBMModel.fit()
       -> LightGBMModel.predict()
       -> LightGBMModel.predict_quantiles()
       -> choose_order_quantity()
       -> attach_inventory_costs()
       -> summarize_predictions()
       -> summarize_costs()
       -> write_run_artifacts()
          -> build_markdown_report()
```

## CLI and Configuration

### `build_parser`
- Archivo: `src/retail_forecasting/run.py`
- Que hace: define los argumentos del CLI.
- Recibe: nada.
- Devuelve: `argparse.ArgumentParser`.
- Se usa en: `main()`.

### `main`
- Archivo: `src/retail_forecasting/run.py`
- Que hace: punto de entrada del CLI. Lee argumentos, carga configuracion, ejecuta el experimento e imprime la ruta del report.
- Recibe: nada.
- Devuelve: `None`.
- Se usa en: ejecucion `python -m retail_forecasting.run`.

### `load_config`
- Archivo: `src/retail_forecasting/config.py`
- Que hace: carga el YAML y construye un objeto `Settings` tipado y validado automáticamente con Pydantic.
- Recibe:
  - `path`: ruta al archivo de configuración.
- Devuelve: `Settings`.
- Se usa en: `main()`.

## Data Loading

### `build_hf_uri`
- Archivo: `src/retail_forecasting/data/dataset.py`
- Que hace: construye la URI `hf://...` del split remoto.
- Recibe:
  - `dataset_id`;
  - `split_path`.
- Devuelve: `str`.
- Se usa en: `load_raw_split()`.

### `processed_panel_path`
- Archivo: `src/retail_forecasting/data/dataset.py`
- Que hace: calcula la ruta del parquet procesado para un split.
- Recibe:
  - `dataset_config`;
  - `split`.
- Devuelve: `Path`.
- Se usa en: `load_prepared_panel()`.

### `load_raw_split`
- Archivo: `src/retail_forecasting/data/dataset.py`
- Que hace: carga el split raw desde cache local o desde Hugging Face.
- Recibe:
  - `dataset_config`;
  - `split`;
  - `columns`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `load_prepared_panel()`.

### `prepare_daily_panel`
- Archivo: `src/retail_forecasting/data/dataset.py`
- Que hace: transforma el dataset bruto en un panel diario listo para modelado.
- Recibe:
  - `frame`;
  - `dataset_config`;
  - `preprocessing_config`.
- Devuelve: `pd.DataFrame`.
- Hace internamente:
  - renombrado de columnas;
  - parseo de fechas;
  - filtrado de ventas negativas;
  - eliminacion de duplicados;
  - creacion de `series_id`;
  - filtrado por historial minimo;
  - seleccion de top series;
  - imputacion basica de nulos.
- Se usa en: `load_prepared_panel()`.

### `load_prepared_panel`
- Archivo: `src/retail_forecasting/data/dataset.py`
- Que hace: devuelve el panel ya procesado, leyendolo de cache si existe o generandolo si no.
- Recibe:
  - `dataset_config`;
  - `preprocessing_config`;
  - `split`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `run_experiment()`.

## Feature Engineering

### `build_feature_frame`
- Archivo: `src/retail_forecasting/features/engineering.py`
- Que hace: construye las features compartidas por entrenamiento e inferencia, sin crear target.
- Recibe:
  - `panel`;
  - `feature_config`.
- Devuelve:
  - `pd.DataFrame` con las columnas originales y las features derivadas;
  - `list[str]` con los nombres de las columnas de features creadas.
- Se usa en: `build_supervised_frame()`, `build_inference_frame()` y `build_inference_frame_with_fallback()`.

### `build_supervised_frame`
- Archivo: `src/retail_forecasting/features/engineering.py`
- Que hace: construye el dataset supervisado para ML con features y target.
- Recibe:
  - `panel`;
  - `feature_config`;
  - `horizon`.
- Devuelve:
  - `pd.DataFrame` con el frame supervisado;
  - `FeatureMetadata` con las columnas de features y metadatos auditables.
- Hace internamente:
  - variables de calendario;
  - lags de demanda;
  - rolling mean, sum y std;
  - lags de descuento;
  - lags de stockout;
  - lags meteorologicos;
  - ids estaticos;
  - target `target_lead_time_demand`.
- Se usa en: `run_experiment_from_frame()`.

### `build_inference_frame`
- Archivo: `src/retail_forecasting/features/engineering.py`
- Que hace: construye una fila lista para prediccion por serie, sin target futuro.
- Recibe:
  - `panel`;
  - `feature_config`.
- Devuelve:
  - `pd.DataFrame` con la ultima fila valida por `series_id`;
  - `FeatureMetadata` con las columnas de features y metadatos auditables.
- Se usa en: futuros flujos de inferencia/despliegue.

### `build_inference_frame_with_fallback`
- Archivo: `src/retail_forecasting/features/engineering.py`
- Que hace: construye la ultima fila por `series_id` para inferencia y decide si cada fila va al modelo o a un fallback jerarquico de `cold start`.
- Recibe:
  - `panel`;
  - `feature_config`;
  - `horizon`.
- Devuelve:
  - `pd.DataFrame` con una fila por serie y columnas de routing:
    - `prediction_source`;
    - `fallback_level`;
    - `fallback_target_lead_time_demand`;
  - `InferenceFallbackMetadata` con contadores de filas del modelo y del fallback.
- Jerarquia de fallback:
  - `series_id`;
  - `product_id`;
  - `third_category_id`;
  - global.
- Se usa en: futuros flujos de inferencia/despliegue con politica explicita de `cold start`.

### `FeatureMetadata`
- Archivo: `src/retail_forecasting/contracts/contracts_engineering.py`
- Que hace: contrato Pydantic congelado con metadatos auditables de feature engineering.
- Campos principales:
  - `mode`;
  - `feature_columns`;
  - `target_column`;
  - `horizon`;
  - `lags`;
  - `rolling_windows`;
  - `input_rows`;
  - `output_rows`;
  - `dropped_rows_missing_target`;
  - `dropped_rows_missing_features`;
  - `rows_not_latest_origin`.
- Se usa en: salidas de `build_feature_frame()`, `build_supervised_frame()` y `build_inference_frame()`.

### `InferenceFallbackMetadata`
- Archivo: `src/retail_forecasting/features/engineering.py`
- Que hace: contrato Pydantic congelado con metadatos auditables del routing de inferencia con fallback.
- Campos principales:
  - `feature_columns`;
  - `horizon`;
  - `input_rows`;
  - `output_rows`;
  - `model_rows`;
  - `cold_start_rows`;
  - `fallback_rows_series`;
  - `fallback_rows_product`;
  - `fallback_rows_third_category`;
  - `fallback_rows_global`.
- Se usa en: salida de `build_inference_frame_with_fallback()`.

### `_build_target`
- Archivo: `src/retail_forecasting/features/engineering.py`
- Que hace: construye el target como suma de demanda observada en el horizonte futuro.
- Recibe:
  - `series_group`;
  - `horizon`.
- Devuelve: `pd.Series`.
- Se usa en: `build_supervised_frame()`.

## Backtesting and Pipeline

### `build_walk_forward_folds`
- Archivo: `src/retail_forecasting/forecasting/backtesting.py`
- Que hace: genera folds temporales walk-forward respetando el horizonte del target.
- Recibe:
  - `panel`;
  - `validation_config`;
  - `horizon`.
- Devuelve: `list[FoldSpec]`, donde cada `FoldSpec` es un contrato Pydantic congelado con `fold_id`, `horizon`, `train_end_date`, `validation_start_date` y `validation_end_date`.
- Invariantes:
  - `horizon > 0`;
  - `fold_id >= 0`;
  - `train_end_date = validation_start_date - horizon`.
- Se usa en: `run_experiment_from_frame()`.

### `run_experiment`
- Archivo: `src/retail_forecasting/forecasting/pipeline.py`
- Que hace: punto de entrada del pipeline experimental.
- Recibe:
  - `settings`.
- Devuelve: `RunArtifacts`.
- Flujo:
  - valida compatibilidad de configuracion;
  - carga el panel preparado;
  - delega en `run_experiment_from_frame()`.
- Se usa en: `main()`.

### `run_experiment_from_frame`
- Archivo: `src/retail_forecasting/forecasting/pipeline.py`
- Que hace: ejecuta el experimento completo desde un panel ya cargado.
- Recibe:
  - `panel`;
  - `settings`.
- Devuelve: `RunArtifacts`.
- Hace internamente:
  - etiquetado de regimen de stockout;
  - feature engineering;
  - construccion de folds;
  - entrenamiento y validacion de modelos;
  - predicciones por fold;
  - evaluacion predictiva y economica;
  - escritura de artefactos finales.

### `_build_baseline_predictions`
- Archivo: `src/retail_forecasting/forecasting/pipeline.py`
- Que hace: genera las predicciones del baseline naive para un fold y las convierte en decision de inventario.
- Recibe:
  - `validation_frame`;
  - `baseline_model`;
  - `fold_id`;
  - `settings`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `run_experiment_from_frame()`.

### `_build_boosting_predictions`
- Archivo: `src/retail_forecasting/forecasting/pipeline.py`
- Que hace: genera prediccion puntual, cuantiles, decision de pedido y costes para el modelo boosting.
- Recibe:
  - `validation_frame`;
  - `feature_columns`;
  - `model`;
  - `fold_id`;
  - `settings`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `run_experiment_from_frame()`.

## Models

### `SeasonalNaiveModel.fit`
- Archivo: `src/retail_forecasting/models/naive.py`
- Que hace: guarda el historial por serie para usarlo en prediccion.
- Recibe:
  - `panel`.
- Devuelve: `SeasonalNaiveModel`.
- Se usa en: `run_experiment_from_frame()`.

### `SeasonalNaiveModel.predict`
- Archivo: `src/retail_forecasting/models/naive.py`
- Que hace: predice la demanda acumulada del horizonte usando estacionalidad simple.
- Recibe:
  - `frame`.
- Devuelve: `np.ndarray`.
- Se usa en: `_build_baseline_predictions()`.

### `LightGBMModel.fit`
- Archivo: `src/retail_forecasting/models/boosting.py`
- Que hace: entrena un modelo puntual y un modelo por cada cuantil solicitado.
- Recibe:
  - `features`;
  - `target`.
- Devuelve: `LightGBMModel`.
- Se usa en: `run_experiment_from_frame()`.

### `LightGBMModel.predict`
- Archivo: `src/retail_forecasting/models/boosting.py`
- Que hace: genera la prediccion puntual y recorta valores negativos.
- Recibe:
  - `features`.
- Devuelve: `np.ndarray`.
- Se usa en: `_build_boosting_predictions()`.

### `LightGBMModel.predict_quantiles`
- Archivo: `src/retail_forecasting/models/boosting.py`
- Que hace: genera cuantiles y fuerza monotonia entre ellos.
- Recibe:
  - `features`.
- Devuelve: `dict[str, np.ndarray]`.
- Se usa en: `_build_boosting_predictions()`.

## Drift and Inventory

### `label_stockout_regime`
- Archivo: `src/retail_forecasting/drift/regime.py`
- Que hace: etiqueta cada fila como `high_stockout` o `low_stockout` segun el stockout observado.
- Recibe:
  - `frame`;
  - `threshold`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `label_all_regimes()`.

### `label_demand_velocity_regime`
- Archivo: `src/retail_forecasting/drift/regime.py`
- Que hace: etiqueta series como `fast_moving` o `slow_moving` según su promedio de demanda.
- Recibe:
  - `frame`;
  - `threshold`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `label_all_regimes()`.

### `label_promotion_regime`
- Archivo: `src/retail_forecasting/drift/regime.py`
- Que hace: etiqueta filas como `on_promotion` o `baseline_price` según si hay descuento activo.
- Recibe:
  - `frame`;
  - `discount_col`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `label_all_regimes()`.

### `label_seasonal_regime`
- Archivo: `src/retail_forecasting/drift/regime.py`
- Que hace: etiqueta filas como `peak_holiday` o `standard_season` según las festividades.
- Recibe:
  - `frame`;
  - `holiday_col`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `label_all_regimes()`.

### `label_all_regimes`
- Archivo: `src/retail_forecasting/drift/regime.py`
- Que hace: aplica todos los etiquetadores de régimen de forma secuencial.
- Recibe:
  - `frame`;
  - `velocity_threshold`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `run_experiment_from_frame()`, `train_and_save_champion()`, y `run_scoring()`.

### `critical_fractile`
- Archivo: `src/retail_forecasting/inventory/newsvendor.py`
- Que hace: calcula el fractil critico a partir del coste de rotura y de sobrestock.
- Recibe:
  - `inventory_config`.
- Devuelve: `float`.
- Se usa en: `choose_order_quantity()`.

### `choose_order_quantity`
- Archivo: `src/retail_forecasting/inventory/newsvendor.py`
- Que hace: convierte predicciones en una cantidad de pedido.
- Recibe:
  - `predictions`;
  - `inventory_config`;
  - `quantile_columns`;
  - `quantile_levels`.
- Devuelve: `pd.Series`.
- Logica:
  - si hay cuantiles, interpola el cuantil correspondiente al fractil critico;
  - si no hay cuantiles, usa la prediccion puntual.
- Se usa en: `_build_baseline_predictions()` y `_build_boosting_predictions()`.

### `attach_inventory_costs`
- Archivo: `src/retail_forecasting/inventory/newsvendor.py`
- Que hace: calcula unidades y costes de sobrestock y rotura.
- Recibe:
  - `predictions`;
  - `inventory_config`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `_build_baseline_predictions()` y `_build_boosting_predictions()`.

### `_interpolate_quantile`
- Archivo: `src/retail_forecasting/inventory/newsvendor.py`
- Que hace: aproxima el valor del cuantil objetivo por interpolacion lineal.
- Recibe:
  - `levels`;
  - `values`;
  - `target_level`.
- Devuelve: `float`.
- Se usa en: `choose_order_quantity()`.

## Evaluation and Reporting

### `summarize_predictions`
- Archivo: `src/retail_forecasting/evaluation/metrics.py`
- Que hace: resume metricas predictivas y probabilisticas por modelo y por fold.
- Recibe:
  - `predictions`.
- Devuelve:
  - `pd.DataFrame` resumen global;
  - `pd.DataFrame` resumen por fold.
- Se usa en: `run_experiment_from_frame()`.

### `summarize_costs`
- Archivo: `src/retail_forecasting/evaluation/metrics.py`
- Que hace: resume el coste operativo por modelo.
- Recibe:
  - `predictions`.
- Devuelve: `pd.DataFrame`.
- Se usa en: `run_experiment_from_frame()`.

### `_build_metric_record`
- Archivo: `src/retail_forecasting/evaluation/metrics.py`
- Que hace: construye un registro de metricas para un subconjunto de predicciones.
- Recibe:
  - `predictions`;
  - `model_name`;
  - `backend_name`.
- Devuelve: `dict[str, float | str]`.
- Se usa en: `summarize_predictions()`.

### `pinball_loss`
- Archivo: `src/retail_forecasting/evaluation/metrics.py`
- Que hace: calcula la perdida pinball para un cuantil.
- Recibe:
  - `actual`;
  - `predicted`;
  - `quantile`.
- Devuelve: `float`.
- Se usa en: `_build_metric_record()`.

### `write_run_artifacts`
- Archivo: `src/retail_forecasting/evaluation/reporting.py`
- Que hace: guarda CSVs, metadata de backtest, plots y report final en el directorio de la corrida.
- Recibe:
  - `artifacts`;
  - `settings`.
- Devuelve: `RunArtifacts`.
- Se usa en: `run_experiment_from_frame()`.

### `BacktestMetadata`
- Archivo: `src/retail_forecasting/evaluation/reporting.py`
- Que hace: contrato Pydantic congelado que resume la trazabilidad de una corrida de backtesting.
- Incluye:
  - resumen del dataset;
  - metadata del frame supervisado;
  - folds ejecutados y filas por fold;
  - modelos ejecutados;
  - hash de configuracion y commit Git.
- Se escribe en: `backtest_metadata.json`.

### `build_markdown_report`
- Archivo: `src/retail_forecasting/evaluation/reporting.py`
- Que hace: construye el texto del `report.md`.
- Recibe:
  - `artifacts`;
  - `settings`.
- Devuelve: `str`.
- Se usa en: `write_run_artifacts()`.

## Web Layer

Funciones de `src/retail_forecasting/api/`. Ver `docs/web_layer.md` para la
arquitectura. La regla que las gobierna: `services/` es pandas puro sin Django,
y las vistas solo leen parametros, llaman a un servicio y renderizan.

### `ArtifactStore`
- Archivo: `src/retail_forecasting/api/services/runs.py`
- Que hace: acceso cacheado y de solo lectura a los artefactos de `reports/`. Localiza el run mas reciente, carga y filtra las predicciones por el champion, y valida nombres de run contra path traversal.
- Metodos principales: `latest_run_path()`, `latest_predictions()`, `grouped_predictions()`, `resolve_run()`, `runs_with()`, `list_eda_runs()`, `invalidate()`.
- Se usa en: todas las vistas, via `api/store.get_store()`.

### `WhatIfParams`
- Archivo: `src/retail_forecasting/api/services/forecast.py`
- Que hace: agrupa los cuatro parametros del simulador (nivel de servicio, coste de rotura, coste de almacenamiento, capacidad) y deriva `alpha` y `critical_ratio`.
- Recibe: `from_mapping()` acepta tanto los nombres de query string como los camelCase del cuerpo JSON.
- Devuelve: instancia inmutable.
- Se usa en: dashboard, tabla de SKU y los endpoints JSON.

### `empirical_conformal`
- Archivo: `src/retail_forecasting/api/services/forecast.py`
- Que hace: calcula la semianchura del intervalo conformal como el cuantil `1 - alpha` de los residuos absolutos, y la cobertura observada.
- Recibe:
  - `sku_df`;
  - `alpha`.
- Devuelve: `ConformalStats`.
- Se usa en: `compute_forecast()` y `compute_sku_table()`.

### `compute_forecast`
- Archivo: `src/retail_forecasting/api/services/forecast.py`
- Que hace: payload completo del dashboard para un SKU: serie del grafico, KPIs (coste, cobertura, MAE, PSI) y la recomendacion Newsvendor.
- Recibe:
  - `grouped`;
  - `params`;
  - `selected_sku`.
- Devuelve: `dict`, o `{"status": "no_predictions"}` si no hay artefactos.
- Se usa en: vista `dashboard` y `POST /api/forecast`.

### `compute_sku_table`
- Archivo: `src/retail_forecasting/api/services/forecast.py`
- Que hace: una fila por SKU con cobertura empirica, PSI, margen, cantidad recomendada y estado operativo.
- Recibe:
  - `grouped`;
  - `params`;
  - `limit`.
- Devuelve: `list[dict]`.
- Se usa en: vista `skus` y `GET /api/skus`.

### `per_sku_psi`
- Archivo: `src/retail_forecasting/api/services/forecast.py`
- Que hace: PSI de la demanda de una serie comparando su mitad antigua con la reciente.
- Recibe:
  - `demand` (array).
- Devuelve: `float` (0.0 si la serie es demasiado corta).
- Se usa en: KPIs del dashboard y estado por SKU.

### `load_feature_drift` / `load_alerts`
- Archivo: `src/retail_forecasting/api/services/forecast.py`
- Que hace: leen `drift_report.json` y `exceptions.csv` del run. Devuelven listas vacias si el artefacto no existe, en lugar de fabricar valores.
- Recibe:
  - `run_path`.
- Devuelve: `list[dict]`.
- Se usa en: vista `drift` y panel de alertas.

### `weekly_summary` / `series_trajectory`
- Archivo: `src/retail_forecasting/api/services/ops.py`
- Que hace: indexan el backtest de origen rodante por origen semanal (rejilla no solapada). `weekly_summary` agrega metricas por cadencia de reentreno y adjunta la comparacion pareada de `cadence_comparison.csv`; `series_trajectory` devuelve la trayectoria de una serie. Las semanas cuyos actuals no han cerrado no reportan metricas, coste incluido.
- Recibe:
  - `simulation` (`OpsSimulation`);
  - `series_id` y `cadence` en el segundo caso.
- Devuelve: `dict`.
- Se usa en: vista `ops`.
- Nota: cobertura y MAE se calculan solo sobre filas con actuals completos; el coste suma el grupo entero.

### `chart_data`
- Archivo: `src/retail_forecasting/api/services/eda.py`
- Que hace: convierte los CSV que escribe el modulo de EDA en payloads listos para graficar (histograma, scatter, boxplot, barras…).
- Recibe:
  - `eda_path`;
  - `name` (validado contra el catalogo de figuras).
- Devuelve: `dict` con la clave `type` que elige el renderizador.
- Se usa en: vista `eda`.

### `imputation_strategies` / `rank_quality`
- Archivo: `src/retail_forecasting/api/services/experiments.py`
- Que hace: reconstruccion latente por estrategia para una serie, y ranking de las estrategias por MAE de reconstruccion con su sesgo y direccion (infra/sobre-imputa).
- Recibe:
  - `run_path` y `series_id`; `quality` en el segundo caso.
- Devuelve: `dict` / `list[dict]`.
- Se usa en: vista `latent`.

### `PipelineRunner`
- Archivo: `src/retail_forecasting/api/services/pipeline.py`
- Que hace: lanza `python -m retail_forecasting.run` como subproceso en background, con lock de ejecucion unica, rate limit por IP y volcado a `reports/active_run.log`.
- Metodos principales: `start()`, `read_log()`, `reset()`, `check_rate_limit()`.
- Se usa en: consola de pipeline del dashboard.

### `forecast_chart` / `ops_trajectory_chart` / `distribution_chart` / `sparkline`
- Archivo: `src/retail_forecasting/api/charts.py`
- Que hace: primitivas SVG renderizadas en servidor. `forecast_chart` devuelve ademas la geometria por punto para que el tooltip no recalcule nada en el navegador.
- Recibe: la serie o los puntos ya calculados.
- Devuelve: `SafeString` (o `dict` con `svg` y `points` en `forecast_chart`).
- Se usa en: dashboard, OPS, drift y tabla de SKU.

### `render`
- Archivo: `src/retail_forecasting/api/eda_charts.py`
- Que hace: despacha un payload de `chart_data()` al renderizador SVG correspondiente segun su `type`.
- Recibe:
  - `data`.
- Devuelve: `SafeString`.
- Se usa en: vista `eda`.
- Renderizadores disponibles: `line_dual`, `histogram`, `histogram_dual`, `bar_horizontal`, `bar_group`, `scatter`, `boxplot`, `series_grid`.

## Utilities

### `ensure_directory`
- Archivo: `src/retail_forecasting/utils/io.py`
- Que hace: crea un directorio si no existe.
- Recibe:
  - `path`.
- Devuelve: `Path`.
- Se usa en: carga de datos y reporting.

### `make_run_directory`
- Archivo: `src/retail_forecasting/utils/io.py`
- Que hace: crea una carpeta de ejecucion con timestamp.
- Recibe:
  - `base_dir`;
  - `run_name`.
- Devuelve: `Path`.
- Se usa en: `write_run_artifacts()`.

### `quantile_column_name`
- Archivo: `src/retail_forecasting/utils/io.py`
- Que hace: transforma un cuantil como `0.1` en un nombre de columna como `q_0_1`.
- Recibe:
  - `quantile`.
- Devuelve: `str`.
- Se usa en: modelos y metricas.

### `dataframe_to_markdown`
- Archivo: `src/retail_forecasting/utils/io.py`
- Que hace: convierte un `DataFrame` en una tabla Markdown simple.
- Recibe:
  - `frame`;
  - `columns`.
- Devuelve: `str`.
- Se usa en: `build_markdown_report()`.

## Nota de uso

Si estas estudiando el proyecto para tutorias o defensa, las funciones mas importantes para memorizar primero son:

1. `main`
2. `load_config`
3. `run_experiment`
4. `load_prepared_panel`
5. `build_supervised_frame`
6. `build_walk_forward_folds`
7. `choose_order_quantity`
8. `summarize_predictions`
9. `summarize_costs`
10. `write_run_artifacts`
