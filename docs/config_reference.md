# Referencia de configuracion

Este documento explica las variables definidas en
`src/retail_forecasting/config.py` y usadas por los YAML de `configs/`.

El flujo general es:

```text
YAML
 -> load_config()
 -> objetos dataclass
 -> Settings
 -> validate_settings()
 -> run_experiment(settings)
```

`Settings` es el objeto principal. Agrupa:

- `project`
- `dataset`
- `preprocessing`
- `features`
- `validation`
- `data_quality`
- `models`
- `inventory`
- `reporting`

## `ProjectConfig`

### `random_seed`

Semilla global de aleatoriedad.

Sirve para que modelos y procesos estocasticos sean mas reproducibles entre
ejecuciones.

Ejemplo:

```yaml
project:
  random_seed: 42
```

Explicacion oral:

> `ProjectConfig` contiene parametros globales del experimento. Ahora mismo
> guarda la semilla aleatoria, que permite que modelos y tuning sean mas
> reproducibles.

### `run_mode`

Modo de ejecucion principal del pipeline.

Valores permitidos:

```yaml
run_mode: backtest
```

o:

```yaml
run_mode: retrain
```

o:

```yaml
run_mode: score_daily
```

Semantica:

- `backtest`: conserva el conjunto completo de artefactos experimentales;
- `retrain`: mantiene el mismo conjunto rico de artefactos, pero representa un
  ciclo de reentrenamiento gobernado operacionalmente;
- `score_daily`: escribe solo artefactos operativos de negocio y metadata
  ligera, sin `report.md`, sin `predictions.csv` y sin `backtest_metadata.json`.

## `DatasetConfig`

Controla de donde salen los datos, como se cachean y que subconjunto se usa.

### `source`

Fuente logica del dataset.

Actualmente solo se acepta:

```yaml
source: fresh_retailnet
```

Se valida porque el pipeline solo soporta `FreshRetailNet-50K`.

### `hf_dataset_id`

Identificador del dataset remoto en Hugging Face.

Ejemplo:

```yaml
hf_dataset_id: Dingdong-Inc/FreshRetailNet-50K
```

Se usa para construir una ruta remota como:

```text
hf://datasets/Dingdong-Inc/FreshRetailNet-50K/data/train.parquet
```

## `DataQualityConfig`

Controla las comprobaciones runtime que se ejecutan antes del pipeline de
forecasting e inventario.

### `max_missing_fraction_warning`

Umbral de missingness a partir del cual una columna genera un warning.

Ejemplo:

```yaml
max_missing_fraction_warning: 0.05
```

Interpretacion:

- `0.05` significa avisar cuando una columna supera el 5% de valores nulos;
- no bloquea por si mismo la corrida;
- debe estar entre `0` y `1`.

### `max_data_age_days`

Limite opcional de antiguedad del dato mas reciente para `score_daily` y
`retrain`.

Ejemplo:

```yaml
max_data_age_days: 2
```

Interpretacion:

- si es `null`, no se aplica control de frescura temporal;
- si tiene valor, una corrida operacional falla cuando el `date` maximo del
  panel supera ese numero de dias respecto a la fecha actual.

### `splits`

Mapa entre nombres logicos de splits y rutas parquet dentro del dataset remoto.

Ejemplo:

```yaml
splits:
  train: data/train.parquet
  eval: data/eval.parquet
```

Cuando el codigo pide `split="train"`, usa `splits["train"]`.

### `local_cache_dir`

Directorio donde se guarda la cache raw local descargada de Hugging Face.

Ejemplo:

```yaml
local_cache_dir: data/raw/fresh_retailnet
```

Puede contener archivos como:

```text
data/raw/fresh_retailnet/train.parquet
```

### `processed_panel_path`

Ruta del panel procesado.

Ejemplo:

```yaml
processed_panel_path: data/processed/fresh_retailnet_train.parquet
```

Este panel ya usa nombres canonicos del proyecto como `date`, `series_id`,
`observed_demand` y `stockout_hours`.

### `use_local_cache`

Controla la cache raw.

- `true`: si existe el parquet raw local, lo reutiliza; si no existe, descarga
  de Hugging Face y lo guarda.
- `false`: lee de Hugging Face cuando necesita raw, sin guardar cache raw.

No es lo mismo que la cache procesada.

### `refresh_processed_cache`

Controla la cache del panel procesado.

- `false`: si existe `processed_panel_path`, lo reutiliza.
- `true`: reconstruye el panel procesado desde raw y sobrescribe el parquet.

Conviene ponerlo en `true` si cambias parametros que afectan al panel, como
`top_n_series`, `min_history_days` o reglas de preprocesamiento.

### `top_n_series`

Numero de series `store_id x product_id` que se conservan, ordenadas por volumen
total de demanda observada.

Ejemplo:

```yaml
top_n_series: 50
```

Reduce tiempo de ejecucion y centra el experimento en las series mas relevantes.

### `min_history_days`

Minimo de dias unicos que debe tener una serie para conservarse.

Ejemplo:

```yaml
min_history_days: 70
```

Evita series demasiado cortas, que no permiten construir lags, ventanas rolling
o folds temporales fiables.

### `max_rows`

Limite opcional de filas raw leidas.

Ejemplo:

```yaml
max_rows: null
```

Puede usarse para pruebas rapidas, pero para experimentos serios debe quedar en
`null`, porque tomar las primeras filas puede sesgar el panel.

### `horizon`

Horizonte de prediccion e inventario en dias.

Ejemplo:

```yaml
horizon: 7
```

El target es demanda acumulada:

```text
target_lead_time_demand(t, h) = demand[t] + ... + demand[t + h - 1]
```

Tambien afecta a la validacion: el entrenamiento debe terminar al menos
`horizon` dias antes de la validacion para evitar leakage.

### `use_eval_as_holdout`

Indica si se usa el split oficial `eval` como holdout.

Actualmente debe ser:

```yaml
use_eval_as_holdout: false
```

Esta bloqueado hasta verificar la semantica temporal del split `eval`.

## `PreprocessingConfig`

Controla limpieza del panel y estrategia de demanda latente.

### `drop_negative_sales`

Si es `true`, elimina filas con demanda observada negativa.

Las ventas negativas pueden representar devoluciones, correcciones o anomalias.

### `fill_missing_values`

Si es `true`, rellena valores faltantes en variables contextuales.

Actualmente:

- `holiday_flag`, `activity_flag`, `precpt` y `stockout_hours` se rellenan con
  `0.0`;
- `discount` se rellena con `1.0`;
- variables meteorologicas se rellenan con la mediana.

### `imputation_strategy`

Estrategia usada por `LatentDemandImputer`.

Opciones implementadas:

- `supervised`
- `historical_mean`
- `clipped_scaling`
- `none`

Con `supervised`, se entrena un LightGBM sobre dias sin stockout y se predice
demanda latente para dias censurados por stockout.

## `FeatureConfig`

Controla las features creadas por `build_supervised_frame()`.

### `lags`

Lags positivos usados para features historicas.

Ejemplo:

```yaml
lags: [1, 7, 14, 28]
```

Generan columnas como `demand_lag_1`, `discount_lag_7` o `stockout_lag_14`.
Solo se permiten lags positivos para evitar leakage.

### `rolling_windows`

Ventanas rolling historicas.

Ejemplo:

```yaml
rolling_windows: [7, 28]
```

Generan columnas como `demand_roll_mean_7`, `demand_roll_sum_28`,
`demand_roll_std_28` o `stockout_roll_mean_7`.

### `include_static_ids`

Si es `true`, incluye identificadores estaticos como features:

- `city_id`
- `store_id`
- `management_group_id`
- `first_category_id`
- `second_category_id`
- `third_category_id`
- `product_id`

Son utiles porque el modelo global entrena muchas series a la vez y necesita
distinguir tiendas, productos y categorias.

### `include_weather_lags`

Si es `true`, incluye lags de variables meteorologicas:

- `precpt_lag_*`
- `avg_temperature_lag_*`
- `avg_humidity_lag_*`
- `avg_wind_level_lag_*`

Se usan como lags porque el tiempo realizado no necesariamente se conoce en la
fecha de decision.

### `include_discount_lags`

Si es `true`, incluye lags de descuento, como `discount_lag_1` o
`discount_lag_7`.

### `include_stockout_lags`

Si es `true`, incluye lags y medias rolling de stockout:

- `stockout_lag_*`
- `stockout_roll_mean_*`

Permite usar informacion historica de roturas sin usar informacion futura.

## `ValidationConfig`

Controla la validacion temporal walk-forward.

### `initial_train_days`

Numero de dias iniciales antes del primer fold de validacion.

### `n_folds`

Numero maximo de folds temporales.

### `fold_size_days`

Duracion de cada bloque de validacion en dias.

Con `n_folds=3` y `fold_size_days=7`, se validan tres bloques semanales.

### `retrain_each_fold`

Si es `true`, los modelos se reentrenan en cada fold con la historia disponible.

Si es `false`, pueden reutilizarse modelos entre folds salvo que otro mecanismo
fuerce reentrenamiento.

### `drift_triggered_retrain`

Si es `true`, una alerta de drift fuerza reentrenamiento en el siguiente fold.

Matiz actual:

- si `retrain_each_fold=true`, este flag es casi redundante;
- si el drift se detecta en el ultimo fold, no cambia metricas porque no hay un
  fold posterior.

## `DriftConfig`

Controla la vigilancia online del rendimiento durante el backtesting.

### `threshold`

Umbral del detector Page-Hinkley. Valores mas altos lo hacen mas conservador.

### `delta`

Pequena tolerancia para ignorar fluctuaciones menores del error monitorizado.

### `min_instances`

Numero minimo de observaciones antes de permitir alertas de drift.

## `ModelConfig`

Controla cuantiles, estacionalidad, hiperparametros de boosting, tuning y
entrenamiento orientado a coste.

### `quantiles`

Cuantiles probabilisticos a predecir.

Ejemplo:

```yaml
quantiles: [0.1, 0.5, 0.9]
```

Generan columnas como:

```text
q_0_1
q_0_5
q_0_9
```

Se usan para pinball loss, cobertura, prediccion probabilistica y decision
newsvendor.

### `seasonal_period`

Periodo estacional para modelos baseline/estadisticos.

En datos diarios:

```yaml
seasonal_period: 7
```

representa estacionalidad semanal.

### `n_estimators`

Numero de arboles o iteraciones en modelos boosting.

Si `use_tuning=false`, es el valor efectivo. Si `use_tuning=true`, Optuna puede
sobrescribirlo. Si Optuna no puede entrenar por falta de datos en su split
interno, se usa como fallback.

### `learning_rate`

Tasa de aprendizaje del boosting.

Valores bajos suelen ser mas estables pero requieren mas arboles. Valores altos
son mas rapidos pero pueden sobreajustar.

### `max_depth`

Profundidad maxima de los arboles.

Controla complejidad, memoria, runtime y riesgo de overfitting.

### `use_tuning`

Si es `true`, ejecuta Optuna para buscar hiperparametros de boosting antes del
backtest.

### `tuning_trials`

Numero de pruebas de Optuna cuando `use_tuning=true`.

Mas trials pueden encontrar mejores parametros, pero aumentan runtime.

### `optimize_for_cost`

Si es `true`, `LightGBMModel` entrena su prediccion puntual como un cuantil
en la fractil critica newsvendor:

```text
critical_fractile = stockout_cost / (stockout_cost + overstock_cost)
```

Con `stockout_cost=4` y `overstock_cost=1`, la fractil es `0.8`. Esto orienta
la prediccion a la decision economica de inventario.

## `InventoryConfig`

Controla la decision newsvendor y la evaluacion economica.

### `overstock_cost`

Coste unitario global de pedir de mas.

### `stockout_cost`

Coste unitario global de quedarse corto.

Junto con `overstock_cost`, define la fractil critica:

```text
c_under / (c_under + c_over)
```

### `use_series_costs`

Si es `false`, todas las series usan los costes globales.

Si es `true`, se construyen costes por `series_id`:

- `c_over`
- `c_under`
- `critical_fractile`

Esto permite decisiones heterogeneas por producto-tienda.

### `series_cost_strategy`

Estrategia para construir costes por serie.

Actualmente solo se permite:

```yaml
series_cost_strategy: synthetic_series
```

`synthetic_series` construye costes heuristicos usando proxies del panel, como
intensidad de demanda, intermitencia, stockouts y categoria de producto.

### `synthetic_cost_config`

Parametros de la heuristica `synthetic_series`.

Ejemplo:

```yaml
synthetic_cost_config:
  perishability_weights: [0.5, 0.3, 0.2]
  slow_moving_weights: [0.6, 0.4]
  criticality_weights: [0.7, 0.3]
  perishability_base: 0.8
  perishability_multiplier: 0.8
  slow_moving_base: 0.9
  slow_moving_multiplier: 0.5
  service_criticality_base: 0.9
  service_criticality_multiplier: 0.5
```

Los pesos deben ser no negativos y sumar aproximadamente `1.0` dentro de cada
dimension. Las bases deben ser positivas y los multiplicadores no negativos.

Interpretacion:

- `perishability_weights` combina inestabilidad de categoria, variabilidad de
  categoria e intermitencia de la serie para ajustar `c_over`.
- `slow_moving_weights` combina intermitencia y baja intensidad de demanda para
  ajustar `c_over`.
- `criticality_weights` combina intensidad de demanda y tension historica de
  stockout para ajustar `c_under`.
- Las bases y multiplicadores controlan cuanto se separan los costes por serie
  de los costes globales `overstock_cost` y `stockout_cost`.

### `clip_negative_orders`

Si es `true`, cantidades de pedido negativas se recortan a cero.

Evita decisiones de inventario invalidas si un modelo genera predicciones
negativas.

### `pareto_order_scales`

Escalas candidatas para analisis de frontera de Pareto.

Ejemplo:

```yaml
pareto_order_scales: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
```

Evalua politicas como:

```text
0.8 * order_quantity
1.0 * order_quantity
1.2 * order_quantity
```

y busca politicas no dominadas en coste, sobrestock y stockout.

## `BusinessConfig`

Controla la politica de revision manual de recomendaciones de reposicion.

Esta capa no cambia el forecast ni la logica del newsvendor. Su funcion es
marcar recomendaciones para revision humana en los artefactos
`reorder_recommendations.csv` y `exceptions.csv`.

### `flag_cold_start`

Si es `true`, filas con `prediction_source = cold_start_fallback` se etiquetan
con `risk_flag = cold_start`.

### `flag_drift_watch`

Si es `true`, filas pertenecientes a folds marcados por el detector de drift se
etiquetan con `risk_flag = drift_watch`, salvo que ya tengan un flag anterior.

### `flag_high_uncertainty`

Si es `true`, recomendaciones con intervalos de prediccion especialmente anchos
se marcan con `risk_flag = high_uncertainty`.

### `high_uncertainty_interval_quantile`

Cuantil usado para definir el umbral de anchura del intervalo.

Ejemplo:

```yaml
high_uncertainty_interval_quantile: 0.95
```

Interpretacion:

- `0.95` significa marcar el 5% superior de anchuras de intervalo dentro de la
  corrida;
- debe estar estrictamente entre `0` y `1`.

### `flag_extreme_order_quantity`

Si es `true`, recomendaciones con cantidades de pedido extremas se marcan con
`risk_flag = extreme_order_quantity`.

### `extreme_order_quantity_quantile`

Cuantil usado para definir el umbral de cantidad extrema.

Ejemplo:

```yaml
extreme_order_quantity_quantile: 0.99
```

Interpretacion:

- `0.99` significa marcar el 1% superior de cantidades de pedido dentro de la
  corrida;
- debe estar estrictamente entre `0` y `1`.

### `champion_data_strategy`

Estrategia actualmente tratada como `champion` para la politica de promocion.

Ejemplo:

```yaml
champion_data_strategy: Observed
```

Si es `null`, la seleccion del champion ignora `data_strategy` y se apoya solo
en `champion_model_name` y `champion_backend_name`.

### `champion_model_name`

Nombre del modelo actualmente desplegado como `champion`.

Ejemplo:

```yaml
champion_model_name: catboost
```

### `champion_backend_name`

Backend asociado al `champion`.

Ejemplo:

```yaml
champion_backend_name: conformal_catboost_official
```

### `champion_min_cost_improvement_pct`

Mejora porcentual minima en `total_cost` que un `challenger` debe lograr para
ser promocionable.

Ejemplo:

```yaml
champion_min_cost_improvement_pct: 5.0
```

Interpretacion:

- `5.0` significa que el challenger debe reducir `total_cost` al menos un 5%
  respecto al champion;
- el valor debe ser mayor o igual que `0`.

### `champion_max_service_level_degradation`

Degradacion maxima permitida en `service_level` al comparar un `challenger`
contra el `champion`.

Ejemplo:

```yaml
champion_max_service_level_degradation: 0.02
```

Interpretacion:

- `0.02` permite perder hasta dos puntos porcentuales de `service_level`;
- el valor debe estar entre `0` y `1`.

## `ReportingConfig`

Controla los artefactos generados.

### `output_dir`

Directorio base de salida.

Ejemplo:

```yaml
output_dir: reports
```

`validate_settings()` evita escribir dentro de `data/`, porque `data/` se
reserva para datasets y caches.

### `run_name`

Prefijo del directorio de ejecucion.

Ejemplo:

```yaml
run_name: fresh_retailnet_v2
```

Produce carpetas como:

```text
reports/fresh_retailnet_v2_YYYYMMDD_HHMMSS/
```

### `make_plots`

Si es `true`, genera plots estandar ademas de CSVs y `report.md`.

Si es `false`, solo escribe artefactos tabulares y Markdown.

## Reglas de validacion

`validate_settings()` bloquea configuraciones que romperian las invariantes del
experimento, por ejemplo:

- dataset no soportado;
- `use_eval_as_holdout=true`;
- horizonte, folds, lags, ventanas, costes o trials no positivos;
- cuantiles vacios, duplicados, desordenados o fuera de `(0, 1)`;
- `reporting.output_dir` dentro de `data/`.

La idea es fallar antes de ejecutar un experimento largo con semantica invalida.
