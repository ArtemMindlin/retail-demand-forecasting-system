# Diseno metodologico del sistema

## Vision general

El sistema sigue una arquitectura modular orientada a experimentacion reproducible. Cada modulo responde a una etapa del ciclo de decision:

1. ingesta de datos;
2. preparacion del panel temporal;
3. analisis exploratorio reproducible del panel;
4. feature engineering sin leakage;
5. entrenamiento y forecasting;
6. decision de inventario;
7. evaluacion economica;
8. reporte y visualizacion.

### Flujo end-to-end

```mermaid
flowchart TD
    A["Config YAML<br/>configs/default.yaml"] --> B["load_config()<br/>typed Settings"]
    B --> C["run_experiment()"]

    C --> D["load_prepared_panel()<br/>load cache or prepare FreshRetailNet panel"]
    D --> E["Canonical panel<br/>date<br/>series_id<br/>observed_demand<br/>stockout_hours<br/>context/static ids"]

    E --> F["Strategy A<br/>Observed demand"]
    E --> G["Strategy B<br/>Latent demand"]
    G --> G1["LatentDemandImputer<br/>stockout-aware demand correction"]
    G1 --> G2["Imputed panel<br/>observed_demand replaced by latent_demand_est"]

    F --> H["run_experiment_from_frame(panel, Observed)"]
    G2 --> I["run_experiment_from_frame(panel, Latent_strategy)"]

    H --> J["Observed predictions + metrics"]
    I --> K["Latent predictions + metrics"]

    J --> L["merge strategy predictions"]
    K --> L

    L --> M["summarize_predictions()<br/>MAE, RMSE, pinball, coverage"]
    L --> N["summarize_costs()<br/>economic ranking"]
    L --> O["run_sensitivity_analysis()<br/>cost-ratio scenarios"]
    L --> P["summarize_pareto_frontier()<br/>multiobjective policies"]

    M --> Q["RunArtifacts"]
    N --> Q
    O --> Q
    P --> Q
    L --> Q

    Q --> R["write_run_artifacts()"]
    R --> R1["predictions.csv"]
    R --> R2["metrics_summary.csv"]
    R --> R3["fold_metrics.csv"]
    R --> R4["cost_summary.csv"]
    R --> R5["reorder_recommendations.csv"]
    R --> R6["exceptions.csv"]
    R --> R7["sensitivity_summary.csv"]
    R --> R8["pareto_frontier.csv"]
    R --> R9["report.md"]
    R --> R10["plots when enabled"]
```

### Detalle de `run_experiment_from_frame()`

```mermaid
flowchart TD
    A["Input panel<br/>Observed or Latent strategy"] --> B["label_stockout_regime()"]

    B --> C["build_supervised_frame()<br/>features + target"]
    C --> C1["Temporal features<br/>positive demand lags<br/>shifted rolling windows<br/>lagged stockout/discount/weather<br/>calendar/static ids"]
    C --> C2["target_lead_time_demand<br/>sum demand from t to t+h-1"]

    B --> D{"inventory.use_series_costs?"}
    D -->|true| D1["build_series_cost_profile()<br/>series-level c_over<br/>series-level c_under<br/>critical_fractile"]
    D -->|false| D2["global inventory costs<br/>config c_over / c_under"]

    B --> E["build_walk_forward_folds()<br/>temporal validation"]
    E --> F["Fold loop"]

    F --> F1["train_frame<br/>date <= train_end_date"]
    F --> F2["validation_frame<br/>validation_start_date to validation_end_date"]
    F1 --> F3["calibration split<br/>sub-train + calibration"]

    F3 --> G["Fit or reuse models"]
    G --> G1["Seasonal naive"]
    G --> G2["Auto boosting + ConformalForecaster"]
    G --> G3["CatBoost + ConformalForecaster"]

    F2 --> H["Build prediction frame"]
    G1 --> H
    G2 --> H
    G3 --> H

    H --> H1["Forecast columns<br/>y_pred<br/>q_* if available<br/>model_name<br/>backend_name<br/>fold_id<br/>data_strategy"]

    H1 --> I["choose_order_quantity()"]
    D1 --> I
    D2 --> I

    I --> I1{"Quantiles available?"}
    I1 -->|yes| I2["interpolate forecast quantile<br/>at critical_fractile"]
    I1 -->|no| I3["use point forecast<br/>order_quantity = y_pred"]

    I2 --> J["attach_inventory_costs()"]
    I3 --> J
    D1 --> J
    D2 --> J

    J --> J1["Inventory columns<br/>overstock_units<br/>stockout_units<br/>overstock_cost<br/>stockout_cost<br/>total_cost"]
    J1 --> K["append fold predictions"]

    K --> L["concatenate folds"]
    L --> M["summarize_predictions()"]
    L --> N["summarize_costs()"]
    L --> O["run_sensitivity_analysis()"]
    L --> P["summarize_pareto_frontier()"]

    M --> Q["RunArtifacts for one strategy"]
    N --> Q
    O --> Q
    P --> Q
    L --> Q
```

## 1. Ingesta de datos

Fuente principal:

- `FreshRetailNet-50K` desde Hugging Face

La capa de datos:

- lee solo las columnas necesarias para la version actual;
- cachea el split bruto en `data/raw/`;
- materializa un panel procesado en `data/processed/`.

Esto permite reproducibilidad y evita volver a descargar el dataset en cada ejecucion.

## 2. Preprocesamiento

El preprocesamiento transforma el split bruto en un panel diario limpio:

- parseo de fechas;
- renombrado semantico de columnas;
- eliminacion de duplicados;
- filtrado de ventas negativas si aparecen;
- construccion de `series_id = store_id + product_id`;
- filtrado de series con historia suficiente;
- seleccion configurable de las series mas relevantes por volumen.

La v2 compara una estrategia de demanda observada con una estrategia heuristica de demanda latente. Los stockouts se mantienen como senal contextual y como insumo para el analisis operativo.

## 3. EDA reproducible

El modulo `eda` opera exclusivamente sobre el panel preparado y no sobre nombres raw del dataset. Su objetivo es producir artefactos de analisis descriptivo auditables sin contaminar la logica de forecasting:

- resumen de cobertura temporal y continuidad por serie;
- perfilado tabular de missingness y variables numericas;
- resumenes de estacionalidad semanal;
- diagnosticos de frecuencia e intensidad de stockout;
- correlaciones descriptivas con `observed_demand`;
- reporte Markdown y plots bajo `reports/eda_*`.

## 4. Feature engineering temporal

### Principios

- ninguna feature puede usar informacion posterior a la fecha de decision;
- las covariables no observables ex ante solo entran como historico;
- el target se alinea con el horizonte de inventario.

### Features incluidas

- lags de demanda
- rolling mean, std y sum de demanda historica
- lags de descuento
- lags de stockout hours
- lags meteorologicos
- calendario
- flags de actividad y festivo
- ids estaticos de tienda y producto

### Target

Para cada fecha `t`, el target es la suma de demanda observada en el horizonte `t ... t+h-1`.

## 5. Entrenamiento

El pipeline soporta dos familias de modelo:

- `seasonal naive` como baseline;
- boosting global para prediccion puntual y cuantiles.

El modelo global se entrena sobre el panel completo filtrado, no serie a serie. Esto mejora eficiencia y permite compartir informacion entre combinaciones de tienda y producto.

## 6. Forecasting probabilistico

La v2 implementa forecasting probabilistico minimo mediante cuantiles `0.1`, `0.5` y `0.9`.

Logica de backends:

1. usar LightGBM si esta instalado;
2. si no, usar XGBoost para punto y fallback para cuantiles;
3. si tampoco esta disponible, usar `scikit-learn`.

Los cuantiles se fuerzan a ser monotonicamente no decrecientes para evitar incoherencias basicas.

## 7. Deteccion de drift

La v2 no implementa aun un detector estadistico completo, pero deja preparado el modulo `drift` para:

- comparar rendimiento por fold temporal;
- segmentar resultados por regimen operativo;
- incorporar detectores posteriores sin reescribir el pipeline.

Como aproximacion inicial, se pueden etiquetar regimenes de alta y baja intensidad de stockout para inspeccionar cambios de comportamiento.

## 8. Reentrenamiento

La interpretacion operacional del proyecto no es ``entrenar siempre el modelo
mas reciente'', sino mantener un `champion` estable y evaluar cada nuevo modelo
como `challenger`.

Politica objetivo:

- `champion`: modelo vigente para generar recomendaciones diarias;
- `challenger`: candidato entrenado con datos mas recientes o activado por
  drift;
- promocion: solo si mejora `total_cost`, mantiene el nivel de servicio dentro
  del umbral aceptado y no presenta fallos de calidad o monitorizacion.

En el estado actual del repo, el backtesting reentrena por fold para estimar
rendimiento bajo distintas fechas de decision. Esa logica experimental no debe
confundirse con la politica de produccion: el objetivo de negocio es un ciclo
de recomendacion diario y un ciclo de promocion de modelos mucho mas
gobernado.

## 9. Ciclo operativo diario

El sistema se concibe como un flujo batch diario:

1. se cierra el dato del dia `t`;
2. se ejecuta la validacion de calidad de datos;
3. el modelo `champion` genera recomendaciones de pedido para `t+1`;
4. se escriben `reorder_recommendations.csv` y `exceptions.csv`;
5. el responsable de reposicion revisa solo los SKUs marcados;
6. las recomendaciones aprobadas se exportan al sistema de compras;
7. la demanda realizada posterior alimenta monitorizacion y futuros
   reentrenamientos.

Este diseno evita depender de serving online y encaja mejor con un problema de
reposicion diaria en retail fresco.

La estrategia experimental sigue usando walk-forward con ventana expansiva y
reentrenamiento por fold para medir rendimiento con rigor temporal. Esa
eleccion sirve para validacion, pero no sustituye la politica operacional: el
modelo champion debe cambiar solo cuando un challenger supera los criterios de
promocion definidos.

## 10. Simulacion / decision de inventario

La v2 implementa una capa de decision newsvendor por periodo:

- si solo hay forecast puntual, la cantidad pedida coincide con la prediccion puntual;
- si hay cuantiles, la cantidad pedida se aproxima al fractil critico definido por la estructura de costes.

Esta capa es deliberadamente simple, pero ya convierte forecast en accion y permite medir impacto economico.

## 11. Evaluacion economica

Para cada prediccion:

- `overstock = max(order_qty - actual_demand, 0)`
- `stockout = max(actual_demand - order_qty, 0)`
- `total_cost = c_over * overstock + c_under * stockout`

Esto permite comparar modelos desde una perspectiva operativa. Un modelo con mejor MAE puede ser peor en coste si se equivoca en la direccion economicamente importante.

La evaluacion tambien genera una frontera de Pareto multiobjetivo. Para cada
modelo se crean politicas candidatas escalando `order_quantity` y se resumen
coste economico, sobrestock, rotura, nivel de servicio y fill rate. Los puntos
Pareto-eficientes muestran politicas no dominadas y hacen explicito el conflicto
entre disponibilidad y desperdicio aproximado.

## 12. Visualizacion de resultados

Cada corrida genera:

- tablas de metricas predictivas;
- tablas de coste agregado;
- frontera de Pareto de politicas candidatas;
- predicciones por fold;
- `backtest_metadata.json` con dataset, features, folds, modelos, hash de configuracion y commit Git;
- graficos simples de coste y trade-off error-coste;
- un reporte Markdown final en `reports/`.

## 13. Como se evita usar informacion futura

La politica anti-leakage es un requisito central del sistema:

1. el target siempre representa el futuro respecto a la fecha de decision;
2. las features historicas usan `shift`;
3. las covariables no disponibles ex ante se usan solo en lag;
4. el entrenamiento de cada fold excluye ejemplos cuyo target se solape con el periodo de validacion.

Esto asegura que el rendimiento estimado sea defendible academicamente.

## 14. Evolucion futura prevista

La arquitectura queda lista para:

- demanda latente o censurada;
- conformal prediction;
- recalibracion de cuantiles;
- detectores de drift;
- reentrenamiento adaptativo;
- base-stock o reorder point;
- dashboard con Streamlit.
