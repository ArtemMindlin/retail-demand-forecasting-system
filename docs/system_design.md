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

## 1. Ingesta de datos

Fuente principal:

- `FreshRetailNet-50K` desde Hugging Face

La capa de datos:

- lee solo las columnas necesarias para la v1;
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

La v1 usa demanda observada. Los stockouts no se corrigen aun para recuperar demanda latente; se mantienen como senal contextual.

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

La v1 implementa forecasting probabilistico minimo mediante cuantiles `0.1`, `0.5` y `0.9`.

Logica de backends:

1. usar LightGBM si esta instalado;
2. si no, usar XGBoost para punto y fallback para cuantiles;
3. si tampoco esta disponible, usar `scikit-learn`.

Los cuantiles se fuerzan a ser monotonicamente no decrecientes para evitar incoherencias basicas.

## 7. Deteccion de drift

La v1 no implementa aun un detector estadistico completo, pero deja preparado el modulo `drift` para:

- comparar rendimiento por fold temporal;
- segmentar resultados por regimen operativo;
- incorporar detectores posteriores sin reescribir el pipeline.

Como aproximacion inicial, se pueden etiquetar regimenes de alta y baja intensidad de stockout para inspeccionar cambios de comportamiento.

## 8. Reentrenamiento

La estrategia inicial es walk-forward con ventana expansiva y reentrenamiento por fold. Esto refleja un proceso operativo sencillo:

- acumular datos historicos;
- reentrenar antes de cada nuevo bloque de decision;
- evaluar si el rendimiento y el coste se sostienen a lo largo del tiempo.

El diseno permite anadir despues:

- politicas de reentrenamiento fijo;
- reentrenamiento gatillado por drift;
- ventanas deslizantes.

## 9. Simulacion / decision de inventario

La v1 implementa una capa de decision newsvendor por periodo:

- si solo hay forecast puntual, la cantidad pedida coincide con la prediccion puntual;
- si hay cuantiles, la cantidad pedida se aproxima al fractil critico definido por la estructura de costes.

Esta capa es deliberadamente simple, pero ya convierte forecast en accion y permite medir impacto economico.

## 10. Evaluacion economica

Para cada prediccion:

- `overstock = max(order_qty - actual_demand, 0)`
- `stockout = max(actual_demand - order_qty, 0)`
- `total_cost = c_over * overstock + c_under * stockout`

Esto permite comparar modelos desde una perspectiva operativa. Un modelo con mejor MAE puede ser peor en coste si se equivoca en la direccion economicamente importante.

## 11. Visualizacion de resultados

Cada corrida genera:

- tablas de metricas predictivas;
- tablas de coste agregado;
- predicciones por fold;
- graficos simples de coste y trade-off error-coste;
- un reporte Markdown final en `reports/`.

## 12. Como se evita usar informacion futura

La politica anti-leakage es un requisito central del sistema:

1. el target siempre representa el futuro respecto a la fecha de decision;
2. las features historicas usan `shift`;
3. las covariables no disponibles ex ante se usan solo en lag;
4. el entrenamiento de cada fold excluye ejemplos cuyo target se solape con el periodo de validacion.

Esto asegura que el rendimiento estimado sea defendible academicamente.

## 13. Evolucion futura prevista

La arquitectura queda lista para:

- demanda latente o censurada;
- conformal prediction;
- recalibracion de cuantiles;
- detectores de drift;
- reentrenamiento adaptativo;
- base-stock o reorder point;
- dashboard con Streamlit.
