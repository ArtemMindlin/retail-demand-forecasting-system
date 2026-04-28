# Propuesta de TFG

## Titulo definitivo

Forecasting probabilístico bajo condiciones de stockout: un enfoque de decisión de inventario para retail.

## Resumen ejecutivo

En el sector retail, las roturas de stock (*stockouts*) actúan como un mecanismo de censura de datos, impidiendo observar la demanda real y sesgando los modelos de previsión tradicionales. Este Trabajo de Fin de Grado propone un sistema de forecasting probabilístico diseñado específicamente para operar en estos entornos de incertidumbre. Mediante el uso de modelos globales de *Machine Learning* y técnicas de recuperación de demanda latente, el sistema permite cuantificar el riesgo de falta de existencias y optimizar la cantidad a pedir basándose en el impacto económico real. La evaluación del sistema demuestra que priorizar la robustez ante stockouts y la calibración probabilística reduce significativamente el coste operativo total, superando los enfoques basados exclusivamente en la reducción del error predictivo puntual.

## Motivacion

El forecasting de demanda tiene valor real cuando se convierte en una decision. En retail, esa decision suele estar asociada a cuanto pedir, cuanto reponer y como gestionar el equilibrio entre sobrestock y rotura. FreshRetailNet-50K es especialmente adecuado para estudiar este problema porque incorpora stockouts organicos, estructura jerarquica de producto y tienda, y covariables de contexto relevantes para demanda perecedera.

## Pregunta de investigacion

Puede un sistema de forecasting probabilistico, evaluado desde una perspectiva economica y no solo predictiva, generar mejores decisiones de inventario que un sistema optimizado unicamente para error medio?

## Hipotesis

1. Un modelo con mejor calibracion probabilistica puede inducir decisiones de inventario con menor coste total, incluso si no obtiene el mejor MAE o RMSE.
2. La evaluacion economica puede alterar el ranking de modelos respecto al ranking obtenido con metricas predictivas clasicas.
3. La adaptacion ante cambios del entorno, como cambios de patron, intensidad promocional o episodios de stockout, puede reducir el coste operativo agregado.

## Objetivo general

Disenar y evaluar un sistema reproducible de forecasting para decisiones de inventario en retail fresco, centrado en incertidumbre, coste operativo y robustez temporal.

## Objetivos especificos

1. Construir un pipeline reproducible para ingesta, preparacion y validacion temporal de series de demanda retail.
2. Implementar un baseline interpretable y un modelo global de machine learning con soporte minimo para cuantiles.
3. Traducir la salida del forecast a una decision de inventario mediante una politica newsvendor de una sola etapa.
4. Comparar modelos con metricas predictivas, probabilisticas y economicas.
5. Documentar explicitamente como se evita leakage temporal y por que la evaluacion economica puede cambiar la conclusion experimental.
6. Dejar una base preparada para extensiones posteriores: demanda latente, conformal prediction, drift detection y politicas de inventario mas ricas.

## Contribucion esperada

La contribucion del TFG no sera proponer el modelo mas sofisticado, sino demostrar una formulacion experimental mas alineada con decision real. Se espera aportar:

- un pipeline modular y reproducible sobre FreshRetailNet-50K;
- una evaluacion centrada en impacto operativo;
- una discusion clara sobre demanda observada frente a demanda censurada;
- una base tecnica preparada para evolucionar a forecasting probabilistico avanzado y adaptacion al drift.

## Metodologia propuesta

### 1. Formulacion del problema

Se modelara la demanda observada agregada por dia para cada combinacion `store_id x product_id`. La variable objetivo de la v1 sera la demanda acumulada durante el horizonte de lead time, de modo que el forecast este alineado con la decision de inventario.

### 2. Datos

Se utilizara FreshRetailNet-50K como dataset principal. La informacion de stockouts no se empleara para reconstruir aun la demanda latente en la v1, pero si se incorporara como contexto historico y como elemento de analisis de robustez.

### 3. Modelos

Se incluiran:

- un baseline seasonal naive;
- un modelo global de boosting sobre panel temporal;
- soporte minimo para cuantiles `0.1`, `0.5` y `0.9`.

### 4. Validacion

La validacion sera exclusivamente temporal, mediante walk-forward con ventanas expansivas. No se usara validacion aleatoria. Los folds se construiran de forma que ninguna muestra de entrenamiento use targets que incluyan informacion posterior al inicio del fold de validacion.

### 5. Evaluacion

Se compararan:

- metricas predictivas: MAE y RMSE;
- metricas probabilisticas: pinball loss y cobertura de intervalo si aplica;
- metricas economicas: coste de sobrestock, coste de rotura y coste total.

### 6. Decision de inventario

La capa de decision de la v1 sera un newsvendor por periodo. La cantidad pedida se obtendra desde el forecast puntual o, cuando existan cuantiles, desde el fractil critico inducido por la razon de costes.

## Riesgos y limitaciones

1. La v1 trabaja con demanda observada y no recupera demanda latente, por lo que el impacto de stockouts puede sesgar parte de la evaluacion.
2. `sale_amount` esta normalizada globalmente, asi que los costes se expresaran en unidades relativas y no monetarias absolutas.
3. El horizonte temporal de 90 dias por serie limita analisis de largo plazo, aunque sigue siendo suficiente para una primera evaluacion temporal rigurosa.
4. Los quantiles de la v1 seran funcionales pero no necesariamente calibrados al nivel de una solucion avanzada con conformal prediction.

## Plan de trabajo por fases

### Fase 1. Base metodologica y tecnica

- definir pregunta de investigacion y formulacion experimental;
- crear estructura del proyecto;
- configurar pipeline reproducible;
- redactar propuesta y diseno metodologico.

### Fase 2. Implementacion v1

- ingesta y preparacion de FreshRetailNet-50K;
- feature engineering temporal;
- baseline seasonal naive;
- modelo global de boosting;
- backtesting walk-forward;
- evaluacion economica y reporte.

### Fase 3. Analisis experimental

- analizar divergencias entre ranking por error y ranking por coste;
- estudiar sensibilidad a costes y periodos con stockout;
- documentar limitaciones y lecciones del sistema.

### Fase 4. Extension para sobresaliente

- quantile forecasting mas robusto;
- conformal prediction;
- demanda censurada o latent demand recovery;
- deteccion de drift y reentrenamiento adaptativo;
- dashboard para analisis interactivo.
