# Diseno experimental

## Definicion del problema

El problema se formula como forecasting de demanda para decision de inventario en retail fresco. La salida del sistema no se evaluara solo por precision predictiva, sino por su capacidad para inducir decisiones con menor coste operativo esperado.

En la v2, cada observacion representa una fecha de decision para una serie `store_id x product_id`. El objetivo es predecir la demanda acumulada durante el horizonte de lead time y evaluar decisiones de inventario bajo incertidumbre.

## Unidad de prediccion

- Serie: `store_id x product_id`
- Contexto jerarquico adicional: `city_id` y categorias de producto
- Frecuencia: diaria

## Horizonte temporal

- Horizonte por defecto: 7 dias
- Interpretacion: demanda acumulada desde el dia de decision hasta el final del lead time
- Motivo: mejor alineacion entre forecast y cantidad a pedir en una politica newsvendor simple

## Variable objetivo

Target v2:

- suma de `sale_amount` observada a lo largo del horizonte `h`

Motivo:

- conecta de forma natural forecast y decision de inventario;
- evita una narrativa demasiado estrecha de prediccion `t+1`;
- deja preparada la extension a politicas base-stock o reorder-point.

## Features

### Historicas

- lags de demanda
- rolling mean y rolling sum de demanda
- lags de stockout hours
- lags de descuento
- lags meteorologicos

### Calendario y contexto conocido ex ante

- dia de la semana
- dia del mes
- mes
- semana del ano
- indicador de fin de semana
- holiday flag
- activity flag

### Identificadores estaticos

- `city_id`
- `store_id`
- `product_id`
- niveles jerarquicos de categoria

## Politica anti-leakage

1. No se usara validacion aleatoria.
2. Todas las features historicas se construyen con `shift` positivo, por lo que solo consumen informacion anterior a la fecha de decision.
3. Variables que no son claramente conocidas ex ante, como weather realizado o stock status, entran solo en forma lagged.
4. El conjunto de entrenamiento de cada fold excluye observaciones cuyo target use dias posteriores al inicio del fold de validacion.

## Estrategia de validacion temporal

Se utilizara walk-forward con ventana expansiva:

- `initial_train_days = 56`
- `fold_size_days = 7`
- `n_folds = 3`

Cada fold produce:

- entrenamiento con toda la historia util hasta el corte temporal permitido;
- validacion en el siguiente bloque temporal;
- reentrenamiento completo antes de cada fold si asi lo indica la configuracion.

## Modelos incluidos

### Baseline

- `seasonal naive`

Justificacion:

- baseline interpretable y muy competitivo en series retail;
- ayuda a separar valor real frente a complejidad gratuita.

### Modelo avanzado v2

- boosting global sobre panel temporal

Justificacion:

- explota informacion cruzada entre series;
- soporta mezcla de lags, calendario y covariables;
- permite una ruta razonable hacia forecasting cuantílico si el backend lo soporta.

Backends:

- prioridad: LightGBM
- alternativa: XGBoost para punto + cuantiles con fallback
- fallback universal: scikit-learn

## Metricas predictivas

- MAE
- RMSE

Estas metricas se reportan como diagnostico de ajuste, no como criterio unico de seleccion.

## Metricas probabilisticas

- pinball loss para cuantiles `0.1`, `0.5`, `0.9`
- cobertura del intervalo `[0.1, 0.9]` cuando este disponible

Estas metricas se usan para estudiar calibracion operacional aproximada y calidad de la incertidumbre reportada.

## Metricas economicas

- unidades de sobrestock
- unidades de rotura
- coste de sobrestock
- coste de rotura
- coste total operativo
- frontera de Pareto entre coste economico, sobrestock y rotura

La v2 usa una estructura de costes configurable y puede activar perfiles de coste por serie:

- `c_over`: penalizacion por exceso
- `c_under`: penalizacion por rotura

La decision de pedido sigue:

- `critical fractile = c_under / (c_under + c_over)`

Ademas de la politica newsvendor seleccionada, el sistema evalua politicas
candidatas que escalan la cantidad pedida. Esto permite visualizar el trade-off
multiobjetivo: reducir desperdicio aproximado mediante menos sobrestock suele
aumentar roturas, mientras que mejorar disponibilidad suele aumentar exceso.

## Escenarios con y sin drift

La v2 no incluye aun un detector formal de drift, pero si prepara dos tipos de analisis:

1. comparacion fold a fold para observar degradacion temporal;
2. segmentacion por intensidad de stockout como proxy de cambio de regimen operativo.

En una fase posterior se anadiran detectores explicitos y politicas de reentrenamiento adaptativo.

## Estrategia de reentrenamiento

Politica operacional objetivo:

- reentrenamiento programado con cadencia fija, por ejemplo semanal;
- reentrenamiento adicional cuando el monitor de drift o degradacion lo
  recomiende;
- evaluacion del nuevo modelo como `challenger` antes de sustituir al
  `champion`.

En el backtesting experimental actual, el sistema sigue reentrenando por fold
con ventana expansiva para estimar rendimiento de forma conservadora. Sin
embargo, la interpretacion de negocio es distinta: en operacion, el modelo en
uso debe permanecer estable hasta que exista evidencia suficiente para promover
un sustituto mejor.

Regla de aceptacion recomendada:

1. el challenger debe mejorar `total_cost`;
2. no debe degradar el nivel de servicio por encima del umbral aceptado;
3. no debe llegar acompañado de alertas bloqueantes de calidad de datos o
   monitorizacion.

## Criterios de comparacion final

El ranking principal se establece por:

1. coste total operativo
2. frontera de Pareto coste-sobrestock-rotura para interpretar trade-offs
3. pinball loss y cobertura, cuando existan cuantiles
4. MAE y RMSE como soporte diagnostico

La conclusion experimental buscara identificar no solo que modelo predice mejor, sino que sistema toma mejores decisiones y bajo que condiciones.

## Limitaciones explicitadas

- la demanda observada puede infraestimar demanda real en presencia de stockouts;
- el dataset trabaja con demanda normalizada;
- la v2 incorpora una estrategia heuristica de demanda latente, pero no estima demanda censurada con un modelo causal completo ni lead times variables;
- la evaluacion economica es de una sola etapa, no una simulacion completa multi-periodo;
- la imputación `fillna` de variables climatológicas (`avg_temperature`, `avg_humidity`, `avg_wind_level`) utiliza la mediana calculada sobre el split completo (incluyendo fechas futuras), lo que constituye un data leakage temporal de baja severidad;
- el cálculo de medias de soporte jerárquico para el fallback de arranque en frío (*cold-start*) se calcula sobre el panel completo disponible en el momento de generación del frame de inferencia, pudiendo filtrar información de demanda futura durante los experimentos de backtesting.
