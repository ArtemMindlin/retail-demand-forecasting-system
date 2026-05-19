# Guión TFG — Explicación del sistema

---

## Bloque 1: Datos

El proyecto empieza con la importación de datos desde **HuggingFace**. El dataset se llama **FreshRetailNet-50K** y está muy bien estructurado: incluye variables que permiten inferir no solo la demanda observada sino también la demanda latente.

En cuanto a tamaño, tiene aproximadamente **4,5 millones de filas de entrenamiento** y **300.000 filas de evaluación**, correspondientes a **300 pares tienda-producto** medidos durante **97 días**.

A partir de esos datos brutos se construye un **panel diario por SKU-tienda**, donde cada fila representa un día de actividad de un par tienda-producto concreto.

### Por qué hace falta imputar la demanda latente

Cuando un producto se queda sin stock, las ventas observadas son 0 (o muy bajas), pero la demanda real no era 0 — simplemente no se podía satisfacer. Si entrenas el modelo directamente con ventas observadas, aprende que "cuando hay stockout, la gente no quiere el producto", lo cual es falso. Esto se llama **sesgo de censura**: el dato está censurado por la restricción de inventario, no por falta de demanda.

El dataset FreshRetailNet incluye `stockout_hours` — cuántas horas del día el SKU estuvo sin stock. El imputador usa esa señal para estimar cuánta demanda adicional había en esos días y corregir la serie. El modelo entonces aprende sobre demanda real, no sobre ventas condicionadas a disponibilidad.

**Para el tutor:** *"El primer problema que resuelve el sistema es que los datos de ventas en retail están sistemáticamente sesgados a la baja: no puedes vender lo que no tienes. Si no corriges ese sesgo antes de entrenar, el modelo subestima la demanda y el ciclo se retroalimenta — pides poco, tienes más stockouts, el modelo aprende que la demanda es baja, y pide aún menos."*

---

## Bloque 2: Feature Engineering (sin leakage temporal)

Tras la importación de datos y la construcción de la demanda latente, comienza el **Feature Engineering**. Se construyen variables lag, rolling windows, medias de weather, discount y de stockout, que permiten que la inferencia de cada día tenga un amplio contexto del pasado.

El principio central de este bloque es **no usar información del futuro** (sin leakage temporal). Cada feature solo puede mirar hacia atrás respecto a la fecha de decisión: los lags usan `.shift()` para desplazar la serie en el tiempo, y las rolling windows solo acumulan datos anteriores al momento de predicción. Si se violara esto, el modelo aprendería con información que en producción no existiría, y las métricas en backtesting serían irrealmente buenas.

Las familias de features son:

- **Lags de demanda** — demanda de los últimos N días de esa serie
- **Rolling windows** — media, std y suma de demanda en ventanas de 7, 14, 28 días
- **Calendario** — día de la semana, mes, festivos, flags de actividad
- **Weather** — medias históricas de temperatura/condiciones meteorológicas
- **Discount** — lags del nivel de descuento aplicado
- **Stockout** — lags de horas de rotura, para que el modelo sepa cuándo el histórico está censurado

**Para el tutor:** *"La política anti-leakage no es un detalle de implementación, es un requisito de validez. Un modelo que en entrenamiento ve el futuro da estimaciones de rendimiento que nunca se reproducirán en producción. Aquí está explícitamente documentado y testeado."*

---

## Bloque 3: Modelado probabilístico

Este bloque tiene dos partes bien diferenciadas.

### Champion: CatBoost / LightGBM

Son modelos de gradient boosting entrenados de forma **global** — una sola instancia del modelo aprende de todas las series (todos los pares tienda-producto) a la vez, no un modelo por serie. Eso es importante porque permite compartir patrones entre productos similares y funciona bien aunque alguna serie tenga poca historia.

La "selección automática vía promotion logic" significa que el sistema entrena ambos candidatos y solo promueve a champion el que gana en **coste de inventario** sobre el holdout, no en MAE. Es una política MLOps: el modelo vigente solo se sustituye si el challenger lo supera de forma clara.

### Conformal Prediction (Mondrian)

**El problema de base**

CatBoost predice un número — por ejemplo "la demanda del lunes será 42 unidades". Pero ese número tiene incertidumbre. Para el newsvendor necesitamos saber el rango probable: ¿puede ser 20? ¿puede ser 80?

Los modelos boosting tienen modos de predicción de cuantiles nativos, pero esos cuantiles **no están calibrados**. Si el modelo dice q_0.9 = 70, no significa que el 90% de las veces la demanda real caiga por debajo de 70 — puede ser el 65% o el 95%. Nadie lo garantiza.

**La analogía del meteorólogo**

Imagina un meteorólogo que dice "90% de probabilidad de que llueva menos de 20 litros mañana". Si llevas un año apuntando sus predicciones y resulta que la demanda real supera ese límite el 40% de las veces, el meteorólogo está mal calibrado. Conformal Prediction es una capa de corrección: toma ese meteorólogo, le muestra sus errores históricos, y ajusta sus márgenes hasta que cuando diga "90%" realmente ocurra el 90% de las veces. No cambia cómo predice, solo afina cuánta confianza tiene.

En el proyecto: CatBoost es el meteorólogo. Conformal Prediction lo corrige usando un set de calibración — datos reales no vistos en entrenamiento — para medir errores históricos y construir márgenes con garantía estadística real.

**Por qué Mondrian**

La variante básica calibra globalmente sobre todas las series juntas. El problema es que un producto de temporada y uno estable tienen distribuciones de error muy distintas. Si calibras juntos, el intervalo del producto estable queda innecesariamente ancho y el del volátil queda estrecho. Mondrian calibra cada categoría de producto por separado, ajustando los márgenes a su propio comportamiento histórico.

**Los tres cuantiles en la práctica**

- `q_0.5` — predicción central, la mejor estimación de demanda
- `q_0.1` — límite inferior: solo hay un 10% de probabilidad de que la demanda real sea menor
- `q_0.9` — límite superior: solo hay un 10% de probabilidad de que la demanda real sea mayor

**Para el tutor:** *"q_0.5 es la predicción central que va al modelo de inventario. q_0.1 y q_0.9 delimitan el rango de incertidumbre con garantía estadística. Sin esa capa, el newsvendor estaría optimizando con incertidumbre mal estimada."*

---
