# Diario de Metodología y Notas Académicas (TFG)

Este documento registra las bases teóricas, explicaciones conceptuales y justificaciones metodológicas de las mejoras implementadas en el sistema de forecasting. Su objetivo es servir de base para la redacción de la memoria del TFG.

---

## 1. El Problema de la Calibración en Modelos Probabilísticos

### 1.1 Evidencia del Problema (Diagnóstico)
Tras ejecutar el baseline del sistema (v1), se ha detectado un fallo crítico en la cuantificación de la incertidumbre. Aunque el modelo solicita cuantiles 0.1 y 0.9 (un intervalo teórico del 80%), los resultados muestran:
- **Cobertura Real:** ~50.7%
- **Error de Calibración:** ~29.3%

**Conclusión:** El modelo es "optimista" y subestima el riesgo. En un entorno real, esto provocaría que el 30% de las veces que el gestor espera tener stock, se produzca una rotura (stockout).

---

## 2. Deep Dive: Conformal Prediction (Predicción Conforme)

### 2.1 Definición Conceptual
**Conformal Prediction (CP)** es un marco de trabajo estadístico que transforma las predicciones puntuales o intervalos heurísticos de un modelo de Machine Learning en **regiones de predicción con garantías de cobertura rigurosas**.

A diferencia de los métodos Bayesianos o la regresión de cuantiles estándar, CP es **"distribution-free"**: no asume que los errores siguen una distribución normal ni ninguna otra forma específica. Solo asume que los datos son intercambiables (exchangeable).

### 2.2 Implementación: Split Conformal en Series Temporales
Se ha implementado un wrapper `ConformalBoostingModel` que utiliza una ventana de **calibración temporal de 28 días**. Esta ventana permite calcular la magnitud de la descalibración reciente y aplicar un factor corrector $\hat{Q}$ a los cuantiles originales.

### 2.3 Resultados y Discusión de la Brecha Residual
Tras la implementación, la cobertura aumentó del **50.7% al 67.6%**. 

**¿Por qué existe una brecha residual hasta el 80%?**
1. **Violación de la Intercambiabilidad:** En retail, la demanda es altamente no estacionaria. El supuesto de que el error de los últimos 28 días es idéntico al de los próximos 7 días se rompe ante cambios de tendencia o estacionalidad (Data Drift).
2. **Autocorrelación del Error:** Debido al horizonte de predicción multi-paso (7 día), los errores no son independientes, lo que reduce la eficiencia del estimador de CP estándar.
3. **Heterocedasticidad Local:** El factor de corrección es global para todas las series. La varianza dispar entre productos (algunos con demanda muy baja y otros alta) sugiere que un ajuste global es insuficiente.

---

## 3. Detección de Drift (Deriva de Datos)

### 3.1 Concepto de Deriva en Series Temporales
En el contexto de este TFG, el **Drift** se define como un cambio impredecible en las propiedades estadísticas de la variable objetivo a lo largo del tiempo. 

Existen dos tipos fundamentales implementados en este sistema:
1. **Concept Drift:** Cambio en la relación entre las variables de entrada (features) y la demanda. Se monitoriza a través de la degradación del error del modelo (MAE/Pinball Loss).
2. **Data Drift (Covariate Shift):** Cambio en la distribución de las variables de entrada (ej. cambio en el comportamiento de precios o clima).

### 3.2 Metodología: Adaptive Windowing (ADWIN)
Para detectar el drift de forma automática, se utiliza la lógica de **ADWIN**. Este algoritmo mantiene una ventana de datos de tamaño variable que se expande cuando no hay cambios y se contrae drásticamente cuando se detecta un cambio estadísticamente significativo en la media de la métrica monitorizada (el error).

### 3.3 Valor Académico
La detección de drift transforma un sistema de forecasting estático en un **Sistema Adaptativo**. Proporciona una capa de "auto-diagnóstico" que alerta al usuario cuando el modelo ha quedado obsoleto, permitiendo disparar estrategias de reentrenamiento o ajuste de inventario de emergencia.

---

## 4. Próximos Pasos de Implementación
*   [x] Creación de `ConformalBoostingModel`.
*   [x] Implementación de la lógica de calibración en el pipeline de backtesting.
*   [x] Comparativa de métricas de cobertura pre y post calibración.
*   [ ] Implementación de `DriftDetector` basado en Page-Hinkley/ADWIN.
*   [ ] Visualización de las alertas de drift en el reporte final.
