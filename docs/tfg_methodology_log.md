# Diario de Metodología y Notas Académicas (TFG)

Este documento registra las bases teóricas, explicaciones conceptuales y justificaciones metodológicas de las mejoras implementadas en el sistema de forecasting. Su objetivo es servir de base para la redacción de la memoria del TFG.

---

## 1. El Problema de la Calibración en Modelos Probabilísticos
... (contenido previo sobre CP) ...

---

## 2. Deep Dive: Conformal Prediction (Predicción Conforme)
... (contenido previo sobre CP) ...

---

## 3. Detección de Drift (Deriva de Datos)
... (contenido previo sobre Drift) ...

---

## 4. Análisis de Sensibilidad Económica

### 4.1 Teoría de la Decisión: El Problema del Newsvendor
En la gestión de inventarios de un solo periodo, la cantidad óptima a pedir ($Q^*$) no es simplemente la media de la demanda, sino que depende del equilibrio entre el coste de pedir de más (Overstock, $C_o$) y el coste de quedarse corto (Stockout, $C_s$). 

La solución óptima viene dada por el **Critical Fractile** (Fractil Crítico):
$$P(D \le Q^*) = \frac{C_s}{C_s + C_o}$$

Donde $D$ es la demanda. El sistema debe pedir la cantidad correspondiente a este cuantil de la distribución de probabilidad de la demanda.

### 4.2 Objetivos del Análisis de Sensibilidad
Un modelo de forecasting no puede evaluarse de forma aislada. Su valor real depende del contexto de negocio:
1. **Ratio de Costes:** Si $C_s >> C_o$, el sistema debe ser agresivo y pedir más. Si $C_s \approx C_o$, el sistema debe ser conservador.
2. **Robustez del Modelo:** Un modelo "bueno" en MAE puede ser "malo" económicamente si sus errores ocurren en los escenarios de mayor coste.
3. **Cambio de Rankings:** El análisis de sensibilidad permite identificar en qué perfiles de negocio (ej. productos de lujo vs. productos básicos) el modelo de Machine Learning aporta más valor frente a heurísticos simples como el Naive.

### 4.3 Metodología de Simulación
Se realiza una re-evaluación *ex-post* de las predicciones variando el ratio $R = C_s / C_o$ en un rango de $[1, 10]$. Para cada ratio:
1. Se recalcula el Fractil Crítico.
2. Se determina la nueva `order_quantity`.
3. Se computan los costes económicos resultantes para todos los modelos en competencia.

---

## 5. Próximos Pasos de Implementación
*   [x] Creación de `ConformalBoostingModel`.
*   [x] Implementación de la lógica de calibración.
*   [x] Implementación de `DriftDetector`.
*   [ ] Implementación de `EconomicSensitivityAnalyzer`.
*   [ ] Visualización de curvas de sensibilidad en el Dashboard.
