# Diario de Metodología y Notas Académicas (TFG)

Este documento registra las bases teóricas, explicaciones conceptuales y justificaciones metodológicas de las mejoras implementadas en el sistema de forecasting. Su objetivo es servir de base para la redacción de la memoria del TFG.

---

## 1. El Problema de la Calibración en Modelos Probabilísticos
... (contenido previo) ...

---

## 2. Deep Dive: Conformal Prediction (Predicción Conforme)
... (contenido previo) ...

---

## 3. Detección de Drift (Deriva de Datos)
... (contenido previo) ...

---

## 4. Análisis de Sensibilidad Económica
... (contenido previo) ...

---

## 5. Recuperación de Demanda Latente (Tratamiento de Censura)

### 5.1 El Problema: Datos Censurados por la Derecha
... (contenido previo) ...

### 5.2 Metodología: Imputación Supervisada (Two-Step)
... (contenido previo) ...

### 5.3 Resultados Finales y Conclusiones del Sistema
Tras integrar la **Imputación Supervisada** con el modelo **Boosting (LightGBM)** y la calibración por **Conformal Prediction**, el sistema ha alcanzado su máximo rendimiento:

1.  **Eficiencia Económica:** El coste total se redujo a **$35,349**, superando al baseline Naive ($38,197).
2.  **Reducción de Roturas (Stockouts):** Las unidades de stockout cayeron de 6,981 (Naive) a **3,069 (Boosting)**, una reducción del **56%**.
3.  **Precisión del Forecast:** El MAE bajó significativamente gracias a la limpieza de los "ceros artificiales" provocados por las roturas de stock.

**Valor Final del TFG:** Se ha construido un pipeline que autocompensa los sesgos de los datos (censura), garantiza estadísticamente sus márgenes de error (conformal) y monitoriza su propia salud temporal (drift). Esto representa una solución completa y profesional para la toma de decisiones de inventario bajo incertidumbre.

---

## 6. Próximos Pasos de Implementación
*   [x] Creación de `ConformalBoostingModel`.
*   [x] Implementación de la lógica de calibración.
*   [x] Implementación de `DriftDetector`.
*   [x] Implementación de `EconomicSensitivityAnalyzer`.
*   [x] Implementación de `SupervisedImputer` para demanda latente.
*   [x] Visualización interactiva completa en Dashboard.
*   [ ] Fortalecimiento de la suite de tests unitarios (Garantía de Calidad).
