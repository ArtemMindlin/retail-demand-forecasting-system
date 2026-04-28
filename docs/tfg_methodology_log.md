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

## 7. Modularización y Reentrenamiento Adaptativo (Hitos Finales)

### 7.1 Imputación Modular
Se ha evolucionado el sistema de recuperación de demanda latente hacia una arquitectura modular (`LatentDemandImputer`). Esto permite comparar científicamente el impacto de diferentes heurísticas (Media Histórica, Escalado) frente al modelo profesor supervisado.

### 7.2 Reentrenamiento Gatillado por Drift
Se ha cerrado el bucle de control del sistema mediante la integración operativa del detector de drift. El pipeline ahora monitoriza el error en tiempo real y decide autónomamente cuándo el modelo requiere una actualización, optimizando el compromiso entre coste computacional y precisión predictiva.

### 7.3 Conclusión Técnica
Con la inclusión del **Makefile** y la **Arquitectura por Contratos**, el proyecto ha alcanzado un nivel de madurez de ingeniería listo para su defensa y entrega.

---

## 8. Estado Actual del Proyecto
*   [x] Refactorización modular de imputación.
*   [x] Lógica de reentrenamiento adaptativo operativa.
*   [x] Makefile de automatización.
*   [x] Documentación técnica integral y manual de dependencias.
*   [x] Implementación de optimización de hiperparámetros con **Optuna**.
*   [ ] Fortalecimiento de la suite de tests unitarios (Garantía de Calidad).
