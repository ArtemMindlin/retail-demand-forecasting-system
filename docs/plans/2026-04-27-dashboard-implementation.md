# Plan: Dashboard Interactivo de Decisiones de Inventario (Streamlit)

Este documento detalla el diseño y la implementación del dashboard para el TFG, orientado a facilitar la interpretación de resultados por parte de un gestor de supply chain.

## 1. Objetivos
*   Visualizar de forma interactiva la eficacia de **Conformal Prediction**.
*   Mostrar alertas de **Drift** de forma clara y accionable.
*   Permitir la comparación entre diferentes ejecuciones de experimentos.
*   Proporcionar una herramienta de simulación de impacto económico.

## 2. Arquitectura de la Interfaz

### Menú Lateral (Sidebar)
*   **Selector de Run:** Listado de directorios en `reports/`.
*   **Configuración de Visualización:** Filtros por `series_id` (producto/tienda).
*   **Parámetros de Inventario:** Sliders para ajustar $C_s$ (stockout) y $C_o$ (overstock) y ver su impacto teórico.

### Cuerpo Principal
1.  **KPI Scorecard:** Tres métricas clave (MAE, Cobertura Real, Coste Total).
2.  **Gráfico de Forecast:** 
    *   Línea de demanda real vs predicción.
    *   Banda de confianza ajustada por Conformal Prediction.
    *   Marcadores de decisiones de inventario (`order_quantity`).
3.  **Panel de Diagnóstico de Drift:** 
    *   Línea de tiempo del error del modelo.
    *   Resaltado de zonas donde el Test de Page-Hinkley detectó cambios.
4.  **Tabla de Datos:** Vista detallada de las predicciones para el periodo seleccionado.

## 3. Stack Tecnológico
*   **Streamlit:** Framework principal.
*   **Plotly:** Gráficos interactivos y dinámicos.
*   **Pandas:** Manipulación de los CSVs de resultados.

## 4. Pasos de Implementación
1.  [ ] Añadir dependencias (`streamlit`, `plotly`).
2.  [ ] Crear `src/retail_forecasting/visualization/dashboard.py`.
3.  [ ] Implementar lógica de carga de archivos desde `reports/`.
4.  [ ] Diseñar los componentes visuales interactivos.
5.  [ ] Añadir instrucciones de ejecución en el `README.md`.

---
*Nota: Este dashboard servirá como la "demo" principal durante la defensa del TFG.*
