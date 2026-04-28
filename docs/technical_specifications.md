# Especificaciones Técnicas y Referencia de Dependencias

Este documento detalla el stack tecnológico, las dependencias del proyecto y la evolución de la arquitectura técnica del sistema.

## 1. Stack Tecnológico Base

- **Lenguaje:** Python 3.11+ (Gestionado con `uv`)
- **Gestor de Paquetes:** `uv` (Fast Python package installer and resolver)
- **Automatización:** GNU Makefile
- **Entorno de Datos:** Pandas + NumPy (Procesamiento de paneles temporales)

## 2. Dependencias Principales (Bibliotecas)

| Biblioteca | Versión | Propósito en el Proyecto |
| :--- | :--- | :--- |
| `pandas` | >=2.2.0 | Gestión de paneles, alineación de series temporales y feature engineering. |
| `lightgbm` | >=4.5.0 | Modelo profesor para imputación de demanda latente y estimación de cuantiles. |
| `catboost` | >=1.2.10 | Modelo de boosting global con soporte nativo para variables categóricas. |
| `pmdarima` | >=2.1.1 | Implementación de Auto-ARIMA para baselines estadísticos adaptativos. |
| `scikit-learn` | >=1.5.0 | Algoritmos de regresión lineal (Ridge) y utilidades de preprocesamiento. |
| `streamlit` | >=1.56.0 | Interfaz de visualización interactiva para el dashboard de resultados. |
| `plotly` | >=6.7.0 | Gráficos interactivos de series temporales, intervalos y sensibilidad económica. |
| `PyYAML` | >=6.0.2 | Gestión de la configuración experimental centralizada en `configs/`. |
| `pytest` | >=9.0.2 | Suite de validación de integridad, contratos y anti-leakage. |

## 3. Arquitectura del Sistema

El sistema sigue un patrón de **Tubería Basada en Contratos**:
1.  **Capa de Datos (`data/`)**: Ingesta y saneamiento. Implementa el `LatentDemandImputer`.
2.  **Capa de Features (`features/`)**: Construcción de variables exógenas con protección de *look-ahead bias*.
3.  **Capa de Modelado (`models/`)**: Wrappers de modelos que cumplen con el protocolo de `Forecaster`. Implementa el decorador `ConformalForecaster`.
4.  **Capa de Orquestación (`forecasting/`)**: Gestión de backtesting *walk-forward* y reentrenamiento adaptativo por *drift*.
5.  **Capa de Decisión (`inventory/`)**: Conversión de forecasts en unidades de pedido vía lógica *Newsvendor*.
6.  **Capa de Evaluación (`evaluation/`)**: Cálculo de métricas predictivas y costes económicos agregados.

## 4. Registro de Evolución Técnica (Log de Implementación)

| Fecha | Cambio / Adición | Descripción Técnica |
| :--- | :--- | :--- |
| 2026-04-28 | **LatentDemandImputer v2** | Refactorización modular para soportar estrategias `supervised`, `historical_mean` y `scaling`. |
| 2026-04-28 | **Adaptive Retraining** | Integración de `PageHinkleyDetector` en el pipeline de backtesting para disparar reentrenamientos. |
| 2026-04-28 | **Unified Makefile** | Estandarización de comandos de ejecución, tests y dashboard. |
| 2026-04-28 | **Reporting Extra** | Inclusión de alertas de drift y comparativa de estrategias de datos en el `report.md`. |
| 2026-04-28 | **Optuna Tuning** | Integración de búsqueda bayesiana de hiperparámetros con validación cruzada temporal interna. |
| 2026-04-28 | **Hardware Parallelization** | Paralelización multihilo de ARIMA (Joblib), Boosting y Optuna para optimizar el uso de CPU. |
| 2026-04-28 | **Prescriptive Training** | Implementación de funciones de pérdida asimétricas orientadas al fractil crítico de inventario. |

## 5. Gestión de Dependencias Futuras

Cualquier nueva biblioteca debe añadirse mediante `uv add` y registrarse en este documento especificando su propósito y el módulo donde se integra.
