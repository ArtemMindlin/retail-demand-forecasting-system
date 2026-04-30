# Forecasting Probabilístico bajo condiciones de Stockout

Este repositorio contiene un sistema de forecasting de demanda retail de nivel investigación para un TFG. El proyecto se centra en la toma de decisiones de inventario bajo incertidumbre, roturas de stock (*stockouts*) y cambios de régimen (*drift*). Utiliza `FreshRetailNet-50K` como dataset base y optimiza el impacto económico mediante políticas de inventario probabilísticas.

## Resumen Ejecutivo

En el sector retail, los *stockouts* generan señales de demanda censuradas que sesgan los modelos convencionales. Este sistema resuelve dicho problema mediante un pipeline de **forecasting probabilístico** que cuantifica la incertidumbre y una capa de **recuperación de demanda latente** que corrige las ventas observadas. El sistema no solo busca minimizar el error predictivo (MAE/RMSE), sino optimizar el coste operativo total bajo una política *Newsvendor*. Las pruebas demuestran una reducción de hasta el **25% en costes operativos** frente a modelos que ignoran la naturaleza censurada de los datos.

## Estructura del Proyecto

- `configs/`: Configuración centralizada de experimentos.
- `data/`: Caché local de datos raw y procesados.
- `docs/`: Documentación metodológica, propuesta y especificaciones técnicas.
- `src/`: Paquete central `retail_forecasting`.
- `tests/`: Suite de tests de integridad, contratos y arquitectura.
- `reports/`: Artefactos generados (Métricas, Costes, Gráficos y Reportes MD).

## Inicio Rápido

El proyecto utiliza `uv` para la gestión de dependencias y un `Makefile` para estandarizar los comandos principales.

```bash
# 1. Instalar dependencias
make install

# 2. Ejecutar EDA reproducible sobre el panel preparado
make eda

# 3. Ejecutar experimento completo (Observed vs Latent)
make run

# 4. Lanzar dashboard interactivo
make dashboard

# 5. Ejecutar tests de integridad
make test-harness
```

## Características Principales

1.  **Imputación de Demanda Latente:** Soporte modular para estrategias supervisadas (LGBM) y baselines estadísticos para corregir la censura por stockout.
2.  **Forecasting Probabilístico:** Wrapper de *Conformal Prediction* para garantizar cobertura de intervalos de confianza.
3.  **Retraining Adaptativo por Drift:** Detección de degradación del modelo vía Page-Hinkley que dispara reentrenamientos dinámicos.
4.  **Evaluación Económica:** Simulación de costes de inventario y análisis de sensibilidad de la razón de costes (Cs/Co).
5.  **Optimización Científica:** Sintonización automática de hiperparámetros mediante búsqueda bayesiana con **Optuna**, optimizando la precisión operativa en cada estrategia de datos.
6.  **Arquitectura por Contratos:** Suite de tests que garantizan la ausencia de *temporal leakage* y la integridad de los dataframes.

---
*Este proyecto es parte de un Trabajo de Fin de Grado (TFG).*
