"""Static metadata for the EDA figures exported to the thesis (memoria).

Each entry pairs a generated figure with its LaTeX caption, label and a short
interpretation paragraph. Kept apart from ``reporting.py`` so the reporting
logic stays readable and the captions are easy to edit in one place.
"""

from __future__ import annotations

MEMORIA_FIGURE_EXPORTS: list[dict[str, str]] = [
    {
        "filename": "observed_demand_distribution.png",
        "caption": "Distribución global de la demanda observada en el panel preparado.",
        "label": "fig:eda_observed_demand_distribution",
        "interpretation": (
            "Interpretación. La distribución de la demanda observada muestra concentración en "
            "rangos bajos y una cola hacia valores mayores, lo que es coherente con un problema "
            "retail heterogéneo y alejado de una distribución gaussiana simple."
        ),
    },
    {
        "filename": "observed_demand_boxplot_top_series.png",
        "caption": "Dispersión de la demanda observada en las series de mayor volumen.",
        "label": "fig:eda_observed_demand_boxplot",
        "interpretation": (
            "Interpretación. Incluso entre las series de mayor volumen persisten diferencias "
            "relevantes en nivel medio y variabilidad, lo que refuerza la conveniencia de "
            "incorporar contexto de serie en el modelado."
        ),
    },
    {
        "filename": "coverage_heatmap.png",
        "caption": "Cobertura temporal del panel por serie y fecha.",
        "label": "fig:eda_coverage_heatmap",
        "interpretation": (
            "Interpretación. La cobertura uniforme indica que el panel preparado es balanceado "
            "tras el filtrado, sin huecos internos ni series truncadas dentro del horizonte "
            "analizado."
        ),
    },
    {
        "filename": "weekday_demand_profile.png",
        "caption": "Perfil semanal de demanda observada con media y mediana.",
        "label": "fig:eda_weekday_profile",
        "interpretation": (
            "Interpretación. El patrón semanal visible justifica el uso de variables de "
            "calendario y retardos de 7 días en la etapa de ingeniería de características."
        ),
    },
    {
        "filename": "zero_demand_rate_by_series.png",
        "caption": "Series más intermitentes según su proporción de demanda cero.",
        "label": "fig:eda_zero_demand_rate",
        "interpretation": (
            "Interpretación. La intermitencia no es homogénea entre series, por lo que el "
            "problema no debe interpretarse como uniforme para todas las combinaciones "
            "tienda-producto."
        ),
    },
    {
        "filename": "stockout_hours_distribution.png",
        "caption": "Distribución de horas de stockout en el panel preparado.",
        "label": "fig:eda_stockout_distribution",
        "interpretation": (
            "Interpretación. La frecuencia de stockouts confirma que la falta de disponibilidad "
            "forma parte del régimen operativo del dataset y no constituye un fenómeno aislado."
        ),
    },
    {
        "filename": "stockout_band_demand.png",
        "caption": "Demanda media y número de observaciones por banda de stockout.",
        "label": "fig:eda_stockout_band_demand",
        "interpretation": (
            "Interpretación. La caída de la demanda observada bajo stockouts severos es "
            "consistente con la hipótesis de censura operativa por falta de disponibilidad."
        ),
    },
    {
        "filename": "stockout_vs_demand_scatter.png",
        "caption": "Relación entre horas de stockout y demanda observada.",
        "label": "fig:eda_stockout_vs_demand",
        "interpretation": (
            "Interpretación. La tendencia agregada negativa sugiere que las horas de stockout "
            "aportan señal contextual relevante, aunque con elevada dispersión entre "
            "observaciones."
        ),
    },
    {
        "filename": "correlation_heatmap.png",
        "caption": "Mapa de correlaciones entre variables numéricas del panel preparado.",
        "label": "fig:eda_correlation_heatmap",
        "interpretation": (
            "Interpretación. Las asociaciones marginales son en general moderadas, lo que "
            "respalda el uso de modelos flexibles capaces de capturar interacciones y no "
            "linealidades."
        ),
    },
    {
        "filename": "covariate_vs_demand_grid.png",
        "caption": "Relaciones muestreadas entre covariables exógenas y demanda observada.",
        "label": "fig:eda_covariates_vs_demand",
        "interpretation": (
            "Interpretación. Las covariables exógenas muestran señal descriptiva, pero su efecto "
            "no es simple ni uniforme, por lo que conviene analizarlas junto con el contexto "
            "temporal y de serie."
        ),
    },
    {
        "filename": "representative_series_panels.png",
        "caption": "Series representativas con demanda observada y overlay de stockout.",
        "label": "fig:eda_representative_series",
        "interpretation": (
            "Interpretación. La visualización conjunta de demanda y stockout resume la "
            "complejidad del problema: estacionalidad, heterogeneidad entre series y posible "
            "compresión de ventas observadas."
        ),
    },
]
