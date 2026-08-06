"""Chart-ready data for the exploratory analysis view.

The EDA module writes PNGs plus the CSVs behind them. The dashboard prefers the
CSVs and redraws each figure as an SVG, so the charts inherit the dashboard's
palette and stay readable at any width; the PNG remains available as a
full-resolution fallback in the lightbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Figure catalogue: the order here is the order shown in the view.
FIGURES: tuple[dict[str, Any], ...] = (
    {
        "name": "observed_demand_distribution",
        "nav_label": "Distribución de demanda",
        "caption": "Distribución global de la demanda observada",
        "interpretation": (
            "La distribución muestra concentración en rangos bajos y una cola hacia valores "
            "mayores, lo que es coherente con un problema retail heterogéneo y alejado de una "
            "distribución gaussiana simple."
        ),
    },
    {
        "name": "weekday_demand_profile",
        "nav_label": "Perfil semanal",
        "caption": "Perfil semanal de demanda (media y mediana)",
        "interpretation": (
            "El patrón semanal visible justifica el uso de variables de calendario y retardos "
            "de 7 días en la etapa de ingeniería de características."
        ),
    },
    {
        "name": "observed_demand_boxplot_top_series",
        "nav_label": "Dispersión top series",
        "caption": "Dispersión de la demanda en las series de mayor volumen",
        "interpretation": (
            "Incluso entre las series de mayor volumen persisten diferencias relevantes en "
            "nivel medio y variabilidad, lo que refuerza la conveniencia de incorporar "
            "contexto de serie en el modelado."
        ),
    },
    {
        "name": "zero_demand_rate_by_series",
        "nav_label": "Intermitencia por serie",
        "caption": "Series más intermitentes (proporción de demanda cero)",
        "interpretation": (
            "La intermitencia no es homogénea entre series, por lo que el problema no debe "
            "interpretarse como uniforme para todas las combinaciones tienda-producto."
        ),
    },
    {
        "name": "stockout_hours_distribution",
        "nav_label": "Distribución stockout",
        "caption": "Distribución de horas de stockout en el panel",
        "interpretation": (
            "La frecuencia de stockouts confirma que la falta de disponibilidad forma parte "
            "del régimen operativo del dataset y no constituye un fenómeno aislado."
        ),
    },
    {
        "name": "stockout_band_demand",
        "nav_label": "Demanda por banda stockout",
        "caption": "Demanda media y observaciones por banda de stockout",
        "interpretation": (
            "La caída de la demanda observada bajo stockouts severos es consistente con la "
            "hipótesis de censura operativa por falta de disponibilidad."
        ),
    },
    {
        "name": "stockout_vs_demand_scatter",
        "nav_label": "Stockout vs demanda",
        "caption": "Relación entre horas de stockout y demanda observada",
        "interpretation": (
            "La tendencia agregada negativa sugiere que las horas de stockout aportan señal "
            "contextual relevante, aunque con elevada dispersión entre observaciones."
        ),
    },
    {
        "name": "correlation_heatmap",
        "nav_label": "Correlaciones",
        "caption": "Correlaciones entre features numéricas y demanda",
        "interpretation": (
            "Las asociaciones marginales son en general moderadas, lo que respalda el uso de "
            "modelos flexibles capaces de capturar interacciones y no linealidades."
        ),
    },
    {
        "name": "representative_series_panels",
        "nav_label": "Series representativas",
        "caption": "Pequeños múltiplos de demanda con overlay de stockout",
        "interpretation": (
            "La visualización conjunta de demanda y stockout resume la complejidad del "
            "problema: estacionalidad, heterogeneidad entre series y posible compresión de "
            "ventas observadas."
        ),
        "wide": True,
    },
)

FIGURE_NAMES = frozenset(figure["name"] for figure in FIGURES)


class ChartDataUnavailableError(Exception):
    """The CSV backing a figure is missing from the EDA run."""


def histogram(values: list[float], bins: int = 25) -> tuple[list[float], list[int]]:
    """Bin centres and counts, ignoring NaNs. Empty lists when there is no data."""
    array = np.array([v for v in values if v is not None and not np.isnan(v)])
    if array.size == 0:
        return [], []
    counts, edges = np.histogram(array, bins=bins)
    centers = [float((edges[i] + edges[i + 1]) / 2) for i in range(len(counts))]
    return centers, counts.tolist()


def available_figures(eda_path: Path) -> list[dict[str, Any]]:
    """Catalogue entries whose PNG exists in this run."""
    return [figure for figure in FIGURES if (eda_path / f"{figure['name']}.png").exists()]


def dataset_summary(eda_path: Path) -> dict[str, Any]:
    """First row of ``dataset_summary.csv`` as a mapping ({} when absent)."""
    path = eda_path / "dataset_summary.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def figure_path(eda_path: Path, name: str) -> Path:
    """Validated filesystem path of a figure PNG.

    Raises:
        ChartDataUnavailableError: for unknown names or missing files. The name
            is checked against the catalogue rather than sanitised, which rules
            out traversal by construction.
    """
    if name not in FIGURE_NAMES:
        raise ChartDataUnavailableError("Figure not found.")
    path = eda_path / f"{name}.png"
    if not path.exists():
        raise ChartDataUnavailableError(f"Figure '{name}' not found.")
    return path


def _read(eda_path: Path, filename: str) -> pd.DataFrame:
    path = eda_path / filename
    if not path.exists():
        raise ChartDataUnavailableError(f"Data not available: {filename}")
    return pd.read_csv(path)


def chart_data(eda_path: Path, name: str) -> dict[str, Any]:  # noqa: C901 - flat dispatch table
    """Chart-ready payload for one named figure.

    Raises:
        ChartDataUnavailableError: for unknown figures or missing source CSVs.
    """
    if name not in FIGURE_NAMES:
        raise ChartDataUnavailableError("Figure not found.")

    if name == "weekday_demand_profile":
        frame = _read(eda_path, "weekday_summary.csv")
        return {
            "type": "line_dual",
            "data": frame[
                ["weekday_name", "observed_demand_mean", "observed_demand_median"]
            ].to_dict("records"),
            "series": [
                {"key": "observed_demand_mean", "label": "Media", "color": "#10b981"},
                {"key": "observed_demand_median", "label": "Mediana", "color": "#3b82f6"},
            ],
        }

    if name == "stockout_band_demand":
        frame = _read(eda_path, "stockout_demand_bands.csv")
        columns = [
            c
            for c in ["stockout_band", "observed_demand_mean", "observations"]
            if c in frame.columns
        ]
        return {
            "type": "bar_group",
            "data": frame[columns].to_dict("records"),
            "x_key": "stockout_band",
        }

    if name == "correlation_heatmap":
        frame = _read(eda_path, "correlation_summary.csv")
        frame = frame.sort_values("absolute_correlation", ascending=False).head(15)
        return {
            "type": "bar_horizontal",
            "data": frame[["feature_name", "correlation_with_observed_demand"]].to_dict("records"),
        }

    if name == "zero_demand_rate_by_series":
        frame = _read(eda_path, "series_summary.csv")
        rates = frame["zero_demand_rate"].dropna().tolist()
        centers, counts = histogram(rates, 25)
        return {
            "type": "histogram",
            "centers": centers,
            "counts": counts,
            "x_label": "Tasa demanda cero",
            "median": float(np.median(rates)) if rates else 0.0,
            "pct_above_50": (
                float(sum(1 for r in rates if r > 0.5) / len(rates) * 100) if rates else 0.0
            ),
        }

    if name == "observed_demand_distribution":
        frame = _read(eda_path, "series_summary.csv")
        means = frame["observed_demand_mean"].dropna().tolist()
        centers, counts = histogram(means, 30)
        return {
            "type": "histogram_dual",
            "centers": centers,
            "counts": counts,
            "log_counts": [float(np.log10(c)) if c > 0 else 0.0 for c in counts],
            "x_label": "Demanda media por serie",
        }

    if name == "stockout_hours_distribution":
        try:
            frame = _read(eda_path, "stockout_by_series_summary.csv")
        except ChartDataUnavailableError:
            frame = _read(eda_path, "series_summary.csv")
        column = (
            "mean_stockout_hours" if "mean_stockout_hours" in frame.columns else "stockout_day_rate"
        )
        centers, counts = histogram(frame[column].dropna().tolist(), 25)
        return {
            "type": "histogram",
            "centers": centers,
            "counts": counts,
            "x_label": column.replace("_", " "),
        }

    if name == "stockout_vs_demand_scatter":
        frame = _read(eda_path, "series_summary.csv")
        subset = frame[["observed_demand_mean", "mean_stockout_hours"]].dropna().head(500)
        return {
            "type": "scatter",
            "data": subset.rename(
                columns={"observed_demand_mean": "x", "mean_stockout_hours": "y"}
            ).to_dict("records"),
        }

    if name == "observed_demand_boxplot_top_series":
        frame = _read(eda_path, "series_summary.csv")
        top = frame.nlargest(20, "observed_demand_sum")[
            ["series_id", "observed_demand_mean", "observed_demand_std"]
        ].dropna()
        # The EDA CSV stores mean and sd, not the five-number summary, so the
        # box is a ±1σ / ±2σ sketch rather than true quartiles.
        boxes = []
        for _, row in top.iterrows():
            mean = float(row["observed_demand_mean"])
            sd = float(row["observed_demand_std"])
            boxes.append(
                {
                    "id": str(row["series_id"]),
                    "min": max(0.0, mean - 2 * sd),
                    "q1": max(0.0, mean - sd),
                    "median": mean,
                    "q3": mean + sd,
                    "max": mean + 2 * sd,
                }
            )
        return {"type": "boxplot", "data": boxes}

    if name == "representative_series_panels":
        frame = _read(eda_path, "series_summary.csv")
        columns = [
            c
            for c in [
                "series_id",
                "observed_demand_mean",
                "observed_demand_std",
                "zero_demand_rate",
                "stockout_day_rate",
                "history_days",
            ]
            if c in frame.columns
        ]
        return {
            "type": "series_grid",
            "series": frame.nlargest(12, "observed_demand_sum")[columns].to_dict("records"),
        }

    raise ChartDataUnavailableError("Figure data not available.")
