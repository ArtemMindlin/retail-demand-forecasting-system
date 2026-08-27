"""How the panel behaves over time: coverage, the weekly cycle, and its autocorrelation.

The category seasonality heatmaps live here too: they are weekday profiles cut by product
category, and they used to sit in a module of their own only because they arrived as a
standalone script rather than as part of the run.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_temporal_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize panel-wide temporal coverage and continuity."""
    date_span_days = int((panel["date"].max() - panel["date"].min()).days) + 1
    expected_rows = date_span_days * panel["series_id"].nunique()

    return pd.DataFrame(
        [
            {
                "date_min": panel["date"].min(),
                "date_max": panel["date"].max(),
                "date_span_days": date_span_days,
                "observed_rows": len(panel),
                "expected_rows_full_grid": expected_rows,
                "coverage_rate_full_grid": len(panel) / expected_rows if expected_rows else 0.0,
                "duplicate_series_date_rows": int(panel.duplicated(["series_id", "date"]).sum()),
            }
        ]
    )


def build_weekday_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize weekly seasonality on the prepared panel."""
    weekday_panel = panel.assign(
        weekday=panel["date"].dt.dayofweek,
        weekday_name=panel["date"].dt.day_name(),
    )

    return (
        weekday_panel.groupby(["weekday", "weekday_name"])
        .agg(
            observations=("observed_demand", "size"),
            observed_demand_mean=("observed_demand", "mean"),
            observed_demand_median=("observed_demand", "median"),
            stockout_day_rate=("stockout_hours", lambda values: (values > 0).mean()),
            mean_stockout_hours=("stockout_hours", "mean"),
        )
        .reset_index()
        .sort_values("weekday")
        .reset_index(drop=True)
    )


def build_series_gap_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing calendar gaps within each series history."""
    rows = []

    for series_id, series_frame in panel.groupby("series_id", sort=False):
        ordered = series_frame.sort_values("date")
        day_deltas = ordered["date"].diff().dt.days.dropna()
        max_gap_days = int(day_deltas.max()) if not day_deltas.empty else 1
        missing_days_within_span = (
            int((day_deltas - 1).clip(lower=0).sum()) if not day_deltas.empty else 0
        )
        rows.append(
            {
                "series_id": series_id,
                "history_days": ordered["date"].nunique(),
                "start_date": ordered["date"].min(),
                "end_date": ordered["date"].max(),
                "max_gap_days": max_gap_days,
                "missing_days_within_span": missing_days_within_span,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["missing_days_within_span", "series_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def render_temporal_figures(
    panel: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Every figure about time: the weekly profile, coverage over the calendar, and the ACF."""
    _plot_weekday_demand_profile(weekday_summary, output_dir / "weekday_demand_profile.png")
    _plot_coverage_heatmap(panel, series_summary, output_dir / "coverage_heatmap.png")
    _plot_acf_demand(panel, output_dir / "acf_demand.png")
    render_category_seasonality_heatmaps(panel, output_dir)


MAX_HEATMAP_SERIES = 120


def _plot_weekday_demand_profile(
    weekday_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    dias = weekday_summary["weekday_name"].map(_DAY_ES).fillna(weekday_summary["weekday_name"])
    ax.plot(
        dias,
        weekday_summary["observed_demand_mean"],
        marker="o",
        linewidth=2,
        color="#2ca02c",
        label="Media",
    )
    ax.plot(
        dias,
        weekday_summary["observed_demand_median"],
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Mediana",
    )
    ax.set_xlabel("Día de la semana")
    ax.set_ylabel("Demanda observada")
    ax.set_title("Perfil semanal de demanda")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_coverage_heatmap(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_series = series_summary.head(MAX_HEATMAP_SERIES)["series_id"].tolist()
    coverage_frame = (
        panel.loc[panel["series_id"].isin(selected_series), ["series_id", "date"]]
        .assign(present=1.0)
        .pivot(index="series_id", columns="date", values="present")
        .fillna(0.0)
    )
    if coverage_frame.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    image = ax.imshow(
        coverage_frame.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Índice de fecha")
    ax.set_ylabel("Serie")
    ax.set_title(f"Mapa de cobertura (las {len(selected_series)} series de mayor demanda)")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Presente (1) / ausente (0)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_acf_demand(panel: pd.DataFrame, output_path: Path, max_lags: int = 28) -> None:
    daily_mean = panel.groupby("date")["observed_demand"].mean().sort_index().to_numpy()
    n = len(daily_mean)
    if n < max_lags + 2:
        return

    mean = daily_mean.mean()
    centered = daily_mean - mean
    var = (centered**2).sum()
    if var == 0:
        return

    acf_values = np.array(
        [(centered[: n - lag] * centered[lag:]).sum() / var for lag in range(max_lags + 1)]
    )
    confidence_bound = 1.96 / np.sqrt(n)
    lags = np.arange(max_lags + 1)

    seasonal_lags = {7, 14, 21, 28}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(
        -confidence_bound, confidence_bound, color="#1f77b4", alpha=0.12, label="IC 95\u00a0%"
    )

    for lag, val in zip(lags, acf_values, strict=False):
        color = "#d62728" if lag in seasonal_lags else "#1f77b4"
        ax.vlines(lag, 0, val, colors=color, linewidth=1.8)
        ax.plot(lag, val, "o", color=color, markersize=4)

    for s_lag in seasonal_lags:
        if s_lag <= max_lags:
            ax.axvline(float(s_lag), color="#d62728", linewidth=0.7, linestyle="--", alpha=0.45)

    ax.set_xlabel("Retardo (días)")
    ax.set_ylabel("Autocorrelación")
    ax.set_title("ACF de la demanda diaria agregada (retardos 0–28)")
    ax.set_xticks(lags)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


_DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

_TIER_COLOR = {"High": "#2166ac", "Medium": "#f4a582", "Low": "#d6604d"}

# Traducciones SOLO para lo que se dibuja. Las claves inglesas se conservan porque vienen de
# `dt.day_name()` y de la clasificacion por niveles, y son las que ordenan y agrupan: cambiarlas
# obligaria a reescribir la logica para ganar unas etiquetas.
_DAY_ES = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}

_TIER_ES = {"High": "alta", "Medium": "media", "Low": "baja"}

_MIN_OBSERVATIONS = 500


def _weekday_profiles(
    panel: pd.DataFrame, min_observations: int
) -> tuple[pd.DataFrame, dict[str, str], pd.Series] | None:
    """Per-category weekday Z-scores, each category's demand tier, and its heterogeneity.

    Returns None when the panel cannot support the figure: no category column, no category
    with enough observations, or a week not fully covered. A synthetic or short panel hits
    all three, and a missing figure is a better answer there than a confident wrong one.
    """
    if "third_category_id" not in panel.columns:
        return None

    counts = panel.groupby("third_category_id")["observed_demand"].count()
    valid_categories = counts[counts >= min_observations].index
    if valid_categories.empty:
        return None

    panel_valid = panel[panel["third_category_id"].isin(valid_categories)].copy()
    panel_valid["date"] = pd.to_datetime(panel_valid["date"])
    panel_valid["day_of_week"] = panel_valid["date"].dt.day_name()

    pivot = panel_valid.groupby(["third_category_id", "day_of_week"])["observed_demand"].mean()
    unstacked = pivot.unstack()
    if not set(_DAY_ORDER).issubset(unstacked.columns):
        return None

    pivot_df = unstacked[_DAY_ORDER]
    pivot_std = pivot_df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

    mean_demand = panel_valid.groupby("third_category_id")["observed_demand"].mean()
    low_cut, high_cut = mean_demand.quantile(1 / 3), mean_demand.quantile(2 / 3)

    def demand_tier(category: str) -> str:
        value = mean_demand.get(category, 0)
        if value >= high_cut:
            return "High"
        return "Medium" if value >= low_cut else "Low"

    tier_map = {category: demand_tier(category) for category in pivot_std.index}

    mean_profile = pivot_std.mean(axis=0)
    deviation = ((pivot_std - mean_profile) ** 2).sum(axis=1)

    return pivot_std, tier_map, deviation


def render_category_seasonality_heatmaps(
    panel: pd.DataFrame,
    output_dir: Path,
    n_per_tier: int = 7,
    min_observations: int = _MIN_OBSERVATIONS,
) -> None:
    """Write one weekly-seasonality heatmap per demand tier into ``output_dir``."""
    profiles = _weekday_profiles(panel, min_observations)
    if profiles is None:
        return
    pivot_std, tier_map, deviation = profiles

    output_dir.mkdir(parents=True, exist_ok=True)

    for tier in ("High", "Medium", "Low"):
        categories = [c for c in pivot_std.index if tier_map[c] == tier]
        if not categories:
            continue
        top = deviation.loc[categories].nlargest(n_per_tier).index
        # Se ordena por la clave inglesa y se renombra solo para dibujar.
        data = pivot_std.loc[top].sort_values("Sunday", ascending=False)
        data = data.rename(columns=_DAY_ES)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            data,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={"label": "Z-score"},
            ax=ax,
        )
        ax.set_title(
            f"Estacionalidad semanal · categorías de demanda {_TIER_ES[tier]}\n"
            f"(Z-score por categoría; las {len(top)} más heterogéneas)",
            color=_TIER_COLOR[tier],
            fontsize=11,
        )
        ax.set_xlabel("Día de la semana")
        ax.set_ylabel("ID de categoría")

        fig.tight_layout()
        fig.savefig(
            output_dir / f"category_seasonality_{tier.lower()}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
