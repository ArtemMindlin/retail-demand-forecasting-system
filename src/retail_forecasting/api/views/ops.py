"""The OPS plane: week-by-week playback of the rolling-origin production backtest."""

from __future__ import annotations

import json
from typing import Any

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from retail_forecasting.api.charts import ops_trajectory_chart
from retail_forecasting.api.services.ops import (
    OpsSimulation,
    SimulationNotFoundError,
    series_trajectory,
    weekly_summary,
)
from retail_forecasting.api.views.empty import empty_state

_simulation: OpsSimulation | None = None

# Human labels for the retrain cadences the simulation compares.
CADENCE_LABELS = {
    "every_7d": "Reentreno semanal",
    "never": "Sin reentreno",
}


def get_simulation() -> OpsSimulation:
    """Shared simulation reader, built from settings on first use."""
    global _simulation
    if _simulation is None:
        _simulation = OpsSimulation()
    return _simulation


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}"


def _coverage_color(coverage: float | None, target: float) -> str:
    """Green within 5 points of target, amber within 25, red beyond."""
    realized = coverage or 0.0
    if realized >= target - 0.05:
        return "var(--c-conf)"
    if realized >= target - 0.25:
        return "var(--c-drift)"
    return "var(--c-crit)"


def _ops_kpi(label: str, value: str, sub: str, color: str) -> dict[str, Any]:
    """One card for the shared KPI grid: colour on the accent bar and on the value."""
    return {"label": label, "value": value, "sub": sub, "color": color, "value_color": color}


def _savings_kpi(comparison: dict[str, Any]) -> tuple[str, str, str]:
    """Saving of weekly retraining vs the baseline, with its bootstrap interval.

    This used to be the ratio of two costs within the displayed week, which moved
    wildly from week to week and read as a result. It is now the paired figure over
    every independent origin, and it says out loud when the window is too short to
    tell the policies apart.
    """
    row = comparison.get("every_7d")
    if not row:
        return "—", "sin comparación pareada", "var(--c-conf)"

    # cost_change_pct is signed as a cost delta: negative cost change = saving.
    saving = -float(row["cost_change_pct"])
    low, high = -float(row["ci95_high_pct"]), -float(row["ci95_low_pct"])
    origins = int(row["n_origins"])
    if row.get("underpowered"):
        sub = f"{origins} orígenes · no concluyente"
        color = "var(--c-drift)"
    elif row.get("conclusive"):
        sub = f"IC95% [{low:+.1f}%, {high:+.1f}%] · {origins} orígenes"
        color = "var(--c-conf)"
    else:
        sub = f"IC95% [{low:+.1f}%, {high:+.1f}%] · indistinguible"
        color = "var(--c-drift)"
    return f"{saving:+.1f}%", sub, color


def ops(request: HttpRequest) -> HttpResponse:
    """Week-by-week playback comparing retrain cadences for one series."""
    try:
        summary = weekly_summary(get_simulation())
    except Exception as exc:
        return empty_state(
            request,
            icon="activity",
            label="SIN BACKTEST",
            title="El backtest de producción no se ha ejecutado",
            detail=(
                f"{exc} El plano OPS lee "
                "simulation/predictions_by_day.parquet de una corrida OPS, que produce el modo "
                "simulate_ops del pipeline."
            ),
            hint="Genéralo con: make simulate",
            page_title="Backtest de producción",
        )

    weeks: list[dict[str, Any]] = summary["weeks"]
    cadences: list[str] = summary["cadences"]
    target: float = summary["target_coverage"]

    cadence = request.GET.get("cadence", "every_7d")
    if cadence not in cadences:
        cadence = cadences[0] if cadences else "every_7d"

    # Default to the last week whose actuals have fully landed; a partial week
    # would show a coverage computed on incomplete data.
    complete = [w for w in weeks if w["by_cadence"].get(cadence, {}).get("actuals_complete")]
    default_week = (complete or weeks)[-1]["week_index"] if weeks else 0
    try:
        week_index = int(request.GET.get("week", default_week))
    except ValueError:
        week_index = default_week

    current = next((w for w in weeks if w["week_index"] == week_index), weeks[0] if weeks else None)
    block = (current or {}).get("by_cadence", {}).get(cadence, {})

    series_list: list[str] = summary["series"]
    series_id = request.GET.get("series") or (series_list[0] if series_list else None)

    points: list[dict[str, Any]] = []
    if series_id:
        try:
            points = series_trajectory(get_simulation(), series_id, cadence)["points"]
        except SimulationNotFoundError:
            points = []

    chart = ops_trajectory_chart(points, week_index)
    savings, savings_sub, savings_color = _savings_kpi(summary.get("comparison", {}))

    # Same card component as the dashboard (partials/kpi_grid.html). These carry no delta:
    # a backtest origin has no previous period to compare itself against, which is why this
    # view used to own a third KPI implementation.
    kpis = [
        _ops_kpi(
            "Cobertura realizada",
            _pct(block.get("coverage")),
            f"objetivo {_pct(target)}",
            _coverage_color(block.get("coverage"), target),
        ),
        _ops_kpi(
            "Coste de inventario (sem.)",
            f"{_num(block.get('total_cost'))} u.m.",
            "política producción" if cadence == "every_7d" else "sin reentreno",
            "var(--c-inv)",
        ),
        _ops_kpi(
            "MAE demanda lead-time",
            _num(block.get("mae")),
            f"{block.get('n_series', 0)} series",
            "var(--c-ai)",
        ),
        _ops_kpi("Ahorro vs sin reentreno", savings, savings_sub, savings_color),
    ]

    return render(
        request,
        "views/ops.html",
        {
            "run": summary["run"],
            "kpis": kpis,
            "cadence": cadence,
            "cadence_options": [{"id": c, "label": CADENCE_LABELS.get(c, c)} for c in cadences],
            "weeks": weeks,
            "week_index": week_index,
            "week_position": next(
                (i + 1 for i, w in enumerate(weeks) if w["week_index"] == week_index), 1
            ),
            "week_count": len(weeks),
            "first_week": weeks[0]["week_index"] if weeks else 0,
            "last_week": weeks[-1]["week_index"] if weeks else 0,
            "origin_date": (current or {}).get("origin_date"),
            "partial_week": block.get("actuals_complete") is False,
            "series_list": series_list,
            "series_id": series_id,
            "target_pct": _pct(target),
            "chart": chart["svg"],
            # Serialized for the pointer handler that positions the hover tooltip.
            "chart_points_json": json.dumps(chart["points"]),
            "chart_width": chart["width"],
            "chart_height": chart["height"],
            "has_points": bool(points),
        },
    )
