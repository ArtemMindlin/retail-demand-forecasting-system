"""The OPS plane: walk-forward playback of the operational simulation."""

from __future__ import annotations

from typing import Any

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from retail_forecasting.api.charts import ops_trajectory_chart
from retail_forecasting.api.services.ops import (
    OpsSimulation,
    SimulationNotFoundError,
    series_trajectory,
    weekly_summary,
)

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
        _simulation = OpsSimulation(settings.REPORTS_DIR)
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
    return "#ef4444"


def ops(request: HttpRequest) -> HttpResponse:
    """Week-by-week playback comparing retrain cadences for one series."""
    try:
        summary = weekly_summary(get_simulation())
    except SimulationNotFoundError as exc:
        return render(request, "views/ops_missing.html", {"detail": str(exc)}, status=200)

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

    cost_weekly = (current or {}).get("by_cadence", {}).get("every_7d", {}).get("total_cost")
    cost_never = (current or {}).get("by_cadence", {}).get("never", {}).get("total_cost")
    savings = _pct(1 - cost_weekly / cost_never) if cost_weekly and cost_never else "—"

    kpis = [
        {
            "label": "Cobertura realizada",
            "value": _pct(block.get("coverage")),
            "sub": f"objetivo {_pct(target)}",
            "color": _coverage_color(block.get("coverage"), target),
        },
        {
            "label": "Coste de inventario (sem.)",
            "value": f"{_num(block.get('total_cost'))} u.m.",
            "sub": "política producción" if cadence == "every_7d" else "sin reentreno",
            "color": "var(--c-inv)",
        },
        {
            "label": "MAE demanda lead-time",
            "value": _num(block.get("mae")),
            "sub": f"{block.get('n_series', 0)} series",
            "color": "var(--c-ai)",
        },
        {
            "label": "Ahorro vs sin reentreno",
            "value": savings,
            "sub": "coste semanal",
            "color": "var(--c-conf)",
        },
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
            "chart": ops_trajectory_chart(points, week_index),
            "has_points": bool(points),
        },
    )
