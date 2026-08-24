"""Research plane: Pareto tuning, cost sensitivity and the fair-cost backtest."""

from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from retail_forecasting.api import eda_charts
from retail_forecasting.api.services import experiments as service
from retail_forecasting.api.services.runs import RunNotFoundError
from retail_forecasting.api.store import get_store
from retail_forecasting.api.views.empty import empty_state


def _pick_run(available: list[str], requested: str | None) -> str | None:
    """The requested run when it exists, otherwise the newest one."""
    if requested and requested in available:
        return requested
    return available[0] if available else None


def pareto(request: HttpRequest) -> HttpResponse:
    """Multi-objective tuning front, cost sensitivity and the fair-cost backtest."""
    store = get_store()
    runs = store.list_runs()
    run_name = _pick_run(runs, request.GET.get("run"))

    if run_name is None:
        return empty_state(
            request,
            icon="sigma",
            label="PLANO DE ANÁLISIS",
            title="Sin runs de experimento",
            detail=(
                "Ningún run contiene predictions.csv, metrics_summary.csv y "
                "cost_summary.csv a la vez."
            ),
            hint="Genéralos con: make run",
        )

    try:
        run_path = store.resolve_run(run_name)
    except RunNotFoundError as exc:
        return empty_state(
            request,
            icon="sigma",
            label="PLANO DE ANÁLISIS",
            title="Run no encontrado",
            detail=str(exc),
        )

    front = service.pareto_front(run_path)
    sensitivity_rows = service.sensitivity(run_path)
    fair = service.fair_cost(store.runs_with)

    # Best configuration on the front: lowest pinball loss among non-dominated.
    best = min(
        (r for r in front if r.get("pinball_loss") is not None),
        key=lambda r: float(r["pinball_loss"]),
        default=None,
    )

    return render(
        request,
        "views/pareto.html",
        {
            "runs": runs,
            "run": run_name,
            "front": front,
            "front_count": len(front),
            "best": best,
            "best_items": sorted(best.items()) if best else [],
            "pareto_chart": eda_charts.pareto_chart(front),
            "sensitivity_chart": eda_charts.sensitivity_chart(sensitivity_rows),
            "has_sensitivity": bool(sensitivity_rows),
            "fair_rows": fair["rows"],
            "fair_run": fair["run"],
            "fair_columns": sorted(fair["rows"][0]) if fair["rows"] else [],
            "summary": service.run_summary(run_path),
        },
    )
