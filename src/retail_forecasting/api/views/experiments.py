"""Research plane: latent-demand imputation and Pareto tuning."""

from __future__ import annotations

from typing import Any

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


def latent(request: HttpRequest) -> HttpResponse:
    """Compare latent-demand reconstruction strategies for one series."""
    store = get_store()
    runs = [d.name for d in store.runs_with("latent_strategies.csv")]
    run_name = _pick_run(runs, request.GET.get("run"))

    if run_name is None:
        return empty_state(
            request,
            icon="sigma",
            label="PLANO DE ANÁLISIS",
            title="Sin comparación de imputación",
            detail=(
                "Ningún run contiene latent_strategies.csv. Ese artefacto lo produce el "
                "modo de comparación de estrategias de imputación."
            ),
            hint=(
                "Genérala con: uv run python -m retail_forecasting.run "
                "--config configs/imputation_compare.yaml"
            ),
        )

    try:
        run_path = store.resolve_run(run_name, requires="latent_strategies.csv")
    except RunNotFoundError as exc:
        return empty_state(
            request,
            icon="sigma",
            label="PLANO DE ANÁLISIS",
            title="Run no encontrado",
            detail=str(exc),
        )

    data = service.imputation_strategies(run_path, request.GET.get("series"))
    quality = service.rank_quality(data.get("quality", []))

    strategy_colors = {name: service.strategy_meta(name)["color"] for name in data["strategies"]}
    censored_days = sum(1 for h in data.get("stockout_hours", []) if h)

    context: dict[str, Any] = {
        "runs": runs,
        "run": run_name,
        "series_list": data["series"],
        "series_id": data.get("series_id"),
        "day_count": len(data["dates"]),
        "censored_days": censored_days,
        "has_data": bool(data["dates"]),
        "chart": eda_charts.latent_compare(data, strategy_colors),
        "quality": quality,
        "best": quality[0] if quality else None,
        "n_eval": quality[0].get("n_eval") if quality else None,
        "series_count": len(data["series"]),
        "strategy_rows": [
            {
                **service.strategy_meta(name),
                "name": name,
                "values": values,
            }
            for name, values in data["strategies"].items()
        ],
    }
    return render(request, "views/latent.html", context)


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
