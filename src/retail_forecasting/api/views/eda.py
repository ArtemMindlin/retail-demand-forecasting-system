"""Exploratory-analysis view: figures redrawn as SVG, with PNG fallback."""

from __future__ import annotations

import logging
from typing import Any

from django.http import FileResponse, Http404, HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from retail_forecasting.api import eda_charts
from retail_forecasting.api.services import eda as eda_service
from retail_forecasting.api.services.runs import RunNotFoundError
from retail_forecasting.api.store import get_store

logger = logging.getLogger(__name__)


# Headline stats surfaced from dataset_summary.csv, in display order. The keys
# are the column names the EDA module writes; each entry carries an optional
# formatter so counts, rates and dates each read naturally.
def _thousands(value: Any) -> str:
    return f"{float(value):,.0f}".replace(",", ".")


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.1f}".replace(".", ",") + " %"


def _plain(value: Any) -> str:
    return str(value)


_SUMMARY_FIELDS = (
    ("unique_series", "Series únicas", _thousands),
    ("rows", "Observaciones", _thousands),
    ("stockout_day_rate", "Tasa de stockout", _percent),
    ("zero_demand_rate", "Demanda cero", _percent),
    ("date_min", "Desde", _plain),
    ("date_max", "Hasta", _plain),
)


def eda(request: HttpRequest) -> HttpResponse:
    """Figure gallery for one EDA run, with a run selector."""
    store = get_store()
    runs = store.list_eda_runs()

    try:
        eda_path = store.resolve_eda_run(request.GET.get("run"))
    except RunNotFoundError as exc:
        return render(request, "views/eda_missing.html", {"detail": str(exc)})

    summary = eda_service.dataset_summary(eda_path)
    stats = [
        {"label": label, "value": fmt(summary[key])}
        for key, label, fmt in _SUMMARY_FIELDS
        if summary.get(key) is not None
    ]

    figures: list[dict[str, Any]] = []
    for figure in eda_service.available_figures(eda_path):
        try:
            chart = eda_charts.render(eda_service.chart_data(eda_path, figure["name"]))
        except eda_service.ChartDataUnavailableError:
            # The PNG exists but its source CSV does not; fall back to the image.
            chart = None
        figures.append({**figure, "chart": chart})

    return render(
        request,
        "views/eda.html",
        {
            "run": eda_path.name,
            "runs": runs,
            "stats": stats,
            "figures": figures,
        },
    )


@require_GET
def eda_figure(request: HttpRequest, name: str) -> HttpResponse:
    """Serve one figure PNG from an EDA run (used by the lightbox)."""
    store = get_store()
    try:
        eda_path = store.resolve_eda_run(request.GET.get("run"))
        path = eda_service.figure_path(eda_path, name)
    except (RunNotFoundError, eda_service.ChartDataUnavailableError) as exc:
        raise Http404(str(exc)) from exc
    return FileResponse(path.open("rb"), content_type="image/png")
