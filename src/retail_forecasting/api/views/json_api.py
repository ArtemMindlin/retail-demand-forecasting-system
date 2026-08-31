"""The JSON surface that survives the migration.

Most of the old ``/api/*`` endpoints existed only to feed the React client and
are replaced by server-rendered HTML. This module keeps the subset that is a
genuine operational interface — the one the report documents — so external
callers and the CSV downloads keep working.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from django.conf import settings
from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from retail_forecasting.api.services import forecast as forecast_service
from retail_forecasting.api.services.runs import ArtifactError, NoPredictionsError
from retail_forecasting.api.store import get_store
from retail_forecasting.config import load_config
from retail_forecasting.forecasting.pipeline import run_experiment, run_scoring

logger = logging.getLogger(__name__)

_START_TIME = time.monotonic()

API_VERSION = "2.0.0"


@require_GET
def health(request: HttpRequest) -> JsonResponse:
    """Liveness probe for the reverse proxy and uptime monitoring."""
    uptime_seconds = int(time.monotonic() - _START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return JsonResponse(
        {
            "status": "ok",
            "service": "Retail Demand Forecasting API",
            "version": API_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "data_loaded": get_store().is_loaded,
        }
    )


def _request_params(request: HttpRequest) -> dict[str, Any]:
    """Merge query string and JSON body into one mapping of what-if inputs."""
    data: dict[str, Any] = dict(request.GET.items())
    if request.body:
        try:
            body = json.loads(request.body)
        except (ValueError, UnicodeDecodeError):
            body = None
        if isinstance(body, dict):
            data.update(body)
    return data


@csrf_exempt
@require_POST
def forecast(request: HttpRequest) -> JsonResponse:
    """Conformal band and Newsvendor recommendation for one SKU.

    CSRF-exempt because this is a documented machine-facing endpoint
    authenticated by the session cookie, not a browser form target — the HTML
    dashboard uses the HTMX fragment views instead.
    """
    payload = _request_params(request)
    params = forecast_service.WhatIfParams.from_mapping(payload)
    selected = payload.get("selectedSkuId") or payload.get("series_id")

    store = get_store()
    try:
        grouped = store.grouped_predictions()
    except NoPredictionsError:
        return JsonResponse({"status": "no_predictions"})
    except ArtifactError as exc:
        return JsonResponse({"detail": str(exc)}, status=500)

    return JsonResponse(forecast_service.compute_forecast(grouped, params, selected))


@require_GET
def skus(request: HttpRequest) -> JsonResponse:
    """Per-SKU operational table: coverage, drift, recommended order quantity."""
    params = forecast_service.WhatIfParams.from_mapping(dict(request.GET.items()))
    store = get_store()
    try:
        grouped = store.grouped_predictions()
    except ArtifactError:
        return JsonResponse([], safe=False)
    return JsonResponse(forecast_service.compute_sku_table(grouped, params), safe=False)


def _download_latest(candidates: list[str], download_name: str) -> HttpResponse:
    """Serve the first existing CSV among ``candidates`` from the latest run."""
    store = get_store()
    try:
        run_path = store.latest_run_path()
    except ArtifactError as exc:
        return JsonResponse({"detail": str(exc)}, status=404)

    path = next((run_path / name for name in candidates if (run_path / name).exists()), None)
    if path is None:
        return JsonResponse({"detail": f"{download_name} not found in the latest run."}, status=404)
    return FileResponse(
        path.open("rb"), as_attachment=True, filename=download_name, content_type="text/csv"
    )


@require_GET
def download_predictions(request: HttpRequest) -> HttpResponse:
    """Download ``predictions.csv`` from the latest run."""
    return _download_latest(["predictions.csv"], "predictions.csv")


@require_GET
def download_costs(request: HttpRequest) -> HttpResponse:
    """Download the cost summary from the latest run."""
    return _download_latest(["cost_summary.csv", "costs.csv"], "costs.csv")


# The documented operational surface. Rendered as a reference page at /api/ —
# this is what replaces FastAPI's generated Swagger UI.
ENDPOINTS: tuple[dict[str, str], ...] = (
    {
        "method": "GET",
        "path": "/health",
        "auth": "público",
        "summary": "Sonda de vida: estado, versión, uptime y si hay predicciones cargadas.",
    },
    {
        "method": "POST",
        "path": "/api/forecast",
        "auth": "sesión",
        "summary": (
            "Banda conformal empírica y cantidad Newsvendor para un SKU. Cuerpo JSON con "
            "serviceLevel, shortageCost, holdingCost y selectedSkuId."
        ),
    },
    {
        "method": "GET",
        "path": "/api/skus",
        "auth": "sesión",
        "summary": (
            "Tabla operativa por SKU: cobertura empírica, PSI de deriva, margen y cantidad "
            "recomendada. Acepta service_level, shortage_cost y holding_cost."
        ),
    },
    {
        "method": "GET",
        "path": "/api/download/predictions",
        "auth": "sesión",
        "summary": "Descarga predictions.csv del run más reciente.",
    },
    {
        "method": "GET",
        "path": "/api/download/costs",
        "auth": "sesión",
        "summary": "Descarga el resumen de costes del run más reciente.",
    },
    {
        "method": "POST",
        "path": "/predict_orders",
        "auth": "sesión",
        "summary": (
            "Ejecuta el pipeline en modo score_daily y devuelve las recomendaciones de "
            "reposición. Cuerpo JSON opcional con config_path y run_name."
        ),
    },
)


@require_GET
def api_index(request: HttpRequest) -> HttpResponse:
    """Human-readable reference for the JSON surface."""
    return render(request, "views/api_docs.html", {"endpoints": ENDPOINTS})


@csrf_exempt
@require_POST
def predict_orders(request: HttpRequest) -> JsonResponse:
    """Run the operational scoring pipeline and return the order recommendations.

    Synchronous by design: callers of this endpoint want the recommendations,
    not a job id. The dashboard's own "run pipeline" button uses the background
    console instead.
    """
    payload = _request_params(request)
    config_path = Path(payload.get("config_path") or settings.CONFIG_PATH)
    if not config_path.exists():
        return JsonResponse({"detail": f"Configuration file not found: {config_path}"}, status=404)

    try:
        config = load_config(config_path)
        config = config.model_copy(
            update={"project": config.project.model_copy(update={"run_mode": "score_daily"})}
        )
        if payload.get("run_name"):
            config = config.model_copy(
                update={
                    "reporting": config.reporting.model_copy(
                        update={"run_name": payload["run_name"]}
                    )
                }
            )

        try:
            artifacts = run_scoring(config)
        except FileNotFoundError:
            # No trained champion on disk yet: fall back to a full experiment.
            artifacts = run_experiment(config)
    except Exception as exc:
        logger.exception("Scoring pipeline failed")
        return JsonResponse({"detail": f"Pipeline failed: {exc}"}, status=500)

    if artifacts.run_directory is None or artifacts.reorder_recommendations is None:
        return JsonResponse(
            {"detail": "Pipeline failed to generate operational artifacts."}, status=500
        )

    recommendations = artifacts.reorder_recommendations.fillna(value="")
    get_store().invalidate()

    return JsonResponse(
        {
            "status": "success",
            "run_directory": str(artifacts.run_directory),
            "recommendations_generated": len(recommendations),
            "recommendations": [
                {str(k): v for k, v in row.items()}
                for row in recommendations.to_dict(orient="records")
            ],
        }
    )
