"""Overlay panels: operational alerts and the configuration editor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods

from retail_forecasting.api.services import forecast as forecast_service
from retail_forecasting.api.services.runs import ArtifactError
from retail_forecasting.api.store import get_store
from retail_forecasting.config import load_config

# Severity styling, mirroring the stylesheet's semantic palette.
SEVERITY_META = {
    "critical": {"color": "var(--c-drift)", "icon": "alertTri", "label": "Crítica"},
    "warning": {"color": "var(--c-ai)", "icon": "info", "label": "Aviso"},
}


@require_GET
def alerts(request: HttpRequest) -> HttpResponse:
    """Slide-over panel listing exceptions from the latest run."""
    store = get_store()
    try:
        run_path = store.latest_run_path()
    except ArtifactError:
        run_path = None

    rows = forecast_service.load_alerts(run_path)
    for row in rows:
        meta = SEVERITY_META.get(row["sev"], SEVERITY_META["warning"])
        row["color"] = meta["color"]
        row["icon"] = meta["icon"]
        row["sev_label"] = meta["label"]

    return render(
        request,
        "partials/alerts_panel.html",
        {"alerts": rows, "run_name": run_path.name if run_path else None},
    )


@require_GET
def alerts_badge(request: HttpRequest) -> HttpResponse:
    """Just the count, so the top bar can show it without the panel open."""
    store = get_store()
    try:
        run_path = store.latest_run_path()
    except ArtifactError:
        run_path = None
    return render(
        request,
        "partials/alerts_badge.html",
        {"count": len(forecast_service.load_alerts(run_path))},
    )


def _validate_config(text: str) -> str | None:
    """Return an error message when ``text`` is not a loadable configuration.

    Validation happens against a temporary file so a bad edit can never
    overwrite the live configuration.
    """
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return f"YAML inválido: {exc}"

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        load_config(tmp_path)
    except Exception as exc:
        return f"La validación de la configuración falló: {exc}"
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return None


@require_http_methods(["GET", "POST"])
def config(request: HttpRequest) -> HttpResponse:
    """View and edit ``configs/experiment.yaml`` with full validation."""
    path: Path = settings.CONFIG_PATH

    if request.method == "GET":
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        return render(
            request,
            "partials/config_modal.html",
            {
                "yaml_text": text,
                "path": str(path),
                "missing": not path.exists(),
            },
        )

    text = request.POST.get("yaml", "")
    error = _validate_config(text)
    if error:
        return render(
            request,
            "partials/config_modal.html",
            {"yaml_text": text, "path": str(path), "error": error},
            status=400,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    # The champion strategy lives in this file, so the cached frame may now be
    # filtered on the wrong model.
    get_store().invalidate()

    return render(
        request,
        "partials/config_modal.html",
        {"yaml_text": text, "path": str(path), "saved": True},
    )
