"""HTML pages for the dashboard tabs.

Each tab is a real URL rendering a real template, so navigation is browser
history rather than component state.
"""

from __future__ import annotations

import json
from typing import Any

from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import render

from retail_forecasting.api import academic
from retail_forecasting.api.charts import distribution_chart, forecast_chart, sparkline
from retail_forecasting.api.services import forecast as forecast_service
from retail_forecasting.api.services.runs import ArtifactError
from retail_forecasting.api.store import get_store
from retail_forecasting.api.views.empty import empty_state

# Severity colour per PSI status, matching the stylesheet's semantic palette.
_PSI_STATUS_COLORS = {
    "critical": "var(--c-drift)",
    "warning": "var(--c-ai)",
    "ok": "var(--c-conf)",
}

PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = forecast_service.PSI_DRIFT_THRESHOLD


def _kpi_card(
    label: str,
    value: str,
    delta: float,
    *,
    color: str,
    sub: str,
    suffix: str = "",
    delta_unit: str = "%",
    delta_decimals: int = 1,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """Assemble one KPI card, resolving the delta's direction and sentiment.

    "Good" is not the same as "up": for cost, MAE and PSI a fall is the win, so
    ``lower_is_better`` flips which colour the delta chip takes.
    """
    is_up = delta >= 0
    good = (not is_up) if lower_is_better else is_up
    return {
        "label": label,
        "value": value,
        "suffix": suffix,
        "color": color,
        "sub": sub,
        "dot": True,
        "delta_icon": "arrowUp" if is_up else "arrowDown",
        "delta_good": good,
        "delta_text": f"{abs(delta):.{delta_decimals}f}{delta_unit}",
    }


def _params(request: HttpRequest) -> forecast_service.WhatIfParams:
    """What-if inputs for this request, read from the query string."""
    return forecast_service.WhatIfParams.from_mapping(dict(request.GET.items()))


def dashboard(request: HttpRequest) -> HttpResponse:
    """Operational dashboard: KPIs, the forecast chart and the academic modules."""
    store = get_store()
    params = _params(request)
    selected_sku = request.GET.get("sku")
    selected_run = request.GET.get("run")

    try:
        grouped = store.grouped_predictions(selected_run)
    except Exception:
        grouped = {}

    try:
        data = forecast_service.compute_forecast(grouped, params, selected_sku)
    except Exception:
        data = {"status": "no_predictions"}

    if data.get("status") == "no_predictions":
        return empty_state(
            request,
            label="SIN ARTEFACTOS",
            title="Todavía no hay predicciones que mostrar",
            detail=(
                "No se ha encontrado ninguna corrida con predictions.csv en el almacén. "
                "Lanza el pipeline para generar el primer conjunto de recomendaciones."
            ),
            show_run_button=True,
            page_title="Sin predicciones",
        )

    chart = forecast_chart(data["forecast"])
    kpis = data["kpis"]
    # Every per-SKU statistic below is computed on this many forecast origins. Shown
    # because coverage, the MAE trend and the PSI are not interpretable on a handful.
    observations = data.get("observations", 0)
    sample = f"n = {observations}"
    psi = kpis["driftPSI"]["value"]
    runs = store.list_runs()

    context = {
        "runs": runs,
        "run": selected_run or (runs[0] if runs else None),
        "sku": data["sku"],
        "recommendation": data["recommendation"],
        "chart_svg": chart["svg"],
        # Serialized for the pointer handler that positions the hover tooltip.
        "chart_points_json": json.dumps(chart["points"]),
        "chart_width": chart["width"],
        # The tooltip maps user-space y to CSS pixels, so it needs the viewBox height.
        "chart_height": chart["height"],
        "kpi_cards": [
            _kpi_card(
                "Coste de Inventario",
                f"{kpis['inventoryCost']['value']:,.0f} u.m.",
                kpis["inventoryCost"]["delta"],
                color="var(--c-inv)",
                sub="vs. base policy",
                delta_unit=" u.m.",
                delta_decimals=2,
                lower_is_better=True,
            ),
            _kpi_card(
                "Cobertura Empírica",
                f"{kpis['coverage']['value']:.1f}",
                kpis["coverage"]["delta"],
                suffix="%",
                color="var(--c-conf)",
                sub=f"target {kpis['coverage']['target']:.1f}% · {sample}",
                delta_unit=" pts",
            ),
            _kpi_card(
                "Precisión IA · MAE",
                f"{kpis['mae']['value']:.2f}",
                kpis["mae"]["delta"],
                suffix="u",
                color="var(--c-ai)",
                sub=f"media de residuos · {sample}",
                delta_unit=" u",
                delta_decimals=2,
                lower_is_better=True,
            ),
            _kpi_card(
                "Alerta de Drift",
                # None means the sample cannot support a PSI; see per_sku_psi. Showing
                # "n/d" keeps the card honest instead of implying stability or an alarm.
                f"{psi:.3f}" if psi is not None else "n/d",
                kpis["driftPSI"]["delta"],
                suffix="PSI" if psi is not None else "",
                color="var(--c-drift)",
                sub=(
                    "threshold 0.20"
                    if psi is not None
                    else f"muestra insuficiente ({sample}) · ver /drift/"
                ),
                lower_is_better=True,
            ),
        ],
        "academic_cards": [
            academic.card_context(module, params, data["recommendation"])
            for module in academic.MODULES
        ],
    }
    return render(request, "views/dashboard.html", context)


def academic_modal(request: HttpRequest, module_id: str) -> HttpResponse:
    """Academic module slide-over: formula, assumptions, literature references."""
    module = academic.MODULES_BY_ID.get(module_id)
    if module is None:
        raise Http404("Unknown academic module.")

    params = _params(request)
    store = get_store()
    selected_run = request.GET.get("run")
    try:
        grouped = store.grouped_predictions(selected_run)
    except ArtifactError:
        grouped = {}

    data = forecast_service.compute_forecast(grouped, params, request.GET.get("sku"))
    recommendation = data.get("recommendation") or {
        "qStar": 0,
        "z": 0.0,
        "criticalRatio": 0.0,
    }
    return render(
        request,
        "partials/academic_modal.html",
        {"module": academic.modal_context(module, params, recommendation)},
    )


def drift(request: HttpRequest) -> HttpResponse:
    """Feature-drift monitor: per-feature PSI with reference-vs-current histograms."""
    store = get_store()
    selected_run = request.GET.get("run")
    try:
        run_path = store.resolve_run(selected_run) if selected_run else store.latest_run_path()
    except Exception:
        run_path = None

    features = forecast_service.load_feature_drift(run_path)

    # Bar scale: never compress below the critical threshold, so the 0.10/0.20
    # markers stay in meaningful positions even when every feature is stable.
    scale = max([f.get("psi", 0.0) for f in features] + [0.3])

    rows: list[dict[str, Any]] = []
    for feature in features:
        psi = float(feature.get("psi", 0.0))
        status = feature.get("status", "ok")
        color = _PSI_STATUS_COLORS.get(status, "var(--c-conf)")
        rows.append(
            {
                "name": feature.get("name", "—"),
                "type": feature.get("type", "numeric"),
                "importance_pct": round(float(feature.get("importance", 0.0)) * 100),
                "psi": psi,
                "status": status,
                "color": color,
                "bar_pct": (psi / scale) * 100 if scale else 0,
                "chart": distribution_chart(feature.get("pre", []), feature.get("post", []), color),
            }
        )

    critical_count = sum(1 for f in rows if f["status"] == "critical")
    average_psi = sum(f["psi"] for f in rows) / len(rows) if rows else None
    runs = store.list_runs()

    context = {
        "runs": runs,
        "run": selected_run or (runs[0] if runs else None),
        "features": rows,
        "critical_count": critical_count,
        "feature_count": len(rows),
        "average_psi": average_psi,
        "run_name": run_path.name if run_path else None,
        "warning_marker_pct": (PSI_WARNING_THRESHOLD / scale) * 100 if scale else 0,
        "critical_marker_pct": (PSI_CRITICAL_THRESHOLD / scale) * 100 if scale else 0,
    }
    return render(request, "views/drift.html", context)


# Status colour and label, matching the stylesheet's semantic palette.
STATUS_META = {
    "ok": {"color": "var(--c-conf)", "label": "OK"},
    "drift": {"color": "var(--c-drift)", "label": "DRIFT"},
    "shortage": {"color": "var(--c-drift)", "label": "SHORTAGE"},
    "overstock": {"color": "var(--c-inv)", "label": "OVERSTOCK"},
}

# Sortable columns: anything outside this set falls back to PSI.
_SKU_SORT_KEYS = {
    "id",
    "cat",
    "lastActual",
    "lastPred",
    "empCoverage",
    "driftPsi",
    "margin",
    "q_star",
    "status",
}


def _coverage_color(coverage: float, target: float) -> str:
    """Colour the coverage bar by how far it sits from the target band."""
    if coverage < target - 3:
        return "var(--c-drift)"
    if coverage > target + 2:
        return "var(--c-inv)"
    return "var(--c-conf)"


def skus(request: HttpRequest) -> HttpResponse:
    """Per-SKU order control: searchable, filterable, sortable table.

    Search, status filter and sort order all live in the query string and are
    applied server-side. The manual order overrides stay in the browser's
    localStorage — they are a private scratchpad that was never persisted
    server-side before, and still isn't.
    """
    store = get_store()
    params = _params(request)
    selected_run = request.GET.get("run")

    try:
        grouped = store.grouped_predictions(selected_run)
    except Exception:
        grouped = {}

    try:
        rows = forecast_service.compute_sku_table(grouped, params)
    except Exception:
        rows = []

    if not rows:
        return empty_state(
            request,
            label="SIN ARTEFACTOS",
            title="Todavía no hay predicciones que mostrar",
            detail=(
                "No se ha encontrado ninguna corrida con predictions.csv en el almacén. "
                "Lanza el pipeline para generar el primer conjunto de recomendaciones."
            ),
            show_run_button=True,
            page_title="Sin predicciones",
        )

    counts = {
        "all": len(rows),
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "drift": sum(1 for r in rows if r["status"] == "drift"),
        "shortage": sum(1 for r in rows if r["status"] == "shortage"),
        "overstock": sum(1 for r in rows if r["status"] == "overstock"),
    }

    status_filter = request.GET.get("status", "all")
    if status_filter != "all":
        rows = [r for r in rows if r["status"] == status_filter]

    search = request.GET.get("q", "").strip().lower()
    if search:
        rows = [r for r in rows if search in r["id"].lower() or search in r["cat"].lower()]

    sort_key = request.GET.get("sort", "driftPsi")
    if sort_key not in _SKU_SORT_KEYS:
        sort_key = "driftPsi"
    descending = request.GET.get("dir", "desc") != "asc"
    # driftPsi is None whenever a SKU's sample can't support the PSI statistic (see
    # per_sku_psi), and `None < None` raises. Sort the rows that have a value and put
    # the rest at the end, in either sort direction — a missing statistic is not "low".
    sortable = [r for r in rows if r[sort_key] is not None]
    unsortable = [r for r in rows if r[sort_key] is None]
    sortable.sort(key=lambda r: r[sort_key], reverse=descending)
    rows[:] = sortable + unsortable

    for row in rows:
        meta = STATUS_META[row["status"]]
        row["status_color"] = meta["color"]
        row["status_label"] = meta["label"]
        row["coverage_color"] = _coverage_color(row["empCoverage"], row["coverageTarget"])
        row["trend"] = sparkline(row["series"], color=meta["color"])

    columns: list[dict[str, Any]] = [
        {"key": "id", "label": "SKU"},
        {"key": "cat", "label": "Categoría"},
        {"key": None, "label": "14d trend"},
        {"key": "lastActual", "label": "Real"},
        {"key": "lastPred", "label": "Predicho"},
        {"key": "empCoverage", "label": "Cobertura"},
        {"key": "driftPsi", "label": "PSI"},
        {"key": "margin", "label": "Margen"},
        {"key": "q_star", "label": "Pedido Final"},
        {"key": "status", "label": "Estado"},
    ]
    for column in columns:
        if column["key"] == sort_key:
            column["arrow"] = "↓" if descending else "↑"
            # Clicking the active column flips direction.
            column["next_dir"] = "asc" if descending else "desc"
        else:
            column["next_dir"] = "desc"

    runs = store.list_runs()
    return render(
        request,
        "views/skus.html",
        {
            "runs": runs,
            "run": selected_run or (runs[0] if runs else None),
            "rows": rows,
            "counts": counts,
            "status_filter": status_filter,
            "search": request.GET.get("q", ""),
            "sort_key": sort_key,
            "sort_dir": "desc" if descending else "asc",
            "columns": columns,
            "filter_tabs": [
                {"id": "all", "label": "Todos", "count": counts["all"], "color": "var(--text-2)"},
                {"id": "ok", "label": "OK", "count": counts["ok"], "color": "var(--c-conf)"},
                {
                    "id": "drift",
                    "label": "Drift",
                    "count": counts["drift"],
                    "color": "var(--c-drift)",
                },
                {
                    "id": "shortage",
                    "label": "Rotura",
                    "count": counts["shortage"],
                    "color": "var(--c-drift)",
                },
                {
                    "id": "overstock",
                    "label": "Overstock",
                    "count": counts["overstock"],
                    "color": "var(--c-inv)",
                },
            ],
        },
    )
