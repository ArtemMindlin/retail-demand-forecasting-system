"""URL map for the dashboard.

Tab names double as the navigation keys used by the ``navigation`` context
processor, so ``url_name`` in ``context.MODES`` must match the names here.
"""

from __future__ import annotations

from django.urls import path
from django.views.generic import RedirectView

from retail_forecasting.api.views import (
    auth,
    eda,
    experiments,
    json_api,
    ops,
    pages,
    panels,
    pipeline,
)

urlpatterns = [
    # ── Auth ──────────────────────────────────────────────────────────────────
    path("login/", auth.login, name="login"),
    path("logout/", auth.logout, name="logout"),
    # ── Tabs ──────────────────────────────────────────────────────────────────
    path("", RedirectView.as_view(pattern_name="dashboard", permanent=False), name="root"),
    path("dashboard/", pages.dashboard, name="dashboard"),
    path(
        "dashboard/modulo/<slug:module_id>/",
        pages.academic_modal,
        name="academic-modal",
    ),
    path("ops/", ops.ops, name="ops"),
    path("skus/", pages.skus, name="skus"),
    path("drift/", pages.drift, name="drift"),
    path("eda/", eda.eda, name="eda"),
    path("eda/figura/<slug:name>.png", eda.eda_figure, name="eda-figure"),
    path("latent/", experiments.latent, name="latent"),
    path("pareto/", experiments.pareto, name="pareto"),
    # ── Overlay panels ────────────────────────────────────────────────────────
    path("alertas/", panels.alerts, name="alerts-panel"),
    path("alertas/contador/", panels.alerts_badge, name="alerts-badge"),
    path("configuracion/", panels.config, name="config"),
    # ── Pipeline console ──────────────────────────────────────────────────────
    path("pipeline/run/", pipeline.run, name="pipeline-run"),
    path("pipeline/status/", pipeline.status, name="pipeline-status"),
    path("pipeline/reset/", pipeline.reset, name="pipeline-reset"),
    # ── Documented JSON surface ───────────────────────────────────────────────
    path("health", json_api.health, name="health"),
    path("api/", json_api.api_index, name="api-index"),
    path("predict_orders", json_api.predict_orders, name="predict-orders"),
    path("api/forecast", json_api.forecast, name="api-forecast"),
    path("api/skus", json_api.skus, name="api-skus"),
    path(
        "api/download/predictions",
        json_api.download_predictions,
        name="api-download-predictions",
    ),
    path("api/download/costs", json_api.download_costs, name="api-download-costs"),
]
