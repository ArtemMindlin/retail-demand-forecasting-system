"""Navigation context shared by every page.

Ports the two-level ``MODES`` structure that lived in the React ``App``
component: a top-level mode (operations vs research) selecting a row of tabs.
Here the active tab comes from the resolved URL name rather than from
``useState``, so navigation is real URLs and the back button works.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest

from retail_forecasting.api.services.forecast import WhatIfParams

MODES: list[dict[str, Any]] = [
    {
        "id": "operacion",
        "label": "Operación",
        "icon": "zap",
        "tabs": [
            {"id": "ops", "label": "Backtest OPS", "icon": "activity", "url_name": "ops"},
            {"id": "dashboard", "label": "Dashboard", "icon": "cpu", "url_name": "dashboard"},
            {"id": "skus", "label": "Análisis SKU", "icon": "package", "url_name": "skus"},
            {
                "id": "drift",
                "label": "Monitor de Drift",
                "icon": "trendingUp",
                "url_name": "drift",
            },
        ],
    },
    {
        "id": "analisis",
        "label": "Análisis",
        "icon": "sigma",
        "tabs": [
            {"id": "eda", "label": "EDA", "icon": "layers", "url_name": "eda"},
            {"id": "latent", "label": "Demanda Latente", "icon": "eye", "url_name": "latent"},
            {"id": "pareto", "label": "Pareto Tuning", "icon": "target", "url_name": "pareto"},
        ],
    },
]

# Tabs that show the what-if sidebar; the rest get the full-width canvas.
SIDEBAR_TABS = frozenset({"dashboard", "skus", "drift"})

_TAB_TO_MODE = {tab["id"]: mode["id"] for mode in MODES for tab in mode["tabs"]}


def navigation(request: HttpRequest) -> dict[str, Any]:
    """Expose the mode/tab tree and which one is active for this request."""
    match = request.resolver_match
    active_tab = match.url_name if match else None
    active_mode = _TAB_TO_MODE.get(active_tab or "", "operacion")

    return {
        "modes": MODES,
        "active_tab": active_tab,
        "active_mode": active_mode,
        "show_sidebar": active_tab in SIDEBAR_TABS,
    }


def whatif(request: HttpRequest) -> dict[str, Any]:
    """Expose the what-if sliders, read from the query string.

    Keeping the parameters in the URL rather than in component state is what
    makes every dashboard screen linkable and reloadable — a scenario can be
    shared by copying the address bar.
    """
    params = WhatIfParams.from_mapping(dict(request.GET.items()))
    return {
        "params": params,
        "critical_ratio": params.critical_ratio,
        "alpha": params.alpha,
    }
