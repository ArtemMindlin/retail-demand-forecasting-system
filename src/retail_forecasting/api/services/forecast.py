"""Empirical conformal intervals, Newsvendor quantities and per-SKU metrics.

Pure computation over an already-loaded predictions frame — no file access, no
framework. Everything here was previously inlined in the FastAPI handlers; it is
extracted so the numbers can be unit-tested without spinning up a web client and
reused by both the HTML views and the JSON endpoints.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.drift.psi import compute_psi

logger = logging.getLogger(__name__)

# Baseline standard-normal equivalent reported alongside the critical ratio.
_BASELINE_Z = 1.65

# PSI above this is conventionally treated as material drift.
PSI_DRIFT_THRESHOLD = 0.20


@dataclass(frozen=True)
class WhatIfParams:
    """The four sliders that drive the operational plane."""

    service_level: float = 95.0
    shortage_cost: float = 18.0
    holding_cost: float = 4.0
    capacity: float = 12000.0

    @property
    def alpha(self) -> float:
        """Miscoverage rate implied by the target service level."""
        return 1.0 - (self.service_level / 100.0)

    @property
    def critical_ratio(self) -> float:
        """Newsvendor critical fractile ``cu / (cu + co)``."""
        return self.shortage_cost / (self.shortage_cost + self.holding_cost)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> WhatIfParams:
        """Build from query params or a JSON body, ignoring unparseable values.

        Accepts both the snake_case query-string names and the camelCase JSON
        names the previous API used, so existing clients keep working.
        """

        def pick(*names: str, default: float) -> float:
            for name in names:
                if name in data and data[name] not in (None, ""):
                    try:
                        return float(data[name])
                    except (TypeError, ValueError):
                        continue
            return default

        return cls(
            service_level=pick("service_level", "serviceLevel", default=95.0),
            shortage_cost=pick("shortage_cost", "shortageCost", default=18.0),
            holding_cost=pick("holding_cost", "holdingCost", default=4.0),
            capacity=pick("capacity", default=12000.0),
        )


@dataclass(frozen=True)
class ConformalStats:
    """Empirical conformal summary for one SKU."""

    residuals: pd.Series = field(repr=False)
    abs_residuals: pd.Series = field(repr=False)
    width: float
    coverage_pct: float


def empirical_conformal(sku_df: pd.DataFrame, alpha: float) -> ConformalStats:
    """Split-conformal interval half-width as the ``1 - alpha`` quantile of |residuals|.

    The coverage returned is in-sample: it is the fraction of the SKU's own
    residuals the band covers, which is exactly ``1 - alpha`` up to ties. It is
    reported so the dashboard can show observed-vs-target, not as an
    out-of-sample guarantee.
    """
    residuals = sku_df["y_true"] - sku_df["y_pred"]
    abs_residuals = residuals.abs()
    width = float(np.quantile(abs_residuals, 1.0 - alpha))
    coverage = float((abs_residuals <= width).mean()) * 100.0
    return ConformalStats(residuals, abs_residuals, width, coverage)


def per_sku_psi(demand: np.ndarray) -> float:
    """PSI of a SKU's own demand, older half vs recent half (0.0 if too short)."""
    half = len(demand) // 2
    if half < 1:
        return 0.0
    return round(compute_psi(demand[:half], demand[half:])[0], 3)


def build_forecast_series(sku_df: pd.DataFrame, conformal_width: float) -> list[dict[str, Any]]:
    """Per-day actual/predicted values with a symmetric conformal band."""
    return [
        {
            "day": idx + 1,
            "label": f"D{idx + 1:02d}",
            "actual": round(float(row.y_true)),
            "predicted": round(float(row.y_pred)),
            "lower": round(max(0.0, float(row.y_pred) - conformal_width)),
            "upper": round(float(row.y_pred) + conformal_width),
        }
        for idx, row in enumerate(sku_df.itertuples())
    ]


def aggregate_inventory_cost(
    sku_df: pd.DataFrame,
    cr_quantile: float,
    params: WhatIfParams,
) -> tuple[float, float]:
    """Newsvendor vs naïve point-forecast inventory cost over the SKU's history.

    Returns ``(newsvendor_cost, delta_vs_naive)``. A negative delta means the
    Newsvendor policy is the cheaper of the two.
    """
    total = 0.0
    naive_total = 0.0
    for row in sku_df.itertuples():
        pred = float(row.y_pred)
        demand = float(row.y_true)
        q = min(max(0.0, pred + cr_quantile), params.capacity)
        q_naive = min(pred, params.capacity)
        total += (
            (demand - q) * params.shortage_cost
            if demand > q
            else (q - demand) * params.holding_cost
        )
        naive_total += (
            (demand - q_naive) * params.shortage_cost
            if demand > q_naive
            else (q_naive - demand) * params.holding_cost
        )
    cost = round(total, 2)
    return cost, round(cost - naive_total, 2)


def compute_forecast(
    grouped: dict[str, pd.DataFrame],
    params: WhatIfParams,
    selected_sku: str | None = None,
) -> dict[str, Any]:
    """Full payload for the dashboard: chart series, KPIs and the order recommendation.

    Returns ``{"status": "no_predictions"}`` when there is nothing to show, which
    the templates render as the empty state.
    """
    if not grouped:
        return {"status": "no_predictions"}

    sku = selected_sku if selected_sku in grouped else next(iter(grouped))
    sku_df = grouped[sku].sort_values("date")

    conformal = empirical_conformal(sku_df, params.alpha)
    cr = params.critical_ratio
    cr_quantile = float(np.quantile(conformal.residuals, cr))

    last_pred = float(sku_df.iloc[-1]["y_pred"])
    q_star = min(max(0.0, last_pred + cr_quantile), params.capacity)

    mae = float(conformal.abs_residuals.mean())
    half = len(conformal.abs_residuals) // 2
    mae_delta = (
        float(
            conformal.abs_residuals.iloc[half:].mean() - conformal.abs_residuals.iloc[:half].mean()
        )
        if half >= 1
        else 0.0
    )

    inventory_cost, inventory_delta = aggregate_inventory_cost(sku_df, cr_quantile, params)
    psi = per_sku_psi(sku_df["y_true"].to_numpy(dtype=float))
    target_cr = params.service_level / 100.0

    return {
        "status": "ok",
        "sku": sku,
        "forecast": build_forecast_series(sku_df, conformal.width),
        "kpis": {
            "inventoryCost": {"value": inventory_cost, "delta": inventory_delta},
            "coverage": {
                "value": conformal.coverage_pct,
                "target": params.service_level,
                "delta": conformal.coverage_pct - params.service_level,
            },
            "mae": {"value": mae, "delta": mae_delta},
            "driftPSI": {"value": psi, "delta": (psi - PSI_DRIFT_THRESHOLD) * 100.0},
        },
        "recommendation": {
            "qStar": round(q_star),
            "z": _BASELINE_Z,
            "criticalRatio": cr,
            "targetCR": target_cr,
            "ratioDelta": (target_cr - cr) * 100.0,
            "utilization": min(100.0, round((q_star / params.capacity) * 100.0, 1)),
        },
    }


def sku_status(coverage_pct: float, drift_psi: float, service_level: float) -> str:
    """Classify a SKU for the status column: drift beats coverage deviation."""
    if drift_psi > PSI_DRIFT_THRESHOLD:
        return "drift"
    if coverage_pct < service_level - 3:
        return "shortage"
    if coverage_pct > service_level + 2:
        return "overstock"
    return "ok"


def sku_category(sku_df: pd.DataFrame) -> str:
    """The SKU's real category, or "N/D" when the artifact does not carry it.

    ``third_category_id`` drives the Mondrian conformal grouping internally but
    is not always persisted into predictions.csv. Returning "N/D" keeps the
    dashboard honest instead of inventing a category.
    """
    if "third_category_id" in sku_df.columns and not sku_df.empty:
        return str(sku_df["third_category_id"].iloc[-1])
    return "N/D"


def compute_sku_table(
    grouped: dict[str, pd.DataFrame],
    params: WhatIfParams,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Per-SKU rows for the SKU table: coverage, drift, order quantity, sparkline."""
    rows: list[dict[str, Any]] = []
    cr = params.critical_ratio

    for sku in list(grouped)[:limit]:
        sku_df = grouped[sku].sort_values("date")
        if sku_df.empty:
            continue

        conformal = empirical_conformal(sku_df, params.alpha)
        cr_quantile = float(np.quantile(conformal.residuals, cr))

        last = sku_df.iloc[-1]
        q_star = min(max(0.0, float(last["y_pred"]) + cr_quantile), params.capacity)
        drift_psi = per_sku_psi(sku_df["y_true"].to_numpy(dtype=float))

        # Cost-asymmetry proxy: the real per-SKU critical fractile when the
        # artifact carries it, otherwise the global one.
        margin = (
            round(float(sku_df["critical_fractile"].iloc[-1]), 2)
            if "critical_fractile" in sku_df.columns
            else round(cr, 2)
        )

        rows.append(
            {
                "id": sku,
                "cat": sku_category(sku_df),
                "series": [float(x) for x in sku_df["y_true"].tail(14)],
                "lastActual": round(float(last["y_true"])),
                "lastPred": round(float(last["y_pred"])),
                "empCoverage": conformal.coverage_pct,
                "coverageTarget": params.service_level,
                "driftPsi": drift_psi,
                "margin": margin,
                "q_star": round(q_star),
                "status": sku_status(conformal.coverage_pct, drift_psi, params.service_level),
            }
        )
    return rows


def load_feature_drift(run_path: Path | None) -> list[dict[str, Any]]:
    """Per-feature PSI from a run's ``drift_report.json``, worst first.

    The artifact is written during an ``experiment`` run: PSI over the
    top-importance features (mean absolute SHAP), older half vs recent half of
    the supervised frame. Runs that predate the artifact return an empty list
    rather than fabricated values.
    """
    if run_path is None:
        return []
    drift_path = run_path / "drift_report.json"
    if not drift_path.exists():
        return []

    try:
        report = json.loads(drift_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to parse drift_report.json at %s", drift_path)
        return []

    if not isinstance(report, list):
        return []
    return sorted(report, key=lambda item: item.get("psi", 0.0), reverse=True)


def load_alerts(run_path: Path | None, limit: int = 5) -> list[dict[str, Any]]:
    """Operational alerts derived from a run's ``exceptions.csv``.

    Empty when no run or no exceptions file exists — the dashboard renders an
    empty state rather than showing synthetic alerts.
    """
    if run_path is None:
        return []
    exceptions_path = run_path / "exceptions.csv"
    if not exceptions_path.exists():
        return []

    try:
        exceptions = pd.read_csv(exceptions_path, nrows=limit)
    except Exception:
        logger.warning("Failed to read %s", exceptions_path)
        return []

    alerts = []
    for idx, row in enumerate(exceptions.itertuples()):
        sku = getattr(row, "series_id", f"SKU-{idx}")
        flag = str(getattr(row, "risk_flag", "high_uncertainty"))
        notes = getattr(row, "notes", "Review recommended.")
        order_qty = getattr(row, "order_quantity", 0)
        severity = "critical" if ("extreme" in flag or "drift" in flag) else "warning"

        alerts.append(
            {
                "id": f"a-{idx + 1:03d}",
                "sev": severity,
                "title": f"Alerta {flag.replace('_', ' ').title()} · {sku}",
                "desc": f"{notes} Cantidad recomendada: {order_qty} u.",
                "meta": [
                    {"k": "SKU", "v": str(sku)},
                    {"k": "Riesgo", "v": flag},
                    {"k": "Orden", "v": f"{order_qty} u"},
                ],
                "read": False,
            }
        )
    return alerts
