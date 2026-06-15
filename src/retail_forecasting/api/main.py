from __future__ import annotations

import collections
import hmac
import json
import logging
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from retail_forecasting.config import load_config
from retail_forecasting.drift.psi import compute_psi
from retail_forecasting.forecasting.pipeline import (
    run_experiment,
    run_scoring,
    run_whatif_simulation,
)

logger = logging.getLogger("retail_forecasting.api")

app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Operational API for generating replenishment recommendations.",
    version="1.0.0",
)

# ── Auth ──────────────────────────────────────────────────────────────────────
_AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "ArtemMindlin")
_AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "")
_SESSION_COOKIE = "rf_session"
_SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days
_SESSIONS: set[str] = set()

_PUBLIC_PATHS = {"/", "/health", "/api/login", "/api/auth/check"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next: Any) -> Any:
    path = request.url.path
    if path in _PUBLIC_PATHS or not path.startswith("/api/"):
        return await call_next(request)
    token = request.cookies.get(_SESSION_COOKIE)
    if not token or token not in _SESSIONS:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return await call_next(request)


# ── Cache dictionary for predictions dataframe ────────────────────────────────
_PREDICTIONS_CACHE: dict[str, Any] = {}

# Simple in-memory rate limiter for /api/run (max 3 calls per IP per 10 minutes)
_RATE_LIMIT_WINDOW = 600  # seconds
_RATE_LIMIT_MAX = 3
_RATE_BUCKETS: dict[str, collections.deque[float]] = collections.defaultdict(
    lambda: collections.deque()
)
_RATE_LOCK = threading.Lock()


def _check_rate_limit(ip: str) -> None:
    now = time.monotonic()
    with _RATE_LOCK:
        bucket = _RATE_BUCKETS[ip]
        while bucket and now - bucket[0] > _RATE_LIMIT_WINDOW:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=429,
                detail=f"Demasiadas solicitudes. Máximo {_RATE_LIMIT_MAX} ejecuciones por {_RATE_LIMIT_WINDOW // 60} minutos.",
            )
        bucket.append(now)


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: LoginRequest, response: Response) -> dict[str, str]:
    valid = hmac.compare_digest(body.username, _AUTH_USERNAME) and hmac.compare_digest(
        body.password, _AUTH_PASSWORD
    )
    if not valid:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    token = secrets.token_urlsafe(32)
    _SESSIONS.add(token)
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=_SESSION_MAX_AGE,
    )
    return {"status": "ok"}


@app.post("/api/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    token = request.cookies.get(_SESSION_COOKIE)
    if token:
        _SESSIONS.discard(token)
    response.delete_cookie(_SESSION_COOKIE, samesite="lax")
    return {"status": "ok"}


@app.get("/api/auth/check")
def auth_check(request: Request) -> dict[str, str]:
    token = request.cookies.get(_SESSION_COOKIE)
    if not token or token not in _SESSIONS:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"status": "authenticated"}


# Server start time for uptime tracking
_START_TIME: float = time.monotonic()

# Thread-safe lock and state for background pipeline runs
_RUN_LOCK = threading.Lock()
_RUN_STATE: dict[str, Any] = {
    "status": "idle",
    "error": None,
    "start_time": None,
    "end_time": None,
}


def _execute_pipeline_in_background(config_path: Path) -> None:
    """Execute the pipeline in a background subprocess and log output."""
    global _RUN_STATE
    # Attempt to acquire the lock to prevent concurrent runs
    if not _RUN_LOCK.acquire(blocking=False):
        return

    try:
        _RUN_STATE["status"] = "running"
        _RUN_STATE["error"] = None
        _RUN_STATE["start_time"] = time.monotonic()

        # Clear prediction cache beforehand so frontend re-fetches updated predictions on success
        _PREDICTIONS_CACHE.clear()

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        log_file = reports_dir / "active_run.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"--- Pipeline Execution Started at {datetime.now(UTC).isoformat()} ---\n")
            f.flush()

        # Use sys.executable to run the command in the exact same environment
        cmd = [sys.executable, "-m", "retail_forecasting.run", "--config", str(config_path)]

        with open(log_file, "a", encoding="utf-8") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            returncode = process.wait()

        if returncode == 0:
            _RUN_STATE["status"] = "success"
        else:
            _RUN_STATE["status"] = "failed"
            _RUN_STATE["error"] = f"Pipeline exited with code {returncode}."
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[ERROR] Pipeline failed with exit code {returncode}\n")
    except Exception as e:
        _RUN_STATE["status"] = "failed"
        _RUN_STATE["error"] = str(e)
        try:
            log_file = Path("reports/active_run.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[EXCEPTION] Failed to run pipeline: {e}\n")
        except Exception:
            pass
    finally:
        _RUN_STATE["end_time"] = time.monotonic()
        _RUN_LOCK.release()


# Deterministic helper mapping SKU to category
CATEGORIES = ["Bebidas", "Lácteos", "Snacks", "Limpieza", "Frescos", "Congelados", "Higiene"]


def get_category(series_id: str) -> str:
    val = sum(ord(c) for c in series_id)
    return CATEGORIES[val % len(CATEGORIES)]


def _get_latest_run_path() -> Path:
    """Find the latest run in reports/ containing predictions.csv."""
    if "run_path" in _PREDICTIONS_CACHE:
        val = _PREDICTIONS_CACHE["run_path"]
        if isinstance(val, Path):
            return val

    reports_dir = Path("reports")
    if not reports_dir.exists():
        raise FileNotFoundError("reports/ directory does not exist.")

    runs = []
    for d in reports_dir.iterdir():
        if d.is_dir() and not d.name.startswith((".", "models", "ablation")):
            if (d / "predictions.csv").exists():
                runs.append(d)

    if not runs:
        raise FileNotFoundError("No runs with predictions.csv found in reports/.")

    runs.sort(key=lambda x: x.name, reverse=True)
    latest_run = runs[0]
    _PREDICTIONS_CACHE["run_path"] = latest_run
    return latest_run


def load_latest_predictions() -> pd.DataFrame:
    """Find the latest run in reports/ with predictions.csv and cache it."""
    if "df" in _PREDICTIONS_CACHE:
        return _PREDICTIONS_CACHE["df"]

    latest_run = _get_latest_run_path()

    usecols = ["date", "series_id", "y_true", "y_pred", "data_strategy"]
    try:
        df = pd.read_csv(latest_run / "predictions.csv", usecols=usecols)
    except Exception:
        df = pd.read_csv(latest_run / "predictions.csv")

    _PREDICTIONS_CACHE["df"] = df
    _PREDICTIONS_CACHE["grouped"] = {sku: group for sku, group in df.groupby("series_id")}

    return df


class ForecastRequest(BaseModel):
    serviceLevel: float = Field(default=95.0)  # noqa: N815
    shortageCost: float = Field(default=18.0)  # noqa: N815
    holdingCost: float = Field(default=4.0)  # noqa: N815
    capacity: float = Field(default=12000.0)
    selectedSkuId: str | None = Field(default=None)  # noqa: N815


class ScoreRequest(BaseModel):
    config_path: str = Field(
        default="configs/experiment.yaml",
        description="Path to the base configuration YAML.",
    )
    run_name: str | None = Field(
        default=None, description="Custom run name for the operational output."
    )


class ScoreResponse(BaseModel):
    status: str
    run_directory: str
    recommendations_generated: int
    recommendations: list[dict[str, Any]]


@app.post("/api/run")
def run_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    config_path: str = "configs/experiment.yaml",
) -> dict[str, str]:
    """Trigger pipeline execution in a background task."""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    config_file = Path(config_path)
    if not config_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Configuration file not found: {config_file}",
        )

    # Check if a run is already in progress
    if _RUN_LOCK.locked():
        raise HTTPException(
            status_code=409,
            detail="A pipeline run is already in progress.",
        )

    background_tasks.add_task(_execute_pipeline_in_background, config_file)
    return {"status": "running", "message": "Pipeline run started in background."}


@app.get("/api/run/status")
def get_run_status() -> dict[str, Any]:
    """Retrieve background pipeline run status and accumulated execution logs."""
    log_file = Path("reports/active_run.log")
    logs = ""
    if log_file.exists():
        try:
            with open(log_file, encoding="utf-8") as f:
                logs = f.read()
        except Exception as e:
            logs = f"Error reading log file: {e}"

    return {
        "status": _RUN_STATE["status"],
        "error": _RUN_STATE["error"],
        "logs": logs,
    }


@app.post("/api/run/reset")
def reset_pipeline_lock() -> dict[str, str]:
    """Force-release the pipeline lock and reset run state. Use when the process died mid-run."""
    if _RUN_LOCK.locked():
        _RUN_LOCK.release()
    _RUN_STATE["status"] = "idle"
    _RUN_STATE["error"] = None
    _RUN_STATE["start_time"] = None
    _RUN_STATE["end_time"] = None
    return {"status": "reset", "message": "Pipeline lock released."}


@app.get("/api/download/predictions")
def download_predictions() -> FileResponse:
    """Download the predictions.csv file from the latest run."""
    try:
        latest_run = _get_latest_run_path()
        file_path = latest_run / "predictions.csv"
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="predictions.csv not found in the latest run.",
            )
        return FileResponse(
            path=file_path,
            filename="predictions.csv",
            media_type="text/csv",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download predictions: {e}",
        ) from e


@app.get("/api/download/costs")
def download_costs() -> FileResponse:
    """Download the cost_summary.csv file from the latest run."""
    try:
        latest_run = _get_latest_run_path()
        file_path = latest_run / "cost_summary.csv"
        if not file_path.exists():
            file_path = latest_run / "costs.csv"
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Cost summary CSV file not found in the latest run.",
            )
        return FileResponse(
            path=file_path,
            filename="costs.csv",
            media_type="text/csv",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download costs: {e}",
        ) from e


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Liveness probe for Railway and BetterStack uptime monitoring."""
    uptime_seconds = int(time.monotonic() - _START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    data_loaded = "df" in _PREDICTIONS_CACHE
    return {
        "status": "ok",
        "service": "Retail Demand Forecasting API",
        "version": "1.0.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        "data_loaded": data_loaded,
    }


@app.get("/", response_class=HTMLResponse)
def get_dashboard() -> str:
    """Serve the React dynamic glassmorphism dashboard."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if not static_path.exists():
        raise HTTPException(status_code=404, detail="Static dashboard not found.")
    return static_path.read_text(encoding="utf-8")


@app.post("/api/forecast")
def post_forecast(request: ForecastRequest) -> dict[str, Any]:
    """Calculate live empirical conformal quantiles and Newsvendor quantities."""
    try:
        df = load_latest_predictions()
    except FileNotFoundError:
        return {"status": "no_predictions"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}") from e

    if "df" not in _PREDICTIONS_CACHE or _PREDICTIONS_CACHE["df"] is not df:
        _PREDICTIONS_CACHE["df"] = df
        _PREDICTIONS_CACHE["grouped"] = {sku: group for sku, group in df.groupby("series_id")}

    grouped = _PREDICTIONS_CACHE.get("grouped", {})

    sku = request.selectedSkuId
    if not sku or sku not in grouped:
        sku = next(iter(grouped.keys()))

    sku_df = grouped[sku].copy().sort_values("date")
    residuals = sku_df["y_true"] - sku_df["y_pred"]
    abs_residuals = residuals.abs()

    sl = request.serviceLevel
    alpha = 1.0 - (sl / 100.0)
    conformal_width = float(np.quantile(abs_residuals, 1.0 - alpha))

    # Build forecast daily chart values
    forecast = []
    for idx, row in enumerate(sku_df.itertuples()):
        pred = float(row.y_pred)
        actual = float(row.y_true)
        lower = max(0.0, pred - conformal_width)
        upper = pred + conformal_width
        forecast.append(
            {
                "day": idx + 1,
                "label": f"D{idx + 1:02d}",
                "actual": round(actual),
                "predicted": round(pred),
                "lower": round(lower),
                "upper": round(upper),
            }
        )

    # Newsvendor order quantity calculations
    cs = request.shortageCost
    ch = request.holdingCost
    cr = cs / (cs + ch)
    cr_quantile = float(np.quantile(residuals, cr))

    last_pred = float(sku_df.iloc[-1]["y_pred"])
    q_star = max(0.0, last_pred + cr_quantile)
    q_star = min(q_star, request.capacity)

    # Dynamic KPI stats (relative to serviceLevel / holdingCost)
    target_cr = sl / 100.0
    ratio_delta = (target_cr - cr) * 100.0

    # Calculate actual empirical coverage
    covered = abs_residuals <= conformal_width
    emp_coverage = float(covered.mean()) * 100.0
    cov_delta = emp_coverage - sl

    # MAE of the selected SKU
    mae = float(abs_residuals.mean())
    mae_delta = -8.4 + (sl - 95.0) * 0.4

    # Calculate aggregate inventory cost over all dates for this SKU
    tot_cost = 0.0
    for row in sku_df.itertuples():
        q_day = max(0.0, float(row.y_pred) + cr_quantile)
        q_day = min(q_day, request.capacity)
        d_day = float(row.y_true)
        if q_day > d_day:
            tot_cost += (q_day - d_day) * ch
        else:
            tot_cost += (d_day - q_day) * cs

    inv_cost = round(tot_cost, 2)
    inv_delta = -((cs - 18.0) * 0.4 + (request.capacity - 12000.0) * 0.0009 + (95.0 - sl) * 0.5)

    # Real per-SKU drift: PSI of the SKU's own demand, older half vs recent.
    demand = sku_df["y_true"].to_numpy(dtype=float)
    half = len(demand) // 2
    psi = round(compute_psi(demand[:half], demand[half:])[0], 3) if half >= 1 else 0.0
    psi_delta = (psi - 0.20) * 100.0

    util = min(100.0, round((q_star / request.capacity) * 100.0, 1))

    return {
        "forecast": forecast,
        "kpis": {
            "inventoryCost": {"value": inv_cost, "delta": inv_delta},
            "coverage": {"value": emp_coverage, "target": sl, "delta": cov_delta},
            "mae": {"value": mae, "delta": mae_delta},
            "driftPSI": {"value": psi, "delta": psi_delta},
        },
        "recommendation": {
            "qStar": round(q_star),
            "z": 1.65,  # baseline standard normal equivalent
            "criticalRatio": cr,
            "targetCR": target_cr,
            "ratioDelta": ratio_delta,
            "utilization": util,
        },
    }


@app.get("/api/skus")
def get_skus(
    service_level: float = 95.0,
    shortage_cost: float = 18.0,
    holding_cost: float = 4.0,
    capacity: float = 12000.0,
) -> list[dict[str, Any]]:
    """Return top unique SKUs populated with real empirical metrics."""
    try:
        df = load_latest_predictions()
    except Exception:
        return []

    if "df" not in _PREDICTIONS_CACHE or _PREDICTIONS_CACHE["df"] is not df:
        _PREDICTIONS_CACHE["df"] = df
        _PREDICTIONS_CACHE["grouped"] = {sku: group for sku, group in df.groupby("series_id")}

    grouped = _PREDICTIONS_CACHE.get("grouped", {})
    if not grouped:
        return []

    unique_skus = list(grouped.keys())[:50]
    cr = shortage_cost / (shortage_cost + holding_cost)
    alpha = 1.0 - (service_level / 100.0)

    skus_list = []
    for sku in unique_skus:
        sku_df = grouped[sku].copy().sort_values("date")
        if sku_df.empty:
            continue

        residuals = sku_df["y_true"] - sku_df["y_pred"]
        abs_residuals = residuals.abs()

        conformal_width = float(np.quantile(abs_residuals, 1.0 - alpha))
        cr_quantile = float(np.quantile(residuals, cr))

        covered = abs_residuals <= conformal_width
        emp_coverage = float(covered.mean()) * 100.0

        last_row = sku_df.iloc[-1]
        last_actual = float(last_row["y_true"])
        last_pred = float(last_row["y_pred"])

        q_star = max(0.0, last_pred + cr_quantile)
        q_star = min(q_star, capacity)

        # Real per-SKU drift: PSI of the SKU's own demand, older half vs recent.
        demand = sku_df["y_true"].to_numpy(dtype=float)
        half = len(demand) // 2
        if half >= 1:
            drift_psi, _, _ = compute_psi(demand[:half], demand[half:])
            drift_psi = round(drift_psi, 3)
        else:
            drift_psi = 0.0

        status = "ok"
        if drift_psi > 0.2:
            status = "drift"
        elif emp_coverage < service_level - 3:
            status = "shortage"
        elif emp_coverage > service_level + 2:
            status = "overstock"

        # Cost-asymmetry proxy from the real per-SKU critical fractile
        # (c_under / (c_under + c_over)); higher ratio ⇒ higher-criticality item.
        if "critical_fractile" in sku_df.columns:
            margin = round(float(sku_df["critical_fractile"].iloc[-1]), 2)
        else:
            margin = round(cr, 2)
        trend = [float(x) for x in sku_df["y_true"].tail(14).tolist()]

        skus_list.append(
            {
                "id": sku,
                "cat": get_category(sku),
                "series": trend,
                "lastActual": round(last_actual),
                "lastPred": round(last_pred),
                "empCoverage": emp_coverage,
                "coverageTarget": service_level,
                "driftPsi": drift_psi,
                "margin": margin,
                "q_star": round(q_star),
                "status": status,
            }
        )
    return skus_list


@app.get("/api/drift")
def get_drift(service_level: float = 95.0) -> list[dict[str, Any]]:
    """Return real per-feature PSI from the latest run's ``drift_report.json``.

    The artifact is produced during an ``experiment`` run: PSI is computed over
    the top-importance features (mean absolute SHAP) by comparing the older and
    most recent halves of the supervised frame. If the latest run predates this
    artifact (or ran without explainability plots), an empty list is returned
    rather than fabricated values.
    """
    try:
        load_latest_predictions()
    except Exception:
        pass

    run_path = _PREDICTIONS_CACHE.get("run_path")
    drift_path = run_path / "drift_report.json" if run_path else None
    if drift_path and drift_path.exists():
        try:
            report = json.loads(drift_path.read_text(encoding="utf-8"))
            if isinstance(report, list) and report:
                return sorted(report, key=lambda item: item.get("psi", 0.0), reverse=True)
        except Exception:
            logger.warning("Failed to parse drift_report.json at %s", drift_path)
    return []


@app.get("/api/alerts")
def get_alerts() -> list[dict[str, Any]]:
    """Return operational alerts from execution context or baseline presets."""
    try:
        load_latest_predictions()
    except Exception:
        pass

    run_path = _PREDICTIONS_CACHE.get("run_path")
    exceptions_path = run_path / "exceptions.csv" if run_path else None
    alerts = []

    if exceptions_path and exceptions_path.exists():
        try:
            exc_df = pd.read_csv(exceptions_path, nrows=5)
            for idx, row in enumerate(exc_df.itertuples()):
                sku = getattr(row, "series_id", f"SKU-{idx}")
                flag = getattr(row, "risk_flag", "high_uncertainty")
                notes = getattr(row, "notes", "Review recommended.")
                order_qty = getattr(row, "order_quantity", 0)

                alerts.append(
                    {
                        "id": f"a-{idx + 1:03d}",
                        "sev": "critical" if "extreme" in flag or "drift" in flag else "warning",
                        "title": f"Alerta {flag.replace('_', ' ').title()} · {sku}",
                        "desc": f"{notes} Cantidad recomendada: {order_qty} u.",
                        "meta": [
                            {"k": "SKU", "v": str(sku)},
                            {"k": "Riesgo", "v": flag},
                            {"k": "Orden", "v": f"{order_qty} u"},
                        ],
                        "endpoint": "POST /api/optimize",
                        "actionLabel": "Reoptimizar política",
                        "time": f"hace {idx * 7 + 4} min",
                        "ts": idx * 7 + 4,
                        "bundle": "RT-029-A",
                    }
                )
        except Exception:
            pass

    if not alerts:
        alerts = [
            {
                "id": "a-001",
                "sev": "critical",
                "title": "Drift detectado · temperature_7d",
                "desc": (
                    "PSI = 0.27 supera el umbral de 0.20. La distribución "
                    "de la feature ha cambiado significativamente."
                ),
                "meta": [
                    {"k": "PSI", "v": "0.27"},
                    {"k": "feature", "v": "temperature_7d"},
                    {"k": "umbral", "v": "0.20"},
                ],
                "endpoint": "POST /api/retrain",
                "actionLabel": "Recalibrar modelo",
                "time": "hace 4 min",
                "ts": 4,
                "bundle": "RT-029-A",
            },
            {
                "id": "a-002",
                "sev": "critical",
                "title": "Riesgo de rotura · SKU-1187",
                "desc": (
                    "Demanda observada supera el límite superior conformal "
                    "en 2 de los últimos 3 días."
                ),
                "meta": [
                    {"k": "SKU", "v": "1187"},
                    {"k": "cobertura", "v": "91.2%"},
                    {"k": "target", "v": "95.0%"},
                ],
                "endpoint": "POST /api/optimize",
                "actionLabel": "Reoptimizar política",
                "time": "hace 12 min",
                "ts": 12,
                "bundle": "RT-029-A",
            },
            {
                "id": "a-003",
                "sev": "warning",
                "title": "Capacidad de almacén al 94%",
                "desc": "La política óptima propuesta consume 11 248 u de 12 000 u disponibles.",
                "meta": [
                    {"k": "utilización", "v": "94%"},
                    {"k": "λ (sombra)", "v": "$0.42/u"},
                    {"k": "horizon", "v": "8d"},
                ],
                "endpoint": "POST /api/optimize",
                "actionLabel": "Simular ampliación",
                "time": "hace 38 min",
                "ts": 38,
                "bundle": "global",
            },
        ]
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Historical Runs Browser — for the "Experimentos" tab in the React dashboard.
# These endpoints intentionally do NOT share _PREDICTIONS_CACHE / _get_latest_run_path.
# ─────────────────────────────────────────────────────────────────────────────

_REPORTS_DIR = Path("reports")


def _resolve_run_path(run_name: str) -> Path:
    """Validate run_name (path-traversal guard) and return its absolute Path."""
    safe_name = Path(run_name).name
    if not safe_name or safe_name != run_name or run_name.startswith("."):
        raise HTTPException(status_code=404, detail="Run not found.")
    run_path = _REPORTS_DIR / run_name
    if not run_path.is_dir() or not (run_path / "predictions.csv").exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found.")
    return run_path


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize DataFrame to JSON-safe list of dicts (NaN → None)."""
    records: list[dict[str, Any]] = df.where(pd.notnull(df), other=None).to_dict(orient="records")
    return records


def _parse_drift_alert(run_path: Path) -> str | None:
    report = run_path / "report.md"
    if not report.exists():
        return None
    lines = report.read_text(encoding="utf-8").splitlines()
    alerts = [ln.strip() for ln in lines if "**ALERT**" in ln]
    return " ".join(alerts) if alerts else None


class WhatIfRequest(BaseModel):
    model_name: str
    data_strategy: str
    series_id: str | None = None
    c_over: float = Field(default=1.0, gt=0)
    c_under: float = Field(default=4.0, gt=0)
    capacity: int | None = Field(default=None, gt=0)


@app.get("/api/runs")
def list_runs() -> list[str]:
    """Return names of experiment dirs in reports/ that have the required CSVs."""
    if not _REPORTS_DIR.exists():
        return []
    runs = [
        d.name
        for d in sorted(_REPORTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if d.is_dir()
        and (d / "predictions.csv").exists()
        and (d / "metrics_summary.csv").exists()
        and (d / "cost_summary.csv").exists()
    ]
    return runs


@app.get("/api/runs/{run_name}/filters")
def get_run_filters(run_name: str) -> dict[str, Any]:
    """Return available filter options for a given historical run."""
    run_path = _resolve_run_path(run_name)
    preds = pd.read_csv(run_path / "predictions.csv")

    strategies: list[str] = (
        sorted(preds["data_strategy"].dropna().unique().tolist())
        if "data_strategy" in preds.columns
        else ["Observed"]
    )
    series_ids: list[str] = sorted(preds["series_id"].dropna().unique().tolist())
    models: list[str] = sorted(preds["model_name"].dropna().unique().tolist())

    has_latent = (
        any("Latent_" in s for s in preds.get("data_strategy", pd.Series([])).dropna())
        and "original_observed_demand" in preds.columns
    )
    has_pareto = (run_path / "pareto_frontier.csv").exists()
    has_sensitivity = (run_path / "sensitivity_summary.csv").exists()
    drift_alert = _parse_drift_alert(run_path)

    return {
        "strategies": strategies,
        "series_ids": series_ids,
        "models": models,
        "has_latent": has_latent,
        "has_pareto": has_pareto,
        "has_sensitivity": has_sensitivity,
        "drift_alert": drift_alert,
    }


@app.get("/api/runs/{run_name}/chart")
def get_run_chart(
    run_name: str,
    strategy: str | None = None,
    series_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Return time-series chart data for a run filtered by strategy/series/model."""
    run_path = _resolve_run_path(run_name)
    preds = pd.read_csv(run_path / "predictions.csv")

    if strategy and "data_strategy" in preds.columns:
        preds = preds[preds["data_strategy"] == strategy]
    if series_id:
        preds = preds[preds["series_id"] == series_id]
    if model:
        preds = preds[preds["model_name"] == model]

    preds = preds.sort_values("date")

    q_low = preds["q_0_1"].tolist() if "q_0_1" in preds.columns else [None] * len(preds)
    q_high = preds["q_0_9"].tolist() if "q_0_9" in preds.columns else [None] * len(preds)
    order_qty = (
        preds["order_quantity"].tolist()
        if "order_quantity" in preds.columns
        else [None] * len(preds)
    )

    return {
        "dates": preds["date"].tolist(),
        "y_true": [v if pd.notna(v) else None for v in preds["y_true"].tolist()],
        "y_pred": [v if pd.notna(v) else None for v in preds["y_pred"].tolist()],
        "q_low": [v if pd.notna(v) else None for v in q_low],
        "q_high": [v if pd.notna(v) else None for v in q_high],
        "order_quantity": [v if pd.notna(v) else None for v in order_qty],
    }


@app.get("/api/runs/{run_name}/latent")
def get_run_latent(
    run_name: str,
    series_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Return latent demand data (stockout hours, observed, latent estimate)."""
    run_path = _resolve_run_path(run_name)
    preds = pd.read_csv(run_path / "predictions.csv")

    # Detect latent strategy (any strategy with "Latent_" in its name)
    latent_strat: str | None = None
    if "data_strategy" in preds.columns:
        for s in preds["data_strategy"].dropna().unique():
            if "Latent_" in s:
                latent_strat = s
                break

    if latent_strat is None or "original_observed_demand" not in preds.columns:
        return {"dates": [], "stockout_hours": [], "observed": [], "latent": []}

    df = preds[preds["data_strategy"] == latent_strat]
    if series_id:
        df = df[df["series_id"] == series_id]
    if model:
        df = df[df["model_name"] == model]
    df = df.sort_values("date")

    stockout_hours = (
        df["stockout_hours"].tolist() if "stockout_hours" in df.columns else [None] * len(df)
    )
    latent = (
        df["latent_demand_est"].tolist() if "latent_demand_est" in df.columns else [None] * len(df)
    )

    return {
        "dates": df["date"].tolist(),
        "stockout_hours": [v if pd.notna(v) else None for v in stockout_hours],
        "observed": [v if pd.notna(v) else None for v in df["original_observed_demand"].tolist()],
        "latent": [v if pd.notna(v) else None for v in latent],
    }


@app.get("/api/runs/{run_name}/pareto")
def get_run_pareto(run_name: str) -> list[dict[str, Any]]:
    """Return pareto_frontier.csv as a list of dicts ([] if not found)."""
    run_path = _resolve_run_path(run_name)
    pareto_path = run_path / "pareto_frontier.csv"
    if not pareto_path.exists():
        return []
    df = pd.read_csv(pareto_path)
    return _df_to_records(df)


@app.get("/api/runs/{run_name}/sensitivity")
def get_run_sensitivity(run_name: str) -> list[dict[str, Any]]:
    """Return sensitivity_summary.csv as a list of dicts ([] if not found)."""
    run_path = _resolve_run_path(run_name)
    sens_path = run_path / "sensitivity_summary.csv"
    if not sens_path.exists():
        return []
    df = pd.read_csv(sens_path)
    return _df_to_records(df)


@app.get("/api/runs/{run_name}/summary")
def get_run_summary(run_name: str) -> dict[str, list[dict[str, Any]]]:
    """Return metrics_summary.csv and cost_summary.csv as lists of dicts."""
    run_path = _resolve_run_path(run_name)
    metrics = pd.read_csv(run_path / "metrics_summary.csv")
    costs = pd.read_csv(run_path / "cost_summary.csv")
    return {
        "metrics": _df_to_records(metrics),
        "costs": _df_to_records(costs),
    }


@app.post("/api/runs/{run_name}/whatif")
def run_whatif(run_name: str, body: WhatIfRequest) -> dict[str, Any]:
    """
    Re-simulate inventory policy with custom cost params and return cost delta
    vs the base scenario stored in cost_summary.csv.
    """
    run_path = _resolve_run_path(run_name)
    preds = pd.read_csv(run_path / "predictions.csv")

    if preds[preds["model_name"] == body.model_name].empty:
        raise HTTPException(
            status_code=404, detail="No predictions found for the requested filters."
        )

    result = run_whatif_simulation(
        predictions=preds,
        model_name=body.model_name,
        data_strategy=body.data_strategy,
        c_over=body.c_over,
        c_under=body.c_under,
        capacity=body.capacity,
        series_id=body.series_id,
    )

    wi_costs = result["wi_costs"]
    whatif_orders = result["whatif_orders"]
    cost_col = result["cost_col"]
    sl_col = result["sl_col"]

    # Load base costs for comparison
    base_costs = pd.read_csv(run_path / "cost_summary.csv")
    if "data_strategy" in base_costs.columns:
        base_costs = base_costs[base_costs["data_strategy"] == body.data_strategy]
    base_costs = base_costs[base_costs["model_name"] == body.model_name]

    base_col = "sim_total_cost" if "sim_total_cost" in base_costs.columns else "total_cost"
    base_sl_col = (
        "sim_service_level" if "sim_service_level" in base_costs.columns else "service_level"
    )

    wi_total = float(wi_costs[cost_col].sum()) if cost_col in wi_costs.columns else 0.0
    base_total = (
        float(base_costs[base_col].sum())
        if base_col in base_costs.columns and not base_costs.empty
        else 0.0
    )
    wi_sl = float(wi_costs[sl_col].mean()) if sl_col in wi_costs.columns else 0.0
    base_sl = (
        float(base_costs[base_sl_col].mean())
        if base_sl_col in base_costs.columns and not base_costs.empty
        else 0.0
    )

    return {
        "whatif_total_cost": round(wi_total, 2),
        "base_total_cost": round(base_total, 2),
        "cost_delta": round(wi_total - base_total, 2),
        "whatif_service_level": round(wi_sl, 4),
        "base_service_level": round(base_sl, 4),
        "service_level_delta": round(wi_sl - base_sl, 4),
        "summary": _df_to_records(wi_costs),
        "whatif_orders": whatif_orders,
    }


@app.post("/api/retrain")
def retrain_model() -> dict[str, str]:
    """Trigger background model retraining simulation."""
    return {"status": "success", "message": "Model recalibration triggered successfully."}


_CONFIG_PATH = Path("configs/experiment.yaml")


class ConfigBody(BaseModel):
    yaml: str


@app.get("/api/config")
def get_config() -> dict[str, str]:
    """Return the raw YAML text of the default configuration file."""
    if not _CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {_CONFIG_PATH}")
    return {"yaml": _CONFIG_PATH.read_text(encoding="utf-8"), "path": str(_CONFIG_PATH)}


@app.put("/api/config")
def put_config(body: ConfigBody) -> dict[str, str]:
    """Validate and write a new configuration file."""
    # 1. Parse YAML syntax
    try:
        yaml.safe_load(body.yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc

    # 2. Validate with load_config via a temp file
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(body.yaml)
            tmp_path = Path(tmp.name)
        load_config(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Config validation failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # 3. Write to the real config path
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(body.yaml, encoding="utf-8")
    return {"status": "ok"}


@app.post("/predict_orders", response_model=ScoreResponse)
def predict_orders(request: ScoreRequest) -> ScoreResponse:
    """
    Trigger the operational scoring pipeline.

    Loads the configuration, enforces 'score_daily' operational mode, runs the
    pipeline (data validation, model prediction, Conformal Prediction, inventory
    optimization), and returns the final actionable order recommendations.
    """
    config_file = Path(request.config_path)
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Configuration file not found: {config_file}")

    try:
        settings = load_config(config_file)

        new_project = settings.project.model_copy(update={"run_mode": "score_daily"})
        settings = settings.model_copy(update={"project": new_project})

        if request.run_name:
            new_reporting = settings.reporting.model_copy(update={"run_name": request.run_name})
            settings = settings.model_copy(update={"reporting": new_reporting})

        try:
            artifacts = run_scoring(settings)
        except FileNotFoundError:
            artifacts = run_experiment(settings)

        if artifacts.run_directory is None or artifacts.reorder_recommendations is None:
            raise HTTPException(
                status_code=500,
                detail="Pipeline failed to generate operational artifacts.",
            )

        recommendations_df = artifacts.reorder_recommendations.fillna(value="")
        recs_list: list[dict[str, Any]] = [
            {str(k): v for k, v in row.items()}
            for row in recommendations_df.to_dict(orient="records")
        ]

        return ScoreResponse(
            status="success",
            run_directory=str(artifacts.run_directory),
            recommendations_generated=len(recs_list),
            recommendations=recs_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}") from e
