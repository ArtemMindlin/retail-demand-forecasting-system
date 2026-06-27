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
from typing import Any, cast

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
# Secure cookies require HTTPS. Safari drops Secure cookies over http://localhost
# (Chrome/Firefox allow them), so for local dev set COOKIE_SECURE=false. Prod keeps
# the default (True) — the cookie is only sent over HTTPS.
_COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").strip().lower() not in {
    "false",
    "0",
    "no",
}

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
        secure=_COOKIE_SECURE,
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


def get_category(sku_df: pd.DataFrame) -> str:
    """Return the SKU's real category if it is present in the predictions frame.

    The dataset categories (``third_category_id``) are used internally for the
    Mondrian conformal grouping but are not currently persisted into the
    predictions artifact the API reads. Until they are, we return "N/D" instead
    of fabricating a category, so the dashboard never shows invented data.
    """
    if "third_category_id" in sku_df.columns and not sku_df.empty:
        return str(sku_df["third_category_id"].iloc[-1])
    return "N/D"


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

    usecols = ["date", "series_id", "y_true", "y_pred", "data_strategy", "model_name"]
    try:
        df = pd.read_csv(latest_run / "predictions.csv", usecols=usecols)
    except Exception:
        df = pd.read_csv(latest_run / "predictions.csv")

    try:
        settings = load_config("configs/experiment.yaml")
        champion_strategy = settings.business.champion_data_strategy
        champion_model = settings.business.champion_model_name
    except Exception:
        champion_strategy = "Observed"
        champion_model = "catboost"

    if "data_strategy" in df.columns:
        df = df[df["data_strategy"] == champion_strategy]
    if "model_name" in df.columns:
        df = df[df["model_name"] == champion_model]

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


def _download_latest_csv(
    candidates: list[str],
    download_name: str,
    missing_detail: str,
    error_label: str,
) -> FileResponse:
    """Serve the first existing CSV among ``candidates`` from the latest run dir."""
    try:
        latest_run = _get_latest_run_path()
        file_path = next(
            (latest_run / name for name in candidates if (latest_run / name).exists()), None
        )
        if file_path is None:
            raise HTTPException(status_code=404, detail=missing_detail)
        return FileResponse(path=file_path, filename=download_name, media_type="text/csv")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{error_label}: {e}") from e


@app.get("/api/download/predictions")
def download_predictions() -> FileResponse:
    """Download the predictions.csv file from the latest run."""
    return _download_latest_csv(
        ["predictions.csv"],
        "predictions.csv",
        "predictions.csv not found in the latest run.",
        "Failed to download predictions",
    )


@app.get("/api/download/costs")
def download_costs() -> FileResponse:
    """Download the cost summary CSV from the latest run."""
    return _download_latest_csv(
        ["cost_summary.csv", "costs.csv"],
        "costs.csv",
        "Cost summary CSV file not found in the latest run.",
        "Failed to download costs",
    )


_EDA_FIGURES: list[dict[str, Any]] = [
    {
        "name": "observed_demand_distribution",
        "nav_label": "Distribución de demanda",
        "caption": "Distribución global de la demanda observada",
        "interpretation": "La distribución muestra concentración en rangos bajos y una cola hacia valores mayores, lo que es coherente con un problema retail heterogéneo y alejado de una distribución gaussiana simple.",
    },
    {
        "name": "weekday_demand_profile",
        "nav_label": "Perfil semanal",
        "caption": "Perfil semanal de demanda (media y mediana)",
        "interpretation": "El patrón semanal visible justifica el uso de variables de calendario y retardos de 7 días en la etapa de ingeniería de características.",
    },
    {
        "name": "observed_demand_boxplot_top_series",
        "nav_label": "Dispersión top series",
        "caption": "Dispersión de la demanda en las series de mayor volumen",
        "interpretation": "Incluso entre las series de mayor volumen persisten diferencias relevantes en nivel medio y variabilidad, lo que refuerza la conveniencia de incorporar contexto de serie en el modelado.",
    },
    {
        "name": "zero_demand_rate_by_series",
        "nav_label": "Intermitencia por serie",
        "caption": "Series más intermitentes (proporción de demanda cero)",
        "interpretation": "La intermitencia no es homogénea entre series, por lo que el problema no debe interpretarse como uniforme para todas las combinaciones tienda-producto.",
    },
    {
        "name": "stockout_hours_distribution",
        "nav_label": "Distribución stockout",
        "caption": "Distribución de horas de stockout en el panel",
        "interpretation": "La frecuencia de stockouts confirma que la falta de disponibilidad forma parte del régimen operativo del dataset y no constituye un fenómeno aislado.",
    },
    {
        "name": "stockout_band_demand",
        "nav_label": "Demanda por banda stockout",
        "caption": "Demanda media y observaciones por banda de stockout",
        "interpretation": "La caída de la demanda observada bajo stockouts severos es consistente con la hipótesis de censura operativa por falta de disponibilidad.",
    },
    {
        "name": "stockout_vs_demand_scatter",
        "nav_label": "Stockout vs demanda",
        "caption": "Relación entre horas de stockout y demanda observada",
        "interpretation": "La tendencia agregada negativa sugiere que las horas de stockout aportan señal contextual relevante, aunque con elevada dispersión entre observaciones.",
    },
    {
        "name": "correlation_heatmap",
        "nav_label": "Correlaciones",
        "caption": "Correlaciones entre features numéricas y demanda",
        "interpretation": "Las asociaciones marginales son en general moderadas, lo que respalda el uso de modelos flexibles capaces de capturar interacciones y no linealidades.",
    },
    {
        "name": "representative_series_panels",
        "nav_label": "Series representativas",
        "caption": "Pequeños múltiplos de demanda con overlay de stockout",
        "interpretation": "La visualización conjunta de demanda y stockout resume la complejidad del problema: estacionalidad, heterogeneidad entre series y posible compresión de ventas observadas.",
        "wide": True,
    },
]


def _get_latest_eda_path() -> Path | None:
    """Find the most-recent EDA report directory in reports/."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None
    eda_dirs = [d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("eda_")]
    if not eda_dirs:
        return None
    return max(eda_dirs, key=lambda d: d.name)


@app.get("/api/eda/runs")
def get_eda_runs() -> list[str]:
    """Return a list of available EDA runs, sorted newest first."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return []
    eda_dirs = [d.name for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("eda_")]
    return sorted(eda_dirs, reverse=True)


@app.get("/api/eda")
def get_eda_meta(run: str | None = None) -> dict[str, Any]:
    """Return EDA summary stats and list of available figure names from an EDA run."""
    eda_path = Path("reports") / run if run else _get_latest_eda_path()
    if eda_path is None:
        raise HTTPException(status_code=404, detail="No EDA report found in reports/.")

    summary: dict[str, Any] = {}
    summary_csv = eda_path / "dataset_summary.csv"
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            if not df.empty:
                summary = json.loads(df.iloc[[0]].to_json(orient="records"))[0]
        except Exception:
            pass

    available_figures = [fig for fig in _EDA_FIGURES if (eda_path / f"{fig['name']}.png").exists()]

    return {"run": eda_path.name, "summary": summary, "figures": available_figures}


@app.get("/api/eda/figure/{name}")
def get_eda_figure(name: str, run: str | None = None) -> FileResponse:
    """Serve a specific EDA figure PNG from an EDA run."""
    safe_name = Path(name).name
    if not safe_name or safe_name != name or "/" in name or ".." in name:
        raise HTTPException(status_code=404, detail="Figure not found.")

    known_names = {fig["name"] for fig in _EDA_FIGURES}
    if safe_name not in known_names:
        raise HTTPException(status_code=404, detail="Figure not found.")

    eda_path = Path("reports") / run if run else _get_latest_eda_path()
    if eda_path is None or not eda_path.exists():
        raise HTTPException(status_code=404, detail="No EDA report found.")

    figure_path = eda_path / f"{safe_name}.png"
    if not figure_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure '{safe_name}' not found.")

    return FileResponse(path=figure_path, media_type="image/png")


def _histogram(values: list[float], n_bins: int = 25) -> tuple[list[float], list[int]]:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return [], []
    counts, edges = np.histogram(arr, bins=n_bins)
    centers = [float((edges[i] + edges[i + 1]) / 2) for i in range(len(counts))]
    return centers, counts.tolist()


@app.get("/api/eda/data/{name}")
def get_eda_chart_data(name: str, run: str | None = None) -> dict[str, Any]:
    """Return chart-ready JSON data for a named EDA figure (replaces static PNG)."""
    known_names = {fig["name"] for fig in _EDA_FIGURES}
    if name not in known_names:
        raise HTTPException(status_code=404, detail="Figure not found.")

    eda_path = Path("reports") / run if run else _get_latest_eda_path()
    if eda_path is None or not eda_path.exists():
        raise HTTPException(status_code=404, detail="No EDA report found.")

    def read(fname: str) -> pd.DataFrame | None:
        p = eda_path / fname
        return pd.read_csv(p) if p.exists() else None

    if name == "weekday_demand_profile":
        df = read("weekday_summary.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        return {
            "type": "line_dual",
            "data": df[["weekday_name", "observed_demand_mean", "observed_demand_median"]].to_dict(
                "records"
            ),
            "series": [
                {"key": "observed_demand_mean", "label": "Media", "color": "#10b981"},
                {"key": "observed_demand_median", "label": "Mediana", "color": "#3b82f6"},
            ],
        }

    if name == "stockout_band_demand":
        df = read("stockout_demand_bands.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        cols = [
            c for c in ["stockout_band", "observed_demand_mean", "observations"] if c in df.columns
        ]
        return {"type": "bar_group", "data": df[cols].to_dict("records"), "x_key": "stockout_band"}

    if name == "correlation_heatmap":
        df = read("correlation_summary.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        df = df.sort_values("absolute_correlation", ascending=False).head(15)
        return {
            "type": "bar_horizontal",
            "data": df[["feature_name", "correlation_with_observed_demand"]].to_dict("records"),
        }

    if name == "zero_demand_rate_by_series":
        df = read("series_summary.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        rates = df["zero_demand_rate"].dropna().tolist()
        centers, counts = _histogram(rates, 25)
        median_r = float(np.median(rates)) if rates else 0.0
        pct50 = float(sum(1 for r in rates if r > 0.5) / len(rates) * 100) if rates else 0.0
        return {
            "type": "histogram",
            "centers": centers,
            "counts": counts,
            "x_label": "Tasa demanda cero",
            "median": median_r,
            "pct_above_50": pct50,
        }

    if name == "observed_demand_distribution":
        df = read("series_summary.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        means = df["observed_demand_mean"].dropna().tolist()
        centers, counts = _histogram(means, 30)
        log_counts = [float(np.log10(c)) if c > 0 else 0.0 for c in counts]
        return {
            "type": "histogram_dual",
            "centers": centers,
            "counts": counts,
            "log_counts": log_counts,
            "x_label": "Demanda media por serie",
        }

    if name == "stockout_hours_distribution":
        df = read("stockout_by_series_summary.csv")
        if df is None:
            df = read("series_summary.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        col = "mean_stockout_hours" if "mean_stockout_hours" in df.columns else "stockout_day_rate"
        vals = df[col].dropna().tolist()
        centers, counts = _histogram(vals, 25)
        return {
            "type": "histogram",
            "centers": centers,
            "counts": counts,
            "x_label": col.replace("_", " "),
        }

    if name == "stockout_vs_demand_scatter":
        df = read("series_summary.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        sub = df[["observed_demand_mean", "mean_stockout_hours"]].dropna().head(500)
        return {
            "type": "scatter",
            "data": sub.rename(
                columns={"observed_demand_mean": "x", "mean_stockout_hours": "y"}
            ).to_dict("records"),
        }

    if name == "observed_demand_boxplot_top_series":
        df = read("series_summary.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        top = df.nlargest(20, "observed_demand_sum")[
            ["series_id", "observed_demand_mean", "observed_demand_std"]
        ].dropna()
        boxes = []
        for _, row in top.iterrows():
            m, s = float(row["observed_demand_mean"]), float(row["observed_demand_std"])
            boxes.append(
                {
                    "id": str(row["series_id"]),
                    "min": max(0.0, m - 2 * s),
                    "q1": max(0.0, m - s),
                    "median": m,
                    "q3": m + s,
                    "max": m + 2 * s,
                }
            )
        return {"type": "boxplot", "data": boxes}

    if name == "representative_series_panels":
        df = read("series_summary.csv")
        if df is None:
            raise HTTPException(status_code=404, detail="Data not available.")
        cols = [
            c
            for c in [
                "series_id",
                "observed_demand_mean",
                "observed_demand_std",
                "zero_demand_rate",
                "stockout_day_rate",
                "history_days",
            ]
            if c in df.columns
        ]
        top = df.nlargest(12, "observed_demand_sum")[cols]
        return {"type": "series_grid", "series": top.to_dict("records")}

    raise HTTPException(status_code=404, detail="Figure data not available.")


# ──────────────────────────────────────────────────────────────────────────
# Operational simulation (OPS plane) — walk-forward week-by-week playback.
# Reads the artifact produced by run_operational_simulation
# (reports/<run>/simulation/predictions_by_day.parquet) and serves it indexed
# by weekly origin. Pure artifact reads — no live compute, no pipeline lock.
# ──────────────────────────────────────────────────────────────────────────

# Conformal band [q_0_1, q_0_9] is a nominal 80% interval (models.quantiles).
_OPS_TARGET_COVERAGE = 0.80
_OPS_LOWER_COL = "q_0_1"
_OPS_UPPER_COL = "q_0_9"
_OPS_CACHE: dict[str, Any] = {}


def _get_ops_sim_df() -> pd.DataFrame | None:
    """Load and cache the operational-simulation prediction artifact."""
    cached = _OPS_CACHE.get("df")
    if cached is not None:
        return cached

    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None
    candidates = sorted(reports_dir.glob("*/simulation/predictions_by_day.parquet"))
    if not candidates:
        return None
    artifact = max(candidates, key=lambda p: p.stat().st_mtime)

    df = pd.read_parquet(artifact)
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    # Weekly playback: keep origins on a 7-day grid (the retrain cadence boundary).
    df["week_index"] = (df["day_index"] // 7).astype(int)
    df["is_weekly_origin"] = df["day_index"] % 7 == 0
    df["covered"] = (df["y_true"] >= df[_OPS_LOWER_COL]) & (df["y_true"] <= df[_OPS_UPPER_COL])
    _OPS_CACHE["df"] = df
    _OPS_CACHE["run"] = artifact.parent.parent.name
    return df


def _ops_cadence_block(group: pd.DataFrame) -> dict[str, Any]:
    """Per-cadence aggregate for one weekly origin (fully-revealed rows only)."""
    complete = group[group["actuals_complete"]]
    base = complete if not complete.empty else group
    coverage = float(base["covered"].mean()) if not base.empty else None
    mae = float((base["y_pred"] - base["y_true"]).abs().mean()) if not base.empty else None
    return {
        "coverage": coverage,
        "total_cost": float(group["total_cost"].sum()),
        "mae": mae,
        "n_series": int(group["series_id"].nunique()),
        "retrained": bool(group["retrained_this_step"].any()),
        "actuals_complete": bool(group["actuals_complete"].all()),
    }


@app.get("/api/ops/weeks")
def get_ops_weeks() -> dict[str, Any]:
    """Per-week summary across cadences plus the series catalogue (for the slider)."""
    df = _get_ops_sim_df()
    if df is None:
        raise HTTPException(status_code=404, detail="No operational simulation artifact found.")

    weekly = df[df["is_weekly_origin"]]
    cadences = sorted(df["cadence"].unique().tolist())

    weeks: list[dict[str, Any]] = []
    for week_index, wk in weekly.groupby("week_index"):
        by_cadence = {cad: _ops_cadence_block(grp) for cad, grp in wk.groupby("cadence")}
        weeks.append(
            {
                "week_index": int(week_index),
                "origin_date": wk["decision_date"].iloc[0].date().isoformat(),
                "by_cadence": by_cadence,
            }
        )

    # Series catalogue ordered by total realized demand (liveliest first).
    volume = weekly.groupby("series_id")["y_true"].sum().sort_values(ascending=False)
    series = [str(s) for s in volume.head(60).index.tolist()]

    return {
        "run": _OPS_CACHE.get("run", "ops_sim"),
        "horizon": 7,
        "target_coverage": _OPS_TARGET_COVERAGE,
        "cadences": cadences,
        "series": series,
        "weeks": weeks,
    }


@app.get("/api/ops/series/{series_id}")
def get_ops_series(series_id: str, cadence: str = "every_7d") -> dict[str, Any]:
    """Weekly forecast/band/actual trajectory for one series (closes the loop)."""
    df = _get_ops_sim_df()
    if df is None:
        raise HTTPException(status_code=404, detail="No operational simulation artifact found.")

    sel = df[
        (df["series_id"].astype(str) == series_id)
        & (df["cadence"] == cadence)
        & (df["is_weekly_origin"])
    ].sort_values("week_index")
    if sel.empty:
        raise HTTPException(status_code=404, detail="Series not found in simulation.")

    points = [
        {
            "week_index": int(r.week_index),
            "origin_date": r.decision_date.date().isoformat(),
            "y_pred": float(r.y_pred),
            "lower": float(getattr(r, _OPS_LOWER_COL)),
            "upper": float(getattr(r, _OPS_UPPER_COL)),
            "y_true": None if pd.isna(r.y_true) else float(r.y_true),
            "order_quantity": float(r.order_quantity),
            "total_cost": float(r.total_cost),
            "covered": bool(r.covered),
            "actuals_complete": bool(r.actuals_complete),
        }
        for r in sel.itertuples(index=False)
    ]
    return {"series_id": series_id, "cadence": cadence, "points": points}


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Liveness probe for Railway and Uptime Kuma monitoring."""
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


def _ensure_grouped_cache(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the {series_id: frame} grouping for ``df``, repopulating the cache
    only when the underlying predictions frame has changed."""
    if "df" not in _PREDICTIONS_CACHE or _PREDICTIONS_CACHE["df"] is not df:
        _PREDICTIONS_CACHE["df"] = df
        _PREDICTIONS_CACHE["grouped"] = {sku: group for sku, group in df.groupby("series_id")}
    return cast("dict[str, pd.DataFrame]", _PREDICTIONS_CACHE.get("grouped", {}))


def _empirical_conformal(
    sku_df: pd.DataFrame, alpha: float
) -> tuple[pd.Series, pd.Series, float, float]:
    """Empirical conformal stats for one SKU.

    Returns ``(residuals, abs_residuals, conformal_width, empirical_coverage_pct)``
    where the interval half-width is the ``(1 - alpha)`` quantile of |residuals|.
    """
    residuals = sku_df["y_true"] - sku_df["y_pred"]
    abs_residuals = residuals.abs()
    conformal_width = float(np.quantile(abs_residuals, 1.0 - alpha))
    emp_coverage = float((abs_residuals <= conformal_width).mean()) * 100.0
    return residuals, abs_residuals, conformal_width, emp_coverage


def _per_sku_psi(demand: np.ndarray) -> float:
    """PSI of a SKU's own demand, older half vs recent half (0.0 if too short)."""
    half = len(demand) // 2
    if half < 1:
        return 0.0
    return round(compute_psi(demand[:half], demand[half:])[0], 3)


def _build_forecast_chart(sku_df: pd.DataFrame, conformal_width: float) -> list[dict[str, Any]]:
    """Per-day actual/predicted values with a symmetric conformal band."""
    forecast = []
    for idx, row in enumerate(sku_df.itertuples()):
        pred = float(row.y_pred)
        forecast.append(
            {
                "day": idx + 1,
                "label": f"D{idx + 1:02d}",
                "actual": round(float(row.y_true)),
                "predicted": round(pred),
                "lower": round(max(0.0, pred - conformal_width)),
                "upper": round(pred + conformal_width),
            }
        )
    return forecast


def _aggregate_inventory_cost(
    sku_df: pd.DataFrame,
    cr_quantile: float,
    shortage_cost: float,
    holding_cost: float,
    capacity: float,
) -> tuple[float, float]:
    """Aggregate Newsvendor vs naïve point-forecast inventory cost over the SKU's history.

    Returns ``(newsvendor_cost, delta_vs_naive)``; a negative delta means the
    Newsvendor policy is cheaper than naïve point-forecast ordering.
    """
    tot_cost = 0.0
    naive_cost = 0.0
    for row in sku_df.itertuples():
        q_day = min(max(0.0, float(row.y_pred) + cr_quantile), capacity)
        q_naive = min(float(row.y_pred), capacity)
        d_day = float(row.y_true)
        tot_cost += (
            (d_day - q_day) * shortage_cost if d_day > q_day else (q_day - d_day) * holding_cost
        )
        naive_cost += (
            (d_day - q_naive) * shortage_cost
            if d_day > q_naive
            else (q_naive - d_day) * holding_cost
        )
    inv_cost = round(tot_cost, 2)
    return inv_cost, round(inv_cost - naive_cost, 2)


@app.post("/api/forecast")
def post_forecast(request: ForecastRequest) -> dict[str, Any]:
    """Calculate live empirical conformal quantiles and Newsvendor quantities."""
    try:
        df = load_latest_predictions()
    except FileNotFoundError:
        return {"status": "no_predictions"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}") from e

    grouped = _ensure_grouped_cache(df)

    sku = request.selectedSkuId
    if not sku or sku not in grouped:
        sku = next(iter(grouped.keys()))

    sku_df = grouped[sku].copy().sort_values("date")

    sl = request.serviceLevel
    alpha = 1.0 - (sl / 100.0)
    residuals, abs_residuals, conformal_width, emp_coverage = _empirical_conformal(sku_df, alpha)

    forecast = _build_forecast_chart(sku_df, conformal_width)

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

    cov_delta = emp_coverage - sl

    # MAE of the selected SKU; delta = second-half MAE − first-half MAE (negative = improving)
    mae = float(abs_residuals.mean())
    half_idx = len(abs_residuals) // 2
    if half_idx >= 1:
        mae_delta = float(
            abs_residuals.iloc[half_idx:].mean() - abs_residuals.iloc[:half_idx].mean()
        )
    else:
        mae_delta = 0.0

    inv_cost, inv_delta = _aggregate_inventory_cost(sku_df, cr_quantile, cs, ch, request.capacity)

    # Real per-SKU drift: PSI of the SKU's own demand, older half vs recent.
    psi = _per_sku_psi(sku_df["y_true"].to_numpy(dtype=float))
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

    grouped = _ensure_grouped_cache(df)
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

        residuals, _, _, emp_coverage = _empirical_conformal(sku_df, alpha)
        cr_quantile = float(np.quantile(residuals, cr))

        last_row = sku_df.iloc[-1]
        last_actual = float(last_row["y_true"])
        last_pred = float(last_row["y_pred"])

        q_star = max(0.0, last_pred + cr_quantile)
        q_star = min(q_star, capacity)

        # Real per-SKU drift: PSI of the SKU's own demand, older half vs recent.
        drift_psi = _per_sku_psi(sku_df["y_true"].to_numpy(dtype=float))

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
                "cat": get_category(sku_df),
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
    """Return operational alerts derived from the latest run's exceptions.csv.

    Returns an empty list when no run or no exceptions are available; the
    dashboard renders an empty state in that case. No synthetic alerts are
    fabricated.
    """
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
                    }
                )
        except Exception:
            pass

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


def _runs_with(*required_files: str) -> list[Path]:
    """Run dirs in reports/ that contain all the given files, newest first."""
    if not _REPORTS_DIR.exists():
        return []
    return [
        d
        for d in sorted(_REPORTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if d.is_dir() and all((d / f).exists() for f in required_files)
    ]


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
    return [
        d.name for d in _runs_with("predictions.csv", "metrics_summary.csv", "cost_summary.csv")
    ]


@app.get("/api/imputation-runs")
def list_imputation_runs() -> list[str]:
    """Return names of imputation-comparison run dirs (those with latent_strategies.csv)."""
    return [d.name for d in _runs_with("latent_strategies.csv")]


def _resolve_imputation_run_path(run_name: str) -> Path:
    """Validate run_name and return its path, requiring latent_strategies.csv."""
    safe_name = Path(run_name).name
    if not safe_name or safe_name != run_name or run_name.startswith("."):
        raise HTTPException(status_code=404, detail="Run not found.")
    run_path = _REPORTS_DIR / run_name
    if not run_path.is_dir() or not (run_path / "latent_strategies.csv").exists():
        raise HTTPException(status_code=404, detail="Imputation comparison run not found.")
    return run_path


@app.get("/api/imputation-runs/{run_name}/strategies")
def get_imputation_strategies(run_name: str, series_id: str | None = None) -> dict[str, Any]:
    """Return the per-strategy latent demand reconstruction for one series.

    Response: dates, observed (censored sale), stockout_hours, and a `strategies` map of
    {strategy_name: [latent values aligned to dates]}, plus the list of available series.
    """
    run_path = _resolve_imputation_run_path(run_name)
    df = pd.read_csv(run_path / "latent_strategies.csv")
    if df.empty:
        return {"dates": [], "observed": [], "stockout_hours": [], "strategies": {}, "series": []}

    series = sorted(df["series_id"].astype(str).unique().tolist())
    chosen = series_id if series_id in series else series[0]
    sub = df[df["series_id"].astype(str) == chosen].sort_values("date")

    dates = sorted(sub["date"].unique().tolist())
    # observed + stockout are strategy-invariant; take them from the first strategy slice.
    base = sub.drop_duplicates(subset=["date"]).set_index("date")
    strategies: dict[str, list[float | None]] = {}
    for name in sorted(sub["strategy"].unique().tolist()):
        s = sub[sub["strategy"] == name].set_index("date")["latent_demand_est"]
        strategies[name] = [float(s[d]) if d in s.index and pd.notna(s[d]) else None for d in dates]

    quality: list[dict[str, Any]] = []
    quality_path = run_path / "imputation_quality.csv"
    if quality_path.exists():
        qdf = pd.read_csv(quality_path)
        quality = [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in qdf.to_dict(orient="records")
        ]

    return {
        "series": series,
        "series_id": chosen,
        "dates": dates,
        "observed": [
            float(base.loc[d, "observed"]) if pd.notna(base.loc[d, "observed"]) else None
            for d in dates
        ],
        "stockout_hours": [
            float(base.loc[d, "stockout_hours"])
            if pd.notna(base.loc[d, "stockout_hours"])
            else None
            for d in dates
        ],
        "strategies": strategies,
        "quality": quality,
    }


@app.get("/api/fair-cost")
def get_fair_cost() -> dict[str, Any]:
    """Return the latest fair inventory-cost backtest (strategies vs a common ground truth).

    Reads the most recent ``fair_cost_backtest.csv`` written by the
    ``fair_cost_backtest`` run-mode. This is the apples-to-apples cost comparison
    (every strategy charged against the same synthetically-censored true demand),
    in contrast to the headline per-strategy cost which is biased by censoring.
    Empty payload (rows=[]) if no backtest has been run yet.
    """
    candidates = _runs_with("fair_cost_backtest.csv")
    if not candidates:
        return {"rows": [], "run": None}

    latest = candidates[0]
    df = pd.read_csv(latest / "fair_cost_backtest.csv")
    return {"rows": _df_to_records(df), "run": latest.name}


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
    has_pareto = (run_path / "tuning_pareto.csv").exists()
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
    """Return tuning_pareto.csv (Pinball vs Winkler front) as a list of dicts ([] if not found)."""
    run_path = _resolve_run_path(run_name)
    pareto_path = run_path / "tuning_pareto.csv"
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
    """Stub endpoint: real retraining runs offline via ``run.py --mode retrain``.

    This endpoint does not trigger any training; it exists only so the UI can
    surface the action. It reports its no-op status honestly rather than
    claiming a success that did not happen.
    """
    return {
        "status": "not_implemented",
        "message": (
            "El reentrenamiento se ejecuta fuera de línea con "
            "'run.py --mode retrain'. Este endpoint no lanza entrenamiento."
        ),
    }


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
