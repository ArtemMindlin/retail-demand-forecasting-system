"""MLflow tracking for experiment runs, alongside the artifacts written under ``reports/``.

What goes here and what stays a file is a size question, not a taste one. Metrics, parameters
and decisions are aggregates -- a few numbers each -- and they are what `mlflow.search_runs`
makes queryable across runs, which is the thing 92 timestamped folders cannot do. The row-level
outputs stay in `reports/`: `predictions.csv` alone is 1.9 GB across runs, MLflow's artifact
store is a plain directory, so moving them would put the same bytes behind a UUID, and
`log_table` only writes JSON (2 MB of it already slows the UI down).

Failure here never fails a run. Everything a run produces is on disk by the time this is
called, and half an hour of forecasting is not worth losing to a locked tracking store.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import pandas as pd

from retail_forecasting.config import Settings, build_config_hash
from retail_forecasting.utils.provenance import get_git_commit

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from retail_forecasting.evaluation.reporting import RunArtifacts

# Same store the imputation search writes to, so every run of every mode is comparable in one
# UI. Module-level and not a literal at the call site: the tests point it at a scratch path.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_RUNS = "retail_forecasting_runs"

# Columns of `metrics_summary` and `cost_summary` that name the row rather than measure it.
# They become part of each metric's name instead of a metric of their own.
_IDENTITY_COLUMNS = ("model_name", "backend_name", "data_strategy", "observations")


def _artifact_root(tracking_uri: str) -> str | None:
    """Where artifacts belong for a given tracking store, as an absolute path.

    Pinned explicitly rather than left to MLflow's default, which resolves `mlruns/` against
    the WORKING DIRECTORY, independently of where the tracking store lives. That independence
    is a trap: pointing a test at a scratch database still wrote an `mlruns/` into the repo,
    and only moving the process's cwd caught both halves. Deriving one from the other means a
    redirected store takes its artifacts with it.
    """
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    return str(Path(tracking_uri.removeprefix(prefix)).resolve().parent / "mlruns")


def _metric_name(prefix: str, row: pd.Series) -> str:
    """`mae` plus the row's identity, since one run scores several models and strategies."""
    parts = [str(row[column]) for column in ("model_name", "data_strategy") if column in row]
    return ".".join([prefix, *(part for part in parts if part and part != "nan")])


def _numeric_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Every numeric cell of a summary table, keyed by column and row identity."""
    metrics: dict[str, float] = {}
    for _, row in frame.iterrows():
        for column, value in row.items():
            if column in _IDENTITY_COLUMNS or not isinstance(value, int | float):
                continue
            if pd.isna(value):
                continue
            metrics[_metric_name(str(column), row)] = float(value)
    return metrics


def _flat_params(settings: Settings) -> dict[str, Any]:
    """The config as `section.field` pairs, skipping what does not identify a run.

    Paths and the split map are left out on purpose: they say where the run read and wrote, not
    what it did, and `search_runs` filters on params -- so a path that differs between two
    machines would split runs that are otherwise the same experiment.
    """
    skip = {"splits", "local_cache_dir", "processed_panel_dir", "models_dir", "output_dir"}
    params: dict[str, Any] = {}
    for section, config in settings.model_dump().items():
        if not isinstance(config, dict):
            continue
        for field, value in config.items():
            if field in skip or isinstance(value, dict):
                continue
            params[f"{section}.{field}"] = value
    return params


def log_run_to_mlflow(artifacts: RunArtifacts, settings: Settings, run_dir: Path) -> None:
    """Record one finished experiment run: its config, its metrics and its decisions."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    root = _artifact_root(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_RUNS) is None:
        mlflow.create_experiment(EXPERIMENT_RUNS, artifact_location=root)
    mlflow.set_experiment(EXPERIMENT_RUNS)

    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params(_flat_params(settings))
        mlflow.set_tags(
            {
                "git_commit": get_git_commit(),
                "run_mode": settings.project.run_mode,
                "config_hash": build_config_hash(settings),
                # The bridge back to the bytes this run left behind, which is where the
                # row-level predictions and the figures live.
                "reports_run_dir": str(run_dir),
                "series": artifacts.prepared_panel["series_id"].nunique(),
                "panel_rows": len(artifacts.prepared_panel),
                "supervised_rows": len(artifacts.supervised_frame),
                "drifts_detected": len(artifacts.drifts),
            }
        )

        mlflow.log_metrics(_numeric_metrics(artifacts.metrics_summary))
        mlflow.log_metrics(_numeric_metrics(artifacts.cost_summary))

        # The summaries whole, so a reader gets the identity columns the metric names flatten.
        # `load_table` reassembles these across runs into one frame.
        mlflow.log_table(artifacts.metrics_summary, artifact_file="metrics_summary.json")
        mlflow.log_table(artifacts.cost_summary, artifact_file="cost_summary.json")

        if artifacts.promotion_decision is not None:
            mlflow.log_dict(
                artifacts.promotion_decision.model_dump(mode="json"), "promotion_decision.json"
            )
        if artifacts.data_quality_report is not None:
            mlflow.log_dict(
                artifacts.data_quality_report.model_dump(mode="json"), "data_quality_report.json"
            )
        if artifacts.backtest_metadata is not None:
            mlflow.log_dict(
                artifacts.backtest_metadata.model_dump(mode="json"), "backtest_metadata.json"
            )

        for figure in sorted(run_dir.glob("*.png")):
            mlflow.log_artifact(str(figure), artifact_path="figures")
