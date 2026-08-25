"""The run store: where every mode records what it did, and writes what it produced.

Its own layer and not part of `evaluation`, where it started: recording a run is something
every mode does, and `eda` is forbidden from importing `evaluation` -- correctly, since the
EDA has nothing to do with summarising predictions and costs. A cross-cutting concern parked
inside one consumer is one the others cannot reach.

`open_run_directory` is the entry point that matters. It opens an MLflow run and hands back
the directory its artifacts live in, and for a local artifact store writing into that
directory IS logging -- MLflow lists and serves whatever it holds, nested paths included. So
the pipeline writes once, straight into the run, instead of writing a directory of its own and
uploading a copy afterwards. Its cost is that the store stops being optional: a store that
cannot be reached is no longer a lost record, it is a lost run.

The `log_*_metadata` functions add what a directory cannot answer -- which config produced
this, how it scored, what it decided -- because that is what `mlflow.search_runs` makes
queryable across runs, and a folder listing never could.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import mlflow
import pandas as pd
from pydantic import BaseModel

from retail_forecasting.config import Settings, build_config_hash
from retail_forecasting.utils.provenance import get_git_commit


class LoggableRun(Protocol):
    """What this module needs off a finished run, named here rather than imported.

    `evaluation.RunArtifacts` satisfies it, but importing that class would invert the layers:
    recording a run is a concern of every mode, so it cannot depend on the one that summarises
    predictions and costs. Structural typing lets this state its requirement instead of
    borrowing a class, and the list doubles as the record of what actually reaches MLflow.
    """

    prepared_panel: pd.DataFrame
    supervised_frame: pd.DataFrame
    metrics_summary: pd.DataFrame
    cost_summary: pd.DataFrame
    promotion_decision: Any
    data_quality_report: Any
    backtest_metadata: Any
    drifts: list[Any]


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_RUNS = "retail_forecasting_runs"

EXPERIMENT_EDA = "retail_forecasting_eda"
EXPERIMENT_OPS = "retail_forecasting_ops"
# Also its own, and for the same reason: reconstruction error against synthetically
# censored days is not forecast error, and sharing an experiment put 21 comparison runs
# in front of the 5 that back chapter 6.
EXPERIMENT_IMPUTATION = "retail_forecasting_imputation"
EXPERIMENT_SENSITIVITY = "retail_forecasting_sensitivity"

_IDENTITY_COLUMNS = ("model_name", "backend_name", "data_strategy", "observations")


def _artifact_root(tracking_uri: str) -> str | None:
    """Where artifacts belong for a given tracking store.

    Pinned explicitly rather than left to MLflow's default, which resolves `mlruns/` against
    the WORKING DIRECTORY, independently of where the tracking store lives. That independence
    is a trap: pointing a test at a scratch database still wrote an `mlruns/` into the repo,
    and only moving the process's cwd caught both halves. Deriving one from the other means a
    redirected store takes its artifacts with it.
    """
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    store = Path(tracking_uri.removeprefix(prefix))
    return str(store.parent / "mlruns")


_ROW_IDENTITY = ("model_name", "data_strategy", "cadence", "strategy")


def _metric_name(prefix: str, row: pd.Series) -> str:
    """`mae` plus the row's identity, since one run scores several models and strategies."""
    parts = [str(row[column]) for column in _ROW_IDENTITY if column in row]
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


def _open_experiment(name: str) -> None:
    """Point MLflow at the store and make sure `name` exists with the right artifact root."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=_artifact_root(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(name)


def log_run_metadata(artifacts: LoggableRun, settings: Settings) -> None:
    """Attach config, metrics and decisions to the run that is already open.

    The files are not this function's business: the pipeline wrote them straight into the
    run's artifact directory, so there is nothing left to upload. What is left is everything
    a directory cannot answer -- which config produced this, how it scored, what it decided.
    """
    mlflow.log_params(_flat_params(settings))
    mlflow.set_tags(
        {
            "git_commit": get_git_commit(),
            "run_mode": settings.project.run_mode,
            "config_hash": build_config_hash(settings),
            "series": artifacts.prepared_panel["series_id"].nunique(),
            "panel_rows": len(artifacts.prepared_panel),
            "supervised_rows": len(artifacts.supervised_frame),
            "drifts_detected": len(artifacts.drifts),
        }
    )

    mlflow.log_metrics(_numeric_metrics(artifacts.metrics_summary))
    mlflow.log_metrics(_numeric_metrics(artifacts.cost_summary))

    mlflow.log_table(artifacts.metrics_summary, artifact_file="metrics_summary.json")
    mlflow.log_table(artifacts.cost_summary, artifact_file="cost_summary.json")


def log_eda_metadata(metadata: BaseModel, dataset_summary: pd.DataFrame) -> None:
    """Attach the EDA run's identity and the panel's statistics to the run already open.

    The figures and tables are already in the run's artifact directory, written there by the
    pipeline. What this adds is what a directory cannot answer: `search_runs` can rank EDA
    runs by zero-demand rate or stockout incidence, not merely tell them apart.
    """
    fields = metadata.model_dump()
    mlflow.log_params({key: "null" if value is None else value for key, value in fields.items()})
    mlflow.set_tags({"run_mode": "eda"})
    mlflow.log_metrics(_numeric_metrics(dataset_summary))


def logged_run_dirs(experiment_name: str) -> dict[str, Path]:
    """Every run under `experiment_name`, newest first, mapped to its artifact directory.

    The directory is the point. `log_artifacts` mirrors a run directory into the store, so what
    comes back here is shaped exactly like the `reports/` folder that produced it -- which is
    what lets a reader move from one to the other without learning a second layout.

    Returns an empty mapping when the experiment does not exist yet, rather than creating it:
    reading the store must not write to it.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {}

    found = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    dirs: dict[str, Path] = {}
    for run in found:
        name = run.info.run_name
        path = Path(run.info.artifact_uri.removeprefix("file://"))
        if name and name not in dirs and path.is_dir():
            dirs[name] = path
    return dirs


def index_run_directory(
    run_dir: Path, experiment_name: str, run_name: str | None = None
) -> str | None:
    """Record a finished run directory in MLflow, reading whatever it happens to contain.

    The regular loggers are handed live objects by the pipeline; this one has only the files.
    It exists because the store is a gitignored sqlite database with no backup: without a way
    to rebuild the index from `reports/`, losing `mlflow.db` would lose every run's record,
    and because runs written before the instrumentation existed are otherwise invisible.
    """
    if not run_dir.is_dir():
        return None

    _open_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name or run_dir.name) as active:
        for name in ("backtest_metadata.json", "eda_metadata.json"):
            candidate = run_dir / name
            if candidate.exists():
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                mlflow.log_params(
                    {
                        key: "null" if value is None else value
                        for key, value in loaded.items()
                        if not isinstance(value, dict | list)
                    }
                )
        for name in ("metrics_summary.csv", "cost_summary.csv", "dataset_summary.csv"):
            candidate = run_dir / name
            if candidate.exists():
                mlflow.log_metrics(_numeric_metrics(pd.read_csv(candidate)))

        mlflow.set_tags({"source_dir": str(run_dir), "backfilled": "true"})
        mlflow.log_artifacts(str(run_dir))
        _write_run_identity(
            Path(active.info.artifact_uri.removeprefix("file://")),
            run_name=run_name or run_dir.name,
            run_id=active.info.run_id,
            experiment_name=experiment_name,
        )
        return str(active.info.run_id)


def resolve_run_dir(value: str | Path) -> Path:
    """A run directory, given either a path to one or the name of a recorded run.

    Command lines used to name runs by path, which stops working once the artifacts live
    behind a UUID. Accepting the name keeps `docs/runs.md` readable: a run is cited by the
    name it was given, not by the directory it happens to sit in.
    """
    candidate = Path(value)
    if candidate.is_dir():
        return candidate
    for experiment in (EXPERIMENT_RUNS, EXPERIMENT_EDA):
        found = logged_run_dirs(experiment).get(str(value))
        if found is not None:
            return found
    raise FileNotFoundError(f"'{value}' no es un directorio ni una corrida registrada.")


RUN_IDENTITY_FILE = "mlflow_run.json"


@contextmanager
def open_run_directory(run_name: str, experiment_name: str) -> Iterator[Path]:
    """Start a run and yield the directory its artifacts live in, to be written into directly.

    For a local artifact store, writing into that directory *is* logging: MLflow lists and
    serves whatever the directory holds, nested paths included. So the pipeline writes once,
    rather than writing a run directory and mirroring it afterwards.

    The run stays open for the duration, which is what lets a caller add params and metrics
    beside the files. A caller that raises leaves the run FAILED with its partial output
    attached -- the same diagnostic value an abandoned `reports/` directory had.
    """
    _open_experiment(experiment_name)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    full_name = f"{run_name}_{timestamp}"

    with mlflow.start_run(run_name=full_name) as active:
        run_dir = Path(active.info.artifact_uri.removeprefix("file://"))
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_identity(
            run_dir,
            run_name=full_name,
            run_id=active.info.run_id,
            experiment_name=experiment_name,
        )
        yield run_dir


def _write_run_identity(run_dir: Path, *, run_name: str, run_id: str, experiment_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_IDENTITY_FILE).write_text(
        json.dumps(
            {"run_name": run_name, "run_id": run_id, "experiment": experiment_name}, indent=2
        ),
        encoding="utf-8",
    )


def log_ops_metadata(
    settings: Settings,
    cadence_summary: pd.DataFrame,
    cadence_comparison: pd.DataFrame,
    n_origins: int,
    n_retrain_events: int,
) -> None:
    """Attach the OPS plane's config and per-cadence results to the run already open.

    The cadence comparison is logged with its `conclusive` and `underpowered` flags beside the
    percentages, because the percentage alone is not citable: `docs/runs.md` allows quoting the
    retrain saving only when the comparison says conclusive, and a metric store that kept the
    number and dropped the caveat would make the unciteable figure the easiest one to find.
    """
    mlflow.log_params(_flat_params(settings))
    mlflow.set_tags(
        {
            "git_commit": get_git_commit(),
            "run_mode": settings.project.run_mode,
            "config_hash": build_config_hash(settings),
            "origins": n_origins,
            "retrain_events": n_retrain_events,
        }
    )
    mlflow.log_metrics(_numeric_metrics(cadence_summary))
    mlflow.log_metrics(_numeric_metrics(cadence_comparison))
    if not cadence_comparison.empty and "conclusive" in cadence_comparison:
        mlflow.set_tags(
            {
                "cadence_conclusive": bool(cadence_comparison["conclusive"].all()),
                "cadence_underpowered": bool(cadence_comparison["underpowered"].any()),
            }
        )
