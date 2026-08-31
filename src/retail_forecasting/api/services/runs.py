"""Discovery and caching of pipeline artifacts on disk.

Runs are discovered through MLflow, which is the index of what exists, and read as
plain files from the artifact directory MLflow hands back. The pipeline writes into that
directory in the first place, so there is nothing to mirror and one copy to read.

Discovery through the index also removes a class of bug rather than guarding against
it: a run name now has to be a key MLflow already knows, so there is no path to
traverse and no name to sanitise.

Deliberately free of any web framework. The Django views construct or receive an
:class:`ArtifactStore` and never touch ``pandas`` file paths themselves, which is
what makes both layers testable in isolation.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from retail_forecasting.config import load_config
from retail_forecasting.tracking import (
    EXPERIMENT_EDA,
    EXPERIMENT_RUNS,
    logged_run_dirs,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/experiment/default.yaml")

_PREDICTION_COLUMNS = ["date", "series_id", "y_true", "y_pred", "data_strategy", "model_name"]


class ArtifactError(Exception):
    """Base class for artifact-access failures."""


class NoPredictionsError(ArtifactError):
    """No run directory containing ``predictions.csv`` exists yet."""


class RunNotFoundError(ArtifactError):
    """A named run does not exist, or does not hold the required artifact."""


class ArtifactStore:
    """Cached, read-only access to the artifacts of a recorded run.

    The cache is keyed on nothing: it holds one latest-run snapshot, and
    :meth:`invalidate` clears it. That is the same lifetime the FastAPI version
    had, and it matches how the data actually changes — only when the pipeline
    finishes a new run.
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}

    # ── Cache control ─────────────────────────────────────────────────────────

    def invalidate(self) -> None:
        """Drop every cached artifact. Called when a pipeline run starts."""
        with self._lock:
            self._cache.clear()

    @property
    def is_loaded(self) -> bool:
        """Whether a predictions frame is currently cached (used by /health)."""
        return "df" in self._cache

    # ── Run discovery ─────────────────────────────────────────────────────────

    def latest_run_path(self) -> Path:
        """Newest run holding ``predictions.csv``.

        Raises:
            NoPredictionsError: if no recorded run holds one.
        """
        cached = self._cache.get("run_path")
        if isinstance(cached, Path):
            return cached

        runs = self.runs_with("predictions.csv")
        if not runs:
            raise NoPredictionsError("No recorded run holds predictions.csv.")

        latest = next(iter(runs.values()))
        with self._lock:
            self._cache["run_path"] = latest
        return latest

    def runs_with(self, *required_files: str, experiment: str = EXPERIMENT_RUNS) -> dict[str, Path]:
        """Recorded runs holding all of ``required_files``, newest first, keyed by name.

        Keyed rather than a bare list because an artifact directory is called `artifacts`
        under a UUID: callers that showed a run's name were showing `artifacts`.
        """
        return {
            name: path
            for name, path in logged_run_dirs(experiment).items()
            if all((path / f).exists() for f in required_files)
        }

    def resolve_run(
        self,
        run_name: str,
        *,
        requires: str = "predictions.csv",
        experiment: str = EXPERIMENT_RUNS,
    ) -> Path:
        """Return the named run's artifact directory.

        Raises:
            RunNotFoundError: if no such run is recorded, or it lacks ``requires``.
        """
        path = logged_run_dirs(experiment).get(run_name)
        if path is None or not (path / requires).exists():
            raise RunNotFoundError(f"Run '{run_name}' not found.")
        return path

    def list_runs(self) -> list[str]:
        """Experiment runs that carry the full metrics/cost artifact set."""
        return list(self.runs_with("predictions.csv", "metrics_summary.csv", "cost_summary.csv"))

    # ── EDA runs ──────────────────────────────────────────────────────────────

    def latest_eda_path(self) -> Path | None:
        """Newest recorded EDA run, or None if the module never ran."""
        recorded = list(logged_run_dirs(EXPERIMENT_EDA).values())
        return recorded[0] if recorded else None

    def list_eda_runs(self) -> list[str]:
        """Names of every recorded EDA run, newest first."""
        return list(logged_run_dirs(EXPERIMENT_EDA))

    def resolve_eda_run(self, run_name: str | None) -> Path:
        """Return the named EDA run, or the newest when unnamed.

        Raises:
            RunNotFoundError: if nothing matches.
        """
        recorded = logged_run_dirs(EXPERIMENT_EDA)
        if run_name:
            path = recorded.get(run_name)
            if path is None:
                raise RunNotFoundError("EDA run not found.")
            return path

        if not recorded:
            raise RunNotFoundError("No EDA run recorded.")
        return next(iter(recorded.values()))

    # ── Predictions ───────────────────────────────────────────────────────────

    def champion(self) -> tuple[str, str]:
        """(data_strategy, model_name) of the champion: the registry first, then the config.

        Same precedence the pipeline applies when it resolves which model to retrain or
        serve. Reading only the config made the dashboard name the configured champion
        after a promotion had already replaced it. Falls back to the historical defaults
        when neither can be read, so a malformed config degrades the dashboard rather than
        breaking it.
        """
        from retail_forecasting.forecasting.pipeline import (
            champion_registry_path,
            load_champion_registry,
            resolve_champion_reference,
        )

        try:
            settings = load_config(self.config_path)
        except Exception:
            logger.warning("Could not load %s; using default champion.", self.config_path)
            return "Observed", "catboost"
        try:
            registry = load_champion_registry(champion_registry_path(settings))
        except Exception:
            logger.warning("Could not read the champion registry; falling back to the config.")
            registry = None
        reference = resolve_champion_reference(settings, registry)
        return str(reference.data_strategy), str(reference.model_name)

    def latest_predictions(self, run_name: str | None = None) -> pd.DataFrame:
        """Champion-filtered predictions frame for the specified or latest run, cached."""
        cache_key = f"df_{run_name or 'latest'}"
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        if run_name:
            run_path = self.resolve_run(run_name)
        else:
            run_path = self.latest_run_path()

        predictions_csv = run_path / "validation_predictions.csv"
        if not predictions_csv.exists():
            predictions_csv = run_path / "predictions.csv"
        try:
            df = pd.read_csv(predictions_csv, usecols=_PREDICTION_COLUMNS)
        except ValueError:
            # Older runs may not carry every column; fall back to reading all.
            df = pd.read_csv(predictions_csv)

        strategy, model = self.champion()
        if "data_strategy" in df.columns:
            if strategy in df["data_strategy"].values:
                df = df[df["data_strategy"] == strategy]
            elif len(df["data_strategy"].unique()) > 0:
                df = df[df["data_strategy"] == df["data_strategy"].unique()[0]]

        if "model_name" in df.columns:
            if model in df["model_name"].values:
                df = df[df["model_name"] == model]
            elif len(df["model_name"].unique()) > 0:
                df = df[df["model_name"] == df["model_name"].unique()[0]]

        grouped = {str(series_id): group for series_id, group in df.groupby("series_id")}

        with self._lock:
            self._cache[cache_key] = df
            self._cache[f"grouped_{run_name or 'latest'}"] = grouped
        return df

    def grouped_predictions(self, run_name: str | None = None) -> dict[str, pd.DataFrame]:
        """``{series_id: frame}`` for the specified or latest predictions, cached."""
        self.latest_predictions(run_name)
        grouped: dict[str, pd.DataFrame] = self._cache.get(f"grouped_{run_name or 'latest'}", {})
        return grouped

    def read_csv(self, run_path: Path, filename: str) -> pd.DataFrame | None:
        """Read ``filename`` from a run directory, or None when absent."""
        path = run_path / filename
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            logger.warning("Failed to read %s", path)
            return None
