"""Discovery and caching of pipeline artifacts on disk.

The system has no database: every figure the dashboard shows is read from a run
directory under ``reports/``. This module owns that access — locating the latest
run, loading and caching its predictions frame, and validating user-supplied run
names against path traversal.

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

logger = logging.getLogger(__name__)

DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_CONFIG_PATH = Path("configs/experiment.yaml")

# Directory-name prefixes that are never experiment runs.
_NON_RUN_PREFIXES = (".", "models", "ablation")

_PREDICTION_COLUMNS = ["date", "series_id", "y_true", "y_pred", "data_strategy", "model_name"]


class ArtifactError(Exception):
    """Base class for artifact-access failures."""


class NoPredictionsError(ArtifactError):
    """No run directory containing ``predictions.csv`` exists yet."""


class RunNotFoundError(ArtifactError):
    """A named run does not exist, or does not hold the required artifact."""


class ArtifactStore:
    """Cached, read-only access to the run artifacts under ``reports/``.

    The cache is keyed on nothing: it holds one latest-run snapshot, and
    :meth:`invalidate` clears it. That is the same lifetime the FastAPI version
    had, and it matches how the data actually changes — only when the pipeline
    finishes a new run.
    """

    def __init__(
        self,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.reports_dir = Path(reports_dir)
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
        """Newest run directory that contains ``predictions.csv``.

        Raises:
            NoPredictionsError: if ``reports/`` is missing or holds no such run.
        """
        cached = self._cache.get("run_path")
        if isinstance(cached, Path):
            return cached

        if not self.reports_dir.exists():
            raise NoPredictionsError(f"{self.reports_dir}/ directory does not exist.")

        runs = [
            d
            for d in self.reports_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith(_NON_RUN_PREFIXES)
            and (d / "predictions.csv").exists()
        ]
        if not runs:
            raise NoPredictionsError(f"No runs with predictions.csv found in {self.reports_dir}/.")

        # Run directories are timestamp-prefixed, so lexical sort is chronological.
        latest = max(runs, key=lambda d: d.name)
        with self._lock:
            self._cache["run_path"] = latest
        return latest

    def runs_with(self, *required_files: str) -> list[Path]:
        """Run directories holding all of ``required_files``, newest first."""
        if not self.reports_dir.exists():
            return []
        return [
            d
            for d in sorted(
                self.reports_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
            )
            if d.is_dir() and all((d / f).exists() for f in required_files)
        ]

    def resolve_run(self, run_name: str, *, requires: str = "predictions.csv") -> Path:
        """Validate ``run_name`` against path traversal and return its directory.

        Raises:
            RunNotFoundError: if the name is unsafe, missing, or lacks ``requires``.
        """
        safe_name = Path(run_name).name
        if not safe_name or safe_name != run_name or run_name.startswith("."):
            raise RunNotFoundError(f"Run '{run_name}' not found.")

        run_path = self.reports_dir / run_name
        if not run_path.is_dir() or not (run_path / requires).exists():
            raise RunNotFoundError(f"Run '{run_name}' not found.")
        return run_path

    def list_runs(self) -> list[str]:
        """Experiment runs that carry the full metrics/cost artifact set."""
        return [
            d.name
            for d in self.runs_with("predictions.csv", "metrics_summary.csv", "cost_summary.csv")
        ]

    # ── EDA runs ──────────────────────────────────────────────────────────────

    def latest_eda_path(self) -> Path | None:
        """Newest ``eda_*`` directory, or None if the EDA module never ran."""
        if not self.reports_dir.exists():
            return None
        eda_dirs = [
            d for d in self.reports_dir.iterdir() if d.is_dir() and d.name.startswith("eda_")
        ]
        return max(eda_dirs, key=lambda d: d.name) if eda_dirs else None

    def list_eda_runs(self) -> list[str]:
        """Names of every EDA run, newest first."""
        if not self.reports_dir.exists():
            return []
        return sorted(
            (
                d.name
                for d in self.reports_dir.iterdir()
                if d.is_dir() and d.name.startswith("eda_")
            ),
            reverse=True,
        )

    def resolve_eda_run(self, run_name: str | None) -> Path:
        """Return the requested EDA directory, or the latest when unnamed.

        Raises:
            RunNotFoundError: if nothing matches.
        """
        if run_name:
            safe_name = Path(run_name).name
            if safe_name != run_name or not run_name.startswith("eda_"):
                raise RunNotFoundError("EDA run not found.")
            path = self.reports_dir / run_name
            if not path.is_dir():
                raise RunNotFoundError("EDA run not found.")
            return path

        latest = self.latest_eda_path()
        if latest is None:
            raise RunNotFoundError(f"No EDA report found in {self.reports_dir}/.")
        return latest

    # ── Predictions ───────────────────────────────────────────────────────────

    def champion(self) -> tuple[str, str]:
        """(data_strategy, model_name) of the champion, per the config file.

        Falls back to the historical defaults when the config cannot be read, so
        a malformed config degrades the dashboard rather than breaking it.
        """
        try:
            settings = load_config(self.config_path)
        except Exception:
            logger.warning("Could not load %s; using default champion.", self.config_path)
            return "Observed", "catboost"
        return (
            str(settings.business.champion_data_strategy),
            str(settings.business.champion_model_name),
        )

    def latest_predictions(self) -> pd.DataFrame:
        """Champion-filtered predictions frame from the latest run, cached.

        Raises:
            NoPredictionsError: if no run with predictions exists.
        """
        cached = self._cache.get("df")
        if cached is not None:
            return cached

        run_path = self.latest_run_path()
        predictions_csv = run_path / "predictions.csv"
        try:
            df = pd.read_csv(predictions_csv, usecols=_PREDICTION_COLUMNS)
        except ValueError:
            # Older runs may not carry every column; fall back to reading all.
            df = pd.read_csv(predictions_csv)

        strategy, model = self.champion()
        if "data_strategy" in df.columns:
            df = df[df["data_strategy"] == strategy]
        if "model_name" in df.columns:
            df = df[df["model_name"] == model]

        with self._lock:
            self._cache["df"] = df
            self._cache["grouped"] = {
                str(series_id): group for series_id, group in df.groupby("series_id")
            }
        return df

    def grouped_predictions(self) -> dict[str, pd.DataFrame]:
        """``{series_id: frame}`` for the latest predictions, cached alongside it."""
        self.latest_predictions()
        grouped: dict[str, pd.DataFrame] = self._cache.get("grouped", {})
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
