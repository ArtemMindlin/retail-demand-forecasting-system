"""Background execution of the forecasting pipeline.

The pipeline runs as a subprocess (``python -m retail_forecasting.run``) writing
its output to ``var/active_run.log``, which the dashboard tails. A lock
allows only one run at a time; a small per-IP rate limit keeps the trigger from
being hammered.

No web framework here — :class:`PipelineRunner` is driven by the views.
"""

from __future__ import annotations

import collections
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_RELATIVE_PATH = Path("active_run.log")


class PipelineBusyError(Exception):
    """A run is already in progress."""


class RateLimitedError(Exception):
    """Too many run requests from one client."""


@dataclass
class RunState:
    """Snapshot of the most recent (or current) pipeline execution."""

    status: str = "idle"  # idle | running | success | failed
    error: str | None = None
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self.status == "running"


@dataclass
class _RateBucket:
    window_seconds: int
    max_calls: int
    hits: dict[str, collections.deque[float]] = field(
        default_factory=lambda: collections.defaultdict(collections.deque)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self.lock:
            bucket = self.hits[key]
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.max_calls:
                minutes = self.window_seconds // 60
                raise RateLimitedError(
                    f"Demasiadas solicitudes. Máximo {self.max_calls} ejecuciones "
                    f"por {minutes} minutos."
                )
            bucket.append(now)


class PipelineRunner:
    """Serialized, observable background runs of the forecasting pipeline."""

    def __init__(
        self,
        state_dir: Path,
        config_path: Path,
        rate_limit_max: int = 3,
        rate_limit_window: int = 600,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.config_path = Path(config_path)
        self.state = RunState()
        self._lock = threading.Lock()
        self._rate = _RateBucket(rate_limit_window, rate_limit_max)

    @property
    def log_path(self) -> Path:
        return self.state_dir / LOG_RELATIVE_PATH

    def check_rate_limit(self, client_key: str) -> None:
        """Raise :class:`RateLimitedError` when ``client_key`` has run too often."""
        self._rate.check(client_key)

    def start(self, on_start: collections.abc.Callable[[], None] | None = None) -> None:
        """Launch a run in a background thread.

        Raises:
            PipelineBusyError: if a run is already in progress.
            FileNotFoundError: if the configuration file is missing.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        if self._lock.locked():
            raise PipelineBusyError("A pipeline run is already in progress.")

        if on_start is not None:
            on_start()

        thread = threading.Thread(target=self._execute, daemon=True, name="pipeline-run")
        thread.start()

    def reset(self) -> None:
        """Force-release the lock. For when the process died mid-run."""
        if self._lock.locked():
            try:
                self._lock.release()
            except RuntimeError:
                # Released by its owning thread in the meantime; nothing to do.
                pass
        self.state = RunState()

    def read_log(self) -> str:
        """Current contents of the active run log ("" when there is none)."""
        if not self.log_path.exists():
            return ""
        try:
            return self.log_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Error reading log file: {exc}"

    def _execute(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self.state = RunState(status="running", started_at=time.monotonic())
            self.state_dir.mkdir(parents=True, exist_ok=True)

            header = f"--- Pipeline Execution Started at {datetime.now(UTC).isoformat()} ---\n"
            self.log_path.write_text(header, encoding="utf-8")

            # sys.executable keeps the subprocess in this exact environment.
            command = [
                sys.executable,
                "-m",
                "retail_forecasting.run",
                "--config",
                str(self.config_path),
            ]
            with self.log_path.open("a", encoding="utf-8") as log_file:
                process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                    command, stdout=log_file, stderr=subprocess.STDOUT, text=True
                )
                return_code = process.wait()

            if return_code == 0:
                self.state.status = "success"
            else:
                self.state.status = "failed"
                self.state.error = f"Pipeline exited with code {return_code}."
                with self.log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n[ERROR] Pipeline failed with exit code {return_code}\n")
        except Exception as exc:  # noqa: BLE001 - surfaced to the console, never swallowed
            logger.exception("Pipeline run failed")
            self.state.status = "failed"
            self.state.error = str(exc)
            try:
                with self.log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n[EXCEPTION] Failed to run pipeline: {exc}\n")
            except OSError:
                pass
        finally:
            self.state.ended_at = time.monotonic()
            self._lock.release()
