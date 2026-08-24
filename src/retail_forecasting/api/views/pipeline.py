"""Trigger the pipeline and stream its log into the dashboard console."""

from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from retail_forecasting.api.services.pipeline import (
    PipelineBusyError,
    PipelineRunner,
    RateLimitedError,
)
from retail_forecasting.api.store import get_store

_runner: PipelineRunner | None = None


def get_runner() -> PipelineRunner:
    """Shared runner, built from settings on first use."""
    global _runner
    if _runner is None:
        _runner = PipelineRunner(
            state_dir=settings.STATE_DIR,
            config_path=settings.CONFIG_PATH,
            rate_limit_max=settings.RUN_RATE_LIMIT_MAX,
            rate_limit_window=settings.RUN_RATE_LIMIT_WINDOW,
        )
    return _runner


def _client_key(request: HttpRequest) -> str:
    return str(request.META.get("REMOTE_ADDR", "unknown"))


def _console(
    request: HttpRequest, runner: PipelineRunner, error: str | None = None
) -> HttpResponse:
    """Render the console overlay for the runner's current state."""
    return render(
        request,
        "partials/pipeline_console.html",
        {
            "status": runner.state.status,
            "error": error or runner.state.error,
            "logs": runner.read_log(),
        },
    )


@require_POST
def run(request: HttpRequest) -> HttpResponse:
    """Start a pipeline run and open the console."""
    runner = get_runner()
    try:
        runner.check_rate_limit(_client_key(request))
    except RateLimitedError as exc:
        return _console(request, runner, error=str(exc))

    try:
        # Predictions are about to change; drop the cached frame so the
        # dashboard re-reads them once the run succeeds.
        runner.start(on_start=get_store().invalidate)
    except PipelineBusyError:
        return _console(request, runner, error="Ya hay una ejecución en curso.")
    except FileNotFoundError as exc:
        return _console(request, runner, error=str(exc))

    return _console(request, runner)


@require_GET
def status(request: HttpRequest) -> HttpResponse:
    """Console body, polled by htmx while a run is in flight."""
    runner = get_runner()
    if runner.state.status == "success":
        # New artifacts on disk: make the next page load pick them up.
        get_store().invalidate()
    return _console(request, runner)


@require_POST
def reset(request: HttpRequest) -> HttpResponse:
    """Force-release a stuck lock after a crashed run."""
    runner = get_runner()
    runner.reset()
    return _console(request, runner)
