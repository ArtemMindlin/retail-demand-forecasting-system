"""The tuned-hyperparameter store: what `tune_forecasting` writes and the pipeline reads.

Its own module rather than part of `forecasting_tuning` so that reading does not depend on
searching. `pipeline` only ever reads, and `forecasting_tuning` reaches into `imputation_tuning`
for its shared progress helpers -- a dependency the experiment pipeline has no business having.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_config import BoostingBackend

logger = logging.getLogger(__name__)


CORE_PARAMS = ("n_estimators", "learning_rate", "max_depth")


def default_params(settings: Settings) -> dict[str, Any]:
    """The untuned config defaults, in the shape a candidate has.

    The baseline both gates measure against: a winner that does not beat this is not worth
    persisting, and the pipeline already has these in its YAML.
    """
    return {
        "n_estimators": settings.models.n_estimators,
        "learning_rate": settings.models.learning_rate,
        "max_depth": settings.models.max_depth,
    }


def read_tuned_params(path: Path, backend: BoostingBackend) -> dict[str, Any] | None:
    """One backend's persisted block, or None when there is nothing to read.

    Returns the whole block -- ``params``, ``tuned_on`` and ``gate`` -- rather than just the
    hyperparameters, because a consumer has to be able to check what the winner was tuned on
    before applying it: hyperparameters do not transfer across training sizes (invariant 41).

    Never raises on a missing or malformed file. A tuned params file is an optimization, and
    the pipeline has working defaults in its YAML, so an unreadable one degrades to those with
    a warning instead of taking a run down.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("no se pudo leer %s (%s), se usan los defaults del YAML", path, exc)
        return None
    block = payload.get(backend) if isinstance(payload, dict) else None
    return block if isinstance(block, dict) else None


def _load_payload(path: Path) -> dict[str, Any]:
    """The whole file as a dict, empty when absent or unreadable."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_backend_block(path: Path, backend: BoostingBackend, block: dict[str, Any]) -> None:
    """Replace one backend's block, leaving the other backend's untouched.

    Read-modify-write rather than one file per backend, because the two searches are separate
    runs of the same mode and the file is meant to be read as one answer.
    """
    payload = _load_payload(path)
    payload[backend] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _drop_backend_block(path: Path, backend: BoostingBackend) -> bool:
    """Remove one backend's block after a failed gate; delete the file once it is empty.

    Only this backend's block: the imputer deletes its whole params file, which is right when
    the file holds one answer. Here the other backend's winner passed its own gates in its own
    run and is not collateral.

    Returns:
        True when something was removed.
    """
    payload = _load_payload(path)
    if backend not in payload:
        return False
    del payload[backend]
    if payload:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        path.unlink()
    return True


def resolve_backend_params(
    settings: Settings, backend: BoostingBackend, n_series: int
) -> tuple[dict[str, Any], str]:
    """The hyperparameters a run should train `backend` with, and where they came from.

    Falls back to the YAML defaults whenever the persisted winner cannot be trusted for THIS
    run's panel. The fingerprint check is the point: invariant 41 measured a winner tuned at 50
    series coming out 12% worse than not tuning at 500, so applying a file across scales is not
    a small risk. Silence would be the dangerous outcome, hence a log line either way.

    Returns:
        ``(params, provenance)``, where provenance is a short label for the run log.
    """
    params_path = settings.models.models_dir / settings.models.forecasting_params_filename
    defaults = default_params(settings)
    block = read_tuned_params(params_path, backend)
    if block is None:
        return defaults, "defaults del YAML (sin fichero afinado)"

    params = block.get("params")
    if not isinstance(params, dict):
        return defaults, "defaults del YAML (bloque sin parámetros)"

    tuned_on = block.get("tuned_on") or {}
    current = {
        "n_series": n_series,
        "horizon": settings.dataset.horizon,
        "lags": list(settings.features.lags),
        "rolling_windows": list(settings.features.rolling_windows),
    }
    mismatched = {
        key: (tuned_on.get(key), value)
        for key, value in current.items()
        if tuned_on.get(key) != value
    }
    if mismatched:
        detail = ", ".join(
            f"{k}: afinado {a} != run {b}" for k, (a, b) in sorted(mismatched.items())
        )
        return defaults, f"defaults del YAML (el fichero no encaja -- {detail})"

    gate = block.get("gate") or {}
    improvement = gate.get("improvement_pct")
    suffix = "" if improvement is None else f", mejora {improvement:+.2f}%"
    return dict(params), f"afinado por tune_forecasting{suffix}"
