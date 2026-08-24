from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Sentinel fold id marking holdout (non walk-forward) predictions so global
# summaries can exclude them. Written by the pipeline, read by evaluation.metrics.
HOLDOUT_FOLD_ID = 999


def model_file_path(models_dir: Path | str, backend_name: str) -> Path:
    """Where a trained backend's pickle lives. In `utils` because `evaluation` needs it too."""
    return Path(models_dir) / f"{backend_name}.pkl"


def quantile_column_name(quantile: float) -> str:
    """Build the canonical column name for a quantile prediction."""
    normalized = str(quantile).replace(".", "_")
    return f"q_{normalized}"


def quantile_level_from_column(column: str) -> float:
    """Recover the quantile level from a canonical quantile column name.

    Inverse of :func:`quantile_column_name` (e.g. ``"q_0_9" -> 0.9``).
    """
    return float(column.replace("q_", "").replace("_", "."))


def winkler_score(actual: Any, lower: Any, upper: Any, alpha: float) -> float:
    """Winkler interval score for a central ``(1 - alpha)`` prediction interval.

    A proper scoring rule that penalizes both wide intervals and observations
    falling outside them (asymmetric ``2/alpha`` penalty). Lower is better.
    Accepts pandas Series or numpy arrays. Lives in ``utils`` so both the
    ``models`` and ``evaluation`` layers can reuse it without crossing layer
    boundaries.
    """
    width = upper - lower
    under_penalty = (2.0 / alpha) * (lower - actual) * (actual < lower)
    over_penalty = (2.0 / alpha) * (actual - upper) * (actual > upper)
    return float(np.mean(width + under_penalty + over_penalty))


def rearrange_quantiles(raw_predictions: list[np.ndarray]) -> np.ndarray:
    """Apply Chernozhukov rearrangement to enforce quantile monotonicity.

    Sorts predicted quantile values in ascending order per sample, ensuring
    q[1] <= q[2] <= ... <= q[M] without worsening the Pinball loss
    (Chernozhukov, Fernández-Val & Galichon, Econometrica 2010).

    Args:
        raw_predictions: List of 1-D arrays, one per quantile level (sorted).

    Returns:
        2-D array of shape (n_samples, n_quantiles) with monotone rows.
    """
    matrix = np.column_stack(raw_predictions)
    return np.sort(matrix, axis=1)


def _format_markdown_value(value: object) -> object:
    """Format a value for Markdown rendering.

    Args:
        value: Value to normalize for display.

    Returns:
        The formatted value.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    return value
