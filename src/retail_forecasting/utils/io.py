from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def make_run_directory(base_dir: str | Path, run_name: str) -> Path:
    """Create a timestamped run directory below a base output path.

    Args:
        base_dir: Parent directory where run folders are stored.
        run_name: Prefix used in the generated run directory name.

    Returns:
        The created run directory path.

    Notes:
        The directory name includes a UTC timestamp to provide stable,
        time-based uniqueness across runs.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def quantile_column_name(quantile: float) -> str:
    """Build the canonical column name for a quantile prediction."""
    normalized = str(quantile).replace(".", "_")
    return f"q_{normalized}"


def quantile_level_from_column(column: str) -> float:
    """Recover the quantile level from a canonical quantile column name.

    Inverse of :func:`quantile_column_name` (e.g. ``"q_0_9" -> 0.9``).
    """
    return float(column.replace("q_", "").replace("_", "."))


def dataframe_to_markdown(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    """Render a DataFrame as a simple Markdown table.

    Args:
        frame: Source DataFrame to render.
        columns: Optional subset of columns to include in order.

    Returns:
        A Markdown table string, or ``"_No data available._"`` when the
        selected frame is empty.

    Notes:
        Float values are formatted through ``_format_markdown_value`` before
        rendering.
    """
    subset = frame if columns is None else frame.loc[:, list(columns)]
    if subset.empty:
        return "_No data available._"

    text_frame = subset.copy()
    for column in text_frame.columns:
        text_frame[column] = text_frame[column].map(_format_markdown_value)

    headers = " | ".join(str(column) for column in text_frame.columns)
    separator = " | ".join(["---"] * len(text_frame.columns))
    rows = [
        " | ".join(str(value) for value in row)
        for row in text_frame.itertuples(index=False, name=None)
    ]
    return "\n".join([f"| {headers} |", f"| {separator} |", *[f"| {row} |" for row in rows]])


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
