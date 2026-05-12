from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path.

    Args:
        path: Directory path to create or normalize.

    Returns:
        The normalized directory path.

    Notes:
        The directory is created with ``parents=True`` and ``exist_ok=True``.
    """
    directory = path if isinstance(path, Path) else Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
    run_dir = ensure_directory(Path(base_dir) / f"{run_name}_{timestamp}")
    return run_dir


def quantile_column_name(quantile: float) -> str:
    """Build the canonical column name for a quantile prediction.

    Args:
        quantile: Quantile level used by the forecast output.

    Returns:
        The column name associated with the quantile.
    """
    normalized = str(quantile).replace(".", "_")
    return f"q_{normalized}"


def dataframe_to_markdown(
    frame: pd.DataFrame, columns: Iterable[str] | None = None
) -> str:
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
    return "\n".join(
        [f"| {headers} |", f"| {separator} |", *[f"| {row} |" for row in rows]]
    )


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
