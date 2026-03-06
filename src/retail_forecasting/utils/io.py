from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = path if isinstance(path, Path) else Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_run_directory(base_dir: str | Path, run_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_directory(Path(base_dir) / f"{run_name}_{timestamp}")
    return run_dir


def quantile_column_name(quantile: float) -> str:
    normalized = str(quantile).replace(".", "_")
    return f"q_{normalized}"


def dataframe_to_markdown(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
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


def _format_markdown_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value
