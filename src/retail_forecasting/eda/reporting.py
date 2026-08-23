from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from retail_forecasting.eda.figure_exports import MEMORIA_FIGURE_EXPORTS
from retail_forecasting.eda.plots import render_eda_plots
from retail_forecasting.utils.io import make_run_directory


@dataclass
class EdaArtifacts:
    panel: pd.DataFrame
    dataset_summary: pd.DataFrame
    missingness_summary: pd.DataFrame
    numeric_summary: pd.DataFrame
    series_summary: pd.DataFrame
    temporal_summary: pd.DataFrame
    weekday_summary: pd.DataFrame
    series_gap_summary: pd.DataFrame
    stockout_summary: pd.DataFrame
    stockout_by_series_summary: pd.DataFrame
    stockout_demand_bands: pd.DataFrame
    correlation_summary: pd.DataFrame
    run_directory: Path | None = None


def write_eda_artifacts(
    artifacts: EdaArtifacts,
    output_dir: str | Path,
    run_name: str,
    make_plots: bool,
    memoria_dir: str | Path | None = None,
) -> EdaArtifacts:
    """Write the summary tables and the figures, and copy the thesis ones into memoria."""
    run_dir = make_run_directory(output_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "dataset_summary.csv": artifacts.dataset_summary,
        "missingness_summary.csv": artifacts.missingness_summary,
        "numeric_summary.csv": artifacts.numeric_summary,
        "series_summary.csv": artifacts.series_summary,
        "temporal_summary.csv": artifacts.temporal_summary,
        "weekday_summary.csv": artifacts.weekday_summary,
        "series_gap_summary.csv": artifacts.series_gap_summary,
        "stockout_summary.csv": artifacts.stockout_summary,
        "stockout_by_series_summary.csv": artifacts.stockout_by_series_summary,
        "stockout_demand_bands.csv": artifacts.stockout_demand_bands,
        "correlation_summary.csv": artifacts.correlation_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(run_dir / filename, index=False)

    if make_plots:
        render_eda_plots(
            panel=artifacts.panel,
            weekday_summary=artifacts.weekday_summary,
            series_summary=artifacts.series_summary,
            output_dir=run_dir,
        )

    if memoria_dir is not None:
        export_figures_to_memoria(
            run_directory=run_dir,
            memoria_dir=memoria_dir,
        )

    artifacts.run_directory = run_dir
    return artifacts


def export_figures_to_memoria(
    run_directory: str | Path,
    memoria_dir: str | Path,
) -> None:
    """Copy the figures the thesis includes into its tree."""
    run_dir = Path(run_directory)
    memoria_root = Path(memoria_dir)
    target_dir = memoria_root / "figures" / "eda"
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in MEMORIA_FIGURE_EXPORTS:
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)
