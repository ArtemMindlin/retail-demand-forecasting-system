from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from retail_forecasting.config import Settings
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.eda.plots import render_eda_plots
from retail_forecasting.eda.profiling import (
    build_correlation_summary,
    build_dataset_summary,
    build_missingness_summary,
    build_numeric_summary,
    build_series_summary,
)
from retail_forecasting.eda.stockout import (
    build_stockout_by_series_summary,
    build_stockout_demand_bands,
    build_stockout_summary,
)
from retail_forecasting.eda.temporal import (
    build_series_gap_summary,
    build_temporal_summary,
    build_weekday_summary,
)
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


def run_eda(settings: Settings, split: str = "train") -> EdaArtifacts:
    """Run EDA on the canonical prepared panel and persist artifacts."""
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split=split,
    )

    artifacts = EdaArtifacts(
        panel=panel,
        dataset_summary=build_dataset_summary(panel),
        missingness_summary=build_missingness_summary(panel),
        numeric_summary=build_numeric_summary(panel),
        series_summary=build_series_summary(panel),
        temporal_summary=build_temporal_summary(panel),
        weekday_summary=build_weekday_summary(panel),
        series_gap_summary=build_series_gap_summary(panel),
        stockout_summary=build_stockout_summary(panel),
        stockout_by_series_summary=build_stockout_by_series_summary(panel),
        stockout_demand_bands=build_stockout_demand_bands(panel),
        correlation_summary=build_correlation_summary(panel),
    )

    return write_eda_artifacts(
        artifacts=artifacts,
        output_dir=settings.reporting.output_dir,
        run_name=f"eda_{settings.reporting.run_name}",
    )


def write_eda_artifacts(
    artifacts: EdaArtifacts,
    output_dir: str | Path,
    run_name: str,
) -> EdaArtifacts:
    """Write the summary tables and the figures into a fresh run directory."""
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

    render_eda_plots(
        panel=artifacts.panel,
        weekday_summary=artifacts.weekday_summary,
        series_summary=artifacts.series_summary,
        output_dir=run_dir,
    )

    artifacts.run_directory = run_dir
    return artifacts
