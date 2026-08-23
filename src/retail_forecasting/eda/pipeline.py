from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retail_forecasting.config import Settings, build_config_hash
from retail_forecasting.contracts.contracts_quality import EdaRunMetadata
from retail_forecasting.data.dataset import load_prepared_panel, panel_cache_filename
from retail_forecasting.eda.profiling import (
    build_correlation_summary,
    build_dataset_summary,
    build_missingness_summary,
    build_numeric_summary,
    render_profiling_figures,
)
from retail_forecasting.eda.series import build_series_summary, render_series_figures
from retail_forecasting.eda.stockout import (
    build_stockout_by_series_summary,
    build_stockout_demand_bands,
    build_stockout_summary,
    render_stockout_figures,
)
from retail_forecasting.eda.temporal import (
    build_series_gap_summary,
    build_temporal_summary,
    build_weekday_summary,
    render_temporal_figures,
)
from retail_forecasting.utils.io import make_run_directory
from retail_forecasting.utils.provenance import get_git_commit, utc_timestamp


def build_run_metadata(
    settings: Settings, panel: pd.DataFrame, split: str, config_path: Path | None
) -> EdaRunMetadata:
    """Describe the run: what it read, what the config asked for, and from which commit."""
    return EdaRunMetadata(
        split=split,
        panel_source=str(
            settings.dataset.processed_panel_dir / panel_cache_filename(settings.dataset, split)
        ),
        n_series=int(panel["series_id"].nunique()),
        rows=len(panel),
        date_min=str(panel["date"].min().date()),
        date_max=str(panel["date"].max().date()),
        configured_top_n_series=settings.dataset.top_n_series,
        configured_min_history_days=settings.dataset.min_history_days,
        configured_max_rows=settings.dataset.max_rows,
        imputation_strategy=settings.preprocessing.imputation_strategy,
        drop_negative_sales=settings.preprocessing.drop_negative_sales,
        fill_missing_values=settings.preprocessing.fill_missing_values,
        config_hash=build_config_hash(settings),
        config_path=None if config_path is None else str(config_path),
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
    )


def run_eda(settings: Settings, split: str = "train", config_path: Path | None = None) -> Path:
    """Read the panel, summarise it, draw it, and return the directory it all landed in."""
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split=split,
    )
    run_dir = make_run_directory(
        settings.reporting.output_dir, f"eda_{settings.reporting.run_name}"
    )

    series_summary = build_series_summary(panel)
    weekday_summary = build_weekday_summary(panel)
    stockout_demand_bands = build_stockout_demand_bands(panel)

    summaries = {
        "dataset_summary.csv": build_dataset_summary(panel),
        "missingness_summary.csv": build_missingness_summary(panel),
        "numeric_summary.csv": build_numeric_summary(panel),
        "series_summary.csv": series_summary,
        "temporal_summary.csv": build_temporal_summary(panel),
        "weekday_summary.csv": weekday_summary,
        "series_gap_summary.csv": build_series_gap_summary(panel),
        "stockout_summary.csv": build_stockout_summary(panel),
        "stockout_by_series_summary.csv": build_stockout_by_series_summary(panel),
        "stockout_demand_bands.csv": stockout_demand_bands,
        "correlation_summary.csv": build_correlation_summary(panel),
    }
    for filename, frame in summaries.items():
        frame.to_csv(run_dir / filename, index=False)

    # Each module draws its own figures from its own summaries, so a figure and the table
    # beside it in the run directory come from one calculation rather than two.
    render_profiling_figures(panel, run_dir)
    render_series_figures(panel, series_summary, run_dir)
    render_temporal_figures(panel, weekday_summary, series_summary, run_dir)
    render_stockout_figures(panel, stockout_demand_bands, run_dir)

    metadata = build_run_metadata(settings, panel, split, config_path)
    (run_dir / "eda_metadata.json").write_text(
        json.dumps(metadata.model_dump(), indent=2), encoding="utf-8"
    )
    return run_dir
