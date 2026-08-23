from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_forecasting.config import Settings
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.eda.plots import render_eda_plots
from retail_forecasting.eda.profiling import (
    build_dataset_summary,
    build_missingness_summary,
    build_numeric_summary,
    build_series_summary,
)
from retail_forecasting.eda.reporting import EdaArtifacts, write_eda_artifacts
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


def run_eda(settings: Settings, split: str = "train") -> EdaArtifacts:
    """Run EDA on the canonical prepared panel and persist artifacts.

    The dataset config is honoured as written, `top_n_series` and `max_rows` included, so the
    mode can describe either the whole panel or the exact subset a model trains on. Those two
    used to be forced to None here, which cost more than it bought: the alignment guard below
    exists to catch a stale processed cache, and it can only compare the panel against a
    configured series count, so forcing that count to None left half of it unable to fire.
    """
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
        make_plots=settings.reporting.make_plots,
        render_plots=render_eda_plots,
        memoria_dir=Path("memoria"),
    )


def build_correlation_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute numeric correlations against observed demand."""
    numeric_columns = panel.select_dtypes(include=["number"]).columns.tolist()
    if "observed_demand" not in numeric_columns:
        return pd.DataFrame(columns=["feature_name", "correlation_with_observed_demand"])

    correlations = panel.loc[:, numeric_columns].corr(numeric_only=True)["observed_demand"]
    correlation_summary = (
        correlations.drop(labels=["observed_demand"])
        .rename("correlation_with_observed_demand")
        .reset_index()
        .rename(columns={"index": "feature_name"})
    )
    correlation_summary["absolute_correlation"] = correlation_summary[
        "correlation_with_observed_demand"
    ].abs()
    return correlation_summary.sort_values(
        ["absolute_correlation", "feature_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
