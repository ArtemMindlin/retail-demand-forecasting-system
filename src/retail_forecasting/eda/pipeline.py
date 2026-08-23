from __future__ import annotations

from retail_forecasting.config import Settings
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.eda.profiling import (
    build_correlation_summary,
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
    mode describes either the whole panel or the exact subset a model trains on depending on
    what the config asks for. `configs/eda/default.yaml` sets both to null, which is what makes
    the default the whole panel.
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
        memoria_dir=settings.reporting.memoria_dir,
    )
