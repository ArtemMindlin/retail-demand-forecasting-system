from __future__ import annotations

from pathlib import Path

import pytest

from retail_forecasting.config import DatasetConfig
from retail_forecasting.eda.profiling import (
    build_dataset_summary,
    build_missingness_summary,
    build_numeric_summary,
    build_series_summary,
)
from retail_forecasting.eda.reporting import EdaArtifacts, write_eda_artifacts
from retail_forecasting.eda.run import (
    build_config_alignment_summary,
    build_correlation_summary,
    raise_on_alignment_warnings,
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
from tests import make_synthetic_panel


def test_eda_summaries_cover_prepared_panel_contract() -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)

    dataset_summary = build_dataset_summary(panel)
    missingness_summary = build_missingness_summary(panel)
    numeric_summary = build_numeric_summary(panel)
    series_summary = build_series_summary(panel)
    temporal_summary = build_temporal_summary(panel)
    weekday_summary = build_weekday_summary(panel)
    series_gap_summary = build_series_gap_summary(panel)
    stockout_summary = build_stockout_summary(panel)
    stockout_by_series_summary = build_stockout_by_series_summary(panel)
    stockout_demand_bands = build_stockout_demand_bands(panel)
    correlation_summary = build_correlation_summary(panel)
    config_alignment_summary, warnings = build_config_alignment_summary(
        panel=panel,
        dataset_config=DatasetConfig(top_n_series=3, min_history_days=70),
    )

    assert dataset_summary.loc[0, "rows"] == len(panel)
    assert dataset_summary.loc[0, "unique_series"] == panel["series_id"].nunique()
    assert config_alignment_summary.loc[0, "top_n_series_matches_config"]
    assert config_alignment_summary.loc[0, "min_history_matches_config"]
    assert warnings == []
    assert "series_id" in set(missingness_summary["column_name"])
    assert set(["column_name", "mean", "median"]).issubset(numeric_summary.columns)
    assert series_summary["series_id"].nunique() == panel["series_id"].nunique()
    assert temporal_summary.loc[0, "duplicate_series_date_rows"] == 0
    assert list(weekday_summary["weekday"]) == sorted(weekday_summary["weekday"].tolist())
    assert (series_gap_summary["missing_days_within_span"] == 0).all()
    assert 0.0 <= stockout_summary.loc[0, "stockout_row_rate"] <= 1.0
    assert stockout_by_series_summary["series_id"].nunique() == panel["series_id"].nunique()
    assert set(stockout_demand_bands["stockout_band"].astype(str)) == {
        "0",
        "0-2",
        "3-6",
        "7+",
    }
    assert "absolute_correlation" in correlation_summary.columns


def test_eda_artifacts_are_written_as_expected(tmp_path: Path) -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)

    artifacts = EdaArtifacts(
        panel=panel,
        dataset_summary=build_dataset_summary(panel),
        config_alignment_summary=build_config_alignment_summary(
            panel=panel,
            dataset_config=DatasetConfig(top_n_series=2, min_history_days=70),
        )[0],
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
        warnings=[],
    )

    written = write_eda_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
        run_name="eda_test",
        make_plots=False,
    )

    assert written.run_directory is not None
    assert (written.run_directory / "eda_report.md").exists()
    assert (written.run_directory / "dataset_summary.csv").exists()
    assert (written.run_directory / "config_alignment_summary.csv").exists()
    assert (written.run_directory / "stockout_summary.csv").exists()
    assert (written.run_directory / "correlation_summary.csv").exists()


def test_eda_plots_are_written_as_expected(tmp_path: Path) -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)

    artifacts = EdaArtifacts(
        panel=panel,
        dataset_summary=build_dataset_summary(panel),
        config_alignment_summary=build_config_alignment_summary(
            panel=panel,
            dataset_config=DatasetConfig(top_n_series=3, min_history_days=70),
        )[0],
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
        warnings=[],
    )

    from retail_forecasting.eda.plots import render_eda_plots

    written = write_eda_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
        run_name="eda_plot_test",
        make_plots=True,
        render_plots=render_eda_plots,
    )

    assert written.run_directory is not None
    assert (written.run_directory / "coverage_heatmap.png").exists()
    assert (written.run_directory / "observed_demand_distribution.png").exists()
    assert (written.run_directory / "observed_demand_boxplot_top_series.png").exists()
    assert (written.run_directory / "stockout_hours_distribution.png").exists()
    assert (written.run_directory / "weekday_demand_profile.png").exists()
    assert (written.run_directory / "zero_demand_rate_by_series.png").exists()
    assert (written.run_directory / "stockout_band_demand.png").exists()
    assert (written.run_directory / "stockout_vs_demand_scatter.png").exists()
    assert (written.run_directory / "correlation_heatmap.png").exists()
    assert (written.run_directory / "covariate_vs_demand_grid.png").exists()
    assert (written.run_directory / "representative_series_panels.png").exists()


def test_eda_exports_selected_figures_to_memoria(tmp_path: Path) -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)
    memoria_dir = tmp_path / "memoria"

    artifacts = EdaArtifacts(
        panel=panel,
        dataset_summary=build_dataset_summary(panel),
        config_alignment_summary=build_config_alignment_summary(
            panel=panel,
            dataset_config=DatasetConfig(top_n_series=3, min_history_days=70),
        )[0],
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
        warnings=[],
    )

    from retail_forecasting.eda.plots import render_eda_plots

    write_eda_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
        run_name="eda_memoria_test",
        make_plots=True,
        render_plots=render_eda_plots,
        memoria_dir=memoria_dir,
    )

    tex_fragment = memoria_dir / "figures" / "eda" / "eda_figures.tex"
    assert tex_fragment.exists()
    assert (memoria_dir / "figures" / "eda" / "coverage_heatmap.png").exists()
    assert (memoria_dir / "figures" / "eda" / "representative_series_panels.png").exists()
    assert "Interpretación." in tex_fragment.read_text(encoding="utf-8")


def test_eda_alignment_flags_stale_processed_panel_shape() -> None:
    panel = make_synthetic_panel(num_series=4, num_days=90)

    summary, warnings = build_config_alignment_summary(
        panel=panel,
        dataset_config=DatasetConfig(top_n_series=2, min_history_days=70),
    )

    assert not summary.loc[0, "top_n_series_matches_config"]
    assert summary.loc[0, "min_history_matches_config"]
    assert warnings != []
    assert "stale processed cache" in warnings[0]


def test_eda_alignment_raises_on_stale_processed_panel_shape() -> None:
    panel = make_synthetic_panel(num_series=4, num_days=90)
    _, warnings = build_config_alignment_summary(
        panel=panel,
        dataset_config=DatasetConfig(top_n_series=2, min_history_days=70),
    )

    with pytest.raises(RuntimeError, match="EDA aborted"):
        raise_on_alignment_warnings(warnings)
