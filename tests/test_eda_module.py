from __future__ import annotations

from pathlib import Path

import pytest

from retail_forecasting.eda.profiling import (
    build_correlation_summary,
    build_dataset_summary,
    build_missingness_summary,
    build_numeric_summary,
)
from retail_forecasting.eda.series import build_series_summary
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

    assert dataset_summary.loc[0, "rows"] == len(panel)
    assert dataset_summary.loc[0, "unique_series"] == panel["series_id"].nunique()
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


def test_category_heatmaps_land_in_the_run_directory(tmp_path: Path) -> None:
    """The three figures chapter 3 cites have to come out of the EDA run itself.

    They used to be produced by running a script by hand, so nothing tied them to the panel
    the rest of the report describes and they could go stale in silence.
    """
    from retail_forecasting.eda.temporal import render_category_seasonality_heatmaps

    # Enough rows per category to clear the minimum, which is what the real panel has.
    panel = make_synthetic_panel(num_series=12, num_days=120)
    render_category_seasonality_heatmaps(panel, tmp_path, min_observations=50)

    written = sorted(path.name for path in tmp_path.glob("category_seasonality_*.png"))
    assert written == [
        "category_seasonality_high.png",
        "category_seasonality_low.png",
        "category_seasonality_medium.png",
    ]


def test_category_heatmaps_are_skipped_rather_than_invented_on_a_thin_panel(
    tmp_path: Path,
) -> None:
    """No category with enough observations means no figure, not a confident one from noise."""
    from retail_forecasting.eda.temporal import render_category_seasonality_heatmaps

    panel = make_synthetic_panel(num_series=2, num_days=30)
    render_category_seasonality_heatmaps(panel, tmp_path, min_observations=10_000)

    assert list(tmp_path.glob("category_seasonality_*.png")) == []


def test_category_heatmaps_survive_a_panel_with_no_category_column(tmp_path: Path) -> None:
    """`third_category_id` is a static id column, not part of the canonical panel contract."""
    from retail_forecasting.eda.temporal import render_category_seasonality_heatmaps

    panel = make_synthetic_panel(num_series=4, num_days=90).drop(columns=["third_category_id"])
    render_category_seasonality_heatmaps(panel, tmp_path, min_observations=1)

    assert list(tmp_path.glob("category_seasonality_*.png")) == []


def _eda_settings(tmp_path: Path, top_n_series: int | None) -> object:
    """Settings for an EDA run whose panel comes from a stub, not from disk."""
    from retail_forecasting.config import ReportingConfig, Settings

    settings = Settings()
    return settings.model_copy(
        update={
            "dataset": settings.dataset.model_copy(
                update={"top_n_series": top_n_series, "min_history_days": 1}
            ),
            "reporting": ReportingConfig(output_dir=tmp_path, run_name="eda_test"),
        }
    )


def test_run_eda_honours_the_series_count_the_config_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`top_n_series` used to be forced to None here, so the config could not choose."""
    from retail_forecasting.eda import pipeline

    panel = make_synthetic_panel(num_series=3, num_days=90)
    seen: list[int | None] = []

    def stub_loader(*, dataset_config, preprocessing_config, split):  # type: ignore[no-untyped-def]
        seen.append(dataset_config.top_n_series)
        return panel

    monkeypatch.setattr(pipeline, "load_prepared_panel", stub_loader)
    pipeline.run_eda(_eda_settings(tmp_path, top_n_series=3), split="train")

    assert seen == [3], "run_eda debe pasar la config tal cual, sin forzar top_n_series"


def test_run_eda_still_describes_the_whole_panel_when_the_config_says_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The EDA config declares null, so the default behaviour is unchanged by the switch."""
    from retail_forecasting.eda import pipeline

    seen: list[int | None] = []

    def stub_loader(*, dataset_config, preprocessing_config, split):  # type: ignore[no-untyped-def]
        seen.append(dataset_config.top_n_series)
        return make_synthetic_panel(num_series=3, num_days=90)

    monkeypatch.setattr(pipeline, "load_prepared_panel", stub_loader)
    pipeline.run_eda(_eda_settings(tmp_path, top_n_series=None), split="train")

    assert seen == [None]


def _run_eda_on(panel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Drive a full EDA run over `panel`, with the loader stubbed out."""
    from retail_forecasting.eda import pipeline

    monkeypatch.setattr(
        pipeline,
        "load_prepared_panel",
        lambda **_: panel,
    )
    return pipeline.run_eda(_eda_settings(tmp_path, top_n_series=None), split="train")


def test_a_run_writes_every_summary_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The eleven CSVs are what the dashboard reads, so a missing one is a broken view."""
    run_dir = _run_eda_on(make_synthetic_panel(num_series=2, num_days=80), tmp_path, monkeypatch)

    written = {path.name for path in run_dir.glob("*.csv")}
    assert written == {
        "correlation_summary.csv",
        "dataset_summary.csv",
        "missingness_summary.csv",
        "numeric_summary.csv",
        "series_gap_summary.csv",
        "series_summary.csv",
        "stockout_by_series_summary.csv",
        "stockout_demand_bands.csv",
        "stockout_summary.csv",
        "temporal_summary.csv",
        "weekday_summary.csv",
    }


def test_a_run_writes_its_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Figures are the point of the mode and there is no flag to turn them off."""
    run_dir = _run_eda_on(make_synthetic_panel(num_series=3, num_days=90), tmp_path, monkeypatch)

    drawn = {path.name for path in run_dir.glob("*.png")}
    assert "observed_demand_distribution.png" in drawn
    assert "weekday_demand_profile.png" in drawn
    assert "acf_demand.png" in drawn
    assert len(drawn) >= 12


def test_a_run_records_what_it_analysed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory of figures with no record of its commit is not citable under docs/runs.md.

    Every other run mode leaves one; the EDA left none, and chapter 3 cites numbers out of its
    figures.
    """
    import json

    panel = make_synthetic_panel(num_series=3, num_days=90)
    run_dir = _run_eda_on(panel, tmp_path, monkeypatch)

    metadata = json.loads((run_dir / "eda_metadata.json").read_text(encoding="utf-8"))

    assert metadata["split"] == "train"
    assert metadata["n_series"] == 3
    assert metadata["rows"] == len(panel)
    # The parquet the run actually read: the cache key covers four dataset fields only, not the
    # preprocessing config nor the code version, so the panel on disk can predate the commit.
    assert metadata["panel_source"].endswith(".parquet")
    assert "git_commit" in metadata and "created_at" in metadata
    # What the config asked for sits beside what was measured, which is the pair that says
    # whether the panel on disk is the panel the config wanted.
    assert metadata["configured_top_n_series"] is None
    assert metadata["imputation_strategy"] == "supervised"
