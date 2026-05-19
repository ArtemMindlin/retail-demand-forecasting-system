from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retail_forecasting.config import DatasetConfig, Settings, load_config
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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for EDA execution."""
    parser = argparse.ArgumentParser(
        description="Run reproducible EDA on the prepared retail panel.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the EDA reporting output directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to analyze after panel preparation.",
    )
    return parser


def run_eda(settings: Settings, split: str = "train") -> EdaArtifacts:
    """Run EDA on the canonical prepared panel and persist artifacts."""
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split=split,
    )
    panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)
    config_alignment_summary, warnings = build_config_alignment_summary(
        panel=panel,
        dataset_config=settings.dataset,
    )
    raise_on_alignment_warnings(warnings)

    artifacts = EdaArtifacts(
        panel=panel,
        dataset_summary=build_dataset_summary(panel),
        config_alignment_summary=config_alignment_summary,
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
        warnings=warnings,
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


def build_config_alignment_summary(
    panel: pd.DataFrame,
    dataset_config: DatasetConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Check whether the loaded prepared panel matches key dataset settings."""
    actual_unique_series = int(panel["series_id"].nunique())
    history_lengths = panel.groupby("series_id")["date"].nunique()
    actual_min_history_days = int(history_lengths.min()) if not history_lengths.empty else 0
    expected_max_series = dataset_config.top_n_series

    if expected_max_series is None or expected_max_series == 0:
        top_n_matches = True
    else:
        top_n_matches = actual_unique_series <= expected_max_series
    min_history_matches = actual_min_history_days >= dataset_config.min_history_days

    summary = pd.DataFrame(
        [
            {
                "processed_panel_dir": str(dataset_config.processed_panel_dir),
                "use_cache": dataset_config.use_cache,
                "configured_top_n_series": expected_max_series,
                "actual_unique_series": actual_unique_series,
                "top_n_series_matches_config": top_n_matches,
                "configured_min_history_days": dataset_config.min_history_days,
                "actual_min_history_days": actual_min_history_days,
                "min_history_matches_config": min_history_matches,
            }
        ]
    )

    warnings: list[str] = []
    if not top_n_matches:
        warnings.append(
            "Loaded prepared panel contains more series than allowed by "
            f"`dataset.top_n_series` ({actual_unique_series} > {expected_max_series}). "
            "This usually means a stale processed cache was reused."
        )
    if not min_history_matches:
        warnings.append(
            "Loaded prepared panel violates `dataset.min_history_days` "
            f"({actual_min_history_days} < {dataset_config.min_history_days})."
        )

    return summary, warnings


def raise_on_alignment_warnings(warnings: list[str]) -> None:
    """Fail fast when the prepared panel does not match critical config."""
    if not warnings:
        return

    joined_warnings = " ".join(warnings)
    raise RuntimeError(
        "EDA aborted because the loaded prepared panel does not match the active dataset "
        f"configuration. {joined_warnings}"
    )


def main() -> None:
    """Parse CLI arguments and execute the configured EDA run."""
    args = build_parser().parse_args()
    settings = load_config(args.config)
    if args.output_dir is not None:
        settings.reporting.output_dir = Path(args.output_dir)

    artifacts = run_eda(settings=settings, split=args.split)
    assert artifacts.run_directory is not None
    print(f"EDA report written to: {artifacts.run_directory / 'eda_report.md'}")


if __name__ == "__main__":
    main()
