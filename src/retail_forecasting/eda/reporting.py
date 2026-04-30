from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from retail_forecasting.utils.io import (
    dataframe_to_markdown,
    ensure_directory,
    make_run_directory,
)


@dataclass
class EdaArtifacts:
    panel: pd.DataFrame
    dataset_summary: pd.DataFrame
    config_alignment_summary: pd.DataFrame
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
    warnings: list[str] = field(default_factory=list)
    run_directory: Path | None = None


def write_eda_artifacts(
    artifacts: EdaArtifacts,
    output_dir: str | Path,
    run_name: str,
    make_plots: bool,
    render_plots: callable | None = None,
) -> EdaArtifacts:
    """Persist EDA summaries, plots, and a Markdown report."""
    run_dir = make_run_directory(output_dir, run_name)
    ensure_directory(run_dir)

    outputs = {
        "dataset_summary.csv": artifacts.dataset_summary,
        "config_alignment_summary.csv": artifacts.config_alignment_summary,
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

    if make_plots and render_plots is not None:
        render_plots(
            panel=artifacts.panel,
            weekday_summary=artifacts.weekday_summary,
            series_summary=artifacts.series_summary,
            output_dir=run_dir,
        )

    report_text = build_eda_report(artifacts)
    (run_dir / "eda_report.md").write_text(report_text, encoding="utf-8")

    artifacts.run_directory = run_dir
    return artifacts


def build_eda_report(artifacts: EdaArtifacts) -> str:
    """Render the Markdown report for an EDA run."""
    report = [
        "# Exploratory Data Analysis Report",
        "",
        "## Dataset Summary",
        "",
        dataframe_to_markdown(artifacts.dataset_summary),
        "",
        "## Configuration Alignment",
        "",
        dataframe_to_markdown(artifacts.config_alignment_summary),
        "",
        "## Alerts",
        "",
        *(
            [f"- **ALERT**: {warning}" for warning in artifacts.warnings]
            if artifacts.warnings
            else ["- No configuration-alignment issues detected."]
        ),
        "",
        "## Temporal Coverage",
        "",
        dataframe_to_markdown(artifacts.temporal_summary),
        "",
        "## Missingness Summary",
        "",
        dataframe_to_markdown(artifacts.missingness_summary.head(12)),
        "",
        "## Weekly Seasonality",
        "",
        dataframe_to_markdown(artifacts.weekday_summary),
        "",
        "## Stockout Summary",
        "",
        dataframe_to_markdown(artifacts.stockout_summary),
        "",
        "## Demand by Stockout Band",
        "",
        dataframe_to_markdown(artifacts.stockout_demand_bands),
        "",
        "## Top Series by Observed Demand",
        "",
        dataframe_to_markdown(
            artifacts.series_summary.head(10),
            columns=[
                "series_id",
                "history_days",
                "observed_demand_sum",
                "observed_demand_mean",
                "stockout_day_rate",
            ],
        ),
        "",
        "## Correlation Summary",
        "",
        dataframe_to_markdown(artifacts.correlation_summary.head(12)),
        "",
        "## Interpretation Notes",
        "",
        "- The EDA is computed on the canonical prepared panel, not on raw column names.",
        "- Demand and stockout summaries are descriptive only and do not alter target semantics.",
        "- Weekly and stockout diagnostics are intended to guide feature and experiment design.",
    ]
    return "\n".join(report)
