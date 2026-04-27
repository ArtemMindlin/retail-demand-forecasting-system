from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

import pandas as pd

from retail_forecasting.config import Settings, settings_to_dict
from retail_forecasting.utils.io import dataframe_to_markdown, ensure_directory, make_run_directory
from retail_forecasting.visualization.plots import render_standard_plots


@dataclass
class RunArtifacts:
    prepared_panel: pd.DataFrame
    supervised_frame: pd.DataFrame
    predictions: pd.DataFrame
    metrics_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    cost_summary: pd.DataFrame
    sensitivity_summary: Optional[pd.DataFrame] = None
    drifts: list[dict[str, Any]] = field(default_factory=list)
    run_directory: Path | None = None


def write_run_artifacts(artifacts: RunArtifacts, settings: Settings) -> RunArtifacts:
    run_dir = make_run_directory(settings.reporting.output_dir, settings.reporting.run_name)
    ensure_directory(run_dir)

    artifacts.predictions.to_csv(run_dir / "predictions.csv", index=False)
    artifacts.metrics_summary.to_csv(run_dir / "metrics_summary.csv", index=False)
    artifacts.fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    artifacts.cost_summary.to_csv(run_dir / "cost_summary.csv", index=False)
    
    if artifacts.sensitivity_summary is not None:
        artifacts.sensitivity_summary.to_csv(run_dir / "sensitivity_summary.csv", index=False)

    if settings.reporting.make_plots:
        render_standard_plots(
            metrics_summary=artifacts.metrics_summary,
            cost_summary=artifacts.cost_summary,
            output_dir=run_dir,
        )

    report_text = build_markdown_report(artifacts=artifacts, settings=settings)
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")

    artifacts.run_directory = run_dir
    return artifacts


def build_markdown_report(artifacts: RunArtifacts, settings: Settings) -> str:
    serializable_settings = settings_to_dict(settings)
    settings_lines = [
        f"- `{section}`: `{values}`"
        for section, values in serializable_settings.items()
    ]

    report = [
        "# Experiment Report",
        "",
        "## Executive Summary",
        "",
        "This report compares forecasting systems under predictive, probabilistic, and economic criteria. "
        "The primary ranking uses total operating cost under a single-period newsvendor policy.",
        "",
        "## Configuration",
        "",
        *settings_lines,
        "",
        "## Dataset Summary",
        "",
        f"- Rows in prepared panel: `{len(artifacts.prepared_panel)}`",
        f"- Rows in supervised frame: `{len(artifacts.supervised_frame)}`",
        f"- Unique series: `{artifacts.prepared_panel['series_id'].nunique()}`",
        f"- Date range: `{artifacts.prepared_panel['date'].min().date()}` to `{artifacts.prepared_panel['date'].max().date()}`",
        "",
        "## Metrics Summary",
        "",
        dataframe_to_markdown(artifacts.metrics_summary),
        "",
        "## Cost Summary",
        "",
        dataframe_to_markdown(artifacts.cost_summary),
        "",
        "## Fold Diagnostics",
        "",
        dataframe_to_markdown(artifacts.fold_metrics),
        "",
        "## Drift Analysis",
        "",
        *(
            [f"- **ALERT**: Detected drift on `{d['date']}` (Score: `{d['score']:.2f}`, Threshold: `{d['threshold']:.2f}`)" for d in artifacts.drifts]
            if artifacts.drifts else ["- No statistically significant drift detected during this run."]
        ),
        "",
        "## Economic Sensitivity Analysis",
        "",
        "Performance under varying stockout/overstock cost ratios (Cs/Co):",
        "",
        dataframe_to_markdown(artifacts.sensitivity_summary) if artifacts.sensitivity_summary is not None else "_No sensitivity analysis available._",
        "",
        "## Interpretation Notes",
        "",
        "- `MAE` and `RMSE` are included as diagnostics, not as the primary decision criterion.",
        "- `pinball_*` columns evaluate quantile quality when quantile forecasts are available.",
        "- `total_cost` is the main ranking metric because the TFG focuses on inventory decisions.",
    ]

    return "\n".join(report)
