from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from retail_forecasting.config import Settings
from retail_forecasting.utils.io import (
    dataframe_to_markdown,
    ensure_directory,
    make_run_directory,
)
from retail_forecasting.visualization.plots import render_standard_plots


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int = Field(ge=0)
    series: int = Field(ge=0)
    unique_dates: int = Field(ge=0)
    date_min: str
    date_max: str


class FeaturePipelineMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(gt=0)
    lags: list[int] = Field(min_length=1)
    rolling_windows: list[int] = Field(min_length=1)
    feature_columns: int = Field(ge=0)
    input_rows: int = Field(ge=0)
    supervised_rows: int = Field(ge=0)
    dropped_rows_missing_target: int = Field(ge=0)
    dropped_rows_missing_features: int = Field(ge=0)


class FoldRunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fold_id: int = Field(ge=0)
    horizon: int = Field(gt=0)
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    train_rows: int = Field(ge=0)
    validation_rows: int = Field(ge=0)
    train_series: int = Field(ge=0)
    validation_series: int = Field(ge=0)


class ValidationMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_train_days: int = Field(gt=0)
    n_folds_requested: int = Field(gt=0)
    fold_size_days: int = Field(gt=0)
    folds_created: int = Field(ge=0)
    folds: list[FoldRunMetadata]


class ModelRunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    models_run: list[str] = Field(min_length=1)
    quantiles: list[float] = Field(min_length=1)
    optimize_for_cost: bool
    use_tuning: bool
    retrain_each_fold: bool


class TuningRunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str
    n_trials_requested: int = Field(gt=0)
    best_score: float | None = Field(default=None, ge=0)
    train_rows: int = Field(ge=0)
    validation_rows: int = Field(ge=0)
    validation_cutoff: str
    feature_count: int = Field(ge=0)
    target_col: str
    best_params: dict[str, int | float]


class BacktestMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name: str
    data_strategy: str
    created_at: str
    git_commit: str | None
    config_hash: str
    dataset: DatasetMetadata
    features: FeaturePipelineMetadata
    validation: ValidationMetadata
    models: ModelRunMetadata
    tuning: TuningRunMetadata | None = None


@dataclass
class RunArtifacts:
    prepared_panel: pd.DataFrame
    supervised_frame: pd.DataFrame
    predictions: pd.DataFrame
    metrics_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    cost_summary: pd.DataFrame
    sensitivity_summary: Optional[pd.DataFrame] = None
    pareto_frontier: Optional[pd.DataFrame] = None
    drifts: list[dict[str, Any]] = field(default_factory=list)
    report_extra: str = ""
    backtest_metadata: BacktestMetadata | None = None
    run_directory: Path | None = None


def write_run_artifacts(artifacts: RunArtifacts, settings: Settings) -> RunArtifacts:
    run_dir = make_run_directory(
        settings.reporting.output_dir, settings.reporting.run_name
    )
    ensure_directory(run_dir)

    artifacts.predictions.to_csv(run_dir / "predictions.csv", index=False)
    artifacts.metrics_summary.to_csv(run_dir / "metrics_summary.csv", index=False)
    artifacts.fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    artifacts.cost_summary.to_csv(run_dir / "cost_summary.csv", index=False)

    if artifacts.sensitivity_summary is not None:
        artifacts.sensitivity_summary.to_csv(
            run_dir / "sensitivity_summary.csv", index=False
        )
    if artifacts.pareto_frontier is not None:
        artifacts.pareto_frontier.to_csv(run_dir / "pareto_frontier.csv", index=False)
    if artifacts.backtest_metadata is not None:
        (run_dir / "backtest_metadata.json").write_text(
            artifacts.backtest_metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )

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


def build_config_hash(settings: Settings) -> str:
    serialized = json.dumps(settings.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def build_markdown_report(artifacts: RunArtifacts, settings: Settings) -> str:
    serializable_settings = settings.model_dump()
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
            [
                f"- **ALERT**: Detected drift on `{d['date']}` (Score: `{d['score']:.2f}`, Threshold: `{d['threshold']:.2f}`)"
                for d in artifacts.drifts
            ]
            if artifacts.drifts
            else ["- No statistically significant drift detected during this run."]
        ),
        "",
        "## Economic Sensitivity Analysis",
        "",
        "Performance under varying stockout/overstock cost ratios (Cs/Co):",
        "",
        dataframe_to_markdown(artifacts.sensitivity_summary)
        if artifacts.sensitivity_summary is not None
        else "_No sensitivity analysis available._",
        "",
        "## Pareto Frontier",
        "",
        "Candidate inventory policies are generated by scaling each model's selected order quantity. "
        "Pareto-efficient rows are not dominated simultaneously on economic cost, overstock units, and stockout units.",
        "",
        dataframe_to_markdown(artifacts.pareto_frontier)
        if artifacts.pareto_frontier is not None
        else "_No Pareto frontier available._",
        "",
        "## Interpretation Notes",
        "",
        "- `MAE` and `RMSE` are included as diagnostics, not as the primary decision criterion.",
        "- `pinball_*` columns evaluate quantile quality when quantile forecasts are available.",
        "- `total_cost` is the main ranking metric because the TFG focuses on inventory decisions.",
    ]

    return "\n".join(report)
