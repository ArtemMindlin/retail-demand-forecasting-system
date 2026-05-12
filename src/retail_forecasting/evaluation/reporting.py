from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from retail_forecasting.config import Settings
from retail_forecasting.contracts.backtesting import FoldRunMetadata
from retail_forecasting.contracts.drift import DriftDetectorMetadata, DriftEvent
from retail_forecasting.contracts.tuning import TuningMetadata
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


class PromotionDecisionMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    champion_data_strategy: str | None
    champion_model_name: str
    champion_backend_name: str
    challenger_data_strategy: str | None
    challenger_model_name: str
    challenger_backend_name: str
    champion_total_cost: float = Field(ge=0.0)
    challenger_total_cost: float = Field(ge=0.0)
    cost_improvement_pct: float
    champion_service_level: float = Field(ge=0.0, le=1.0)
    challenger_service_level: float = Field(ge=0.0, le=1.0)
    service_level_delta: float
    min_cost_improvement_pct: float = Field(ge=0.0)
    max_service_level_degradation: float = Field(ge=0.0, le=1.0)
    promote: bool
    decision_reason: str


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
    tuning: TuningMetadata | None = None
    drift: DriftDetectorMetadata
    promotion: PromotionDecisionMetadata | None = None


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
    reorder_recommendations: pd.DataFrame | None = None
    exceptions: pd.DataFrame | None = None
    promotion_decision: PromotionDecisionMetadata | None = None
    drifts: list[DriftEvent] = field(default_factory=list)
    report_extra: str = ""
    backtest_metadata: BacktestMetadata | None = None
    run_directory: Path | None = None


def write_run_artifacts(artifacts: RunArtifacts, settings: Settings) -> RunArtifacts:
    run_dir = make_run_directory(
        settings.reporting.output_dir, settings.reporting.run_name
    )
    ensure_directory(run_dir)

    reorder_recommendations = build_reorder_recommendations(artifacts, settings)
    exceptions = build_exceptions_frame(reorder_recommendations)
    promotion_decision = build_promotion_decision(artifacts, settings)
    if artifacts.backtest_metadata is not None:
        artifacts.backtest_metadata = artifacts.backtest_metadata.model_copy(
            update={"promotion": promotion_decision}
        )

    artifacts.predictions.to_csv(run_dir / "predictions.csv", index=False)
    artifacts.metrics_summary.to_csv(run_dir / "metrics_summary.csv", index=False)
    artifacts.fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    artifacts.cost_summary.to_csv(run_dir / "cost_summary.csv", index=False)
    reorder_recommendations.to_csv(run_dir / "reorder_recommendations.csv", index=False)
    exceptions.to_csv(run_dir / "exceptions.csv", index=False)
    if promotion_decision is not None:
        (run_dir / "promotion_decision.json").write_text(
            promotion_decision.model_dump_json(indent=2),
            encoding="utf-8",
        )

    if artifacts.sensitivity_summary is not None:
        artifacts.sensitivity_summary.to_csv(
            run_dir / "sensitivity_summary.csv", index=False
        )
    if artifacts.pareto_frontier is not None:
        artifacts.pareto_frontier.to_csv(run_dir / "pareto_frontier.csv", index=False)
    if settings.reporting.make_plots:
        render_standard_plots(
            metrics_summary=artifacts.metrics_summary,
            cost_summary=artifacts.cost_summary,
            output_dir=run_dir,
        )

    report_text = build_markdown_report(artifacts=artifacts, settings=settings)
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")

    artifacts.reorder_recommendations = reorder_recommendations
    artifacts.exceptions = exceptions
    artifacts.promotion_decision = promotion_decision
    if artifacts.backtest_metadata is not None:
        (run_dir / "backtest_metadata.json").write_text(
            artifacts.backtest_metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
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
                f"- **ALERT**: Detected drift on `{event.date}` (Score: `{event.score:.2f}`, Threshold: `{event.threshold:.2f}`)"
                for event in artifacts.drifts
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


def build_reorder_recommendations(
    artifacts: RunArtifacts, settings: Settings
) -> pd.DataFrame:
    recommendations = artifacts.predictions.copy()
    recommendations["decision_date"] = recommendations["date"]
    recommendations["predicted_lead_time_demand"] = recommendations["y_pred"]

    series_parts = (
        recommendations["series_id"].astype(str).str.split("_", n=1, expand=True)
    )
    recommendations["store_id"] = series_parts[0]
    recommendations["product_id"] = series_parts[1]

    if "prediction_source" not in recommendations.columns:
        recommendations["prediction_source"] = "model"
    if "fallback_level" not in recommendations.columns:
        recommendations["fallback_level"] = pd.Series(
            pd.NA, index=recommendations.index, dtype="object"
        )

    recommendations["risk_flag"] = pd.Series(
        pd.NA, index=recommendations.index, dtype="object"
    )
    recommendations["notes"] = pd.Series(
        pd.NA, index=recommendations.index, dtype="object"
    )

    if settings.business.flag_cold_start:
        cold_start_mask = recommendations["prediction_source"] == "cold_start_fallback"
        recommendations.loc[cold_start_mask, "risk_flag"] = "cold_start"
        recommendations.loc[cold_start_mask, "notes"] = (
            "Fallback recommendation generated due to incomplete model features."
        )

    if settings.business.flag_drift_watch and artifacts.drifts:
        drift_fold_ids = {event.fold_id for event in artifacts.drifts}
        drift_mask = (
            recommendations["fold_id"].isin(drift_fold_ids)
            & recommendations["risk_flag"].isna()
        )
        recommendations.loc[drift_mask, "risk_flag"] = "drift_watch"
        recommendations.loc[drift_mask, "notes"] = (
            "Recommendation belongs to a fold flagged for drift monitoring."
        )

    lower_quantile = "q_0_1"
    upper_quantile = "q_0_9"
    if (
        settings.business.flag_high_uncertainty
        and lower_quantile in recommendations.columns
        and upper_quantile in recommendations.columns
    ):
        interval_width = (
            recommendations[upper_quantile] - recommendations[lower_quantile]
        )
        positive_width = interval_width[interval_width > 0]
        if not positive_width.empty:
            width_threshold = float(
                positive_width.quantile(
                    settings.business.high_uncertainty_interval_quantile
                )
            )
            high_uncertainty_mask = (
                interval_width >= width_threshold
            ) & recommendations["risk_flag"].isna()
            recommendations.loc[high_uncertainty_mask, "risk_flag"] = "high_uncertainty"
            recommendations.loc[high_uncertainty_mask, "notes"] = (
                "Prediction interval width exceeds the configured review threshold."
            )

    extreme_order_threshold = float(
        recommendations["order_quantity"].quantile(
            settings.business.extreme_order_quantity_quantile
        )
    )
    if settings.business.flag_extreme_order_quantity and extreme_order_threshold > 0:
        extreme_order_mask = (
            recommendations["order_quantity"] >= extreme_order_threshold
        ) & recommendations["risk_flag"].isna()
        recommendations.loc[extreme_order_mask, "risk_flag"] = "extreme_order_quantity"
        recommendations.loc[extreme_order_mask, "notes"] = (
            "Order quantity exceeds the configured review threshold."
        )

    preferred_columns = [
        "decision_date",
        "series_id",
        "store_id",
        "product_id",
        "predicted_lead_time_demand",
        "order_quantity",
        "prediction_source",
        "fallback_level",
        "risk_flag",
        "notes",
        "data_strategy",
        "model_name",
        "backend_name",
        "fold_id",
        "stockout_hours",
        "stockout_regime",
        lower_quantile,
        "q_0_5",
        upper_quantile,
    ]
    output_columns = [
        column for column in preferred_columns if column in recommendations.columns
    ]
    return recommendations.loc[:, output_columns]


def build_exceptions_frame(recommendations: pd.DataFrame) -> pd.DataFrame:
    exceptions = recommendations.loc[recommendations["risk_flag"].notna()].copy()
    preferred_columns = [
        "decision_date",
        "series_id",
        "store_id",
        "product_id",
        "risk_flag",
        "order_quantity",
        "prediction_source",
        "fallback_level",
        "notes",
        "predicted_lead_time_demand",
        "model_name",
        "backend_name",
        "fold_id",
        "q_0_1",
        "q_0_5",
        "q_0_9",
    ]
    output_columns = [
        column for column in preferred_columns if column in exceptions.columns
    ]
    return exceptions.loc[:, output_columns]


def build_promotion_decision(
    artifacts: RunArtifacts, settings: Settings
) -> PromotionDecisionMetadata | None:
    cost_summary = artifacts.cost_summary
    if cost_summary.empty:
        return None

    summary = cost_summary.copy()
    required_columns = {
        "model_name",
        "backend_name",
        "total_cost",
        "service_level",
    }
    if not required_columns.issubset(summary.columns):
        return None

    champion_mask = (summary["model_name"] == settings.business.champion_model_name) & (
        summary["backend_name"] == settings.business.champion_backend_name
    )
    if (
        "data_strategy" in summary.columns
        and settings.business.champion_data_strategy is not None
    ):
        champion_mask &= (
            summary["data_strategy"] == settings.business.champion_data_strategy
        )

    champion_rows = summary.loc[champion_mask].copy()
    if champion_rows.empty:
        return None

    champion = champion_rows.sort_values("total_cost").iloc[0]
    candidate_mask = (summary["model_name"] != champion["model_name"]) | (
        summary["backend_name"] != champion["backend_name"]
    )
    if "data_strategy" in summary.columns:
        candidate_mask |= summary["data_strategy"] != champion.get("data_strategy")
    challengers = summary.loc[candidate_mask].copy()
    if challengers.empty:
        return None

    champion_total_cost = float(champion["total_cost"])
    champion_service_level = float(champion["service_level"])
    challengers["cost_improvement_pct"] = (
        (champion_total_cost - challengers["total_cost"].astype(float))
        / champion_total_cost
        * 100.0
    )
    challengers["service_level_delta"] = (
        challengers["service_level"].astype(float) - champion_service_level
    )
    challengers["promote"] = (
        challengers["cost_improvement_pct"]
        >= settings.business.champion_min_cost_improvement_pct
    ) & (
        challengers["service_level_delta"]
        >= -settings.business.champion_max_service_level_degradation
    )

    promotable = challengers.loc[challengers["promote"]].copy()
    if not promotable.empty:
        selected = promotable.sort_values(
            ["total_cost", "service_level"],
            ascending=[True, False],
        ).iloc[0]
        decision_reason = "Challenger improves total cost and respects the configured service-level tolerance."
        promote = True
    else:
        selected = challengers.sort_values(
            ["total_cost", "service_level"],
            ascending=[True, False],
        ).iloc[0]
        decision_reason = "No challenger satisfied both the minimum cost improvement and service-level guardrail."
        promote = False

    return PromotionDecisionMetadata(
        champion_data_strategy=_optional_string(champion.get("data_strategy")),
        champion_model_name=str(champion["model_name"]),
        champion_backend_name=str(champion["backend_name"]),
        challenger_data_strategy=_optional_string(selected.get("data_strategy")),
        challenger_model_name=str(selected["model_name"]),
        challenger_backend_name=str(selected["backend_name"]),
        champion_total_cost=champion_total_cost,
        challenger_total_cost=float(selected["total_cost"]),
        cost_improvement_pct=float(selected["cost_improvement_pct"]),
        champion_service_level=champion_service_level,
        challenger_service_level=float(selected["service_level"]),
        service_level_delta=float(selected["service_level_delta"]),
        min_cost_improvement_pct=settings.business.champion_min_cost_improvement_pct,
        max_service_level_degradation=settings.business.champion_max_service_level_degradation,
        promote=promote,
        decision_reason=decision_reason,
    )


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
