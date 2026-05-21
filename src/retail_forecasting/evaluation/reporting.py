from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import shap
from pydantic import BaseModel, ConfigDict, Field

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_backtesting import FoldRunMetadata
from retail_forecasting.contracts.contracts_business import ChampionRecord, ChampionRegistry
from retail_forecasting.contracts.contracts_drift import DriftDetectorMetadata, DriftEvent
from retail_forecasting.contracts.contracts_quality import DataQualityReport
from retail_forecasting.contracts.contracts_tuning import TuningMetadata
from retail_forecasting.utils.io import (
    dataframe_to_markdown,
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

    champion_source: Literal["config", "registry"]
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


class OperationalRunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name: str
    run_mode: Literal["experiment", "retrain", "score_daily", "simulate_ops"]
    created_at: str
    git_commit: str | None
    config_hash: str
    recommendation_rows: int = Field(ge=0)
    exception_rows: int = Field(ge=0)
    champion_model_name: str | None = None
    champion_backend_name: str | None = None
    promotion_executed: bool
    promotion_approved: bool | None = None


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
    data_quality: DataQualityReport | None = None


@dataclass
class RunArtifacts:
    prepared_panel: pd.DataFrame
    supervised_frame: pd.DataFrame
    predictions: pd.DataFrame
    metrics_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    cost_summary: pd.DataFrame
    sensitivity_summary: pd.DataFrame | None = None
    pareto_frontier: pd.DataFrame | None = None
    reorder_recommendations: pd.DataFrame | None = None
    exceptions: pd.DataFrame | None = None
    promotion_decision: PromotionDecisionMetadata | None = None
    champion_registry: ChampionRegistry | None = None
    operational_metadata: OperationalRunMetadata | None = None
    data_quality_report: DataQualityReport | None = None
    drifts: list[DriftEvent] = field(default_factory=list)
    report_extra: str = ""
    backtest_metadata: BacktestMetadata | None = None
    run_directory: Path | None = None
    shap_values: shap.Explanation | None = None


def write_run_artifacts(artifacts: RunArtifacts, settings: Settings) -> RunArtifacts:
    run_dir = make_run_directory(settings.reporting.output_dir, settings.reporting.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    reorder_recommendations = build_reorder_recommendations(artifacts, settings)
    exceptions = build_exceptions_frame(reorder_recommendations)
    registry_path = champion_registry_path(settings)
    champion_registry = load_champion_registry(registry_path)
    promotion_decision = build_promotion_decision(artifacts, settings, champion_registry)
    champion_registry = update_champion_registry(
        artifacts=artifacts,
        settings=settings,
        current_registry=champion_registry,
        promotion_decision=promotion_decision,
    )
    if artifacts.backtest_metadata is not None:
        artifacts.backtest_metadata = artifacts.backtest_metadata.model_copy(
            update={
                "promotion": promotion_decision,
                "data_quality": artifacts.data_quality_report,
            }
        )
    operational_metadata = build_operational_run_metadata(
        settings=settings,
        reorder_recommendations=reorder_recommendations,
        exceptions=exceptions,
        champion_registry=champion_registry,
        promotion_decision=promotion_decision,
    )

    reorder_recommendations.to_csv(run_dir / "reorder_recommendations.csv", index=False)
    exceptions.to_csv(run_dir / "exceptions.csv", index=False)
    if promotion_decision is not None:
        (run_dir / "promotion_decision.json").write_text(
            promotion_decision.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if champion_registry is not None:
        registry_path.write_text(
            champion_registry.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if artifacts.data_quality_report is not None:
        (run_dir / "data_quality_report.json").write_text(
            artifacts.data_quality_report.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if settings.project.run_mode == "score_daily":
        (run_dir / "operational_run_metadata.json").write_text(
            operational_metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
    else:
        artifacts.predictions.to_csv(run_dir / "predictions.csv", index=False)
        artifacts.metrics_summary.to_csv(run_dir / "metrics_summary.csv", index=False)
        artifacts.fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
        artifacts.cost_summary.to_csv(run_dir / "cost_summary.csv", index=False)
        if artifacts.sensitivity_summary is not None:
            artifacts.sensitivity_summary.to_csv(run_dir / "sensitivity_summary.csv", index=False)
        if artifacts.pareto_frontier is not None:
            artifacts.pareto_frontier.to_csv(run_dir / "pareto_frontier.csv", index=False)
        if settings.reporting.make_plots:
            render_standard_plots(
                metrics_summary=artifacts.metrics_summary,
                cost_summary=artifacts.cost_summary,
                output_dir=run_dir,
            )
            if artifacts.shap_values is not None:
                from retail_forecasting.visualization.plots import render_shap_summary

                render_shap_summary(
                    shap_values=artifacts.shap_values,
                    output_path=run_dir / "shap_summary.png",
                )

        report_text = build_markdown_report(artifacts=artifacts, settings=settings)
        (run_dir / "report.md").write_text(report_text, encoding="utf-8")

    artifacts.reorder_recommendations = reorder_recommendations
    artifacts.exceptions = exceptions
    artifacts.promotion_decision = promotion_decision
    artifacts.champion_registry = champion_registry
    artifacts.operational_metadata = operational_metadata
    if artifacts.backtest_metadata is not None and settings.project.run_mode != "score_daily":
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
        f"- `{section}`: `{values}`" for section, values in serializable_settings.items()
    ]

    from retail_forecasting.evaluation.post_mortem import generate_post_mortem_report

    post_mortem_text = generate_post_mortem_report(artifacts, settings)

    report = [
        "# Experiment Report",
        "",
        "## Executive Summary",
        "",
        "This report compares forecasting systems under predictive, probabilistic, and economic"
        " criteria. The primary ranking uses total operating cost under a newsvendor policy.",
        "",
        "## Post-Mortem Analysis (Top 5 Problematic SKUs)",
        "",
        post_mortem_text,
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
        f"- Date range: `{artifacts.prepared_panel['date'].min().date()}`"
        f" to `{artifacts.prepared_panel['date'].max().date()}`",
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
                f"- **ALERT**: Detected drift on `{event.date}`"
                f" (Score: `{event.score:.2f}`, Threshold: `{event.threshold:.2f}`)"
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
        "Candidate inventory policies are generated by scaling each model's order quantity."
        " Pareto-efficient rows are not dominated on cost, overstock, and stockout units.",
        "",
        dataframe_to_markdown(artifacts.pareto_frontier)
        if artifacts.pareto_frontier is not None
        else "_No Pareto frontier available._",
        "",
        "## Interpretation Notes",
        "",
        "- `MAE` and `RMSE` are included as diagnostics, not as the primary decision criterion.",
        "- `pinball_*` columns evaluate quantile quality when quantile forecasts are available.",
        "- `interval_coverage` (PICP): Fraction of observations in the P10-P90 interval. ~0.80.",
        "- `interval_width` (MPIW): Average width of the prediction interval. Narrower is better.",
        "- `winkler_score`: Proper scoring rule for intervals. Penalizes width and miscoverage.",
        "- `total_cost`: Static single-period inventory cost (Newsvendor). Good for SKU ranking.",
        "- `sim_total_cost`: Dynamic multi-period cost. Accounts for carry-over stock and backlog.",
        "- `sim_service_level`: Service level achieved in the dynamic simulation.",
        "- `total_cost` is the main ranking metric because the TFG focuses on inventory decisions.",
    ]

    return "\n".join(report)


def build_reorder_recommendations(artifacts: RunArtifacts, settings: Settings) -> pd.DataFrame:
    recommendations = artifacts.predictions.copy()
    recommendations["decision_date"] = recommendations["date"]
    recommendations["predicted_lead_time_demand"] = recommendations["y_pred"]

    series_parts = recommendations["series_id"].astype(str).str.split("_", n=1, expand=True)
    recommendations["store_id"] = series_parts[0]
    recommendations["product_id"] = series_parts[1]

    if "prediction_source" not in recommendations.columns:
        recommendations["prediction_source"] = "model"
    if "fallback_level" not in recommendations.columns:
        recommendations["fallback_level"] = pd.Series(
            pd.NA, index=recommendations.index, dtype="object"
        )

    recommendations["risk_flag"] = pd.Series(pd.NA, index=recommendations.index, dtype="object")
    recommendations["notes"] = pd.Series(pd.NA, index=recommendations.index, dtype="object")

    if settings.business.flag_cold_start:
        cold_start_mask = recommendations["prediction_source"] == "cold_start_fallback"
        recommendations.loc[cold_start_mask, "risk_flag"] = "cold_start"
        recommendations.loc[cold_start_mask, "notes"] = (
            "Fallback recommendation generated due to incomplete model features."
        )

    if settings.business.flag_drift_watch and artifacts.drifts:
        drift_fold_ids = {event.fold_id for event in artifacts.drifts}
        drift_mask = (
            recommendations["fold_id"].isin(drift_fold_ids) & recommendations["risk_flag"].isna()
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
        interval_width = recommendations[upper_quantile] - recommendations[lower_quantile]
        positive_width = interval_width[interval_width > 0]
        if not positive_width.empty:
            width_threshold = float(
                positive_width.quantile(settings.business.high_uncertainty_interval_quantile)
            )
            high_uncertainty_mask = (interval_width >= width_threshold) & recommendations[
                "risk_flag"
            ].isna()
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
    output_columns = [column for column in preferred_columns if column in recommendations.columns]
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
    output_columns = [column for column in preferred_columns if column in exceptions.columns]
    return exceptions.loc[:, output_columns]


def build_promotion_decision(
    artifacts: RunArtifacts,
    settings: Settings,
    champion_registry: ChampionRegistry | None = None,
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

    champion_reference = resolve_champion_reference(settings, champion_registry)
    champion_rows = summary.loc[_champion_mask(summary, champion_reference)].copy()
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
        challengers["cost_improvement_pct"] >= settings.business.champion_min_cost_improvement_pct
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
        decision_reason = (
            "Challenger improves total cost and respects the configured service-level tolerance."
        )
        promote = True
    else:
        selected = challengers.sort_values(
            ["total_cost", "service_level"],
            ascending=[True, False],
        ).iloc[0]
        decision_reason = (
            "No challenger satisfied both the minimum cost improvement and service-level guardrail."
        )
        promote = False

    return PromotionDecisionMetadata(
        champion_source=champion_reference.source,
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


def build_operational_run_metadata(
    settings: Settings,
    reorder_recommendations: pd.DataFrame,
    exceptions: pd.DataFrame,
    champion_registry: ChampionRegistry | None,
    promotion_decision: PromotionDecisionMetadata | None,
) -> OperationalRunMetadata:
    current_champion = champion_registry.current_champion if champion_registry else None
    return OperationalRunMetadata(
        run_name=settings.reporting.run_name,
        run_mode=settings.project.run_mode,
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
        config_hash=build_config_hash(settings),
        recommendation_rows=len(reorder_recommendations),
        exception_rows=len(exceptions),
        champion_model_name=(current_champion.model_name if current_champion is not None else None),
        champion_backend_name=(
            current_champion.backend_name if current_champion is not None else None
        ),
        promotion_executed=promotion_decision is not None,
        promotion_approved=(promotion_decision.promote if promotion_decision is not None else None),
    )


def champion_registry_path(settings: Settings) -> Path:
    return Path(settings.reporting.output_dir) / "champion_registry.json"


def load_champion_registry(path: Path) -> ChampionRegistry | None:
    if not path.exists():
        return None
    return ChampionRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def update_champion_registry(
    artifacts: RunArtifacts,
    settings: Settings,
    current_registry: ChampionRegistry | None,
    promotion_decision: PromotionDecisionMetadata | None,
) -> ChampionRegistry | None:
    if current_registry is not None and (
        promotion_decision is None or not promotion_decision.promote
    ):
        return current_registry

    reference = resolve_champion_reference(settings, current_registry)
    champion_rows = artifacts.cost_summary.loc[
        _champion_mask(artifacts.cost_summary, reference)
    ].copy()
    if champion_rows.empty:
        return current_registry

    base_row = champion_rows.sort_values("total_cost").iloc[0]
    promoted_row = base_row
    reason = "Registry bootstrapped from configured champion."

    if promotion_decision is not None and promotion_decision.promote:
        promoted_rows = artifacts.cost_summary.loc[
            _candidate_mask_from_decision(artifacts.cost_summary, promotion_decision)
        ].copy()
        if not promoted_rows.empty:
            promoted_row = promoted_rows.sort_values("total_cost").iloc[0]
            reason = promotion_decision.decision_reason

    _backend = str(promoted_row["backend_name"])
    _model_file = settings.reporting.output_dir / "models" / f"{_backend}.pkl"
    return ChampionRegistry(
        updated_at=utc_timestamp(),
        current_champion=ChampionRecord(
            data_strategy=_optional_string(promoted_row.get("data_strategy")),
            model_name=str(promoted_row["model_name"]),
            backend_name=_backend,
            promoted_at=utc_timestamp(),
            run_name=settings.reporting.run_name,
            git_commit=get_git_commit(),
            config_hash=build_config_hash(settings),
            reason=reason,
            model_path=str(_model_file) if _model_file.exists() else None,
        ),
    )


@dataclass(frozen=True)
class _ChampionReference:
    source: Literal["config", "registry"]
    data_strategy: str | None
    model_name: str
    backend_name: str


def resolve_champion_reference(
    settings: Settings, champion_registry: ChampionRegistry | None
) -> _ChampionReference:
    if champion_registry is not None:
        return _ChampionReference(
            source="registry",
            data_strategy=champion_registry.current_champion.data_strategy,
            model_name=champion_registry.current_champion.model_name,
            backend_name=champion_registry.current_champion.backend_name,
        )
    return _ChampionReference(
        source="config",
        data_strategy=settings.business.champion_data_strategy,
        model_name=settings.business.champion_model_name,
        backend_name=settings.business.champion_backend_name,
    )


def _champion_mask(summary: pd.DataFrame, reference: _ChampionReference) -> pd.Series:
    mask = (summary["model_name"] == reference.model_name) & (
        summary["backend_name"] == reference.backend_name
    )
    if "data_strategy" in summary.columns and reference.data_strategy is not None:
        mask &= summary["data_strategy"] == reference.data_strategy
    return mask


def _candidate_mask_from_decision(
    summary: pd.DataFrame, decision: PromotionDecisionMetadata
) -> pd.Series:
    mask = (summary["model_name"] == decision.challenger_model_name) & (
        summary["backend_name"] == decision.challenger_backend_name
    )
    if "data_strategy" in summary.columns and decision.challenger_data_strategy is not None:
        mask &= summary["data_strategy"] == decision.challenger_data_strategy
    return mask


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
