from __future__ import annotations

import pandas as pd

from retail_forecasting.config import BusinessConfig, Settings
from retail_forecasting.contracts.drift import DriftEvent
from retail_forecasting.evaluation.reporting import (
    RunArtifacts,
    build_promotion_decision,
    build_exceptions_frame,
    build_reorder_recommendations,
)


def test_build_reorder_recommendations_flags_cold_start_and_exceptions() -> None:
    artifacts = RunArtifacts(
        prepared_panel=pd.DataFrame(),
        supervised_frame=pd.DataFrame(),
        predictions=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
                "series_id": ["1_101", "2_202"],
                "y_pred": [12.0, 18.0],
                "order_quantity": [13.0, 19.0],
                "prediction_source": ["cold_start_fallback", "model"],
                "fallback_level": ["series", pd.NA],
                "model_name": ["seasonal_naive", "catboost"],
                "backend_name": ["heuristic", "catboost"],
                "fold_id": [0, 0],
                "stockout_hours": [3.0, 0.0],
                "stockout_regime": ["high", "low"],
            }
        ),
        metrics_summary=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        cost_summary=pd.DataFrame(),
    )
    settings = Settings(
        business=BusinessConfig(
            flag_drift_watch=False,
            flag_high_uncertainty=False,
            flag_extreme_order_quantity=False,
        )
    )

    recommendations = build_reorder_recommendations(artifacts, settings)
    exceptions = build_exceptions_frame(recommendations)

    cold_start_row = recommendations.loc[recommendations["series_id"] == "1_101"].iloc[
        0
    ]
    normal_row = recommendations.loc[recommendations["series_id"] == "2_202"].iloc[0]

    assert cold_start_row["risk_flag"] == "cold_start"
    assert cold_start_row["fallback_level"] == "series"
    assert normal_row["risk_flag"] is pd.NA or pd.isna(normal_row["risk_flag"])
    assert exceptions["series_id"].tolist() == ["1_101"]
    assert exceptions["risk_flag"].tolist() == ["cold_start"]


def test_build_reorder_recommendations_flags_drift_watch() -> None:
    artifacts = RunArtifacts(
        prepared_panel=pd.DataFrame(),
        supervised_frame=pd.DataFrame(),
        predictions=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-06-01", "2024-06-08"]),
                "series_id": ["1_101", "2_202"],
                "y_pred": [12.0, 15.0],
                "order_quantity": [12.0, 15.0],
                "model_name": ["catboost", "catboost"],
                "backend_name": ["catboost", "catboost"],
                "fold_id": [0, 1],
            }
        ),
        metrics_summary=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        cost_summary=pd.DataFrame(),
        drifts=[DriftEvent(date="2024-06-08", score=20.0, threshold=15.0, fold_id=1)],
    )
    settings = Settings(
        business=BusinessConfig(
            flag_high_uncertainty=False,
            flag_extreme_order_quantity=False,
        )
    )

    recommendations = build_reorder_recommendations(artifacts, settings)

    stable_row = recommendations.loc[recommendations["fold_id"] == 0].iloc[0]
    drift_row = recommendations.loc[recommendations["fold_id"] == 1].iloc[0]

    assert pd.isna(stable_row["risk_flag"])
    assert drift_row["risk_flag"] == "drift_watch"


def test_build_reorder_recommendations_flags_high_uncertainty_from_quantiles() -> None:
    artifacts = RunArtifacts(
        prepared_panel=pd.DataFrame(),
        supervised_frame=pd.DataFrame(),
        predictions=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-06-01"]),
                "series_id": ["1_101", "2_202", "3_303"],
                "y_pred": [10.0, 10.0, 10.0],
                "order_quantity": [10.0, 10.0, 10.0],
                "model_name": ["catboost", "catboost", "catboost"],
                "backend_name": ["catboost", "catboost", "catboost"],
                "fold_id": [0, 0, 0],
                "q_0_1": [9.0, 8.0, 1.0],
                "q_0_5": [10.0, 10.0, 10.0],
                "q_0_9": [11.0, 12.0, 30.0],
            }
        ),
        metrics_summary=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        cost_summary=pd.DataFrame(),
    )
    settings = Settings(
        business=BusinessConfig(
            high_uncertainty_interval_quantile=0.8,
            flag_extreme_order_quantity=False,
        )
    )

    recommendations = build_reorder_recommendations(artifacts, settings)
    flagged = recommendations.loc[
        recommendations["risk_flag"] == "high_uncertainty", "series_id"
    ].tolist()

    assert flagged == ["3_303"]


def test_build_reorder_recommendations_flags_extreme_order_quantity() -> None:
    artifacts = RunArtifacts(
        prepared_panel=pd.DataFrame(),
        supervised_frame=pd.DataFrame(),
        predictions=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-06-01"]),
                "series_id": ["1_101", "2_202", "3_303"],
                "y_pred": [10.0, 20.0, 300.0],
                "order_quantity": [10.0, 20.0, 300.0],
                "model_name": ["catboost", "catboost", "catboost"],
                "backend_name": ["catboost", "catboost", "catboost"],
                "fold_id": [0, 0, 0],
            }
        ),
        metrics_summary=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        cost_summary=pd.DataFrame(),
    )
    settings = Settings(
        business=BusinessConfig(
            flag_high_uncertainty=False,
            extreme_order_quantity_quantile=0.8,
        )
    )

    recommendations = build_reorder_recommendations(artifacts, settings)
    flagged = recommendations.loc[
        recommendations["risk_flag"] == "extreme_order_quantity", "series_id"
    ].tolist()

    assert flagged == ["3_303"]


def test_build_promotion_decision_recommends_promotion_when_guardrails_pass() -> None:
    artifacts = RunArtifacts(
        prepared_panel=pd.DataFrame(),
        supervised_frame=pd.DataFrame(),
        predictions=pd.DataFrame(),
        metrics_summary=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        cost_summary=pd.DataFrame(
            {
                "data_strategy": ["Observed", "Observed", "Latent_supervised"],
                "model_name": ["catboost", "seasonal_naive", "auto_boosting"],
                "backend_name": ["catboost", "heuristic", "lightgbm"],
                "observations": [10, 10, 10],
                "mean_order_quantity": [11.0, 10.0, 10.5],
                "total_overstock_units": [5.0, 8.0, 4.0],
                "total_stockout_units": [4.0, 7.0, 3.0],
                "total_overstock_cost": [5.0, 8.0, 4.0],
                "total_stockout_cost": [16.0, 28.0, 12.0],
                "total_cost": [21.0, 36.0, 18.0],
                "mean_cost": [2.1, 3.6, 1.8],
                "service_level": [0.80, 0.70, 0.79],
                "fill_rate": [0.92, 0.85, 0.94],
            }
        ),
    )
    settings = Settings(
        business=BusinessConfig(
            champion_data_strategy="Observed",
            champion_model_name="catboost",
            champion_backend_name="catboost",
            champion_min_cost_improvement_pct=10.0,
            champion_max_service_level_degradation=0.02,
        )
    )

    decision = build_promotion_decision(artifacts, settings)

    assert decision is not None
    assert decision.promote is True
    assert decision.challenger_model_name == "auto_boosting"
    assert decision.challenger_backend_name == "lightgbm"


def test_build_promotion_decision_blocks_promotion_when_service_level_degrades_too_much() -> (
    None
):
    artifacts = RunArtifacts(
        prepared_panel=pd.DataFrame(),
        supervised_frame=pd.DataFrame(),
        predictions=pd.DataFrame(),
        metrics_summary=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        cost_summary=pd.DataFrame(
            {
                "data_strategy": ["Observed", "Latent_supervised"],
                "model_name": ["catboost", "auto_boosting"],
                "backend_name": ["catboost", "lightgbm"],
                "observations": [10, 10],
                "mean_order_quantity": [11.0, 10.5],
                "total_overstock_units": [5.0, 4.0],
                "total_stockout_units": [4.0, 3.0],
                "total_overstock_cost": [5.0, 4.0],
                "total_stockout_cost": [16.0, 12.0],
                "total_cost": [21.0, 18.0],
                "mean_cost": [2.1, 1.8],
                "service_level": [0.80, 0.70],
                "fill_rate": [0.92, 0.94],
            }
        ),
    )
    settings = Settings(
        business=BusinessConfig(
            champion_data_strategy="Observed",
            champion_model_name="catboost",
            champion_backend_name="catboost",
            champion_min_cost_improvement_pct=10.0,
            champion_max_service_level_degradation=0.02,
        )
    )

    decision = build_promotion_decision(artifacts, settings)

    assert decision is not None
    assert decision.promote is False
    assert decision.challenger_model_name == "auto_boosting"
