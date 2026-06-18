from __future__ import annotations

from pathlib import Path

import numpy as np

from retail_forecasting.config import DatasetConfig, ReportingConfig, Settings
from retail_forecasting.forecasting.pipeline import run_experiment_from_frame
from tests import make_synthetic_panel


def test_pipeline_artifacts_follow_dataframe_contracts(tmp_path: Path) -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)
    settings = Settings(
        dataset=DatasetConfig(
            top_n_series=3,
            min_history_days=70,
            horizon=7,
        ),
        reporting=ReportingConfig(
            output_dir=tmp_path,
            run_name="dataframe_contract",
            make_plots=False,
        ),
    )

    artifacts = run_experiment_from_frame(panel, settings)

    _assert_prepared_panel_contract(artifacts.prepared_panel)
    _assert_supervised_frame_contract(
        supervised_frame=artifacts.supervised_frame,
        horizon=settings.dataset.horizon,
    )
    _assert_prediction_frame_contract(artifacts.predictions)
    _assert_metrics_summary_contract(artifacts.metrics_summary)
    _assert_fold_metrics_contract(artifacts.fold_metrics)
    _assert_cost_summary_contract(artifacts.cost_summary)
    _assert_tuning_pareto_contract(artifacts.tuning_pareto)


def _assert_prepared_panel_contract(frame) -> None:
    required_columns = {
        "date",
        "series_id",
        "observed_demand",
        "stockout_hours",
        "stockout_regime",
        "velocity_regime",
        "promo_regime",
        "seasonal_regime",
        "store_id",
        "product_id",
    }

    assert required_columns.issubset(frame.columns)
    assert not frame.duplicated(subset=["series_id", "date"]).any()
    assert (frame["observed_demand"] >= 0).all()


def _assert_supervised_frame_contract(supervised_frame, horizon: int) -> None:
    required_columns = {
        "date",
        "series_id",
        "target_lead_time_demand",
        "target_horizon_days",
    }

    assert required_columns.issubset(supervised_frame.columns)
    assert supervised_frame["target_lead_time_demand"].notna().all()
    assert (supervised_frame["target_horizon_days"] == horizon).all()


def _assert_prediction_frame_contract(predictions) -> None:
    required_columns = {
        "date",
        "series_id",
        "target_lead_time_demand",
        "y_true",
        "y_pred",
        "model_name",
        "backend_name",
        "fold_id",
        "order_quantity",
        "overstock_units",
        "stockout_units",
        "overstock_cost",
        "stockout_cost",
        "total_cost",
    }

    assert required_columns.issubset(predictions.columns)
    assert predictions["fold_id"].notna().all()
    assert np.allclose(predictions["y_true"], predictions["target_lead_time_demand"])
    assert (predictions["order_quantity"] >= 0).all()
    assert (predictions["overstock_units"] >= 0).all()
    assert (predictions["stockout_units"] >= 0).all()
    assert (predictions["total_cost"] >= 0).all()
    _assert_quantile_columns_are_monotonic(predictions)


def _assert_metrics_summary_contract(metrics_summary) -> None:
    required_columns = {
        "model_name",
        "backend_name",
        "observations",
        "mae",
        "rmse",
    }

    assert required_columns.issubset(metrics_summary.columns)
    assert (metrics_summary["observations"] > 0).all()
    assert (metrics_summary["mae"] >= 0).all()
    assert (metrics_summary["rmse"] >= 0).all()


def _assert_fold_metrics_contract(fold_metrics) -> None:
    required_columns = {
        "fold_id",
        "model_name",
        "backend_name",
        "observations",
        "mae",
        "rmse",
    }

    assert required_columns.issubset(fold_metrics.columns)
    assert fold_metrics["fold_id"].notna().all()
    assert (fold_metrics["observations"] > 0).all()


def _assert_cost_summary_contract(cost_summary) -> None:
    required_columns = {
        "model_name",
        "backend_name",
        "observations",
        "mean_order_quantity",
        "total_overstock_units",
        "total_stockout_units",
        "total_overstock_cost",
        "total_stockout_cost",
        "total_cost",
        "mean_cost",
        "service_level",
        "fill_rate",
    }

    assert required_columns.issubset(cost_summary.columns)
    assert (cost_summary["total_cost"] >= 0).all()
    assert cost_summary["service_level"].between(0.0, 1.0).all()
    assert cost_summary["fill_rate"].between(0.0, 1.0).all()


def _assert_tuning_pareto_contract(tuning_pareto) -> None:
    # The Pareto artifact is now the Optuna tuning frontier (Pinball vs Winkler),
    # populated when use_tuning is enabled (the default).
    assert tuning_pareto is not None
    required_columns = {
        "data_strategy",
        "trial_number",
        "pinball",
        "winkler",
        "is_on_front",
        "is_selected",
    }

    assert required_columns.issubset(tuning_pareto.columns)
    assert (tuning_pareto["trial_number"] >= 0).all()
    assert (tuning_pareto["pinball"] >= 0).all()
    assert tuning_pareto["is_on_front"].any()
    assert tuning_pareto["is_selected"].any()


def _assert_quantile_columns_are_monotonic(predictions) -> None:
    quantile_columns = [column for column in predictions.columns if column.startswith("q_")]
    if len(quantile_columns) < 2:
        return

    sorted_columns = sorted(quantile_columns, key=_quantile_level)
    quantiles = predictions.loc[:, sorted_columns].dropna(how="any")
    if quantiles.empty:
        return

    differences = np.diff(quantiles.to_numpy(dtype=float), axis=1)
    assert (differences >= 0).all()


def _quantile_level(column: str) -> float:
    return float(column.replace("q_", "").replace("_", "."))


def test_stockout_regime_thresholds() -> None:
    import pandas as pd

    from retail_forecasting.drift import label_all_regimes, label_stockout_regime

    # 1. Verify handling of zero-stockout case
    df_zero = pd.DataFrame({"stockout_hours": [0.0, 0.0, 0.0]})
    # With a custom threshold of 1.0 (or default 1.0)
    res_zero = label_stockout_regime(df_zero, threshold=1.0)
    assert (res_zero["stockout_regime"] == "low_stockout").all()

    # 2. Verify that strict inequality > is used rather than >=
    # If stockout_hours is exactly equal to the threshold, it should be "low_stockout"
    df_exact = pd.DataFrame({"stockout_hours": [1.0, 2.0, 0.5]})
    res_exact = label_stockout_regime(df_exact, threshold=1.0)
    assert res_exact.loc[0, "stockout_regime"] == "low_stockout"  # 1.0 is equal to threshold
    assert res_exact.loc[1, "stockout_regime"] == "high_stockout"  # 2.0 is > threshold
    assert res_exact.loc[2, "stockout_regime"] == "low_stockout"  # 0.5 is < threshold

    # 3. Verify label_all_regimes forwards the custom stockout threshold correctly
    df_all = pd.DataFrame(
        {
            "series_id": ["1", "1", "1"],
            "observed_demand": [10.0, 15.0, 20.0],
            "stockout_hours": [2.0, 0.0, 1.0],
            "discount": [0.0, 0.1, 0.0],
            "holiday_flag": [0.0, 0.0, 1.0],
        }
    )
    res_all = label_all_regimes(df_all, velocity_threshold=5.0, stockout_threshold=1.5)
    # Series average demand is 15.0 > 5.0 -> fast_moving
    # For stockout: 2.0 > 1.5 -> high_stockout. 0.0 and 1.0 are <= 1.5 -> low_stockout.
    assert res_all.loc[0, "stockout_regime"] == "high_stockout"
    assert res_all.loc[1, "stockout_regime"] == "low_stockout"
    assert res_all.loc[2, "stockout_regime"] == "low_stockout"
