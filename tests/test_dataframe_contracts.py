from __future__ import annotations

from pathlib import Path

import numpy as np

from retail_forecasting.config import DatasetConfig, ReportingConfig, Settings
from retail_forecasting.forecasting.pipeline import run_experiment_from_frame
from tests import make_synthetic_panel


def test_pipeline_artifacts_follow_dataframe_contracts(tmp_path: Path) -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)
    settings = Settings()
    settings.dataset = DatasetConfig(
        top_n_series=3,
        min_history_days=70,
        horizon=7,
    )
    settings.reporting = ReportingConfig(
        output_dir=tmp_path,
        run_name="dataframe_contract",
        make_plots=False,
    )

    artifacts = run_experiment_from_frame(panel, settings)

    _assert_prepared_panel_contract(artifacts.prepared_panel)
    _assert_supervised_frame_contract(
        supervised_frame=artifacts.supervised_frame,
        horizon=settings.dataset.horizon,
    )
    _assert_prediction_frame_contract(artifacts.predictions)
    _assert_cost_summary_contract(artifacts.cost_summary)


def _assert_prepared_panel_contract(frame) -> None:
    required_columns = {
        "date",
        "series_id",
        "observed_demand",
        "stockout_hours",
        "stockout_regime",
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
    assert np.allclose(predictions["y_true"], predictions["target_lead_time_demand"])
    assert (predictions["order_quantity"] >= 0).all()
    assert (predictions["overstock_units"] >= 0).all()
    assert (predictions["stockout_units"] >= 0).all()
    assert (predictions["total_cost"] >= 0).all()


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
    }

    assert required_columns.issubset(cost_summary.columns)
    assert cost_summary["total_cost"].is_monotonic_increasing
