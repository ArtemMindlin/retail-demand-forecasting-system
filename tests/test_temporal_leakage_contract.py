from __future__ import annotations

import numpy as np
import pandas as pd

from retail_forecasting.config import FeatureConfig, ValidationConfig
from retail_forecasting.features.engineering import build_supervised_frame
from retail_forecasting.forecasting.backtesting import build_walk_forward_folds
from tests import make_synthetic_panel


def test_supervised_frame_uses_future_demand_only_as_target() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=30)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])
    horizon = 4

    supervised, metadata = build_supervised_frame(
        panel=panel,
        feature_config=feature_config,
        horizon=horizon,
    )
    feature_columns = metadata.feature_columns

    row = supervised.iloc[0]
    source = _series_source(panel, row["series_id"])
    source_index = _source_index_for_date(source, row["date"])

    expected_target = source.loc[
        source_index : source_index + horizon - 1,
        "observed_demand",
    ].sum()

    assert "target_lead_time_demand" not in feature_columns
    assert row["target_lead_time_demand"] == expected_target
    assert row["target_horizon_days"] == horizon


def test_historical_features_exclude_current_row_values() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=30)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    supervised, metadata = build_supervised_frame(
        panel=panel,
        feature_config=feature_config,
        horizon=3,
    )
    feature_columns = metadata.feature_columns

    row = supervised.iloc[0]
    source = _series_source(panel, row["series_id"])
    source_index = _source_index_for_date(source, row["date"])
    past_demand_window = source.loc[
        source_index - 3 : source_index - 1, "observed_demand"
    ]

    assert row["demand_lag_1"] == source.loc[source_index - 1, "observed_demand"]
    assert row["demand_lag_2"] == source.loc[source_index - 2, "observed_demand"]
    assert np.isclose(row["demand_roll_mean_3"], past_demand_window.mean())
    assert np.isclose(row["demand_roll_sum_3"], past_demand_window.sum())
    assert np.isclose(row["demand_roll_std_3"], past_demand_window.std())

    assert "observed_demand" not in feature_columns


def test_realized_context_enters_model_only_as_lagged_features() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=30)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    supervised, metadata = build_supervised_frame(
        panel=panel,
        feature_config=feature_config,
        horizon=3,
    )
    feature_columns = metadata.feature_columns

    row = supervised.iloc[0]
    source = _series_source(panel, row["series_id"])
    source_index = _source_index_for_date(source, row["date"])

    realized_same_day_columns = {
        "discount",
        "stockout_hours",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    }
    assert realized_same_day_columns.isdisjoint(feature_columns)

    assert row["discount_lag_1"] == source.loc[source_index - 1, "discount"]
    assert row["stockout_lag_2"] == source.loc[source_index - 2, "stockout_hours"]
    assert np.isclose(
        row["stockout_roll_mean_3"],
        source.loc[
            source_index - 3 : source_index - 1,
            "stockout_hours",
        ].mean(),
    )
    assert row["precpt_lag_1"] == source.loc[source_index - 1, "precpt"]
    assert (
        row["avg_temperature_lag_2"]
        == source.loc[
            source_index - 2,
            "avg_temperature",
        ]
    )


def test_walk_forward_folds_leave_horizon_gap_before_validation() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=100)
    validation = ValidationConfig(initial_train_days=40, n_folds=3, fold_size_days=5)

    for horizon in [1, 3, 7]:
        folds = build_walk_forward_folds(panel, validation, horizon=horizon)

        for fold in folds:
            assert fold.train_end_date == fold.validation_start_date - pd.Timedelta(
                days=horizon,
            )

            latest_training_target_end = fold.train_end_date + pd.Timedelta(
                days=horizon - 1,
            )
            assert latest_training_target_end < fold.validation_start_date


def _series_source(panel: pd.DataFrame, series_id: str) -> pd.DataFrame:
    return (
        panel.loc[panel["series_id"] == series_id]
        .sort_values("date")
        .reset_index(drop=True)
    )


def _source_index_for_date(source: pd.DataFrame, date: pd.Timestamp) -> int:
    matches = np.flatnonzero(source["date"].to_numpy() == np.datetime64(date))
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one source row for date {date}.")
    return int(matches[0])
