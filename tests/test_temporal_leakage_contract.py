from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecasting.config import (
    DatasetConfig,
    FeatureConfig,
    ModelConfig,
    ReportingConfig,
    Settings,
    ValidationConfig,
)
from retail_forecasting.drift import label_all_regimes
from retail_forecasting.features.engineering import build_supervised_frame
from retail_forecasting.forecasting.backtesting import build_walk_forward_folds
from retail_forecasting.forecasting.pipeline import (
    _split_train_calibration,
    run_experiment_from_frame,
)
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
    past_demand_window = source.loc[source_index - 3 : source_index - 1, "observed_demand"]

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


def test_regime_labels_never_become_features() -> None:
    """The regime labels ride in the panel as metadata and must stay out of the model.

    `run_experiment` and `run_scoring` label the panel before building features, so these
    four columns ARE present when the supervised frame is built. Two of them would leak if
    picked up: `stockout_regime` thresholds the same day's `stockout_hours`, which the hard
    rules say may only enter lagged, and `velocity_regime` thresholds a per-series mean taken
    over the whole frame, future dates included -- the shape invariant 8 forbids.

    What keeps them out is the allowlist in `build_feature_frame`, which names every feature
    it adds. That was a convention no test held: the sibling check above only blocks the raw
    same-day columns, so adding the labels as categoricals passed green with a leak inside.
    """
    panel = label_all_regimes(make_synthetic_panel(num_series=2, num_days=40))
    regime_columns = {
        "stockout_regime",
        "velocity_regime",
        "promo_regime",
        "seasonal_regime",
    }
    assert regime_columns.issubset(panel.columns), "the panel is meant to carry the labels"

    _, metadata = build_supervised_frame(
        panel=panel,
        feature_config=FeatureConfig(lags=[1, 2], rolling_windows=[3]),
        horizon=3,
    )
    assert regime_columns.isdisjoint(metadata.feature_columns)


def test_walk_forward_folds_leave_horizon_gap_before_validation() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=100)
    validation = ValidationConfig(initial_train_days=40, n_folds=3, fold_size_days=5)

    for horizon in [1, 3, 7]:
        folds = build_walk_forward_folds(panel, validation, horizon=horizon)

        for fold in folds:
            assert fold.horizon == horizon
            assert fold.train_end_date == fold.validation_start_date - pd.Timedelta(
                days=horizon,
            )

            latest_training_target_end = fold.train_end_date + pd.Timedelta(
                days=horizon - 1,
            )
            assert latest_training_target_end < fold.validation_start_date


def test_calibration_split_embargoes_the_horizon() -> None:
    """Conformal calibration targets must not overlap the sub-training targets.

    Without the embargo the last ``horizon - 1`` training rows carry demand from
    days inside the calibration window, so the conformity scores come out
    optimistically small and the resulting intervals undercover.
    """
    panel = make_synthetic_panel(num_series=2, num_days=120)
    horizon = 7
    supervised, _ = build_supervised_frame(
        panel=panel,
        feature_config=FeatureConfig(lags=[1, 2], rolling_windows=[3]),
        horizon=horizon,
    )
    settings = Settings(
        dataset=DatasetConfig(top_n_series=2, min_history_days=70, horizon=horizon),
        validation=ValidationConfig(calibration_days=14),
    )

    sub_train, calib, _ = _split_train_calibration(supervised, settings)

    assert not sub_train.empty
    assert not calib.empty

    latest_training_target_end = sub_train["date"].max() + pd.Timedelta(days=horizon - 1)
    assert latest_training_target_end < calib["date"].min()


def test_forecast_metrics_cover_every_validation_origin(tmp_path: Path) -> None:
    """Forecast metrics must summarize all validation origins.

    Every origin carries its own single-period decision, so the decision frame and
    the validation frame hold the same rows. A coarser ordering cadence would make
    the former a strict subset, and measuring MAE, Winkler or coverage on it would
    silently discard predictions -- which is what this test exists to catch.
    """
    panel = make_synthetic_panel(num_series=3, num_days=90)
    settings = Settings(
        dataset=DatasetConfig(top_n_series=3, min_history_days=70, horizon=7),
        models=ModelConfig(use_tuning=False, models_dir=tmp_path / "models"),
        reporting=ReportingConfig(make_plots=False),
    )

    artifacts = run_experiment_from_frame(panel, settings, save_artifacts=False)

    validation = artifacts.validation_predictions
    assert validation is not None
    # One decision per validation origin: the two frames must not diverge.
    assert len(artifacts.predictions) == len(validation)

    group_cols = ["data_strategy", "model_name", "backend_name"]
    observations = (
        artifacts.metrics_summary.set_index(group_cols)["observations"]
        if all(col in artifacts.metrics_summary.columns for col in group_cols)
        else artifacts.metrics_summary.set_index("model_name")["observations"]
    )
    expected = validation.groupby(group_cols).size()
    for key, count in expected.items():
        assert observations.loc[key] == count


def _series_source(panel: pd.DataFrame, series_id: str) -> pd.DataFrame:
    return panel.loc[panel["series_id"] == series_id].sort_values("date").reset_index(drop=True)


def _source_index_for_date(source: pd.DataFrame, date: pd.Timestamp) -> int:
    matches = np.flatnonzero(source["date"].to_numpy() == np.datetime64(date))
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one source row for date {date}.")
    return int(matches[0])
