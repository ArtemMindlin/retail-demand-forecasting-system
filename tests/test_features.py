from __future__ import annotations

import pytest

from retail_forecasting.config import FeatureConfig
from retail_forecasting.features.engineering import (
    FeatureFrameMetadata,
    build_feature_frame,
    build_inference_frame,
    build_supervised_frame,
)
from tests import make_synthetic_panel


def test_build_supervised_frame_uses_only_past_features() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    supervised, feature_columns = build_supervised_frame(
        panel, feature_config, horizon=3
    )
    assert "demand_lag_1" in feature_columns
    assert "target_lead_time_demand" in supervised.columns

    row = supervised.iloc[0]
    source = (
        panel.loc[panel["series_id"] == row["series_id"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    source_index = source.index[source["date"] == row["date"]][0]

    expected_lag_1 = source.loc[source_index - 1, "observed_demand"]
    expected_target = source.loc[
        source_index : source_index + 2, "observed_demand"
    ].sum()

    assert row["demand_lag_1"] == expected_lag_1
    assert row["target_lead_time_demand"] == expected_target


def test_build_inference_frame_returns_latest_complete_feature_row_per_series() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    inference, feature_columns = build_inference_frame(panel, feature_config)

    assert len(inference) == panel["series_id"].nunique()
    assert "target_lead_time_demand" not in inference.columns
    assert inference[feature_columns].notna().all().all()

    latest_dates = panel.groupby("series_id")["date"].max()
    for row in inference.itertuples(index=False):
        assert row.date == latest_dates.loc[row.series_id]


def test_build_supervised_frame_can_return_pydantic_metadata() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])
    horizon = 3

    supervised, metadata = build_supervised_frame(
        panel,
        feature_config,
        horizon=horizon,
        return_metadata=True,
    )

    assert isinstance(metadata, FeatureFrameMetadata)
    assert metadata.mode == "supervised"
    assert metadata.target_column == "target_lead_time_demand"
    assert metadata.horizon == horizon
    assert metadata.input_rows == len(panel)
    assert metadata.output_rows == len(supervised)
    assert metadata.feature_columns
    assert "target_lead_time_demand" not in metadata.feature_columns
    assert metadata.dropped_rows_missing_target == horizon - 1
    assert metadata.dropped_rows_missing_features > 0
    assert metadata.lags == [1, 2]
    assert metadata.rolling_windows == [3]


def test_build_inference_frame_can_return_pydantic_metadata() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    inference, metadata = build_inference_frame(
        panel,
        feature_config,
        return_metadata=True,
    )

    assert isinstance(metadata, FeatureFrameMetadata)
    assert metadata.mode == "inference"
    assert metadata.target_column is None
    assert metadata.horizon is None
    assert metadata.input_rows == len(panel)
    assert metadata.output_rows == len(inference)
    assert metadata.output_rows == panel["series_id"].nunique()
    assert metadata.rows_not_latest_origin > 0


def test_build_feature_frame_validates_required_columns() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=20).drop(
        columns=["observed_demand"]
    )
    feature_config = FeatureConfig(lags=[1], rolling_windows=[3])

    with pytest.raises(ValueError, match="observed_demand"):
        build_feature_frame(panel, feature_config)
