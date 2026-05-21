from __future__ import annotations

import pandas as pd
import pytest

from retail_forecasting.config import FeatureConfig
from retail_forecasting.contracts import FeatureMetadata
from retail_forecasting.features.engineering import (
    build_feature_frame,
    build_inference_frame,
    build_inference_frame_with_fallback,
    build_supervised_frame,
)
from tests import make_synthetic_panel


def test_build_supervised_frame_uses_only_past_features() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    supervised, metadata = build_supervised_frame(panel, feature_config, horizon=3)
    feature_columns = metadata.feature_columns
    assert "demand_lag_1" in feature_columns
    assert "target_lead_time_demand" in supervised.columns

    row = supervised.iloc[0]
    source = (
        panel.loc[panel["series_id"] == row["series_id"]].sort_values("date").reset_index(drop=True)
    )
    source_index = source.index[source["date"] == row["date"]][0]

    expected_lag_1 = source.loc[source_index - 1, "observed_demand"]
    expected_target = source.loc[source_index : source_index + 2, "observed_demand"].sum()

    assert row["demand_lag_1"] == expected_lag_1
    assert row["target_lead_time_demand"] == expected_target


def test_build_inference_frame_returns_latest_complete_feature_row_per_series() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    inference, metadata = build_inference_frame(panel, feature_config)
    feature_columns = metadata.feature_columns

    assert len(inference) == panel["series_id"].nunique()
    assert "target_lead_time_demand" not in inference.columns
    assert inference[feature_columns].notna().all().all()

    latest_dates = panel.groupby("series_id")["date"].max()
    for row in inference.itertuples(index=False):
        assert row.date == latest_dates.loc[row.series_id]


def test_build_supervised_frame_returns_pydantic_metadata() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])
    horizon = 3

    supervised, metadata = build_supervised_frame(
        panel,
        feature_config,
        horizon=horizon,
    )

    assert isinstance(metadata, FeatureMetadata)
    assert metadata.mode == "supervised"
    assert metadata.target_column == "target_lead_time_demand"
    assert metadata.horizon == horizon
    assert metadata.input_rows == len(panel)
    assert metadata.output_rows == 15
    assert metadata.feature_columns
    assert "target_lead_time_demand" not in metadata.feature_columns
    assert metadata.dropped_rows_missing_target == horizon - 1
    assert metadata.dropped_rows_missing_features == 3
    assert metadata.lags == [1, 2]
    assert metadata.rolling_windows == [3]


def test_build_inference_frame_returns_pydantic_metadata() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=20)
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    inference, metadata = build_inference_frame(
        panel,
        feature_config,
    )

    assert isinstance(metadata, FeatureMetadata)
    assert metadata.mode == "inference"
    assert metadata.target_column is None
    assert metadata.horizon is None
    assert metadata.input_rows == len(panel)
    assert metadata.output_rows == len(inference)
    assert metadata.output_rows == panel["series_id"].nunique()
    assert metadata.rows_not_latest_origin == 32


def test_build_feature_frame_validates_required_columns() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=20).drop(columns=["observed_demand"])
    feature_config = FeatureConfig(lags=[1], rolling_windows=[3])

    with pytest.raises(ValueError, match="observed_demand"):
        build_feature_frame(panel, feature_config)


def test_build_supervised_frame_fails_when_no_usable_rows_remain() -> None:
    panel = make_synthetic_panel(num_series=1, num_days=2)
    feature_config = FeatureConfig(lags=[1], rolling_windows=[3])

    with pytest.raises(
        ValueError,
        match="Feature engineering produced no usable rows for supervised mode.",
    ):
        build_supervised_frame(panel, feature_config, horizon=3)


def test_build_supervised_frame_warns_on_large_row_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    panel = make_synthetic_panel(num_series=1, num_days=20)
    feature_config = FeatureConfig(lags=[1], rolling_windows=[15])

    with caplog.at_level("WARNING"):
        build_supervised_frame(panel, feature_config, horizon=3)

    assert "Feature engineering dropped" in caplog.text


def test_build_inference_frame_with_fallback_marks_model_and_cold_start_rows() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=20)
    short_history = panel.loc[panel["series_id"] == "1_101"].tail(2).copy()
    mature_history = panel.loc[panel["series_id"] == "2_102"].copy()
    panel = (
        pd.concat([short_history, mature_history], ignore_index=True)
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    inference_plan, metadata = build_inference_frame_with_fallback(
        panel,
        feature_config,
        horizon=3,
    )

    assert len(inference_plan) == 2
    assert metadata.model_rows == 1
    assert metadata.cold_start_rows == 1
    assert metadata.fallback_rows_series == 1

    by_series = inference_plan.set_index("series_id")

    assert by_series.loc["2_102", "prediction_source"] == "model"
    assert pd.isna(by_series.loc["2_102", "fallback_level"])
    assert pd.isna(by_series.loc["2_102", "fallback_target_lead_time_demand"])

    expected_series_mean = short_history["observed_demand"].mean() * 3
    assert by_series.loc["1_101", "prediction_source"] == "cold_start_fallback"
    assert by_series.loc["1_101", "fallback_level"] == "series"
    assert by_series.loc["1_101", "fallback_target_lead_time_demand"] == pytest.approx(
        expected_series_mean
    )


def test_build_inference_frame_with_fallback_uses_hierarchical_levels() -> None:
    donor_product = make_synthetic_panel(num_series=1, num_days=5).copy()
    donor_product["series_id"] = "10_200"
    donor_product["store_id"] = 10
    donor_product["product_id"] = 200
    donor_product["third_category_id"] = 900
    donor_product["observed_demand"] = 4.0

    cold_product = donor_product.tail(1).copy()
    cold_product["series_id"] = "11_200"
    cold_product["store_id"] = 11
    cold_product["observed_demand"] = pd.NA

    donor_category = make_synthetic_panel(num_series=1, num_days=5).copy()
    donor_category["series_id"] = "20_300"
    donor_category["store_id"] = 20
    donor_category["product_id"] = 300
    donor_category["third_category_id"] = 901
    donor_category["observed_demand"] = 6.0

    cold_category = donor_category.tail(1).copy()
    cold_category["series_id"] = "21_301"
    cold_category["store_id"] = 21
    cold_category["product_id"] = 301
    cold_category["observed_demand"] = pd.NA

    donor_global = make_synthetic_panel(num_series=1, num_days=5).copy()
    donor_global["series_id"] = "30_400"
    donor_global["store_id"] = 30
    donor_global["product_id"] = 400
    donor_global["third_category_id"] = 902
    donor_global["observed_demand"] = 8.0

    cold_global = donor_global.tail(1).copy()
    cold_global["series_id"] = "31_401"
    cold_global["store_id"] = 31
    cold_global["product_id"] = 401
    cold_global["third_category_id"] = 903
    cold_global["observed_demand"] = pd.NA

    panel = (
        pd.concat(
            [
                donor_product,
                cold_product,
                donor_category,
                cold_category,
                donor_global,
                cold_global,
            ],
            ignore_index=True,
        )
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )
    feature_config = FeatureConfig(lags=[1, 2], rolling_windows=[3])

    inference_plan, metadata = build_inference_frame_with_fallback(
        panel,
        feature_config,
        horizon=2,
    )

    by_series = inference_plan.set_index("series_id")

    assert by_series.loc["11_200", "fallback_level"] == "product"
    assert by_series.loc["11_200", "fallback_target_lead_time_demand"] == pytest.approx(8.0)

    assert by_series.loc["21_301", "fallback_level"] == "third_category"
    assert by_series.loc["21_301", "fallback_target_lead_time_demand"] == pytest.approx(12.0)

    expected_global = panel["observed_demand"].mean() * 2
    assert by_series.loc["31_401", "fallback_level"] == "global"
    assert by_series.loc["31_401", "fallback_target_lead_time_demand"] == pytest.approx(
        expected_global
    )

    assert metadata.cold_start_rows >= 3
    assert metadata.fallback_rows_product >= 1
    assert metadata.fallback_rows_third_category >= 1
    assert metadata.fallback_rows_global >= 1
