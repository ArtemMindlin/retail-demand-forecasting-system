from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy
from pydantic import BaseModel, ConfigDict, Field

from retail_forecasting.config import FeatureConfig

logger = logging.getLogger(__name__)

DROP_WARNING_FRACTION = 0.2


class FeatureFrameMetadata(BaseModel):
    """Auditable metadata for feature frame construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["features", "supervised", "inference"]
    feature_columns: list[str] = Field(min_length=1)
    target_column: str | None = None
    horizon: int | None = Field(default=None, gt=0)
    lags: list[int] = Field(min_length=1)
    rolling_windows: list[int] = Field(min_length=1)
    input_rows: int = Field(ge=0)
    output_rows: int = Field(ge=0)
    dropped_rows_missing_target: int = Field(default=0, ge=0)
    dropped_rows_missing_features: int = Field(default=0, ge=0)
    rows_not_latest_origin: int = Field(default=0, ge=0)


class InferenceFallbackMetadata(BaseModel):
    """Auditable metadata for inference-time cold-start fallback planning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_columns: list[str] = Field(min_length=1)
    horizon: int = Field(gt=0)
    lags: list[int] = Field(min_length=1)
    rolling_windows: list[int] = Field(min_length=1)
    input_rows: int = Field(ge=0)
    output_rows: int = Field(ge=0)
    model_rows: int = Field(ge=0)
    cold_start_rows: int = Field(ge=0)
    fallback_rows_series: int = Field(default=0, ge=0)
    fallback_rows_product: int = Field(default=0, ge=0)
    fallback_rows_third_category: int = Field(default=0, ge=0)
    fallback_rows_global: int = Field(default=0, ge=0)


def build_feature_frame(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, FeatureFrameMetadata]:
    """Build reusable feature columns from a prepared daily panel.

    Args:
        panel: Prepared daily panel with one row per series and date.
        feature_config: Feature engineering configuration.

    Returns:
        A tuple containing the feature frame and metadata with the ordered
        feature column names used for modeling. The returned frame is not
        filtered for missing feature values, so callers can apply training or
        inference policies.

    Notes:
        Historical features are built with positive lags so they only use past
        information relative to each forecast origin.
    """
    _validate_required_columns(panel=panel, feature_config=feature_config)
    frame = panel.copy().sort_values(["series_id", "date"]).reset_index(drop=True)
    grouped = frame.groupby("series_id", sort=False)

    feature_columns: list[str] = []

    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_month"] = frame["date"].dt.day
    frame["month"] = frame["date"].dt.month
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    feature_columns.extend(
        [
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "is_weekend",
            "holiday_flag",
            "activity_flag",
        ]
    )

    # Demand lags: these features store past observed demand values for each series.
    demand_series = grouped["observed_demand"]
    for lag in sorted(set(feature_config.lags)):
        column = f"demand_lag_{lag}"
        frame[column] = demand_series.shift(lag)
        feature_columns.append(column)

    # Rolling window features: these summarize recent observed demand over a window of past days for each series.
    for window in sorted(set(feature_config.rolling_windows)):
        mean_column = f"demand_roll_mean_{window}"
        sum_column = f"demand_roll_sum_{window}"
        std_column = f"demand_roll_std_{window}"
        frame[mean_column] = grouped["observed_demand"].transform(
            lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).mean()
        )
        frame[sum_column] = grouped["observed_demand"].transform(
            lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).sum()
        )
        frame[std_column] = grouped["observed_demand"].transform(
            lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).std()
        )
        frame[std_column] = frame[std_column].fillna(0.0)
        feature_columns.extend([mean_column, sum_column, std_column])

    if feature_config.include_discount_lags:
        for lag in sorted(set(feature_config.lags)):
            column = f"discount_lag_{lag}"
            frame[column] = grouped["discount"].shift(lag)
            feature_columns.append(column)

    if feature_config.include_stockout_lags:
        for lag in sorted(set(feature_config.lags)):
            column = f"stockout_lag_{lag}"
            frame[column] = grouped["stockout_hours"].shift(lag)
            feature_columns.append(column)
        for window in sorted(set(feature_config.rolling_windows)):
            column = f"stockout_roll_mean_{window}"
            frame[column] = grouped["stockout_hours"].transform(
                lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).mean()
            )
            feature_columns.append(column)

    if feature_config.include_weather_lags:
        weather_columns = [
            "precpt",
            "avg_temperature",
            "avg_humidity",
            "avg_wind_level",
        ]
        for source_column in weather_columns:
            for lag in sorted(set(feature_config.lags)):
                column = f"{source_column}_lag_{lag}"
                frame[column] = grouped[source_column].shift(lag)
                feature_columns.append(column)

    if feature_config.include_static_ids:
        static_columns = [
            "city_id",
            "store_id",
            "management_group_id",
            "first_category_id",
            "second_category_id",
            "third_category_id",
            "product_id",
        ]
        feature_columns.extend(static_columns)

    return frame, FeatureFrameMetadata(
        mode="features",
        feature_columns=feature_columns,
        lags=sorted(set(feature_config.lags)),
        rolling_windows=sorted(set(feature_config.rolling_windows)),
        input_rows=len(panel),
        output_rows=len(frame),
    )


def build_supervised_frame(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
    horizon: int,
) -> tuple[pd.DataFrame, FeatureFrameMetadata]:
    """Build the supervised modeling frame and feature list from a panel.

    Args:
        panel: Prepared daily panel with one row per series and date.
        feature_config: Feature engineering configuration.
        horizon: Forecast horizon expressed in days.

    Returns:
        A tuple containing the supervised frame and metadata with the ordered
        feature column names used for modeling.
    """
    frame, feature_metadata = build_feature_frame(
        panel=panel,
        feature_config=feature_config,
    )
    feature_columns = feature_metadata.feature_columns
    grouped = frame.groupby("series_id", sort=False)

    frame["target_lead_time_demand"] = _build_target(grouped["observed_demand"], horizon)
    frame["target_horizon_days"] = horizon

    before_target_drop = len(frame)
    frame = frame.loc[frame["target_lead_time_demand"].notna()].copy()
    dropped_rows_missing_target = before_target_drop - len(frame)
    _warn_on_large_drop(
        step="missing target rows",
        input_rows=before_target_drop,
        dropped_rows=dropped_rows_missing_target,
    )

    before_feature_drop = len(frame)
    frame = frame.dropna(subset=feature_columns)
    dropped_rows_missing_features = before_feature_drop - len(frame)
    _warn_on_large_drop(
        step="missing feature rows",
        input_rows=before_feature_drop,
        dropped_rows=dropped_rows_missing_features,
    )
    frame = frame.reset_index(drop=True)
    _raise_if_empty(frame, mode="supervised")

    return frame, FeatureFrameMetadata(
        mode="supervised",
        feature_columns=feature_columns,
        target_column="target_lead_time_demand",
        horizon=horizon,
        lags=feature_metadata.lags,
        rolling_windows=feature_metadata.rolling_windows,
        input_rows=feature_metadata.input_rows,
        output_rows=len(frame),
        dropped_rows_missing_target=dropped_rows_missing_target,
        dropped_rows_missing_features=dropped_rows_missing_features,
    )


def build_inference_frame(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, FeatureFrameMetadata]:
    """Build one prediction-ready row per series from the latest valid origin.

    Args:
        panel: Prepared daily panel containing all history available at
            inference time.
        feature_config: Feature engineering configuration.

    Returns:
        A tuple containing the latest row per series with complete features and
        metadata with the ordered feature column names used for modeling.
    """
    frame, feature_metadata = build_feature_frame(
        panel=panel,
        feature_config=feature_config,
    )
    feature_columns = feature_metadata.feature_columns
    before_feature_drop = len(frame)
    frame = frame.dropna(subset=feature_columns)
    dropped_rows_missing_features = before_feature_drop - len(frame)
    _warn_on_large_drop(
        step="missing feature rows",
        input_rows=before_feature_drop,
        dropped_rows=dropped_rows_missing_features,
    )
    feature_complete_rows = len(frame)
    frame = frame.groupby("series_id", sort=False, as_index=False).tail(1)
    rows_not_latest_origin = feature_complete_rows - len(frame)
    frame = frame.reset_index(drop=True)
    _raise_if_empty(frame, mode="inference")

    return frame, FeatureFrameMetadata(
        mode="inference",
        feature_columns=feature_columns,
        lags=feature_metadata.lags,
        rolling_windows=feature_metadata.rolling_windows,
        input_rows=feature_metadata.input_rows,
        output_rows=len(frame),
        dropped_rows_missing_features=dropped_rows_missing_features,
        rows_not_latest_origin=rows_not_latest_origin,
    )


def build_inference_frame_with_fallback(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
    horizon: int,
) -> tuple[pd.DataFrame, InferenceFallbackMetadata]:
    """Build one latest-origin row per series with model-vs-fallback routing.

    Rows with complete feature availability are marked for model inference.
    Rows without complete features are preserved and assigned a hierarchical
    cold-start fallback prediction using daily observed-demand means scaled to
    the requested lead-time horizon.
    """
    _validate_fallback_columns(panel)
    frame, feature_metadata = build_feature_frame(
        panel=panel,
        feature_config=feature_config,
    )
    feature_columns = feature_metadata.feature_columns

    latest_rows = frame.groupby("series_id", sort=False, as_index=False).tail(1).copy()
    feature_complete_mask = latest_rows[feature_columns].notna().all(axis=1)

    latest_rows["prediction_source"] = "model"
    latest_rows["fallback_level"] = pd.Series(pd.NA, index=latest_rows.index, dtype="object")
    latest_rows["fallback_target_lead_time_demand"] = pd.NA

    cold_start_mask = ~feature_complete_mask
    if cold_start_mask.any():
        fallback_assignments = _resolve_cold_start_fallbacks(
            latest_rows.loc[cold_start_mask],
            panel=panel,
            horizon=horizon,
        )
        latest_rows.loc[cold_start_mask, "prediction_source"] = "cold_start_fallback"
        latest_rows.loc[cold_start_mask, "fallback_level"] = fallback_assignments[
            "fallback_level"
        ].to_numpy()
        latest_rows.loc[cold_start_mask, "fallback_target_lead_time_demand"] = fallback_assignments[
            "fallback_target_lead_time_demand"
        ].to_numpy()

    latest_rows = latest_rows.reset_index(drop=True)
    _raise_if_empty(latest_rows, mode="inference")

    fallback_counts = latest_rows.loc[
        latest_rows["prediction_source"] == "cold_start_fallback", "fallback_level"
    ].value_counts()

    return latest_rows, InferenceFallbackMetadata(
        feature_columns=feature_columns,
        horizon=horizon,
        lags=feature_metadata.lags,
        rolling_windows=feature_metadata.rolling_windows,
        input_rows=feature_metadata.input_rows,
        output_rows=len(latest_rows),
        model_rows=int((latest_rows["prediction_source"] == "model").sum()),
        cold_start_rows=int((latest_rows["prediction_source"] == "cold_start_fallback").sum()),
        fallback_rows_series=int(fallback_counts.get("series", 0)),
        fallback_rows_product=int(fallback_counts.get("product", 0)),
        fallback_rows_third_category=int(fallback_counts.get("third_category", 0)),
        fallback_rows_global=int(fallback_counts.get("global", 0)),
    )


def _build_target(series_group: SeriesGroupBy, horizon: int) -> pd.Series:
    """Aggregate future demand over the configured forecast horizon.

    Args:
        series_group: Grouped demand series by ``series_id``.
        horizon: Forecast horizon expressed in days.

    Returns:
        A Series containing lead-time demand targets for each row.
    """
    future_terms = [series_group.shift(-offset) for offset in range(horizon)]
    return pd.concat(future_terms, axis=1).sum(axis=1, min_count=horizon)


def _validate_required_columns(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
) -> None:
    required_columns = {
        "series_id",
        "date",
        "observed_demand",
        "holiday_flag",
        "activity_flag",
    }
    if feature_config.include_discount_lags:
        required_columns.add("discount")
    if feature_config.include_stockout_lags:
        required_columns.add("stockout_hours")
    if feature_config.include_weather_lags:
        required_columns.update(
            {
                "precpt",
                "avg_temperature",
                "avg_humidity",
                "avg_wind_level",
            }
        )
    if feature_config.include_static_ids:
        required_columns.update(
            {
                "city_id",
                "store_id",
                "management_group_id",
                "first_category_id",
                "second_category_id",
                "third_category_id",
                "product_id",
            }
        )

    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(
            "Cannot build feature frame without required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )


def _validate_fallback_columns(panel: pd.DataFrame) -> None:
    required_columns = {
        "series_id",
        "product_id",
        "third_category_id",
        "observed_demand",
    }
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(
            "Cannot build inference fallback plan without required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )


def _resolve_cold_start_fallbacks(
    cold_rows: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    series_means = panel.groupby("series_id")["observed_demand"].mean()
    product_means = panel.groupby("product_id")["observed_demand"].mean()
    third_category_means = panel.groupby("third_category_id")["observed_demand"].mean()
    global_mean = panel["observed_demand"].mean()

    fallback_levels: list[str] = []
    fallback_values: list[float] = []

    for row in cold_rows.itertuples(index=False):
        series_mean = series_means.get(row.series_id)
        if pd.notna(series_mean):
            fallback_levels.append("series")
            fallback_values.append(float(series_mean) * horizon)
            continue

        product_mean = product_means.get(row.product_id)
        if pd.notna(product_mean):
            fallback_levels.append("product")
            fallback_values.append(float(product_mean) * horizon)
            continue

        third_category_mean = third_category_means.get(row.third_category_id)
        if pd.notna(third_category_mean):
            fallback_levels.append("third_category")
            fallback_values.append(float(third_category_mean) * horizon)
            continue

        if pd.isna(global_mean):
            raise ValueError(
                "Cannot resolve cold-start fallback because observed_demand has no non-null history."
            )

        fallback_levels.append("global")
        fallback_values.append(float(global_mean) * horizon)

    return pd.DataFrame(
        {
            "fallback_level": fallback_levels,
            "fallback_target_lead_time_demand": fallback_values,
        },
        index=cold_rows.index,
    )


def _warn_on_large_drop(step: str, input_rows: int, dropped_rows: int) -> None:
    if input_rows <= 0 or dropped_rows <= 0:
        return

    dropped_fraction = dropped_rows / input_rows
    if dropped_fraction >= DROP_WARNING_FRACTION:
        logger.warning(
            "Feature engineering dropped %s of rows at %s (%s/%s).",
            f"{dropped_fraction:.1%}",
            step,
            dropped_rows,
            input_rows,
        )


def _raise_if_empty(frame: pd.DataFrame, mode: Literal["supervised", "inference"]) -> None:
    if frame.empty:
        raise ValueError(f"Feature engineering produced no usable rows for {mode} mode.")
