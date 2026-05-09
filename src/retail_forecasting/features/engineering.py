from __future__ import annotations

import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy

from retail_forecasting.config import FeatureConfig


def build_feature_frame(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Build reusable feature columns from a prepared daily panel.

    Args:
        panel: Prepared daily panel with one row per series and date.
        feature_config: Feature engineering configuration.

    Returns:
        A tuple containing the feature frame and the ordered feature column
        names used for modeling. The returned frame is not filtered for missing
        feature values, so callers can apply training or inference policies.

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
            lambda series: series.shift(1)
            .rolling(window=window, min_periods=window)
            .mean()
        )
        frame[sum_column] = grouped["observed_demand"].transform(
            lambda series: series.shift(1)
            .rolling(window=window, min_periods=window)
            .sum()
        )
        frame[std_column] = grouped["observed_demand"].transform(
            lambda series: series.shift(1)
            .rolling(window=window, min_periods=window)
            .std()
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
                lambda series: series.shift(1)
                .rolling(window=window, min_periods=window)
                .mean()
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

    return frame, feature_columns


def build_supervised_frame(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
    horizon: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the supervised modeling frame and feature list from a panel.

    Args:
        panel: Prepared daily panel with one row per series and date.
        feature_config: Feature engineering configuration.
        horizon: Forecast horizon expressed in days.

    Returns:
        A tuple containing the supervised frame and the ordered feature column
        names used for modeling.
    """
    frame, feature_columns = build_feature_frame(
        panel=panel,
        feature_config=feature_config,
    )
    grouped = frame.groupby("series_id", sort=False)

    frame["target_lead_time_demand"] = _build_target(
        grouped["observed_demand"], horizon
    )
    frame["target_horizon_days"] = horizon

    frame = frame.loc[frame["target_lead_time_demand"].notna()].copy()
    frame = frame.dropna(subset=feature_columns)
    frame = frame.reset_index(drop=True)

    return frame, feature_columns


def build_inference_frame(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one prediction-ready row per series from the latest valid origin.

    Args:
        panel: Prepared daily panel containing all history available at
            inference time.
        feature_config: Feature engineering configuration.

    Returns:
        A tuple containing the latest row per series with complete features and
        the ordered feature column names used for modeling.
    """
    frame, feature_columns = build_feature_frame(
        panel=panel,
        feature_config=feature_config,
    )
    frame = frame.dropna(subset=feature_columns)
    frame = frame.groupby("series_id", sort=False, as_index=False).tail(1)
    frame = frame.reset_index(drop=True)

    return frame, feature_columns


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
