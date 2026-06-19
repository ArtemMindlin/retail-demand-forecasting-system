from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

from retail_forecasting.config import FeatureConfig
from retail_forecasting.contracts import FeatureMetadata, InferenceFallbackMetadata

logger = logging.getLogger(__name__)

DROP_WARNING_FRACTION = 0.2


def _add_lag_features(
    frame: pd.DataFrame,
    grouped: DataFrameGroupBy,
    source_col: str,
    lags: list[int],
    prefix: str,
) -> list[str]:
    """Add shifted lag columns ``{prefix}_lag_{lag}`` to ``frame``; return their names."""
    added = []
    for lag in sorted(set(lags)):
        column = f"{prefix}_lag_{lag}"
        frame[column] = grouped[source_col].shift(lag)
        added.append(column)
    return added


def _rolling_feature(
    grouped: DataFrameGroupBy,
    source_col: str,
    window: int,
    agg: str,
) -> pd.Series:
    """Past-only rolling aggregate (shift(1) avoids leaking the current day)."""
    return grouped[source_col].transform(
        lambda series, w=window: getattr(series.shift(1).rolling(window=w, min_periods=w), agg)()
    )


def build_feature_frame(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Build reusable feature columns from a prepared daily panel."""
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

    feature_columns.extend(
        _add_lag_features(frame, grouped, "observed_demand", feature_config.lags, "demand")
    )

    for window in sorted(set(feature_config.rolling_windows)):
        mean_column = f"demand_roll_mean_{window}"
        sum_column = f"demand_roll_sum_{window}"
        std_column = f"demand_roll_std_{window}"
        frame[mean_column] = _rolling_feature(grouped, "observed_demand", window, "mean")
        frame[sum_column] = _rolling_feature(grouped, "observed_demand", window, "sum")
        frame[std_column] = _rolling_feature(grouped, "observed_demand", window, "std").fillna(0.0)
        feature_columns.extend([mean_column, sum_column, std_column])

    if feature_config.include_discount_lags:
        feature_columns.extend(
            _add_lag_features(frame, grouped, "discount", feature_config.lags, "discount")
        )

    if feature_config.include_stockout_lags:
        feature_columns.extend(
            _add_lag_features(frame, grouped, "stockout_hours", feature_config.lags, "stockout")
        )
        for window in sorted(set(feature_config.rolling_windows)):
            column = f"stockout_roll_mean_{window}"
            frame[column] = _rolling_feature(grouped, "stockout_hours", window, "mean")
            feature_columns.append(column)

    if feature_config.include_weather_lags:
        weather_columns = [
            "precpt",
            "avg_temperature",
            "avg_humidity",
            "avg_wind_level",
        ]
        for source_column in weather_columns:
            feature_columns.extend(
                _add_lag_features(frame, grouped, source_column, feature_config.lags, source_column)
            )

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
) -> tuple[pd.DataFrame, FeatureMetadata]:
    """Build the supervised modeling frame and feature list from a panel.

    Args:
        panel: Prepared daily panel with one row per series and date.
        feature_config: Feature engineering configuration.
        horizon: Forecast horizon expressed in days.

    Returns:
        A tuple containing the supervised frame and metadata with the ordered
        feature column names used for modeling.
    """
    frame, feature_columns = build_feature_frame(
        panel=panel,
        feature_config=feature_config,
    )
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
    if frame.empty:
        raise ValueError("Feature engineering produced no usable rows for supervised mode.")

    return frame, FeatureMetadata(
        mode="supervised",
        feature_columns=feature_columns,
        target_column="target_lead_time_demand",
        horizon=horizon,
        lags=sorted(set(feature_config.lags)),
        rolling_windows=sorted(set(feature_config.rolling_windows)),
        input_rows=len(panel),
        output_rows=len(frame),
        dropped_rows_missing_target=dropped_rows_missing_target,
        dropped_rows_missing_features=dropped_rows_missing_features,
    )


def build_inference_frame(
    panel: pd.DataFrame,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, FeatureMetadata]:
    """Build one prediction-ready row per series from the latest valid origin.

    Args:
        panel: Prepared daily panel containing all history available at
            inference time.
        feature_config: Feature engineering configuration.

    Returns:
        A tuple containing the latest row per series with complete features and
        metadata with the ordered feature column names used for modeling.
    """
    frame, feature_columns = build_feature_frame(
        panel=panel,
        feature_config=feature_config,
    )
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
    if frame.empty:
        raise ValueError("Feature engineering produced no usable rows for inference mode.")

    return frame, FeatureMetadata(
        mode="inference",
        feature_columns=feature_columns,
        lags=sorted(set(feature_config.lags)),
        rolling_windows=sorted(set(feature_config.rolling_windows)),
        input_rows=len(panel),
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
    frame, feature_columns = build_feature_frame(
        panel=panel,
        feature_config=feature_config,
    )

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
    if latest_rows.empty:
        raise ValueError("Feature engineering produced no usable rows for inference mode.")

    fallback_counts = latest_rows.loc[
        latest_rows["prediction_source"] == "cold_start_fallback", "fallback_level"
    ].value_counts()

    return latest_rows, InferenceFallbackMetadata(
        feature_columns=feature_columns,
        horizon=horizon,
        lags=sorted(set(feature_config.lags)),
        rolling_windows=sorted(set(feature_config.rolling_windows)),
        input_rows=len(panel),
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

    if pd.isna(global_mean):
        raise ValueError(
            "Cannot resolve cold-start fallback because observed_demand has no non-null history."
        )

    val_series = cold_rows["series_id"].map(series_means)
    val_product = cold_rows["product_id"].map(product_means)
    val_third = cold_rows["third_category_id"].map(third_category_means)

    fallback_values = val_series.fillna(val_product).fillna(val_third).fillna(global_mean) * horizon

    conditions = [val_series.notna(), val_product.notna(), val_third.notna()]
    choices = ["series", "product", "third_category"]
    fallback_levels = np.select(conditions, choices, default="global")

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
