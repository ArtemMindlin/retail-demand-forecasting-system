from __future__ import annotations

import pandas as pd

from retail_forecasting.config import DatasetConfig, PreprocessingConfig

STATIC_ID_COLUMNS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]

RAW_COLUMNS = [
    *STATIC_ID_COLUMNS,
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


def load_raw_split(
    dataset_config: DatasetConfig,
    split: str = "train",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a raw dataset split from local cache or Hugging Face.

    Args:
        dataset_config: Dataset-level configuration values.
        split: Dataset split to load.
        columns: Optional subset of columns to read.

    Returns:
        The raw split as a DataFrame.
    """

    selected_columns = columns or RAW_COLUMNS
    split_path = dataset_config.splits[split]
    local_path = dataset_config.local_cache_dir / f"{split}.parquet"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_config.use_cache and local_path.exists():
        return pd.read_parquet(local_path, columns=selected_columns)

    remote_uri = f"hf://datasets/{dataset_config.hf_dataset_id}/{split_path}"
    frame = pd.read_parquet(remote_uri, columns=selected_columns)

    if dataset_config.use_cache:
        frame.to_parquet(local_path, index=False)

    return frame


def prepare_daily_panel(
    frame: pd.DataFrame,
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
) -> pd.DataFrame:
    """Clean and filter the raw split into the daily modeling panel.

    Args:
        frame: Raw split loaded from parquet.
        dataset_config: Dataset-level configuration values.
        preprocessing_config: Preprocessing controls for filtering and filling.

    Returns:
        A cleaned daily panel ready for feature engineering.
    """

    if dataset_config.max_rows:
        panel = frame.head(dataset_config.max_rows).copy()
    else:
        panel = frame.copy()

    panel = panel.rename(
        columns={
            "dt": "date",
            "sale_amount": "observed_demand",
            "stock_hour6_22_cnt": "stockout_hours",
        }
    )
    panel["date"] = pd.to_datetime(panel["date"])

    if preprocessing_config.drop_negative_sales:
        panel = panel.loc[panel["observed_demand"] >= 0].copy()

    panel = panel.drop_duplicates(subset=["store_id", "product_id", "date"])
    panel["series_id"] = panel["store_id"].astype(str) + "_" + panel["product_id"].astype(str)
    panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)

    history_lengths = panel.groupby("series_id")["date"].nunique()
    valid_series = history_lengths.loc[history_lengths >= dataset_config.min_history_days].index
    panel = panel.loc[panel["series_id"].isin(valid_series)].copy()

    if dataset_config.top_n_series:
        top_series = (
            panel.groupby("series_id")["observed_demand"]
            .sum()
            .nlargest(dataset_config.top_n_series)
            .index
        )
        panel = panel.loc[panel["series_id"].isin(top_series)].copy()

    if preprocessing_config.fill_missing_values:
        zero_fill_columns = [
            "holiday_flag",
            "activity_flag",
            "precpt",
            "stockout_hours",
        ]
        for column in zero_fill_columns:
            if column in panel.columns:
                panel[column] = panel[column].fillna(0.0)

        if "discount" in panel.columns:
            panel["discount"] = panel["discount"].fillna(1.0)

        weather_columns = ["avg_temperature", "avg_humidity", "avg_wind_level"]
        for column in weather_columns:
            if column in panel.columns:
                panel[column] = panel[column].fillna(panel[column].median())

    return panel.reset_index(drop=True)


def load_prepared_panel(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    split: str = "train",
) -> pd.DataFrame:
    """Load or build the processed panel for a dataset split.

    Args:
        dataset_config: Dataset-level configuration values.
        preprocessing_config: Preprocessing controls for panel preparation.
        split: Dataset split to materialize.

    Returns:
        The processed panel as a DataFrame.
    """

    target_path = dataset_config.processed_panel_dir / f"{split}.parquet"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_config.use_cache and target_path.exists():
        return pd.read_parquet(target_path)

    raw_frame = load_raw_split(dataset_config=dataset_config, split=split)
    panel = prepare_daily_panel(
        frame=raw_frame,
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
    )
    panel.to_parquet(target_path, index=False)
    return panel
