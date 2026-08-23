from __future__ import annotations

import hashlib

import pandas as pd

from retail_forecasting.config import DatasetConfig, PreprocessingConfig
from retail_forecasting.data.censorship import OPERATIVE_WINDOW_HOURS

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
    restrict_to_series: set[str] | None = None,
) -> pd.DataFrame:
    """Clean and filter the raw split into the daily modeling panel.

    Args:
        frame: Raw split loaded from parquet.
        dataset_config: Dataset-level configuration values.
        preprocessing_config: Preprocessing controls for filtering and filling.
        restrict_to_series: When given, keep exactly these ``series_id`` values and skip
            both series-selection filters. Required for any split other than ``train``,
            whose series universe must be inherited rather than recomputed -- see the
            comment at the filters below.

    Returns:
        A cleaned daily panel ready for feature engineering.
    """

    # `max_rows` truncates by raw row order, so it cuts whole series off the tail. That is an
    # acceptable way to shrink train (it defines the universe, and the cut lands before the
    # series filters), but destructive on an inherited split: it would drop rows of series the
    # holdout is supposed to cover, leaving some with fewer days than others. The series
    # restriction already bounds the size there (50 series x 7 days), so the cap is not needed.
    if dataset_config.max_rows and restrict_to_series is None:
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

    # Both filters below define the series UNIVERSE from the split's own rows, which is only
    # meaningful for train. Applied to a short forward split such as the official 7-day eval
    # they are actively wrong, in two different ways:
    #   * `min_history_days` (70) counts days within the split, so a 7-day split loses every
    #     series and the panel comes out empty. The history those series need lives in train.
    #   * `top_n_series` ranks by demand summed over the split, so the top 50 of a 7 days
    #     window is a DIFFERENT set of series than the top 50 of train's 90 days -- the more
    #     insidious failure, since it yields a populated panel of the wrong series and the
    #     holdout would score a model on series it never trained on.
    # A non-train split therefore inherits train's series set instead of recomputing one.
    if restrict_to_series is not None:
        panel = panel[panel["series_id"].isin(restrict_to_series)].copy()
    else:
        history_lengths = panel.groupby("series_id")["date"].count()
        valid_series = history_lengths[history_lengths >= dataset_config.min_history_days].index
        panel = panel[panel["series_id"].isin(valid_series)].copy()

        if dataset_config.top_n_series:
            top_series = (
                panel.groupby("series_id")["observed_demand"]
                .sum()
                .nlargest(dataset_config.top_n_series)
                .index
            )
            panel = panel[panel["series_id"].isin(top_series)].copy()

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

        if "stockout_hours" in panel.columns:
            panel["stockout_hours"] = panel["stockout_hours"].clip(0.0, OPERATIVE_WINDOW_HOURS)

        if "discount" in panel.columns:
            panel["discount"] = panel["discount"].fillna(1.0)

        # Weather covariates are imputed causally within each series: a
        # forward-fill carries the last observed reading (past information only),
        # a backward-fill covers leading gaps at the very start of a series, and
        # a global median is the final fallback for series with no reading at
        # all. This avoids the look-ahead bias of a split-wide median, which
        # would let future observations inform the imputed values of past rows.
        weather_columns = ["avg_temperature", "avg_humidity", "avg_wind_level"]
        for column in weather_columns:
            if column not in panel.columns:
                continue
            filled = panel.groupby("series_id")[column].ffill()
            filled = filled.groupby(panel["series_id"]).bfill()
            panel[column] = filled.fillna(filled.median())

    return panel.reset_index(drop=True)


def panel_cache_filename(dataset_config: DatasetConfig, split: str) -> str:
    """File name ``load_prepared_panel`` reads and writes for ``split``.

    Public so that pre-built panels (the OPS simulation split) can be written under
    the name the loader will look for, instead of duplicating the convention.
    """
    return f"{split}_{_panel_cache_key(dataset_config)}.parquet"


def _panel_cache_key(dataset_config: DatasetConfig) -> str:
    """Short fingerprint of the settings that change the shape of the prepared panel.

    Keying the cache on the split name alone made every config share one file, so
    running the 50-series subset after the 500-series one silently reused the
    500-series panel and ``top_n_series`` had no effect. The base subset and the
    scale validation answer different questions and must not collapse into
    whichever panel happens to be on disk.
    """
    shape = (
        dataset_config.top_n_series,
        dataset_config.min_history_days,
        dataset_config.max_rows,
        dataset_config.horizon,
    )
    return hashlib.sha1(repr(shape).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def _sorted_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """The canonical row order, applied on every path out of ``load_prepared_panel``.

    Ordering used to be inherited rather than guaranteed: ``prepare_daily_panel`` sorts, and
    the filters after it preserve that order, so the rebuild path came out sorted by accident
    of construction while the cache path returned whatever order the parquet happened to hold.
    Callers made up the difference by re-sorting, which is the shape of an assumption nobody
    owns. Cheap enough not to think about: 4ms against the 4ms the read itself costs.
    """
    return panel.sort_values(["series_id", "date"]).reset_index(drop=True)


def load_prepared_panel(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    split: str = "train",
) -> pd.DataFrame:
    """Load or build the processed panel for a dataset split.

    A split other than ``train`` inherits train's series universe, which means loading the
    train panel first (recursion bottoms out immediately, since train inherits nothing).
    Pre-built panels written under ``panel_cache_filename`` -- the OPS backtest split -- are
    returned from cache before any of that, so this path never touches them.

    Args:
        dataset_config: Dataset-level configuration values.
        preprocessing_config: Preprocessing controls for panel preparation.
        split: Dataset split to materialize.

    Returns:
        The processed panel as a DataFrame, sorted by ``series_id`` then ``date``.

    Raises:
        ValueError: If a non-train split prepares to zero rows. Silently returning an empty
            holdout is what let the eval evaluation go missing from every run to date.
    """

    target_path = dataset_config.processed_panel_dir / panel_cache_filename(dataset_config, split)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_config.use_cache and target_path.exists():
        cached = pd.read_parquet(target_path)
        # An empty cached non-train panel is never a legitimate cache entry: it is the artifact
        # of the universe filters this function now bypasses. Rebuild rather than serve it, so
        # the fix applies without invalidating the cache key -- bumping the key would orphan the
        # pre-built OPS split, whose `.built` sentinel would stop `make simulate` regenerating it.
        if split == "train" or not cached.empty:
            return _sorted_panel(cached)

    restrict_to_series: set[str] | None = None
    if split != "train":
        train_panel = load_prepared_panel(
            dataset_config=dataset_config,
            preprocessing_config=preprocessing_config,
            split="train",
        )
        restrict_to_series = set(train_panel["series_id"].unique())

    raw_frame = load_raw_split(dataset_config=dataset_config, split=split)
    panel = prepare_daily_panel(
        frame=raw_frame,
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        restrict_to_series=restrict_to_series,
    )

    if split != "train" and panel.empty:
        raise ValueError(
            f"The prepared '{split}' panel is empty: none of the {len(restrict_to_series or ())} "
            f"series selected from train appear in the raw '{split}' split "
            f"({len(raw_frame):,} raw rows). A holdout that silently evaluates nothing is worse "
            "than a failed run, so this raises instead of returning an empty frame."
        )

    panel.to_parquet(target_path, index=False)
    return _sorted_panel(panel)
