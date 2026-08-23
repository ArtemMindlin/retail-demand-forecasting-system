"""The official eval split as an external holdout (invariant 14).

This path had no coverage at all, which is how it went unnoticed that the holdout prepared to
zero rows under every experiment config and was then skipped in silence. These tests pin both
halves: the series universe is inherited rather than recomputed, and a holdout that vanishes
raises instead of being dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retail_forecasting.config import DatasetConfig, PreprocessingConfig, ReportingConfig, Settings
from retail_forecasting.data.dataset import (
    load_prepared_panel,
    panel_cache_filename,
    prepare_daily_panel,
)
from retail_forecasting.forecasting.pipeline import run_experiment_from_frame
from tests import make_synthetic_panel

RAW_RENAMES = {
    "date": "dt",
    "observed_demand": "sale_amount",
    "stockout_hours": "stock_hour6_22_cnt",
}


def _make_raw_frame(series_ids: list[tuple[int, int]], num_days: int, start: str) -> pd.DataFrame:
    """A raw-shaped split: raw column names, one row per series-day.

    Demand is deliberately ordered so the LAST series is the highest seller, which is what
    lets a test show that recomputing ``top_n_series`` per split picks a different universe.
    """
    dates = pd.date_range(start, periods=num_days, freq="D")
    rows = []
    for rank, (store_id, product_id) in enumerate(series_ids):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "city_id": 1,
                    "store_id": store_id,
                    "management_group_id": 1,
                    "first_category_id": 1,
                    "second_category_id": 1,
                    "third_category_id": store_id,
                    "product_id": product_id,
                    "dt": date,
                    "sale_amount": 1.0 + rank * 10.0,
                    "stock_hour6_22_cnt": float(index % 3),
                    "discount": 1.0,
                    "holiday_flag": 0,
                    "activity_flag": 0,
                    "precpt": 0.0,
                    "avg_temperature": 15.0,
                    "avg_humidity": 50.0,
                    "avg_wind_level": 3.0,
                }
            )
    return pd.DataFrame(rows)


def test_restricted_split_keeps_short_series_that_min_history_days_would_drop() -> None:
    """A 7-day forward split holds 7 days per series; ``min_history_days`` is 70.

    Counting history within the split is what emptied the panel: the history those series
    need lives in train, not here.
    """
    raw = _make_raw_frame([(1, 101), (2, 102)], num_days=7, start="2024-06-26")
    config = DatasetConfig(top_n_series=2, min_history_days=70, horizon=7)

    unrestricted = prepare_daily_panel(raw, config, PreprocessingConfig())
    restricted = prepare_daily_panel(
        raw, config, PreprocessingConfig(), restrict_to_series={"1_101", "2_102"}
    )

    assert unrestricted.empty, "the old behaviour this test documents: the whole split is lost"
    assert set(restricted["series_id"]) == {"1_101", "2_102"}
    assert len(restricted) == 14


def test_restricted_split_does_not_recompute_the_top_n_series_universe() -> None:
    """Ranking by demand summed over the split picks a different universe than train's.

    The more insidious half of the defect: it yields a POPULATED panel of the wrong series,
    so the holdout would score a model on series it never trained on.
    """
    raw = _make_raw_frame([(1, 101), (2, 102), (3, 103)], num_days=7, start="2024-06-26")
    config = DatasetConfig(top_n_series=1, min_history_days=7, horizon=7)

    own_universe = prepare_daily_panel(raw, config, PreprocessingConfig())
    inherited = prepare_daily_panel(
        raw, config, PreprocessingConfig(), restrict_to_series={"1_101"}
    )

    assert set(own_universe["series_id"]) == {"3_103"}, "highest seller within the split"
    assert set(inherited["series_id"]) == {"1_101"}, "train's universe, regardless of split sales"


def test_unrestricted_preparation_still_applies_both_universe_filters() -> None:
    """The train path must be untouched by the holdout fix."""
    raw = pd.concat(
        [
            _make_raw_frame([(1, 101), (2, 102)], num_days=90, start="2024-03-28"),
            _make_raw_frame([(9, 109)], num_days=10, start="2024-03-28"),
        ],
        ignore_index=True,
    )
    config = DatasetConfig(top_n_series=1, min_history_days=70, horizon=7)

    panel = prepare_daily_panel(raw, config, PreprocessingConfig())

    assert "9_109" not in set(panel["series_id"]), "min_history_days still drops short series"
    assert set(panel["series_id"]) == {"2_102"}, "top_n_series still ranks the survivors"


def _write_raw_splits(tmp_path: Path, eval_series: list[tuple[int, int]]) -> DatasetConfig:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_series = [(1, 101), (2, 102), (3, 103)]
    _make_raw_frame(train_series, num_days=90, start="2024-03-28").to_parquet(
        raw_dir / "train.parquet", index=False
    )
    _make_raw_frame(eval_series, num_days=7, start="2024-06-26").to_parquet(
        raw_dir / "eval.parquet", index=False
    )
    return DatasetConfig(
        local_cache_dir=raw_dir,
        processed_panel_dir=tmp_path / "processed",
        top_n_series=2,
        min_history_days=70,
        horizon=7,
    )


def test_eval_split_inherits_the_train_series_universe(tmp_path: Path) -> None:
    config = _write_raw_splits(tmp_path, eval_series=[(1, 101), (2, 102), (3, 103)])
    preprocessing = PreprocessingConfig()

    train = load_prepared_panel(config, preprocessing, split="train")
    holdout = load_prepared_panel(config, preprocessing, split="eval")

    assert set(holdout["series_id"]) == set(train["series_id"])
    assert len(set(train["series_id"])) == 2, "top_n_series=2 selected the universe on train"
    assert holdout["date"].min() > train["date"].max(), "a forward holdout, not an overlap"


def test_eval_split_raises_when_it_shares_no_series_with_train(tmp_path: Path) -> None:
    config = _write_raw_splits(tmp_path, eval_series=[(7, 107), (8, 108)])

    with pytest.raises(ValueError, match="prepared 'eval' panel is empty"):
        load_prepared_panel(config, PreprocessingConfig(), split="eval")


def test_an_empty_cached_holdout_is_rebuilt_rather_than_served(tmp_path: Path) -> None:
    """Stale empty caches are the artifact of the bug, so they must not survive the fix."""

    config = _write_raw_splits(tmp_path, eval_series=[(1, 101), (2, 102), (3, 103)])
    preprocessing = PreprocessingConfig()
    load_prepared_panel(config, preprocessing, split="train")

    cache_path = config.processed_panel_dir / panel_cache_filename(config, "eval")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    load_prepared_panel(config, preprocessing, split="eval").head(0).to_parquet(
        cache_path, index=False
    )

    assert pd.read_parquet(cache_path).empty, "precondition: a stale empty cache on disk"
    assert not load_prepared_panel(config, preprocessing, split="eval").empty


def test_an_empty_holdout_frame_fails_the_run_instead_of_being_skipped(tmp_path: Path) -> None:
    """The second silent drop: a requested holdout that vanishes in feature engineering."""
    panel = make_synthetic_panel(num_series=3, num_days=90)
    settings = Settings(
        dataset=DatasetConfig(top_n_series=3, min_history_days=70, horizon=7),
        reporting=ReportingConfig(output_dir=tmp_path, run_name="empty_holdout", make_plots=False),
    )

    with pytest.raises(ValueError, match="supervised frame is empty"):
        run_experiment_from_frame(
            panel, settings, holdout_panel=panel.head(0), save_artifacts=False
        )


def test_no_holdout_requested_is_not_an_error(tmp_path: Path) -> None:
    """``None`` means the caller did not ask for a holdout; that path must stay silent."""
    panel = make_synthetic_panel(num_series=3, num_days=90)
    settings = Settings(
        dataset=DatasetConfig(top_n_series=3, min_history_days=70, horizon=7),
        reporting=ReportingConfig(output_dir=tmp_path, run_name="no_holdout", make_plots=False),
    )

    artifacts = run_experiment_from_frame(panel, settings, holdout_panel=None, save_artifacts=False)

    assert not artifacts.predictions.empty


def test_a_cached_panel_comes_back_in_canonical_order(tmp_path: Path) -> None:
    """The cache path returned the parquet's own order, and callers re-sorted to make up for it.

    Row order is a documented property of the prepared panel, so the loader owns it on every
    path rather than leaving it to whoever wrote the file. The OPS plane writes its split
    directly under the cache name, which is exactly a path that never went through the
    rebuilding code.
    """
    config = _write_raw_splits(tmp_path, eval_series=[(1, 101), (2, 102)])
    preprocessing = PreprocessingConfig()

    panel = load_prepared_panel(config, preprocessing, split="train")
    target = config.processed_panel_dir / panel_cache_filename(config, "train")
    shuffled = panel.sample(frac=1.0, random_state=0)
    shuffled.to_parquet(target, index=False)
    assert not shuffled["series_id"].is_monotonic_increasing, "el fixture debe estar desordenado"

    reloaded = load_prepared_panel(config, preprocessing, split="train")

    expected = panel.sort_values(["series_id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(reloaded, expected)
