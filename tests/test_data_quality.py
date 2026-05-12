from __future__ import annotations

import pandas as pd
import pytest

from retail_forecasting.config import DataQualityConfig, ProjectConfig, Settings
from retail_forecasting.data.quality import (
    DataQualityError,
    raise_on_blocking_data_quality,
    validate_prepared_panel,
)
from tests import make_synthetic_panel


def test_validate_prepared_panel_blocks_duplicate_series_dates() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    duplicate_row = panel.iloc[[0]].copy()
    broken_panel = pd.concat([panel, duplicate_row], ignore_index=True)
    settings = Settings()

    report = validate_prepared_panel(broken_panel, settings)

    assert report.passed is False
    assert report.blocking_error_count > 0
    assert report.blocking_errors[0].code == "duplicate_series_date_rows"


def test_validate_prepared_panel_warns_on_high_missingness() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    panel["discount"] = pd.NA
    settings = Settings(
        data_quality=DataQualityConfig(max_missing_fraction_warning=0.01)
    )

    report = validate_prepared_panel(panel, settings)

    assert report.warning_count > 0
    assert report.warnings[0].code == "high_missingness"


def test_validate_prepared_panel_blocks_stale_operational_data() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    panel["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        panel.groupby("series_id").cumcount(), unit="D"
    )
    settings = Settings(
        project=ProjectConfig(run_mode="score_daily"),
        data_quality=DataQualityConfig(max_data_age_days=1),
    )

    report = validate_prepared_panel(panel, settings)

    assert report.passed is False
    assert any(issue.code == "stale_data" for issue in report.blocking_errors)


def test_raise_on_blocking_data_quality_raises_error() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    broken_panel = panel.drop(columns=["observed_demand"])
    settings = Settings()

    report = validate_prepared_panel(broken_panel, settings)

    with pytest.raises(DataQualityError, match="Blocking data-quality checks failed"):
        raise_on_blocking_data_quality(report)
