from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_quality import (
    DataQualityError,
    DataQualityIssue,
    DataQualityReport,
)

REQUIRED_PANEL_COLUMNS = {
    "date",
    "series_id",
    "observed_demand",
    "stockout_hours",
}

KEY_COLUMNS = ["date", "series_id", "observed_demand"]


def validate_prepared_panel(panel: pd.DataFrame, settings: Settings) -> DataQualityReport:
    warnings: list[DataQualityIssue] = []
    blocking_errors: list[DataQualityIssue] = []

    missing_columns = sorted(REQUIRED_PANEL_COLUMNS - set(panel.columns))
    if missing_columns:
        blocking_errors.append(
            DataQualityIssue(
                severity="blocking",
                code="missing_required_columns",
                message=(
                    f"Prepared panel is missing required columns: {', '.join(missing_columns)}."
                ),
            )
        )
        return _build_report(panel, settings, warnings, blocking_errors)

    duplicate_rows = int(panel.duplicated(subset=["series_id", "date"]).sum())
    if duplicate_rows > 0:
        blocking_errors.append(
            DataQualityIssue(
                severity="blocking",
                code="duplicate_series_date_rows",
                message=(
                    f"Prepared panel contains duplicate `series_id + date` rows: {duplicate_rows}."
                ),
            )
        )

    null_counts_by_key = {column: int(panel[column].isna().sum()) for column in KEY_COLUMNS}
    null_key_counts = {column: count for column, count in null_counts_by_key.items() if count > 0}
    if null_key_counts:
        blocking_errors.append(
            DataQualityIssue(
                severity="blocking",
                code="null_key_columns",
                message=(
                    "Prepared panel contains nulls in key columns: "
                    + ", ".join(
                        f"{column}={count}" for column, count in sorted(null_key_counts.items())
                    )
                    + "."
                ),
            )
        )

    parsed_dates = pd.to_datetime(panel["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates > 0:
        blocking_errors.append(
            DataQualityIssue(
                severity="blocking",
                code="invalid_date_format",
                message=f"Prepared panel contains {invalid_dates} rows with unparseable dates.",
            )
        )

    missing_fractions = panel.isna().mean(numeric_only=False)
    warned_columns = [
        f"{column}={fraction:.3f}"
        for column, fraction in missing_fractions.items()
        if fraction > settings.data_quality.max_missing_fraction_warning
        and column not in KEY_COLUMNS
    ]
    if warned_columns:
        warnings.append(
            DataQualityIssue(
                severity="warning",
                code="high_missingness",
                message=(
                    "Prepared panel columns exceed the configured missingness warning threshold: "
                    + ", ".join(warned_columns)
                    + "."
                ),
            )
        )

    if (
        settings.data_quality.max_data_age_days is not None
        and settings.project.run_mode != "experiment"
        and invalid_dates == 0
        and not parsed_dates.empty
    ):
        latest_date = parsed_dates.max()
        data_age_days = (datetime.now(UTC).date() - latest_date.date()).days
        if data_age_days > settings.data_quality.max_data_age_days:
            blocking_errors.append(
                DataQualityIssue(
                    severity="blocking",
                    code="stale_data",
                    message=(
                        "Prepared panel is older than the configured freshness limit: "
                        f"{data_age_days} days > {settings.data_quality.max_data_age_days}."
                    ),
                )
            )

    return _build_report(panel, settings, warnings, blocking_errors)


def raise_on_blocking_data_quality(report: DataQualityReport) -> None:
    if report.passed:
        return
    details = " ".join(issue.message for issue in report.blocking_errors)
    raise DataQualityError(f"Blocking data-quality checks failed. {details}")


def _build_report(
    panel: pd.DataFrame,
    settings: Settings,
    warnings: list[DataQualityIssue],
    blocking_errors: list[DataQualityIssue],
) -> DataQualityReport:
    date_min = None
    date_max = None
    if "date" in panel.columns and not panel.empty:
        parsed_dates = pd.to_datetime(panel["date"], errors="coerce")
        if parsed_dates.notna().any():
            date_min = parsed_dates.min().isoformat()
            date_max = parsed_dates.max().isoformat()

    return DataQualityReport(
        run_mode=settings.project.run_mode,
        checked_rows=len(panel),
        checked_series=(int(panel["series_id"].count()) if "series_id" in panel.columns else 0),
        date_min=date_min,
        date_max=date_max,
        warning_count=len(warnings),
        blocking_error_count=len(blocking_errors),
        warnings=warnings,
        blocking_errors=blocking_errors,
        passed=not blocking_errors,
    )
