from __future__ import annotations

import pandas as pd


def build_temporal_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize panel-wide temporal coverage and continuity."""
    date_span_days = int((panel["date"].max() - panel["date"].min()).days) + 1
    expected_rows = date_span_days * panel["series_id"].nunique()

    return pd.DataFrame(
        [
            {
                "date_min": panel["date"].min(),
                "date_max": panel["date"].max(),
                "date_span_days": date_span_days,
                "observed_rows": len(panel),
                "expected_rows_full_grid": expected_rows,
                "coverage_rate_full_grid": len(panel) / expected_rows if expected_rows else 0.0,
                "duplicate_series_date_rows": int(panel.duplicated(["series_id", "date"]).sum()),
            }
        ]
    )


def build_weekday_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize weekly seasonality on the prepared panel."""
    weekday_panel = panel.assign(
        weekday=panel["date"].dt.dayofweek,
        weekday_name=panel["date"].dt.day_name(),
    )

    return (
        weekday_panel.groupby(["weekday", "weekday_name"])
        .agg(
            observations=("observed_demand", "size"),
            observed_demand_mean=("observed_demand", "mean"),
            observed_demand_median=("observed_demand", "median"),
            stockout_day_rate=("stockout_hours", lambda values: (values > 0).mean()),
            mean_stockout_hours=("stockout_hours", "mean"),
        )
        .reset_index()
        .sort_values("weekday")
        .reset_index(drop=True)
    )


def build_series_gap_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing calendar gaps within each series history."""
    rows = []

    for series_id, series_frame in panel.groupby("series_id", sort=False):
        ordered = series_frame.sort_values("date")
        day_deltas = ordered["date"].diff().dt.days.dropna()
        max_gap_days = int(day_deltas.max()) if not day_deltas.empty else 1
        missing_days_within_span = (
            int((day_deltas - 1).clip(lower=0).sum()) if not day_deltas.empty else 0
        )
        rows.append(
            {
                "series_id": series_id,
                "history_days": ordered["date"].nunique(),
                "start_date": ordered["date"].min(),
                "end_date": ordered["date"].max(),
                "max_gap_days": max_gap_days,
                "missing_days_within_span": missing_days_within_span,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["missing_days_within_span", "series_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
