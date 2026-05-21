from __future__ import annotations

import pandas as pd


def build_dataset_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize the prepared panel at dataset level."""
    series_lengths = panel.groupby("series_id")["date"].nunique()

    return pd.DataFrame(
        [
            {
                "rows": len(panel),
                "unique_series": panel["series_id"].nunique(),
                "date_min": panel["date"].min(),
                "date_max": panel["date"].max(),
                "observed_demand_sum": panel["observed_demand"].sum(),
                "observed_demand_mean": panel["observed_demand"].mean(),
                "observed_demand_std": panel["observed_demand"].std(ddof=0),
                "zero_demand_rate": (panel["observed_demand"] == 0).mean(),
                "stockout_day_rate": (panel["stockout_hours"] > 0).mean(),
                "mean_stockout_hours": panel["stockout_hours"].mean(),
                "median_history_days": series_lengths.median(),
                "min_history_days": series_lengths.min(),
                "max_history_days": series_lengths.max(),
            }
        ]
    )


def build_missingness_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize null rates and uniqueness by column."""
    rows = []
    total_rows = len(panel)

    for column in panel.columns:
        null_count = int(panel[column].isna().sum())
        rows.append(
            {
                "column_name": column,
                "dtype": str(panel[column].dtype),
                "null_count": null_count,
                "null_rate": null_count / total_rows if total_rows else 0.0,
                "n_unique": int(panel[column].nunique(dropna=True)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["null_rate", "column_name"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_series_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize each series for ranking and inspection."""
    series_summary = (
        panel.groupby("series_id")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            history_days=("date", "nunique"),
            observed_demand_sum=("observed_demand", "sum"),
            observed_demand_mean=("observed_demand", "mean"),
            observed_demand_std=("observed_demand", "std"),
            zero_demand_rate=("observed_demand", lambda values: (values == 0).mean()),
            stockout_day_rate=("stockout_hours", lambda values: (values > 0).mean()),
            mean_stockout_hours=("stockout_hours", "mean"),
        )
        .reset_index()
        .sort_values(
            ["observed_demand_sum", "series_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    series_summary["observed_demand_std"] = series_summary["observed_demand_std"].fillna(0.0)
    return series_summary


def build_numeric_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Render descriptive statistics for numeric columns."""
    numeric_panel = panel.select_dtypes(include=["number"])
    if numeric_panel.empty:
        return pd.DataFrame(
            columns=[
                "column_name",
                "count",
                "mean",
                "std",
                "min",
                "p25",
                "median",
                "p75",
                "max",
            ]
        )

    summary = numeric_panel.describe().transpose()
    summary = summary.rename(
        columns={
            "25%": "p25",
            "50%": "median",
            "75%": "p75",
        }
    )
    summary.index.name = "column_name"
    return summary.reset_index()
