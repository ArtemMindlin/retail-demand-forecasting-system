from __future__ import annotations

import pandas as pd


def build_stockout_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize stockout frequency and severity at dataset level."""
    stockout_mask = panel["stockout_hours"] > 0
    stockout_panel = panel.loc[stockout_mask]

    return pd.DataFrame(
        [
            {
                "stockout_rows": int(stockout_mask.sum()),
                "stockout_row_rate": stockout_mask.mean(),
                "mean_stockout_hours_all_rows": panel["stockout_hours"].mean(),
                "mean_stockout_hours_stockout_rows": (
                    stockout_panel["stockout_hours"].mean()
                    if not stockout_panel.empty
                    else 0.0
                ),
                "zero_demand_rate_stockout_rows": (
                    (stockout_panel["observed_demand"] == 0).mean()
                    if not stockout_panel.empty
                    else 0.0
                ),
                "zero_demand_rate_non_stockout_rows": (
                    (panel.loc[~stockout_mask, "observed_demand"] == 0).mean()
                    if (~stockout_mask).any()
                    else 0.0
                ),
            }
        ]
    )


def build_stockout_by_series_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize stockout behavior by series."""
    return (
        panel.groupby("series_id")
        .agg(
            observations=("stockout_hours", "size"),
            stockout_days=("stockout_hours", lambda values: int((values > 0).sum())),
            stockout_day_rate=("stockout_hours", lambda values: (values > 0).mean()),
            mean_stockout_hours=("stockout_hours", "mean"),
            max_stockout_hours=("stockout_hours", "max"),
            observed_demand_mean=("observed_demand", "mean"),
        )
        .reset_index()
        .sort_values(["stockout_day_rate", "series_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_stockout_demand_bands(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare demand under stockout intensity bands."""
    banded = panel.assign(
        stockout_band=pd.cut(
            panel["stockout_hours"],
            bins=[-0.01, 0.0, 2.0, 6.0, float("inf")],
            labels=["0", "0-2", "3-6", "7+"],
        )
    )

    return (
        banded.groupby("stockout_band", observed=False)
        .agg(
            observations=("observed_demand", "size"),
            observed_demand_mean=("observed_demand", "mean"),
            observed_demand_median=("observed_demand", "median"),
            stockout_hours_mean=("stockout_hours", "mean"),
        )
        .reset_index()
    )
