"""The panel as a whole: its shape, its gaps, its numeric columns and how they relate."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail_forecasting.utils.plotting import make_grid


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


def build_correlation_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute numeric correlations against observed demand."""
    numeric_columns = panel.select_dtypes(include=["number"]).columns.tolist()
    if "observed_demand" not in numeric_columns:
        return pd.DataFrame(columns=["feature_name", "correlation_with_observed_demand"])

    correlations = panel.loc[:, numeric_columns].corr(numeric_only=True)["observed_demand"]
    correlation_summary = (
        correlations.drop(labels=["observed_demand"])
        .rename("correlation_with_observed_demand")
        .reset_index()
        .rename(columns={"index": "feature_name"})
    )
    correlation_summary["absolute_correlation"] = correlation_summary[
        "correlation_with_observed_demand"
    ].abs()
    return correlation_summary.sort_values(
        ["absolute_correlation", "feature_name"],
        ascending=[False, True],
    ).reset_index(drop=True)


_MEANINGFUL_NUMERIC = [
    "observed_demand",
    "stockout_hours",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


def render_profiling_figures(panel: pd.DataFrame, output_dir: Path) -> None:
    """Every figure about the panel as a whole rather than about its series."""
    _plot_observed_demand_distribution(panel, output_dir / "observed_demand_distribution.png")
    _plot_correlation_heatmap(panel, output_dir / "correlation_heatmap.png")
    _plot_covariate_vs_demand_grid(panel, output_dir / "covariate_vs_demand_grid.png")


def _plot_observed_demand_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        panel["observed_demand"],
        bins=50,
        color="#1f77b4",
        edgecolor="white",
    )
    axes[0].set_xlabel("Observed demand")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Observed demand distribution (linear scale)")

    axes[1].hist(
        panel["observed_demand"],
        bins=50,
        color="#1f77b4",
        edgecolor="white",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Observed demand")
    axes[1].set_ylabel("Frequency (log scale)")
    axes[1].set_title("Observed demand distribution (log scale)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_correlation_heatmap(panel: pd.DataFrame, output_path: Path) -> None:
    cols = [c for c in _MEANINGFUL_NUMERIC if c in panel.columns]
    if not cols:
        return

    variable_numeric = panel[cols].dropna(how="all")
    variable_numeric = variable_numeric.loc[:, variable_numeric.nunique(dropna=True) > 1]
    if variable_numeric.empty:
        return

    correlation = variable_numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        correlation.to_numpy(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(correlation.index)))
    ax.set_yticklabels(correlation.index)
    ax.set_title("Correlation heatmap")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Correlation")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_covariate_vs_demand_grid(panel: pd.DataFrame, output_path: Path) -> None:
    candidate_columns = [
        "discount",
        "avg_temperature",
        "precpt",
        "avg_humidity",
        "avg_wind_level",
    ]
    columns = [c for c in candidate_columns if c in panel.columns]
    if not columns:
        return

    n_bins = 20
    n_cols = 3
    fig, axes_flat = make_grid(len(columns), n_cols, width=14, row_height=4)

    for axis, column in zip(axes_flat, columns, strict=False):
        col_data = panel[[column, "observed_demand"]].dropna()
        col_data = col_data[col_data[column] > col_data[column].quantile(0.01)]
        col_data = col_data[col_data[column] < col_data[column].quantile(0.99)]

        if col_data.empty or col_data[column].nunique() < 2:
            axis.set_visible(False)
            continue

        col_data["_bin"] = pd.cut(col_data[column], bins=n_bins)
        binned = (
            col_data.groupby("_bin", observed=True)["observed_demand"]
            .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)))
            .reset_index()
        )
        binned["x_mid"] = binned["_bin"].apply(lambda b: b.mid)

        axis.fill_between(
            binned["x_mid"],
            binned["mean"] - 1.96 * binned["sem"],
            binned["mean"] + 1.96 * binned["sem"],
            alpha=0.25,
            color="#1f77b4",
            label="95% CI",
        )
        axis.plot(
            binned["x_mid"], binned["mean"], color="#1f77b4", linewidth=2, label="Mean demand"
        )
        axis.set_xlabel(column)
        axis.set_ylabel("Mean observed demand")
        axis.set_title(f"{column} vs observed demand")
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
