from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail_forecasting.utils.io import ensure_directory

TOP_SERIES_PLOT_COUNT = 12
MAX_HEATMAP_SERIES = 120
SCATTER_SAMPLE_SIZE = 5000
REPRESENTATIVE_SERIES_COUNT = 12


def render_eda_plots(
    panel: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Render a comprehensive static plot set for EDA runs."""
    target_dir = ensure_directory(output_dir)

    _plot_observed_demand_distribution(
        panel,
        target_dir / "observed_demand_distribution.png",
    )
    _plot_observed_demand_boxplot_top_series(
        panel,
        series_summary,
        target_dir / "observed_demand_boxplot_top_series.png",
    )
    _plot_stockout_hours_distribution(
        panel,
        target_dir / "stockout_hours_distribution.png",
    )
    _plot_weekday_demand_profile(
        weekday_summary,
        target_dir / "weekday_demand_profile.png",
    )
    _plot_top_series_total_demand(
        series_summary,
        target_dir / "top_series_total_demand.png",
    )
    _plot_coverage_heatmap(
        panel,
        series_summary,
        target_dir / "coverage_heatmap.png",
    )
    _plot_zero_demand_rate_by_series(
        series_summary,
        target_dir / "zero_demand_rate_by_series.png",
    )
    _plot_stockout_band_demand(
        panel,
        target_dir / "stockout_band_demand.png",
    )
    _plot_stockout_vs_demand_scatter(
        panel,
        target_dir / "stockout_vs_demand_scatter.png",
    )
    _plot_correlation_heatmap(
        panel,
        target_dir / "correlation_heatmap.png",
    )
    _plot_covariate_vs_demand_grid(
        panel,
        target_dir / "covariate_vs_demand_grid.png",
    )
    _plot_representative_series_panels(
        panel,
        series_summary,
        target_dir / "representative_series_panels.png",
    )


def _plot_observed_demand_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        panel["observed_demand"],
        bins=30,
        color="#1f77b4",
        edgecolor="white",
    )
    axes[0].set_xlabel("Observed demand")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Observed demand distribution")

    strictly_positive = panel.loc[panel["observed_demand"] > 0, "observed_demand"]
    axes[1].hist(
        strictly_positive,
        bins=30,
        color="#17becf",
        edgecolor="white",
    )
    axes[1].set_xlabel("Observed demand > 0")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Positive-demand distribution")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_observed_demand_boxplot_top_series(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    top_series = series_summary.head(TOP_SERIES_PLOT_COUNT)["series_id"].tolist()
    subset = panel.loc[panel["series_id"].isin(top_series)].copy()
    if subset.empty:
        return

    ordered_labels = top_series
    distributions = [
        subset.loc[subset["series_id"] == series_id, "observed_demand"].to_numpy()
        for series_id in ordered_labels
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        distributions,
        tick_labels=ordered_labels,
        patch_artist=True,
        boxprops={"facecolor": "#aec7e8"},
        medianprops={"color": "#d62728"},
    )
    ax.set_xlabel("Series")
    ax.set_ylabel("Observed demand")
    ax.set_title("Observed demand by top-demand series")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stockout_hours_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        panel["stockout_hours"],
        bins=24,
        color="#d62728",
        edgecolor="white",
    )
    axes[0].set_xlabel("Stockout hours")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Stockout hours distribution")

    positive_stockout = panel.loc[panel["stockout_hours"] > 0, "stockout_hours"]
    axes[1].hist(
        positive_stockout,
        bins=24,
        color="#ff9896",
        edgecolor="white",
    )
    axes[1].set_xlabel("Positive stockout hours")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Positive stockout-hour distribution")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_weekday_demand_profile(
    weekday_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        weekday_summary["weekday_name"],
        weekday_summary["observed_demand_mean"],
        marker="o",
        linewidth=2,
        color="#2ca02c",
        label="Mean",
    )
    ax.plot(
        weekday_summary["weekday_name"],
        weekday_summary["observed_demand_median"],
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Median",
    )
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Observed demand")
    ax.set_title("Weekly demand profile")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_top_series_total_demand(
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    top_series = series_summary.head(10)
    if top_series.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        top_series["series_id"][::-1],
        top_series["observed_demand_sum"][::-1],
        color="#9467bd",
    )
    ax.set_xlabel("Total observed demand")
    ax.set_ylabel("Series")
    ax.set_title("Top 10 series by observed demand")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_coverage_heatmap(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_series = series_summary.head(MAX_HEATMAP_SERIES)["series_id"].tolist()
    coverage_frame = (
        panel.loc[panel["series_id"].isin(selected_series), ["series_id", "date"]]
        .assign(present=1.0)
        .pivot(index="series_id", columns="date", values="present")
        .fillna(0.0)
    )
    if coverage_frame.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    image = ax.imshow(
        coverage_frame.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Date index")
    ax.set_ylabel("Series")
    ax.set_title(f"Coverage heatmap (top {len(selected_series)} series by demand)")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Observed row")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_zero_demand_rate_by_series(
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    zero_demand_rank = series_summary.sort_values(
        ["zero_demand_rate", "series_id"],
        ascending=[False, True],
    ).head(20)
    if zero_demand_rank.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        zero_demand_rank["series_id"][::-1],
        zero_demand_rank["zero_demand_rate"][::-1],
        color="#8c564b",
    )
    ax.set_xlabel("Zero-demand rate")
    ax.set_ylabel("Series")
    ax.set_title("Top 20 most intermittent series")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stockout_band_demand(panel: pd.DataFrame, output_path: Path) -> None:
    stockout_band_frame = (
        panel.assign(
            stockout_band=pd.cut(
                panel["stockout_hours"],
                bins=[-0.01, 0.0, 2.0, 6.0, float("inf")],
                labels=["0", "0-2", "3-6", "7+"],
            )
        )
        .groupby("stockout_band", observed=False)["observed_demand"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(
        stockout_band_frame["stockout_band"].astype(str),
        stockout_band_frame["mean"],
        color="#ff7f0e",
    )
    axes[0].set_xlabel("Stockout band")
    axes[0].set_ylabel("Mean observed demand")
    axes[0].set_title("Mean demand by stockout band")

    axes[1].bar(
        stockout_band_frame["stockout_band"].astype(str),
        stockout_band_frame["count"],
        color="#bcbd22",
    )
    axes[1].set_xlabel("Stockout band")
    axes[1].set_ylabel("Observations")
    axes[1].set_title("Observation count by stockout band")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stockout_vs_demand_scatter(panel: pd.DataFrame, output_path: Path) -> None:
    sampled = _sample_panel(panel)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        sampled["stockout_hours"],
        sampled["observed_demand"],
        alpha=0.15,
        s=12,
        color="#1f77b4",
    )

    grouped = (
        panel.groupby("stockout_hours", as_index=False)["observed_demand"]
        .mean()
        .rename(columns={"observed_demand": "observed_demand_mean"})
    )
    ax.plot(
        grouped["stockout_hours"],
        grouped["observed_demand_mean"],
        color="#d62728",
        linewidth=2,
        label="Mean by stockout hour",
    )
    ax.set_xlabel("Stockout hours")
    ax.set_ylabel("Observed demand")
    ax.set_title("Observed demand vs stockout hours")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_correlation_heatmap(panel: pd.DataFrame, output_path: Path) -> None:
    numeric = panel.select_dtypes(include=["number"])
    if numeric.empty:
        return

    variable_numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if variable_numeric.empty:
        return

    correlation = variable_numeric.corr(numeric_only=True)
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
    columns = [column for column in candidate_columns if column in panel.columns]
    if not columns:
        return

    sampled = _sample_panel(panel)
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
    axes_flat = axes.flatten()

    for axis, column in zip(axes_flat, columns, strict=False):
        axis.scatter(
            sampled[column],
            sampled["observed_demand"],
            alpha=0.15,
            s=10,
            color="#1f77b4",
        )
        binned = _binned_average(sampled, feature_column=column)
        if not binned.empty:
            axis.plot(
                binned[column],
                binned["observed_demand_mean"],
                color="#d62728",
                linewidth=2,
            )
        axis.set_xlabel(column)
        axis.set_ylabel("Observed demand")
        axis.set_title(f"{column} vs observed demand")

    for axis in axes_flat[len(columns) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_representative_series_panels(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_series = series_summary.head(REPRESENTATIVE_SERIES_COUNT)[
        "series_id"
    ].tolist()
    if not selected_series:
        return

    subset = panel.loc[panel["series_id"].isin(selected_series)].copy()
    n_cols = 3
    n_rows = int(np.ceil(len(selected_series) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.6 * n_rows), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for axis, series_id in zip(axes_flat, selected_series, strict=False):
        series_frame = subset.loc[subset["series_id"] == series_id].sort_values("date")
        axis.plot(
            series_frame["date"],
            series_frame["observed_demand"],
            color="#1f77b4",
            linewidth=1.8,
        )
        axis.fill_between(
            series_frame["date"],
            0,
            series_frame["stockout_hours"],
            color="#d62728",
            alpha=0.18,
        )
        axis.set_title(series_id)
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("Demand")

    for axis in axes_flat[len(selected_series) :]:
        axis.axis("off")

    fig.suptitle(
        "Representative series panels (line: demand, shaded: stockout hours)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _sample_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if len(panel) <= SCATTER_SAMPLE_SIZE:
        return panel
    return panel.sample(SCATTER_SAMPLE_SIZE, random_state=42)


def _binned_average(
    panel: pd.DataFrame,
    feature_column: str,
    bins: int = 20,
) -> pd.DataFrame:
    feature = panel[feature_column]
    if feature.nunique(dropna=True) <= 1:
        return pd.DataFrame(columns=[feature_column, "observed_demand_mean"])

    quantile_bins = min(bins, feature.nunique(dropna=True))
    try:
        binned = pd.qcut(feature, q=quantile_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=[feature_column, "observed_demand_mean"])

    grouped = (
        panel.assign(_bin=binned)
        .groupby("_bin", observed=False)
        .agg(
            feature_center=(feature_column, "mean"),
            observed_demand_mean=("observed_demand", "mean"),
        )
        .dropna()
        .reset_index(drop=True)
        .rename(columns={"feature_center": feature_column})
    )
    return grouped
