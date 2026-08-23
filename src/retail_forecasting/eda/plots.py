from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail_forecasting.eda.category_heatmap import render_category_seasonality_heatmaps
from retail_forecasting.eda.stockout import render_stockout_figures
from retail_forecasting.eda.temporal import render_temporal_figures

TOP_SERIES_PLOT_COUNT = 12
REPRESENTATIVE_SERIES_COUNT = 12


def render_eda_plots(
    panel: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    series_summary: pd.DataFrame,
    stockout_demand_bands: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Render a comprehensive static plot set for EDA runs."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    _plot_observed_demand_distribution(
        panel,
        target_dir / "observed_demand_distribution.png",
    )
    _plot_observed_demand_boxplot_top_series(
        panel,
        series_summary,
        target_dir / "observed_demand_boxplot_top_series.png",
    )
    _plot_top_series_total_demand(
        series_summary,
        target_dir / "top_series_total_demand.png",
    )
    _plot_zero_demand_rate_by_series(
        series_summary,
        target_dir / "zero_demand_rate_by_series.png",
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
    render_stockout_figures(panel, stockout_demand_bands, target_dir)
    render_temporal_figures(panel, weekday_summary, series_summary, target_dir)
    render_category_seasonality_heatmaps(panel, target_dir)


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


def _plot_observed_demand_boxplot_top_series(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    # Split the plot budget into three equal bands: top / middle / bottom volume.
    per_group = TOP_SERIES_PLOT_COUNT // 3
    top = series_summary.head(per_group)["series_id"].tolist()
    mid_start = len(series_summary) // 2
    mid = series_summary.iloc[mid_start : mid_start + per_group]["series_id"].tolist()
    bottom = series_summary.tail(per_group)["series_id"].tolist()

    subset = panel.loc[panel["series_id"].isin(top + mid + bottom)].copy()
    if subset.empty:
        return

    def _distributions(ids: list[str]) -> list[np.ndarray]:
        return [subset.loc[subset["series_id"] == s, "observed_demand"].to_numpy() for s in ids]

    def _draw(ax: plt.Axes, ids: list[str], color: str, title: str) -> None:
        bp = ax.boxplot(
            _distributions(ids),
            tick_labels=ids,
            patch_artist=True,
            medianprops={"color": "#d62728", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_xlabel("Series")
        ax.set_ylabel("Observed demand")
        ax.tick_params(axis="x", rotation=45)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw(axes[0], top, "#2166ac", "High-volume series")
    _draw(axes[1], mid + bottom, "#74add1", "Mid / low-volume series")

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


def _plot_zero_demand_rate_by_series(
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    rates = series_summary["zero_demand_rate"].dropna()
    if rates.empty:
        return

    median_rate = rates.median()
    pct_above_50 = (rates > 0.5).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(rates, bins=40, color="#8c564b", edgecolor="white", alpha=0.85)
    ax.axvline(
        median_rate,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label=f"Median: {median_rate:.2f}",
    )
    ax.axvline(
        0.5,
        color="#ff7f0e",
        linewidth=1.4,
        linestyle=":",
        label=f">50% zero: {pct_above_50:.0f}% of series",
    )
    ax.set_xlabel("Zero-demand rate")
    ax.set_ylabel("Number of series")
    ax.set_title("Distribution of zero-demand rate across all series")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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


def _make_grid(
    n_items: int,
    n_cols: int,
    width: float,
    row_height: float,
    sharex: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a grid of subplots sized for ``n_items`` and hide the unused cells."""
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, row_height * n_rows), sharex=sharex)
    axes_flat = np.atleast_1d(axes).flatten()
    for axis in axes_flat[n_items:]:
        axis.axis("off")
    return fig, axes_flat


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
    fig, axes_flat = _make_grid(len(columns), n_cols, width=14, row_height=4)

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


def _select_diverse_series(panel: pd.DataFrame, n: int = 12) -> list[str]:
    """Pick n series spanning different stores, demand levels and stockout exposure."""
    min_days = panel.groupby("series_id")["date"].count()
    threshold = min_days.quantile(0.5)
    valid = min_days[min_days >= threshold].index

    df = panel[panel["series_id"].isin(valid)].copy()
    stats = df.groupby("series_id").agg(
        mean_demand=("observed_demand", "mean"),
        zero_rate=("observed_demand", lambda x: (x == 0).mean()),
        stockout_rate=("stockout_hours", lambda x: (x > 0).mean()),
        store=("store_id", "first"),
    )

    n_quantiles = min(3, stats["mean_demand"].nunique())
    stats["demand_tier"] = pd.qcut(
        stats["mean_demand"],
        q=n_quantiles,
        labels=["low", "mid", "high"][:n_quantiles],
        duplicates="drop",
    )
    stats["stockout_tier"] = pd.cut(
        stats["stockout_rate"], bins=[-0.01, 0.05, 0.3, 1.0], labels=["low", "mid", "high"]
    )

    selected: list[str] = []
    used_stores: set[str] = set()
    rng = np.random.default_rng(42)

    for demand_tier in ["high", "mid", "low"]:
        for stockout_tier in ["high", "mid", "low"]:
            candidates = stats[
                (stats["demand_tier"] == demand_tier)
                & (stats["stockout_tier"] == stockout_tier)
                & (~stats["store"].isin(used_stores))
            ]
            if candidates.empty:
                candidates = stats[
                    (stats["demand_tier"] == demand_tier) & (~stats["store"].isin(used_stores))
                ]
            if candidates.empty:
                continue
            pick = rng.choice(candidates.index)
            selected.append(pick)
            used_stores.add(stats.loc[pick, "store"])
            if len(selected) >= n:
                return selected

    return selected[:n]


def _plot_representative_series_panels(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_series = _select_diverse_series(panel, n=REPRESENTATIVE_SERIES_COUNT)
    if not selected_series:
        return

    subset = panel.loc[panel["series_id"].isin(selected_series)].copy()
    n_cols = 3
    fig, axes_flat = _make_grid(len(selected_series), n_cols, width=16, row_height=3.6, sharex=True)

    for axis, series_id in zip(axes_flat, selected_series, strict=False):
        series_frame = subset.loc[subset["series_id"] == series_id].sort_values("date")
        demand_max = series_frame["observed_demand"].max()
        stockout_max = series_frame["stockout_hours"].max()
        # Scale stockout overlay to demand range so it doesn't dominate the axis
        stockout_scaled = (
            series_frame["stockout_hours"] / stockout_max * demand_max
            if stockout_max > 0
            else series_frame["stockout_hours"]
        )
        axis.fill_between(
            series_frame["date"],
            0,
            stockout_scaled,
            color="#d62728",
            alpha=0.25,
            label="Stockout (scaled)",
        )
        axis.plot(
            series_frame["date"],
            series_frame["observed_demand"],
            color="#1f77b4",
            linewidth=1.8,
            label="Demand",
        )
        axis.set_title(series_id)
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("Demand")

    fig.suptitle(
        "Representative series panels (line: demand, shaded: stockout hours)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
