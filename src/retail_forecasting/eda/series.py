"""Per-series profile of the panel: how much each series sells, how idle it is, how it looks.

The summary and the three figures drawn from it live together, so a figure and the table beside
it in the run directory cannot drift apart -- which is what happened when the stockout band
statistic was derived once here and once six hundred lines away.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from retail_forecasting.utils.plotting import make_grid

TOP_SERIES_PLOT_COUNT = 12
REPRESENTATIVE_SERIES_COUNT = 12


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


def render_series_figures(
    panel: pd.DataFrame, series_summary: pd.DataFrame, output_dir: Path
) -> None:
    """Every figure that ranks, selects or portrays individual series."""
    _plot_observed_demand_boxplot_top_series(
        panel, series_summary, output_dir / "observed_demand_boxplot_top_series.png"
    )
    _plot_zero_demand_rate_by_series(series_summary, output_dir / "zero_demand_rate_by_series.png")
    _plot_representative_series_panels(
        panel, series_summary, output_dir / "representative_series_panels.png"
    )


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

    def _draw(ax: Axes, ids: list[str], color: str, title: str) -> None:
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
    fig, axes_flat = make_grid(len(selected_series), n_cols, width=16, row_height=3.6, sharex=True)

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
