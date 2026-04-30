from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from retail_forecasting.utils.io import ensure_directory


def render_eda_plots(
    panel: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Render a standard plot set for EDA runs."""
    target_dir = ensure_directory(output_dir)
    _plot_observed_demand_distribution(
        panel, target_dir / "observed_demand_distribution.png"
    )
    _plot_stockout_hours_distribution(
        panel, target_dir / "stockout_hours_distribution.png"
    )
    _plot_weekday_demand_profile(
        weekday_summary, target_dir / "weekday_demand_profile.png"
    )
    _plot_top_series_total_demand(
        series_summary, target_dir / "top_series_total_demand.png"
    )


def _plot_observed_demand_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(panel["observed_demand"], bins=30, color="#1f77b4", edgecolor="white")
    plt.xlabel("Observed demand")
    plt.ylabel("Frequency")
    plt.title("Observed demand distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_stockout_hours_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(panel["stockout_hours"], bins=24, color="#d62728", edgecolor="white")
    plt.xlabel("Stockout hours")
    plt.ylabel("Frequency")
    plt.title("Stockout hours distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_weekday_demand_profile(
    weekday_summary: pd.DataFrame, output_path: Path
) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(
        weekday_summary["weekday_name"],
        weekday_summary["observed_demand_mean"],
        marker="o",
        color="#2ca02c",
    )
    plt.xlabel("Weekday")
    plt.ylabel("Mean observed demand")
    plt.title("Weekly demand profile")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_top_series_total_demand(
    series_summary: pd.DataFrame, output_path: Path
) -> None:
    top_series = series_summary.head(10)
    if top_series.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.barh(
        top_series["series_id"][::-1],
        top_series["observed_demand_sum"][::-1],
        color="#9467bd",
    )
    plt.xlabel("Total observed demand")
    plt.ylabel("Series")
    plt.title("Top 10 series by observed demand")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
