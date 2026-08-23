from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
                    stockout_panel["stockout_hours"].mean() if not stockout_panel.empty else 0.0
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


def render_stockout_figures(
    panel: pd.DataFrame, stockout_demand_bands: pd.DataFrame, output_dir: Path
) -> None:
    """Every stockout figure, drawn from this module's own summaries."""
    _plot_stockout_hours_distribution(panel, output_dir / "stockout_hours_distribution.png")
    _plot_stockout_band_demand(stockout_demand_bands, output_dir / "stockout_band_demand.png")
    _plot_stockout_vs_demand_scatter(panel, output_dir / "stockout_vs_demand_scatter.png")


# ── Figuras ───────────────────────────────────────────────────────────────────
# Junto a los resúmenes que dibujan, y no en un módulo de figuras aparte: la figura de bandas
# de stockout llegó a reconstruir las bandas por su cuenta, con el mismo `pd.cut` que el
# constructor de arriba, porque los dos cálculos vivían a seiscientas líneas de distancia.

SCATTER_SAMPLE_SIZE = 5000


def _sample_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if len(panel) <= SCATTER_SAMPLE_SIZE:
        return panel
    return panel.sample(SCATTER_SAMPLE_SIZE, random_state=42)


def _plot_stockout_hours_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    max_hours = int(panel["stockout_hours"].max()) + 1
    bins = np.arange(-0.5, max_hours + 0.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        panel["stockout_hours"],
        bins=bins,
        color="#d62728",
        edgecolor="white",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Stockout hours")
    axes[0].set_ylabel("Frequency (log scale)")
    axes[0].set_title("Stockout hours distribution (log scale)")
    axes[0].set_xticks(range(0, max_hours, 2))

    positive_stockout = panel.loc[panel["stockout_hours"] > 0, "stockout_hours"]
    bins_pos = np.arange(0.5, max_hours + 0.5, 1)
    axes[1].hist(
        positive_stockout,
        bins=bins_pos,
        color="#ff9896",
        edgecolor="white",
    )
    axes[1].set_xlabel("Positive stockout hours")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Positive stockout-hour distribution")
    axes[1].set_xticks(range(1, max_hours, 2))

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stockout_band_demand(stockout_demand_bands: pd.DataFrame, output_path: Path) -> None:
    """Draw the band summary rather than recomputing it.

    The bands and their cut points used to be built a second time here, from the panel, with
    the same `pd.cut` literal that `build_stockout_demand_bands` uses. Two derivations of one
    statistic drift silently: move a cut point in one place and chapter 3 ends up with a
    figure and a table that disagree, with nothing failing.
    """
    stockout_band_frame = stockout_demand_bands

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(
        stockout_band_frame["stockout_band"].astype(str),
        stockout_band_frame["observed_demand_mean"],
        color="#ff7f0e",
    )
    axes[0].set_xlabel("Stockout band")
    axes[0].set_ylabel("Mean observed demand")
    axes[0].set_title("Mean demand by stockout band")

    axes[1].bar(
        stockout_band_frame["stockout_band"].astype(str),
        stockout_band_frame["observations"],
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
