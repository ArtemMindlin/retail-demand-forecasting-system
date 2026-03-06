from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from retail_forecasting.utils.io import ensure_directory


def render_standard_plots(
    metrics_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    target_dir = ensure_directory(output_dir)
    _plot_total_cost(cost_summary, target_dir / "cost_by_model.png")
    _plot_error_cost_tradeoff(metrics_summary, cost_summary, target_dir / "error_cost_tradeoff.png")


def _plot_total_cost(cost_summary: pd.DataFrame, output_path: Path) -> None:
    if cost_summary.empty:
        return

    plt.figure(figsize=(8, 4))
    plt.bar(cost_summary["model_name"], cost_summary["total_cost"])
    plt.ylabel("Total cost")
    plt.xlabel("Model")
    plt.title("Total operating cost by model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_error_cost_tradeoff(
    metrics_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if metrics_summary.empty or cost_summary.empty:
        return

    merged = metrics_summary.merge(
        cost_summary[["model_name", "backend_name", "total_cost"]],
        on=["model_name", "backend_name"],
        how="inner",
    )
    if merged.empty:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(merged["mae"], merged["total_cost"])
    for row in merged.itertuples(index=False):
        plt.annotate(row.model_name, (row.mae, row.total_cost))
    plt.xlabel("MAE")
    plt.ylabel("Total cost")
    plt.title("Error-cost tradeoff")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
