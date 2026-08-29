from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap


def render_standard_plots(
    metrics_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Render the standard backtest plots (cost-by-model, error-cost tradeoff) to disk."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    _plot_total_cost(cost_summary, target_dir / "cost_by_model.png")
    _plot_error_cost_tradeoff(metrics_summary, cost_summary, target_dir / "error_cost_tradeoff.png")


def render_shap_summary(
    shap_values: shap.Explanation,
    output_path: Path,
) -> None:
    """
    Render a SHAP summary plot (dot plot) and save it to disk.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, show=False)
    plt.title("SHAP Feature Importance (Global)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_total_cost(cost_summary: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of total operating cost by model."""
    if cost_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cost_summary["model_name"], cost_summary["total_cost"])
    ax.set_ylabel("Coste total")
    ax.set_xlabel("Modelo")
    ax.set_title("Coste operativo total por modelo")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_error_cost_tradeoff(
    metrics_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter of point error (MAE) against total logistic cost per model."""
    if metrics_summary.empty or cost_summary.empty:
        return

    merged = metrics_summary.merge(
        cost_summary[["model_name", "backend_name", "total_cost"]],
        on=["model_name", "backend_name"],
        how="inner",
    )
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged["mae"], merged["total_cost"])
    for row in merged.itertuples(index=False):
        ax.annotate(row.model_name, (row.mae, row.total_cost))
    ax.set_xlabel("MAE")
    ax.set_ylabel("Coste total")
    ax.set_title("Compromiso entre error y coste")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_pareto_front(
    pareto_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render the multi-objective Pareto front (Pinball Loss vs. Winkler Score) to disk."""
    if pareto_df.empty or "pinball" not in pareto_df.columns or "winkler" not in pareto_df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    if "is_on_front" in pareto_df.columns:
        dominated = pareto_df[~pareto_df["is_on_front"]]
        front = pareto_df[pareto_df["is_on_front"]].sort_values("pinball")
        if not dominated.empty:
            ax.scatter(
                dominated["pinball"],
                dominated["winkler"],
                color="#94a3b8",
                alpha=0.6,
                label="Evaluaciones dominadas",
            )
        if not front.empty:
            ax.scatter(
                front["pinball"],
                front["winkler"],
                color="#3b82f6",
                s=70,
                label="Frontera de Pareto",
            )
            ax.plot(
                front["pinball"],
                front["winkler"],
                color="#3b82f6",
                linestyle="--",
                alpha=0.5,
            )
        if "is_selected" in pareto_df.columns:
            selected = pareto_df[pareto_df["is_selected"]]
            if not selected.empty:
                ax.scatter(
                    selected["pinball"],
                    selected["winkler"],
                    color="#ef4444",
                    s=130,
                    zorder=5,
                    label="Configuración seleccionada",
                )
    else:
        ax.scatter(
            pareto_df["pinball"],
            pareto_df["winkler"],
            color="#3b82f6",
            s=70,
            label="Frontera de Pareto",
        )

    ax.set_xlabel("Pinball Loss (q*) [Decisión de Inventario]")
    ax.set_ylabel("Winkler Score (P10-P90) [Calidad de Intervalo]")
    ax.set_title("Frente de Pareto Multiobjetivo (Optuna)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
