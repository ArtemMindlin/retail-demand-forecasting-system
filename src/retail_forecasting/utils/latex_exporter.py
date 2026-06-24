from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

CostMode = Literal["fair", "summary"]


def _fair_cost_table(costs_df: pd.DataFrame) -> str:
    """Fair-cost backtest table: every strategy charged against a common ground truth."""
    cost_cols = ["strategy", "signal_mae", "total_cost", "fill_rate", "mean_order"]
    cost_table = costs_df[cost_cols].copy()
    cost_table["strategy"] = cost_table["strategy"].str.replace("_", " ").str.title()
    latex: str = cost_table.to_latex(
        index=False,
        header=["Estrategia", "MAE Señal", "Coste Total", "Fill Rate (\\%)", "Pedido Medio"],
        float_format="%.2f",
        bold_rows=False,
        column_format="@{}lcccc@{}",
        label="tab:metrics_cost",
        caption="Comparativa de costes operativos bajo evaluación justa (evaluada contra la misma demanda real).",
        position="h",
    )
    return latex


def _cost_summary_table(costs_df: pd.DataFrame) -> str:
    """Legacy per-model cost summary table (each strategy graded against its own target)."""
    cost_cols = ["data_strategy", "model_name", "total_cost", "mean_cost"]
    mask_costs = costs_df["model_name"].isin(["catboost", "seasonal_naive", "auto_boosting"])
    cost_table = costs_df.loc[mask_costs, cost_cols].sort_values(["total_cost"])
    cost_table["model_name"] = cost_table["model_name"].str.replace("_", " ").str.title()
    cost_table["data_strategy"] = cost_table["data_strategy"].str.replace("_", " ").str.title()
    latex: str = cost_table.to_latex(
        index=False,
        header=["Estrategia", "Modelo", "Coste Total", "Coste Medio"],
        float_format="%.2f",
        bold_rows=False,
        column_format="@{}llcc@{}",
        label="tab:metrics_cost_gen",
        caption="Comparativa de costes operativos (Generada automáticamente).",
        position="h",
    )
    return latex


def export_to_latex(
    metrics_path: str | Path,
    costs_path: str | Path,
    output_dir: str | Path,
    cost_mode: CostMode = "fair",
) -> None:
    """Convert CSV results to LaTeX tables.

    ``cost_mode`` selects which cost table to build (the caller knows which CSV it
    passed): ``"fair"`` for a ``fair_cost_backtest.csv``, ``"summary"`` for a
    legacy ``cost_summary.csv``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(metrics_path)
    costs_df = pd.read_csv(costs_path)

    # 1. Predictive Metrics Table (MAE/RMSE) — keep only representative models.
    pred_cols = ["data_strategy", "model_name", "mae", "rmse"]
    mask = metrics_df["model_name"].isin(["catboost", "seasonal_naive", "auto_boosting"])
    pred_table = metrics_df.loc[mask, pred_cols].sort_values(["data_strategy", "mae"])
    pred_table["model_name"] = pred_table["model_name"].str.replace("_", " ").str.title()
    pred_table["data_strategy"] = pred_table["data_strategy"].str.replace("_", " ").str.title()

    latex_pred = pred_table.to_latex(
        index=False,
        header=["Estrategia", "Modelo", "MAE", "RMSE"],
        float_format="%.2f",
        bold_rows=False,
        column_format="@{}llcc@{}",
        label="tab:metrics_predictive",
        caption="Comparativa de errores predictivos (Generada automáticamente).",
        position="h",
    )

    # 2. Economic Metrics Table (Total Cost)
    latex_cost = (
        _fair_cost_table(costs_df) if cost_mode == "fair" else _cost_summary_table(costs_df)
    )

    (output_dir / "table_predictive.tex").write_text(latex_pred)
    (output_dir / "table_costs.tex").write_text(latex_cost)

    print(f"LaTeX tables exported to {output_dir}")


if __name__ == "__main__":
    # Reference runs from the audit report
    metrics_run = "reports/fresh_retailnet_v2_20260620_080615/metrics_summary.csv"
    fair_cost_run = "reports/fresh_retailnet_v2_20260621_105332/fair_cost_backtest.csv"
    export_to_latex(metrics_run, fair_cost_run, "memoria/tables", cost_mode="fair")
