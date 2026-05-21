from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_to_latex(
    metrics_path: str | Path, costs_path: str | Path, output_dir: str | Path
) -> None:
    """Converts CSV results to LaTeX tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(metrics_path)
    costs_df = pd.read_csv(costs_path)

    # 1. Predictive Metrics Table (MAE/RMSE)
    # We filter and format according to the thesis structure
    pred_cols = ["data_strategy", "model_name", "mae", "rmse"]
    # Only keep some representative models to avoid cluttering the table
    mask = metrics_df["model_name"].isin(["catboost", "seasonal_naive", "auto_boosting"])
    pred_table = metrics_df.loc[mask, pred_cols].sort_values(["data_strategy", "mae"])

    # Capitalize and format for beauty
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
    cost_cols = ["data_strategy", "model_name", "total_cost", "mean_cost"]
    mask_costs = costs_df["model_name"].isin(["catboost", "seasonal_naive", "auto_boosting"])
    cost_table = costs_df.loc[mask_costs, cost_cols].sort_values(["total_cost"])

    cost_table["model_name"] = cost_table["model_name"].str.replace("_", " ").str.title()
    cost_table["data_strategy"] = cost_table["data_strategy"].str.replace("_", " ").str.title()

    latex_cost = cost_table.to_latex(
        index=False,
        header=["Estrategia", "Modelo", "Coste Total", "Coste Medio"],
        float_format="%.2f",
        bold_rows=False,
        column_format="@{}llcc@{}",
        label="tab:metrics_cost_gen",
        caption="Comparativa de costes operativos (Generada automáticamente).",
        position="h",
    )

    # Clean up standard to_latex output to fit the \input style (remove table environment if needed,
    # but here we keep it as a standalone snippet)
    (output_dir / "table_predictive.tex").write_text(latex_pred)
    (output_dir / "table_costs.tex").write_text(latex_cost)

    print(f"LaTeX tables exported to {output_dir}")


if __name__ == "__main__":
    # Example usage for the latest report
    latest_report = "reports/fresh_retailnet_v2_YYYYMMDD_HHMMSS"
    export_to_latex(
        f"{latest_report}/metrics_summary.csv",
        f"{latest_report}/cost_summary.csv",
        "memoria/tables",
    )
