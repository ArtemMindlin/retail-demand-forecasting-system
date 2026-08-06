from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

CostMode = Literal["fair", "summary"]

# Internal identifiers -> the names used in the prose of the report. Without this map the
# generated tables read "Auto Boosting"/"Catboost"/"Seasonal Naive" while the chapters say
# LightGBM/CatBoost/Seasonal Naïve, which looks like a different experiment to the reader.
_DISPLAY_NAMES = {
    "auto_boosting": "LightGBM",
    "catboost": "CatBoost",
    "seasonal_naive": "Seasonal Naïve",
    "observed": "Observed",
    "latent_supervised": "Latent Supervised",
    "latent_historical_mean": "Latent Historical Mean",
    "latent_clipped_scaling": "Latent Clipped Scaling",
}


def _es_number(value: float) -> str:
    """Format a number with Spanish conventions: '.' thousands, ',' decimals.

    The memoria is written in Spanish, so the generated tables must match the prose
    (12.881,98) instead of pandas' default (12881.98). Without this the tables had to be
    reformatted by hand after every export, which is how they drifted from the exporter.
    """
    return f"{value:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _prettify(column: pd.Series) -> pd.Series:
    """Map internal snake_case identifiers to their report-facing names."""
    normalised = column.astype(str).str.strip().str.lower()
    fallback = normalised.str.replace("_", " ").str.title()
    return normalised.map(_DISPLAY_NAMES).fillna(fallback)


def _centered(latex: str) -> str:
    """Insert the ``\\centering`` that ``DataFrame.to_latex`` does not emit."""
    return latex.replace("\n\\caption{", "\n\\centering\n\\caption{", 1)


def _fair_cost_table(costs_df: pd.DataFrame) -> str:
    """Fair-cost backtest table: every strategy charged against a common ground truth."""
    cost_cols = ["strategy", "signal_mae", "total_cost", "fill_rate", "mean_order"]
    cost_table = costs_df[cost_cols].copy()
    cost_table["strategy"] = _prettify(cost_table["strategy"])
    n_eval = int(costs_df["n_eval"].iloc[0]) if "n_eval" in costs_df.columns else 0
    # Spanish thousands separator, applied to the number alone (never to the whole caption).
    n_eval_es = f"{n_eval:,}".replace(",", "\\,")
    caption = (
        "Comparativa de costes operativos bajo evaluación justa: todas las estrategias se "
        "puntúan contra la misma demanda real mediante censura sintética sobre días limpios "
        f"($n = {n_eval_es}$ días censurados). La política de pedido es común a todas ellas "
        "(aproximación normal \\textit{order-up-to} con costes aplanados), de modo que la "
        "diferencia de coste es atribuible únicamente a la señal de demanda."
    )
    latex: str = cost_table.to_latex(
        index=False,
        header=["Estrategia", "MAE Señal", "Coste Total", "Fill Rate (\\%)", "Pedido Medio"],
        float_format=_es_number,
        bold_rows=False,
        column_format="@{}lcccc@{}",
        label="tab:metrics_cost",
        caption=caption,
        position="h",
    )
    return latex


def _cost_summary_table(costs_df: pd.DataFrame) -> str:
    """Legacy per-model cost summary table (each strategy graded against its own target)."""
    cost_cols = ["data_strategy", "model_name", "total_cost", "mean_cost"]
    mask_costs = costs_df["model_name"].isin(["catboost", "seasonal_naive", "auto_boosting"])
    cost_table = costs_df.loc[mask_costs, cost_cols].sort_values(["total_cost"])
    cost_table["model_name"] = _prettify(cost_table["model_name"])
    cost_table["data_strategy"] = _prettify(cost_table["data_strategy"])
    latex: str = cost_table.to_latex(
        index=False,
        header=["Estrategia", "Modelo", "Coste Total", "Coste Medio"],
        float_format=_es_number,
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
    pred_table["model_name"] = _prettify(pred_table["model_name"])
    pred_table["data_strategy"] = _prettify(pred_table["data_strategy"])

    latex_pred = pred_table.to_latex(
        index=False,
        header=["Estrategia", "Modelo", "MAE", "RMSE"],
        float_format=_es_number,
        bold_rows=False,
        column_format="@{}llcc@{}",
        label="tab:metrics_predictive",
        caption=(
            "Comparativa de errores predictivos en el backtesting walk-forward "
            "($K=3$ folds, $H=7$ días). Cada estrategia se mide contra su propio target, "
            "por lo que los errores no son comparables entre estrategias "
            "(véase la discusión en la Sección~\\ref{sec:predictivo_walkforward})."
        ),
        position="h",
    )

    # 2. Economic Metrics Table (Total Cost)
    latex_cost = (
        _fair_cost_table(costs_df) if cost_mode == "fair" else _cost_summary_table(costs_df)
    )

    (output_dir / "table_predictive.tex").write_text(_centered(latex_pred), encoding="utf-8")
    (output_dir / "table_costs.tex").write_text(_centered(latex_cost), encoding="utf-8")

    print(f"LaTeX tables exported to {output_dir}")


if __name__ == "__main__":
    # Reference runs from the audit report
    metrics_run = "reports/fresh_retailnet_v2_20260620_080615/metrics_summary.csv"
    fair_cost_run = "reports/fresh_retailnet_v2_20260621_105332/fair_cost_backtest.csv"
    export_to_latex(metrics_run, fair_cost_run, "memoria/tables", cost_mode="fair")
