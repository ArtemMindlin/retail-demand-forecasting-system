"""Convert run artifacts into the LaTeX tables the results chapter inputs.

Both tables are generated, never hand-edited: `memoria/tables/table_predictive.tex`
(`tab:metrics_predictive`) and `memoria/tables/table_costs.tex` (`tab:metrics_cost`).

The runs are passed in, deliberately. This module used to pin them in `__main__`, so
regenerating a table silently republished whichever run was hardcoded there — that is
how both tables drifted to runs predating the August 2026 audit fixes while
`docs/runs.md` declared newer ones. Pass the run that `docs/runs.md` declares.

Usage:
    python -m retail_forecasting.utils.latex_exporter \
        --metrics-run reports/fresh_retailnet_large_20260811_125735 \
        --fair-cost-run reports/fresh_retailnet_large_20260811_184959
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd

CostMode = Literal["fair", "summary"]

# Internal identifiers -> the names used in the prose of the report. Without this map the
# generated tables read "Auto Boosting"/"Catboost"/"Seasonal Naive" while the chapters say
# LightGBM/CatBoost/Seasonal Naïve, which looks like a different experiment to the reader.
_DISPLAY_NAMES = {
    "auto_boosting": "LightGBM",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "seasonal_naive": "Seasonal Naïve",
    "observed": "Observed",
    "latent_supervised": "Latent Supervised",
    "latent_historical_mean": "Latent Historical Mean",
    "latent_clipped_scaling": "Latent Clipped Scaling",
}

# Models represented in the report tables. `lightgbm` and `auto_boosting` are the same
# backend under two names the pipeline has used over time; both must be listed or the
# filter drops the LightGBM rows without saying so.
_REPORTED_MODELS = ("catboost", "lightgbm", "auto_boosting", "seasonal_naive")


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


def _tabular(
    frame: pd.DataFrame,
    header: list[str],
    column_format: str,
    caption: str,
    label: str,
) -> str:
    """Emit a booktabs table for ``frame``, one row per record, in document order.

    Written by hand rather than via ``DataFrame.to_latex`` because pandas routes that
    call through ``Styler``, which needs Jinja2 — dropped from this project's
    dependencies in `14ad8b4`. The exporter was therefore unrunnable, which is the
    mechanical reason both tables drifted from their declared runs: regenerating them
    raised ``ImportError`` and the stale files stayed in place.
    """
    cells = [
        [value if isinstance(value, str) else _es_number(float(value)) for value in record]
        for record in frame.itertuples(index=False)
    ]
    body = "\n".join(" & ".join(row) + r" \\" for row in cells)
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{column_format}}}\n"
        "\\toprule\n"
        f"{' & '.join(header)} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _fair_cost_table(costs_df: pd.DataFrame) -> str:
    """Fair-cost backtest table: every strategy charged against a common ground truth."""
    cost_cols = ["strategy", "signal_mae", "total_cost", "fill_rate", "mean_order"]
    cost_table = costs_df[cost_cols].copy()
    cost_table["strategy"] = _prettify(cost_table["strategy"])
    n_eval = int(costs_df["n_eval"].iloc[0]) if "n_eval" in costs_df.columns else 0
    # Spanish thousands separator, applied to the number alone (never to the whole caption).
    n_eval_es = f"{n_eval:,}".replace(",", "\\,")
    # The ranking of this table depends on the panel the 30 series were drawn from, so the
    # caption states it. Sampling the 50-series subset instead inverts the order.
    panel = ""
    if {"source_panel_series", "sampled_series"} <= set(costs_df.columns):
        sampled = int(costs_df["sampled_series"].iloc[0])
        source = int(costs_df["source_panel_series"].iloc[0])
        panel = f", en {sampled} series muestreadas del panel de {source}"
    caption = (
        "Comparativa de costes operativos bajo evaluación justa: todas las estrategias se "
        "puntúan contra la misma demanda real mediante censura sintética sobre días limpios"
        f"{panel} ($n = {n_eval_es}$ días censurados). La política de pedido es común a todas "
        "ellas (aproximación normal \\textit{order-up-to} con costes aplanados), de modo que "
        "la diferencia de coste es atribuible únicamente a la señal de demanda."
    )
    return _tabular(
        cost_table,
        header=["Estrategia", "MAE Señal", "Coste Total", "Fill Rate (\\%)", "Pedido Medio"],
        column_format="@{}lcccc@{}",
        caption=caption,
        label="tab:metrics_cost",
    )


def _cost_summary_table(costs_df: pd.DataFrame) -> str:
    """Legacy per-model cost summary table (each strategy graded against its own target)."""
    cost_cols = ["data_strategy", "model_name", "total_cost", "mean_cost"]
    mask_costs = costs_df["model_name"].isin(_REPORTED_MODELS)
    cost_table = costs_df.loc[mask_costs, cost_cols].sort_values(["total_cost"])
    cost_table["model_name"] = _prettify(cost_table["model_name"])
    cost_table["data_strategy"] = _prettify(cost_table["data_strategy"])
    return _tabular(
        cost_table,
        header=["Estrategia", "Modelo", "Coste Total", "Coste Medio"],
        column_format="@{}llcc@{}",
        caption="Comparativa de costes operativos (Generada automáticamente).",
        label="tab:metrics_cost_gen",
    )


def export_to_latex(
    metrics_path: str | Path,
    costs_path: str | Path,
    output_dir: str | Path,
    cost_mode: CostMode = "fair",
    panel_series: int | None = None,
) -> None:
    """Convert CSV results to LaTeX tables.

    ``cost_mode`` selects which cost table to build (the caller knows which CSV it
    passed): ``"fair"`` for a ``fair_cost_backtest.csv``, ``"summary"`` for a
    legacy ``cost_summary.csv``.

    ``panel_series`` is stated in the predictive caption. The base subset and the scale
    run produce very different errors, so a table that does not name its panel invites
    the reader to attribute it to whichever one the surrounding section describes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(metrics_path)
    costs_df = pd.read_csv(costs_path)

    # 1. Predictive Metrics Table (MAE/RMSE) — keep only representative models.
    pred_cols = ["data_strategy", "model_name", "mae", "rmse"]
    mask = metrics_df["model_name"].isin(_REPORTED_MODELS)
    pred_table = metrics_df.loc[mask, pred_cols].sort_values(["data_strategy", "mae"])
    pred_table["model_name"] = _prettify(pred_table["model_name"])
    pred_table["data_strategy"] = _prettify(pred_table["data_strategy"])

    panel = f", panel de {panel_series} series" if panel_series else ""
    latex_pred = _tabular(
        pred_table,
        header=["Estrategia", "Modelo", "MAE", "RMSE"],
        column_format="@{}llcc@{}",
        caption=(
            "Comparativa de errores predictivos en el backtesting walk-forward "
            f"($K=3$ folds, $H=7$ días{panel}). Cada estrategia se mide contra su propio "
            "target, por lo que los errores no son comparables entre estrategias "
            "(véase la discusión en la Sección~\\ref{sec:predictivo_walkforward})."
        ),
        label="tab:metrics_predictive",
    )

    # 2. Economic Metrics Table (Total Cost)
    latex_cost = (
        _fair_cost_table(costs_df) if cost_mode == "fair" else _cost_summary_table(costs_df)
    )

    (output_dir / "table_predictive.tex").write_text(latex_pred, encoding="utf-8")
    (output_dir / "table_costs.tex").write_text(latex_cost, encoding="utf-8")

    print(f"LaTeX tables exported to {output_dir}")


def _panel_series(run_dir: Path) -> int | None:
    """Read the panel size a run declares, so the caption can state it."""
    metadata_path = run_dir / "backtest_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    series = metadata.get("dataset", {}).get("series")
    return int(series) if series else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-run",
        type=Path,
        required=True,
        help="Run directory holding metrics_summary.csv (backs tab:metrics_predictive).",
    )
    parser.add_argument(
        "--fair-cost-run",
        type=Path,
        required=True,
        help="Run directory holding fair_cost_backtest.csv (backs tab:metrics_cost).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("memoria/tables"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_to_latex(
        args.metrics_run / "metrics_summary.csv",
        args.fair_cost_run / "fair_cost_backtest.csv",
        args.output_dir,
        cost_mode="fair",
        panel_series=_panel_series(args.metrics_run),
    )


if __name__ == "__main__":
    main()
