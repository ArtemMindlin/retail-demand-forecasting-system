"""Convert run artifacts into the LaTeX tables the results chapter inputs.

Both tables are generated, never hand-edited: `memoria/tables/table_predictive.tex`
(`tab:metrics_predictive`) and `memoria/tables/table_costs.tex` (`tab:metrics_cost`).

The runs are passed in, deliberately. This module used to pin them in `__main__`, so
regenerating a table silently republished whichever run was hardcoded there — that is
how both tables drifted to runs predating the August 2026 audit fixes while
`docs/runs.md` declared newer ones. Pass the run that `docs/runs.md` declares -- by the
name it is cited under, which `resolve_run_dir` looks up in MLflow, or by directory.

It lives in `evaluation` and not in `utils` for that lookup: `utils` imports no
first-party layer, and `tracking` imports `utils`, so reaching the run index from there
would have closed a cycle.

Usage:
    python -m retail_forecasting.evaluation.latex_exporter \
        --metrics-run <corrida Observed> <corrida Latent> \
        --fair-cost-run <corrida de coste justo>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from retail_forecasting.tracking import resolve_run_dir

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
    # Shrink to the text block if the table is wider, never stretch it if narrower. The
    # exporter cannot know how wide its own numbers will be: adding the cost-gap column
    # overflowed the page by 113pt, and the next re-run changes every figure in it.
    fit = "\\resizebox{\\ifdim\\width>\\textwidth \\textwidth\\else\\width\\fi}{!}{%"
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{fit}\n"
        f"\\begin{{tabular}}{{{column_format}}}\n"
        "\\toprule\n"
        f"{' & '.join(header)} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table}\n"
    )


def _signed_pct_column(metrics_df: pd.DataFrame) -> pd.Series:
    """Render `bias_pct` with an explicit sign, or `n/d` for runs that predate the metric.

    Runs older than the metric carry no column at all, and a silent zero there would read as
    an unbiased model rather than as a missing measurement.
    """
    if "bias_pct" not in metrics_df.columns:
        return pd.Series(["n/d"] * len(metrics_df), index=metrics_df.index)
    return metrics_df["bias_pct"].map(
        lambda value: (
            "n/d"
            if pd.isna(value)
            else f"{'+' if value >= 0 else '-'}{_es_number(abs(float(value)))}"
        )
    )


def _cost_gap_column(costs_df: pd.DataFrame) -> pd.Series:
    """The paired cost gap against the baseline, rendered as ``mean [low; high]``.

    Pre-formatted here rather than left to ``_tabular``, which prices one number per cell.
    The baseline row is a dash: its gap against itself is not zero, it is undefined, and a
    zero-width interval on the reference reads as a finding.
    """
    if "cost_ci95_low" not in costs_df.columns:
        return pd.Series(["--"] * len(costs_df), index=costs_df.index)

    def render(row: pd.Series) -> str:
        if pd.isna(row["cost_delta"]):
            return "--"
        return (
            f"{_es_number(row['cost_delta'])} "
            f"[{_es_number(row['cost_ci95_low'])}; {_es_number(row['cost_ci95_high'])}]"
        )

    return costs_df.apply(render, axis=1)


def _fair_cost_table(costs_df: pd.DataFrame) -> str:
    """Fair-cost backtest table: every strategy charged against a common ground truth."""
    cost_cols = ["strategy", "total_cost", "fill_rate", "mean_order"]
    cost_table = costs_df[cost_cols].copy()
    cost_table["strategy"] = _prettify(cost_table["strategy"])
    cost_table["cost_gap"] = _cost_gap_column(costs_df)
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
    draws = ""
    if "n_draws" in costs_df.columns:
        n_draws = int(costs_df["n_draws"].iloc[0])
        draws = (
            f" Cada cifra es la media de {n_draws} sorteos de censura, y el hueco de coste es "
            "una diferencia EMPAREJADA frente al observado, con su intervalo de confianza al "
            "95\\% en unidades de coste (no en puntos porcentuales)."
        )
    caption = (
        "Comparativa de costes operativos bajo evaluación justa: todas las estrategias se "
        "puntúan contra la misma demanda real mediante censura sintética sobre días limpios"
        f"{panel} ($n = {n_eval_es}$ días censurados). La política de pedido es común a todas "
        "ellas y deliberadamente ingenua --- se pide exactamente lo que dice la señal, sin "
        "término de seguridad, y se tarifa con el único par de costes del catálogo ---, de modo "
        f"que el coste ES el error asimétrico de la señal y nada más.{draws}"
    )
    return _tabular(
        cost_table,
        header=[
            "Estrategia",
            "Coste Total",
            "Fill Rate (\\%)",
            "Pedido Medio",
            "$\\Delta$ vs Observado (IC 95\\%)",
        ],
        column_format="@{}lcccc@{}",
        caption=caption,
        label="tab:metrics_cost",
    )


def export_to_latex(
    metrics_path: str | Path | Sequence[str | Path],
    costs_path: str | Path,
    output_dir: str | Path,
    panel_series: int | None = None,
) -> None:
    """Convert CSV results to LaTeX tables from a run's `metrics_summary.csv` and its
    `fair_cost_backtest.csv`.

    ``panel_series`` is stated in the predictive caption. The base subset and the scale
    run produce very different errors, so a table that does not name its panel invites
    the reader to attribute it to whichever one the surrounding section describes.

    ``metrics_path`` accepts several runs and concatenates them. An experiment scores one
    demand strategy per run, so the arms of `tab:metrics_predictive` arrive as separate
    runs; the table still lists them together because it reports each against its own
    target, which is what its caption says.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = [metrics_path] if isinstance(metrics_path, str | Path) else list(metrics_path)
    metrics_df = pd.concat([pd.read_csv(path) for path in metrics_paths], ignore_index=True)
    costs_df = pd.read_csv(costs_path)

    # 1. Predictive Metrics Table (MAE/RMSE/bias) — keep only representative models.
    pred_cols = ["data_strategy", "model_name", "mae", "rmse"]
    mask = metrics_df["model_name"].isin(_REPORTED_MODELS)
    pred_table = metrics_df.loc[mask, pred_cols].sort_values(["data_strategy", "mae"])
    pred_table["model_name"] = _prettify(pred_table["model_name"])
    pred_table["data_strategy"] = _prettify(pred_table["data_strategy"])
    # Signed, and beside the absolute errors on purpose: the point estimator is fitted at the
    # critical fractile, so MAE alone cannot tell a systematic shift from raw dispersion.
    pred_table["bias_pct"] = _signed_pct_column(metrics_df.loc[mask, :])

    panel = f", panel de {panel_series} series" if panel_series else ""
    latex_pred = _tabular(
        pred_table,
        header=["Estrategia", "Modelo", "MAE", "RMSE", "Sesgo (\\%)"],
        column_format="@{}llccc@{}",
        caption=(
            "Comparativa de errores predictivos en el backtesting walk-forward "
            f"($K=3$ folds, $H=7$ días{panel}). Cada estrategia se mide contra su propio "
            "target, por lo que los errores no son comparables entre estrategias "
            "(véase la discusión en la Sección~\\ref{sec:predictivo_walkforward}). El sesgo "
            "es el error medio con signo como fracción de la demanda, positivo cuando el "
            "modelo sobrestima."
        ),
        label="tab:metrics_predictive",
    )

    latex_cost = _fair_cost_table(costs_df)

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
        type=resolve_run_dir,
        required=True,
        nargs="+",
        help="One run name or directory per demand strategy, each holding a metrics_summary.csv "
        "(backs tab:metrics_predictive). An experiment scores one strategy per run, so pass "
        "the Observed run and the Latent one.",
    )
    parser.add_argument(
        "--fair-cost-run",
        type=resolve_run_dir,
        required=True,
        help="Run name or directory holding fair_cost_backtest.csv (backs tab:metrics_cost).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("memoria/tables"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_to_latex(
        [run / "metrics_summary.csv" for run in args.metrics_run],
        args.fair_cost_run / "fair_cost_backtest.csv",
        args.output_dir,
        panel_series=_panel_series(args.metrics_run[0]),
    )


if __name__ == "__main__":
    main()
