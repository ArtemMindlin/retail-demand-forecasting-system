"""Contract tests for the memoria table exporter.

The generated tables are the only path by which run artifacts reach the thesis, and
both of them had drifted to runs predating the August 2026 audit. Two defects made that
possible and each has a test here: the exporter could not run at all (Jinja2 had been
dropped, so `DataFrame.to_latex` raised), and its model filter omitted `lightgbm`, so
the LightGBM rows would have vanished silently from a regenerated table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retail_forecasting.evaluation.latex_exporter import _panel_series, export_to_latex

METRICS = pd.DataFrame(
    {
        "model_name": ["catboost", "lightgbm", "seasonal_naive"],
        "data_strategy": ["Latent_supervised"] * 3,
        "mae": [6.90850877458881, 5.910826635972128, 4.471199898008478],
        "rmse": [9.437776122560328, 8.366484278704425, 6.046985358695393],
    }
)

FAIR_COST = pd.DataFrame(
    {
        "strategy": ["Observed", "Latent_supervised"],
        "source_panel_series": [500, 500],
        "sampled_series": [30, 30],
        "signal_mae": [3.3912308020477817, 1.2909867391653858],
        "total_cost": [1843.4099137659625, 1774.8228848638441],
        "fill_rate": [90.42729586868771, 99.74686927967038],
        "mean_order": [10.442437074900662, 13.918321151333373],
        "n_eval": [293, 293],
    }
)


def _export(tmp_path: Path, panel_series: int | None = 500) -> tuple[str, str]:
    metrics_path = tmp_path / "metrics_summary.csv"
    costs_path = tmp_path / "fair_cost_backtest.csv"
    METRICS.to_csv(metrics_path, index=False)
    FAIR_COST.to_csv(costs_path, index=False)
    out = tmp_path / "tables"
    export_to_latex(metrics_path, costs_path, out, cost_mode="fair", panel_series=panel_series)
    return (
        (out / "table_predictive.tex").read_text(encoding="utf-8"),
        (out / "table_costs.tex").read_text(encoding="utf-8"),
    )


def test_export_runs_without_jinja2(tmp_path: Path) -> None:
    """The exporter must not depend on an optional pandas extra to produce a table."""
    predictive, costs = _export(tmp_path)
    assert predictive.startswith("\\begin{table}")
    assert costs.startswith("\\begin{table}")
    assert "\\toprule" in predictive and "\\bottomrule" in costs


def test_lightgbm_rows_reach_the_predictive_table(tmp_path: Path) -> None:
    """`lightgbm` must survive the model filter; it used to be dropped in silence."""
    predictive, _ = _export(tmp_path)
    assert "LightGBM" in predictive
    for name in ("CatBoost", "Seasonal Naïve"):
        assert name in predictive


def test_numbers_use_spanish_conventions(tmp_path: Path) -> None:
    """Thousands with '.', decimals with ',' — the memoria is written in Spanish."""
    predictive, costs = _export(tmp_path)
    assert "1.843,41" in costs
    assert "5,91" in predictive
    assert "1843.41" not in costs


def test_captions_declare_the_panel_they_came_from(tmp_path: Path) -> None:
    """A table that omits its panel gets attributed to whichever one the section names."""
    predictive, costs = _export(tmp_path)
    assert "panel de 500 series" in predictive
    assert "30 series muestreadas del panel de 500" in costs
    assert "$n = 293$" in costs


def test_panel_series_reads_run_metadata(tmp_path: Path) -> None:
    (tmp_path / "backtest_metadata.json").write_text(
        json.dumps({"dataset": {"series": 500}}), encoding="utf-8"
    )
    assert _panel_series(tmp_path) == 500
    assert _panel_series(tmp_path / "missing") is None
