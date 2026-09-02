"""Historical-run browser: Pareto tuning and fair cost.

These read run artifacts directly and deliberately do not share the dashboard's
predictions cache: the research plane browses arbitrary historical runs, while
the operational plane always looks at the latest one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from retail_forecasting.tracking import EXPERIMENT_FORECASTING_TUNING

logger = logging.getLogger(__name__)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame to JSON-safe records, turning NaN into None."""
    rows: list[dict[str, Any]] = frame.where(pd.notnull(frame), other=None).to_dict(
        orient="records"
    )
    return rows


def fair_cost(runs_with: Any) -> dict[str, Any]:
    """Latest fair inventory-cost backtest, or an empty payload.

    This is the apples-to-apples comparison: every strategy is charged against
    the same synthetically-censored ground truth, unlike the headline per-strategy
    cost which is biased by censoring.
    """
    candidates = runs_with("fair_cost_backtest.csv")
    if not candidates:
        return {"rows": [], "run": None}
    name, latest = next(iter(candidates.items()))
    return {
        "rows": records(pd.read_csv(latest / "fair_cost_backtest.csv")),
        "run": name,
    }


def _optional_csv(run_path: Path, filename: str) -> list[dict[str, Any]]:
    path = run_path / filename
    if not path.exists():
        return []
    try:
        return records(pd.read_csv(path))
    except Exception:
        logger.warning("Failed to read %s", path)
        return []


def pareto_front(runs_with: Any) -> dict[str, Any]:
    """Latest tuning run's ``tuning_pareto.csv`` — the Pinball vs Winkler front.

    Read from the tuning experiment rather than from the selected experiment run: the search
    is its own run mode, so its front never lived inside a walk-forward run and looking for
    it there found nothing.
    """
    candidates = runs_with("tuning_pareto.csv", experiment=EXPERIMENT_FORECASTING_TUNING)
    if not candidates:
        return {"rows": [], "run": None}
    name, latest = next(iter(candidates.items()))
    return {"rows": records(pd.read_csv(latest / "tuning_pareto.csv")), "run": name}


def sensitivity(run_path: Path) -> list[dict[str, Any]]:
    """``sensitivity_summary.csv`` — cost sensitivity to the Cs/Co ratio."""
    return _optional_csv(run_path, "sensitivity_summary.csv")


def run_summary(run_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Metrics and cost summaries for one run."""
    return {
        "metrics": _optional_csv(run_path, "metrics_summary.csv"),
        "costs": _optional_csv(run_path, "cost_summary.csv"),
    }
