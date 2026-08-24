"""Historical-run browser: latent-demand imputation, Pareto tuning, fair cost.

These read run artifacts directly and deliberately do not share the dashboard's
predictions cache: the research plane browses arbitrary historical runs, while
the operational plane always looks at the latest one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Display metadata for the three imputation strategies compared in the run.
STRATEGY_META: dict[str, dict[str, str]] = {
    "supervised": {
        "color": "var(--c-conf)",
        "label": "Supervised (LGBM)",
        "short": "Supervised",
    },
    "historical_mean": {"color": "#60a5fa", "label": "Historical mean", "short": "Hist. mean"},
    "clipped_scaling": {"color": "#a78bfa", "label": "Clipped scaling", "short": "Clipped scal."},
}

DEFAULT_STRATEGY_COLOR = "#94a3b8"


def strategy_meta(name: str) -> dict[str, str]:
    """Colour and labels for a strategy, with a neutral fallback."""
    return STRATEGY_META.get(name, {"color": DEFAULT_STRATEGY_COLOR, "label": name, "short": name})


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame to JSON-safe records, turning NaN into None."""
    rows: list[dict[str, Any]] = frame.where(pd.notnull(frame), other=None).to_dict(
        orient="records"
    )
    return rows


def imputation_strategies(run_path: Path, series_id: str | None = None) -> dict[str, Any]:
    """Per-strategy latent reconstruction for one series, plus run-level quality.

    ``observed`` and ``stockout_hours`` are strategy-invariant, so they are read
    once from the de-duplicated slice rather than per strategy.
    """
    frame = pd.read_csv(run_path / "latent_strategies.csv")
    if frame.empty:
        return {"series": [], "dates": [], "observed": [], "strategies": {}, "quality": []}

    series = sorted(frame["series_id"].astype(str).unique())
    chosen = series_id if series_id in series else series[0]
    subset = frame[frame["series_id"].astype(str) == chosen].sort_values("date")

    dates = sorted(subset["date"].unique())
    base = subset.drop_duplicates(subset=["date"]).set_index("date")

    strategies: dict[str, list[float | None]] = {}
    for name in sorted(subset["strategy"].unique()):
        values = subset[subset["strategy"] == name].set_index("date")["latent_demand_est"]
        strategies[str(name)] = [
            float(values[d]) if d in values.index and pd.notna(values[d]) else None for d in dates
        ]

    def column(name: str) -> list[float | None]:
        return [float(base.loc[d, name]) if pd.notna(base.loc[d, name]) else None for d in dates]

    quality: list[dict[str, Any]] = []
    quality_path = run_path / "imputation_quality.csv"
    if quality_path.exists():
        quality = records(pd.read_csv(quality_path))

    return {
        "series": series,
        "series_id": chosen,
        "dates": [str(d) for d in dates],
        "observed": column("observed"),
        "stockout_hours": column("stockout_hours"),
        "strategies": strategies,
        "quality": quality,
    }


def rank_quality(quality: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strategies ordered by reconstruction MAE, best first, with display metadata.

    ``bias`` is read as a direction: materially negative means the strategy
    under-imputes the lost demand, positive means it over-imputes.
    """
    ranked = sorted((q for q in quality if q.get("mae") is not None), key=lambda q: float(q["mae"]))
    worst_mae = max((float(q["mae"]) for q in ranked), default=1e-9) or 1e-9

    rows = []
    for index, entry in enumerate(ranked):
        bias = float(entry.get("bias") or 0.0)
        meta = strategy_meta(str(entry.get("strategy")))
        if bias < -0.5:
            direction = "infra-imputa"
        elif bias > 0.5:
            direction = "sobre-imputa"
        else:
            direction = "—"

        if abs(bias) < 0.2:
            bias_color = "var(--c-conf)"
        elif bias < 0:
            bias_color = "#ef4444"
        else:
            bias_color = "#f59e0b"

        rows.append(
            {
                **entry,
                "label": meta["label"],
                "color": meta["color"],
                "is_best": index == 0,
                "bar_pct": (float(entry["mae"]) / worst_mae) * 100,
                "direction": direction,
                "bias_color": bias_color,
            }
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


def pareto_front(run_path: Path) -> list[dict[str, Any]]:
    """``tuning_pareto.csv`` — the Pinball vs Winkler trade-off front."""
    return _optional_csv(run_path, "tuning_pareto.csv")


def sensitivity(run_path: Path) -> list[dict[str, Any]]:
    """``sensitivity_summary.csv`` — cost sensitivity to the Cs/Co ratio."""
    return _optional_csv(run_path, "sensitivity_summary.csv")


def run_summary(run_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Metrics and cost summaries for one run."""
    return {
        "metrics": _optional_csv(run_path, "metrics_summary.csv"),
        "costs": _optional_csv(run_path, "cost_summary.csv"),
    }
