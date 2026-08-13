"""Walk-forward operational simulation (the OPS plane).

Reads the artifact written by ``run_operational_simulation``
(``reports/<run>/simulation/predictions_by_day.parquet``) and indexes it by
weekly origin, so the dashboard can play the simulation back week by week and
compare retrain cadences. Pure artifact reads — no live compute.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pandas as pd

# The conformal band [q_0_1, q_0_9] is a nominal 80% interval (models.quantiles).
TARGET_COVERAGE = 0.80
LOWER_COLUMN = "q_0_1"
UPPER_COLUMN = "q_0_9"
HORIZON_DAYS = 7

ARTIFACT_GLOB = "*/simulation/predictions_by_day.parquet"


class SimulationNotFoundError(Exception):
    """No operational-simulation artifact exists under ``reports/``."""


class OpsSimulation:
    """Cached access to the walk-forward simulation artifact."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)
        self._lock = threading.Lock()
        self._frame: pd.DataFrame | None = None
        self._run_name: str | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._frame = None
            self._run_name = None

    @property
    def run_name(self) -> str:
        return self._run_name or "ops_sim"

    def frame(self) -> pd.DataFrame:
        """Load (and cache) the newest simulation artifact.

        Raises:
            SimulationNotFoundError: when no run has produced one.
        """
        if self._frame is not None:
            return self._frame

        if not self.reports_dir.exists():
            raise SimulationNotFoundError("No operational simulation artifact found.")
        candidates = sorted(self.reports_dir.glob(ARTIFACT_GLOB))
        if not candidates:
            raise SimulationNotFoundError("No operational simulation artifact found.")

        artifact = max(candidates, key=lambda p: p.stat().st_mtime)
        df = pd.read_parquet(artifact)
        df["decision_date"] = pd.to_datetime(df["decision_date"])
        # Weekly playback: origins sit on a 7-day grid, the retrain boundary.
        df["week_index"] = (df["day_index"] // 7).astype(int)
        df["is_weekly_origin"] = df["day_index"] % 7 == 0
        df["covered"] = (df["y_true"] >= df[LOWER_COLUMN]) & (df["y_true"] <= df[UPPER_COLUMN])

        with self._lock:
            self._frame = df
            self._run_name = artifact.parent.parent.name
        return df


def _cadence_block(group: pd.DataFrame) -> dict[str, Any]:
    """Aggregate one cadence at one weekly origin.

    Every metric -- cost included -- uses only fully-revealed rows. Cost used to be
    summed over the whole group on the grounds that an order placed is a cost
    incurred, but the cost of that order is computed *against* ``y_true``: on a
    partial week ``y_true`` is a truncated window sum, so the shortage half of the
    cost comes out artificially low. A partial week reports no metrics at all and
    the view flags it instead.
    """
    complete = group[group["actuals_complete"]]
    if complete.empty:
        return {
            "coverage": None,
            "total_cost": None,
            "mae": None,
            "n_series": int(group["series_id"].nunique()),
            "retrained": bool(group["retrained_this_step"].any()),
            "actuals_complete": False,
        }
    return {
        "coverage": float(complete["covered"].mean()),
        "total_cost": float(complete["total_cost"].sum()),
        "mae": float((complete["y_pred"] - complete["y_true"]).abs().mean()),
        "n_series": int(complete["series_id"].nunique()),
        "retrained": bool(group["retrained_this_step"].any()),
        "actuals_complete": bool(group["actuals_complete"].all()),
    }


def weekly_summary(simulation: OpsSimulation, series_limit: int = 60) -> dict[str, Any]:
    """Per-week metrics for every cadence, plus the series catalogue."""
    df = simulation.frame()
    weekly = df[df["is_weekly_origin"]]

    weeks = [
        {
            "week_index": int(week_index),
            "origin_date": week["decision_date"].iloc[0].date().isoformat(),
            "by_cadence": {
                str(cadence): _cadence_block(group) for cadence, group in week.groupby("cadence")
            },
        }
        for week_index, week in weekly.groupby("week_index")
    ]

    # Catalogue ordered by realized demand, liveliest series first.
    volume = weekly.groupby("series_id")["y_true"].sum().sort_values(ascending=False)

    return {
        "run": simulation.run_name,
        "horizon": HORIZON_DAYS,
        "target_coverage": TARGET_COVERAGE,
        "cadences": sorted(str(c) for c in df["cadence"].unique()),
        "series": [str(s) for s in volume.head(series_limit).index],
        "weeks": weeks,
    }


def series_trajectory(
    simulation: OpsSimulation,
    series_id: str,
    cadence: str = "every_7d",
) -> dict[str, Any]:
    """Weekly forecast/band/actual trajectory for one series under one cadence."""
    df = simulation.frame()
    selection = df[
        (df["series_id"].astype(str) == series_id)
        & (df["cadence"] == cadence)
        & df["is_weekly_origin"]
    ].sort_values("week_index")

    if selection.empty:
        raise SimulationNotFoundError(f"Series '{series_id}' not found in the simulation.")

    points = [
        {
            "week_index": int(row.week_index),
            "origin_date": row.decision_date.date().isoformat(),
            "y_pred": float(row.y_pred),
            "lower": float(getattr(row, LOWER_COLUMN)),
            "upper": float(getattr(row, UPPER_COLUMN)),
            "y_true": None if pd.isna(row.y_true) else float(row.y_true),
            "order_quantity": float(row.order_quantity),
            "total_cost": float(row.total_cost),
            "covered": bool(row.covered),
            "actuals_complete": bool(row.actuals_complete),
        }
        for row in selection.itertuples(index=False)
    ]
    return {"series_id": series_id, "cadence": cadence, "points": points}
