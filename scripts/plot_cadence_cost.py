"""Render the cumulative-cost curve per retraining cadence for the results chapter.

The streaming simulation reaches the thesis as a summary table, which reports where the
two policies END but not how they got there. This figure plots the running total over the
evaluation window, which is what shows whether the gap opens steadily (the static model
decaying as it ages) or comes from one bad week.

It draws only the NON-OVERLAPPING origins the comparison is computed on, the same ones
`_independent_origins` keeps, so the curve cannot suggest more evidence than the interval
in the caption is based on: each marker is one of the independent decision points.

Usage:
    python scripts/plot_cadence_cost.py --run <corrida de simulate_ops>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd

from retail_forecasting.config import load_config
from retail_forecasting.simulation.operations import _independent_origins
from retail_forecasting.tracking import resolve_run_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT = Path("memoria/figures/cadence_cost.pdf")
logger = logging.getLogger(__name__)

# The static baseline is the one whose cost is expected to run away, so it gets the warm
# colour; the active policy keeps the cool one used for the "good" series elsewhere.
_COLOURS = {"never": "#c2410c", "every_7d": "#1d4ed8"}
_LABELS = {"never": "nunca (modelo estático)", "every_7d": "cada 7 días"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=resolve_run_dir, required=True)
    parser.add_argument("--config", default="configs/simulate_ops/default.yaml")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()

    simulation = args.run / "simulation"
    frame = pd.read_parquet(simulation / "predictions_by_day.parquet")
    horizon = int(load_config(args.config).dataset.horizon)

    # Reuses the pipeline's own definition of an independent origin rather than
    # re-deriving it: the grid and the "actuals landed" filter both matter, and a curve
    # built on a different set of origins would not add up to the table beside it.
    independent = _independent_origins(frame, horizon)
    by_origin = (
        independent.groupby(["cadence", "decision_date"])["total_cost"]
        .sum()
        .rename("cost")
        .reset_index()
        .sort_values("decision_date")
    )
    kept = sorted(by_origin["decision_date"].unique())

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for cadence, group in by_origin.groupby("cadence"):
        cumulative = group["cost"].cumsum()
        ax.plot(
            group["decision_date"],
            cumulative,
            marker="o",
            markersize=4,
            lw=1.9,
            color=_COLOURS.get(str(cadence), "#334155"),
            label=_LABELS.get(str(cadence), str(cadence)),
        )
        logger.info("  %s: coste acumulado final %.0f", cadence, cumulative.iloc[-1])

    # One tick per origin, short-form: the default date locator picks its own grid and
    # collides the labels, and the origins are what the reader needs to count anyway.
    ax.set_xticks(list(kept))
    ax.set_xticklabels([pd.Timestamp(d).strftime("%d/%m") for d in kept])
    ax.set_ylabel("Coste de inventario acumulado (u.m.)")
    ax.set_xlabel("Origen de decisión")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    logger.info("escrito %s (%d orígenes independientes)", args.output, len(kept))


if __name__ == "__main__":
    main()
