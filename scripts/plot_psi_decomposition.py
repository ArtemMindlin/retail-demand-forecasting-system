"""Render the per-bin decomposition of one feature's PSI for the results chapter.

The PSI reaches the thesis as a single number per feature, which says that a distribution
moved but not how. This figure opens one of them up: because the bin edges are the deciles
of the reference window, every reference bin holds exactly 10% of the observations, so the
current histogram can be read directly against that flat line. A bar above it is a decile
that gained mass and one below is a decile that lost it.

Reads `drift_report.json` from a run, which already carries the `pre` and `post` histograms
precisely so they can be plotted without recomputing anything.

Usage:
    python scripts/plot_psi_decomposition.py --run <corrida> [--feature demand_roll_mean_14]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

from retail_forecasting.tracking import resolve_run_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT = Path("memoria/figures/psi_decomposition.pdf")
logger = logging.getLogger(__name__)

_GAIN = "#c2410c"
_LOSS = "#1d4ed8"
_RULE = "#334155"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=resolve_run_dir, required=True)
    parser.add_argument("--feature", default="demand_roll_mean_14")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()

    report = json.loads((args.run / "drift_report.json").read_text(encoding="utf-8"))
    matches = [row for row in report if row["name"] == args.feature]
    if not matches:
        raise SystemExit(
            f"{args.feature} no está en drift_report.json. Disponibles: "
            + ", ".join(row["name"] for row in report)
        )
    record = matches[0]

    pre = np.asarray(record["pre"], dtype=float)
    post = np.asarray(record["post"], dtype=float)
    # The published PSI is recomputed here rather than read, so the figure cannot disagree
    # with its own caption if the stored value ever comes from a different code path.
    contribution = (post - pre) * np.log(post / pre)
    bins = np.arange(1, len(post) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colours = [_GAIN if p > r else _LOSS for p, r in zip(post, pre, strict=True)]
    ax.bar(bins, post * 100, color=colours, width=0.68, zorder=3)
    ax.axhline(
        pre[0] * 100, color=_RULE, ls="--", lw=1.2, zorder=4, label="referencia (10 % por tramo)"
    )

    # Label only the deciles that carry the index, so the annotation is not wallpaper.
    for index in np.argsort(contribution)[::-1][:3]:
        ax.annotate(
            f"aporta {contribution[index]:.3f}".replace(".", ","),
            xy=(bins[index], post[index] * 100),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=_RULE,
        )

    ax.set_xticks(bins)
    ax.set_xlabel("Decil de la distribución de referencia")
    ax.set_ylabel("Masa de la ventana actual (%)")
    ax.set_ylim(0, max(post * 100) * 1.22)
    ax.grid(axis="y", alpha=0.25, lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    logger.info("escrito %s", args.output)
    logger.info(
        "  %s · PSI recalculado %.4f (reportado %.4f)",
        args.feature,
        contribution.sum(),
        record["psi"],
    )


if __name__ == "__main__":
    main()
