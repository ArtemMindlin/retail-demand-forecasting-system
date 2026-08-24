"""Render the per-fold conformal coverage figure used in the results chapter.

The figure contrasts the base subset (50 series) against the scale validation
(500 series) for one model, showing that the fold-to-fold variance narrows while a
residual gap to the nominal level persists.

It reads `fold_metrics.csv` from two run directories, so the figure is reproducible
from artifacts instead of being a hand-made PDF with no source.

Usage:
    python scripts/plot_coverage_folds.py \
        --base   fresh_retailnet_v2_20260811_123002 \
        --scale  fresh_retailnet_large_20260811_125735
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from retail_forecasting.tracking import resolve_run_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NOMINAL = 0.80
OUTPUT = Path("memoria/figures/cobertura_folds.pdf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=resolve_run_dir,
        required=True,
        help="Base-subset run directory, or the name of a recorded run.",
    )
    parser.add_argument(
        "--scale",
        type=resolve_run_dir,
        required=True,
        help="Scale run directory, or the name of a recorded run.",
    )
    parser.add_argument("--model", default="catboost", help="Model to plot.")
    parser.add_argument("--strategy", default="Latent_supervised", help="Data strategy to plot.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def coverage_by_fold(run_dir: Path, model: str, strategy: str) -> pd.Series:
    frame = pd.read_csv(run_dir / "fold_metrics.csv")
    column = next(c for c in frame.columns if "coverage" in c)
    selected = frame[(frame["model_name"] == model) & (frame["data_strategy"] == strategy)]
    if selected.empty:
        raise SystemExit(f"No rows for {model}/{strategy} in {run_dir}")
    return selected.set_index("fold_id")[column].sort_index()


def main() -> None:
    args = build_parser().parse_args()
    base = coverage_by_fold(args.base, args.model, args.strategy)
    scale = coverage_by_fold(args.scale, args.model, args.strategy)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    folds = [f"Fold {i}" for i in base.index]

    for series, label, colour, marker in (
        (base, f"Subset base (50 series) · media {base.mean():.1%}", "#c2410c", "o"),
        (scale, f"Escala (500 series) · media {scale.mean():.1%}", "#1d4ed8", "s"),
    ):
        ax.plot(folds, series.to_numpy() * 100, marker=marker, color=colour, label=label, lw=1.8)
        ax.axhline(series.mean() * 100, color=colour, ls=":", lw=1.0, alpha=0.7)

    ax.axhline(NOMINAL * 100, color="#334155", ls="--", lw=1.2)
    ax.annotate(
        f"Cobertura nominal {NOMINAL:.0%}",
        xy=(len(folds) - 1, NOMINAL * 100),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#334155",
    )

    ax.set_ylabel("Cobertura empírica $[q_{0,1}, q_{0,9}]$ (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"✅ Wrote {args.output}")
    print(f"   base  : {[f'{v:.1%}' for v in base]} → media {base.mean():.1%}")
    print(f"   escala: {[f'{v:.1%}' for v in scale]} → media {scale.mean():.1%}")


if __name__ == "__main__":
    main()
