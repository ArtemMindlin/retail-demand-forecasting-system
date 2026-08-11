"""Render the MAE-versus-simulated-cost figure used in the results chapter.

Shows the dissociation that is the central result of the work: the ranking of the
models by point error is the inverse of their ranking by simulated logistic cost.

Reads `metrics_summary.csv` and `cost_summary.csv` from one run directory, so the
figure is reproducible from artifacts instead of being a hand-made PDF.

Usage:
    python scripts/plot_mae_vs_cost.py --run reports/fresh_retailnet_large_20260811_125735
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT = Path("memoria/figures/mae_vs_coste.pdf")
LABELS = {"seasonal_naive": "Seasonal Naïve", "lightgbm": "LightGBM", "catboost": "CatBoost"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory to read.")
    parser.add_argument("--strategy", default="Latent_supervised", help="Data strategy to plot.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def load_frame(run_dir: Path, strategy: str) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "metrics_summary.csv")
    costs = pd.read_csv(run_dir / "cost_summary.csv")
    cost_column = "sim_total_cost" if "sim_total_cost" in costs.columns else "total_cost"

    merged = metrics.merge(
        costs[["data_strategy", "model_name", cost_column]],
        on=["data_strategy", "model_name"],
    )
    merged = merged[merged["data_strategy"] == strategy].copy()
    if merged.empty:
        raise SystemExit(f"No rows for strategy {strategy} in {run_dir}")
    merged = merged.rename(columns={cost_column: "cost"}).sort_values("mae")
    merged["label"] = merged["model_name"].map(lambda m: LABELS.get(m, m))
    return merged


def main() -> None:
    args = build_parser().parse_args()
    frame = load_frame(args.run, args.strategy)

    fig, (ax_mae, ax_cost) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    positions = range(len(frame))
    colours = ["#0f766e", "#1d4ed8", "#c2410c"]

    ax_mae.barh(list(positions), frame["mae"], color=colours, height=0.55)
    ax_mae.set_yticks(list(positions), frame["label"])
    ax_mae.set_xlabel("MAE (unidades)")
    ax_mae.set_title("Error puntual · menor es mejor", fontsize=10)
    for y, value in zip(positions, frame["mae"], strict=True):
        ax_mae.annotate(
            f"{value:.2f}",
            (value, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax_cost.barh(list(positions), frame["cost"] / 1000.0, color=colours, height=0.55)
    ax_cost.set_yticks(list(positions), frame["label"])
    ax_cost.set_xlabel("Coste simulado Order-Up-To (miles de u.m.)")
    ax_cost.set_title("Coste logístico · menor es mejor", fontsize=10)
    for y, value in zip(positions, frame["cost"], strict=True):
        ax_cost.annotate(
            f"{value / 1000:.0f}k",
            (value / 1000.0, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    for axis in (ax_mae, ax_cost):
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.16)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"✅ Wrote {args.output}")
    print(frame[["label", "mae", "cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
