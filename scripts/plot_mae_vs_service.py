"""Render the MAE-versus-service figure used in the results chapter.

Shows the dissociation that is the central result of the work: ranking the models by
point error says one thing, inventory cost barely separates them, and service level
reverses the order outright.

Reads `metrics_summary.csv` and `cost_summary.csv` from one run directory, so the
figure is reproducible from artifacts instead of being a hand-made PDF.

Usage:
    python scripts/plot_mae_vs_service.py --run <corrida de experimento>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from retail_forecasting.tracking import resolve_run_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT = Path("memoria/figures/mae_vs_servicio.pdf")
LABELS = {"seasonal_naive": "Seasonal Naïve", "lightgbm": "LightGBM", "catboost": "CatBoost"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=resolve_run_dir,
        required=True,
        help="Run directory, or the name of a recorded run.",
    )
    parser.add_argument("--strategy", default="Latent_supervised", help="Data strategy to plot.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def load_frame(run_dir: Path, strategy: str) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "metrics_summary.csv")
    costs = pd.read_csv(run_dir / "cost_summary.csv")

    merged = metrics.merge(
        costs[["data_strategy", "model_name", "total_cost", "service_level"]],
        on=["data_strategy", "model_name"],
    )
    merged = merged[merged["data_strategy"] == strategy].copy()
    if merged.empty:
        raise SystemExit(f"No rows for strategy {strategy} in {run_dir}")
    merged = merged.rename(columns={"total_cost": "cost"}).sort_values("mae")
    merged["service_pct"] = merged["service_level"] * 100.0
    merged["label"] = merged["model_name"].map(lambda m: LABELS.get(m, m))
    return merged


def main() -> None:
    args = build_parser().parse_args()
    frame = load_frame(args.run, args.strategy)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
        }
    )

    fig, (ax_mae, ax_cost, ax_svc) = plt.subplots(1, 3, figsize=(7.2, 2.4))
    positions = range(len(frame))
    colours = ["#0f766e", "#1d4ed8", "#c2410c"]

    # 1. Error puntual (MAE)
    ax_mae.barh(list(positions), frame["mae"], color=colours, height=0.55)
    ax_mae.set_yticks(list(positions), frame["label"])
    ax_mae.set_xlabel("MAE (unidades)")
    ax_mae.set_title("Error puntual · menor es mejor", pad=7)
    for y, value in zip(positions, frame["mae"], strict=True):
        ax_mae.annotate(
            f"{value:.2f}",
            (value, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="medium",
        )

    # 2. Coste logístico
    ax_cost.barh(list(positions), frame["cost"] / 1000.0, color=colours, height=0.55)
    ax_cost.set_yticks(list(positions), frame["label"])
    ax_cost.set_xlabel("Coste inventario (miles u.m.)")
    ax_cost.set_title("Coste · no discrimina", pad=7)
    for y, value in zip(positions, frame["cost"], strict=True):
        ax_cost.annotate(
            f"{value / 1000:.0f}k",
            (value / 1000.0, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="medium",
        )

    # 3. Nivel de servicio
    ax_svc.barh(list(positions), frame["service_pct"], color=colours, height=0.55)
    ax_svc.set_yticks(list(positions), frame["label"])
    ax_svc.set_xlabel("Nivel de servicio (%)")
    ax_svc.set_title("Servicio · mayor es mejor", pad=7)
    for y, value in zip(positions, frame["service_pct"], strict=True):
        ax_svc.annotate(
            f"{value:.1f}%",
            (value, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="medium",
        )

    for axis in (ax_mae, ax_cost, ax_svc):
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.20)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"✅ Wrote {args.output}")
    print(frame[["label", "mae", "cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
