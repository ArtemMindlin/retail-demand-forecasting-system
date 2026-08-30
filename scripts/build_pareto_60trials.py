"""Compute and plot empirical Pareto frontiers (Cost vs. Winkler Score) for LightGBM and CatBoost."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    memoria_fig_dir = Path("memoria/figures")
    memoria_fig_dir.mkdir(parents=True, exist_ok=True)

    lgb_df = pd.read_csv(output_dir / "pareto_lightgbm.csv")
    cat_df = pd.read_csv(output_dir / "pareto_catboost.csv")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    # Panel 1: LightGBM Pareto
    lgb_dom = lgb_df[~lgb_df["is_on_front"]]
    lgb_front = lgb_df[lgb_df["is_on_front"]].sort_values("cost")
    lgb_winner = lgb_df[lgb_df["is_selected"]].iloc[0]

    ax1.scatter(
        lgb_dom["cost"],
        lgb_dom["winkler"],
        color="#94a3b8",
        alpha=0.45,
        s=35,
        label="Ensayos dominados (LGBM)",
    )
    ax1.scatter(
        lgb_front["cost"],
        lgb_front["winkler"],
        color="#2563eb",
        s=80,
        zorder=4,
        label="Frontera de Pareto",
    )
    ax1.plot(
        lgb_front["cost"],
        lgb_front["winkler"],
        color="#2563eb",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        zorder=3,
    )
    ax1.scatter(
        [lgb_winner["cost"]],
        [lgb_winner["winkler"]],
        color="#dc2626",
        s=140,
        marker="*",
        zorder=5,
        label=f"Óptimo Seleccionado (Trial {int(lgb_winner['trial_number']) + 1})",
    )

    ax1.set_xlabel("Coste Logístico de Inventario (Menor es mejor)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Winkler Score P10–P90 (Menor es mejor)", fontsize=11, fontweight="bold")
    ax1.set_title("Frente de Pareto · LightGBM (60 Ensayos)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    # Panel 2: CatBoost Pareto
    cat_dom = cat_df[~cat_df["is_on_front"]]
    cat_front = cat_df[cat_df["is_on_front"]].sort_values("cost")
    cat_winner = cat_df[cat_df["is_selected"]].iloc[0]

    ax2.scatter(
        cat_dom["cost"],
        cat_dom["winkler"],
        color="#94a3b8",
        alpha=0.45,
        s=35,
        label="Ensayos dominados (CatBoost)",
    )
    ax2.scatter(
        cat_front["cost"],
        cat_front["winkler"],
        color="#ea580c",
        s=80,
        zorder=4,
        label=f"Frontera de Pareto ({len(cat_front)} soluciones)",
    )
    ax2.plot(
        cat_front["cost"],
        cat_front["winkler"],
        color="#ea580c",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        zorder=3,
    )
    ax2.scatter(
        [cat_winner["cost"]],
        [cat_winner["winkler"]],
        color="#dc2626",
        s=140,
        marker="*",
        zorder=5,
        label=f"Óptimo Seleccionado (Trial {int(cat_winner['trial_number']) + 1})",
    )

    ax2.set_xlabel("Coste Logístico de Inventario (Menor es mejor)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Winkler Score P10–P90 (Menor es mejor)", fontsize=11, fontweight="bold")
    ax2.set_title("Frente de Pareto · CatBoost (60 Ensayos)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "pareto_front_60trials.png")
    fig.savefig(memoria_fig_dir / "pareto_front_60trials.png")
    plt.close(fig)

    print(
        f"Generated {output_dir / 'pareto_front_60trials.png'} and copied to memoria/figures successfully!"
    )


if __name__ == "__main__":
    main()
