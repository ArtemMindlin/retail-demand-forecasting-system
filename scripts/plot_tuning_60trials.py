"""Plot the exact 60-trial tuning history and Pareto convergence for LightGBM and CatBoost."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_tuning_log(log_path: Path) -> dict[str, pd.DataFrame]:
    content = log_path.read_text(encoding="utf-8")

    sections = {
        "lightgbm": content[
            content.find("TUNING DE FORECASTING · LIGHTGBM") : content.find(
                "TUNING DE FORECASTING · CATBOOST"
            )
        ],
        "catboost": content[content.find("TUNING DE FORECASTING · CATBOOST") :],
    }

    results = {}
    pattern = re.compile(r"^\s*(\d+)/60\s+([\d\.]+)\s+([\d\.]+)", re.MULTILINE)

    for backend, text in sections.items():
        matches = pattern.findall(text)
        records = [
            {"trial": int(m[0]), "cost": float(m[1]), "best_so_far": float(m[2])} for m in matches
        ]
        results[backend] = pd.DataFrame(records)

    return results


def main() -> None:
    log_path = Path("var/tune_forecasting_chain.log")
    data = parse_tuning_log(log_path)

    lgb_df = data["lightgbm"]
    cat_df = data["catboost"]

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    lgb_df.to_csv(output_dir / "tuning_lightgbm_60trials.csv", index=False)
    cat_df.to_csv(output_dir / "tuning_catboost_60trials.csv", index=False)

    # Figure 1: Convergence and Trial Distribution (60 trials per backend)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Subplot 1: Convergence Curve (Best Cost So Far)
    ax1.plot(
        lgb_df["trial"],
        lgb_df["best_so_far"],
        color="#3b82f6",
        linewidth=2.5,
        label="LightGBM (Mejor: 22.39)",
    )
    ax1.scatter(
        lgb_df["trial"],
        lgb_df["cost"],
        color="#3b82f6",
        alpha=0.35,
        s=25,
        label="LightGBM (trials)",
    )

    ax1.plot(
        cat_df["trial"],
        cat_df["best_so_far"],
        color="#f97316",
        linewidth=2.5,
        label="CatBoost (Mejor: 23.97)",
    )
    ax1.scatter(
        cat_df["trial"],
        cat_df["cost"],
        color="#f97316",
        alpha=0.35,
        s=25,
        label="CatBoost (trials)",
    )

    ax1.set_xlabel("Número de Ensayo (Trial 1 a 60)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Coste de Selección (In-Sample)", fontsize=11, fontweight="bold")
    ax1.set_title("Convergencia del Tuning (60 Trials por Backend)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)

    # Subplot 2: Out-of-Sample Validation Verdict vs. Defaults
    backends = ["LightGBM", "CatBoost"]
    defaults = [44.00, 33.26]
    winners = [33.66, 28.89]
    gains = [-23.52, -13.15]

    x = [0, 1]
    width = 0.32
    ax2.bar(
        [p - width / 2 for p in x],
        defaults,
        width,
        label="Configuración por defecto",
        color="#94a3b8",
    )
    ax2.bar(
        [p + width / 2 for p in x],
        winners,
        width,
        label="Ganador del Tuning (60 trials)",
        color="#10b981",
    )

    ax2.set_ylabel(
        "Coste de Inventario en Validación (Out-of-Sample)", fontsize=11, fontweight="bold"
    )
    ax2.set_title("Veredicto Final en Validación (25 Extracciones)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(backends, fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    # Annotate bars
    for i in range(2):
        ax2.annotate(
            f"{gains[i]:.1f}%\n(IC95)",
            (x[i] + width / 2, winners[i] / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "tuning_comparison_60trials.png")
    plt.close(fig)

    print(f"Generated {output_dir / 'tuning_comparison_60trials.png'} successfully!")


if __name__ == "__main__":
    main()
