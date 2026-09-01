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
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    memoria_fig_dir = Path("memoria/figures")
    memoria_fig_dir.mkdir(parents=True, exist_ok=True)

    # Load latest LightGBM biobjective Pareto trials
    pareto_lgbm_path = output_dir / "pareto_lightgbm.csv"
    if pareto_lgbm_path.exists():
        raw_lgb = pd.read_csv(pareto_lgbm_path)
        lgb_df = pd.DataFrame(
            {
                "trial": raw_lgb["trial_number"] + 1,
                "cost": raw_lgb["cost"],
                "best_so_far": raw_lgb["cost"].cummin(),
            }
        )
    else:
        log_path = Path("var/tune_forecasting_chain.log")
        data = parse_tuning_log(log_path)
        lgb_df = data["lightgbm"]

    # Load latest CatBoost biobjective Pareto trials
    pareto_cat_path = output_dir / "pareto_catboost.csv"
    if pareto_cat_path.exists():
        raw_cat = pd.read_csv(pareto_cat_path)
        cat_df = pd.DataFrame(
            {
                "trial": raw_cat["trial_number"] + 1,
                "cost": raw_cat["cost"],
                "best_so_far": raw_cat["cost"].cummin(),
            }
        )
    else:
        log_path = Path("var/tune_forecasting_chain.log")
        data = parse_tuning_log(log_path)
        cat_df = data["catboost"]

    # Save CSVs
    lgb_df.to_csv(output_dir / "tuning_lightgbm_60trials.csv", index=False)
    cat_df.to_csv(output_dir / "tuning_catboost_60trials.csv", index=False)

    # Configure publication styling
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.8,
            "figure.autolayout": True,
        }
    )

    # Figure: 2-Panel Tuning Analysis (Convergence + Out-of-Sample Stability)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 2.85), dpi=300)

    # Panel A: Convergence Curve in Selection (N=30 draws)
    ax1.scatter(
        lgb_df["trial"],
        lgb_df["cost"],
        color="#3b82f6",
        alpha=0.35,
        s=20,
        edgecolors="none",
    )
    ax1.scatter(
        cat_df["trial"],
        cat_df["cost"],
        color="#f97316",
        alpha=0.35,
        s=20,
        edgecolors="none",
    )
    ax1.plot(
        lgb_df["trial"],
        lgb_df["best_so_far"],
        color="#2563eb",
        linewidth=2.2,
        label=r"LightGBM (mín: $22{,}07$)",
    )
    ax1.plot(
        cat_df["trial"],
        cat_df["best_so_far"],
        color="#ea580c",
        linewidth=2.2,
        label=r"CatBoost (mín: $23{,}77$)",
    )

    ax1.set_xlabel("Número de ensayo (Optuna)", fontweight="semibold")
    ax1.set_ylabel("Coste en selección (u.m.)", fontweight="semibold")
    ax1.set_title("A. Convergencia en Selección ($N=30$)", fontweight="bold")
    ax1.set_xlim(0, 61)
    ax1.set_ylim(21.5, 32.5)
    ax1.set_xticks([1, 15, 30, 45, 60])
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="upper right", framealpha=0.92, edgecolor="#cbd5e1")

    # Panel B: In-Sample vs Out-of-Sample vs Defaults (N=25 draws)
    stages = ["Selección\n(In-Sample)", "Validación\n(Out-Sample)", "Por Defecto\n(Baseline)"]
    import numpy as np

    x = np.arange(len(stages))
    width = 0.35

    lgb_vals = [22.07, 34.03, 44.00]
    cat_vals = [23.77, 29.93, 33.26]

    rects1 = ax2.bar(x - width / 2, lgb_vals, width, label="LightGBM", color="#2563eb", alpha=0.88)
    rects2 = ax2.bar(x + width / 2, cat_vals, width, label="CatBoost", color="#ea580c", alpha=0.88)

    ax2.set_ylabel("Coste logístico de inventario (u.m.)", fontweight="semibold")
    ax2.set_title("B. Estabilidad Fuera de Muestra ($N=25$)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontweight="semibold")
    ax2.set_ylim(0, 52)
    ax2.grid(axis="y", linestyle=":", alpha=0.55)
    ax2.legend(loc="upper left", framealpha=0.92, edgecolor="#cbd5e1")

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    # Save vector PDF and PNG
    for path in [
        output_dir / "tuning_comparison_60trials.pdf",
        output_dir / "tuning_comparison_60trials.png",
        memoria_fig_dir / "tuning_comparison_60trials.pdf",
        memoria_fig_dir / "tuning_comparison_60trials.png",
    ]:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    print("Generated tuning_comparison_60trials.pdf and PNG successfully!")


if __name__ == "__main__":
    main()
