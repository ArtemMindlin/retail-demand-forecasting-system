"""Compute and plot empirical Pareto frontiers for the 60 real trials of LightGBM and CatBoost."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Compute non-dominated Pareto front points (minimizing both x and y)."""
    sorted_df = df.sort_values(x_col).copy()
    is_front = []
    min_y = float("inf")
    for val in sorted_df[y_col]:
        if val < min_y:
            is_front.append(True)
            min_y = val
        else:
            is_front.append(False)
    sorted_df["is_on_front"] = is_front
    return sorted_df


def main() -> None:
    client = mlflow.tracking.MlflowClient("sqlite:///mlflow.db")
    runs = client.search_runs(experiment_ids=["9"], max_results=500)

    records = []
    for r in runs:
        m = r.data.metrics
        p = r.data.params
        t = r.data.tags
        cost = m.get("trial_cost") or m.get("cost")
        if cost is None or cost > 50:
            continue

        backend = "catboost" if "l2_leaf_reg" in p or "border_count" in p else "lightgbm"
        n_est = int(float(p.get("n_estimators", 200)))
        lr = float(p.get("learning_rate", 0.05))
        max_d = int(float(p.get("max_depth", 6)))

        records.append(
            {
                "run_id": r.info.run_id,
                "run_name": t.get("mlflow.runName", r.info.run_id),
                "backend": backend,
                "cost": float(cost),
                "n_estimators": n_est,
                "learning_rate": lr,
                "max_depth": max_d,
            }
        )

    all_df = pd.DataFrame(records)

    # Isolate the exact 60 trials for LightGBM and 60 trials for CatBoost
    cat_df = all_df[all_df["backend"] == "catboost"].head(60)
    lgb_df = all_df[all_df["backend"] == "lightgbm"].head(60)

    cat_pareto = compute_pareto_front(cat_df, "cost", "n_estimators")
    lgb_pareto = compute_pareto_front(lgb_df, "cost", "n_estimators")

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_pareto.to_csv(output_dir / "pareto_catboost_60trials.csv", index=False)
    lgb_pareto.to_csv(output_dir / "pareto_lightgbm_60trials.csv", index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    # Panel 1: LightGBM Pareto
    lgb_dom = lgb_pareto[~lgb_pareto["is_on_front"]]
    lgb_front = lgb_pareto[lgb_pareto["is_on_front"]].sort_values("cost")
    lgb_winner = lgb_pareto.loc[lgb_pareto["cost"].idxmin()]

    ax1.scatter(
        lgb_dom["cost"],
        lgb_dom["n_estimators"],
        color="#94a3b8",
        alpha=0.45,
        s=35,
        label="Ensayos dominados (LGBM)",
    )
    ax1.scatter(
        lgb_front["cost"],
        lgb_front["n_estimators"],
        color="#2563eb",
        s=80,
        zorder=4,
        label="Frontera de Pareto",
    )
    ax1.plot(
        lgb_front["cost"],
        lgb_front["n_estimators"],
        color="#2563eb",
        linestyle="--",
        alpha=0.6,
        zorder=3,
    )
    ax1.scatter(
        [lgb_winner["cost"]],
        [lgb_winner["n_estimators"]],
        color="#dc2626",
        s=130,
        marker="*",
        zorder=5,
        label=f"Ganador (N={int(lgb_winner['n_estimators'])})",
    )

    ax1.set_xlabel("Coste de Inventario (Menor es mejor)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Complejidad (n_estimators)", fontsize=11, fontweight="bold")
    ax1.set_title("Frente de Pareto · LightGBM (60 Ensayos Reales)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)

    # Panel 2: CatBoost Pareto
    cat_dom = cat_pareto[~cat_pareto["is_on_front"]]
    cat_front = cat_pareto[cat_pareto["is_on_front"]].sort_values("cost")
    cat_winner = cat_pareto.loc[cat_pareto["cost"].idxmin()]

    ax2.scatter(
        cat_dom["cost"],
        cat_dom["n_estimators"],
        color="#94a3b8",
        alpha=0.45,
        s=35,
        label="Ensayos dominados (CatBoost)",
    )
    ax2.scatter(
        cat_front["cost"],
        cat_front["n_estimators"],
        color="#f97316",
        s=80,
        zorder=4,
        label="Frontera de Pareto",
    )
    ax2.plot(
        cat_front["cost"],
        cat_front["n_estimators"],
        color="#f97316",
        linestyle="--",
        alpha=0.6,
        zorder=3,
    )
    ax2.scatter(
        [cat_winner["cost"]],
        [cat_winner["n_estimators"]],
        color="#dc2626",
        s=130,
        marker="*",
        zorder=5,
        label=f"Ganador (N={int(cat_winner['n_estimators'])})",
    )

    ax2.set_xlabel("Coste de Inventario (Menor es mejor)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Complejidad (n_estimators)", fontsize=11, fontweight="bold")
    ax2.set_title("Frente de Pareto · CatBoost (60 Ensayos Reales)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "pareto_front_60trials.png")
    plt.close(fig)

    print("Generated pareto_front_60trials.png successfully!")


if __name__ == "__main__":
    main()
