from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from retail_forecasting.config import load_config
from retail_forecasting.data.dataset import load_prepared_panel


def _load_profiles() -> tuple[pd.DataFrame, dict[str, str], pd.Series]:
    settings = load_config("configs/experiment.yaml")
    eda_dataset_config = settings.dataset.model_copy(
        update={"top_n_series": None, "max_rows": None}
    )
    panel = load_prepared_panel(eda_dataset_config, settings.preprocessing, split="train")

    min_obs = 500
    counts = panel.groupby("third_category_id")["observed_demand"].count()
    valid_categories = counts[counts >= min_obs].index
    panel_valid = panel[panel["third_category_id"].isin(valid_categories)].copy()

    panel_valid["date"] = pd.to_datetime(panel_valid["date"])
    panel_valid["day_of_week"] = panel_valid["date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    pivot_df = (
        panel_valid.groupby(["third_category_id", "day_of_week"])["observed_demand"]
        .mean()
        .unstack()[day_order]
    )
    pivot_std = pivot_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    mean_demand = panel_valid.groupby("third_category_id")["observed_demand"].mean()
    t33, t67 = mean_demand.quantile(1 / 3), mean_demand.quantile(2 / 3)

    def demand_tier(cat: str) -> str:
        v = mean_demand.get(cat, 0)
        if v >= t67:
            return "High"
        elif v >= t33:
            return "Medium"
        return "Low"

    tier_map = {cat: demand_tier(cat) for cat in pivot_std.index}
    mean_profile = pivot_std.mean(axis=0)
    deviation = ((pivot_std - mean_profile) ** 2).sum(axis=1)

    return pivot_std, tier_map, deviation


def generate_category_seasonality_heatmap(output_dir: Path, n_per_tier: int = 7) -> None:
    pivot_std, tier_map, deviation = _load_profiles()

    tier_color = {"High": "#2166ac", "Medium": "#f4a582", "Low": "#d6604d"}
    tiers = ["High", "Medium", "Low"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for tier in tiers:
        cats = [c for c in pivot_std.index if tier_map[c] == tier]
        top = deviation.loc[cats].nlargest(n_per_tier).index
        data = pivot_std.loc[top].sort_values("Sunday", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            data,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={"label": "Z-score"},
            ax=ax,
        )
        ax.set_title(
            f"Weekly seasonality — {tier} demand categories\n"
            "(Z-score per category; 7 most heterogeneous)",
            color=tier_color[tier],
            fontsize=11,
        )
        ax.set_xlabel("Day of the Week")
        ax.set_ylabel("Category ID")

        fig.tight_layout()
        out = output_dir / f"category_seasonality_{tier.lower()}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate_category_seasonality_heatmap(Path("memoria/figures/eda"))
