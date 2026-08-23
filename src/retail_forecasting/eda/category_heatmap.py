"""Weekly seasonality per product category, split by demand tier.

Kept out of ``plots.py`` because it is the only figure set here that reasons about product
categories rather than series, and the only one that needs seaborn. It is called from
``render_eda_plots`` like every other figure, so ``make eda`` produces it and
``figure_exports`` carries it to the thesis.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

_TIER_COLOR = {"High": "#2166ac", "Medium": "#f4a582", "Low": "#d6604d"}

# A category needs this many observations before its weekly profile means anything: the
# Z-score is taken within the category, so a handful of rows produces a confident-looking
# pattern out of noise.
_MIN_OBSERVATIONS = 500


def _weekday_profiles(
    panel: pd.DataFrame, min_observations: int
) -> tuple[pd.DataFrame, dict[str, str], pd.Series] | None:
    """Per-category weekday Z-scores, each category's demand tier, and its heterogeneity.

    Returns None when the panel cannot support the figure: no category column, no category
    with enough observations, or a week not fully covered. A synthetic or short panel hits
    all three, and a missing figure is a better answer there than a confident wrong one.
    """
    if "third_category_id" not in panel.columns:
        return None

    counts = panel.groupby("third_category_id")["observed_demand"].count()
    valid_categories = counts[counts >= min_observations].index
    if valid_categories.empty:
        return None

    panel_valid = panel[panel["third_category_id"].isin(valid_categories)].copy()
    panel_valid["date"] = pd.to_datetime(panel_valid["date"])
    panel_valid["day_of_week"] = panel_valid["date"].dt.day_name()

    pivot = panel_valid.groupby(["third_category_id", "day_of_week"])["observed_demand"].mean()
    unstacked = pivot.unstack()
    if not set(_DAY_ORDER).issubset(unstacked.columns):
        return None

    pivot_df = unstacked[_DAY_ORDER]
    pivot_std = pivot_df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

    mean_demand = panel_valid.groupby("third_category_id")["observed_demand"].mean()
    low_cut, high_cut = mean_demand.quantile(1 / 3), mean_demand.quantile(2 / 3)

    def demand_tier(category: str) -> str:
        value = mean_demand.get(category, 0)
        if value >= high_cut:
            return "High"
        return "Medium" if value >= low_cut else "Low"

    tier_map = {category: demand_tier(category) for category in pivot_std.index}

    # Squared distance from the panel's average weekly shape: the figure shows the categories
    # that depart from it most, since those are what make the case that one weekly cycle for
    # the whole panel would be a systematic error.
    mean_profile = pivot_std.mean(axis=0)
    deviation = ((pivot_std - mean_profile) ** 2).sum(axis=1)

    return pivot_std, tier_map, deviation


def render_category_seasonality_heatmaps(
    panel: pd.DataFrame,
    output_dir: Path,
    n_per_tier: int = 7,
    min_observations: int = _MIN_OBSERVATIONS,
) -> None:
    """Write one weekly-seasonality heatmap per demand tier into ``output_dir``."""
    profiles = _weekday_profiles(panel, min_observations)
    if profiles is None:
        return
    pivot_std, tier_map, deviation = profiles

    output_dir.mkdir(parents=True, exist_ok=True)

    for tier in ("High", "Medium", "Low"):
        categories = [c for c in pivot_std.index if tier_map[c] == tier]
        if not categories:
            continue
        top = deviation.loc[categories].nlargest(n_per_tier).index
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
            f"(Z-score per category; {len(top)} most heterogeneous)",
            color=_TIER_COLOR[tier],
            fontsize=11,
        )
        ax.set_xlabel("Day of the Week")
        ax.set_ylabel("Category ID")

        fig.tight_layout()
        fig.savefig(
            output_dir / f"category_seasonality_{tier.lower()}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
