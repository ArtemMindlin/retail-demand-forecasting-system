from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP_SERIES_PLOT_COUNT = 12
MAX_HEATMAP_SERIES = 120
SCATTER_SAMPLE_SIZE = 5000
REPRESENTATIVE_SERIES_COUNT = 12


def render_eda_plots(
    panel: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Render a comprehensive static plot set for EDA runs."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    _plot_observed_demand_distribution(
        panel,
        target_dir / "observed_demand_distribution.png",
    )
    _plot_observed_demand_boxplot_top_series(
        panel,
        series_summary,
        target_dir / "observed_demand_boxplot_top_series.png",
    )
    _plot_stockout_hours_distribution(
        panel,
        target_dir / "stockout_hours_distribution.png",
    )
    _plot_weekday_demand_profile(
        weekday_summary,
        target_dir / "weekday_demand_profile.png",
    )
    _plot_top_series_total_demand(
        series_summary,
        target_dir / "top_series_total_demand.png",
    )
    _plot_coverage_heatmap(
        panel,
        series_summary,
        target_dir / "coverage_heatmap.png",
    )
    _plot_zero_demand_rate_by_series(
        series_summary,
        target_dir / "zero_demand_rate_by_series.png",
    )
    _plot_stockout_band_demand(
        panel,
        target_dir / "stockout_band_demand.png",
    )
    _plot_stockout_vs_demand_scatter(
        panel,
        target_dir / "stockout_vs_demand_scatter.png",
    )
    _plot_correlation_heatmap(
        panel,
        target_dir / "correlation_heatmap.png",
    )
    _plot_covariate_vs_demand_grid(
        panel,
        target_dir / "covariate_vs_demand_grid.png",
    )
    _plot_representative_series_panels(
        panel,
        series_summary,
        target_dir / "representative_series_panels.png",
    )
    _plot_acf_demand(
        panel,
        target_dir / "acf_demand.png",
    )


def _plot_observed_demand_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        panel["observed_demand"],
        bins=50,
        color="#1f77b4",
        edgecolor="white",
    )
    axes[0].set_xlabel("Observed demand")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Observed demand distribution (linear scale)")

    axes[1].hist(
        panel["observed_demand"],
        bins=50,
        color="#1f77b4",
        edgecolor="white",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Observed demand")
    axes[1].set_ylabel("Frequency (log scale)")
    axes[1].set_title("Observed demand distribution (log scale)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_observed_demand_boxplot_top_series(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    n = TOP_SERIES_PLOT_COUNT // 3 * 3
    per_group = n // 3
    top = series_summary.head(per_group)["series_id"].tolist()
    mid_start = len(series_summary) // 2
    mid = series_summary.iloc[mid_start : mid_start + per_group]["series_id"].tolist()
    bottom = series_summary.tail(per_group)["series_id"].tolist()

    subset = panel.loc[panel["series_id"].isin(top + mid + bottom)].copy()
    if subset.empty:
        return

    def _distributions(ids: list[str]) -> list[np.ndarray]:
        return [subset.loc[subset["series_id"] == s, "observed_demand"].to_numpy() for s in ids]

    def _draw(ax: plt.Axes, ids: list[str], color: str, title: str) -> None:
        bp = ax.boxplot(
            _distributions(ids),
            tick_labels=ids,
            patch_artist=True,
            medianprops={"color": "#d62728", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_xlabel("Series")
        ax.set_ylabel("Observed demand")
        ax.tick_params(axis="x", rotation=45)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw(axes[0], top, "#2166ac", "High-volume series")
    _draw(axes[1], mid + bottom, "#74add1", "Mid / low-volume series")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stockout_hours_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    max_hours = int(panel["stockout_hours"].max()) + 1
    bins = np.arange(-0.5, max_hours + 0.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        panel["stockout_hours"],
        bins=bins,
        color="#d62728",
        edgecolor="white",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Stockout hours")
    axes[0].set_ylabel("Frequency (log scale)")
    axes[0].set_title("Stockout hours distribution (log scale)")
    axes[0].set_xticks(range(0, max_hours, 2))

    positive_stockout = panel.loc[panel["stockout_hours"] > 0, "stockout_hours"]
    bins_pos = np.arange(0.5, max_hours + 0.5, 1)
    axes[1].hist(
        positive_stockout,
        bins=bins_pos,
        color="#ff9896",
        edgecolor="white",
    )
    axes[1].set_xlabel("Positive stockout hours")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Positive stockout-hour distribution")
    axes[1].set_xticks(range(1, max_hours, 2))

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_weekday_demand_profile(
    weekday_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        weekday_summary["weekday_name"],
        weekday_summary["observed_demand_mean"],
        marker="o",
        linewidth=2,
        color="#2ca02c",
        label="Mean",
    )
    ax.plot(
        weekday_summary["weekday_name"],
        weekday_summary["observed_demand_median"],
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Median",
    )
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Observed demand")
    ax.set_title("Weekly demand profile")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_top_series_total_demand(
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    top_series = series_summary.head(10)
    if top_series.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        top_series["series_id"][::-1],
        top_series["observed_demand_sum"][::-1],
        color="#9467bd",
    )
    ax.set_xlabel("Total observed demand")
    ax.set_ylabel("Series")
    ax.set_title("Top 10 series by observed demand")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_coverage_heatmap(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_series = series_summary.head(MAX_HEATMAP_SERIES)["series_id"].tolist()
    coverage_frame = (
        panel.loc[panel["series_id"].isin(selected_series), ["series_id", "date"]]
        .assign(present=1.0)
        .pivot(index="series_id", columns="date", values="present")
        .fillna(0.0)
    )
    if coverage_frame.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    image = ax.imshow(
        coverage_frame.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Date index")
    ax.set_ylabel("Series")
    ax.set_title(f"Coverage heatmap (top {len(selected_series)} series by demand)")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Observed row")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_zero_demand_rate_by_series(
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    rates = series_summary["zero_demand_rate"].dropna()
    if rates.empty:
        return

    median_rate = rates.median()
    pct_above_50 = (rates > 0.5).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(rates, bins=40, color="#8c564b", edgecolor="white", alpha=0.85)
    ax.axvline(
        median_rate,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label=f"Median: {median_rate:.2f}",
    )
    ax.axvline(
        0.5,
        color="#ff7f0e",
        linewidth=1.4,
        linestyle=":",
        label=f">50% zero: {pct_above_50:.0f}% of series",
    )
    ax.set_xlabel("Zero-demand rate")
    ax.set_ylabel("Number of series")
    ax.set_title("Distribution of zero-demand rate across all series")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stockout_band_demand(panel: pd.DataFrame, output_path: Path) -> None:
    stockout_band_frame = (
        panel.assign(
            stockout_band=pd.cut(
                panel["stockout_hours"],
                bins=[-0.01, 0.0, 2.0, 6.0, float("inf")],
                labels=["0", "0-2", "3-6", "7+"],
            )
        )
        .groupby("stockout_band", observed=False)["observed_demand"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(
        stockout_band_frame["stockout_band"].astype(str),
        stockout_band_frame["mean"],
        color="#ff7f0e",
    )
    axes[0].set_xlabel("Stockout band")
    axes[0].set_ylabel("Mean observed demand")
    axes[0].set_title("Mean demand by stockout band")

    axes[1].bar(
        stockout_band_frame["stockout_band"].astype(str),
        stockout_band_frame["count"],
        color="#bcbd22",
    )
    axes[1].set_xlabel("Stockout band")
    axes[1].set_ylabel("Observations")
    axes[1].set_title("Observation count by stockout band")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stockout_vs_demand_scatter(panel: pd.DataFrame, output_path: Path) -> None:
    sampled = _sample_panel(panel)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        sampled["stockout_hours"],
        sampled["observed_demand"],
        alpha=0.15,
        s=12,
        color="#1f77b4",
    )

    grouped = (
        panel.groupby("stockout_hours", as_index=False)["observed_demand"]
        .mean()
        .rename(columns={"observed_demand": "observed_demand_mean"})
    )
    ax.plot(
        grouped["stockout_hours"],
        grouped["observed_demand_mean"],
        color="#d62728",
        linewidth=2,
        label="Mean by stockout hour",
    )
    ax.set_xlabel("Stockout hours")
    ax.set_ylabel("Observed demand")
    ax.set_title("Observed demand vs stockout hours")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


_MEANINGFUL_NUMERIC = [
    "observed_demand",
    "stockout_hours",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


def _plot_correlation_heatmap(panel: pd.DataFrame, output_path: Path) -> None:
    cols = [c for c in _MEANINGFUL_NUMERIC if c in panel.columns]
    if not cols:
        return

    variable_numeric = panel[cols].dropna(how="all")
    variable_numeric = variable_numeric.loc[:, variable_numeric.nunique(dropna=True) > 1]
    if variable_numeric.empty:
        return

    correlation = variable_numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        correlation.to_numpy(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(correlation.index)))
    ax.set_yticklabels(correlation.index)
    ax.set_title("Correlation heatmap")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Correlation")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_covariate_vs_demand_grid(panel: pd.DataFrame, output_path: Path) -> None:
    candidate_columns = [
        "discount",
        "avg_temperature",
        "precpt",
        "avg_humidity",
        "avg_wind_level",
    ]
    columns = [c for c in candidate_columns if c in panel.columns]
    if not columns:
        return

    n_bins = 20
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for axis, column in zip(axes_flat, columns, strict=False):
        col_data = panel[[column, "observed_demand"]].dropna()
        col_data = col_data[col_data[column] > col_data[column].quantile(0.01)]
        col_data = col_data[col_data[column] < col_data[column].quantile(0.99)]

        col_data["_bin"] = pd.cut(col_data[column], bins=n_bins)
        binned = (
            col_data.groupby("_bin", observed=True)["observed_demand"]
            .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)))
            .reset_index()
        )
        binned["x_mid"] = binned["_bin"].apply(lambda b: b.mid)

        axis.fill_between(
            binned["x_mid"],
            binned["mean"] - 1.96 * binned["sem"],
            binned["mean"] + 1.96 * binned["sem"],
            alpha=0.25,
            color="#1f77b4",
            label="95% CI",
        )
        axis.plot(
            binned["x_mid"], binned["mean"], color="#1f77b4", linewidth=2, label="Mean demand"
        )
        axis.set_xlabel(column)
        axis.set_ylabel("Mean observed demand")
        axis.set_title(f"{column} vs observed demand")
        axis.legend(fontsize=8)

    for axis in axes_flat[len(columns) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _select_diverse_series(panel: pd.DataFrame, n: int = 12) -> list[str]:
    """Pick n series spanning different stores, demand levels and stockout exposure."""
    min_days = panel.groupby("series_id")["date"].count()
    threshold = min_days.quantile(0.5)
    valid = min_days[min_days >= threshold].index

    df = panel[panel["series_id"].isin(valid)].copy()
    stats = df.groupby("series_id").agg(
        mean_demand=("observed_demand", "mean"),
        zero_rate=("observed_demand", lambda x: (x == 0).mean()),
        stockout_rate=("stockout_hours", lambda x: (x > 0).mean()),
        store=("store_id", "first"),
    )

    stats["demand_tier"] = pd.qcut(stats["mean_demand"], q=3, labels=["low", "mid", "high"])
    stats["stockout_tier"] = pd.cut(
        stats["stockout_rate"], bins=[-0.01, 0.05, 0.3, 1.0], labels=["low", "mid", "high"]
    )

    selected: list[str] = []
    used_stores: set[str] = set()
    rng = np.random.default_rng(42)

    for demand_tier in ["high", "mid", "low"]:
        for stockout_tier in ["high", "mid", "low"]:
            candidates = stats[
                (stats["demand_tier"] == demand_tier)
                & (stats["stockout_tier"] == stockout_tier)
                & (~stats["store"].isin(used_stores))
            ]
            if candidates.empty:
                candidates = stats[
                    (stats["demand_tier"] == demand_tier) & (~stats["store"].isin(used_stores))
                ]
            if candidates.empty:
                continue
            pick = rng.choice(candidates.index)
            selected.append(pick)
            used_stores.add(stats.loc[pick, "store"])
            if len(selected) >= n:
                return selected

    return selected[:n]


def _plot_representative_series_panels(
    panel: pd.DataFrame,
    series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_series = _select_diverse_series(panel, n=REPRESENTATIVE_SERIES_COUNT)
    if not selected_series:
        return

    subset = panel.loc[panel["series_id"].isin(selected_series)].copy()
    n_cols = 3
    n_rows = int(np.ceil(len(selected_series) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.6 * n_rows), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for axis, series_id in zip(axes_flat, selected_series, strict=False):
        series_frame = subset.loc[subset["series_id"] == series_id].sort_values("date")
        demand_max = series_frame["observed_demand"].max()
        stockout_max = series_frame["stockout_hours"].max()
        # Scale stockout overlay to demand range so it doesn't dominate the axis
        stockout_scaled = (
            series_frame["stockout_hours"] / stockout_max * demand_max
            if stockout_max > 0
            else series_frame["stockout_hours"]
        )
        axis.fill_between(
            series_frame["date"],
            0,
            stockout_scaled,
            color="#d62728",
            alpha=0.25,
            label="Stockout (scaled)",
        )
        axis.plot(
            series_frame["date"],
            series_frame["observed_demand"],
            color="#1f77b4",
            linewidth=1.8,
            label="Demand",
        )
        axis.set_title(series_id)
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("Demand")

    for axis in axes_flat[len(selected_series) :]:
        axis.axis("off")

    fig.suptitle(
        "Representative series panels (line: demand, shaded: stockout hours)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_acf_demand(panel: pd.DataFrame, output_path: Path, max_lags: int = 28) -> None:
    daily_mean = panel.groupby("date")["observed_demand"].mean().sort_index().to_numpy()
    n = len(daily_mean)
    if n < max_lags + 2:
        return

    mean = daily_mean.mean()
    centered = daily_mean - mean
    var = (centered**2).sum()
    if var == 0:
        return

    acf_values = np.array(
        [(centered[: n - lag] * centered[lag:]).sum() / var for lag in range(max_lags + 1)]
    )
    confidence_bound = 1.96 / np.sqrt(n)
    lags = np.arange(max_lags + 1)

    seasonal_lags = {7, 14, 21, 28}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-confidence_bound, confidence_bound, color="#1f77b4", alpha=0.12, label="95% CI")

    for lag, val in zip(lags, acf_values, strict=False):
        color = "#d62728" if lag in seasonal_lags else "#1f77b4"
        ax.vlines(lag, 0, val, colors=color, linewidth=1.8)
        ax.plot(lag, val, "o", color=color, markersize=4)

    for s_lag in seasonal_lags:
        if s_lag <= max_lags:
            ax.axvline(float(s_lag), color="#d62728", linewidth=0.7, linestyle="--", alpha=0.45)

    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF of daily aggregate demand (lags 0–28)")
    ax.set_xticks(lags)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _sample_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if len(panel) <= SCATTER_SAMPLE_SIZE:
        return panel
    return panel.sample(SCATTER_SAMPLE_SIZE, random_state=42)


def _binned_average(
    panel: pd.DataFrame,
    feature_column: str,
    bins: int = 20,
) -> pd.DataFrame:
    feature = panel[feature_column]
    if feature.nunique(dropna=True) <= 1:
        return pd.DataFrame(columns=[feature_column, "observed_demand_mean"])

    quantile_bins = min(bins, feature.nunique(dropna=True))
    try:
        binned = pd.qcut(feature, q=quantile_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=[feature_column, "observed_demand_mean"])

    grouped = (
        panel.assign(_bin=binned)
        .groupby("_bin", observed=False)
        .agg(
            feature_center=(feature_column, "mean"),
            observed_demand_mean=("observed_demand", "mean"),
        )
        .dropna()
        .reset_index(drop=True)
        .rename(columns={"feature_center": feature_column})
    )
    return grouped
