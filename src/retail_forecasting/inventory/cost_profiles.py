from __future__ import annotations

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig


def build_series_cost_profile(
    panel: pd.DataFrame,
    inventory_config: InventoryConfig,
) -> pd.DataFrame:
    """Build synthetic per-series cost coefficients for the newsvendor layer.

    The profile remains intentionally heuristic: it uses demand intensity,
    intermittency, stockout tension, and category-level instability as
    proxies for service criticality and perishability.
    """
    required_columns = {
        "series_id",
        "observed_demand",
        "stockout_hours",
        "third_category_id",
    }
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(
            "Cannot build series cost profile without required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    conf = inventory_config.synthetic_cost_config

    series_summary = (
        panel.groupby("series_id")
        .agg(
            mean_observed_demand=("observed_demand", "mean"),
            demand_std=("observed_demand", "std"),
            zero_demand_rate=(
                "observed_demand",
                lambda values: float((values == 0).mean()),
            ),
            stockout_day_rate=(
                "stockout_hours",
                lambda values: float((values > 0).mean()),
            ),
            third_category_id=("third_category_id", "first"),
        )
        .reset_index()
    )
    series_summary["demand_std"] = series_summary["demand_std"].fillna(0.0)
    denominator = series_summary["mean_observed_demand"].replace(0.0, np.nan)
    series_summary["coefficient_variation"] = (series_summary["demand_std"] / denominator).fillna(
        0.0
    )

    category_summary = (
        series_summary.groupby("third_category_id")
        .agg(
            category_zero_demand_rate=("zero_demand_rate", "mean"),
            category_cv=("coefficient_variation", "mean"),
        )
        .reset_index()
    )
    category_summary["category_zero_rank"] = _percentile_rank(
        category_summary["category_zero_demand_rate"]
    )
    category_summary["category_cv_rank"] = _percentile_rank(category_summary["category_cv"])

    profile = series_summary.merge(
        category_summary[
            [
                "third_category_id",
                "category_zero_rank",
                "category_cv_rank",
            ]
        ],
        on="third_category_id",
        how="left",
    )

    profile["demand_rank"] = _percentile_rank(profile["mean_observed_demand"])
    profile["intermittency_rank"] = _percentile_rank(profile["zero_demand_rate"])
    profile["stockout_rank"] = _percentile_rank(profile["stockout_day_rate"])

    # Dimensional Scoring using parameterized weights
    pw = conf.perishability_weights
    profile["synthetic_perishability_score"] = (
        pw[0] * profile["category_zero_rank"]
        + pw[1] * profile["category_cv_rank"]
        + pw[2] * profile["intermittency_rank"]
    )

    sw = conf.slow_moving_weights
    profile["slow_moving_score"] = sw[0] * profile["intermittency_rank"] + sw[1] * (
        1.0 - profile["demand_rank"]
    )

    cw = conf.criticality_weights
    profile["service_criticality_score"] = (
        cw[0] * profile["demand_rank"] + cw[1] * profile["stockout_rank"]
    )

    # Rescaling using parameterized base and multipliers
    profile["perishability_factor"] = (
        conf.perishability_base
        + conf.perishability_multiplier * profile["synthetic_perishability_score"]
    )
    profile["slow_moving_factor"] = (
        conf.slow_moving_base + conf.slow_moving_multiplier * profile["slow_moving_score"]
    )
    profile["service_criticality_factor"] = (
        conf.service_criticality_base
        + conf.service_criticality_multiplier * profile["service_criticality_score"]
    )

    profile["c_over"] = (
        inventory_config.overstock_cost
        * profile["perishability_factor"]
        * profile["slow_moving_factor"]
    )
    profile["c_under"] = inventory_config.stockout_cost * profile["service_criticality_factor"]
    profile["critical_fractile"] = profile["c_under"] / (profile["c_under"] + profile["c_over"])

    return profile[
        [
            "series_id",
            "c_over",
            "c_under",
            "critical_fractile",
            "synthetic_perishability_score",
            "service_criticality_score",
        ]
    ].copy()


def attach_series_costs(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
    series_cost_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach per-row inventory cost coefficients to a prediction frame."""
    enriched = predictions.copy()
    required_cost_columns = [
        "c_over",
        "c_under",
        "critical_fractile",
    ]
    optional_cost_columns = [
        "synthetic_perishability_score",
        "service_criticality_score",
    ]

    if inventory_config.use_series_costs:
        if all(column in enriched.columns for column in required_cost_columns):
            return enriched
        if series_cost_profile is None:
            raise ValueError(
                "Series cost profiles are enabled but no `series_cost_profile` was provided."
            )
        if "series_id" not in enriched.columns:
            raise ValueError("Series cost profiles require a `series_id` column in predictions.")

        enriched = enriched.merge(
            series_cost_profile,
            on="series_id",
            how="left",
            validate="many_to_one",
        )
        if enriched[["c_over", "c_under", "critical_fractile"]].isna().any().any():
            raise ValueError("Missing cost profile values after merging `series_id` costs.")
        return enriched

    cost_columns = required_cost_columns + optional_cost_columns
    existing_cost_columns = [column for column in cost_columns if column in enriched.columns]
    if existing_cost_columns:
        enriched = enriched.drop(columns=existing_cost_columns)

    enriched["c_over"] = inventory_config.overstock_cost
    enriched["c_under"] = inventory_config.stockout_cost
    enriched["critical_fractile"] = inventory_config.stockout_cost / (
        inventory_config.stockout_cost + inventory_config.overstock_cost
    )
    return enriched


def _percentile_rank(values: pd.Series) -> pd.Series:
    if values.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(values), 0.5), index=values.index, dtype=float)
    return values.rank(method="average", pct=True).astype(float)
