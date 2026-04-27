from __future__ import annotations

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig

# The critical fractile is calculated to find the quantile of the demand distribution that minimizes expected costs, and order quantities are chosen accordingly.
def critical_fractile(inventory_config: InventoryConfig) -> float:
    return inventory_config.stockout_cost / (
        inventory_config.stockout_cost + inventory_config.overstock_cost
    )

# The order quantity is determined by either directly using the point forecast or interpolating between quantile forecasts based on the critical fractile, which represents the optimal service level for the newsvendor problem.
def choose_order_quantity(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
    quantile_columns: list[str],
    quantile_levels: list[float],
) -> pd.Series:
    alpha = critical_fractile(inventory_config)

    # Boosting models provide quantile forecasts.
    if quantile_columns:
        sorted_pairs = sorted(zip(quantile_levels, quantile_columns), key=lambda item: item[0])
        levels = np.asarray([pair[0] for pair in sorted_pairs], dtype=float)
        values = predictions[[pair[1] for pair in sorted_pairs]].to_numpy(dtype=float)
        orders = np.apply_along_axis(lambda row: _interpolate_quantile(levels, row, alpha), 1, values)

    # Heuristic models provide only point forecasts.
    else:
        orders = predictions["y_pred"].to_numpy(dtype=float)

    if inventory_config.clip_negative_orders:
        orders = np.maximum(orders, 0.0)
    return pd.Series(orders, index=predictions.index, name="order_quantity")


def attach_inventory_costs(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
) -> pd.DataFrame:
    evaluated = predictions.copy()

    # Calculate overstock units.
    evaluated["overstock_units"] = np.maximum(
        evaluated["order_quantity"] - evaluated["y_true"],
        0.0,
    )

    # Calculate stockout units.
    evaluated["stockout_units"] = np.maximum(
        evaluated["y_true"] - evaluated["order_quantity"],
        0.0,
    )

    # Calculate costs based on the overstock and stockout units.
    evaluated["overstock_cost"] = (
        inventory_config.overstock_cost * evaluated["overstock_units"]
    )

    evaluated["stockout_cost"] = (
        inventory_config.stockout_cost * evaluated["stockout_units"]
    )

    # Calculate total cost as the sum of overstock and stockout costs.
    evaluated["total_cost"] = evaluated["overstock_cost"] + evaluated["stockout_cost"]
    return evaluated


def _interpolate_quantile(levels: np.ndarray, values: np.ndarray, target_level: float) -> float:
    if target_level <= levels[0]:
        return float(values[0])
    if target_level >= levels[-1]:
        return float(values[-1])
    return float(np.interp(target_level, levels, values))


def run_sensitivity_analysis(
    predictions: pd.DataFrame,
    base_inventory_config: InventoryConfig,
    ratios: list[float] | None = None,
) -> pd.DataFrame:
    """Run sensitivity analysis across multiple cost ratios.

    Args:
        predictions: The prediction frame containing y_true and forecast quantiles.
        base_inventory_config: Base inventory settings (used for overstock_cost reference).
        ratios: List of Cs/Co ratios to test. Defaults to [1, 2, 4, 8, 10].

    Returns:
        A summary dataframe with costs for each model and each ratio.
    """
    if ratios is None:
        ratios = [1.0, 2.0, 4.0, 8.0, 10.0]

    # Identify available quantile columns
    quantile_columns = [c for c in predictions.columns if c.startswith("q_")]
    quantile_levels = [
        float(c.replace("q_", "").replace("_", ".")) for c in quantile_columns
    ]

    results = []

    for ratio in ratios:
        # Create a temporary config for this ratio
        temp_config = InventoryConfig(
            overstock_cost=base_inventory_config.overstock_cost,
            stockout_cost=base_inventory_config.overstock_cost * ratio,
            clip_negative_orders=base_inventory_config.clip_negative_orders,
        )

        for model_name in predictions["model_name"].unique():
            model_preds = predictions[predictions["model_name"] == model_name].copy()

            # Identify if this model has quantiles
            has_quantiles = model_name in ["auto_boosting", "catboost"]

            # Recalculate order quantity for this specific ratio
            model_preds["order_quantity"] = choose_order_quantity(
                predictions=model_preds,
                inventory_config=temp_config,
                quantile_columns=quantile_columns if has_quantiles else [],
                quantile_levels=quantile_levels if has_quantiles else [],
            )

            # Fallback for models without quantiles
            if not has_quantiles:
                model_preds["order_quantity"] = model_preds["y_pred"]

            # Calculate resulting costs
            cost_preds = attach_inventory_costs(model_preds, temp_config)

            # Summarize
            results.append(
                {
                    "ratio": ratio,
                    "model_name": model_name,
                    "total_cost": cost_preds["total_cost"].sum(),
                    "overstock_cost": cost_preds["overstock_cost"].sum(),
                    "stockout_cost": cost_preds["stockout_cost"].sum(),
                    "mean_order_quantity": cost_preds["order_quantity"].mean(),
                }
            )

    return pd.DataFrame(results)
