from __future__ import annotations

import pandas as pd
from retail_forecasting.config import InventoryConfig
from retail_forecasting.inventory.newsvendor import attach_inventory_costs, choose_order_quantity


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
    quantile_levels = [float(c.replace("q_", "").replace("_", ".")) for c in quantile_columns]

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

            # Recalculate order quantity for this specific ratio
            model_preds["order_quantity"] = choose_order_quantity(
                predictions=model_preds,
                inventory_config=temp_config,
                quantile_columns=quantile_columns if model_name != "seasonal_naive" else [],
                quantile_levels=quantile_levels if model_name != "seasonal_naive" else [],
            )

            # Calculate resulting costs
            cost_preds = attach_inventory_costs(model_preds, temp_config)

            # Summarize
            results.append({
                "ratio": ratio,
                "model_name": model_name,
                "total_cost": cost_preds["total_cost"].sum(),
                "overstock_cost": cost_preds["overstock_cost"].sum(),
                "stockout_cost": cost_preds["stockout_cost"].sum(),
                "mean_order_quantity": cost_preds["order_quantity"].mean(),
            })

    return pd.DataFrame(results)
