from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig
from retail_forecasting.inventory.cost_profiles import attach_series_costs


def critical_fractile(inventory_config: InventoryConfig) -> float:
    return float(
        inventory_config.stockout_cost
        / (inventory_config.stockout_cost + inventory_config.overstock_cost)
    )


def choose_order_quantity(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
    quantile_columns: list[str],
    quantile_levels: list[float],
) -> pd.Series:
    """Determine the single-period Newsvendor order quantity.

    The quantity is the critical-fractile quantile of the lead-time demand
    distribution. There is no inventory position to net out: each forecast origin
    is decided independently.
    """
    enriched = attach_series_costs(predictions, inventory_config)
    alpha = enriched["critical_fractile"].to_numpy(dtype=float)

    if quantile_columns:
        sorted_pairs = sorted(
            zip(quantile_levels, quantile_columns, strict=False), key=lambda item: item[0]
        )
        levels = np.asarray([pair[0] for pair in sorted_pairs], dtype=float)
        values = enriched[[pair[1] for pair in sorted_pairs]].to_numpy(dtype=float)
        s_levels = np.asarray(
            [
                _interpolate_quantile(levels, row, row_alpha)
                for row, row_alpha in zip(values, alpha, strict=False)
            ],
            dtype=float,
        )
    else:
        # Heuristic models: the point forecast is the quantity
        s_levels = enriched["y_pred"].to_numpy(dtype=float)

    orders = s_levels

    if inventory_config.clip_negative_orders:
        orders = np.maximum(orders, 0.0)

    return pd.Series(orders, index=predictions.index, name="order_quantity")


def attach_inventory_costs(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
) -> pd.DataFrame:
    evaluated = attach_series_costs(predictions, inventory_config)

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
    evaluated["overstock_cost"] = evaluated["c_over"] * evaluated["overstock_units"]

    evaluated["stockout_cost"] = evaluated["c_under"] * evaluated["stockout_units"]

    # Calculate total cost as the sum of overstock and stockout costs.
    evaluated["total_cost"] = evaluated["overstock_cost"] + evaluated["stockout_cost"]
    return evaluated


def _interpolate_quantile(levels: np.ndarray, values: np.ndarray, target_level: float) -> float:
    if target_level <= levels[0]:
        return float(values[0])
    if target_level >= levels[-1]:
        return float(values[-1])
    return float(np.interp(target_level, levels, values))


def _config_for_ratio(base_config: InventoryConfig, ratio: float) -> InventoryConfig:
    """Clone the base inventory config with stockout_cost set to overstock_cost * ratio."""
    return InventoryConfig(
        overstock_cost=base_config.overstock_cost,
        stockout_cost=base_config.overstock_cost * ratio,
        clip_negative_orders=base_config.clip_negative_orders,
    )


def _summarize_group(
    model_preds: pd.DataFrame,
    model_name: str,
    data_strategy: str | None,
    ratio: float,
    temp_config: InventoryConfig,
    quantile_columns: list[str],
    quantile_levels: list[float],
) -> dict[str, Any]:
    """Re-cost one model group under `temp_config` and return its summary row."""
    # Use quantiles only when the model actually produced (non-null) ones.
    has_quantiles = any(
        c.startswith("q_") and model_preds[c].notna().any() for c in model_preds.columns
    )
    model_preds["order_quantity"] = choose_order_quantity(
        predictions=model_preds,
        inventory_config=temp_config,
        quantile_columns=quantile_columns if has_quantiles else [],
        quantile_levels=quantile_levels if has_quantiles else [],
    )
    if not has_quantiles:
        model_preds["order_quantity"] = model_preds["y_pred"]

    cost_preds = attach_inventory_costs(model_preds, temp_config)

    res_dict = {
        "ratio": ratio,
        "model_name": model_name,
        "total_cost": cost_preds["total_cost"].sum(),
        "overstock_cost": cost_preds["overstock_cost"].sum(),
        "stockout_cost": cost_preds["stockout_cost"].sum(),
        "mean_order_quantity": cost_preds["order_quantity"].mean(),
    }
    if data_strategy is not None:
        res_dict["data_strategy"] = data_strategy
    return res_dict


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

    quantile_columns = [c for c in predictions.columns if c.startswith("q_")]
    quantile_levels = [float(c.replace("q_", "").replace("_", ".")) for c in quantile_columns]

    group_cols = ["model_name"]
    if "data_strategy" in predictions.columns:
        group_cols.append("data_strategy")

    results = []
    for ratio in ratios:
        temp_config = _config_for_ratio(base_inventory_config, ratio)

        for keys, group_df in predictions.groupby(group_cols):
            if len(group_cols) == 2:
                model_name, data_strategy = keys
            else:
                model_name, data_strategy = keys, None
            results.append(
                _summarize_group(
                    group_df.copy(),
                    model_name,
                    data_strategy,
                    ratio,
                    temp_config,
                    quantile_columns,
                    quantile_levels,
                )
            )

    return pd.DataFrame(results)
