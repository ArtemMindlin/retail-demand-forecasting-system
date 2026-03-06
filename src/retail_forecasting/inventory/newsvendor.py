from __future__ import annotations

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig


def critical_fractile(inventory_config: InventoryConfig) -> float:
    return inventory_config.stockout_cost / (
        inventory_config.stockout_cost + inventory_config.overstock_cost
    )


def choose_order_quantity(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
    quantile_columns: list[str],
    quantile_levels: list[float],
) -> pd.Series:
    alpha = critical_fractile(inventory_config)

    if quantile_columns:
        sorted_pairs = sorted(zip(quantile_levels, quantile_columns), key=lambda item: item[0])
        levels = np.asarray([pair[0] for pair in sorted_pairs], dtype=float)
        values = predictions[[pair[1] for pair in sorted_pairs]].to_numpy(dtype=float)
        orders = np.apply_along_axis(lambda row: _interpolate_quantile(levels, row, alpha), 1, values)
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
    evaluated["overstock_units"] = np.maximum(
        evaluated["order_quantity"] - evaluated["y_true"],
        0.0,
    )
    evaluated["stockout_units"] = np.maximum(
        evaluated["y_true"] - evaluated["order_quantity"],
        0.0,
    )
    evaluated["overstock_cost"] = (
        inventory_config.overstock_cost * evaluated["overstock_units"]
    )
    evaluated["stockout_cost"] = (
        inventory_config.stockout_cost * evaluated["stockout_units"]
    )
    evaluated["total_cost"] = evaluated["overstock_cost"] + evaluated["stockout_cost"]
    return evaluated


def _interpolate_quantile(levels: np.ndarray, values: np.ndarray, target_level: float) -> float:
    if target_level <= levels[0]:
        return float(values[0])
    if target_level >= levels[-1]:
        return float(values[-1])
    return float(np.interp(target_level, levels, values))
