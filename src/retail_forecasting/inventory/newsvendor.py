from __future__ import annotations

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
    series_cost_profile: pd.DataFrame | None = None,
    current_stock: pd.Series | None = None,
    on_order: pd.Series | None = None,
) -> pd.Series:
    """Determine the optimal order quantity using an Order-Up-To (S) policy.

    If current_stock and on_order are provided, the function calculates the
    Order-Up-To level (S) and subtracts the current Inventory Position.
    Otherwise, it defaults to the static Newsvendor quantity.
    """
    enriched = attach_series_costs(
        predictions=predictions,
        inventory_config=inventory_config,
        series_cost_profile=series_cost_profile,
    )
    alpha = enriched["critical_fractile"].to_numpy(dtype=float)

    # 1. Determine the 'Order-Up-To' level (S) based on lead-time demand distribution
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
        # Heuristic models: S = Point Forecast
        s_levels = enriched["y_pred"].to_numpy(dtype=float)

    # 2. Subtract current Inventory Position (On-Hand + On-Order - Backlog)
    # Note: In our simulation context, 'current_stock' represents On-Hand minus Backlog
    inventory_position = np.zeros_like(s_levels)
    if current_stock is not None:
        inventory_position += current_stock.reindex(predictions.index).fillna(0.0).to_numpy()
    if on_order is not None:
        inventory_position += on_order.reindex(predictions.index).fillna(0.0).to_numpy()

    orders = s_levels - inventory_position

    if inventory_config.clip_negative_orders:
        orders = np.maximum(orders, 0.0)

    return pd.Series(orders, index=predictions.index, name="order_quantity")


def attach_inventory_costs(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
    series_cost_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    evaluated = attach_series_costs(
        predictions=predictions,
        inventory_config=inventory_config,
        series_cost_profile=series_cost_profile,
    )

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


def summarize_pareto_frontier(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
) -> pd.DataFrame:
    """Evaluate candidate inventory policies and mark Pareto-efficient trade-offs.

    Candidate policies scale the already selected order quantity. This keeps the
    frontier independent from model training and works for point and quantile
    forecasters.
    """
    required_columns = {
        "y_true",
        "order_quantity",
        "c_over",
        "c_under",
        "model_name",
        "backend_name",
    }
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Cannot compute Pareto frontier without required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    order_scales = _validated_order_scales(inventory_config.pareto_order_scales)
    group_cols = ["model_name", "backend_name"]
    if "data_strategy" in predictions.columns:
        group_cols.insert(0, "data_strategy")

    records = []
    for keys, subset in predictions.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_cols, key_values, strict=False))
        for order_scale in order_scales:
            candidate = subset.copy()
            candidate["candidate_order_quantity"] = (
                candidate["order_quantity"].to_numpy(dtype=float) * order_scale
            )
            if inventory_config.clip_negative_orders:
                candidate["candidate_order_quantity"] = np.maximum(
                    candidate["candidate_order_quantity"],
                    0.0,
                )

            overstock_units = np.maximum(
                candidate["candidate_order_quantity"] - candidate["y_true"],
                0.0,
            )
            stockout_units = np.maximum(
                candidate["y_true"] - candidate["candidate_order_quantity"],
                0.0,
            )
            overstock_cost = candidate["c_over"].to_numpy(dtype=float) * overstock_units
            stockout_cost = candidate["c_under"].to_numpy(dtype=float) * stockout_units
            actual_demand = candidate["y_true"].to_numpy(dtype=float)
            filled_units = np.minimum(
                actual_demand,
                candidate["candidate_order_quantity"].to_numpy(dtype=float),
            )
            demand_denominator = actual_demand.sum()

            records.append(
                {
                    **group_values,
                    "policy_name": f"order_scale_{order_scale:g}",
                    "order_scale": order_scale,
                    "observations": int(len(candidate)),
                    "mean_order_quantity": float(candidate["candidate_order_quantity"].mean()),
                    "total_overstock_units": float(overstock_units.sum()),
                    "total_stockout_units": float(stockout_units.sum()),
                    "total_overstock_cost": float(overstock_cost.sum()),
                    "total_stockout_cost": float(stockout_cost.sum()),
                    "total_cost": float(overstock_cost.sum() + stockout_cost.sum()),
                    "mean_cost": float((overstock_cost + stockout_cost).mean()),
                    "service_level": float((stockout_units <= 0.0).mean()),
                    "fill_rate": float(
                        filled_units.sum() / demand_denominator if demand_denominator > 0.0 else 1.0
                    ),
                }
            )

    frontier = pd.DataFrame(records)
    if frontier.empty:
        return frontier

    frontier["is_pareto_efficient"] = False
    for _, index in frontier.groupby(group_cols, dropna=False).groups.items():
        objectives = frontier.loc[
            index,
            ["total_cost", "total_overstock_units", "total_stockout_units"],
        ].to_numpy(dtype=float)
        frontier.loc[index, "is_pareto_efficient"] = _pareto_efficient_mask(objectives)

    sort_columns = group_cols + ["is_pareto_efficient", "total_cost"]
    sort_ascending = [True] * len(group_cols) + [False, True]
    return frontier.sort_values(
        sort_columns,
        ascending=sort_ascending,
    ).reset_index(drop=True)


def _interpolate_quantile(levels: np.ndarray, values: np.ndarray, target_level: float) -> float:
    if target_level <= levels[0]:
        return float(values[0])
    if target_level >= levels[-1]:
        return float(values[-1])
    return float(np.interp(target_level, levels, values))


def _validated_order_scales(order_scales: list[float]) -> list[float]:
    unique_scales = sorted({float(scale) for scale in order_scales})
    if not unique_scales:
        raise ValueError("Pareto frontier requires at least one order scale.")
    if any(scale < 0.0 for scale in unique_scales):
        raise ValueError("Pareto order scales must be non-negative.")
    return unique_scales


def _pareto_efficient_mask(objectives: np.ndarray) -> np.ndarray:
    efficient = np.ones(objectives.shape[0], dtype=bool)
    for idx, candidate in enumerate(objectives):
        if not efficient[idx]:
            continue
        dominated = np.all(objectives <= candidate, axis=1) & np.any(
            objectives < candidate,
            axis=1,
        )
        dominated[idx] = False
        if dominated.any():
            efficient[idx] = False
    return efficient


def run_sensitivity_analysis(
    predictions: pd.DataFrame,
    base_inventory_config: InventoryConfig,
    ratios: list[float] | None = None,
    series_cost_profile: pd.DataFrame | None = None,
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
            use_series_costs=base_inventory_config.use_series_costs,
            clip_negative_orders=base_inventory_config.clip_negative_orders,
        )
        adjusted_series_cost_profile = series_cost_profile
        if series_cost_profile is not None and temp_config.use_series_costs:
            adjusted_series_cost_profile = series_cost_profile.copy()
            target_stockout_cost = base_inventory_config.overstock_cost * ratio
            stockout_scale = target_stockout_cost / base_inventory_config.stockout_cost
            adjusted_series_cost_profile["c_under"] = (
                adjusted_series_cost_profile["c_under"] * stockout_scale
            )
            adjusted_series_cost_profile["critical_fractile"] = adjusted_series_cost_profile[
                "c_under"
            ] / (adjusted_series_cost_profile["c_under"] + adjusted_series_cost_profile["c_over"])

        # Determine grouping keys based on column presence
        group_cols = ["model_name"]
        if "data_strategy" in predictions.columns:
            group_cols.append("data_strategy")

        for keys, group_df in predictions.groupby(group_cols):
            model_preds = group_df.copy()
            if len(group_cols) == 2:
                model_name, data_strategy = keys
            else:
                model_name = keys
                data_strategy = None

            if (
                adjusted_series_cost_profile is None
                and temp_config.use_series_costs
                and {"c_over", "c_under", "critical_fractile"}.issubset(model_preds.columns)
            ):
                target_stockout_cost = base_inventory_config.overstock_cost * ratio
                stockout_scale = target_stockout_cost / base_inventory_config.stockout_cost
                model_preds["c_under"] = model_preds["c_under"] * stockout_scale
                model_preds["critical_fractile"] = model_preds["c_under"] / (
                    model_preds["c_under"] + model_preds["c_over"]
                )

            # Identify if this model has quantiles with actual (non-null) values
            has_quantiles = any(
                c.startswith("q_") and model_preds[c].notna().any() for c in model_preds.columns
            )

            # Recalculate order quantity for this specific ratio
            model_preds["order_quantity"] = choose_order_quantity(
                predictions=model_preds,
                inventory_config=temp_config,
                quantile_columns=quantile_columns if has_quantiles else [],
                quantile_levels=quantile_levels if has_quantiles else [],
                series_cost_profile=adjusted_series_cost_profile,
            )

            # Fallback for models without quantiles
            if not has_quantiles:
                model_preds["order_quantity"] = model_preds["y_pred"]

            # Calculate resulting costs
            cost_preds = attach_inventory_costs(
                model_preds,
                temp_config,
                series_cost_profile=adjusted_series_cost_profile,
            )

            # Summarize
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
            results.append(res_dict)

    return pd.DataFrame(results)
