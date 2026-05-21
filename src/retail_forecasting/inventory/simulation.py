from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from retail_forecasting.config import InventoryConfig


@dataclass
class InventoryState:
    """Tracks the physical state of a SKU during simulation."""

    series_id: str
    on_hand: float = 0.0
    on_order: float = 0.0
    backlog: float = 0.0

    # Configuration
    lead_time: int = 7  # matching horizon usually
    c_over: float = 1.0
    c_under: float = 4.0

    # History for tracking
    history: list[dict[str, float]] = field(default_factory=list)

    def step(self, demand: float, order_quantity: float, arrivals: float) -> None:
        """Advances the simulation by one decision period (fold)."""
        # 1. Update on-hand with arrivals
        self.on_hand += arrivals

        # 2. Satisfy backlog if any
        if self.backlog > 0:
            filled_backlog = min(self.on_hand, self.backlog)
            self.on_hand -= filled_backlog
            self.backlog -= filled_backlog

        # 3. Satisfy current demand
        served_demand = min(self.on_hand, demand)
        self.on_hand -= served_demand
        missed_demand = demand - served_demand

        # 4. Update backlog (assuming backorders are allowed in this simulation)
        self.backlog += missed_demand

        # 5. Overstock is what remains
        overstock = self.on_hand

        # 6. Costs
        cost = (overstock * self.c_over) + (self.backlog * self.c_under)

        # 7. Place new order
        self.on_order = order_quantity

        self.history.append(
            {
                "demand": demand,
                "order_placed": order_quantity,
                "arrivals": arrivals,
                "on_hand_end": self.on_hand,
                "backlog_end": self.backlog,
                "cost": cost,
            }
        )


def simulate_inventory_policy(
    predictions: pd.DataFrame,
    inventory_config: InventoryConfig,
    series_cost_profile: pd.DataFrame | None = None,
    initial_on_hand: float = 0.0,
) -> pd.DataFrame:
    """
    Simulates a multi-period inventory policy across backtest folds.

    This replaces the static newsvendor logic with a dynamic one where
    the ending state of fold N is the starting state of fold N+1.

    Now uses 'choose_order_quantity' iteratively to implement an
    Inventory-Aware Order-Up-To policy, and applies LP optimization
    if a global capacity constraint is set.
    """
    from retail_forecasting.inventory.newsvendor import (
        attach_inventory_costs,
        choose_order_quantity,
    )
    from retail_forecasting.inventory.optimization import optimize_orders_lp

    if "fold_id" not in predictions.columns:
        return predictions  # Cannot simulate without temporal order

    results = []

    # Identify available quantile columns for the decider
    quantile_columns = [c for c in predictions.columns if c.startswith("q_")]
    quantile_levels = [float(c.replace("q_", "").replace("_", ".")) for c in quantile_columns]

    # Process independently for each experimental strategy (data_strategy, model_name, backend_name)
    model_group_cols = ["data_strategy", "model_name", "backend_name"]
    model_group_cols = [c for c in model_group_cols if c in predictions.columns]

    for _model_keys, model_subset in predictions.groupby(model_group_cols):
        # Initialize state for all series in this model strategy
        states: dict[str, InventoryState] = {}
        pending_orders: dict[str, list[tuple[int, float]]] = {}
        for series_id in model_subset["series_id"].unique():
            series_subset = model_subset[model_subset["series_id"] == series_id]
            states[series_id] = InventoryState(
                series_id=series_id,
                on_hand=initial_on_hand,
                c_over=series_subset["c_over"].iloc[0]
                if "c_over" in series_subset.columns
                else 1.0,
                c_under=series_subset["c_under"].iloc[0]
                if "c_under" in series_subset.columns
                else 4.0,
            )
            pending_orders[series_id] = []

        # Iterate fold by fold to enable global constraints
        sorted_folds = sorted(model_subset["fold_id"].unique())
        for fold_id in sorted_folds:
            fold_subset = model_subset[model_subset["fold_id"] == fold_id]

            unconstrained_orders = {}
            marginal_utilities = {}
            row_map = {}
            arrivals_map = {}

            # 1. First pass: Determine unconstrained orders for all SKUs
            for _, row in fold_subset.iterrows():
                series_id = row["series_id"]
                row_map[series_id] = row
                state = states[series_id]

                # Collect arrivals
                arrivals = sum(q for f, q in pending_orders[series_id] if f == fold_id)
                arrivals_map[series_id] = arrivals
                pending_orders[series_id] = [
                    (f, q) for f, q in pending_orders[series_id] if f != fold_id
                ]

                decider_input = pd.DataFrame([row])
                net_stock = pd.Series([state.on_hand - state.backlog], index=decider_input.index)
                on_order_val = sum(q for f, q in pending_orders[series_id])
                on_order_series = pd.Series([on_order_val], index=decider_input.index)

                new_order_qty = choose_order_quantity(
                    predictions=decider_input,
                    inventory_config=inventory_config,
                    quantile_columns=quantile_columns
                    if row["model_name"] != "seasonal_naive"
                    else [],
                    quantile_levels=quantile_levels
                    if row["model_name"] != "seasonal_naive"
                    else [],
                    series_cost_profile=series_cost_profile,
                    current_stock=net_stock,
                    on_order=on_order_series,
                ).iloc[0]

                unconstrained_orders[series_id] = new_order_qty
                marginal_utilities[series_id] = state.c_under

            # 2. Global Optimization Pass
            global_cap = getattr(inventory_config, "global_capacity_units", None)
            if global_cap is not None:
                constrained_orders = optimize_orders_lp(
                    unconstrained_orders=unconstrained_orders,
                    marginal_utilities=marginal_utilities,
                    global_capacity=global_cap,
                )
            else:
                constrained_orders = unconstrained_orders

            # 3. Apply Constrained Orders and Advance State
            for series_id, row in row_map.items():
                state = states[series_id]
                arrivals = arrivals_map[series_id]
                final_order_qty = constrained_orders[series_id]

                # Advance physical simulation
                state.step(
                    demand=row["y_true"],
                    order_quantity=final_order_qty,
                    arrivals=arrivals,
                )

                # Schedule next arrival
                pending_orders[series_id].append((fold_id + 1, final_order_qty))

                # Save results
                sim_row = row.copy()
                sim_row["order_quantity"] = final_order_qty
                sim_row["sim_on_hand"] = state.on_hand
                sim_row["sim_backlog"] = state.backlog
                sim_row["sim_arrivals"] = arrivals
                sim_row["sim_total_cost"] = state.history[-1]["cost"]

                cost_row = attach_inventory_costs(
                    pd.DataFrame([sim_row]), inventory_config, series_cost_profile
                ).iloc[0]

                results.append(cost_row)

    return pd.DataFrame(results)
