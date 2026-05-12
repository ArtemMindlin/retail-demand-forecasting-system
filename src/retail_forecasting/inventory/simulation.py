from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd
from typing import Any


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
    history: list[dict] = field(default_factory=list)

    def step(self, demand: float, order_quantity: float, arrivals: float):
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
    inventory_config: Any,
    series_cost_profile: pd.DataFrame | None = None,
    initial_on_hand: float = 0.0,
) -> pd.DataFrame:
    """
    Simulates a multi-period inventory policy across backtest folds.

    This replaces the static newsvendor logic with a dynamic one where
    the ending state of fold N is the starting state of fold N+1.

    Now uses 'choose_order_quantity' iteratively to implement an
    Inventory-Aware Order-Up-To policy.
    """
    from retail_forecasting.inventory.newsvendor import (
        attach_inventory_costs,
        choose_order_quantity,
    )

    if "fold_id" not in predictions.columns:
        return predictions  # Cannot simulate without temporal order

    results = []

    # Identify available quantile columns for the decider
    quantile_columns = [c for c in predictions.columns if c.startswith("q_")]
    quantile_levels = [
        float(c.replace("q_", "").replace("_", ".")) for c in quantile_columns
    ]

    # Group by model and series to maintain state continuity
    group_cols = ["data_strategy", "model_name", "backend_name", "series_id"]
    group_cols = [c for c in group_cols if c in predictions.columns]

    for keys, subset in predictions.groupby(group_cols):
        subset = subset.sort_values("fold_id")

        # Initialize state
        state = InventoryState(
            series_id=subset["series_id"].iloc[0],
            on_hand=initial_on_hand,
            c_over=subset["c_over"].iloc[0] if "c_over" in subset.columns else 1.0,
            c_under=subset["c_under"].iloc[0] if "c_under" in subset.columns else 4.0,
        )

        # Tracking pipeline: (arrival_fold, quantity)
        pending_orders = []

        for _, row in subset.iterrows():
            current_fold = row["fold_id"]

            # 1. Collect arrivals for this fold (orders placed in previous folds)
            arrivals = sum(q for f, q in pending_orders if f == current_fold)
            pending_orders = [(f, q) for f, q in pending_orders if f != current_fold]

            # 2. DECISION: Choose order quantity considering current inventory position
            # We use a single-row dataframe for the decider call
            decider_input = pd.DataFrame([row])

            # On-Hand minus Backlog is the net stock available
            net_stock = pd.Series(
                [state.on_hand - state.backlog], index=decider_input.index
            )
            # Sum of all pending orders is the On-Order amount
            on_order_val = sum(q for f, q in pending_orders)
            on_order_series = pd.Series([on_order_val], index=decider_input.index)

            # Recalculate order_quantity dynamically
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

            # 3. ADVANCE: Advance the physical simulation
            state.step(
                demand=row["y_true"], order_quantity=new_order_qty, arrivals=arrivals
            )

            # 4. SCHEDULE: Arrival for next fold (lead time = 1 fold/horizon)
            pending_orders.append((current_fold + 1, new_order_qty))

            # 5. RECORD: Save enriched row
            sim_row = row.copy()
            sim_row["order_quantity"] = (
                new_order_qty  # Update with inventory-aware decision
            )
            sim_row["sim_on_hand"] = state.on_hand
            sim_row["sim_backlog"] = state.backlog
            sim_row["sim_arrivals"] = arrivals
            sim_row["sim_total_cost"] = state.history[-1]["cost"]

            # Re-calculate costs for this row based on updated order_quantity (for static baseline comparison)
            cost_row = attach_inventory_costs(
                pd.DataFrame([sim_row]), inventory_config, series_cost_profile
            ).iloc[0]

            results.append(cost_row)

    return pd.DataFrame(results)
