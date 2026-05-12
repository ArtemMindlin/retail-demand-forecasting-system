from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


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
    predictions: pd.DataFrame, initial_on_hand: float = 0.0
) -> pd.DataFrame:
    """
    Simulates a multi-period inventory policy across backtest folds.

    This replaces the static newsvendor logic with a dynamic one where
    the ending state of fold N is the starting state of fold N+1.
    """
    if "fold_id" not in predictions.columns:
        return predictions  # Cannot simulate without temporal order

    results = []

    # Group by model and series
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

        # Pipeline: Order at end of fold N arrives at start of fold N+1
        pending_orders = []  # List of (arrival_fold, quantity)

        for _, row in subset.iterrows():
            current_fold = row["fold_id"]

            # 1. Collect arrivals for this fold
            arrivals = sum(q for f, q in pending_orders if f == current_fold)
            pending_orders = [(f, q) for f, q in pending_orders if f != current_fold]

            # 2. Step the simulation
            # Note: y_true in our pipeline is the lead-time demand for the horizon
            # order_quantity was chosen at the START of the horizon
            state.step(
                demand=row["y_true"],
                order_quantity=row["order_quantity"],
                arrivals=arrivals,
            )

            # 3. Schedule next arrival (assuming lead time = 1 fold/horizon)
            pending_orders.append((current_fold + 1, row["order_quantity"]))

            # 4. Record enriched row
            sim_row = row.copy()
            sim_row["sim_on_hand"] = state.on_hand
            sim_row["sim_backlog"] = state.backlog
            sim_row["sim_arrivals"] = arrivals
            sim_row["sim_total_cost"] = state.history[-1]["cost"]
            results.append(sim_row)

    return pd.DataFrame(results)
