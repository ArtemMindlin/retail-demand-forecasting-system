from __future__ import annotations

import pandas as pd

from retail_forecasting.config import InventoryConfig
from retail_forecasting.inventory.newsvendor import (
    attach_inventory_costs,
    choose_order_quantity,
    critical_fractile,
)


def test_newsvendor_costs_follow_critical_fractile() -> None:
    inventory = InventoryConfig(overstock_cost=1.0, stockout_cost=4.0)
    predictions = pd.DataFrame(
        {
            "y_true": [10.0],
            "y_pred": [9.0],
            "q_0_1": [8.0],
            "q_0_5": [10.0],
            "q_0_9": [12.0],
        }
    )

    order_quantity = choose_order_quantity(
        predictions=predictions,
        inventory_config=inventory,
        quantile_columns=["q_0_1", "q_0_5", "q_0_9"],
        quantile_levels=[0.1, 0.5, 0.9],
    )
    evaluated = attach_inventory_costs(
        predictions.assign(order_quantity=order_quantity),
        inventory_config=inventory,
    )

    assert critical_fractile(inventory) == 0.8
    assert float(order_quantity.iloc[0]) > 10.0
    assert float(evaluated["total_cost"].iloc[0]) >= 0.0
