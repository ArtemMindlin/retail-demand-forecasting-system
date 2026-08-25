from __future__ import annotations

import pandas as pd

from retail_forecasting.config import InventoryConfig
from retail_forecasting.inventory.cost_profiles import attach_series_costs
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
    assert 10.0 < float(order_quantity.iloc[0]) <= 12.0
    assert float(evaluated["total_cost"].iloc[0]) >= 0.0


def test_attach_series_costs_uses_one_cost_pair_for_the_catalogue() -> None:
    """One pair for everything, and no way to vary it per series.

    A synthetic per-series profile used to sit behind a `use_series_costs` flag. It is
    gone, so this pins the surviving contract: the coefficients come from the config and
    every row gets the same ones.
    """
    inventory = InventoryConfig(overstock_cost=1.5, stockout_cost=5.0)
    predictions = pd.DataFrame({"series_id": ["a", "b", "c"], "y_pred": [1.0, 2.0, 3.0]})

    enriched = attach_series_costs(predictions, inventory)

    assert (enriched["c_over"] == 1.5).all()
    assert (enriched["c_under"] == 5.0).all()
    assert enriched["critical_fractile"].nunique() == 1
