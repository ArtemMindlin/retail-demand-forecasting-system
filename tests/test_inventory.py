from __future__ import annotations

import pandas as pd
import pytest

from retail_forecasting.config import InventoryConfig
from retail_forecasting.inventory.cost_profiles import (
    attach_series_costs,
    build_series_cost_profile,
)
from retail_forecasting.inventory.newsvendor import (
    attach_inventory_costs,
    choose_order_quantity,
    critical_fractile,
)
from retail_forecasting.inventory.optimization import optimize_orders_lp
from tests import make_synthetic_panel


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


def test_series_cost_profile_builds_row_specific_costs() -> None:
    panel = make_synthetic_panel(num_series=4, num_days=90)
    inventory = InventoryConfig(
        overstock_cost=1.0,
        stockout_cost=4.0,
        use_series_costs=True,
    )

    profile = build_series_cost_profile(panel, inventory)

    assert profile["series_id"].nunique() == panel["series_id"].nunique()
    assert (profile["c_over"] > 0).all()
    assert (profile["c_under"] > 0).all()
    assert profile["critical_fractile"].between(0.0, 1.0).all()
    assert profile["c_over"].nunique() > 1 or profile["c_under"].nunique() > 1


def test_newsvendor_uses_series_specific_critical_fractile() -> None:
    inventory = InventoryConfig(
        overstock_cost=1.0,
        stockout_cost=4.0,
        use_series_costs=True,
    )
    predictions = pd.DataFrame(
        {
            "series_id": ["low_service", "high_service"],
            "y_true": [10.0, 10.0],
            "y_pred": [10.0, 10.0],
            "q_0_1": [8.0, 8.0],
            "q_0_5": [10.0, 10.0],
            "q_0_9": [12.0, 12.0],
        }
    )
    series_cost_profile = pd.DataFrame(
        {
            "series_id": ["low_service", "high_service"],
            "c_over": [2.0, 1.0],
            "c_under": [2.0, 9.0],
            "critical_fractile": [0.5, 0.9],
            "synthetic_perishability_score": [0.4, 0.4],
            "service_criticality_score": [0.2, 0.9],
        }
    )

    order_quantity = choose_order_quantity(
        predictions=predictions,
        inventory_config=inventory,
        quantile_columns=["q_0_1", "q_0_5", "q_0_9"],
        quantile_levels=[0.1, 0.5, 0.9],
        series_cost_profile=series_cost_profile,
    )
    evaluated = attach_inventory_costs(
        predictions.assign(order_quantity=order_quantity),
        inventory_config=inventory,
        series_cost_profile=series_cost_profile,
    )

    assert float(order_quantity.iloc[1]) > float(order_quantity.iloc[0])
    assert {"c_over", "c_under", "critical_fractile"}.issubset(evaluated.columns)
    assert float(evaluated.loc[1, "critical_fractile"]) == 0.9


def test_attach_series_costs_falls_back_to_global_costs() -> None:
    inventory = InventoryConfig(overstock_cost=1.5, stockout_cost=5.0)
    predictions = pd.DataFrame({"series_id": ["a"], "y_pred": [1.0]})

    enriched = attach_series_costs(predictions, inventory)

    assert float(enriched.loc[0, "c_over"]) == 1.5
    assert float(enriched.loc[0, "c_under"]) == 5.0


def test_attach_series_costs_reuses_existing_series_cost_columns() -> None:
    inventory = InventoryConfig(use_series_costs=True)
    predictions = pd.DataFrame(
        {
            "series_id": ["a"],
            "y_pred": [1.0],
            "c_over": [1.2],
            "c_under": [4.8],
            "critical_fractile": [0.8],
        }
    )

    enriched = attach_series_costs(predictions, inventory)

    assert float(enriched.loc[0, "c_over"]) == 1.2
    assert float(enriched.loc[0, "c_under"]) == 4.8
    assert float(enriched.loc[0, "critical_fractile"]) == 0.8


def test_capacity_lp_is_inactive_when_the_constraint_does_not_bind() -> None:
    """The early exit matters: every experiment config sets capacity far above demand."""
    orders = {"a": 10.0, "b": 20.0}
    utilities = {"a": 4.0, "b": 4.0}

    allocated = optimize_orders_lp(orders, utilities, global_capacity=1_000.0)

    assert allocated == orders


def test_capacity_lp_fills_the_most_valuable_units_first() -> None:
    """With a demand distribution the LP must spread, not starve whole SKUs.

    Two identical SKUs and half the requested capacity. The first tranche of each --
    the units below ``q_0.1``, which are almost certain to sell -- outvalues any deeper
    tranche, so both SKUs must receive at least that much before either goes further.
    A constant marginal utility would instead fill one to its cap and zero the other.

    The split of the leftover capacity is not asserted: with identical value curves the
    LP has degenerate optima and any vertex summing to the budget is equally optimal.
    """
    orders = {"a": 10.0, "b": 10.0}
    utilities = {"a": 4.0, "b": 4.0}
    holding = {"a": 1.0, "b": 1.0}
    first_tranche = 2.0
    quantiles = {
        "a": [(0.1, first_tranche), (0.5, 6.0), (0.9, 10.0)],
        "b": [(0.1, first_tranche), (0.5, 6.0), (0.9, 10.0)],
    }

    allocated = optimize_orders_lp(
        orders,
        utilities,
        global_capacity=10.0,
        demand_quantiles=quantiles,
        holding_costs=holding,
    )

    assert sum(allocated.values()) == pytest.approx(10.0)
    assert allocated["a"] >= first_tranche
    assert allocated["b"] >= first_tranche


def test_capacity_lp_prioritises_the_costlier_stockout() -> None:
    """When shortage costs differ, the scarce capacity leans to the expensive SKU."""
    orders = {"cheap": 10.0, "costly": 10.0}
    utilities = {"cheap": 2.0, "costly": 20.0}
    holding = {"cheap": 1.0, "costly": 1.0}
    quantiles = {
        "cheap": [(0.1, 2.0), (0.5, 6.0), (0.9, 10.0)],
        "costly": [(0.1, 2.0), (0.5, 6.0), (0.9, 10.0)],
    }

    allocated = optimize_orders_lp(
        orders,
        utilities,
        global_capacity=10.0,
        demand_quantiles=quantiles,
        holding_costs=holding,
    )

    assert sum(allocated.values()) == pytest.approx(10.0)
    assert allocated["costly"] > allocated["cheap"]


def test_capacity_lp_never_exceeds_the_unconstrained_request() -> None:
    orders = {"a": 3.0, "b": 40.0}
    utilities = {"a": 4.0, "b": 4.0}
    quantiles = {"a": [(0.5, 3.0)], "b": [(0.5, 40.0)]}

    allocated = optimize_orders_lp(
        orders, utilities, global_capacity=20.0, demand_quantiles=quantiles
    )

    assert sum(allocated.values()) <= 20.0 + 1e-9
    for series_id, quantity in allocated.items():
        assert -1e-9 <= quantity <= orders[series_id] + 1e-9
