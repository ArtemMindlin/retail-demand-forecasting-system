from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def optimize_orders_lp(
    unconstrained_orders: dict[str, float],
    marginal_utilities: dict[str, float],
    global_capacity: float,
) -> dict[str, float]:
    """
    Optimize order quantities across SKUs to respect a global capacity constraint,
    maximizing the total marginal utility (e.g., avoided stockout cost).

    This is formulated as a Linear Programming problem (Continuous Knapsack).

    Args:
        unconstrained_orders: Mapping of series_id to optimal unconstrained order quantity.
        marginal_utilities: Mapping of series_id to the marginal utility of ordering a unit (e.g., c_under).
        global_capacity: The maximum total units that can be ordered across all SKUs.

    Returns:
        Mapping of series_id to constrained optimal order quantity.
    """
    series_ids = list(unconstrained_orders.keys())
    n_items = len(series_ids)

    if n_items == 0:
        return {}

    bounds = []
    c = []

    total_unconstrained = 0.0
    for sid in series_ids:
        max_qty = unconstrained_orders[sid]
        total_unconstrained += max_qty
        bounds.append((0.0, max_qty))

        # We want to MAXIMIZE utility * quantity. linprog MINIMIZES.
        # So we negate the coefficients.
        c.append(-marginal_utilities[sid])

    # If the total unconstrained orders already fit within capacity, no need to optimize
    if total_unconstrained <= global_capacity:
        return unconstrained_orders.copy()

    # Global capacity constraint: sum(x_i) <= global_capacity
    A_ub = np.ones((1, n_items))
    b_ub = np.array([global_capacity])

    # "highs" is the standard modern solver in scipy
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        # Fallback to proportional scaling if LP fails (should be rare)
        scale = global_capacity / total_unconstrained
        return {sid: unconstrained_orders[sid] * scale for sid in series_ids}

    constrained_orders = {}
    for i, sid in enumerate(series_ids):
        constrained_orders[sid] = float(result.x[i])

    return constrained_orders
