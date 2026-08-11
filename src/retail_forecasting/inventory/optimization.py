from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def _demand_segments(
    order_cap: float,
    quantiles: list[tuple[float, float]],
    c_under: float,
    c_over: float,
) -> list[tuple[float, float]]:
    """Piecewise-linear marginal value of successive units for one SKU.

    Returns ``[(width, marginal_value), ...]`` from the most to the least valuable
    unit. The marginal value of the unit at position ``q`` is

        dV/dq = c_under - (c_under + c_over) * F(q)

    which is the derivative of the Newsvendor expected cost: a unit avoids a
    shortage with probability ``1 - F(q)`` and becomes overstock with probability
    ``F(q)``. It is positive only while ``F(q)`` stays below the critical ratio
    ``c_under / (c_under + c_over)``, so the segments beyond that point carry a
    negative value and the solver leaves them empty on its own.

    ``F`` is read off the model's own predicted quantiles, taking the midpoint level
    of each interval as the representative probability of the segment.
    """
    points = sorted({(float(level), float(value)) for level, value in quantiles})
    if not points:
        return [(max(0.0, order_cap), c_under)]

    segments: list[tuple[float, float]] = []
    previous_level, previous_value = 0.0, 0.0
    remaining = max(0.0, order_cap)
    total = c_under + c_over

    for level, value in points:
        if remaining <= 0.0:
            break
        width = min(max(0.0, value - previous_value), remaining)
        if width > 0.0:
            midpoint = (previous_level + level) / 2.0
            segments.append((width, c_under - total * midpoint))
            remaining -= width
        previous_level, previous_value = level, value

    # Anything the Newsvendor asked for beyond the top predicted quantile keeps the
    # marginal value implied by that upper tail.
    if remaining > 0.0:
        segments.append((remaining, c_under - total * previous_level))
    return segments


def optimize_orders_lp(
    unconstrained_orders: dict[str, float],
    marginal_utilities: dict[str, float],
    global_capacity: float,
    demand_quantiles: dict[str, list[tuple[float, float]]] | None = None,
    holding_costs: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Optimize order quantities across SKUs to respect a global capacity constraint.

    Two formulations, selected by whether the caller supplies a demand distribution:

    * With ``demand_quantiles``, each SKU is split into segments of decreasing
      marginal value derived from its predicted quantiles, and the LP fills the
      globally most valuable units first. This is the correct formulation: the value
      of a unit falls as it covers deeper into the demand distribution.
    * Without it, every unit of an SKU carries the same value ``c_under``. That makes
      the objective linear in ``q_i`` over box constraints, so the optimum is a corner
      solution -- SKUs with high ``c_under`` are filled to their cap and the rest are
      starved to zero. Kept only for callers that have no distribution to pass; it
      loses to a blind proportional cutback whenever orders carry slack over realized
      demand, because zeroing an SKU guarantees a stockout that spreading avoids.

    Args:
        unconstrained_orders: Mapping of series_id to optimal unconstrained order quantity.
        marginal_utilities: Mapping of series_id to the shortage cost per unit (``c_under``).
        global_capacity: The maximum total units that can be ordered across all SKUs.
        demand_quantiles: Optional mapping of series_id to ``[(level, value), ...]``
            predicted demand quantiles, used to build the piecewise value curve.
        holding_costs: Optional mapping of series_id to the overstock cost per unit
            (``c_over``). Defaults to 0, which makes every unit up to the cap valuable.

    Returns:
        Mapping of series_id to constrained optimal order quantity.
    """
    series_ids = list(unconstrained_orders.keys())
    n_items = len(series_ids)

    if n_items == 0:
        return {}

    total_unconstrained = sum(max(0.0, unconstrained_orders[sid]) for sid in series_ids)

    # If the total unconstrained orders already fit within capacity, no need to optimize
    if total_unconstrained <= global_capacity:
        return unconstrained_orders.copy()

    # One LP variable per (series, value segment). Without a demand distribution each
    # series contributes a single full-width segment, which reproduces the original
    # flat-utility formulation exactly.
    owners: list[str] = []
    bounds: list[tuple[float, float]] = []
    c: list[float] = []
    for sid in series_ids:
        cap = max(0.0, unconstrained_orders[sid])
        if demand_quantiles is None:
            segments = [(cap, marginal_utilities[sid])]
        else:
            segments = _demand_segments(
                order_cap=cap,
                quantiles=demand_quantiles.get(sid, []),
                c_under=marginal_utilities[sid],
                c_over=(holding_costs or {}).get(sid, 0.0),
            )
        for width, value in segments:
            owners.append(sid)
            bounds.append((0.0, width))
            # We want to MAXIMIZE value * quantity. linprog MINIMIZES, so negate.
            c.append(-value)

    # Global capacity constraint: sum(x_j) <= global_capacity
    A_ub = np.ones((1, len(c)))
    b_ub = np.array([global_capacity])

    # "highs" is the standard modern solver in scipy
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        # Fallback to proportional scaling if LP fails (should be rare)
        scale = global_capacity / total_unconstrained
        return {sid: unconstrained_orders[sid] * scale for sid in series_ids}

    constrained_orders = dict.fromkeys(series_ids, 0.0)
    for owner, allocated in zip(owners, result.x, strict=True):
        constrained_orders[owner] += float(allocated)

    return constrained_orders
