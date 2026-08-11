"""Compare the capacity LP against a blind proportional cutback under a binding limit.

The default warehouse capacity in every experiment config sits orders of magnitude
above what the evaluated panels request, so `optimize_orders_lp` always returns
through its early exit and the solver never runs. This script forces the constraint
to bind and quantifies what the LP buys over the naive alternative -- the fallback
the module itself uses when the solver fails -- so the claim is backed by an
artifact instead of an ad-hoc measurement.

For each decision period it takes the unconstrained Newsvendor orders from a run's
predictions and allocates a reduced capacity two ways:

    LP           maximise sum(u_i * q_i)  s.t.  sum(q_i) <= C,  0 <= q_i <= d_i
    proportional q_i = d_i * C / sum(d_i)

Both allocations are then charged against the same realized demand, so the only
difference is how the shortfall was distributed across SKUs.

Usage:
    python scripts/capacity_lp_experiment.py \
        --run reports/fresh_retailnet_v2_20260811_123002 --capacity-fraction 0.75
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retail_forecasting.inventory.optimization import optimize_orders_lp

OUTPUT_NAME = "capacity_lp_experiment.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory to read.")
    parser.add_argument("--strategy", default="Latent_supervised", help="Data strategy to use.")
    parser.add_argument("--model", default="catboost", help="Model whose orders are constrained.")
    parser.add_argument(
        "--capacity-fraction",
        type=float,
        nargs="+",
        default=[0.75, 0.6, 0.5, 0.4, 0.3, 0.2],
        help="Capacity as a fraction of the unconstrained request. Accepts several to sweep.",
    )
    return parser


def penalty(
    orders: dict[str, float], demand: dict[str, float], shortage: dict[str, float]
) -> float:
    """Stockout penalty of an allocation against the realized demand."""
    return sum(max(demand[sid] - orders[sid], 0.0) * shortage[sid] for sid in orders)


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_csv(args.run / "predictions.csv", low_memory=False)
    frame = frame[
        (frame["data_strategy"] == args.strategy) & (frame["model_name"] == args.model)
    ].copy()
    if frame.empty:
        raise SystemExit(f"No rows for {args.model}/{args.strategy} in {args.run}")

    periods = []
    for fold_id, period in frame.groupby("fold_id"):
        periods.append(
            (
                fold_id,
                {
                    str(row.series_id): max(0.0, float(row.order_quantity))
                    for row in period.itertuples()
                },
                {str(row.series_id): float(row.c_under) for row in period.itertuples()},
                {str(row.series_id): float(row.y_true) for row in period.itertuples()},
            )
        )

    records = []
    for fraction in args.capacity_fraction:
        lp_total = prop_total = 0.0
        lp_stockouts = prop_stockouts = 0
        zeroed = 0
        for _fold_id, unconstrained, utilities, demand in periods:
            requested = sum(unconstrained.values())
            capacity = requested * fraction
            scale = capacity / requested if requested > 0 else 0.0

            lp_orders = optimize_orders_lp(unconstrained, utilities, capacity)
            proportional = {sid: qty * scale for sid, qty in unconstrained.items()}

            lp_total += penalty(lp_orders, demand, utilities)
            prop_total += penalty(proportional, demand, utilities)
            lp_stockouts += sum(1 for sid in lp_orders if demand[sid] > lp_orders[sid])
            prop_stockouts += sum(1 for sid in proportional if demand[sid] > proportional[sid])
            zeroed += sum(1 for qty in lp_orders.values() if qty < 1e-9)

        records.append(
            {
                "capacity_fraction": fraction,
                "lp_stockout_penalty": round(lp_total, 2),
                "proportional_stockout_penalty": round(prop_total, 2),
                "penalty_reduction_pct": round(
                    (prop_total - lp_total) / prop_total * 100.0 if prop_total else 0.0, 2
                ),
                "lp_series_stocked_out": lp_stockouts,
                "proportional_series_stocked_out": prop_stockouts,
                "lp_series_zeroed": zeroed,
            }
        )

    result = pd.DataFrame(records)
    output = args.run / OUTPUT_NAME
    result.to_csv(output, index=False)

    print("\n── Capacity LP vs proportional cutback (positive reduction = LP wins) ──")
    print(result.to_string(index=False))
    wins = result[result["penalty_reduction_pct"] > 0]
    if wins.empty:
        print("\n⚠️  The LP never beats the proportional cutback on this panel.")
    else:
        best = wins.loc[wins["penalty_reduction_pct"].idxmax()]
        print(
            f"\n✅ Best for the LP: {best['capacity_fraction']:.0%} capacity, "
            f"{best['penalty_reduction_pct']:.1f}% lower stockout penalty"
        )
    print(f"   written to {output}\n")


if __name__ == "__main__":
    main()
