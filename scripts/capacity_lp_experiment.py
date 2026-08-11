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

    quantile_columns = sorted(c for c in frame.columns if c.startswith("q_"))
    quantile_levels = [float(c.replace("q_", "").replace("_", ".")) for c in quantile_columns]

    periods = []
    for fold_id, period in frame.groupby("fold_id"):
        periods.append(
            {
                "fold_id": fold_id,
                "orders": {
                    str(row.series_id): max(0.0, float(row.order_quantity))
                    for row in period.itertuples()
                },
                "c_under": {str(row.series_id): float(row.c_under) for row in period.itertuples()},
                "c_over": {str(row.series_id): float(row.c_over) for row in period.itertuples()},
                "demand": {str(row.series_id): float(row.y_true) for row in period.itertuples()},
                "quantiles": {
                    str(row["series_id"]): list(
                        zip(quantile_levels, [float(row[c]) for c in quantile_columns], strict=True)
                    )
                    for _, row in period.iterrows()
                },
            }
        )

    records = []
    for fraction in args.capacity_fraction:
        totals = {"piecewise": 0.0, "flat": 0.0, "proportional": 0.0}
        zeroed = {"piecewise": 0, "flat": 0}
        for period in periods:
            orders, c_under, c_over = period["orders"], period["c_under"], period["c_over"]
            demand, quantiles = period["demand"], period["quantiles"]
            requested = sum(orders.values())
            capacity = requested * fraction
            scale = capacity / requested if requested > 0 else 0.0

            allocations = {
                "piecewise": optimize_orders_lp(
                    orders, c_under, capacity, demand_quantiles=quantiles, holding_costs=c_over
                ),
                "flat": optimize_orders_lp(orders, c_under, capacity),
                "proportional": {sid: qty * scale for sid, qty in orders.items()},
            }
            for name, allocation in allocations.items():
                totals[name] += penalty(allocation, demand, c_under)
                if name in zeroed:
                    zeroed[name] += sum(1 for qty in allocation.values() if qty < 1e-9)

        baseline = totals["proportional"]
        records.append(
            {
                "capacity_fraction": fraction,
                "piecewise_lp_penalty": round(totals["piecewise"], 2),
                "flat_lp_penalty": round(totals["flat"], 2),
                "proportional_penalty": round(baseline, 2),
                "piecewise_vs_proportional_pct": round(
                    (baseline - totals["piecewise"]) / baseline * 100.0 if baseline else 0.0, 2
                ),
                "flat_vs_proportional_pct": round(
                    (baseline - totals["flat"]) / baseline * 100.0 if baseline else 0.0, 2
                ),
                "piecewise_series_zeroed": zeroed["piecewise"],
                "flat_series_zeroed": zeroed["flat"],
            }
        )

    result = pd.DataFrame(records)
    output = args.run / OUTPUT_NAME
    result.to_csv(output, index=False)

    print("\n── Capacity allocation vs proportional cutback (positive = beats it) ──")
    print(result.to_string(index=False))
    for column, label in (
        ("piecewise_vs_proportional_pct", "piecewise LP"),
        ("flat_vs_proportional_pct", "flat-utility LP"),
    ):
        wins = result[result[column] > 0]
        if wins.empty:
            print(f"\n⚠️  The {label} never beats the proportional cutback.")
        else:
            best = wins.loc[wins[column].idxmax()]
            print(
                f"\n✅ {label}: beats proportional at {len(wins)}/{len(result)} capacity levels, "
                f"best {best[column]:.1f}% at {best['capacity_fraction']:.0%}"
            )
    print(f"\n   written to {output}\n")


if __name__ == "__main__":
    main()
