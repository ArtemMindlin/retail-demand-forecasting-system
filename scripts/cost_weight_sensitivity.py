"""Measure how far the inventory cost moves when the cost heuristic's weights move.

The synthetic per-series cost profile is built from nine constants in
`inventory/cost_profiles.py` that were chosen by domain reasoning, not calibrated: the
dataset publishes no shelf life, margin or acquisition cost, so there is no target to fit
them against. This script answers the question that leaves open -- how much does the result
actually depend on them.

It never retrains. The weights do not reach the forecast: the model's own critical fractile
comes from the GLOBAL `stockout_cost`/`overstock_cost`, while the nine constants only shape
the PER-SERIES profile, which enters at `choose_order_quantity` to pick where on the
already-predicted quantile curve the order sits. So a finished run's `predictions.csv` is
enough, and the whole study is a re-decision of stored quantiles.

Two designs, because they answer different questions:

* one-at-a-time: perturb one constant, hold the rest. Says WHICH constant matters.
* joint: sample all nine at once. Says HOW MUCH the cost can move at all, and unlike the
  one-at-a-time pass it sees interactions -- `c_over` multiplies two of the three factors,
  so they cannot be read independently.

The reference that makes the numbers mean something is not the default weights, it is FLAT
costs (`use_series_costs: False`), which charge every series alike. If the spread from
perturbing the weights is small next to the gap between the profile and flat costs, then
what earns its place is differentiating series at all, not the particular numbers used to
differentiate them.

Scope, so the result is not over-read: this measures the SENSITIVITY of cost to the
coefficients. It cannot say the heuristic is well specified. "The weights barely matter" is
not "the cost model is right" -- it is "it barely matters how you parameterise it".

Usage:
    python scripts/cost_weight_sensitivity.py \
        --run    fresh_retailnet_large_20260811_125735 \
        --config configs/experiment/large.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig, load_config
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.inventory.cost_profiles import (
    CostHeuristicCoefficients,
    build_series_cost_profile,
)
from retail_forecasting.inventory.newsvendor import (
    attach_inventory_costs,
    choose_order_quantity,
)
from retail_forecasting.tracking import resolve_run_dir
from retail_forecasting.utils.io import quantile_level_from_column
from retail_forecasting.utils.logging import Table, configure, fields, get_logger, rule

# Named under the package root on purpose: `configure()` attaches the handler to
# `retail_forecasting`, and a script's `__name__` is `__main__`, which sits outside that
# tree and would emit nothing.
logger = get_logger("retail_forecasting.scripts.cost_weight_sensitivity")

# Written by `attach_series_costs` on the run that produced the predictions. They MUST go:
# `attach_series_costs` returns early when it finds them, so leaving them in would silently
# re-charge the stored profile for every perturbation and report perfect insensitivity.
PROFILE_COLUMNS = [
    "c_over",
    "c_under",
    "critical_fractile",
    "synthetic_perishability_score",
    "service_criticality_score",
]
CHARGED_COLUMNS = [
    "overstock_units",
    "stockout_units",
    "overstock_cost",
    "stockout_cost",
    "total_cost",
]
STORED_COST_COLUMNS = [*PROFILE_COLUMNS, "order_quantity", *CHARGED_COLUMNS]

OUTPUT = Path("reports/cost_weight_sensitivity.csv")


def _evaluate(
    predictions: pd.DataFrame,
    settings_inventory: InventoryConfig,
    panel: pd.DataFrame,
    coefficients: CostHeuristicCoefficients | None,
    flat: bool = False,
) -> dict[str, float]:
    """Re-decide one prediction frame under one set of coefficients, and charge it twice.

    Charging twice is the point. A profile does not only DECIDE differently, it also PRICES
    differently: its `c_over`/`c_under` are the global costs times factors above 1, so its
    `total_cost` comes out higher on the same orders. Comparing that against flat costs
    measures the price tag, not the decision -- the same apples-to-oranges that
    `evaluate_fair_inventory_cost` exists to avoid.

    So `total_cost_own` is what the pipeline would report (decision and charge from the same
    profile), and `total_cost_flat` charges every case with the SAME flat coefficients. Only
    the second is comparable across cases, and it is the one the conclusion rests on.
    """
    inventory = settings_inventory
    flat_inventory = inventory.model_copy(update={"use_series_costs": False})

    profile = None
    if not flat:
        profile = build_series_cost_profile(panel, inventory, coefficients)

    frame = predictions.drop(columns=STORED_COST_COLUMNS, errors="ignore").copy()
    # Only the columns actually populated. `seasonal_naive` carries no quantiles, and handing
    # `choose_order_quantity` all-NaN columns makes it interpolate NaN instead of taking the
    # point-forecast branch, which reported a cost of exactly zero.
    quantile_columns = [
        column
        for column in frame.columns
        if column.startswith("q_") and frame[column].notna().any()
    ]
    quantile_levels = [quantile_level_from_column(column) for column in quantile_columns]

    frame["order_quantity"] = choose_order_quantity(
        frame,
        flat_inventory if flat else inventory,
        quantile_columns=quantile_columns,
        quantile_levels=quantile_levels,
        series_cost_profile=profile,
    )
    own = attach_inventory_costs(frame, flat_inventory if flat else inventory, profile)
    # Same orders, one price list for everybody.
    charged_flat = attach_inventory_costs(
        # The profile's own coefficients go, so the flat ones apply; `order_quantity` stays,
        # because charging a different price for the SAME orders is the whole point.
        frame.drop(columns=[*PROFILE_COLUMNS, *CHARGED_COLUMNS], errors="ignore"),
        flat_inventory,
        None,
    )

    demand = float(charged_flat["y_true"].sum())
    stockout_units = float(charged_flat["stockout_units"].sum())
    return {
        "total_cost_own": float(own["total_cost"].sum()),
        "total_cost_flat": float(charged_flat["total_cost"].sum()),
        "fill_rate": (1.0 - stockout_units / demand) * 100.0 if demand > 0 else float("nan"),
        "mean_order": float(frame["order_quantity"].mean()),
        "uses_quantiles": float(bool(quantile_columns)),
    }


def _one_at_a_time(rng: np.random.Generator) -> list[tuple[str, CostHeuristicCoefficients]]:
    """Perturb one constant at a time, renormalising the weight group it belongs to."""
    del rng
    base = CostHeuristicCoefficients()
    cases: list[tuple[str, CostHeuristicCoefficients]] = []

    for factor in (0.5, 0.8, 1.2, 1.5):
        # Weight groups: scale one entry, then renormalise so the group still sums to 1.
        for field_name, index in (
            ("perishability_weights", 0),
            ("perishability_weights", 1),
            ("perishability_weights", 2),
            ("slow_moving_weights", 0),
            ("slow_moving_weights", 1),
            ("criticality_weights", 0),
            ("criticality_weights", 1),
        ):
            weights = list(getattr(base, field_name))
            weights[index] *= factor
            total = sum(weights)
            weights = [weight / total for weight in weights]
            cases.append(
                (
                    f"{field_name}[{index}] x{factor}",
                    replace(base, **{field_name: tuple(weights)}),
                )
            )
        # Scale factors are unconstrained, so they move on their own.
        for field_name in (
            "perishability_base",
            "perishability_multiplier",
            "slow_moving_base",
            "slow_moving_multiplier",
            "service_criticality_base",
            "service_criticality_multiplier",
        ):
            value = getattr(base, field_name) * factor
            cases.append((f"{field_name} x{factor}", replace(base, **{field_name: value})))
    return cases


def _joint(rng: np.random.Generator, draws: int) -> list[tuple[str, CostHeuristicCoefficients]]:
    """Sample every constant at once: Dirichlet for the weight groups, uniform for scales."""
    cases: list[tuple[str, CostHeuristicCoefficients]] = []
    for draw in range(draws):
        cases.append(
            (
                f"joint#{draw}",
                CostHeuristicCoefficients(
                    perishability_weights=tuple(rng.dirichlet([1.0, 1.0, 1.0])),
                    slow_moving_weights=tuple(rng.dirichlet([1.0, 1.0])),
                    criticality_weights=tuple(rng.dirichlet([1.0, 1.0])),
                    perishability_base=float(rng.uniform(0.4, 1.6)),
                    perishability_multiplier=float(rng.uniform(0.4, 1.6)),
                    slow_moving_base=float(rng.uniform(0.45, 1.8)),
                    slow_moving_multiplier=float(rng.uniform(0.25, 1.0)),
                    service_criticality_base=float(rng.uniform(0.45, 1.8)),
                    service_criticality_multiplier=float(rng.uniform(0.25, 1.0)),
                ),
            )
        )
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=resolve_run_dir,
        required=True,
        help="Run name or directory holding predictions.csv.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="The config that produced the run: the profile is rebuilt from its panel, and "
        "a profile is relative to the panel it was built on.",
    )
    parser.add_argument("--draws", type=int, default=200, help="Joint samples (default 200).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure()
    rule(logger, "sensibilidad del coste a los pesos de la heurística")

    settings = load_config(args.config)
    predictions = pd.read_csv(args.run / "predictions.csv")
    # `use_cache` off means "rebuild the panel" for a real run; here it would mean "re-download
    # 45k rows to rebuild a panel we already have". The cache key is a hash of the dataset
    # config, so the cached file IS what this config produces.
    cached_dataset = settings.dataset.model_copy(update={"use_cache": True})
    panel = load_prepared_panel(
        dataset_config=cached_dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )

    arms = sorted(predictions["data_strategy"].dropna().unique())
    models = sorted(predictions["model_name"].dropna().unique())
    fields(
        logger,
        {
            "corrida": args.run.name,
            "panel": f"{panel['series_id'].nunique()} series, {len(panel)} filas",
            "predicciones": f"{len(predictions)} filas",
            "brazos": " · ".join(arms),
            "modelos": " · ".join(models),
            "casos": f"{len(_one_at_a_time(np.random.default_rng(0)))} uno-a-uno "
            f"+ {args.draws} conjuntos, por brazo y modelo",
        },
    )

    rng = np.random.default_rng(args.seed)
    cases = [("default", None), ("__flat__", None)]
    cases += _one_at_a_time(rng)
    cases += _joint(rng, args.draws)

    records: list[dict[str, object]] = []
    progress = Table(logger, {"brazo": 18, "modelo": 15, "casos": 7, "coste def.": 12})
    for arm in arms:
        for model in models:
            subset = predictions[
                (predictions["data_strategy"] == arm) & (predictions["model_name"] == model)
            ]
            if subset.empty:
                continue
            for label, coefficients in cases:
                result = _evaluate(
                    subset,
                    settings.inventory,
                    panel,
                    coefficients,
                    flat=label == "__flat__",
                )
                records.append(
                    {
                        "data_strategy": arm,
                        "model_name": model,
                        "case": "flat" if label == "__flat__" else label,
                        "design": (
                            "reference"
                            if label in ("default", "__flat__")
                            else ("joint" if label.startswith("joint#") else "one_at_a_time")
                        ),
                        **result,
                    }
                )
            default_cost = next(
                float(str(record["total_cost_flat"]))
                for record in records
                if record["data_strategy"] == arm
                and record["model_name"] == model
                and record["case"] == "default"
            )
            progress.row(
                {
                    "brazo": arm,
                    "modelo": model,
                    "casos": len(cases),
                    "coste def.": f"{default_cost:.2f}",
                }
            )

    frame = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    rule(logger, "resultado")
    summary = Table(
        logger,
        {"brazo": 18, "modelo": 15, "def.": 10, "plano": 10, "p5–p95 pesos": 20, "rango/plano": 12},
    )
    for (arm, model), group in frame.groupby(["data_strategy", "model_name"]):
        if not float(group["uses_quantiles"].iloc[0]):
            # No quantile curve, so the critical fractile has nothing to move along:
            # `choose_order_quantity` takes the point forecast and ignores it entirely. The
            # sensitivity is zero by construction, which is a finding, not a measurement.
            summary.row({"brazo": arm, "modelo": model, "def.": "sin cuantiles"})
            continue
        default_cost = float(group.loc[group["case"] == "default", "total_cost_flat"].iloc[0])
        flat_cost = float(group.loc[group["case"] == "flat", "total_cost_flat"].iloc[0])
        perturbed = group.loc[group["design"] != "reference", "total_cost_flat"].astype(float)
        low, high = perturbed.quantile(0.05), perturbed.quantile(0.95)
        gap = abs(default_cost - flat_cost)
        summary.row(
            {
                "brazo": arm,
                "modelo": model,
                "def.": f"{default_cost:.1f}",
                "plano": f"{flat_cost:.1f}",
                "p5–p95 pesos": f"{low:.1f} – {high:.1f}",
                "rango/plano": f"{(high - low) / gap:.2f}x" if gap > 0 else "—",
            }
        )
    fields(logger, {"escrito": args.output})


if __name__ == "__main__":
    main()
