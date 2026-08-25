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

The sampling breadth is config, not a flag: it turned out to BE half the conclusion, so
it is declared and versioned alongside the result. See `inventory.sensitivity_*` in
`docs/config_reference.md`.

Entered through `run_mode = cost_sensitivity`; `--run` names the finished run to analyse.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig, Settings
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.inventory.cost_profiles import (
    CostHeuristicCoefficients,
    build_series_cost_profile,
)
from retail_forecasting.inventory.newsvendor import (
    attach_inventory_costs,
    choose_order_quantity,
)
from retail_forecasting.tracking import (
    EXPERIMENT_SENSITIVITY,
    open_run_directory,
)
from retail_forecasting.utils.io import quantile_level_from_column
from retail_forecasting.utils.logging import Table, fields, get_logger, rule

logger = get_logger(__name__)

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


SCALE_FIELDS = (
    "perishability_base",
    "perishability_multiplier",
    "slow_moving_base",
    "slow_moving_multiplier",
    "service_criticality_base",
    "service_criticality_multiplier",
)
WEIGHT_FIELDS = ("perishability_weights", "slow_moving_weights", "criticality_weights")


def _one_at_a_time(factors: tuple[float, ...]) -> list[tuple[str, CostHeuristicCoefficients]]:
    """Perturb one constant at a time, renormalising the weight group it belongs to."""
    base = CostHeuristicCoefficients()
    cases: list[tuple[str, CostHeuristicCoefficients]] = []

    for factor in factors:
        for field_name in WEIGHT_FIELDS:
            for index in range(len(getattr(base, field_name))):
                weights = list(getattr(base, field_name))
                weights[index] *= factor
                total = sum(weights)
                # Renormalised, so this asks "does the SPLIT between dimensions matter",
                # not "does the overall scale matter" -- that is what the scale fields test.
                weights = [weight / total for weight in weights]
                cases.append(
                    (
                        f"{field_name}[{index}] x{factor}",
                        replace(base, **{field_name: tuple(weights)}),  # type: ignore[arg-type]
                    )
                )
        for field_name in SCALE_FIELDS:
            value = getattr(base, field_name) * factor
            cases.append((f"{field_name} x{factor}", replace(base, **{field_name: value})))
    return cases


def _joint(
    rng: np.random.Generator,
    draws: int,
    scale_span: float,
    weight_concentration: float,
) -> list[tuple[str, CostHeuristicCoefficients]]:
    """Sample every constant at once, centred on the defaults.

    The Dirichlet is parameterised as `concentration * default_weights`, so its MEAN is the
    default split and `concentration` sets how tightly draws cluster around it. A flat
    Dirichlet(1,...,1) would instead be uniform over the simplex and would spend most of its
    draws on degenerate splits like (0.99, 0.005, 0.005) -- that answers "what if I had no
    idea", where the useful question is "how precisely do these need to be right".

    Scales are uniform in `default * (1 +- scale_span)`.
    """
    base = CostHeuristicCoefficients()
    cases: list[tuple[str, CostHeuristicCoefficients]] = []
    for draw in range(draws):
        values: dict[str, Any] = {}
        for field_name in WEIGHT_FIELDS:
            defaults = np.asarray(getattr(base, field_name), dtype=float)
            values[field_name] = tuple(rng.dirichlet(weight_concentration * defaults))
        for field_name in SCALE_FIELDS:
            default = getattr(base, field_name)
            values[field_name] = float(
                rng.uniform(default * (1.0 - scale_span), default * (1.0 + scale_span))
            )
        cases.append((f"joint#{draw}", replace(base, **values)))
    return cases


def _analysed_run_name(run_dir: Path) -> str:
    """The analysed run's NAME, not its artifact directory.

    A run's artifacts live under a UUID, so the directory name is unreadable. Every run
    directory carries `mlflow_run.json` for exactly this: the store is a gitignored sqlite
    database, so nothing else in the directory says which run it is.
    """
    marker = run_dir / "mlflow_run.json"
    if marker.exists():
        try:
            return str(json.loads(marker.read_text(encoding="utf-8"))["run_name"])
        except (KeyError, ValueError):
            pass
    return run_dir.name


def run_cost_sensitivity(settings: Settings, run_dir: Path) -> Path:
    """Perturb the cost heuristic over a finished run and write the sweep to a new run.

    `run_dir` is another run's artifact directory: this mode analyses a finished experiment
    rather than producing predictions of its own. It still opens a run of its own, so the
    sweep carries the commit, the config hash and the sampling parameters that produced it --
    two sweeps of the same panel with different breadths are otherwise indistinguishable
    files, and the breadth changes the answer.

    Returns:
        The created run directory path.
    """
    rule(logger, "sensibilidad del coste a los coeficientes de la heurística")

    predictions = pd.read_csv(run_dir / "predictions.csv")
    # `use_cache` off means "rebuild the panel" for a real run; here it would mean
    # re-downloading a panel we already have. The cache key is a hash of the dataset config,
    # so the cached file IS what this config produces -- and it must be the same panel the
    # analysed run used, because a profile is relative to the panel it was built on.
    panel = load_prepared_panel(
        dataset_config=settings.dataset.model_copy(update={"use_cache": True}),
        preprocessing_config=settings.preprocessing,
        split="train",
    )

    inventory = settings.inventory
    factors = tuple(inventory.sensitivity_oat_factors)
    arms = sorted(predictions["data_strategy"].dropna().unique())
    models = sorted(predictions["model_name"].dropna().unique())

    fields(
        logger,
        {
            "analiza": _analysed_run_name(run_dir),
            "panel": f"{panel['series_id'].nunique()} series, {len(panel)} filas",
            "predicciones": f"{len(predictions)} filas",
            "brazos": " · ".join(arms),
            "modelos": " · ".join(models),
            "uno-a-uno": f"{len(_one_at_a_time(factors))} casos, factores "
            f"{' · '.join(f'x{factor:g}' for factor in factors)}",
            "conjunto": f"{inventory.sensitivity_draws} muestras, escalas "
            f"±{inventory.sensitivity_scale_span:.0%}, concentración "
            f"{inventory.sensitivity_weight_concentration:g}",
        },
    )

    rng = np.random.default_rng(settings.project.random_seed)
    cases: list[tuple[str, CostHeuristicCoefficients | None]] = [
        ("default", None),
        ("__flat__", None),
    ]
    cases += _one_at_a_time(factors)
    cases += _joint(
        rng,
        inventory.sensitivity_draws,
        inventory.sensitivity_scale_span,
        inventory.sensitivity_weight_concentration,
    )

    records: list[dict[str, object]] = []
    for arm in arms:
        for model in models:
            subset = predictions[
                (predictions["data_strategy"] == arm) & (predictions["model_name"] == model)
            ]
            if subset.empty:
                continue
            for label, coefficients in cases:
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
                        **_evaluate(
                            subset, inventory, panel, coefficients, flat=label == "__flat__"
                        ),
                    }
                )

    frame = pd.DataFrame(records)
    with open_run_directory(
        f"cost_sensitivity_{settings.reporting.run_name}", EXPERIMENT_SENSITIVITY
    ) as out_dir:
        frame.to_csv(out_dir / "cost_weight_sensitivity.csv", index=False)
        _report(frame)
        fields(logger, {"escrito": out_dir})
        return out_dir


def _report(frame: pd.DataFrame) -> None:
    """Percent change against the default, split into the weights and the scale factors.

    Not a ratio against the profile-vs-flat gap: that gap is ~2% of the cost, and dividing
    by it inflated a 47% move into "24x", which impresses without informing.
    """
    rule(logger, "resultado")
    summary = Table(
        logger,
        {
            "brazo": 18,
            "modelo": 14,
            "coste def.": 11,
            "vs plano": 9,
            "pesos p5–p95": 17,
            "escalas p5–p95": 17,
            "fill def.": 9,
        },
    )
    for (arm, model), group in frame.groupby(["data_strategy", "model_name"]):
        if not float(group["uses_quantiles"].iloc[0]):
            # No quantile curve, so the critical fractile has nothing to move along:
            # `choose_order_quantity` takes the point forecast and ignores it entirely. The
            # sensitivity is zero by construction, which is a finding, not a measurement.
            summary.row({"brazo": arm, "modelo": model, "coste def.": "sin cuantiles"})
            continue

        reference = group.loc[group["case"] == "default"]
        default_cost = float(reference["total_cost_flat"].iloc[0])
        flat_cost = float(group.loc[group["case"] == "flat", "total_cost_flat"].iloc[0])
        one_at_a_time = group.loc[group["design"] == "one_at_a_time"]

        def _span(subset: pd.DataFrame, base: float = default_cost) -> str:
            if subset.empty:
                return "—"
            change = (subset["total_cost_flat"].astype(float) - base) / base * 100.0
            return f"{change.quantile(0.05):+.1f}% {change.quantile(0.95):+.1f}%"

        summary.row(
            {
                "brazo": arm,
                "modelo": model,
                "coste def.": f"{default_cost:.0f}",
                "vs plano": f"{(default_cost - flat_cost) / flat_cost * 100.0:+.2f}%",
                "pesos p5–p95": _span(one_at_a_time[one_at_a_time["case"].str.contains("weights")]),
                "escalas p5–p95": _span(
                    one_at_a_time[~one_at_a_time["case"].str.contains("weights")]
                ),
                "fill def.": f"{float(reference['fill_rate'].iloc[0]):.2f}%",
            }
        )

    rule(logger, "muestreo conjunto")
    joint = Table(
        logger, {"brazo": 18, "modelo": 14, "mediana": 10, "p5–p95": 17, "baten al def.": 14}
    )
    for (arm, model), group in frame.groupby(["data_strategy", "model_name"]):
        if not float(group["uses_quantiles"].iloc[0]):
            continue
        default_cost = float(group.loc[group["case"] == "default", "total_cost_flat"].iloc[0])
        sample = group.loc[group["design"] == "joint", "total_cost_flat"].astype(float)
        if sample.empty:
            continue
        change = (sample - default_cost) / default_cost * 100.0
        joint.row(
            {
                "brazo": arm,
                "modelo": model,
                "mediana": f"{change.median():+.2f}%",
                "p5–p95": f"{change.quantile(0.05):+.1f}% {change.quantile(0.95):+.1f}%",
                "baten al def.": f"{int((sample < default_cost).sum())}/{len(sample)}",
            }
        )
