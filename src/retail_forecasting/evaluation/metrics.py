from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from retail_forecasting.utils.io import (
    HOLDOUT_FOLD_ID,
    quantile_level_from_column,
    winkler_score,
)

__all__ = ["summarize_predictions", "summarize_costs", "pinball_loss", "winkler_score"]

# The benchmark `rel_mae_naive` divides by. Duplicated from `models/naive.py` rather than
# imported: `evaluation` may not import `models`, and closing that boundary for one string
# would be a worse trade than repeating it.
NAIVE_MODEL_NAME = "seasonal_naive"

N_BOOTSTRAP_RESAMPLES = 2000

# Below this many clusters the interval is too wide to rank anything, and the honest reading is
# that the run cannot tell the models apart. Same threshold `simulation/operations.py` applies to
# its own clusters.
MIN_SERIES_FOR_INFERENCE = 8


def _exclude_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the holdout fold so global summaries don't mix it with walk-forward folds."""
    if "fold_id" in df.columns:
        return df[df["fold_id"] != HOLDOUT_FOLD_ID]
    return df


def _split_group_keys(keys: Any, group_cols: list[str]) -> dict[str, Any]:
    """Map a pandas groupby key (tuple or scalar) to a {column: value} dict."""
    values = keys if isinstance(keys, tuple) else (keys,)
    return dict(zip(group_cols, values, strict=True))


def summarize_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    fold_records = []

    group_cols = ["model_name", "backend_name"]
    if "data_strategy" in predictions.columns:
        group_cols.insert(0, "data_strategy")

    # Global summary metrics exclude the holdout fold to avoid mixing it with walk-forward folds.
    for keys, subset in _exclude_holdout(predictions).groupby(group_cols, dropna=False):
        key_map = _split_group_keys(keys, group_cols)
        record = _build_metric_record(subset, key_map["model_name"], key_map["backend_name"])
        if key_map.get("data_strategy"):
            record["data_strategy"] = key_map["data_strategy"]
        records.append(record)

    fold_group_cols = ["fold_id", *group_cols]
    for keys, subset in predictions.groupby(fold_group_cols, dropna=False):
        key_map = _split_group_keys(keys, fold_group_cols)
        record = _build_metric_record(subset, key_map["model_name"], key_map["backend_name"])
        record["fold_id"] = key_map["fold_id"]
        if key_map.get("data_strategy"):
            record["data_strategy"] = key_map["data_strategy"]
        fold_records.append(record)

    summary = _attach_rel_mae_naive(pd.DataFrame(records), group_cols)
    folds = _attach_rel_mae_naive(pd.DataFrame(fold_records), ["fold_id", *group_cols])
    return summary, folds


def _attach_rel_mae_naive(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Scale each model's MAE by the seasonal naive's, on the SAME rows. Below 1 beats it.

    This is a RelMAE, and calling it MASE would be an abuse worth avoiding in writing. Textbook
    MASE divides by the IN-SAMPLE one-step seasonal-naive error, and there is no such quantity
    here: the target is demand summed over `horizon` days, for which a one-step naive error is
    undefined. What this divides by is the seasonal naive's MAE over the same evaluation rows,
    which is the benchmark the thesis actually argues against.

    The naive is scored inside every run, so the denominator costs nothing. It is NaN when the
    naive is absent -- a scoring run carries the champion alone -- rather than silently absent,
    since a missing column reads as "not measured" and a missing value as "not comparable".
    """
    if summary.empty or "mae" not in summary.columns:
        return summary
    # Scaled WITHIN its own comparison group: an experiment scores one demand strategy, and each
    # fold has its own difficulty, so a single global denominator would mix both.
    scope = [column for column in group_cols if column not in ("model_name", "backend_name")]
    baseline = summary.loc[summary["model_name"] == NAIVE_MODEL_NAME, [*scope, "mae"]]
    if baseline.empty:
        summary["rel_mae_naive"] = float("nan")
        return summary

    if scope:
        baseline = baseline.drop_duplicates(subset=scope).rename(columns={"mae": "_naive_mae"})
        # Left merge on de-duplicated keys, so length and row order survive and the positional
        # assignment below lines up.
        denominator = summary.merge(baseline, on=scope, how="left")["_naive_mae"]
    else:
        denominator = pd.Series([float(baseline["mae"].iloc[0])] * len(summary))

    scaled = denominator.to_numpy(dtype=float)
    summary["rel_mae_naive"] = np.where(
        scaled > 0.0, summary["mae"].to_numpy(dtype=float) / scaled, np.nan
    )
    return summary


def _scope_key(frame: pd.DataFrame, scope: list[str]) -> pd.Series:
    """One string per row identifying its comparison group, or a constant when there is none."""
    if not scope:
        return pd.Series("", index=frame.index)
    return frame[scope].astype(str).agg("|".join, axis=1)


def _attach_cost_gap_naive(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    group_cols: list[str],
    random_seed: int,
) -> pd.DataFrame:
    """Each model's total-cost gap against the seasonal naive, with a paired bootstrap interval.

    Without this the run reports a cost per model and nothing about whether the gap between two
    of them is real. The champion comparison turned on a 0.05% difference, which is plainly
    within noise -- but "plainly" is not a measurement, and the methodology this project declares
    requires that a difference exceed the experiment's own variability before it counts.

    Resamples whole SERIES as clusters. `simulation/operations.py` runs the same statistic over
    whole ORIGINS, and the difference is deliberate: there the scoring grid is one origin per
    week and the series inside a week share that week's demand shock. Here the comparison is
    PAIRED on identical rows, so the day's shock reaches both models and cancels in the
    difference, leaving series-idiosyncratic misfit. Series are also the better-powered unit in
    this design, which has 500 of them against 21 validation dates.
    """
    for column in ("cost_change_pct", "ci95_low_pct", "ci95_high_pct"):
        summary[column] = float("nan")
    summary["conclusive"] = False
    if summary.empty or "series_id" not in predictions.columns:
        return summary

    scope = [column for column in group_cols if column not in ("model_name", "backend_name")]
    predictions = predictions.assign(_scope=_scope_key(predictions, scope))
    rng = np.random.default_rng(random_seed)
    gaps: dict[tuple[str, str], dict[str, float | bool]] = {}

    for scope_value, subset in predictions.groupby("_scope", dropna=False):
        paired = subset.pivot_table(
            index="series_id",
            columns="model_name",
            values="total_cost",
            aggfunc="sum",
            fill_value=0.0,
        )
        if NAIVE_MODEL_NAME not in paired.columns or paired.empty:
            continue
        baseline = paired[NAIVE_MODEL_NAME].to_numpy(dtype=float)
        n_series = len(paired)
        total_baseline = float(baseline.sum())
        draws = rng.integers(0, n_series, size=(N_BOOTSTRAP_RESAMPLES, n_series))
        resampled_baseline = baseline[draws].sum(axis=1)

        for model in paired.columns:
            delta = paired[model].to_numpy(dtype=float) - baseline
            change = 100.0 * float(delta.sum()) / total_baseline if total_baseline else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = 100.0 * delta[draws].sum(axis=1) / resampled_baseline
            finite = boot[np.isfinite(boot)]
            low, high = (
                (float(x) for x in np.percentile(finite, [2.5, 97.5]))
                if finite.size
                else (float("nan"), float("nan"))
            )
            gaps[(str(scope_value), str(model))] = {
                "cost_change_pct": change,
                "ci95_low_pct": low,
                "ci95_high_pct": high,
                # The naive against itself is a zero gap by construction, never a finding.
                "conclusive": bool(
                    model != NAIVE_MODEL_NAME
                    and n_series >= MIN_SERIES_FOR_INFERENCE
                    and (low > 0.0 or high < 0.0)
                ),
            }

    keys = list(zip(_scope_key(summary, scope), summary["model_name"].astype(str), strict=True))
    for column in ("cost_change_pct", "ci95_low_pct", "ci95_high_pct"):
        summary[column] = [float(gaps.get(key, {}).get(column, float("nan"))) for key in keys]
    # Default False rather than missing: a model with no comparison has not been shown to differ.
    summary["conclusive"] = [bool(gaps.get(key, {}).get("conclusive", False)) for key in keys]
    return summary


def summarize_costs(predictions: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    enriched = predictions.copy()
    enriched["service_level_hit"] = (
        enriched["stockout_units"].to_numpy(dtype=float) <= 0.0
    ).astype(float)
    enriched["served_units"] = np.minimum(
        enriched["y_true"].to_numpy(dtype=float),
        enriched["order_quantity"].to_numpy(dtype=float),
    )

    group_cols = ["model_name", "backend_name"]
    if "data_strategy" in predictions.columns:
        group_cols.insert(0, "data_strategy")

    agg_map = {
        "observations": ("y_true", "size"),
        "mean_order_quantity": ("order_quantity", "mean"),
        "total_overstock_units": ("overstock_units", "sum"),
        "total_stockout_units": ("stockout_units", "sum"),
        "total_overstock_cost": ("overstock_cost", "sum"),
        "total_stockout_cost": ("stockout_cost", "sum"),
        "total_cost": ("total_cost", "sum"),
        "mean_cost": ("total_cost", "mean"),
        "service_level": ("service_level_hit", "mean"),
        "served_units": ("served_units", "sum"),
        "total_demand": ("y_true", "sum"),
    }

    # Global summary costs exclude the holdout fold to avoid mixing it with walk-forward folds.
    summary = (
        _exclude_holdout(enriched).groupby(group_cols, dropna=False).agg(**agg_map).reset_index()
    )

    summary["fill_rate"] = np.where(
        summary["total_demand"] > 0.0,
        summary["served_units"] / summary["total_demand"],
        1.0,
    )
    summary = (
        summary.drop(columns=["served_units", "total_demand"])
        .sort_values("total_cost")
        .reset_index(drop=True)
    )
    return _attach_cost_gap_naive(
        summary, _exclude_holdout(enriched), group_cols, random_seed=random_seed
    )


def _ratio_of_sums(numerator: float, denominator: float) -> float:
    """`numerator / denominator` as a percentage, NaN when there is no demand to divide by."""
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator * 100.0


def _build_metric_record(
    predictions: pd.DataFrame,
    model_name: str,
    backend_name: str,
) -> dict[str, float | str]:
    errors = predictions["y_pred"] - predictions["y_true"]
    demand = float(predictions["y_true"].sum())
    record: dict[str, float | str] = {
        "model_name": model_name,
        "backend_name": backend_name,
        "observations": int(len(predictions)),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(math.sqrt(np.square(errors).mean())),
        # MAE as a fraction of the demand it was made against. Within ONE evaluation set this
        # cannot reorder models -- every model scores the same rows, so `wape` is `mae` times the
        # constant `n / sum(y)` -- and that is not what it is for. It is for reading an error
        # ACROSS panels of different scale, which a raw MAE cannot: the base subset and the
        # 500-series run are not comparable in units. A ratio of sums, so unlike MAPE it
        # survives the 4.5% of days this panel has with zero demand.
        "wape": _ratio_of_sums(float(np.abs(errors).sum()), demand),
        # SIGNED, and reported beside the absolute errors on purpose: MAE and RMSE cannot tell
        # over-prediction from under-prediction, and in inventory that is the difference between
        # a stockout and tied-up capital. The system's central claim is about a downward BIAS
        # from censoring, which the absolute metrics are blind to.
        "mean_error": float(errors.mean()),
        "bias_pct": _ratio_of_sums(float(errors.sum()), demand),
    }

    quantile_pairs = _find_quantile_columns(predictions)
    for quantile, column in quantile_pairs:
        record[f"pinball_{column}"] = float(
            pinball_loss(predictions["y_true"], predictions[column], quantile)
        )

    if len(quantile_pairs) >= 2:
        # Use the outermost quantiles to evaluate the prediction interval
        lower_q, lower_column = quantile_pairs[0]
        upper_q, upper_column = quantile_pairs[-1]
        alpha = lower_q + (1.0 - upper_q)

        y_true = predictions["y_true"]
        lower = predictions[lower_column]
        upper = predictions[upper_column]

        # PICP: Prediction Interval Coverage Probability
        coverage_val = float(((y_true >= lower) & (y_true <= upper)).mean())
        record["interval_coverage"] = coverage_val
        # Legacy name kept for backward compatibility (asserted by tests/test_quantile_contract.py)
        record[f"coverage_{lower_column}_{upper_column}"] = coverage_val

        # MPIW: Mean Prediction Interval Width
        record["interval_width"] = float((upper - lower).mean())

        # Winkler Score
        record["winkler_score"] = winkler_score(y_true, lower, upper, alpha)

    return record


def _find_quantile_columns(predictions: pd.DataFrame) -> list[tuple[float, str]]:
    quantile_pairs = [
        (quantile_level_from_column(column), column)
        for column in predictions.columns
        if column.startswith("q_") and predictions[column].notna().any()
    ]
    return sorted(quantile_pairs, key=lambda item: item[0])


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    return float(mean_pinball_loss(actual, predicted, alpha=quantile))
