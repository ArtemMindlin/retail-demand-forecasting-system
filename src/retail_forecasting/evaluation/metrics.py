from __future__ import annotations

import math

import numpy as np
import pandas as pd


def summarize_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    fold_records = []

    group_cols = ["model_name", "backend_name"]
    if "data_strategy" in predictions.columns:
        group_cols.insert(0, "data_strategy")

    for keys, subset in predictions.groupby(group_cols, dropna=False):
        if "data_strategy" in group_cols:
            strategy, model_name, backend_name = keys
        else:
            model_name, backend_name = keys
            strategy = None

        record = _build_metric_record(subset, model_name, backend_name)
        if strategy:
            record["data_strategy"] = strategy
        records.append(record)

    fold_group_cols = ["fold_id"] + group_cols
    for keys, subset in predictions.groupby(fold_group_cols, dropna=False):
        if "data_strategy" in group_cols:
            fold_id, strategy, model_name, backend_name = keys
        else:
            fold_id, model_name, backend_name = keys
            strategy = None

        record = _build_metric_record(subset, model_name, backend_name)
        record["fold_id"] = fold_id
        if strategy:
            record["data_strategy"] = strategy
        fold_records.append(record)

    return pd.DataFrame(records), pd.DataFrame(fold_records)


def summarize_costs(predictions: pd.DataFrame) -> pd.DataFrame:
    enriched = predictions.copy()
    enriched["service_level_hit"] = (
        enriched["stockout_units"].to_numpy(dtype=float) <= 0.0
    ).astype(float)
    enriched["served_units"] = np.minimum(
        enriched["y_true"].to_numpy(dtype=float),
        enriched["order_quantity"].to_numpy(dtype=float),
    )

    if "sim_backlog" in enriched.columns:
        enriched["sim_service_level_hit"] = (
            enriched["sim_backlog"].to_numpy(dtype=float) <= 0.0
        ).astype(float)

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

    if "sim_total_cost" in enriched.columns:
        agg_map.update(
            {
                "sim_total_cost": ("sim_total_cost", "sum"),
                "sim_mean_cost": ("sim_total_cost", "mean"),
                "sim_service_level": ("sim_service_level_hit", "mean"),
            }
        )

    summary = enriched.groupby(group_cols, dropna=False).agg(**agg_map).reset_index()

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
    return summary


def _build_metric_record(
    predictions: pd.DataFrame,
    model_name: str,
    backend_name: str,
) -> dict[str, float | str]:
    errors = predictions["y_pred"] - predictions["y_true"]
    record: dict[str, float | str] = {
        "model_name": model_name,
        "backend_name": backend_name,
        "observations": int(len(predictions)),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(math.sqrt(np.square(errors).mean())),
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
        # Legacy name for backward compatibility with existing tests
        record[f"coverage_{lower_column}_{upper_column}"] = coverage_val

        # MPIW: Mean Prediction Interval Width
        record["interval_width"] = float((upper - lower).mean())

        # Winkler Score
        record["winkler_score"] = winkler_score(y_true, lower, upper, alpha)

    return record


def _find_quantile_columns(predictions: pd.DataFrame) -> list[tuple[float, str]]:
    quantile_pairs = [
        (_parse_quantile_column(column), column)
        for column in predictions.columns
        if column.startswith("q_") and predictions[column].notna().any()
    ]
    return sorted(quantile_pairs, key=lambda item: item[0])


def _parse_quantile_column(column: str) -> float:
    return float(column.replace("q_", "").replace("_", "."))


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    diff = actual - predicted
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


def winkler_score(
    actual: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float
) -> float:
    """
    Calculate the Winkler Score for prediction intervals.
    A proper scoring rule that penalizes both wide intervals and values outside the interval.
    Lower is better.
    """
    width = upper - lower
    under_penalty = (2.0 / alpha) * (lower - actual) * (actual < lower)
    over_penalty = (2.0 / alpha) * (actual - upper) * (actual > upper)
    score = width + under_penalty + over_penalty
    return float(np.mean(score))
