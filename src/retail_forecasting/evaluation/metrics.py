from __future__ import annotations

import math

import numpy as np
import pandas as pd


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    fold_records = []

    for keys, subset in predictions.groupby(["model_name", "backend_name"], dropna=False):
        model_name, backend_name = keys
        records.append(_build_metric_record(subset, model_name, backend_name))

    for keys, subset in predictions.groupby(
        ["fold_id", "model_name", "backend_name"],
        dropna=False,
    ):
        fold_id, model_name, backend_name = keys
        record = _build_metric_record(subset, model_name, backend_name)
        record["fold_id"] = fold_id
        fold_records.append(record)

    return pd.DataFrame(records), pd.DataFrame(fold_records)


def summarize_costs(predictions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        predictions.groupby(["model_name", "backend_name"], dropna=False)
        .agg(
            observations=("y_true", "size"),
            mean_order_quantity=("order_quantity", "mean"),
            total_overstock_units=("overstock_units", "sum"),
            total_stockout_units=("stockout_units", "sum"),
            total_overstock_cost=("overstock_cost", "sum"),
            total_stockout_cost=("stockout_cost", "sum"),
            total_cost=("total_cost", "sum"),
            mean_cost=("total_cost", "mean"),
        )
        .reset_index()
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

    quantile_columns = sorted(
        [
            column
            for column in predictions.columns
            if column.startswith("q_") and predictions[column].notna().any()
        ]
    )
    for column in quantile_columns:
        quantile = float(column.replace("q_", "").replace("_", "."))
        record[f"pinball_{column}"] = float(
            pinball_loss(predictions["y_true"], predictions[column], quantile)
        )

    lower = "q_0_1"
    upper = "q_0_9"
    if (
        lower in predictions.columns
        and upper in predictions.columns
        and predictions[lower].notna().any()
        and predictions[upper].notna().any()
    ):
        record["coverage_q_0_1_q_0_9"] = float(
            ((predictions["y_true"] >= predictions[lower]) & (predictions["y_true"] <= predictions[upper])).mean()
        )
    else:
        record["coverage_q_0_1_q_0_9"] = np.nan

    return record


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    diff = actual - predicted
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))
