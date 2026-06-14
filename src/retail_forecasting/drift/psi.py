from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Standard PSI interpretation thresholds (Population Stability Index).
PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.20


def psi_status(psi: float) -> str:
    """Classify a PSI value following the conventional 0.10 / 0.20 bands."""
    if psi >= PSI_CRITICAL_THRESHOLD:
        return "critical"
    if psi >= PSI_WARNING_THRESHOLD:
        return "warning"
    return "ok"


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> tuple[float, list[float], list[float]]:
    """Population Stability Index between a reference and a current sample.

    ``PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)`` over ``bins`` bins whose
    edges are the quantiles of the reference distribution. Bin proportions are
    floored with a small epsilon to avoid division by zero or ``log(0)`` on
    empty bins.

    Returns the PSI together with the normalized reference and current
    histograms over the shared bin edges, so the caller can plot the pre/post
    distributions.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        zeros = [0.0] * bins
        return 0.0, zeros, zeros

    # Quantile-based edges from the reference; collapse to unique values so a
    # low-cardinality feature does not produce empty degenerate bins.
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        # Constant reference: a single bin is all we can form.
        edges = np.array([reference.min() - 0.5, reference.min() + 0.5])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_prop = ref_counts / ref_counts.sum()
    cur_prop = cur_counts / cur_counts.sum()

    epsilon = 1e-4
    ref_safe = np.clip(ref_prop, epsilon, None)
    cur_safe = np.clip(cur_prop, epsilon, None)

    psi = float(np.sum((cur_safe - ref_safe) * np.log(cur_safe / ref_safe)))
    return psi, [round(v, 4) for v in ref_prop.tolist()], [round(v, 4) for v in cur_prop.tolist()]


def _temporal_split(frame: pd.DataFrame, date_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks splitting a frame into an older reference and recent half.

    Drift is measured "since training": the reference window is the older half
    of the observations (what the model was largely fitted on) and the current
    window is the most recent half. The cut is the median observation date,
    with an index-based fallback when dates are unusable.
    """
    n = len(frame)
    if date_column in frame.columns:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        if dates.notna().any():
            split = dates.quantile(0.5)
            ref_mask = (dates < split).to_numpy()
            cur_mask = ~ref_mask & dates.notna().to_numpy()
            if ref_mask.sum() > 0 and cur_mask.sum() > 0:
                return ref_mask, cur_mask
    half = n // 2
    idx = np.arange(n)
    return idx < half, idx >= half


def build_feature_drift_report(
    supervised_frame: pd.DataFrame,
    shap_values: Any,
    top_n: int = 7,
    bins: int = 10,
    date_column: str = "date",
) -> list[dict[str, Any]]:
    """Real per-feature PSI for the ``top_n`` most important model features.

    Feature importance is the mean absolute SHAP value of the explained model;
    the reference/current distributions come from a temporal split of the
    supervised frame. Returns a list of records ready to serialize as
    ``drift_report.json`` and serve from the API.
    """
    if shap_values is None or supervised_frame is None or supervised_frame.empty:
        return []

    feature_names = list(getattr(shap_values, "feature_names", []) or [])
    values = np.asarray(getattr(shap_values, "values", []), dtype=float)
    if not feature_names or values.size == 0:
        return []

    importance = np.abs(values).mean(axis=0)
    total = importance.sum()
    importance_share = importance / total if total > 0 else importance

    order = np.argsort(importance)[::-1]
    ranked = [
        (feature_names[i], float(importance_share[i]))
        for i in order
        if i < len(feature_names) and feature_names[i] in supervised_frame.columns
    ][:top_n]

    ref_mask, cur_mask = _temporal_split(supervised_frame, date_column)

    report: list[dict[str, Any]] = []
    for name, share in ranked:
        column = supervised_frame[name]
        ref = column.to_numpy()[ref_mask]
        cur = column.to_numpy()[cur_mask]
        psi, pre, post = compute_psi(ref, cur, bins=bins)
        ftype = "binary" if int(column.nunique(dropna=True)) <= 2 else "numeric"
        report.append(
            {
                "name": name,
                "type": ftype,
                "importance": round(share, 4),
                "psi": round(psi, 4),
                "status": psi_status(psi),
                "pre": pre,
                "post": post,
            }
        )
    report.sort(key=lambda item: item["psi"], reverse=True)
    return report
