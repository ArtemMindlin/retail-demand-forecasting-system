from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retail_forecasting.drift.psi import (
    PSI_CRITICAL_THRESHOLD,
    build_feature_drift_report,
    compute_psi,
    psi_status,
)


def test_compute_psi_zero_for_identical_distributions() -> None:
    rng = np.random.RandomState(0)
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(0, 1, 5000)
    psi, pre, post = compute_psi(reference, current)
    assert psi < PSI_CRITICAL_THRESHOLD
    assert psi_status(psi) == "ok"
    assert len(pre) == len(post)


def test_compute_psi_large_for_shifted_distribution() -> None:
    rng = np.random.RandomState(0)
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(2.0, 1, 5000)
    psi, _, _ = compute_psi(reference, current)
    assert psi >= PSI_CRITICAL_THRESHOLD
    assert psi_status(psi) == "critical"


def test_compute_psi_handles_empty_and_constant() -> None:
    assert compute_psi(np.array([]), np.array([1.0, 2.0]))[0] == 0.0
    psi, _, _ = compute_psi(np.ones(100), np.ones(100))
    assert psi == 0.0


@dataclass
class _FakeExplanation:
    values: np.ndarray
    feature_names: list[str]


def test_build_feature_drift_report_ranks_by_shap_and_detects_drift() -> None:
    n = 400
    half = n // 2
    rng = np.random.RandomState(0)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    # feat_drift shifts in the recent half; feat_stable keeps the same law.
    feat_drift = np.concatenate([rng.normal(0, 1, half), rng.normal(4, 1, n - half)])
    feat_stable = np.concatenate([rng.normal(0, 1, half), rng.normal(0, 1, n - half)])
    frame = pd.DataFrame(
        {
            "date": dates,
            "feat_drift": feat_drift,
            "feat_stable": feat_stable,
        }
    )
    # feat_drift is the most important feature for the model.
    shap_vals = np.column_stack([np.full(n, 3.0), np.full(n, 0.5)])
    explanation = _FakeExplanation(values=shap_vals, feature_names=["feat_drift", "feat_stable"])

    report = build_feature_drift_report(frame, explanation, top_n=2)
    assert len(report) == 2
    # Sorted by PSI desc; the drifting feature must lead and be flagged critical.
    assert report[0]["name"] == "feat_drift"
    assert report[0]["status"] == "critical"
    assert report[0]["psi"] > report[1]["psi"]
    # Importance is a normalized share in (0, 1].
    assert 0.0 < report[0]["importance"] <= 1.0


def test_build_feature_drift_report_empty_without_shap() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "x": [1, 2, 3]})
    assert build_feature_drift_report(frame, None) == []
