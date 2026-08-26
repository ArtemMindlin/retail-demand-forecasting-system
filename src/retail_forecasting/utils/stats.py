"""Statistics shared by the modes that score candidates on repeated random draws."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci95(deltas: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% CI for the mean of the PAIRED per-draw differences.

    Student-t, not the normal's 1.96: at 25 draws the correct multiplier is
    ``t(0.975, 24) = 2.064``, and 1.96 would quote a ~93% interval as 95% -- permissive in
    the one direction a gate must not be.

    Lives in `utils` so the three consumers can share it: both Optuna searches and the
    fair-cost backtest. It was born inside `imputation_tuning`, and importing it from there
    would have made the backtest depend on optuna and torch -- torch being an optional extra
    the backtest has no use for.

    ``simulation/operations.py`` keeps its own bootstrap on purpose: that one resamples whole
    ORIGINS as clusters, which no closed form covers.
    """
    n = len(deltas)
    if n < 2:
        raise ValueError(f"A 95% CI for the mean needs at least 2 draws, got {n}.")
    mean = float(np.mean(deltas))
    half_width = float(stats.t.ppf(0.975, n - 1) * np.std(deltas, ddof=1) / np.sqrt(n))
    return mean - half_width, mean + half_width
