from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# A SHAP run that has not finished by now is not going to: the explainer is quadratic in
# the sample and this step is optional reporting, not a result.
_TIMEOUT_SECONDS = 600


def _explain(model: Any, X_sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values. Runs in the CHILD process; see `calculate_shap_values`.

    Returns plain arrays rather than a `shap.Explanation` so what crosses the process
    boundary is numpy and nothing else.
    """
    booster = model
    if hasattr(model, "point_model_"):
        booster = model.point_model_
    elif hasattr(model, "base_model") and hasattr(model.base_model, "point_model_"):
        booster = model.base_model.point_model_

    try:
        explanation = shap.TreeExplainer(booster)(X_sample)
    except Exception:
        # Fallback to the universal (and much slower) explainer, on a smaller sample.
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 50))
        explanation = explainer(X_sample)

    return np.asarray(explanation.values), np.asarray(explanation.base_values)


def calculate_shap_values(
    model: Any, X: pd.DataFrame, sample_size: int = 500
) -> shap.Explanation | None:
    """SHAP values for a fitted model, computed in a SEPARATE PROCESS.

    The isolation is not tidiness. `shap.TreeExplainer` on the LightGBM champion reaches
    LightGBM's `pred_contrib`, which is native code, and segfaults inside the full pipeline
    process. A SIGSEGV cannot be caught, so the crash took down runs that had already spent
    minutes training every fold, for the sake of an optional figure. Run in a child, the
    same crash surfaces as `BrokenProcessPool` in the parent, which is catchable: the run
    keeps its models, metrics and costs, and loses only the explainability plot.

    Returns ``None`` when the child dies or times out. Callers already treat a missing
    explanation as "skip the figure", so there is nothing further to handle.
    """
    X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X

    # `spawn` starts the child from a clean interpreter: a `fork` would inherit the parent's
    # already-initialised native runtimes, which is the state the crash happens in.
    executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
    try:
        future = executor.submit(_explain, model, X_sample)
        values, base_values = future.result(timeout=_TIMEOUT_SECONDS)
    except Exception as error:  # BrokenProcessPool si el hijo muere; TimeoutError si cuelga
        logger.warning(
            "SHAP no pudo calcularse (%s: %s); la corrida continúa sin la figura.",
            type(error).__name__,
            error,
        )
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_sample.to_numpy(),
        feature_names=list(X_sample.columns),
    )
