from __future__ import annotations

import pandas as pd
import shap
from typing import Any


def calculate_shap_values(
    model: Any, X: pd.DataFrame, sample_size: int = 500
) -> shap.Explanation:
    """
    Calculate SHAP values for a given model and feature set.
    Uses TreeExplainer for boosting models if possible.
    """
    # Sample data if it's too large for performance
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    # Try to get the underlying booster if it's a wrapper
    booster = model
    if hasattr(model, "point_model_"):
        booster = model.point_model_
    elif hasattr(model, "base_model") and hasattr(model.base_model, "point_model_"):
        booster = model.base_model.point_model_

    # Create explainer
    try:
        # CatBoost, LightGBM, XGBoost often work better with TreeExplainer
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer(X_sample)
    except Exception:
        # Fallback to general KernelExplainer (much slower, but universal)
        # We use a smaller sample for KernelExplainer
        X_sample_kernel = shap.sample(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict, X_sample_kernel)
        shap_values = explainer(X_sample)

    return shap_values
