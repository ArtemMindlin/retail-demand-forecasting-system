from __future__ import annotations

import numpy as np
import pandas as pd


def label_stockout_regime(frame: pd.DataFrame, threshold: float | None = 1.0) -> pd.DataFrame:
    """Label rows by stockout regime using a threshold on stockout hours."""

    labeled = frame.copy()
    threshold_value = (
        threshold if threshold is not None else float(labeled["stockout_hours"].median())
    )
    labeled["stockout_regime"] = np.where(
        labeled["stockout_hours"] > threshold_value,
        "high_stockout",
        "low_stockout",
    )
    return labeled


def label_demand_velocity_regime(
    frame: pd.DataFrame,
    threshold: float = 1.0,
    series_means: pd.Series | None = None,
) -> pd.DataFrame:
    """Label series by demand velocity using a threshold on average historical sales.

    The mean is the one aggregate in this module, so it is the only label whose value
    depends on which rows the frame happens to carry. Pass ``series_means`` to pin it to a
    reference universe and make the labeling concatenable: without it, labeling two halves
    separately and labeling their concatenation disagree on the same rows. Series absent
    from ``series_means`` fall back to their mean within ``frame``.
    """

    labeled = frame.copy()
    means = labeled.groupby("series_id", sort=False)["observed_demand"].mean()
    if series_means is not None:
        means = series_means.combine_first(means)
    labeled["velocity_regime"] = np.where(
        labeled["series_id"].map(means) >= threshold,
        "fast_moving",
        "slow_moving",
    )
    return labeled


def label_promotion_regime(frame: pd.DataFrame, discount_col: str = "discount") -> pd.DataFrame:
    """Label rows by whether they represent an active discount or promotional event."""

    labeled = frame.copy()
    discount_val = labeled[discount_col].fillna(0.0)
    labeled["promo_regime"] = np.where(
        discount_val > 0.0,
        "on_promotion",
        "baseline_price",
    )
    return labeled


def label_seasonal_regime(frame: pd.DataFrame, holiday_col: str = "holiday_flag") -> pd.DataFrame:
    """Label rows by operational seasonal periods."""

    labeled = frame.copy()
    holiday_val = labeled[holiday_col].fillna(0.0)
    labeled["seasonal_regime"] = np.where(
        holiday_val > 0.0,
        "peak_holiday",
        "standard_season",
    )
    return labeled


def label_all_regimes(
    frame: pd.DataFrame,
    velocity_threshold: float = 1.0,
    stockout_threshold: float | None = 1.0,
    velocity_series_means: pd.Series | None = None,
) -> pd.DataFrame:
    """Helper function to apply all operational regime labelers to a frame sequentially.

    With ``velocity_series_means`` supplied and ``stockout_threshold`` not None, every label
    is a row-wise function of the frame, so labeling is concatenable.
    """

    labeled = label_stockout_regime(frame, threshold=stockout_threshold)
    labeled = label_demand_velocity_regime(
        labeled, threshold=velocity_threshold, series_means=velocity_series_means
    )
    labeled = label_promotion_regime(labeled)
    labeled = label_seasonal_regime(labeled)
    return labeled
