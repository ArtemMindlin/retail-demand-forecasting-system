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


def label_demand_velocity_regime(frame: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Label series by demand velocity using a threshold on average historical sales."""

    labeled = frame.copy()
    mean_demand = labeled.groupby("series_id", sort=False)["observed_demand"].transform("mean")
    labeled["velocity_regime"] = np.where(
        mean_demand >= threshold,
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
) -> pd.DataFrame:
    """Helper function to apply all operational regime labelers to a frame sequentially."""

    labeled = label_stockout_regime(frame, threshold=stockout_threshold)
    labeled = label_demand_velocity_regime(labeled, threshold=velocity_threshold)
    labeled = label_promotion_regime(labeled)
    labeled = label_seasonal_regime(labeled)
    return labeled
