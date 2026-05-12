from __future__ import annotations

import numpy as np
import pandas as pd


def label_stockout_regime(
    frame: pd.DataFrame, threshold: float | None = None
) -> pd.DataFrame:
    """Label rows by stockout regime using a threshold on stockout hours.

    Args:
        frame: Input frame containing a ``stockout_hours`` column.
        threshold: Optional threshold used to split low and high stockout
            regimes.

    Returns:
        A copy of the input frame with a ``stockout_regime`` column.

    Notes:
        When no threshold is provided, the median of ``stockout_hours`` is
        used.
    """
    labeled = frame.copy()
    threshold_value = (
        threshold
        if threshold is not None
        else float(labeled["stockout_hours"].median())
    )
    labeled["stockout_regime"] = np.where(
        labeled["stockout_hours"] >= threshold_value,
        "high_stockout",
        "low_stockout",
    )
    return labeled
