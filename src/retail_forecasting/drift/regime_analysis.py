from __future__ import annotations

import numpy as np
import pandas as pd


def label_stockout_regime(frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    labeled = frame.copy()
    threshold_value = threshold if threshold is not None else float(labeled["stockout_hours"].median())
    labeled["stockout_regime"] = np.where(
        labeled["stockout_hours"] >= threshold_value,
        "high_stockout",
        "low_stockout",
    )
    return labeled
