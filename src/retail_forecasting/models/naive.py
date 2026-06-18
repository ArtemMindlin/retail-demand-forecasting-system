from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SeasonalNaiveModel:
    seasonal_period: int
    horizon: int
    model_name: str = "seasonal_naive"
    history_: Any = field(init=False, default=None)

    def fit(self, panel: pd.DataFrame) -> SeasonalNaiveModel:
        self.history_ = (
            panel.loc[:, ["series_id", "date", "observed_demand"]]
            .sort_values(["series_id", "date"])
            .groupby("series_id", sort=False)
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = []
        for row in frame.itertuples(index=False):
            # Seasonal naive: sum the observed demand at the most recent seasonal
            # lags covering the forecast horizon. Falls back to zero (via NaN) when
            # a series lacks enough history for a given lag.
            row_date = pd.Timestamp(row.date)
            series_history = self.history_.get_group(row.series_id)
            history = series_history.loc[series_history["date"] < row_date]
            history_by_date = history.set_index("date")["observed_demand"]
            prediction = 0.0
            for step in range(self.horizon):
                lag = math.ceil((step + 1) / self.seasonal_period) * self.seasonal_period - step
                reference_date = row_date - pd.Timedelta(days=lag)
                prediction += float(history_by_date.get(reference_date, np.nan))
            predictions.append(np.nan if math.isnan(prediction) else max(prediction, 0.0))
        return np.asarray(predictions, dtype=float)
