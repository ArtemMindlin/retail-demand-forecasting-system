from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import pmdarima as pm
from tqdm import tqdm
from joblib import Parallel, delayed


def _fit_predict_single_series(series_id, history, min_date, seasonal_period, horizon):
    """Worker function for parallel ARIMA fitting."""
    series_history = history[
        (history["series_id"] == series_id) & 
        (history["date"] < min_date)
    ].sort_values("date")
    
    y_train = series_history["observed_demand"].values
    
    # Require at least 30 days for a decent statistical fit
    if len(y_train) < 30:
        return series_id, float(np.sum([np.mean(y_train[-7:])] * horizon)) if len(y_train) > 0 else (series_id, 0.0)
        
    try:
        # Fit a simple ARIMA model
        model = pm.auto_arima(
            y_train, 
            seasonal=True, 
            m=seasonal_period,
            max_p=1, max_q=1, d=1, 
            stepwise=True, 
            suppress_warnings=True, 
            error_action='ignore'
        )
        
        forecast = model.predict(n_periods=horizon)
        return series_id, float(np.maximum(np.sum(forecast), 0.0))
    except Exception:
        return series_id, float(np.sum([np.mean(y_train[-7:])] * horizon))


@dataclass
class AutoArimaModel:
    """A local statistical baseline using Auto-ARIMA.
    
    This model fits an independent ARIMA model for each series in parallel.
    """
    seasonal_period: int
    horizon: int
    n_jobs: int = -1  # Use all cores on M5
    model_name: str = "auto_arima"
    backend_name: str = field(init=False, default="pmdarima_stepwise")

    def fit(self, panel: pd.DataFrame) -> "AutoArimaModel":
        self.history_ = panel.copy()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict using per-series ARIMA models in parallel."""
        unique_series = frame["series_id"].unique()
        min_date = frame["date"].min()
        
        # Parallel execution
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict_single_series)(
                sid, self.history_, min_date, self.seasonal_period, self.horizon
            ) 
            for sid in tqdm(unique_series, desc="ARIMA parallel fitting")
        )
        
        forecast_cache = dict(results)
        final_preds = [forecast_cache.get(sid, 0.0) for sid in frame["series_id"]]
        return np.asarray(final_preds, dtype=float)
