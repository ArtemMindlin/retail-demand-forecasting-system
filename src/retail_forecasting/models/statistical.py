from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import pmdarima as pm
from tqdm import tqdm


@dataclass
class AutoArimaModel:
    """A local statistical baseline using Auto-ARIMA.
    
    This model fits an independent ARIMA model for each series. It automatically 
    selects the best (p, d, q) parameters using the AIC criterion.
    """
    seasonal_period: int
    horizon: int
    model_name: str = "auto_arima"
    backend_name: str = field(init=False, default="pmdarima_stepwise")

    def fit(self, panel: pd.DataFrame) -> "AutoArimaModel":
        # ARIMA is a local model, but for compliance with the pipeline, 
        # we store the history here and fit during 'predict' for the 
        # relevant validation window, or we can pre-fit. 
        # In retail_forecasting pipeline, it's cleaner to store history.
        self.history_ = panel.copy()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict using per-series ARIMA models, optimized to fit once per series-origin."""
        # Identification of the forecast origin: the date before the first date in the frame
        # In this TFG pipeline, each call to predict is usually for a single validation fold.
        # We assume all rows in 'frame' can share the same model fit per series for speed.
        
        unique_series = frame["series_id"].unique()
        min_date = frame["date"].min()
        print(f"Fitting ARIMA for {len(unique_series)} series (origin < {min_date.date()})...")
        
        # Cache of sum-forecasts for this validation window: {series_id: forecast_sum}
        forecast_cache = {}
        
        for series_id in tqdm(unique_series, desc="ARIMA fitting"):
            # Get historical data strictly before the validation window
            series_history = self.history_[
                (self.history_["series_id"] == series_id) & 
                (self.history_["date"] < min_date)
            ].sort_values("date")
            
            y_train = series_history["observed_demand"].values
            
            # Require at least 30 days for a decent statistical fit
            if len(y_train) < 30:
                forecast_cache[series_id] = np.sum([np.mean(y_train[-7:])] * self.horizon) if len(y_train) > 0 else 0.0
                continue
                
            try:
                # Fit a simple ARIMA model
                model = pm.auto_arima(
                    y_train, 
                    seasonal=True, 
                    m=self.seasonal_period,
                    max_p=1, max_q=1, d=1, # Reduced complexity for speed
                    stepwise=True, 
                    suppress_warnings=True, 
                    error_action='ignore'
                )
                
                forecast = model.predict(n_periods=self.horizon)
                forecast_cache[series_id] = np.maximum(np.sum(forecast), 0.0)
            except Exception:
                forecast_cache[series_id] = np.sum([np.mean(y_train[-7:])] * self.horizon)
                
        # Map cache back to frame rows
        final_preds = [forecast_cache.get(row.series_id, 0.0) for row in frame.itertuples(index=False)]
        return np.asarray(final_preds, dtype=float)
