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
        """Predict using per-series ARIMA models."""
        predictions_map = {}
        unique_series = frame["series_id"].unique()
        
        print(f"Fitting ARIMA for {len(unique_series)} series...")
        
        for series_id in tqdm(unique_series, desc="ARIMA Progress"):
            # Get historical data for this series
            series_history = self.history_[self.history_["series_id"] == series_id].sort_values("date")
            y_train = series_history["observed_demand"].values
            
            try:
                # Fit Auto-ARIMA
                # We disable seasonality for speed if the window is short, 
                # but keep it if requested via seasonal_period
                model = pm.auto_arima(
                    y_train, 
                    seasonal=True, 
                    m=self.seasonal_period,
                    stepwise=True, 
                    suppress_warnings=True, 
                    error_action='ignore', 
                    max_p=3, max_q=3
                )
                
                # Forecast
                forecast = model.predict(n_periods=self.horizon)
                predictions_map[series_id] = np.maximum(forecast, 0.0)
            except Exception:
                # Fallback to mean if ARIMA fails
                predictions_map[series_id] = np.array([np.mean(y_train)] * self.horizon)

        # Map predictions back to the frame structure
        # Since our pipeline expects 1 prediction per row (multi-step is handled 
        # as a single target_lead_time_demand), ARIMA needs to sum its forecast horizon.
        
        final_preds = []
        for row in frame.itertuples(index=False):
            series_forecast = predictions_map.get(row.series_id, [0.0])
            # The target in this TFG is the SUM of the next 'horizon' days
            final_preds.append(np.sum(series_forecast))
            
        return np.asarray(final_preds, dtype=float)
