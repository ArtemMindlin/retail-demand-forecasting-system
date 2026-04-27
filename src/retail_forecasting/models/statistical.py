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
        """Predict using per-series ARIMA models, sensitive to the row date."""
        unique_series = frame["series_id"].unique()
        print(f"Fitting ARIMA for {len(unique_series)} series (dynamic mode)...")
        
        # Cache models per series to avoid refitting if multiple dates exist for the same series
        # but the history hasn't changed much (simplified for TFG performance).
        series_models = {}
        
        final_preds = []
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="ARIMA rows"):
            series_id = row.series_id
            target_date = row.date
            
            # Get historical data strictly before the row date
            series_history = self.history_[
                (self.history_["series_id"] == series_id) & 
                (self.history_["date"] < target_date)
            ].sort_values("date")
            
            y_train = series_history["observed_demand"].values
            
            if len(y_train) < 14: # Need minimum data for ARIMA
                final_preds.append(0.0)
                continue
                
            try:
                # If we don't have a model for this series, or to be simple, 
                # fit a quick model. Stepwise=True and simple p,q.
                model = pm.auto_arima(
                    y_train, 
                    seasonal=True, 
                    m=self.seasonal_period,
                    max_p=2, max_q=2, d=1,
                    stepwise=True, 
                    suppress_warnings=True, 
                    error_action='ignore'
                )
                
                forecast = model.predict(n_periods=self.horizon)
                final_preds.append(np.maximum(np.sum(forecast), 0.0))
            except Exception:
                # Fallback to sum of recent mean
                final_preds.append(np.sum([np.mean(y_train[-7:])] * self.horizon))
                
        return np.asarray(final_preds, dtype=float)
