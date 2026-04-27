from __future__ import annotations

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List


class SupervisedImputer:
    """Imputes latent demand for periods with stockouts using a teacher model.
    
    This imputer trains a simple model on non-stockout days and predicts 
    demand for stockout days, correcting the observed sales if they are 
    lower than the predicted demand.
    """
    
    def __init__(self, stockout_col: str = "stockout_hours", target_col: str = "observed_demand"):
        self.stockout_col = stockout_col
        self.target_col = target_col
        self.model = None
        
    def impute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Correct censored demand in the input panel.
        
        Args:
            panel: Daily panel containing date, series_id, observed_demand, and stockout_hours.
            
        Returns:
            A panel with a new column 'imputed_demand' and corrected 'observed_demand'.
        """
        df = panel.copy()
        
        # 1. Feature Engineering for the Teacher Model (Basic)
        # We use simple features that don't depend on lags to avoid circularity
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        
        # 2. Identify Clean vs Censored rows
        # A row is clean if stockout_hours is 0
        is_clean = df[self.stockout_col] == 0
        is_censored = ~is_clean
        
        if not is_censored.any():
            df["is_imputed"] = False
            return df
            
        # 3. Train the Teacher Model on Clean data
        # We use a global model approach
        train_df = df[is_clean]
        
        # Simple categorical encoding for series_id
        train_df["series_cat"] = train_df["series_id"].astype("category")
        X_train = train_df[["month", "day_of_week", "series_cat"]]
        y_train = train_df[self.target_col]
        
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        self.model.fit(X_train, y_train)
        
        # 4. Impute for Censored rows
        censored_df = df[is_censored].copy()
        censored_df["series_cat"] = censored_df["series_id"].astype("category")
        X_censored = censored_df[["month", "day_of_week", "series_cat"]]
        
        predicted_latent_demand = self.model.predict(X_censored)
        
        # 5. Apply correction
        # Rule: New Demand = max(Observed Sales, Predicted Latent Demand)
        # This ensures we don't reduce sales if the model under-predicts
        df.loc[is_censored, "latent_demand_est"] = np.maximum(
            df.loc[is_censored, self.target_col],
            predicted_latent_demand
        )
        # For clean rows, latent demand is just observed sales
        df.loc[is_clean, "latent_demand_est"] = df.loc[is_clean, self.target_col]
        
        df["is_imputed"] = is_censored
        
        # For the rest of the pipeline, we swap the target to the estimated demand
        # but keep a backup of the original
        df["original_observed_demand"] = df[self.target_col]
        df[self.target_col] = df["latent_demand_est"]
        
        return df
