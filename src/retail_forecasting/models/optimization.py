from __future__ import annotations

import optuna
import pandas as pd
import numpy as np
from typing import Any, Dict
from retail_forecasting.config import Settings
from retail_forecasting.models.boosting import AutoBoostingModel


class HyperparameterTuner:
    """Orchestrates hyperparameter optimization using Optuna."""

    def __init__(self, settings: Settings, n_trials: int = 20):
        self.settings = settings
        self.n_trials = n_trials
        self.best_params: Dict[str, Any] = {}

    def tune_boosting(
        self,
        train_frame: pd.DataFrame,
        feature_columns: list[str],
        target_col: str = "target_lead_time_demand",
    ) -> Dict[str, Any]:
        """Find best parameters for the Boosting model.

        Args:
            train_frame: Training data (will be split for internal validation).
            feature_columns: List of features to use.
            target_col: Target variable name.

        Returns:
            Dictionary with the best hyperparameters found.
        """
        # Internal simple temporal split for tuning
        # (e.g., use last 14 days of train_frame as validation for the tuner)
        max_date = train_frame["date"].max()
        val_cutoff = max_date - pd.Timedelta(days=14)

        t_train = train_frame[train_frame["date"] <= val_cutoff]
        t_val = train_frame[train_frame["date"] > val_cutoff]

        if t_train.empty or t_val.empty:
            # Fallback if train frame is too small
            return {
                "n_estimators": self.settings.models.n_estimators,
                "learning_rate": self.settings.models.learning_rate,
                "max_depth": self.settings.models.max_depth,
            }

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 800),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }

            model = AutoBoostingModel(
                quantiles=self.settings.models.quantiles,
                random_seed=self.settings.project.random_seed,
                **params,
            )

            model.fit(t_train.loc[:, feature_columns], t_train[target_col])
            preds = model.predict(t_val.loc[:, feature_columns])

            # Minimize MAE
            mae = np.abs(preds - t_val[target_col].values).mean()
            return float(mae)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=-1)

        self.best_params = study.best_params
        print(f"✅ Optuna Optimization Finished. Best MAE: {study.best_value:.4f}")
        print(f"Best Params: {self.best_params}")

        return self.best_params
