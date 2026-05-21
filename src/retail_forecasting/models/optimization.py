from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger(__name__)

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_tuning import (
    BoostingParams,
    TuningMetadata,
    TuningResult,
)
from retail_forecasting.models.boosting import AutoBoostingModel
from retail_forecasting.utils.io import quantile_column_name


@dataclass(slots=True)
class HyperparameterTuner:
    """Orchestrates multi-objective hyperparameter optimization using Optuna."""

    settings: Settings
    n_trials: int = 20
    best_params: BoostingParams | None = field(default=None, init=False)
    tuning_metadata: TuningMetadata | None = field(default=None, init=False)

    def tune_boosting(
        self,
        train_frame: pd.DataFrame,
        feature_columns: list[str],
        target_col: str = "target_lead_time_demand",
    ) -> TuningResult:
        """Find best parameters using multi-objective optimization (MAE vs Winkler)."""
        max_date = train_frame["date"].max()
        val_cutoff = max_date - pd.Timedelta(days=14)

        t_train = train_frame[train_frame["date"] <= val_cutoff]
        t_val = train_frame[train_frame["date"] > val_cutoff]

        if t_train.empty or t_val.empty:
            fallback_params = BoostingParams(
                n_estimators=self.settings.models.n_estimators,
                learning_rate=self.settings.models.learning_rate,
                max_depth=self.settings.models.max_depth,
            )
            self.best_params = fallback_params
            self.tuning_metadata = TuningMetadata(
                strategy="default_fallback",
                n_trials_requested=self.n_trials,
                best_score=None,
                train_rows=len(t_train),
                validation_rows=len(t_val),
                validation_cutoff=val_cutoff,
                feature_count=len(feature_columns),
                target_col=target_col,
                best_params=fallback_params,
            )
            return TuningResult(best_params=fallback_params, metadata=self.tuning_metadata)

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            params = BoostingParams(
                n_estimators=trial.suggest_int("n_estimators", 50, 800),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 12),
            )
            # Introduce loss_function as a categorical hyperparameter
            loss_function = trial.suggest_categorical("loss_function", ["RMSE", "MAE"])

            model = AutoBoostingModel(
                quantiles=self.settings.models.quantiles,
                random_seed=self.settings.project.random_seed,
                n_estimators=params.n_estimators,
                learning_rate=params.learning_rate,
                max_depth=params.max_depth,
                loss_function=loss_function,  # Note: AutoBoostingModel must support this argument
            )

            model.fit(t_train.loc[:, feature_columns], t_train[target_col])

            # Predict point and quantiles
            preds = model.predict(t_val.loc[:, feature_columns])
            q_preds = model.predict_quantiles(t_val.loc[:, feature_columns])

            y_true = t_val[target_col].values

            # Objective 1: Predictive Error (MAE)
            mae = np.abs(preds - y_true).mean()

            # Objective 2: Interval Quality (Winkler Score)
            if len(self.settings.models.quantiles) >= 2:
                q_low = self.settings.models.quantiles[0]
                q_high = self.settings.models.quantiles[-1]
                alpha = q_low + (1.0 - q_high)

                low_col = quantile_column_name(q_low)
                high_col = quantile_column_name(q_high)

                if low_col in q_preds and high_col in q_preds:
                    y_low = q_preds[low_col]
                    y_high = q_preds[high_col]

                    width = y_high - y_low
                    under = (2.0 / alpha) * (y_low - y_true) * (y_true < y_low)
                    over = (2.0 / alpha) * (y_true - y_high) * (y_true > y_high)
                    winkler = (width + under + over).mean()
                else:
                    winkler = mae * 2.0  # Fallback penalty
            else:
                winkler = mae * 2.0

            return float(mae), float(winkler)

        # Multi-objective study: Minimize both MAE and Winkler
        study = optuna.create_study(directions=["minimize", "minimize"])

        # Apply stratified sampling for large datasets to keep tuning feasible
        MAX_TUNING_ROWS = 500_000
        if len(t_train) > MAX_TUNING_ROWS:
            logger.info(
                "Dataset too large for tuning (%s rows). Performing stratified sampling.",
                len(t_train),
            )
            series_count = t_train["series_id"].nunique()
            rows_per_series = MAX_TUNING_ROWS // series_count
            t_train = t_train.groupby("series_id", group_keys=False).apply(
                lambda x: x.tail(rows_per_series)
            )

        study.optimize(objective, n_trials=self.n_trials, n_jobs=-1)

        # Select the best trial from the Pareto front
        # We can pick the one that minimizes the sum of normalized objectives,
        # or simply the one with the lowest MAE on the front for simplicity,
        # but storing the fact it was a multi-objective search.
        best_trial = min(study.best_trials, key=lambda t: t.values[0] + (t.values[1] * 0.1))

        best_params = BoostingParams(
            n_estimators=int(best_trial.params["n_estimators"]),
            learning_rate=float(best_trial.params["learning_rate"]),
            max_depth=int(best_trial.params["max_depth"]),
        )
        self.best_params = best_params

        self.tuning_metadata = TuningMetadata(
            strategy="optuna_multiobjective_pareto",
            n_trials_requested=self.n_trials,
            best_score=float(best_trial.values[0]),
            train_rows=len(t_train),
            validation_rows=len(t_val),
            validation_cutoff=val_cutoff,
            feature_count=len(feature_columns),
            target_col=target_col,
            best_params=best_params,
        )
        print(
            f"✅ Optuna Multi-Objective Finished. Best Selected MAE: {best_trial.values[0]:.4f}, Winkler: {best_trial.values[1]:.4f}"
        )

        return TuningResult(best_params=best_params, metadata=self.tuning_metadata)
