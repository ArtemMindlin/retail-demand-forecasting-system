from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from retail_forecasting.config import Settings
from retail_forecasting.models.boosting import AutoBoostingModel


class BoostingParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_depth: int = Field(gt=0)


class TuningMetadata(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    strategy: Literal["optuna_temporal_holdout", "default_fallback"]
    n_trials_requested: int = Field(gt=0)
    best_score: float | None = Field(default=None, ge=0)
    train_rows: int = Field(ge=0)
    validation_rows: int = Field(ge=0)
    validation_cutoff: pd.Timestamp
    feature_count: int = Field(ge=0)
    target_col: str
    best_params: BoostingParams

    @field_validator("validation_cutoff", mode="before")
    @classmethod
    def _coerce_timestamp(cls, value: Any) -> pd.Timestamp:
        return pd.Timestamp(value)


class TuningResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    best_params: BoostingParams
    metadata: TuningMetadata


@dataclass(slots=True)
class HyperparameterTuner:
    """Orchestrates hyperparameter optimization using Optuna."""

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
        """Find best parameters for the Boosting model.

        Args:
            train_frame: Training data (will be split for internal validation).
            feature_columns: List of features to use.
            target_col: Target variable name.

        Returns:
            Best hyperparameters plus auditable tuning metadata.
        """
        # Internal simple temporal split for tuning
        # (e.g., use last 14 days of train_frame as validation for the tuner)
        max_date = train_frame["date"].max()
        val_cutoff = max_date - pd.Timedelta(days=14)

        t_train = train_frame[train_frame["date"] <= val_cutoff]
        t_val = train_frame[train_frame["date"] > val_cutoff]

        if t_train.empty or t_val.empty:
            # Fallback if train frame is too small
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
            return TuningResult(
                best_params=fallback_params,
                metadata=self.tuning_metadata,
            )

        def objective(trial: optuna.Trial) -> float:
            params = BoostingParams(
                n_estimators=trial.suggest_int("n_estimators", 50, 800),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.005, 0.2, log=True
                ),
                max_depth=trial.suggest_int("max_depth", 3, 12),
            )

            model = AutoBoostingModel(
                quantiles=self.settings.models.quantiles,
                random_seed=self.settings.project.random_seed,
                n_estimators=params.n_estimators,
                learning_rate=params.learning_rate,
                max_depth=params.max_depth,
            )

            model.fit(t_train.loc[:, feature_columns], t_train[target_col])
            preds = model.predict(t_val.loc[:, feature_columns])

            # Minimize MAE
            mae = np.abs(preds - t_val[target_col].values).mean()
            return float(mae)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=-1)

        best_params = BoostingParams(
            n_estimators=int(study.best_params["n_estimators"]),
            learning_rate=float(study.best_params["learning_rate"]),
            max_depth=int(study.best_params["max_depth"]),
        )
        self.best_params = best_params
        self.tuning_metadata = TuningMetadata(
            strategy="optuna_temporal_holdout",
            n_trials_requested=self.n_trials,
            best_score=float(study.best_value),
            train_rows=len(t_train),
            validation_rows=len(t_val),
            validation_cutoff=val_cutoff,
            feature_count=len(feature_columns),
            target_col=target_col,
            best_params=best_params,
        )
        print(f"✅ Optuna Optimization Finished. Best MAE: {study.best_value:.4f}")
        print(f"Best Params: {self.best_params}")

        return TuningResult(
            best_params=best_params,
            metadata=self.tuning_metadata,
        )
