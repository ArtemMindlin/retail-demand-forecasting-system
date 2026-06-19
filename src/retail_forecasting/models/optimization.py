from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import optuna
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_tuning import (
    BoostingParams,
    ParetoTrial,
    TuningMetadata,
    TuningResult,
)
from retail_forecasting.models.boosting import AutoBoostingModel
from retail_forecasting.utils.io import quantile_column_name, winkler_score

logger = logging.getLogger(__name__)


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
        """Find best parameters using multi-objective optimization (Pinball vs Winkler)."""
        t_train, t_val, val_cutoff = self._temporal_split(train_frame)

        if t_train.empty or t_val.empty:
            return self._fallback_result(t_train, t_val, val_cutoff, feature_columns, target_col)

        t_train = self._subsample_if_large(t_train)
        objective = self._build_objective(t_train, t_val, feature_columns, target_col)

        # Multi-objective study: minimize both Pinball loss (at q*) and Winkler score
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(objective, n_trials=self.n_trials, n_jobs=-1)

        # Select the best trial from the Pareto front by the sum of the two objectives.
        best_trial = min(study.best_trials, key=lambda t: t.values[0] + t.values[1])
        pareto_front = self._collect_pareto_front(study, best_trial)

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
        logger.info(
            "Optuna multi-objective finished. Best selected Pinball(q*)=%.4f, Winkler=%.4f",
            best_trial.values[0],
            best_trial.values[1],
        )

        return TuningResult(
            best_params=best_params,
            metadata=self.tuning_metadata,
            pareto_front=pareto_front,
        )

    def _temporal_split(
        self, train_frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        """Hold out the most recent ~20% of days (clamped to [3, 14]) for internal validation."""
        max_date = train_frame["date"].max()
        min_date = train_frame["date"].min()
        available_days = (max_date - min_date).days
        val_days = max(3, min(14, int(available_days * 0.2)))
        val_cutoff = max_date - pd.Timedelta(days=val_days)
        t_train = train_frame[train_frame["date"] <= val_cutoff]
        t_val = train_frame[train_frame["date"] > val_cutoff]
        return t_train, t_val, val_cutoff

    def _fallback_result(
        self,
        t_train: pd.DataFrame,
        t_val: pd.DataFrame,
        val_cutoff: pd.Timestamp,
        feature_columns: list[str],
        target_col: str,
    ) -> TuningResult:
        """Return config defaults when the internal split leaves no train/val rows."""
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

    def _subsample_if_large(self, t_train: pd.DataFrame) -> pd.DataFrame:
        """Stratified tail-sampling per series to keep tuning feasible on large data."""
        max_tuning_rows = 500_000
        if len(t_train) <= max_tuning_rows:
            return t_train
        logger.info(
            "Dataset too large for tuning (%s rows). Performing stratified sampling.",
            len(t_train),
        )
        series_count = t_train["series_id"].nunique()
        rows_per_series = max_tuning_rows // series_count
        return t_train.groupby("series_id", group_keys=False).apply(
            lambda x: x.tail(rows_per_series)
        )

    def _build_objective(
        self,
        t_train: pd.DataFrame,
        t_val: pd.DataFrame,
        feature_columns: list[str],
        target_col: str,
    ) -> Callable[[optuna.Trial], tuple[float, float]]:
        """Build the Optuna objective returning (Pinball at q*, Winkler) for a trial."""

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            params = BoostingParams(
                n_estimators=trial.suggest_int("n_estimators", 50, 800),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 12),
            )

            model = AutoBoostingModel(
                quantiles=self.settings.models.quantiles,
                random_seed=self.settings.project.random_seed,
                n_estimators=params.n_estimators,
                learning_rate=params.learning_rate,
                max_depth=params.max_depth,
                overstock_cost=self.settings.inventory.overstock_cost,
                stockout_cost=self.settings.inventory.stockout_cost,
            )

            model.fit(t_train.loc[:, feature_columns], t_train[target_col])

            # Predict point and quantiles
            preds = model.predict(t_val.loc[:, feature_columns])
            q_preds = model.predict_quantiles(t_val.loc[:, feature_columns])

            y_true = t_val[target_col].to_numpy()

            # Objective 1: Pinball Loss at critical fractile q* (consistent with training)
            critical_fractile = self.settings.inventory.stockout_cost / (
                self.settings.inventory.stockout_cost + self.settings.inventory.overstock_cost
            )
            pinball = mean_pinball_loss(y_true, preds, alpha=critical_fractile)

            # Objective 2: Interval Quality (Winkler Score)
            # Fallback when no interval is available: penalize with twice the
            # point loss so trials without usable quantiles never look optimal.
            pinball_penalty = float(pinball) * 2.0
            if len(self.settings.models.quantiles) >= 2:
                q_low = self.settings.models.quantiles[0]
                q_high = self.settings.models.quantiles[-1]
                alpha = q_low + (1.0 - q_high)

                low_col = quantile_column_name(q_low)
                high_col = quantile_column_name(q_high)

                if low_col in q_preds and high_col in q_preds:
                    winkler = winkler_score(y_true, q_preds[low_col], q_preds[high_col], alpha)
                else:
                    winkler = pinball_penalty
            else:
                winkler = pinball_penalty

            return float(pinball), float(winkler)

        return objective

    def _collect_pareto_front(
        self, study: optuna.Study, best_trial: optuna.trial.FrozenTrial
    ) -> list[ParetoTrial]:
        """Capture every completed trial as a (Pinball, Winkler) point, flagging the
        Pareto-front members and the finally selected trial."""
        front_numbers = {trial.number for trial in study.best_trials}
        return [
            ParetoTrial(
                trial_number=trial.number,
                pinball=float(trial.values[0]),
                winkler=float(trial.values[1]),
                is_on_front=trial.number in front_numbers,
                is_selected=trial.number == best_trial.number,
                n_estimators=int(trial.params["n_estimators"])
                if "n_estimators" in trial.params
                else None,
                learning_rate=float(trial.params["learning_rate"])
                if "learning_rate" in trial.params
                else None,
                max_depth=int(trial.params["max_depth"]) if "max_depth" in trial.params else None,
            )
            for trial in study.trials
            if trial.values is not None
        ]
