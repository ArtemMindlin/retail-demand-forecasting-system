from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_tuning import BoostingParams, ImputationTuningMetadata
from retail_forecasting.data.censorship import (
    IMPUTATION_LGBM_PARAMS_FILENAME,
    LatentDemandImputer,
)
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.evaluation.reporting import get_git_commit, utc_timestamp
from retail_forecasting.forecasting.pipeline import _synthetic_censor_holdout


def tune_imputation_lgbm(
    settings: Settings, n_trials: int | None = None, seed: int | None = None
) -> Path:
    """Search LGBM hyperparameters for the supervised imputer and persist the winner.

    Uses only the ``train`` split (never ``eval``, per the eval-holdout invariant in
    docs/invariants.md) and scores each trial with the same synthetic-censoring
    reconstruction error used by ``_evaluate_imputation_quality``: held-out clean days are
    synthetically censored so the true demand is known, and each trial's MAE against that
    ground truth is what Optuna minimizes.

    Only the winning hyperparameters are persisted (not fitted weights), so
    ``LatentDemandImputer`` still re-fits on each panel's own clean days -- this run only
    changes which 3 numbers that fit uses.

    Returns:
        The path of the written ``imputation_lgbm_params.json``.
    """
    n_trials = n_trials if n_trials is not None else settings.models.tuning_trials
    seed = seed if seed is not None else settings.project.random_seed

    print("\n" + "=" * 50)
    print("🔍 IMPUTATION LGBM TUNING (supervised strategy, no forecasting)")
    print("=" * 50 + "\n")
    print("📂 Loading train panel...")
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )

    censored, eval_idx, true_demand = _synthetic_censor_holdout(panel, seed=seed)
    if len(eval_idx) == 0:
        raise ValueError(
            "Cannot tune the imputer: the train panel has no clean/censored rows to build "
            "a synthetic evaluation set from."
        )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
        imputed = LatentDemandImputer(strategy="supervised", lgbm_params=params).impute(censored)
        pred = imputed.loc[eval_idx, "latent_demand_est"].astype(float).to_numpy()
        return float(np.mean(np.abs(pred - true_demand)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = BoostingParams(
        n_estimators=int(study.best_params["n_estimators"]),
        learning_rate=float(study.best_params["learning_rate"]),
        max_depth=int(study.best_params["max_depth"]),
    )
    print(
        f"✅ Best trial: MAE={study.best_value:.4f} "
        f"(n_estimators={best_params.n_estimators}, "
        f"learning_rate={best_params.learning_rate:.4f}, max_depth={best_params.max_depth})"
    )

    metadata = ImputationTuningMetadata(
        n_trials_requested=n_trials,
        best_mae=float(study.best_value),
        train_rows=int(len(censored) - len(eval_idx)),
        eval_rows=int(len(eval_idx)),
        seed=seed,
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
        best_params=best_params,
    )

    models_dir = settings.models.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    params_path = models_dir / IMPUTATION_LGBM_PARAMS_FILENAME
    params_path.write_text(json.dumps(best_params.model_dump(), indent=2), encoding="utf-8")
    metadata_path = models_dir / "imputation_lgbm_tuning_metadata.json"
    metadata_path.write_text(json.dumps(metadata.model_dump(), indent=2), encoding="utf-8")

    print(f"\n✅ Tuned imputation hyperparameters written to: {params_path}\n")
    return params_path
