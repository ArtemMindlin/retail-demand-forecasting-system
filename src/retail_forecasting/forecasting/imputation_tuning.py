from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_tuning import BoostingParams, ImputationTuningMetadata
from retail_forecasting.data.censorship import (
    DEFAULT_SUPERVISED_LGBM_PARAMS,
    IMPUTATION_LGBM_PARAMS_FILENAME,
    LatentDemandImputer,
)
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.evaluation.reporting import get_git_commit, utc_timestamp
from retail_forecasting.forecasting.pipeline import _synthetic_censor_holdout

# A single synthetic-censoring draw scores ~650 rows, and the trial-to-trial MAE spread on one
# draw (12-40%) dwarfs the ~1% differences between hyperparameter sets. Averaging the objective
# over several independent draws shrinks that selection noise by ~sqrt(N).
N_SELECTION_HOLDOUTS = 5

# Holdouts the search never sees, used only to report whether the winner actually beats the
# untuned defaults. Scoring the winner on the draws that chose it is what made an earlier run
# report a 1.4% gain that did not replicate.
N_VALIDATION_HOLDOUTS = 5

# Offset keeping validation seeds disjoint from selection seeds derived from the same base seed.
_VALIDATION_SEED_OFFSET = 10_000

Holdout = tuple[pd.DataFrame, np.ndarray, np.ndarray]


def _build_holdouts(panel: pd.DataFrame, seeds: list[int]) -> list[Holdout]:
    """Draw one synthetic-censoring holdout per seed.

    Built once and reused by every trial, so all candidates are scored on identical data and
    differences between them are hyperparameters rather than resampling luck.
    """
    holdouts = []
    for seed in seeds:
        censored, eval_idx, true_demand = _synthetic_censor_holdout(panel, seed=seed)
        if len(eval_idx) == 0:
            raise ValueError(
                "Cannot tune the imputer: the train panel has no clean/censored rows to build "
                "a synthetic evaluation set from."
            )
        holdouts.append((censored, eval_idx, true_demand))
    return holdouts


def _mean_mae(holdouts: list[Holdout], params: dict[str, int | float]) -> float:
    """Mean reconstruction MAE of one hyperparameter set across every holdout."""
    maes = []
    for censored, eval_idx, true_demand in holdouts:
        imputed = LatentDemandImputer(strategy="supervised", lgbm_params=params).impute(censored)
        pred = imputed.loc[eval_idx, "latent_demand_est"].astype(float).to_numpy()
        maes.append(float(np.mean(np.abs(pred - true_demand))))
    return float(np.mean(maes))


def tune_imputation_lgbm(
    settings: Settings,
    n_trials: int | None = None,
    seed: int | None = None,
    n_selection_holdouts: int = N_SELECTION_HOLDOUTS,
    n_validation_holdouts: int = N_VALIDATION_HOLDOUTS,
) -> Path:
    """Search LGBM hyperparameters for the supervised imputer and persist the winner.

    Uses only the ``train`` split (never ``eval``, per the eval-holdout invariant in
    docs/invariants.md) and scores each trial with the same synthetic-censoring
    reconstruction error used by ``_evaluate_imputation_quality``: held-out clean days are
    synthetically censored so the true demand is known, and each trial's MAE against that
    ground truth is what Optuna minimizes.

    The objective averages that MAE over ``n_selection_holdouts`` independent draws, and the
    winner is then re-scored against the untuned defaults on ``n_validation_holdouts`` further
    draws the search never saw. Both guard the same failure: on a single draw the noise between
    trials is an order of magnitude larger than the difference between hyperparameter sets, so
    a single-draw search selects the lucky trial and reports its in-sample score as a gain.

    The winning params are persisted ONLY if they beat the defaults on the validation draws --
    otherwise the pipeline would silently switch to hyperparameters chosen by noise. The
    metadata file is written either way, recording both scores and the decision.

    Only the winning hyperparameters are persisted (not fitted weights), so
    ``LatentDemandImputer`` still re-fits on each panel's own clean days -- this run only
    changes which 3 numbers that fit uses.

    Returns:
        The path of the written ``imputation_lgbm_params.json`` when the winner beat the
        defaults, otherwise the path of the metadata file recording why it did not.
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

    selection_seeds = [seed + i for i in range(n_selection_holdouts)]
    validation_seeds = [seed + _VALIDATION_SEED_OFFSET + i for i in range(n_validation_holdouts)]
    selection = _build_holdouts(panel, selection_seeds)
    validation = _build_holdouts(panel, validation_seeds)
    print(
        f"🎲 {len(selection)} selection holdouts (seeds {selection_seeds}), "
        f"{len(validation)} validation holdouts (seeds {validation_seeds})"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
        return _mean_mae(selection, params)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_params = BoostingParams(
        n_estimators=int(study.best_params["n_estimators"]),
        learning_rate=float(study.best_params["learning_rate"]),
        max_depth=int(study.best_params["max_depth"]),
    )
    print(
        f"✅ Best trial: selection MAE={study.best_value:.4f} "
        f"(n_estimators={best_params.n_estimators}, "
        f"learning_rate={best_params.learning_rate:.4f}, max_depth={best_params.max_depth})"
    )

    print("🧪 Scoring the winner and the untuned defaults on the validation holdouts...")
    best_mae_validation = _mean_mae(validation, best_params.model_dump())
    default_mae_validation = _mean_mae(validation, dict(DEFAULT_SUPERVISED_LGBM_PARAMS))
    improvement_pct = float(
        (best_mae_validation - default_mae_validation) / default_mae_validation * 100
    )
    beats_default = best_mae_validation < default_mae_validation

    metadata = ImputationTuningMetadata(
        n_trials_requested=n_trials,
        best_mae_selection=float(study.best_value),
        best_mae_validation=best_mae_validation,
        default_mae_validation=default_mae_validation,
        improvement_pct=improvement_pct,
        persisted=beats_default,
        n_selection_holdouts=len(selection),
        n_validation_holdouts=len(validation),
        selection_seeds=selection_seeds,
        validation_seeds=validation_seeds,
        train_rows=int(len(selection[0][0]) - len(selection[0][1])),
        eval_rows=int(len(selection[0][1])),
        seed=seed,
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
        best_params=best_params,
    )

    models_dir = settings.models.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = models_dir / "imputation_lgbm_tuning_metadata.json"
    metadata_path.write_text(json.dumps(metadata.model_dump(), indent=2), encoding="utf-8")

    print(
        f"\n📏 Validation: tuned={best_mae_validation:.4f} vs "
        f"default={default_mae_validation:.4f} ({improvement_pct:+.2f}%)"
    )
    if not beats_default:
        print(
            "⚠️  The tuned hyperparameters do NOT beat the untuned defaults out of sample, so "
            "they were NOT persisted -- the pipeline keeps using the defaults.\n"
            f"    Decision recorded in: {metadata_path}\n"
        )
        return metadata_path

    params_path = models_dir / IMPUTATION_LGBM_PARAMS_FILENAME
    params_path.write_text(json.dumps(best_params.model_dump(), indent=2), encoding="utf-8")
    print(f"\n✅ Tuned imputation hyperparameters written to: {params_path}\n")
    return params_path
