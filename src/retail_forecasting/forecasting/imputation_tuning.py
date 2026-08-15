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
    _synthetic_censor_holdout,
)
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.evaluation.reporting import get_git_commit, utc_timestamp

# Number of selection holdouts used to average trial MAE and shrink selection noise.
N_SELECTION_HOLDOUTS = 5

# Validation holdouts unseen by Optuna used for statistical decision against defaults.
N_VALIDATION_HOLDOUTS = 10

# Offset ensuring validation seeds remain disjoint from selection seeds.
_VALIDATION_SEED_OFFSET = 10_000

_BOOTSTRAP_RESAMPLES = 10_000

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


def _holdout_maes(holdouts: list[Holdout], params: dict[str, int | float]) -> np.ndarray:
    """Reconstruction MAE of one hyperparameter set on each holdout, one value per draw."""
    maes = []
    for censored, eval_idx, true_demand in holdouts:
        imputed = LatentDemandImputer(strategy="supervised", lgbm_params=params).impute(censored)
        pred = imputed.loc[eval_idx, "latent_demand_est"].astype(float).to_numpy()
        maes.append(float(np.mean(np.abs(pred - true_demand))))
    return np.asarray(maes, dtype=float)


def _mean_mae(holdouts: list[Holdout], params: dict[str, int | float]) -> float:
    """Mean reconstruction MAE of one hyperparameter set across every holdout."""
    return float(np.mean(_holdout_maes(holdouts, params)))


def _bootstrap_ci95(deltas: np.ndarray, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of the paired per-holdout MAE differences."""
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(_BOOTSTRAP_RESAMPLES, len(deltas)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tune_imputation_lgbm(
    settings: Settings,
    n_trials: int | None = None,
    seed: int | None = None,
    n_selection_holdouts: int = N_SELECTION_HOLDOUTS,
    n_validation_holdouts: int = N_VALIDATION_HOLDOUTS,
) -> Path:
    """Search LGBM hyperparameters for the supervised imputer and persist the winner.

    The objective averages that MAE over ``n_selection_holdouts`` independent draws, and the
    winner is then re-scored against the untuned defaults on ``n_validation_holdouts`` further
    draws the search never saw. Both guard the same failure: on a single draw the noise between
    trials is an order of magnitude larger than the difference between hyperparameter sets, so
    a single-draw search selects the lucky trial and reports its in-sample score as a gain.

    The winning params are persisted ONLY if the bootstrap CI95 of their per-draw improvement
    over the defaults lies entirely below zero -- a mean that merely happens to be negative is
    what a coin flip looks like, and persisting it would silently switch the pipeline to
    hyperparameters chosen by noise. The metadata file is written either way, recording both
    scores, the interval and the decision.

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
    tuned_maes = _holdout_maes(validation, best_params.model_dump())
    default_maes = _holdout_maes(validation, dict(DEFAULT_SUPERVISED_LGBM_PARAMS))
    best_mae_validation = float(np.mean(tuned_maes))
    default_mae_validation = float(np.mean(default_maes))
    improvement_pct = float(
        (best_mae_validation - default_mae_validation) / default_mae_validation * 100
    )

    # Decide on the interval, not the point estimate. A mean that happens to land below zero
    # is exactly what a coin-flip result looks like: one winner cleared this gate at -1.14%
    # and then measured -0.45% with a CI straddling zero on fresh draws.
    ci_lo, ci_hi = _bootstrap_ci95(tuned_maes - default_maes, seed=seed)
    beats_default = ci_hi < 0.0

    metadata = ImputationTuningMetadata(
        n_trials_requested=n_trials,
        best_mae_selection=float(study.best_value),
        best_mae_validation=best_mae_validation,
        default_mae_validation=default_mae_validation,
        improvement_pct=improvement_pct,
        improvement_ci95=[ci_lo, ci_hi],
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
        f"default={default_mae_validation:.4f} ({improvement_pct:+.2f}%), "
        f"CI95 of the difference [{ci_lo:+.4f}, {ci_hi:+.4f}]"
    )
    params_path = models_dir / IMPUTATION_LGBM_PARAMS_FILENAME
    if not beats_default:
        print(
            "⚠️  The tuned hyperparameters do NOT beat the untuned defaults by a margin the "
            "validation draws can distinguish from zero, so they were NOT persisted -- the "
            "pipeline keeps using the defaults.\n"
            f"    Decision recorded in: {metadata_path}\n"
        )
        # A params file from an earlier run would otherwise survive a search that just
        # failed revalidation, and the pipeline would go on using a winner this run rejected.
        if params_path.exists():
            params_path.unlink()
            print(f"🧹 Removed the superseded params file: {params_path}\n")
        return metadata_path

    params_path.write_text(json.dumps(best_params.model_dump(), indent=2), encoding="utf-8")
    print(f"\n✅ Tuned imputation hyperparameters written to: {params_path}\n")
    return params_path
