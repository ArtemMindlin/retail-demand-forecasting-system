from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_tuning import (
    ImputationBoostingParams,
    ImputationTuningMetadata,
)
from retail_forecasting.data.censorship import (
    DEFAULT_SUPERVISED_LGBM_PARAMS,
    IMPUTATION_LGBM_PARAMS_FILENAME,
    LatentDemandImputer,
    _synthetic_censor_holdout,
)
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.evaluation.reporting import get_git_commit, utc_timestamp

N_SELECTION_HOLDOUTS = 15

N_VALIDATION_HOLDOUTS = 25

# Offset ensuring validation seeds remain disjoint from selection seeds.
_VALIDATION_SEED_OFFSET = 10_000

_VALIDATION_SERIES_FRACTION = 0.30

_BOOTSTRAP_RESAMPLES = 10_000

# Random trials before the GP starts guiding the search. Kept below Optuna's default of 10 so a
# 40-trial budget does not spend a quarter of itself sampling at random.
_N_STARTUP_TRIALS = 5

# MLflow experiment collecting every imputation search, so runs stay comparable across time.
_MLFLOW_EXPERIMENT = "imputation_lgbm_tuning"

_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

Holdout = tuple[pd.DataFrame, np.ndarray, np.ndarray]


def _split_series_panels(
    panel: pd.DataFrame, seed: int, validation_fraction: float = _VALIDATION_SERIES_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition the panel by ``series_id`` into a selection panel and a validation panel.

    Series are sorted before shuffling so the split depends only on ``seed`` and the set of
    series present, not on the panel's row order. Both sides are guaranteed at least one
    series, which is what lets the synthetic-panel tests run this path with 3 series.

    Returns:
        ``(selection_panel, validation_panel)`` sharing no series and therefore no rows.
    """
    series = np.sort(panel["series_id"].unique())
    if len(series) < 2:
        raise ValueError(
            "Cannot split the imputation tuning panel by series: it holds "
            f"{len(series)} series, and a disjoint validation set needs at least 2."
        )
    shuffled = np.random.default_rng(seed).permutation(series)
    n_validation = min(len(series) - 1, max(1, round(len(series) * validation_fraction)))
    validation_series = set(shuffled[:n_validation])

    is_validation = panel["series_id"].isin(validation_series)
    return panel.loc[~is_validation].copy(), panel.loc[is_validation].copy()


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
        pred = imputed.loc[eval_idx, "latent_demand_est"].to_numpy(dtype=float)
        maes.append(np.mean(np.abs(pred - true_demand)))
    return np.asarray(maes, dtype=float)


def _bootstrap_ci95(deltas: np.ndarray, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of the paired per-holdout MAE differences."""
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(_BOOTSTRAP_RESAMPLES, len(deltas)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _configure_mlflow() -> None:
    """Point MLflow at its tracking store and select the imputation-tuning experiment.

    Imported locally, like torch elsewhere in this module, so importing this module does not
    require the ``ml`` extra -- the contract tests exercise the persist gate without it.
    """
    import mlflow

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)


def _log_study_to_mlflow(
    study: optuna.Study,
    metadata: ImputationTuningMetadata,
    metadata_path: Path,
    params_path: Path,
    beats_default: bool,
) -> None:
    """Record one completed search in MLflow: winner, validation verdict, and its artifacts.

    Called once, after the search and the persist decision are already final -- this only
    translates what ``tune_imputation_lgbm`` already computed and already wrote to disk, so a
    tracking failure here cannot cost the run its actual output.

    Also logs the per-trial convergence curve from ``study`` (persisted in step 1's Optuna
    storage), as two metric series sharing the trial number as their step: ``trial_mae``, the
    raw value the GP explored trial by trial, and ``trial_mae_best_so_far``, its running
    minimum -- the one that actually answers "is this still improving or has it stalled".

    Imported locally, like torch and in ``_configure_mlflow``, so importing this module does not
    require the ``ml`` extra.
    """
    import mlflow

    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(metadata.best_params.model_dump())
        mlflow.log_params(
            {
                "n_trials": metadata.n_trials_requested,
                "seed": metadata.seed,
                "n_selection_series": metadata.n_selection_series,
                "n_validation_series": metadata.n_validation_series,
            }
        )
        mlflow.log_metrics(
            {
                "mae_selection_best": metadata.best_mae_selection,
                "mae_validation_tuned": metadata.best_mae_validation,
                "mae_validation_default": metadata.default_mae_validation,
                "improvement_pct": metadata.improvement_pct,
                "improvement_ci95_low": metadata.improvement_ci95[0],
                "improvement_ci95_high": metadata.improvement_ci95[1],
            }
        )
        mlflow.set_tags(
            {
                "persisted": str(metadata.persisted),
                "git_commit": metadata.git_commit or "unknown",
            }
        )

        best_so_far = float("inf")
        for trial in study.trials:
            if trial.value is None:
                continue
            best_so_far = min(best_so_far, trial.value)
            mlflow.log_metric("trial_mae", trial.value, step=trial.number)
            mlflow.log_metric("trial_mae_best_so_far", best_so_far, step=trial.number)

        mlflow.log_artifact(str(metadata_path))
        if beats_default:
            mlflow.log_artifact(str(params_path))


def tune_imputation_lgbm(
    settings: Settings,
    n_trials: int | None = None,
    seed: int | None = None,
    n_selection_holdouts: int = N_SELECTION_HOLDOUTS,
    n_validation_holdouts: int = N_VALIDATION_HOLDOUTS,
) -> Path:
    """Search LGBM hyperparameters for the supervised imputer and persist the winner.

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

    selection_panel, validation_panel = _split_series_panels(panel, seed)
    n_selection_series = int(selection_panel["series_id"].nunique())
    n_validation_series = int(validation_panel["series_id"].nunique())

    selection_seeds = [seed + i for i in range(n_selection_holdouts)]
    validation_seeds = [seed + _VALIDATION_SEED_OFFSET + i for i in range(n_validation_holdouts)]
    selection = _build_holdouts(selection_panel, selection_seeds)
    validation = _build_holdouts(validation_panel, validation_seeds)
    print(
        f"🧬 Series split: {n_selection_series} for selection, {n_validation_series} for "
        f"validation (disjoint — no shared rows)"
    )
    print(
        f"🎲 {len(selection)} selection holdouts (seeds {selection_seeds}), "
        f"{len(validation)} validation holdouts (seeds {validation_seeds})"
    )

    def objective(trial: optuna.Trial) -> float:
        subsample = trial.suggest_float("subsample", 0.4, 1.0)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 8000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "num_leaves": trial.suggest_int("num_leaves", 4, 1024, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": subsample,
            "subsample_freq": 1 if subsample < 1.0 else 0,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "min_data_per_group": trial.suggest_int("min_data_per_group", 1, 100),
            "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 50.0),
            "max_bin": trial.suggest_int("max_bin", 8, 255),
        }
        return float(np.mean(_holdout_maes(selection, params)))

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the installed extras
        raise ModuleNotFoundError(
            "Imputation tuning uses Optuna's GPSampler, which needs PyTorch. Install the "
            "optional ML backends with: uv sync --extra dev --extra ml"
        ) from exc

    torch.set_num_threads(1)

    models_dir = settings.models.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"imputation_lgbm_seed{seed}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=_N_STARTUP_TRIALS),
        storage=f"sqlite:///{models_dir / 'imputation_tuning_studies.db'}",
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials)

    subsample_best = float(study.best_params["subsample"])
    best_params = ImputationBoostingParams(
        n_estimators=int(study.best_params["n_estimators"]),
        learning_rate=float(study.best_params["learning_rate"]),
        max_depth=int(study.best_params["max_depth"]),
        num_leaves=int(study.best_params["num_leaves"]),
        min_child_samples=int(study.best_params["min_child_samples"]),
        colsample_bytree=float(study.best_params["colsample_bytree"]),
        subsample=subsample_best,
        subsample_freq=1 if subsample_best < 1.0 else 0,
        reg_alpha=float(study.best_params["reg_alpha"]),
        reg_lambda=float(study.best_params["reg_lambda"]),
        min_data_per_group=int(study.best_params["min_data_per_group"]),
        cat_smooth=float(study.best_params["cat_smooth"]),
        max_bin=int(study.best_params["max_bin"]),
    )
    print(
        f"✅ Best trial: selection MAE={study.best_value:.4f} "
        f"(n_estimators={best_params.n_estimators}, "
        f"learning_rate={best_params.learning_rate:.4f}, max_depth={best_params.max_depth}, "
        f"num_leaves={best_params.num_leaves}, min_child_samples={best_params.min_child_samples}, "
        f"colsample_bytree={best_params.colsample_bytree:.3f}, subsample={best_params.subsample:.3f}, "
        f"reg_alpha={best_params.reg_alpha:.4e}, reg_lambda={best_params.reg_lambda:.4e}, "
        f"min_data_per_group={best_params.min_data_per_group}, cat_smooth={best_params.cat_smooth:.2f}, "
        f"max_bin={best_params.max_bin})"
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
        n_selection_series=n_selection_series,
        n_validation_series=n_validation_series,
        selection_seeds=selection_seeds,
        validation_seeds=validation_seeds,
        train_rows=int(len(selection[0][0]) - len(selection[0][1])),
        eval_rows=int(len(selection[0][1])),
        seed=seed,
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
        best_params=best_params,
    )

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
        _configure_mlflow()
        _log_study_to_mlflow(
            study=study,
            metadata=metadata,
            metadata_path=metadata_path,
            params_path=params_path,
            beats_default=beats_default,
        )
        return metadata_path

    params_path.write_text(json.dumps(best_params.model_dump(), indent=2), encoding="utf-8")
    print(f"\n✅ Tuned imputation hyperparameters written to: {params_path}\n")
    _configure_mlflow()
    _log_study_to_mlflow(
        study=study,
        metadata=metadata,
        metadata_path=metadata_path,
        params_path=params_path,
        beats_default=beats_default,
    )
    return params_path
