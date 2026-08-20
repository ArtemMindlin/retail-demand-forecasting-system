from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import mlflow
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from scipy import stats

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_tuning import (
    ImputationBoostingParams,
    ImputationTuningMetadata,
)
from retail_forecasting.data.censorship import (
    DEFAULT_SUPERVISED_LGBM_PARAMS,
    IMPUTATION_LGBM_PARAMS_FILENAME,
    LatentDemandImputer,
    synthetic_censor_holdout,
)
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.evaluation.reporting import get_git_commit, utc_timestamp

N_SELECTION_HOLDOUTS = 15

N_VALIDATION_HOLDOUTS = 25

_VALIDATION_SEED_OFFSET = 10_000

_VALIDATION_WINDOW_FRACTION = 1.0 / 3.0

# Optuna's own default for GPSampler. Five was below it, which is thin ground for fitting a
# Gaussian process over 13 dimensions.
_N_STARTUP_TRIALS = 10

# MLflow experiment collecting every imputation search, so runs stay comparable across time.
_MLFLOW_EXPERIMENT = "imputation_lgbm_tuning"

_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# What `objective` actually suggests. `subsample_freq` is absent on purpose: it is derived from
# `subsample` rather than searched, so enqueueing it would name a parameter the space has no
# distribution for.
_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "n_estimators": (50, 3000),
    "max_depth": (2, 12),
    "num_leaves": (2, 1024),
    "min_child_samples": (2, 100),
    "min_data_per_group": (1, 100),
    "max_bin": (8, 255),
}

_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "subsample": (0.4, 1.0),
    "learning_rate": (0.005, 0.3),
    "colsample_bytree": (0.3, 1.0),
    "reg_alpha": (1e-8, 100.0),
    "reg_lambda": (1e-8, 100.0),
    "cat_smooth": (0.0, 50.0),
}


class Holdout(NamedTuple):
    """One synthetic-censoring draw: a panel with some clean days faked, plus the answer key."""

    censored: pd.DataFrame
    eval_idx: np.ndarray
    true_demand: np.ndarray


def _split_temporal_windows(
    panel: pd.DataFrame, validation_fraction: float = _VALIDATION_WINDOW_FRACTION
) -> tuple[pd.Series, pd.Series]:
    """Cut the panel's calendar into an early selection window and a late validation window.

    Nothing is removed from the panel: these are MASKS over the full panel.

    Returns:
        ``(selection_mask, validation_mask)``, disjoint and covering the panel.
    """
    days = np.sort(panel["date"].unique())
    if len(days) < 2:
        raise ValueError(
            "Cannot split the imputation tuning panel by time: it spans "
            f"{len(days)} distinct dates, and a disjoint validation window needs at least 2."
        )
    n_validation_days = min(len(days) - 1, max(1, round(len(days) * validation_fraction)))
    is_validation = panel["date"] >= pd.Timestamp(days[len(days) - n_validation_days])
    return ~is_validation, is_validation


@dataclass(frozen=True)
class HoldoutSet:
    """Every draw over one window, together with the facts that describe the window as a whole.

    One class for both roles, with no marker saying which: nothing here behaves differently for
    selection than for validation, so a role field would be dead weight. The role lives in the
    variable name and in what the caller does with it.

    The boundary dates are owned HERE rather than derived by the caller, because that derivation
    is where a bug lived: recovering the selection window's last day as ``cut_date - 1 day``
    names a date absent from the panel wherever the calendar has a gap at the boundary. Born
    beside the mask they come from, there is nowhere left to derive them wrongly.
    """

    draws: list[Holdout]
    seeds: list[int]
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    @property
    def n_eval_rows(self) -> int:
        """Scorable rows per draw. Identical across draws by construction: the count is the
        window's clean-row pool times a fixed fraction, so draw 0 stands for all of them."""
        return len(self.draws[0].eval_idx)

    @property
    def teacher_fit_rows(self) -> int:
        """Clean rows the LGBM teacher actually fits on.

        The variable that decides whether tuning helps at all: measured against the untuned
        defaults, the gain shrinks monotonically as this grows and reverses at scale. See
        invariant 41.
        """
        return int((self.draws[0].censored["stockout_hours"] == 0).sum())


def _build_holdout_set(
    panel: pd.DataFrame, seeds: list[int], censorable_mask: pd.Series
) -> HoldoutSet:
    """Draw one synthetic-censoring holdout per seed, censoring only within ``censorable_mask``.

    Built once and reused by every trial, so all candidates are scored on identical data and
    differences between them are hyperparameters rather than resampling luck. Every draw keeps
    the FULL panel -- only the eligible evaluation rows differ between the two windows.
    """
    draws = []
    for seed in seeds:
        censored, eval_idx, true_demand = synthetic_censor_holdout(
            panel, seed=seed, censorable_mask=censorable_mask
        )
        draws.append(Holdout(censored, eval_idx, true_demand))

    window_dates = panel.loc[censorable_mask, "date"]
    return HoldoutSet(
        draws=draws,
        seeds=list(seeds),
        window_start=pd.Timestamp(window_dates.min()),
        window_end=pd.Timestamp(window_dates.max()),
    )


def _holdout_maes(holdouts: list[Holdout], params: dict[str, int | float]) -> np.ndarray:
    """Reconstruction MAE of one hyperparameter set on each holdout, one value per draw."""

    def holdout_mae(holdout: Holdout) -> float:
        censored, eval_idx, true_demand = holdout
        imputed = LatentDemandImputer(strategy="supervised", lgbm_params=params).impute(censored)
        pred = imputed.loc[eval_idx, "latent_demand_est"].to_numpy(dtype=float)
        return float(np.mean(np.abs(pred - true_demand)))

    n_jobs = min(len(holdouts), cpu_count())
    maes = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(holdout_mae)(h) for h in holdouts)
    return np.asarray(maes, dtype=float)


def _mean_ci95(deltas: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% CI for the mean of the paired per-holdout MAE differences.

    Student-t, not the normal's 1.96: at ``N_VALIDATION_HOLDOUTS`` draws the correct multiplier
    is ``t(0.975, n-1) = 2.064``, and 1.96 would quote a ~93% interval as 95% -- permissive in
    the one direction a gate must not be.

    ``simulation/operations.py`` keeps its own bootstrap on purpose: that one resamples whole
    ORIGINS as clusters, which no closed form covers.
    """
    n = len(deltas)
    if n < 2:
        raise ValueError(f"A 95% CI for the mean needs at least 2 draws, got {n}.")
    mean = float(np.mean(deltas))
    half_width = float(stats.t.ppf(0.975, n - 1) * np.std(deltas, ddof=1) / np.sqrt(n))
    return mean - half_width, mean + half_width


def _configure_mlflow() -> None:
    """Point MLflow at its tracking store and select the imputation-tuning experiment.

    The tracking store is one database holding many experiments, separated by name -- not one
    database per use case -- so extending tracking elsewhere means a new experiment name here,
    not a new store.
    """
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
    """
    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(metadata.best_params.model_dump())
        mlflow.log_params(
            {
                "n_trials": metadata.n_trials_requested,
                "seed": metadata.seed,
                "n_series": metadata.n_series,
                "teacher_fit_rows": metadata.teacher_fit_rows,
                "validation_window_start": metadata.validation_window_start,
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

    The winning params are persisted only if they clear TWO gates, both judged on the paired
    per-draw confidence interval rather than the point estimate (a mean that merely happens to
    land below zero is what a coin flip looks like):

    * vs the untuned defaults -- does tuning buy anything at all? This is the number the thesis
      cites, and failing it deletes any params file an earlier run left behind, since the
      pipeline should fall back to defaults rather than use a winner this run rejected.
    * vs the incumbent already on disk -- is THIS search better than the last one that passed?
      The defaults comparison is blind to the incumbent, so without this a worse search
      overwrote a better one (observed: a -4.65% run replaced a -5.33% winner, scoring better on
      selection and worse on validation). Failing this keeps the incumbent untouched.

    The metadata file is written either way, recording both comparisons and the decision.

    Only the winning hyperparameters are persisted (not fitted weights), so
    ``LatentDemandImputer`` still re-fits on each panel's own clean days -- this run only
    changes which 13 hyperparameters that fit uses.

    Returns:
        The path of the written ``imputation_lgbm_params.json`` when the winner cleared both
        gates, otherwise the path of the metadata file recording why it did not.
    """
    n_trials = n_trials if n_trials is not None else settings.models.tuning_trials
    seed = seed if seed is not None else settings.project.random_seed

    print("\n" + "=" * 50)
    print("IMPUTATION LGBM TUNING")
    print("=" * 50 + "\n")
    print("Loading train panel...")
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )

    selection_mask, validation_mask = _split_temporal_windows(panel)
    n_series = int(panel["series_id"].nunique())

    selection_seeds = [seed + i for i in range(n_selection_holdouts)]
    validation_seeds = [seed + _VALIDATION_SEED_OFFSET + i for i in range(n_validation_holdouts)]
    selection = _build_holdout_set(panel, selection_seeds, selection_mask)
    validation = _build_holdout_set(panel, validation_seeds, validation_mask)
    print(
        f"Temporal split of {n_series} series over {len(panel):,} rows: selection through "
        f"{selection.window_end.date().isoformat()}, validation from "
        f"{validation.window_start.date().isoformat()} ({selection.n_eval_rows:,} vs "
        f"{validation.n_eval_rows:,} scorable rows per draw)"
    )
    print(
        f"The teacher fits on the FULL panel either way: ~{selection.teacher_fit_rows:,} clean rows"
    )
    print(
        f"{len(selection.draws)} selection holdouts (seeds {selection.seeds}), "
        f"{len(validation.draws)} validation holdouts (seeds {validation.seeds})"
    )

    def objective(trial: optuna.Trial) -> float:
        subsample = trial.suggest_float("subsample", *_FLOAT_BOUNDS["subsample"])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *_INT_BOUNDS["n_estimators"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *_FLOAT_BOUNDS["learning_rate"], log=True
            ),
            "max_depth": trial.suggest_int("max_depth", *_INT_BOUNDS["max_depth"]),
            "num_leaves": trial.suggest_int("num_leaves", *_INT_BOUNDS["num_leaves"], log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *_INT_BOUNDS["min_child_samples"]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *_FLOAT_BOUNDS["colsample_bytree"]
            ),
            "subsample": subsample,
            "subsample_freq": 1 if subsample < 1.0 else 0,
            "reg_alpha": trial.suggest_float("reg_alpha", *_FLOAT_BOUNDS["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *_FLOAT_BOUNDS["reg_lambda"], log=True),
            "min_data_per_group": trial.suggest_int(
                "min_data_per_group", *_INT_BOUNDS["min_data_per_group"]
            ),
            "cat_smooth": trial.suggest_float("cat_smooth", *_FLOAT_BOUNDS["cat_smooth"]),
            "max_bin": trial.suggest_int("max_bin", *_INT_BOUNDS["max_bin"]),
        }
        return float(np.mean(_holdout_maes(selection.draws, params)))

    try:
        import torch  # noqa: F401  -- imported for the check, GPSampler is what uses it
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the installed extras
        raise ModuleNotFoundError(
            "Imputation tuning uses Optuna's GPSampler, which needs PyTorch. Install the "
            "optional ML backends with: uv sync --extra dev --extra ml"
        ) from exc

    models_dir = settings.models.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Read the incumbent before the search, not after: the persist decision below either
    # overwrites or deletes this exact file, so by then there is nothing left to compare against.
    # None means no incumbent, which is the first ever run rather than an error.
    params_path = models_dir / IMPUTATION_LGBM_PARAMS_FILENAME
    incumbent_params: dict[str, int | float] | None = (
        json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else None
    )

    study_name = f"imputation_lgbm_seed{seed}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GPSampler(
            seed=seed,
            n_startup_trials=_N_STARTUP_TRIALS,
            deterministic_objective=True,
        ),
        storage=f"sqlite:///{models_dir / 'imputation_tuning_studies.db'}",
        study_name=study_name,
    )

    study.enqueue_trial(dict(DEFAULT_SUPERVISED_LGBM_PARAMS))
    print("Enqueued the untuned defaults as a reference trial")
    if incumbent_params is not None:
        study.enqueue_trial(dict(incumbent_params))
        print("Enqueued the incumbent params on disk as a reference trial")

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
        f"Best trial: selection MAE={study.best_value:.4f} "
        f"(n_estimators={best_params.n_estimators}, "
        f"learning_rate={best_params.learning_rate:.4f}, max_depth={best_params.max_depth}, "
        f"num_leaves={best_params.num_leaves}, min_child_samples={best_params.min_child_samples}, "
        f"colsample_bytree={best_params.colsample_bytree:.3f}, subsample={best_params.subsample:.3f}, "
        f"reg_alpha={best_params.reg_alpha:.4e}, reg_lambda={best_params.reg_lambda:.4e}, "
        f"min_data_per_group={best_params.min_data_per_group}, cat_smooth={best_params.cat_smooth:.2f}, "
        f"max_bin={best_params.max_bin})"
    )

    print("Scoring the winner and the untuned defaults on the validation holdouts...")
    tuned_maes = _holdout_maes(validation.draws, best_params.model_dump())
    default_maes = _holdout_maes(validation.draws, dict(DEFAULT_SUPERVISED_LGBM_PARAMS))
    best_mae_validation = float(np.mean(tuned_maes))
    default_mae_validation = float(np.mean(default_maes))
    improvement_pct = float(
        (best_mae_validation - default_mae_validation) / default_mae_validation * 100
    )

    ci_lo, ci_hi = _mean_ci95(tuned_maes - default_maes)
    beats_default = ci_hi < 0.0

    incumbent_mae_validation: float | None = None
    incumbent_ci95: list[float] | None = None
    beats_incumbent: bool | None = None
    if incumbent_params is not None:
        print("Scoring the incumbent params already on disk on the same draws...")
        incumbent_maes = _holdout_maes(validation.draws, incumbent_params)
        incumbent_mae_validation = float(np.mean(incumbent_maes))
        inc_lo, inc_hi = _mean_ci95(tuned_maes - incumbent_maes)
        incumbent_ci95 = [inc_lo, inc_hi]
        # The MEAN decides here, unlike the defaults gate above. That gate guards a CLAIM the
        # thesis makes, where an unresolvable difference must be reported as no difference. This
        # one only picks which of two files sits on disk, and when the draws cannot separate them
        # "keep whichever arrived first" is not a more defensible tiebreak than "keep the better
        # mean" -- it just looks more cautious. Note this reduces to one comparison: the interval
        # is symmetric about the mean, so a decisive verdict either way already agrees with it,
        # and the mean only does real work in the straddling case.
        beats_incumbent = bool(np.mean(tuned_maes - incumbent_maes) < 0.0)
        decisive = inc_hi < 0.0 or inc_lo > 0.0
        print(
            f"   challenger={best_mae_validation:.4f} vs incumbent="
            f"{incumbent_mae_validation:.4f}, CI95 [{inc_lo:+.4f}, {inc_hi:+.4f}] "
            f"({'decisive' if decisive else 'indistinguishable, decided on the mean'}) -> "
            f"{'replaces it' if beats_incumbent else 'does NOT replace it'}"
        )

    should_persist = beats_default and beats_incumbent is not False

    metadata = ImputationTuningMetadata(
        n_trials_requested=n_trials,
        best_mae_selection=float(study.best_value),
        best_mae_validation=best_mae_validation,
        default_mae_validation=default_mae_validation,
        improvement_pct=improvement_pct,
        improvement_ci95=[ci_lo, ci_hi],
        incumbent_mae_validation=incumbent_mae_validation,
        incumbent_ci95=incumbent_ci95,
        beats_incumbent=beats_incumbent,
        persisted=should_persist,
        n_selection_holdouts=len(selection.draws),
        n_validation_holdouts=len(validation.draws),
        n_series=n_series,
        selection_window_end=selection.window_end.date().isoformat(),
        validation_window_start=validation.window_start.date().isoformat(),
        n_selection_eval_rows=selection.n_eval_rows,
        n_validation_eval_rows=validation.n_eval_rows,
        selection_seeds=selection.seeds,
        validation_seeds=validation.seeds,
        teacher_fit_rows=selection.teacher_fit_rows,
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
    elif beats_incumbent is False:
        # Deliberately NOT deleted: unlike the branch above, tuning did beat the defaults here,
        # so the incumbent on disk is still a validated winner -- and a better one than this
        # run produced. Leaving it in place is the whole point of this gate.
        print(
            "⚖️  The tuned hyperparameters beat the untuned defaults but NOT the incumbent "
            "already on disk, so the incumbent was KEPT and this run's winner discarded.\n"
            f"    Decision recorded in: {metadata_path}\n"
        )
    else:
        params_path.write_text(json.dumps(best_params.model_dump(), indent=2), encoding="utf-8")
        print(f"\n✅ Tuned imputation hyperparameters written to: {params_path}\n")

    _configure_mlflow()
    _log_study_to_mlflow(
        study=study,
        metadata=metadata,
        metadata_path=metadata_path,
        params_path=params_path,
        beats_default=should_persist,
    )
    return params_path if should_persist else metadata_path
