from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import mlflow
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from mlflow.data.pandas_dataset import from_pandas
from scipy import stats

from retail_forecasting.config import Settings, build_config_hash
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
from retail_forecasting.data.dataset import load_prepared_panel, panel_cache_filename
from retail_forecasting.utils.provenance import get_git_commit, utc_timestamp

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


def _log_study_to_mlflow(
    study: optuna.Study,
    metadata: ImputationTuningMetadata,
    metadata_path: Path,
    params_path: Path,
    beats_default: bool,
    panel: pd.DataFrame,
    panel_source: Path,
    config_hash: str,
    config_path: Path | None,
) -> None:
    """Record one completed search in MLflow: winner, validation verdict, and its artifacts."""
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    # Digest of the panel this search actually read, alongside the parquet it came from.
    # `n_series` and `teacher_fit_rows` pin the SHAPE; nothing pinned the content, and the panel
    # cache is keyed on four dataset settings only -- not on the preprocessing config nor on the
    # code version -- so a panel built before a change to `prepare_daily_panel` is served
    # unchanged afterwards, under a run whose git_commit says otherwise.
    #
    # Both filters are for MLflow talking about itself. It registers LocalArtifactDatasetSource
    # twice, so resolving any local path reports it as ambiguous while resolving it correctly;
    # and it warns that integer columns cannot hold missing values, which is advice about
    # enforcing a MODEL signature at inference time. No model is logged here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted")
        warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")
        dataset = from_pandas(panel, source=str(panel_source), name="imputation_train_panel")
        # The schema is computed lazily, and it is the access that warns -- so force it here,
        # inside the filter, rather than letting `log_input` trip it further down.
        _ = dataset.schema

    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_input(dataset, context="training")
        mlflow.log_params(metadata.best_params.model_dump())
        # The seed lists are deliberately absent: both are `seed` plus an offset and an index,
        # so they are reconstructible from what is here, and the metadata artifact below carries
        # them verbatim anyway.
        mlflow.log_params(
            {
                "n_trials": metadata.n_trials_requested,
                "seed": metadata.seed,
                "n_series": metadata.n_series,
                "n_selection_holdouts": metadata.n_selection_holdouts,
                "n_validation_holdouts": metadata.n_validation_holdouts,
                "selection_window_end": metadata.selection_window_end,
                "validation_window_start": metadata.validation_window_start,
            }
        )
        # The row counts are metrics rather than params despite measuring nothing: MLflow stores
        # params as strings, so `params.teacher_fit_rows > 10000` in the UI compares text. These
        # are the sizes a search is judged by -- invariant 41 makes `teacher_fit_rows` the
        # variable that decides whether tuning helps at all -- and they have to sort as numbers.
        mlflow.log_metrics(
            {
                "mae_selection_best": metadata.best_mae_selection,
                "mae_validation_tuned": metadata.best_mae_validation,
                "mae_validation_default": metadata.default_mae_validation,
                "improvement_pct": metadata.improvement_pct,
                "improvement_ci95_low": metadata.improvement_ci95[0],
                "improvement_ci95_high": metadata.improvement_ci95[1],
                "teacher_fit_rows": metadata.teacher_fit_rows,
                "n_selection_eval_rows": metadata.n_selection_eval_rows,
                "n_validation_eval_rows": metadata.n_validation_eval_rows,
            }
        )
        # Only when there was an incumbent to compete against. A run with none logs nothing
        # here rather than a sentinel: absent means "no incumbent", where a -1 or a 0 is a
        # number someone eventually averages.
        if metadata.incumbent_mae_validation is not None and metadata.incumbent_ci95 is not None:
            mlflow.log_metrics(
                {
                    "mae_validation_incumbent": metadata.incumbent_mae_validation,
                    "incumbent_ci95_low": metadata.incumbent_ci95[0],
                    "incumbent_ci95_high": metadata.incumbent_ci95[1],
                }
            )

        # The three branches of the persist decision, resolved into one searchable value. Worth
        # a tag of its own because the two gates do not decide alike: the defaults gate needs
        # the whole interval below zero, the incumbent gate goes on the mean. Left to reassemble
        # from the raw fields, a run that replaced its incumbent on a CI straddling zero reads
        # as a bug. Order matters -- a run that loses to both is reported against the defaults,
        # which is the branch that also deletes the superseded file.
        if metadata.persisted:
            outcome = "persisted"
        elif not beats_default:
            outcome = "no_gain_over_defaults"
        else:
            outcome = "lost_to_incumbent"

        mlflow.set_tags(
            {
                "persisted": str(metadata.persisted),
                "beats_incumbent": str(metadata.beats_incumbent),
                "outcome": outcome,
                "git_commit": metadata.git_commit or "unknown",
                # Two answers to "which configuration was this". The hash is the identity: it
                # covers the RESOLVED settings, so a hand-edited YAML or an environment override
                # moves it. The path is what a human recognises in the UI, where a bare sha256
                # identifies nothing. They differ exactly when someone changed the file.
                "config_hash": config_hash,
                "config_path": str(config_path) if config_path is not None else "unknown",
            }
        )

        # Two views of the same 300 trials, because they answer different questions. The metric
        # series on this run draws the convergence curve -- is it still improving, or has it
        # stalled. The nested run per trial carries that trial's own hyperparameters, which the
        # curve cannot hold: a step number is not a place to put thirteen values. Only the
        # nested form lets the UI plot a hyperparameter against the objective across the search.
        #
        # Optuna's own storage has the same trials, so this is a duplicate on purpose: reading
        # it there means a second tool over a second database, and this is where the run's
        # verdict, panel digest and configuration already live.
        #
        # `best_trial_number` names the winner among the children, which are otherwise 300 rows
        # sorted by nothing in particular.
        mlflow.log_param("best_trial_number", study.best_trial.number)

        best_so_far = float("inf")
        for trial in study.trials:
            # None means the trial never produced a value -- failed, or still running when the
            # study was read. There is nothing to log and nothing to plot.
            if trial.value is None:
                continue
            best_so_far = min(best_so_far, trial.value)
            mlflow.log_metric("trial_mae", trial.value, step=trial.number)
            mlflow.log_metric("trial_mae_best_so_far", best_so_far, step=trial.number)

            # Zero-padded so the UI's alphabetical run list runs 0001, 0002, ... rather than
            # 1, 10, 100.
            with mlflow.start_run(run_name=f"trial-{trial.number:04d}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("mae", trial.value)
                if trial.duration is not None:
                    # Cost per trial varies by more than an order of magnitude with the tree
                    # count, so this is what makes "was that gain worth the runtime" answerable.
                    mlflow.log_metric("duration_seconds", trial.duration.total_seconds())

        mlflow.log_artifact(str(metadata_path))
        if metadata.persisted:
            mlflow.log_artifact(str(params_path))


def tune_imputation_lgbm(
    settings: Settings,
    n_trials: int | None = None,
    seed: int | None = None,
    n_selection_holdouts: int = N_SELECTION_HOLDOUTS,
    n_validation_holdouts: int = N_VALIDATION_HOLDOUTS,
    config_path: Path | None = None,
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
        f"\nValidation: tuned={best_mae_validation:.4f} vs "
        f"default={default_mae_validation:.4f} ({improvement_pct:+.2f}%), "
        f"CI95 of the difference [{ci_lo:+.4f}, {ci_hi:+.4f}]"
    )

    if not beats_default:
        print(
            "The tuned hyperparameters do NOT beat the untuned defaults by a margin the "
            "validation draws can distinguish from zero, so they were NOT persisted -- the "
            "pipeline keeps using the defaults.\n"
            f"    Decision recorded in: {metadata_path}\n"
        )

        if params_path.exists():
            params_path.unlink()
            print(f"Removed the superseded params file: {params_path}\n")
    elif beats_incumbent is False:
        print(
            "The tuned hyperparameters beat the untuned defaults but NOT the incumbent "
            "already on disk, so the incumbent was KEPT and this run's winner discarded.\n"
            f"    Decision recorded in: {metadata_path}\n"
        )
    else:
        params_path.write_text(json.dumps(best_params.model_dump(), indent=2), encoding="utf-8")
        print(f"\nTuned imputation hyperparameters written to: {params_path}\n")

    _log_study_to_mlflow(
        study=study,
        metadata=metadata,
        metadata_path=metadata_path,
        params_path=params_path,
        beats_default=beats_default,
        panel=panel,
        panel_source=settings.dataset.processed_panel_dir
        / panel_cache_filename(settings.dataset, "train"),
        config_hash=build_config_hash(settings),
        config_path=config_path,
    )
    return params_path if should_persist else metadata_path
