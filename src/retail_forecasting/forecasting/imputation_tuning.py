from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from hashlib import sha256
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
from retail_forecasting.utils.logging import Table, fields, get_logger, rule, thousands
from retail_forecasting.utils.provenance import get_git_commit, utc_timestamp

logger = get_logger(__name__)

N_SELECTION_HOLDOUTS = 30

# One progress line per trial. Optuna's own INFO logging prints a paragraph per trial with the
# full parameter dict, which buries everything else; it is silenced below and replaced by this.
# Every trial and not every tenth: measured at 10-35s a trial on average but with a 60x spread
# between the cheapest and dearest, a coarser interval leaves no way to tell a working search
# from a hung one. At the search's own scale 300 lines is nothing.
_PROGRESS_EVERY_TRIALS = 1

# Only COMPLETE counts as a trial the search got something out of. Studies from before the
# pruner was removed hold PRUNED trials, and counting those against the budget would retire
# it without the sampler ever having seen a value for them.
_FINISHED_STATES = (optuna.trial.TrialState.COMPLETE,)

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
    # Lower bounds pulled up to where the six searches of reports/sampler_ab agreed: every
    # winner sat at max_depth 10-11 with colsample 0.55-0.66 and subsample 0.93-1.0. They stop
    # at the untuned defaults (max_depth 6, colsample 1.0) rather than at the winners' range,
    # because the defaults have to stay expressible: they are enqueued as a reference trial and
    # pinned by `test_the_untuned_defaults_sit_inside_the_search_space`.
    "max_depth": (6, 12),
    "num_leaves": (2, 1024),
    "min_child_samples": (2, 100),
    "min_data_per_group": (1, 100),
    "max_bin": (8, 255),
}

_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "subsample": (0.85, 1.0),
    "learning_rate": (0.005, 0.3),
    "colsample_bytree": (0.5, 1.0),
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


def _hhmm(seconds: float) -> str:
    """Duration as ``1h52m``, ``12m`` or ``45s``, all the precision a progress line needs.

    Seconds below the minute and not just ``0m``: a trial here can cost one second, and a
    search that reports ``restante 0m`` from start to finish reports nothing.
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def _pace_seconds(study: optuna.Study) -> float | None:
    """Seconds per trial, measured on the trials PAST the sampler's random startup.

    The startup trials are drawn uniformly and are not representative of the pace: measured
    across the first six, the cheapest cost 1s and the dearest 61s, and the GP then converges
    on the expensive corner of the space, so a rate that includes them projects an ETA far
    below the truth. Read from the durations the study itself stores rather than from this
    process's clock, so the pace survives a resume. None until there is anything to measure.
    """
    all_seconds: list[float] = []
    past_startup: list[float] = []
    for trial in study.get_trials(deepcopy=False, states=_FINISHED_STATES):
        if trial.duration is None:
            continue
        seconds = trial.duration.total_seconds()
        all_seconds.append(seconds)
        if trial.number >= _N_STARTUP_TRIALS:
            past_startup.append(seconds)

    timed = past_startup or all_seconds
    if not timed:
        return None
    return sum(timed) / len(timed)


def _objective_mae(
    trial: optuna.Trial, draws: list[Holdout], params: dict[str, int | float]
) -> float:
    """Mean reconstruction MAE over every draw.

    Every trial sees the SAME draws, so the mean is a paired comparison against the other
    trials: the between-draw variance that dominates the objective is shared rather than
    resampled, which is what makes two trial values comparable at all.

    `trial` is unused, and kept because Optuna hands the objective a trial and because the
    signature is what the search calls through.
    """
    del trial
    return float(np.mean(_holdout_maes(draws, params)))


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

    rule(logger, "tuning del imputador supervisado")
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
    fields(
        logger,
        {
            "panel": f"{n_series} series, {thousands(len(panel))} filas",
            "ventanas": (
                f"selección hasta {selection.window_end.date().isoformat()} · "
                f"validación desde {validation.window_start.date().isoformat()}"
            ),
            "puntuables": (
                f"{thousands(selection.n_eval_rows)} vs "
                f"{thousands(validation.n_eval_rows)} filas por extracción"
            ),
            "teacher": f"~{thousands(selection.teacher_fit_rows)} filas limpias (panel completo)",
            "extracciones": (
                f"{len(selection.draws)} selección · {len(validation.draws)} validación"
            ),
            "presupuesto": f"{n_trials} trials · sin pruning",
            "semillas": f"selección {seed}+ · validación {seed + _VALIDATION_SEED_OFFSET}+",
        },
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
        return _objective_mae(trial, selection.draws, params)

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the installed extras
        raise ModuleNotFoundError(
            "Imputation tuning uses Optuna's GPSampler, which needs PyTorch. Install the "
            "optional ML backends with: uv sync --extra dev --extra ml"
        ) from exc

    # Not a speed setting -- measured at 69.0s against 68.5s over 300 trials, and identical
    # results either way. Without it the process SEGFAULTS at the first GP fit, reproducibly,
    # once LightGBM has run under joblib's threads in the same process: two OpenMP runtimes,
    # torch's and LightGBM's, and torch left free to spawn its own pool. Removing it as dead
    # weight is what 1c7a490 did, on a measurement that exercised the sampler alone.
    torch.set_num_threads(1)

    models_dir = settings.models.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Read the incumbent before the search, not after: the persist decision below either
    # overwrites or deletes this exact file, so by then there is nothing left to compare against.
    # None means no incumbent, which is the first ever run rather than an error.
    params_path = models_dir / IMPUTATION_LGBM_PARAMS_FILENAME
    incumbent_params: dict[str, int | float] | None = (
        json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else None
    )

    # Stable name, so an interrupted search RESUMES instead of starting over: at minutes per
    # trial, losing the work to a closed lid is not an acceptable failure mode. The name binds
    # the study to what makes its trials comparable, the seed, the panel size and the search
    # space itself, so narrowing a bound starts a fresh study rather than silently resuming one
    # whose trials came from a different space.
    space_digest = sha256(
        json.dumps({"int": _INT_BOUNDS, "float": _FLOAT_BOUNDS}, sort_keys=True).encode()
    ).hexdigest()[:8]
    study_name = f"imputation_lgbm_seed{seed}_{n_series}series_{space_digest}"

    # Its INFO handler emits one paragraph per trial. Ours reports every
    # `_PROGRESS_EVERY_TRIALS` instead, so warnings from Optuna still get through.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Known and accepted: the sampler choice is measured in reports/sampler_ab. Left to fire
    # once per import would still put it in the middle of the setup block on every run.
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GPSampler(
            seed=seed,
            n_startup_trials=_N_STARTUP_TRIALS,
            deterministic_objective=True,
        ),
        # No pruning, and not for want of trying: a MedianPruner here deadlocked the search.
        # Both halves of Optuna read COMPLETE trials only -- the pruner takes its median over
        # them (`pruners/_percentile.py`) and GPSampler fits on them (`samplers/_gp/sampler.py`)
        # -- so a pruned trial teaches the sampler nothing, and a deterministic GP with no new
        # data re-proposes the point it just had cut. Measured on a 300-trial run: 76 of the
        # first 91 trials pruned, the last COMPLETE at trial 52, and 31 consecutive trials
        # covering 8 distinct parameter vectors. The median only tightens too, since a trial
        # joins the reference set only by beating it. The saving was never needed: 300 unpruned
        # trials cost two to three hours.
        pruner=optuna.pruners.NopPruner(),
        storage=f"sqlite:///{models_dir / 'imputation_tuning_studies.db'}",
        study_name=study_name,
        load_if_exists=True,
    )

    # `n_trials` is a TARGET, not an increment: on a resumed study Optuna would otherwise run
    # that many MORE trials on top of the ones already stored.
    already_done = len(study.get_trials(deepcopy=False, states=_FINISHED_STATES))
    trials_todo = max(0, n_trials - already_done)

    if already_done:
        # The reference trials are only enqueued for a fresh study. Enqueueing them again on
        # every resume would re-measure two configurations the study already scored.
        fields(logger, {"reanudando": f"{already_done} trials ya en el estudio {study_name}"})
    else:
        study.enqueue_trial(dict(DEFAULT_SUPERVISED_LGBM_PARAMS))
        fields(logger, {"referencia": "defaults sin sintonizar, encolados como trial"})
        if incumbent_params is not None:
            study.enqueue_trial(dict(incumbent_params))
            fields(logger, {"referencia": "campeón en disco, encolado como trial"})

    started = time.monotonic()
    # Widths chosen so the whole row fits a narrow terminal pane without wrapping, which is
    # the only thing that actually separates one trial from the next.
    table = Table(
        logger,
        {
            "trial": len(f"{n_trials}/{n_trials}"),
            "MAE": 6,
            "mejor": 6,
            "s/trial": 7,
            "pasado": 6,
            "queda": 6,
        },
    )
    table.header()

    def report_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # `trial.number` counts from the start of the study, resumed trials included, so it is
        # the right numerator against the target rather than against what is left to run.
        done = trial.number + 1
        if done % _PROGRESS_EVERY_TRIALS and done != n_trials:
            return
        pace = _pace_seconds(study)
        table.row(
            {
                "trial": f"{done}/{n_trials}",
                "MAE": "-" if trial.value is None else f"{trial.value:.4f}",
                "mejor": f"{study.best_value:.4f}",
                "s/trial": "-" if pace is None else f"{pace:.0f}",
                "pasado": _hhmm(time.monotonic() - started),
                "queda": "-" if pace is None else _hhmm(pace * max(n_trials - done, 0)),
            }
        )

    study.optimize(objective, n_trials=trials_todo, callbacks=[report_progress])

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
    rule(logger, "mejor trial")
    fields(
        logger,
        {
            "MAE selección": f"{study.best_value:.4f}",
            "árboles": (
                f"{best_params.n_estimators} · lr {best_params.learning_rate:.4f} · "
                f"profundidad {best_params.max_depth} · hojas {best_params.num_leaves}"
            ),
            "muestreo": (
                f"filas {best_params.subsample:.3f} · columnas {best_params.colsample_bytree:.3f} · "
                f"min_child {best_params.min_child_samples}"
            ),
            "regularización": (
                f"alpha {best_params.reg_alpha:.2e} · lambda {best_params.reg_lambda:.2e}"
            ),
            "categóricas": (
                f"min_data {best_params.min_data_per_group} · smooth {best_params.cat_smooth:.2f} · "
                f"bins {best_params.max_bin}"
            ),
        },
    )

    logger.info("")
    logger.info("Puntuando al ganador y a los defaults sobre las extracciones de validación")
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
        logger.info("Puntuando al campeón en disco sobre las mismas extracciones")
        incumbent_maes = _holdout_maes(validation.draws, incumbent_params)
        incumbent_mae_validation = float(np.mean(incumbent_maes))
        inc_lo, inc_hi = _mean_ci95(tuned_maes - incumbent_maes)
        incumbent_ci95 = [inc_lo, inc_hi]

        beats_incumbent = bool(np.mean(tuned_maes - incumbent_maes) < 0.0)
        decisive = inc_hi < 0.0 or inc_lo > 0.0
        logger.info(
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

    rule(logger, "veredicto")
    fields(
        logger,
        {
            "ganador": f"MAE {best_mae_validation:.4f} en validación",
            "defaults": f"MAE {default_mae_validation:.4f} en validación",
            "mejora": f"{improvement_pct:+.2f}%",
            "IC95": f"[{ci_lo:+.4f}, {ci_hi:+.4f}] sobre la diferencia pareada",
        },
    )

    if not beats_default:
        # Two different failures share this gate, and calling both of them a tie misreports
        # one: a CI that straddles zero is a coin flip, but a CI entirely above it means the
        # winner is measurably WORSE than the defaults it was supposed to beat.
        verdict = (
            "es peor que los defaults, y el intervalo no toca el cero"
            if ci_lo > 0
            else "no se distingue de cero en las extracciones de validación"
        )
        logger.info(
            "  NO persistido: la mejora %s, así que el pipeline sigue con los defaults", verdict
        )
        fields(logger, {"decisión": metadata_path})

        if params_path.exists():
            params_path.unlink()
            fields(logger, {"retirado": params_path})
    elif beats_incumbent is False:
        logger.info(
            "  NO persistido: bate a los defaults pero no al campeón en disco, que se mantiene"
        )
        fields(logger, {"decisión": metadata_path})
    else:
        params_path.write_text(json.dumps(best_params.model_dump(), indent=2), encoding="utf-8")
        fields(logger, {"persistido": params_path})

    # Everything this run produces is already on disk by now, and the caller is about to be
    # handed the path to it. A tracking store that is locked, missing or out of disk would
    # otherwise take a search of half an hour down with it at its last step, having lost
    # nothing but the record. Broad on purpose: no failure to write a log is worth the run.
    try:
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
    except Exception as exc:  # noqa: BLE001 - see above
        logger.warning(
            "el registro en MLflow falló y la búsqueda no se ve afectada: %s: %s. Todo lo que "
            "produjo está en disco, empezando por %s",
            type(exc).__name__,
            exc,
            metadata_path,
        )
    return params_path if should_persist else metadata_path
