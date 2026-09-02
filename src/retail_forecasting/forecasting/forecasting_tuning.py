"""Hyperparameter search for the forecasting models (``run_mode = tune_forecasting``).

Separate from `run_experiment` for the reason `imputation_tuning` is separate: the search
costs far more than one run, so it is paid once here and its winner persisted, instead of
being repeated inside every experiment via `models.use_tuning`.

One BACKEND per run, picked by `models.tuning_backend`. Two backends mean two independent
searches, two gates and two verdicts, so they are two executions.

Two layers, as in the imputer. The CALENDAR is deterministic: an early selection window
drives the search, a later validation window judges the winner, and neither is drawn at
random because validation must come after selection in time. The DRAWS are random, inside
the already-fixed window.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd

# Imported for its side effect and its ORDER, not for its API. Optuna's GPSampler loads torch
# lazily on its first modelled trial, by which time LightGBM and CatBoost have initialised their
# own runtimes -- and torch loading second segfaults the process on macOS (measured: SIGSEGV at
# trial `n_startup_trials + 1`, and no crash when torch goes first). Ruff sorts third-party
# above first-party, so this line stays ahead of the model imports that pull those two in.
import torch  # noqa: F401
from joblib import Parallel, cpu_count, delayed
from mlflow.data.pandas_dataset import from_pandas

from retail_forecasting.config import Settings, build_config_hash
from retail_forecasting.contracts.contracts_config import BoostingBackend
from retail_forecasting.contracts.contracts_tuning import ForecastingTuningMetadata
from retail_forecasting.data.dataset import load_prepared_panel, panel_cache_filename
from retail_forecasting.drift import label_all_regimes
from retail_forecasting.features.engineering import build_supervised_frame
from retail_forecasting.forecasting.imputation_tuning import hhmm, pace_seconds
from retail_forecasting.forecasting.pipeline import (
    _split_train_calibration,
    _train_conformal_model,
)
from retail_forecasting.forecasting.tuned_params import (
    CORE_PARAMS,
    _drop_backend_block,
    _write_backend_block,
    default_params,
    read_tuned_params,
)
from retail_forecasting.inventory.newsvendor import attach_inventory_costs, choose_order_quantity
from retail_forecasting.models.boosting import LightGBMModel
from retail_forecasting.models.catboosting import CatBoostingModel
from retail_forecasting.tracking import EXPERIMENT_FORECASTING_TUNING, MLFLOW_TRACKING_URI
from retail_forecasting.utils.io import winkler_score
from retail_forecasting.utils.logging import Table, fields, rule, thousands
from retail_forecasting.utils.provenance import get_git_commit, utc_timestamp
from retail_forecasting.utils.stats import mean_ci95
from retail_forecasting.visualization.plots import render_pareto_front

logger = logging.getLogger(__name__)

EVAL_DAYS = 7

# Draws per window. The search averages over the selection draws; the gate is a paired
# comparison over the validation ones, which is what turns "wins by 2%" into a confidence
# interval. Same counts as the imputer.
N_SELECTION_DRAWS = 30
N_VALIDATION_DRAWS = 25

# Validation seeds must not collide with selection seeds, or the two sets stop being
# independent draws of the same generator.
_VALIDATION_SEED_OFFSET = 10_000

# One thread per fit: the draws run in parallel instead, the trade `censorship.py` already
# makes for the imputer's LGBM. A fit this size does not scale to ten cores anyway.
_TUNING_MODEL_THREADS = 1

# Trials the GP sampler draws uniformly before it starts modelling the space, and how often the
# progress table prints a row. Both mirror the imputer's search.
_N_STARTUP_TRIALS = 10
_PROGRESS_EVERY_TRIALS = 1
_FINISHED_STATES = (optuna.trial.TrialState.COMPLETE,)

# One experiment for both backends: they answer the same question about the same panel, and a
# `backend` tag separates them without splitting the history in two.
# Bumped whenever `_draw_cost` changes WHAT IT MEASURES. Optuna resumes by study name, so a
# study whose trials were scored under a different objective would be reused in silence: the
# search would see `trials_todo == 0`, return at once, and the gate would validate a winner
# selected under the old cost against defaults measured under the new one -- a mixed verdict
# that looks complete. v1 scored the raw model; v2 scored it through the conformal layer but
# calibrated on the class default of 21 days because `validation` was not wired into this
# mode's settings; v3 calibrates on the 14 days configs/experiment/default.yaml actually
# deploys.
_OBJECTIVE_VERSION = 4

_MLFLOW_EXPERIMENT = EXPERIMENT_FORECASTING_TUNING

# Searched under the DATACLASS's names, not the backend's, so the space and the persisted JSON
# are the same shape for both. CatBoost calls these `iterations` and `depth`; the dataclass
# translates. Everything else a trial suggests travels in `extra_params` under the backend's
# own parameter name.
_BACKENDS: dict[BoostingBackend, type[LightGBMModel] | type[CatBoostingModel]] = {
    "lightgbm": LightGBMModel,
    "catboost": CatBoostingModel,
}

# LightGBM's space is the imputer's thirteen. `cat_smooth` and `min_data_per_group` are here
# because the static ids reach the model as pandas categoricals: they were dead knobs while the
# ids arrived as int64, and they are what keeps a 352-level `store_id` from overfitting now
# that LightGBM partitions its levels instead of carving numeric ranges.
_LGBM_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "n_estimators": (50, 3000),
    "max_depth": (6, 12),
    "num_leaves": (2, 1024),
    "min_child_samples": (1, 100),
    "min_data_per_group": (1, 500),
    "max_bin": (8, 255),
}
_LGBM_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.005, 0.3),
    "subsample": (0.85, 1.0),
    "colsample_bytree": (0.1, 1.0),
    "reg_alpha": (1e-8, 100.0),
    "reg_lambda": (1e-8, 100.0),
    "cat_smooth": (0.0, 50.0),
}

# CatBoost's space follows its own tuning guide, which orders the knobs: trees, learning rate
# and depth first, then l2 and border count. `depth` stops at 10 because the guide calls 4-10
# optimal, not because of the CPU's 16. `grow_policy` stays SymmetricTree (the guide: better
# quality and 10x faster), which is also why `min_data_in_leaf` and `max_leaves` are absent --
# they apply only to Depthwise/Lossguide. `one_hot_max_size` decides which of the seven static
# ids get one-hot encoded rather than target-statistic encoded; the guide recommends it for
# low-cardinality categoricals, and five of the seven have fewer than 20 levels.
_CATBOOST_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "n_estimators": (50, 3000),
    "max_depth": (4, 10),
    "border_count": (32, 254),
    "one_hot_max_size": (2, 255),
}
_CATBOOST_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.005, 0.3),
    "l2_leaf_reg": (0.01, 100.0),
    "random_strength": (0.0, 10.0),
    # Bayesian bootstrap intensity. Tuned instead of `subsample` because the two are mutually
    # exclusive: `subsample` needs bootstrap_type Bernoulli/MVS, and the default is Bayesian.
    "bagging_temperature": (0.0, 10.0),
}


def _suggest_lightgbm(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Suggest one LightGBM candidate."""
    subsample = trial.suggest_float("subsample", *_LGBM_FLOAT_BOUNDS["subsample"])
    return {
        "n_estimators": trial.suggest_int("n_estimators", *_LGBM_INT_BOUNDS["n_estimators"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *_LGBM_FLOAT_BOUNDS["learning_rate"], log=True
        ),
        "max_depth": trial.suggest_int("max_depth", *_LGBM_INT_BOUNDS["max_depth"]),
        "num_leaves": trial.suggest_int("num_leaves", *_LGBM_INT_BOUNDS["num_leaves"], log=True),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", *_LGBM_INT_BOUNDS["min_child_samples"]
        ),
        "min_data_per_group": trial.suggest_int(
            "min_data_per_group", *_LGBM_INT_BOUNDS["min_data_per_group"]
        ),
        "cat_smooth": trial.suggest_float("cat_smooth", *_LGBM_FLOAT_BOUNDS["cat_smooth"]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *_LGBM_FLOAT_BOUNDS["colsample_bytree"]
        ),
        "subsample": subsample,
        # LightGBM ignores `subsample` unless this says how often to resample.
        "subsample_freq": 1 if subsample < 1.0 else 0,
        "reg_alpha": trial.suggest_float("reg_alpha", *_LGBM_FLOAT_BOUNDS["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *_LGBM_FLOAT_BOUNDS["reg_lambda"], log=True
        ),
        "max_bin": trial.suggest_int("max_bin", *_LGBM_INT_BOUNDS["max_bin"]),
    }


def _suggest_catboost(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Suggest one CatBoost candidate."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", *_CATBOOST_INT_BOUNDS["n_estimators"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *_CATBOOST_FLOAT_BOUNDS["learning_rate"], log=True
        ),
        "max_depth": trial.suggest_int("max_depth", *_CATBOOST_INT_BOUNDS["max_depth"]),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", *_CATBOOST_FLOAT_BOUNDS["l2_leaf_reg"], log=True
        ),
        "border_count": trial.suggest_int("border_count", *_CATBOOST_INT_BOUNDS["border_count"]),
        "one_hot_max_size": trial.suggest_int(
            "one_hot_max_size", *_CATBOOST_INT_BOUNDS["one_hot_max_size"]
        ),
        "random_strength": trial.suggest_float(
            "random_strength", *_CATBOOST_FLOAT_BOUNDS["random_strength"]
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", *_CATBOOST_FLOAT_BOUNDS["bagging_temperature"]
        ),
    }


_SUGGEST: dict[BoostingBackend, Callable[[optuna.trial.BaseTrial], dict[str, Any]]] = {
    "lightgbm": _suggest_lightgbm,
    "catboost": _suggest_catboost,
}


def suggest_params(trial: optuna.trial.BaseTrial, backend: BoostingBackend) -> dict[str, Any]:
    """Suggest one candidate for `backend`.

    The three core keys carry the DATACLASS's names in every backend, so the space and the
    persisted JSON have one shape; the rest use the backend's own parameter names and travel
    to the estimator through `extra_params`.

    Typed against `BaseTrial` rather than `Trial` so a `FixedTrial` can replay a stored
    `best_params` through it. That replay is how the derived keys -- `subsample_freq`, which
    Optuna never suggested -- are recovered without duplicating the rule here and there.
    """
    return _SUGGEST[backend](trial)


@dataclass(frozen=True)
class Window:
    """One fit/score split of the supervised frame, with the horizon embargo applied.

    `fit_frame` ends `horizon` days before `eval_start`, so no training target reaches into a
    scored day: the last training origin's lead-time sum closes the day before scoring opens.
    Without that gap a training row's LABEL would contain the demand of the days it is about
    to be scored on -- the same embargo as invariants 11 and 12.
    """

    fit_frame: pd.DataFrame
    eval_frame: pd.DataFrame
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp

    def describe(self) -> str:
        """One-line summary for the run log."""
        return (
            f"entrena ≤{self.train_end.date()} ({thousands(len(self.fit_frame))} filas) · "
            f"puntúa {self.eval_start.date()}→{self.eval_end.date()} "
            f"({thousands(len(self.eval_frame))} filas)"
        )


def _window(frame: pd.DataFrame, eval_start: pd.Timestamp, horizon: int) -> Window:
    """Cut one window whose scoring starts at `eval_start`."""
    eval_end = eval_start + pd.Timedelta(days=EVAL_DAYS - 1)
    train_end = eval_start - pd.Timedelta(days=horizon)
    window = Window(
        fit_frame=frame.loc[frame["date"] <= train_end].copy(),
        eval_frame=frame.loc[frame["date"].between(eval_start, eval_end)].copy(),
        train_end=train_end,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    if window.fit_frame.empty or window.eval_frame.empty:
        raise ValueError(
            f"Window scoring {eval_start.date()}→{eval_end.date()} has "
            f"{len(window.fit_frame)} fit rows and {len(window.eval_frame)} scored rows, and "
            "both must be non-empty. The panel is too short for two nested windows at "
            f"horizon {horizon}: it needs {2 * EVAL_DAYS + 2 * horizon} dates of usable "
            "origins beyond the feature cold start."
        )
    return window


def split_windows(frame: pd.DataFrame, horizon: int) -> tuple[Window, Window]:
    """Cut the supervised frame into a selection window and a later validation window.

    Validation takes the last `EVAL_DAYS` of origins. Selection sits `EVAL_DAYS + horizon - 1`
    days earlier, which is the latest it can start while its own targets still close before
    validation opens: the gap matters because a selection target is a `horizon`-day sum, so
    without it the two objectives share demand days and the gate stops being independent of
    the search it is meant to judge.

    Returns:
        ``(selection, validation)``.
    """
    dates = np.sort(frame["date"].unique())
    validation_start = pd.Timestamp(dates[-EVAL_DAYS])
    selection_start = validation_start - pd.Timedelta(days=EVAL_DAYS + horizon - 1)
    return _window(frame, selection_start, horizon), _window(frame, validation_start, horizon)


def _bootstrap_draws(fit_frame: pd.DataFrame, seeds: list[int]) -> list[np.ndarray]:
    """One bootstrap resample of the training rows per seed, as positional indices.

    Resamples whole SERIES rather than individual rows: the origins of one series share lags,
    rolling windows and demand level, so treating them as independent understates the variance
    the gate is built to measure. `simulation/operations.py` resamples origins as clusters for
    the same reason.

    Sampling WITH replacement keeps the training set the same size, which is what separates
    this from the series-holdout design invariant 41 rejects: that one shrank the training set,
    and the measured gain against the defaults reverses with training size.

    Built once and reused by every trial, so two candidates differ by their hyperparameters
    rather than by resampling luck -- and so the gate's per-draw differences are paired.
    """
    positions = fit_frame.groupby("series_id", sort=False).indices
    series_ids = np.array(list(positions))
    draws = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(series_ids, size=len(series_ids), replace=True)
        draws.append(np.concatenate([positions[series_id] for series_id in sampled]))
    return draws


@dataclass(frozen=True)
class DrawSet:
    """Every bootstrap draw over one window, plus the window itself.

    One class for both roles, with no field saying which: nothing here behaves differently for
    selection than for validation. The role lives in the variable name and in what the caller
    does with it.

    Only the TRAINING rows are drawn. Every draw scores the window's full `eval_frame`, so
    draw `i` of one candidate and draw `i` of another differ by hyperparameters and by the
    bootstrap sample, never by which rows were scored.
    """

    window: Window
    draws: list[np.ndarray]
    seeds: list[int]

    @property
    def fit_rows(self) -> int:
        """Training rows per draw. Identical across draws: a bootstrap keeps the size."""
        return len(self.draws[0])

    def describe(self) -> str:
        """One-line summary for the run log."""
        return f"{len(self.draws)} extracciones · {self.window.describe()}"


def build_draw_sets(frame: pd.DataFrame, horizon: int, seed: int) -> tuple[DrawSet, DrawSet]:
    """Build the selection and validation draw sets from the supervised frame."""
    selection_window, validation_window = split_windows(frame, horizon)
    selection_seeds = [seed + i for i in range(N_SELECTION_DRAWS)]
    validation_seeds = [seed + _VALIDATION_SEED_OFFSET + i for i in range(N_VALIDATION_DRAWS)]
    return (
        DrawSet(
            window=selection_window,
            draws=_bootstrap_draws(selection_window.fit_frame, selection_seeds),
            seeds=selection_seeds,
        ),
        DrawSet(
            window=validation_window,
            draws=_bootstrap_draws(validation_window.fit_frame, validation_seeds),
            seeds=validation_seeds,
        ),
    )


def _draw_cost(
    draw: np.ndarray,
    window: Window,
    backend: BoostingBackend,
    params: dict[str, Any],
    settings: Settings,
    feature_columns: list[str],
    target_column: str,
) -> tuple[float, float]:
    """Mean simulated inventory cost of one hyperparameter set on one bootstrap draw.

    Cost, not MAE or Winkler: it is the criterion the champion is selected by, and the only
    one that prices the 4:1 asymmetry between a stockout and a unit of overstock. Point-error
    and cost rankings do invert, so optimizing the wrong one picks the wrong model.

    The candidate is scored THROUGH the conformal layer, and reusing the pipeline's own
    `_split_train_calibration` and `_train_conformal_model` rather than reimplementing them is
    the point: there must be one definition of the deployed decision path. Scoring the raw
    model instead was measuring a policy the system never runs. Conformal widens the outer
    quantiles by the calibrated radius, and `choose_order_quantity` interpolates the critical
    fractile between the median and the upper quantile, so the deployed order sits `0.75 * q_hat`
    above the raw one -- measured on the persisted champion, 4.57 units against a panel whose
    99th percentile of demand is 5.8. Optimizing without it selects for a different decision.
    """
    fit_frame = window.fit_frame.iloc[draw]
    base_model = _BACKENDS[backend](
        quantiles=settings.models.quantiles,
        random_seed=settings.project.random_seed,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        overstock_cost=settings.inventory.overstock_cost,
        stockout_cost=settings.inventory.stockout_cost,
        n_jobs=_TUNING_MODEL_THREADS,
        extra_params={k: v for k, v in params.items() if k not in CORE_PARAMS},
    )
    # Carved out of the resampled fit rows, with the same embargo the pipeline applies: the
    # calibration slice is the tail of the training calendar and the model trains on what
    # precedes it by at least `horizon` days.
    sub_train, calib, calib_groups = _split_train_calibration(fit_frame, settings)
    model = _train_conformal_model(
        base_model, sub_train, calib, calib_groups, feature_columns, settings
    )

    features = window.eval_frame.loc[:, feature_columns]
    eval_groups = (
        window.eval_frame["third_category_id"]
        if "third_category_id" in window.eval_frame.columns
        else None
    )
    quantiles = model.predict_quantiles(features, group_ids=eval_groups)
    levels = sorted(set(settings.models.quantiles))
    columns = list(quantiles)
    if len(columns) != len(levels):
        raise ValueError(
            f"predict_quantiles returned {len(columns)} columns for {len(levels)} configured "
            "quantiles; choose_order_quantity pairs them by position."
        )

    predictions = window.eval_frame.loc[:, ["date", "series_id"]].copy()
    predictions["y_true"] = window.eval_frame[target_column]
    predictions["y_pred"] = model.predict(features)
    for column, values in quantiles.items():
        predictions[column] = values
    predictions["order_quantity"] = choose_order_quantity(
        predictions=predictions,
        inventory_config=settings.inventory,
        quantile_columns=columns,
        quantile_levels=levels,
    )
    evaluated = attach_inventory_costs(predictions, settings.inventory)
    cost = float(evaluated["total_cost"].mean())
    winkler = winkler_score(
        predictions["y_true"],
        predictions[columns[0]],
        predictions[columns[-1]],
        alpha=0.2,
    )
    return cost, winkler


def draw_metrics(
    draw_set: DrawSet,
    backend: BoostingBackend,
    params: dict[str, Any],
    settings: Settings,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Cost and Winkler score of one hyperparameter set on each draw.

    Parallel over draws with each fit pinned to one thread, so the cores go to the 30
    independent fits rather than to one fit that cannot use them.
    """
    n_jobs = min(len(draw_set.draws), cpu_count())
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_draw_cost)(
            draw, draw_set.window, backend, params, settings, feature_columns, target_column
        )
        for draw in draw_set.draws
    )
    costs = np.asarray([r[0] for r in results], dtype=float)
    winklers = np.asarray([r[1] for r in results], dtype=float)
    return costs, winklers


def draw_costs(
    draw_set: DrawSet,
    backend: BoostingBackend,
    params: dict[str, Any],
    settings: Settings,
    feature_columns: list[str],
    target_column: str,
) -> np.ndarray:
    """Cost of one hyperparameter set on each draw, one value per draw."""
    costs, _ = draw_metrics(draw_set, backend, params, settings, feature_columns, target_column)
    return costs


def _open_study(backend: BoostingBackend, seed: int, models_dir: Path) -> optuna.Study:
    """Open (or resume) the multi-objective study for one backend.

    Persisted to SQLite so a search killed at hour four continues where it stopped, and keyed
    by backend so the two searches never share trials -- their spaces do not even have the same
    parameter names.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    models_dir.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=_N_STARTUP_TRIALS,
            multivariate=True,
        ),
        storage=f"sqlite:///{models_dir / 'forecasting_tuning_studies.db'}",
        study_name=f"forecasting_{backend}_v{_OBJECTIVE_VERSION}",
        load_if_exists=True,
    )


def run_search(
    selection: DrawSet,
    backend: BoostingBackend,
    settings: Settings,
    feature_columns: list[str],
    target_column: str,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Search `backend`'s hyperparameters on the selection window via multi-objective TPE."""
    study = _open_study(backend, seed, settings.models.models_dir)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = suggest_params(trial, backend)
        costs, winklers = draw_metrics(
            selection, backend, params, settings, feature_columns, target_column
        )
        return float(np.mean(costs)), float(np.mean(winklers))

    # `n_trials` is a TARGET, not an increment: on a resumed study Optuna would otherwise run
    # that many MORE trials on top of the ones already stored.
    already_done = len(study.get_trials(deepcopy=False, states=_FINISHED_STATES))
    trials_todo = max(0, n_trials - already_done)
    if already_done:
        fields(logger, {"reanudado": f"{already_done} trials ya en el estudio"})
    if not trials_todo:
        return study

    started = time.monotonic()
    table = Table(
        logger,
        {
            "trial": len(f"{n_trials}/{n_trials}"),
            "coste": 8,
            "winkler": 8,
            "mejor_coste": 11,
            "mejor_winkler": 13,
            "s/trial": 7,
            "pasado": 6,
            "queda": 6,
        },
    )

    def report_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = trial.number + 1
        if done % _PROGRESS_EVERY_TRIALS and done != n_trials:
            return
        pace = pace_seconds(study, _N_STARTUP_TRIALS)
        cost_val = trial.values[0] if trial.values else None
        winkler_val = trial.values[1] if trial.values and len(trial.values) > 1 else None
        best_candidate = (
            min(study.best_trials, key=lambda t: t.values[0]) if study.best_trials else None
        )
        best_cost = best_candidate.values[0] if best_candidate else float("nan")
        best_winkler = (
            best_candidate.values[1]
            if best_candidate and len(best_candidate.values) > 1
            else float("nan")
        )
        table.row(
            {
                "trial": f"{done}/{n_trials}",
                "coste": "-" if cost_val is None else f"{cost_val:.4f}",
                "winkler": "-" if winkler_val is None else f"{winkler_val:.4f}",
                "mejor_coste": f"{best_cost:.4f}",
                "mejor_winkler": f"{best_winkler:.4f}",
                "s/trial": "-" if pace is None else f"{pace:.0f}",
                "pasado": hhmm(time.monotonic() - started),
                "queda": "-" if pace is None else hhmm(pace * max(n_trials - done, 0)),
            }
        )

    study.optimize(objective, n_trials=trials_todo, callbacks=[report_progress])
    return study


@dataclass(frozen=True)
class Verdict:
    """What the validation draws say about the winner, and whether it earns the file."""

    best_cost: float
    default_cost: float
    improvement_pct: float
    improvement_ci95: tuple[float, float]
    incumbent_cost: float | None
    incumbent_ci95: tuple[float, float] | None
    beats_incumbent: bool | None

    @property
    def beats_default(self) -> bool:
        """True only when the whole interval sits below zero.

        The point estimate is not enough: the between-draw spread dwarfs the difference
        between candidates, so a mean that merely happens to land below zero is what a coin
        flip looks like. Invariant 41 measured a search reporting -1.4% in-sample that came
        out at -0.32% with the interval crossing zero on fresh draws.
        """
        return self.improvement_ci95[1] < 0.0

    @property
    def should_persist(self) -> bool:
        """Both gates: beats the defaults, and does not lose to the incumbent on disk."""
        return self.beats_default and self.beats_incumbent is not False


def judge(
    validation: DrawSet,
    backend: BoostingBackend,
    winner_params: dict[str, Any],
    settings: Settings,
    feature_columns: list[str],
    target_column: str,
    incumbent_params: dict[str, Any] | None = None,
) -> Verdict:
    """Score the winner, the untuned defaults and any incumbent on the SAME validation draws.

    Same draws for all three, so every comparison is paired: the between-draw variance that
    dominates each cost is shared rather than resampled, which is what makes two numbers
    comparable at all.
    """
    winner = draw_costs(
        validation, backend, winner_params, settings, feature_columns, target_column
    )
    defaults = draw_costs(
        validation, backend, default_params(settings), settings, feature_columns, target_column
    )
    best_cost = float(np.mean(winner))
    default_cost = float(np.mean(defaults))
    ci_lo, ci_hi = mean_ci95(winner - defaults)

    incumbent_cost: float | None = None
    incumbent_ci: tuple[float, float] | None = None
    beats_incumbent: bool | None = None
    if incumbent_params is not None:
        incumbent = draw_costs(
            validation, backend, incumbent_params, settings, feature_columns, target_column
        )
        incumbent_cost = float(np.mean(incumbent))
        incumbent_ci = mean_ci95(winner - incumbent)
        # Decided on the MEAN, not on the interval: this gate only has to break a tie between
        # two candidates that both already cleared the defaults, and refusing to replace an
        # incumbent whenever the interval straddles zero would freeze the file forever.
        beats_incumbent = bool(np.mean(winner - incumbent) < 0.0)

    return Verdict(
        best_cost=best_cost,
        default_cost=default_cost,
        improvement_pct=(best_cost - default_cost) / default_cost * 100.0,
        improvement_ci95=(ci_lo, ci_hi),
        incumbent_cost=incumbent_cost,
        incumbent_ci95=incumbent_ci,
        beats_incumbent=beats_incumbent,
    )


PARETO_ARTIFACT = "tuning_pareto.csv"


def build_pareto_frame(study: optuna.Study, selected_trial: int) -> pd.DataFrame:
    """One row per scored trial, named as the dashboard reads it.

    `pinball_loss`/`winkler_score` are the column names the web layer consumes; the search
    itself calls them cost and winkler. Renaming here rather than at the reader keeps the
    artifact self-describing for anyone who opens the CSV without the view.
    """
    records = [
        {
            "trial_number": trial.number,
            "pinball_loss": float(trial.values[0]),
            "winkler_score": float(trial.values[1]),
            "is_on_front": trial in study.best_trials,
            "is_selected": trial.number == selected_trial,
            **trial.params,
        }
        for trial in study.trials
        if trial.values and len(trial.values) >= 2
    ]
    return pd.DataFrame(records)


def _build_metadata(
    verdict: Verdict,
    study: optuna.Study,
    winner_params: dict[str, Any],
    selection: DrawSet,
    validation: DrawSet,
    settings: Settings,
    backend: BoostingBackend,
    n_trials: int,
    seed: int,
    n_series: int,
    incumbent_tuned_on: dict[str, Any] | None,
) -> ForecastingTuningMetadata:
    """Assemble the decision record, written whether or not the winner is persisted."""
    min_cost = min((t.values[0] for t in study.best_trials), default=0.0)
    return ForecastingTuningMetadata(
        backend=backend,
        n_trials_requested=n_trials,
        best_cost_selection=float(min_cost),
        best_cost_validation=verdict.best_cost,
        default_cost_validation=verdict.default_cost,
        improvement_pct=verdict.improvement_pct,
        improvement_ci95=list(verdict.improvement_ci95),
        incumbent_cost_validation=verdict.incumbent_cost,
        incumbent_ci95=None if verdict.incumbent_ci95 is None else list(verdict.incumbent_ci95),
        beats_incumbent=verdict.beats_incumbent,
        incumbent_tuned_on=incumbent_tuned_on,
        persisted=verdict.should_persist,
        n_selection_draws=len(selection.draws),
        n_validation_draws=len(validation.draws),
        selection_seeds=selection.seeds,
        validation_seeds=validation.seeds,
        selection_train_end=selection.window.train_end.date().isoformat(),
        selection_eval_start=selection.window.eval_start.date().isoformat(),
        selection_eval_end=selection.window.eval_end.date().isoformat(),
        validation_train_end=validation.window.train_end.date().isoformat(),
        validation_eval_start=validation.window.eval_start.date().isoformat(),
        validation_eval_end=validation.window.eval_end.date().isoformat(),
        n_selection_fit_rows=selection.fit_rows,
        n_validation_fit_rows=validation.fit_rows,
        n_selection_eval_rows=len(selection.window.eval_frame),
        n_validation_eval_rows=len(validation.window.eval_frame),
        n_series=n_series,
        horizon=settings.dataset.horizon,
        lags=list(settings.features.lags),
        rolling_windows=list(settings.features.rolling_windows),
        seed=seed,
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
        best_params=winner_params,
    )


def _log_study_to_mlflow(
    study: optuna.Study,
    metadata: ForecastingTuningMetadata,
    metadata_path: Path,
    params_path: Path,
    panel: pd.DataFrame,
    panel_source: Path,
    config_hash: str,
    config_path: Path | None,
) -> None:
    """Record one completed search in MLflow: winner, validation verdict, and its artifacts."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    # Both filters are for MLflow talking about itself: it registers LocalArtifactDatasetSource
    # twice, so resolving a local path reports it as ambiguous while resolving it correctly, and
    # it warns that integer columns cannot hold missing values -- advice about enforcing a MODEL
    # signature at inference time. No model is logged here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted")
        warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")
        dataset = from_pandas(panel, source=str(panel_source), name="forecasting_train_panel")
        # The schema is computed lazily and it is the access that warns, so force it inside the
        # filter rather than letting `log_input` trip it further down.
        _ = dataset.schema

    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_input(dataset, context="training")
        mlflow.log_params(metadata.best_params)
        mlflow.log_params(
            {
                "backend": metadata.backend,
                "n_trials": metadata.n_trials_requested,
                "seed": metadata.seed,
                "n_series": metadata.n_series,
                "horizon": metadata.horizon,
                "lags": metadata.lags,
                "rolling_windows": metadata.rolling_windows,
                "n_selection_draws": metadata.n_selection_draws,
                "n_validation_draws": metadata.n_validation_draws,
                "selection_eval": f"{metadata.selection_eval_start}..{metadata.selection_eval_end}",
                "validation_eval": (
                    f"{metadata.validation_eval_start}..{metadata.validation_eval_end}"
                ),
            }
        )
        # Row counts are metrics rather than params despite measuring nothing: MLflow stores
        # params as strings, so a `params.n_selection_fit_rows > 10000` filter would compare
        # text. Training size is the variable invariant 41 makes decisive, so it has to sort as
        # a number.
        mlflow.log_metrics(
            {
                "cost_selection_best": metadata.best_cost_selection,
                "cost_validation_tuned": metadata.best_cost_validation,
                "cost_validation_default": metadata.default_cost_validation,
                "improvement_pct": metadata.improvement_pct,
                "improvement_ci95_low": metadata.improvement_ci95[0],
                "improvement_ci95_high": metadata.improvement_ci95[1],
                "n_selection_fit_rows": metadata.n_selection_fit_rows,
                "n_validation_fit_rows": metadata.n_validation_fit_rows,
                "n_selection_eval_rows": metadata.n_selection_eval_rows,
                "n_validation_eval_rows": metadata.n_validation_eval_rows,
            }
        )
        # Only when there was an incumbent. A run with none logs nothing here rather than a
        # sentinel: absent means "no incumbent", where a -1 is a number someone eventually
        # averages.
        if metadata.incumbent_cost_validation is not None and metadata.incumbent_ci95 is not None:
            mlflow.log_metrics(
                {
                    "cost_validation_incumbent": metadata.incumbent_cost_validation,
                    "incumbent_ci95_low": metadata.incumbent_ci95[0],
                    "incumbent_ci95_high": metadata.incumbent_ci95[1],
                }
            )

        # The three branches of the persist decision as one searchable value. Worth its own tag
        # because the gates do not decide alike -- the defaults gate needs the whole interval
        # below zero, the incumbent gate goes on the mean -- so a run that replaced its
        # incumbent on an interval straddling zero reads as a bug when reassembled by hand.
        # Order matters: a run that loses to both is reported against the defaults, the branch
        # that also drops the superseded block.
        if metadata.persisted:
            outcome = "persisted"
        elif metadata.improvement_ci95[1] >= 0.0:
            outcome = "no_gain_over_defaults"
        else:
            outcome = "lost_to_incumbent"

        mlflow.set_tags(
            {
                "backend": metadata.backend,
                "persisted": str(metadata.persisted),
                "beats_incumbent": str(metadata.beats_incumbent),
                "outcome": outcome,
                "git_commit": metadata.git_commit or "unknown",
                # Two answers to "which configuration was this". The hash is the identity -- it
                # covers the RESOLVED settings, so a hand-edited YAML moves it. The path is what
                # a human recognises in the UI. They differ exactly when someone edited the file.
                "config_hash": config_hash,
                "config_path": str(config_path) if config_path is not None else "unknown",
            }
        )

        # Two views of the same trials. The metric series draws the convergence curve; the
        # nested run per trial carries that trial's own hyperparameters.
        best_cost_trial = min(study.best_trials, key=lambda t: t.values[0])
        mlflow.log_param("best_trial_number", best_cost_trial.number)
        mlflow.log_param("n_pareto_trials", len(study.best_trials))
        best_cost_so_far = float("inf")
        best_winkler_so_far = float("inf")
        for trial in study.trials:
            if not trial.values or len(trial.values) < 2:
                continue
            cost_val = float(trial.values[0])
            winkler_val = float(trial.values[1])
            best_cost_so_far = min(best_cost_so_far, cost_val)
            best_winkler_so_far = min(best_winkler_so_far, winkler_val)
            mlflow.log_metric("trial_cost", cost_val, step=trial.number)
            mlflow.log_metric("trial_cost_best_so_far", best_cost_so_far, step=trial.number)
            mlflow.log_metric("trial_winkler", winkler_val, step=trial.number)
            mlflow.log_metric("trial_winkler_best_so_far", best_winkler_so_far, step=trial.number)
            # Zero-padded so the UI's alphabetical list runs 0001, 0002, ... not 1, 10, 100.
            with mlflow.start_run(run_name=f"trial-{trial.number:04d}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("cost", cost_val)
                mlflow.log_metric("winkler", winkler_val)
                mlflow.log_metric("is_on_front", 1.0 if trial in study.best_trials else 0.0)
                if trial.duration is not None:
                    mlflow.log_metric("duration_seconds", trial.duration.total_seconds())

        mlflow.log_artifact(str(metadata_path))
        if metadata.persisted and params_path.exists():
            mlflow.log_artifact(str(params_path))

        # The front goes into the RUN, not a path relative to the cwd: it is what the
        # dashboard's Pareto view reads, and a run store is the only place it can find it.
        pareto_df = build_pareto_frame(study, best_cost_trial.number)
        if not pareto_df.empty:
            pareto_path = metadata_path.parent / PARETO_ARTIFACT
            front_png = metadata_path.parent / f"pareto_front_{metadata.backend}.png"
            pareto_df.to_csv(pareto_path, index=False)
            render_pareto_front(pareto_df, front_png)
            mlflow.log_artifact(str(pareto_path))
            mlflow.log_artifact(str(front_png))
            pareto_path.unlink()
            front_png.unlink()


def _track(
    study: optuna.Study,
    metadata: ForecastingTuningMetadata,
    metadata_path: Path,
    params_path: Path,
    panel: pd.DataFrame,
    settings: Settings,
    config_path: Path | None,
) -> None:
    """Log to MLflow, and never let that take the search down with it.

    Everything the run produced is already on disk by this point. A tracking store that is
    locked, missing or out of space would otherwise lose two hours of search at its last step,
    having lost nothing but the record. Broad on purpose: no failure to write a log is worth
    the run.
    """
    try:
        _log_study_to_mlflow(
            study=study,
            metadata=metadata,
            metadata_path=metadata_path,
            params_path=params_path,
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


def tune_forecasting_models(
    settings: Settings,
    n_trials: int | None = None,
    seed: int | None = None,
    config_path: Path | None = None,
) -> Path:
    """Search one backend's hyperparameters and persist the winner.

    Returns:
        The path of the written params file when the winner cleared the gates, otherwise
        the path of the metadata file recording why it did not.
    """
    n_trials = n_trials if n_trials is not None else settings.models.tuning_trials
    seed = seed if seed is not None else settings.project.random_seed
    backend = settings.models.tuning_backend

    # The model classes log the critical fractile once per fit, which is right for a run that
    # fits ten models and useless for one that fits nine thousand: the value is constant, read
    # off the inventory costs. Raised to WARNING for this process only, so an experiment run
    # still reports it.
    logging.getLogger("retail_forecasting.models").setLevel(logging.WARNING)

    rule(logger, f"tuning de forecasting · {backend}")
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    supervised_frame, feature_metadata = build_supervised_frame(
        panel=label_all_regimes(panel),
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )
    # Never None in supervised mode; the annotation covers the inference frames too.
    assert feature_metadata.target_column is not None
    selection, validation = build_draw_sets(supervised_frame, settings.dataset.horizon, seed)

    critical_fractile = settings.inventory.stockout_cost / (
        settings.inventory.stockout_cost + settings.inventory.overstock_cost
    )
    fields(
        logger,
        {
            "panel": f"{panel['series_id'].nunique()} series, {thousands(len(panel))} filas",
            "supervisado": (
                f"{thousands(len(supervised_frame))} filas · "
                f"{len(feature_metadata.feature_columns)} features"
            ),
            "selección": selection.describe(),
            "validación": validation.describe(),
            "semillas": (f"selección {seed}+ · validación {seed + _VALIDATION_SEED_OFFSET}+"),
            "objetivo": f"coste logístico simulado · fractil crítico {critical_fractile:.2f}",
            "defaults a batir": (
                f"n_estimators={settings.models.n_estimators} "
                f"learning_rate={settings.models.learning_rate} "
                f"max_depth={settings.models.max_depth}"
            ),
            "presupuesto": f"{n_trials} trials · semilla {seed}",
        },
    )
    if config_path is not None:
        fields(logger, {"config": str(config_path)})

    study = run_search(
        selection=selection,
        backend=backend,
        settings=settings,
        feature_columns=feature_metadata.feature_columns,
        target_column=feature_metadata.target_column,
        n_trials=n_trials,
        seed=seed,
    )
    winner_trial = min(study.best_trials, key=lambda t: t.values[0])
    winner_params = suggest_params(optuna.trial.FixedTrial(winner_trial.params), backend)
    rule(logger, "mejor trial (frontera de pareto)")
    fields(
        logger,
        {
            "coste selección": f"{winner_trial.values[0]:.4f} (in-sample)",
            "winkler selección": f"{winner_trial.values[1]:.4f}",
            "trials en frente pareto": f"{len(study.best_trials)}/{len(study.trials)}",
            "parámetros": ", ".join(f"{k}={v}" for k, v in sorted(winner_params.items())),
        },
    )

    models_dir = settings.models.models_dir
    params_path = models_dir / settings.models.forecasting_params_filename
    incumbent = read_tuned_params(params_path, backend)
    incumbent_params = None if incumbent is None else incumbent.get("params")
    incumbent_tuned_on = None if incumbent is None else incumbent.get("tuned_on")
    if incumbent_params is not None:
        fields(
            logger,
            {
                "titular en disco": ", ".join(
                    f"{k}={v}" for k, v in sorted(incumbent_params.items())
                )
            },
        )

    logger.info("")
    logger.info(
        "Puntuando al ganador, a los defaults y al titular sobre las extracciones de validación"
    )
    verdict = judge(
        validation=validation,
        backend=backend,
        winner_params=winner_params,
        settings=settings,
        feature_columns=feature_metadata.feature_columns,
        target_column=feature_metadata.target_column,
        incumbent_params=incumbent_params,
    )
    metadata = _build_metadata(
        verdict=verdict,
        study=study,
        winner_params=winner_params,
        selection=selection,
        validation=validation,
        settings=settings,
        backend=backend,
        n_trials=n_trials,
        seed=seed,
        n_series=int(panel["series_id"].nunique()),
        incumbent_tuned_on=incumbent_tuned_on,
    )

    metadata_path = models_dir / f"forecasting_{backend}_tuning_metadata.json"
    metadata_path.write_text(json.dumps(metadata.model_dump(), indent=2), encoding="utf-8")

    ci_lo, ci_hi = verdict.improvement_ci95
    rule(logger, "veredicto")
    fields(
        logger,
        {
            "ganador": f"coste {verdict.best_cost:.4f} en validación",
            "defaults": f"coste {verdict.default_cost:.4f} en validación",
            "mejora": f"{verdict.improvement_pct:+.2f}%",
            "IC95": f"[{ci_lo:+.4f}, {ci_hi:+.4f}] sobre la diferencia pareada, en coste",
            "veredicto": "bate a los defaults" if verdict.beats_default else "NO los bate",
        },
    )
    if verdict.incumbent_ci95 is not None:
        inc_lo, inc_hi = verdict.incumbent_ci95
        decisive = inc_hi < 0.0 or inc_lo > 0.0
        fields(
            logger,
            {
                "vs titular": (
                    f"{verdict.incumbent_cost:.4f} · IC95 [{inc_lo:+.4f}, {inc_hi:+.4f}] "
                    f"({'decisivo' if decisive else 'indistinguible, decidido por la media'}) -> "
                    f"{'lo sustituye' if verdict.beats_incumbent else 'NO lo sustituye'}"
                )
            },
        )

    if not verdict.beats_default:
        reason = (
            "es peor que los defaults y el intervalo no toca el cero"
            if ci_lo > 0
            else "no se distingue de cero en las extracciones de validación"
        )
        logger.info("  NO persistido: la mejora %s, el pipeline sigue con los defaults", reason)
        if _drop_backend_block(params_path, backend):
            fields(logger, {"retirado": f"{params_path} · bloque {backend}"})
        fields(logger, {"decisión": metadata_path})
        _track(study, metadata, metadata_path, params_path, panel, settings, config_path)
        return metadata_path

    if verdict.beats_incumbent is False:
        logger.info(
            "  NO persistido: bate a los defaults pero no al titular en disco, que se mantiene"
        )
        fields(logger, {"decisión": metadata_path})
        _track(study, metadata, metadata_path, params_path, panel, settings, config_path)
        return metadata_path

    _write_backend_block(
        params_path,
        backend,
        {
            "params": winner_params,
            "tuned_on": {
                "n_series": metadata.n_series,
                "train_rows": metadata.n_selection_fit_rows,
                "horizon": metadata.horizon,
                "lags": metadata.lags,
                "rolling_windows": metadata.rolling_windows,
            },
            "gate": {
                "improvement_pct": metadata.improvement_pct,
                "improvement_ci95": metadata.improvement_ci95,
                "n_validation_draws": metadata.n_validation_draws,
                "created_at": metadata.created_at,
                "git_commit": metadata.git_commit,
            },
        },
    )
    fields(logger, {"persistido": f"{params_path} · bloque {backend}", "decisión": metadata_path})
    _track(study, metadata, metadata_path, params_path, panel, settings, config_path)
    return params_path


__all__ = [
    "tune_forecasting_models",
    "read_tuned_params",
    "run_search",
    "draw_costs",
    "suggest_params",
    "default_params",
    "build_draw_sets",
    "split_windows",
    "DrawSet",
    "Window",
    "EVAL_DAYS",
]
