"""GPSampler vs TPESampler on the real imputation objective, paired by seed.

Resumable by construction: every study lives in one SQLite file with `load_if_exists`, and a
finished search appends its verdict to a JSONL that a restart reads back and skips. Killing this
mid-run and relaunching the identical command continues from the last completed trial.

Writes only inside its own directory. Touches no params file, no models/, no mlflow.db.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import torch

torch.set_num_threads(1)

from retail_forecasting.config import load_config
from retail_forecasting.data.censorship import DEFAULT_SUPERVISED_LGBM_PARAMS
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.forecasting.imputation_tuning import (
    _FLOAT_BOUNDS,
    _INT_BOUNDS,
    _N_STARTUP_TRIALS,
    _VALIDATION_SEED_OFFSET,
    N_SELECTION_HOLDOUTS,
    N_VALIDATION_HOLDOUTS,
    Holdout,
    _build_holdout_set,
    _holdout_maes,
    _mean_ci95,
    _split_temporal_windows,
)

HERE = Path(__file__).parent  # this script lives in the directory it writes to
HERE.mkdir(exist_ok=True)
RESULTS = HERE / "results.jsonl"
STORAGE = f"sqlite:///{HERE / 'studies.db'}"

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/imputation_tuning.yaml"
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42, 1234, 7]


def build_sampler(kind: str, seed: int) -> optuna.samplers.BaseSampler:
    if kind == "gp":
        return optuna.samplers.GPSampler(
            seed=seed, n_startup_trials=_N_STARTUP_TRIALS, deterministic_objective=True
        )
    return optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=_N_STARTUP_TRIALS, multivariate=True
    )


def make_objective(draws: list[Holdout]) -> Callable[[optuna.Trial], float]:
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
        return float(np.mean(_holdout_maes(draws, params)))

    return objective


def full_params(best: dict[str, float]) -> dict[str, int | float]:
    out = dict(best)
    out["subsample_freq"] = 1 if float(best["subsample"]) < 1.0 else 0
    return out


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = set()
    if RESULTS.exists():
        done = {
            (r["sampler"], r["seed"])
            for r in (json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip())
        }
        print(f"resuming: {len(done)} searches already recorded", flush=True)

    settings = load_config(CONFIG)
    panel = load_prepared_panel(
        dataset_config=settings.dataset, preprocessing_config=settings.preprocessing, split="train"
    )
    selection_mask, validation_mask = _split_temporal_windows(panel)
    print(
        f"{CONFIG}: {panel['series_id'].nunique()} series, {len(panel):,} rows, "
        f"{N_TRIALS} trials x {len(SEEDS)} seeds x 2 samplers",
        flush=True,
    )

    default_cache: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        selection = _build_holdout_set(
            panel, [seed + i for i in range(N_SELECTION_HOLDOUTS)], selection_mask
        )
        validation = _build_holdout_set(
            panel,
            [seed + _VALIDATION_SEED_OFFSET + i for i in range(N_VALIDATION_HOLDOUTS)],
            validation_mask,
        )
        if seed not in default_cache:
            default_cache[seed] = _holdout_maes(
                validation.draws, dict(DEFAULT_SUPERVISED_LGBM_PARAMS)
            )
        default_maes = default_cache[seed]

        for kind in ("gp", "tpe"):
            if (kind, seed) in done:
                print(f"  skip {kind} seed={seed} (already recorded)", flush=True)
                continue
            name = f"{kind}_seed{seed}_n{N_TRIALS}"
            study = optuna.create_study(
                direction="minimize",
                sampler=build_sampler(kind, seed),
                storage=STORAGE,
                study_name=name,
                load_if_exists=True,
            )
            complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not complete:
                study.enqueue_trial(dict(DEFAULT_SUPERVISED_LGBM_PARAMS))
            remaining = N_TRIALS - len(complete)
            print(f"  {kind} seed={seed}: {len(complete)} done, {remaining} to go", flush=True)
            started = time.perf_counter()
            if remaining > 0:
                study.optimize(make_objective(selection.draws), n_trials=remaining)
            elapsed = time.perf_counter() - started

            tuned_maes = _holdout_maes(validation.draws, full_params(study.best_params))
            delta = tuned_maes - default_maes
            lo, hi = _mean_ci95(delta)
            record = {
                "sampler": kind,
                "seed": seed,
                "n_trials": N_TRIALS,
                "best_mae_selection": float(study.best_value),
                "best_trial_number": study.best_trial.number,
                "mae_validation_tuned": float(np.mean(tuned_maes)),
                "mae_validation_default": float(np.mean(default_maes)),
                "improvement_pct": float(
                    (np.mean(tuned_maes) - np.mean(default_maes)) / np.mean(default_maes) * 100
                ),
                "ci95_low": lo,
                "ci95_high": hi,
                "beats_default": bool(hi < 0.0),
                "seconds": elapsed,
                "best_params": full_params(study.best_params),
            }
            with RESULTS.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(
                f"    -> selection {record['best_mae_selection']:.4f} | "
                f"validation {record['mae_validation_tuned']:.4f} "
                f"({record['improvement_pct']:+.2f}%, CI [{lo:+.4f}, {hi:+.4f}]) | "
                f"{elapsed / 60:.1f} min",
                flush=True,
            )

    print(f"\nALL DONE -- {RESULTS}", flush=True)


main()
