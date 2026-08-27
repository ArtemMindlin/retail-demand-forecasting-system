"""The forecasting hyperparameter search: windows, draws, gates and the params store."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from retail_forecasting.config import DatasetConfig, FeatureConfig, ModelConfig, Settings
from retail_forecasting.features.engineering import build_supervised_frame
from retail_forecasting.forecasting.forecasting_tuning import (
    EVAL_DAYS,
    N_SELECTION_DRAWS,
    N_VALIDATION_DRAWS,
    Verdict,
    _bootstrap_draws,
    build_draw_sets,
    draw_costs,
    split_windows,
    suggest_params,
)
from retail_forecasting.forecasting.tuned_params import (
    CORE_PARAMS,
    _drop_backend_block,
    _write_backend_block,
    default_params,
    read_tuned_params,
    resolve_backend_params,
)
from tests import make_synthetic_panel

HORIZON = 7
FEATURES = FeatureConfig(lags=[1, 7, 14], rolling_windows=[7, 14])


def supervised(num_series: int = 6, num_days: int = 60) -> tuple[pd.DataFrame, list[str], str]:
    """A supervised frame long enough for two windows plus the gap between them."""
    frame, metadata = build_supervised_frame(
        make_synthetic_panel(num_series=num_series, num_days=num_days),
        FEATURES,
        horizon=HORIZON,
    )
    assert metadata.target_column is not None
    return frame, metadata.feature_columns, metadata.target_column


def settings_for(models_dir: Path) -> Settings:
    return Settings(
        dataset=DatasetConfig(horizon=HORIZON),
        features=FEATURES,
        models=ModelConfig(models_dir=models_dir, n_estimators=20, max_depth=4),
    )


# --------------------------------------------------------------------------- windows


def test_windows_are_placed_at_the_end_of_the_calendar() -> None:
    frame, _, _ = supervised()
    selection, validation = split_windows(frame, HORIZON)

    origins = sorted(pd.to_datetime(frame["date"]).unique())
    assert validation.eval_end == pd.Timestamp(origins[-1]), "validation takes the last origins"
    assert validation.eval_start == pd.Timestamp(origins[-EVAL_DAYS])
    for window in (selection, validation):
        assert (window.eval_end - window.eval_start).days == EVAL_DAYS - 1


def test_each_window_embargoes_the_horizon_before_scoring_opens() -> None:
    """The embargo is what keeps a training LABEL out of the days it is scored on.

    A training origin's target is a `horizon`-day sum, so the last one must close the day
    before scoring opens -- one day later and the label already contains scored demand.
    """
    frame, _, target = supervised()
    for window in split_windows(frame, HORIZON):
        assert window.train_end == window.eval_start - pd.Timedelta(days=HORIZON)
        last_label_covers_until = window.train_end + pd.Timedelta(days=HORIZON - 1)
        assert last_label_covers_until == window.eval_start - pd.Timedelta(days=1)
        assert window.fit_frame["date"].max() <= window.train_end
        assert window.eval_frame["date"].min() >= window.eval_start
        assert window.eval_frame[target].notna().all()


def test_the_two_windows_do_not_share_a_day_of_demand() -> None:
    """Selection targets must close before validation scoring opens.

    Not just disjoint scored rows: a selection target is a `horizon`-day sum, so without the
    gap the two objectives are computed from overlapping demand and the gate stops being
    independent of the search it exists to judge.
    """
    frame, _, _ = supervised()
    selection, validation = split_windows(frame, HORIZON)

    selection_target_ends = selection.eval_end + pd.Timedelta(days=HORIZON - 1)
    assert selection_target_ends < validation.eval_start
    assert selection.eval_end < validation.eval_start


def test_a_panel_too_short_for_two_windows_raises() -> None:
    frame, _, _ = supervised(num_series=2, num_days=40)
    with pytest.raises(ValueError, match="too short for two nested windows"):
        split_windows(frame, HORIZON)


# --------------------------------------------------------------------------- draws


def test_bootstrap_draws_keep_the_training_size() -> None:
    """The size is what separates a bootstrap from the series-holdout invariant 41 rejects.

    That design shrank the training set, and the measured gain against the defaults reverses
    with training size, so a draw that trained on less would be answering another question.
    """
    frame, _, _ = supervised()
    selection, _ = split_windows(frame, HORIZON)
    draws = _bootstrap_draws(selection.fit_frame, [1, 2, 3])

    assert len(draws) == 3
    assert {len(draw) for draw in draws} == {len(selection.fit_frame)}


def test_bootstrap_draws_resample_whole_series() -> None:
    """Series, not rows: the origins of one series share lags, windows and demand level."""
    frame, _, _ = supervised()
    selection, _ = split_windows(frame, HORIZON)
    rows_per_series = selection.fit_frame.groupby("series_id", sort=False).size()
    assert rows_per_series.nunique() == 1, "the fixture gives every series the same rows"

    drawn = selection.fit_frame.iloc[_bootstrap_draws(selection.fit_frame, [7])[0]]
    counts = drawn.groupby("series_id", sort=False).size()
    # Sampling with replacement takes each series a whole number of times, never a fraction:
    # a draw that split a series would be resampling rows, which is what this rules out.
    assert set(counts % rows_per_series.iloc[0]) == {0}
    # And it is a resample, not a copy: some series drop out and the survivors repeat to keep
    # the size. Not asserted through the counts, which are all equal whenever every survivor
    # happens to be taken the same number of times.
    assert drawn["series_id"].nunique() < selection.fit_frame["series_id"].nunique()
    assert len(drawn) == len(selection.fit_frame)


def test_bootstrap_draws_are_deterministic_per_seed_and_differ_across_seeds() -> None:
    """Every trial must see the SAME draws, or two candidates differ by resampling luck."""
    frame, _, _ = supervised()
    selection, _ = split_windows(frame, HORIZON)

    assert np.array_equal(
        _bootstrap_draws(selection.fit_frame, [11])[0],
        _bootstrap_draws(selection.fit_frame, [11])[0],
    )
    assert not np.array_equal(
        _bootstrap_draws(selection.fit_frame, [11])[0],
        _bootstrap_draws(selection.fit_frame, [12])[0],
    )


def test_draw_sets_use_disjoint_seeds() -> None:
    frame, _, _ = supervised()
    selection, validation = build_draw_sets(frame, HORIZON, seed=42)

    assert len(selection.draws) == N_SELECTION_DRAWS
    assert len(validation.draws) == N_VALIDATION_DRAWS
    assert set(selection.seeds).isdisjoint(validation.seeds)


# --------------------------------------------------------------------------- search space


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("lightgbm", 13), ("catboost", 8)],
)
def test_the_search_space_has_the_documented_size(backend: str, expected: int) -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    seen: dict[str, object] = {}

    def objective(trial: optuna.Trial) -> float:
        seen.update(suggest_params(trial, backend))  # type: ignore[arg-type]
        return 0.0

    study.optimize(objective, n_trials=1)

    assert len(seen) == expected
    assert set(CORE_PARAMS) <= set(seen), "the three core keys use the dataclass's names"


def test_lightgbm_subsample_freq_follows_subsample() -> None:
    """LightGBM ignores `subsample` unless told how often to resample."""
    params = suggest_params(
        optuna.trial.FixedTrial(
            {
                "subsample": 1.0,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "num_leaves": 31,
                "min_child_samples": 20,
                "min_data_per_group": 100,
                "cat_smooth": 10.0,
                "colsample_bytree": 0.8,
                "reg_alpha": 1e-3,
                "reg_lambda": 1e-3,
                "max_bin": 255,
            }
        ),
        "lightgbm",
    )
    assert params["subsample_freq"] == 0


def test_default_params_are_the_config_values() -> None:
    settings = settings_for(Path("models"))
    assert default_params(settings) == {
        "n_estimators": settings.models.n_estimators,
        "learning_rate": settings.models.learning_rate,
        "max_depth": settings.models.max_depth,
    }


# --------------------------------------------------------------------------- the gates


def _verdict(ci: tuple[float, float], beats_incumbent: bool | None = None) -> Verdict:
    return Verdict(
        best_cost=1.0,
        default_cost=2.0,
        improvement_pct=-50.0,
        improvement_ci95=ci,
        incumbent_cost=None if beats_incumbent is None else 1.5,
        incumbent_ci95=None if beats_incumbent is None else (-1.0, 1.0),
        beats_incumbent=beats_incumbent,
    )


def test_the_defaults_gate_needs_the_whole_interval_below_zero() -> None:
    """A mean that merely lands below zero is what a coin flip looks like.

    Invariant 41 measured a search reporting -1.4% in-sample that came out at -0.32% with the
    interval crossing zero on fresh draws.
    """
    assert _verdict((-2.0, -0.5)).beats_default
    assert not _verdict((-2.0, 0.5)).beats_default, "straddling zero is a null result"
    assert not _verdict((0.5, 2.0)).beats_default


def test_persisting_needs_both_gates() -> None:
    assert _verdict((-2.0, -0.5)).should_persist, "no incumbent to lose to"
    assert _verdict((-2.0, -0.5), beats_incumbent=True).should_persist
    assert not _verdict((-2.0, -0.5), beats_incumbent=False).should_persist
    assert not _verdict((-2.0, 0.5), beats_incumbent=True).should_persist


# --------------------------------------------------------------------------- params store


def test_each_backend_owns_its_block(tmp_path: Path) -> None:
    path = tmp_path / "forecasting_params.json"
    assert read_tuned_params(path, "lightgbm") is None

    _write_backend_block(path, "lightgbm", {"params": {"n_estimators": 300}})
    _write_backend_block(path, "catboost", {"params": {"n_estimators": 900}})
    assert sorted(json.loads(path.read_text())) == ["catboost", "lightgbm"]

    _write_backend_block(path, "lightgbm", {"params": {"n_estimators": 111}})
    catboost = read_tuned_params(path, "catboost")
    assert catboost is not None
    assert catboost["params"] == {"n_estimators": 900}, "rewriting one leaves the other alone"


def test_a_failed_gate_drops_only_its_own_block(tmp_path: Path) -> None:
    """The imputer deletes its whole file; here the other backend passed its own gates."""
    path = tmp_path / "forecasting_params.json"
    _write_backend_block(path, "lightgbm", {"params": {"n_estimators": 300}})
    _write_backend_block(path, "catboost", {"params": {"n_estimators": 900}})

    assert _drop_backend_block(path, "lightgbm") is True
    assert sorted(json.loads(path.read_text())) == ["catboost"]
    assert _drop_backend_block(path, "lightgbm") is False, "dropping twice is not an error"

    assert _drop_backend_block(path, "catboost") is True
    assert not path.exists(), "the file goes once its last block does"


def test_an_unreadable_params_file_falls_back_instead_of_raising(tmp_path: Path) -> None:
    """A tuned file is an optimization and the YAML defaults work, so it must not take a run down."""
    path = tmp_path / "forecasting_params.json"
    path.write_text("{ not json")
    assert read_tuned_params(path, "lightgbm") is None


def test_tuned_params_are_refused_when_the_panel_does_not_match(tmp_path: Path) -> None:
    """Hyperparameters do not transfer across training sizes.

    Invariant 41 measured a winner tuned at 50 series coming out 12% WORSE than not tuning at
    500, which is why the file records what it was tuned on and this check exists.
    """
    settings = settings_for(tmp_path)
    path = tmp_path / settings.models.forecasting_params_filename
    tuned = {"n_estimators": 2855, "learning_rate": 0.1, "max_depth": 10, "num_leaves": 4}
    _write_backend_block(
        path,
        "lightgbm",
        {
            "params": tuned,
            "tuned_on": {
                "n_series": 500,
                "horizon": HORIZON,
                "lags": list(FEATURES.lags),
                "rolling_windows": list(FEATURES.rolling_windows),
            },
            "gate": {"improvement_pct": -3.41},
        },
    )

    matched, provenance = resolve_backend_params(settings, "lightgbm", n_series=500)
    assert matched == tuned
    assert "tune_forecasting" in provenance

    mismatched, provenance = resolve_backend_params(settings, "lightgbm", n_series=50)
    assert mismatched == default_params(settings)
    assert "n_series" in provenance, "the log has to name what did not match"

    # The other backend has no block at all and takes the defaults without complaint.
    other, provenance = resolve_backend_params(settings, "catboost", n_series=500)
    assert other == default_params(settings)


# --------------------------------------------------------------------------- the objective


@pytest.mark.parametrize("backend", ["lightgbm", "catboost"])
def test_the_objective_prices_a_worse_model_higher(backend: str, tmp_path: Path) -> None:
    """Cost, not MAE: it is the criterion the champion is selected by.

    Same draw for both candidates, so the difference is the hyperparameters rather than
    resampling luck -- which is also what makes the gate's per-draw differences paired.
    """
    frame, feature_columns, target = supervised()
    settings = settings_for(tmp_path)
    selection, _ = build_draw_sets(frame, HORIZON, seed=5)
    # One draw: this asks whether the cost ranks two candidates, not how tight the interval is.
    one_draw = dataclasses.replace(selection, draws=selection.draws[:1])

    baseline = draw_costs(
        one_draw, backend, default_params(settings), settings, feature_columns, target
    )
    crippled = draw_costs(
        one_draw,
        backend,
        {"n_estimators": 1, "learning_rate": 0.001, "max_depth": 1},
        settings,
        feature_columns,
        target,
    )

    assert baseline.shape == crippled.shape == (1,)
    assert (baseline > 0).all(), "a cost of zero would mean perfect foresight"
    assert crippled[0] > baseline[0], "one tree at a glacial rate must cost more"


def test_the_objective_scores_through_the_conformal_layer() -> None:
    """The tuning must optimise the policy the pipeline deploys, not the raw model's.

    Conformal widens the outer quantiles by the calibrated radius, and `choose_order_quantity`
    interpolates the critical fractile between the median and the upper quantile, so the
    deployed order sits `0.75 * q_hat` above the raw one. Scoring the raw model measured a
    policy the system never runs: on the real panel the two costs came out 15.52 against 39.28,
    a 153% gap. This pins that `_draw_cost` goes through the calibrated forecaster.
    """
    import inspect

    from retail_forecasting.forecasting import forecasting_tuning as module

    source = inspect.getsource(module._draw_cost)

    assert "_train_conformal_model" in source
    assert "_split_train_calibration" in source
    # And the Mondrian group has to reach prediction, or the calibration is only global.
    assert "group_ids=eval_groups" in source


def test_the_study_name_carries_the_objective_version() -> None:
    """A changed objective must not resume trials scored under the old one.

    Optuna resumes by study name with `load_if_exists=True`. Without a version in the name, a
    study holding 300 trials from the previous objective would be reused in silence: the search
    would see nothing left to do, return at once, and the gate would validate a winner selected
    under the old cost against defaults measured under the new one.
    """
    import inspect

    from retail_forecasting.forecasting import forecasting_tuning as module

    assert module._OBJECTIVE_VERSION >= 2
    source = inspect.getsource(module._open_study)
    assert "_OBJECTIVE_VERSION" in source
    assert 'study_name=f"forecasting_{backend}"' not in source
