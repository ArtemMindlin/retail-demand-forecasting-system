from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.data.pandas_dataset import from_pandas

from retail_forecasting.config import ModelConfig, Settings, build_config_hash
from retail_forecasting.data.censorship import (
    DEFAULT_SUPERVISED_LGBM_PARAMS,
    IMPUTATION_LGBM_PARAMS_FILENAME,
    LatentDemandImputer,
    synthetic_censor_holdout,
)
from retail_forecasting.forecasting.imputation_tuning import (
    _FLOAT_BOUNDS,
    _INT_BOUNDS,
    _MLFLOW_EXPERIMENT,
    _build_holdout_set,
    _holdout_maes,
    _split_temporal_windows,
    tune_imputation_lgbm,
)
from tests import make_synthetic_panel

METADATA_FILENAME = "imputation_lgbm_tuning_metadata.json"


@pytest.fixture(autouse=True)
def _mlflow_never_lands_in_the_repo(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run these tests from a scratch directory so MLflow's output follows them.

    Both halves of it. `mlflow.db` is a relative path, and the artifact root is resolved
    against the working directory too -- separately, so redirecting one leaves the other
    behind: a search with MLFLOW_TRACKING_URI pointed at /tmp still wrote an mlruns/ folder
    into the repo. Moving the working directory is what catches both.

    Safe here specifically because this module stubs the panel loader and passes an absolute
    models_dir, so nothing else it touches is relative to the repo root.
    """
    monkeypatch.chdir(tmp_path_factory.mktemp("mlflow_cwd"))


@pytest.fixture
def patched_train_only_loader(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    train_panel = make_synthetic_panel(num_series=3, num_days=70)

    def fake_load(*, dataset_config, preprocessing_config, split: str) -> pd.DataFrame:
        assert split == "train", "imputation tuning must never load the eval holdout split"
        return train_panel.copy()

    monkeypatch.setattr(
        "retail_forecasting.forecasting.imputation_tuning.load_prepared_panel", fake_load
    )
    return train_panel


def _settings(tmp_path: Path) -> Settings:
    return Settings(models=ModelConfig(models_dir=tmp_path, tuning_trials=2))


INCUMBENT_MARKER_ESTIMATORS = 999


def _write_incumbent(tmp_path: Path, params_path: Path | None = None) -> Path:
    """Plant a params file on disk so the run under test has an incumbent to be judged against.

    Marked by a sentinel ``n_estimators`` so ``_stub_holdout_maes`` can tell it apart from both
    the untuned defaults and the challenger the search produces.
    """
    target = params_path if params_path is not None else tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME
    incumbent = dict(DEFAULT_SUPERVISED_LGBM_PARAMS) | {"n_estimators": INCUMBENT_MARKER_ESTIMATORS}
    target.write_text(json.dumps(incumbent), encoding="utf-8")
    return target


def _stub_holdout_maes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    default_maes: list[float],
    other_maes: list[float],
    incumbent_maes: list[float] | None = None,
) -> None:
    """Score the untuned defaults, the planted incumbent, and everything else at fixed MAEs.

    One value per validation draw, so the tests drive the confidence interval the persist gate
    decides on -- not just its mean.
    """

    def fake_holdout_maes(holdouts, params) -> np.ndarray:
        if incumbent_maes is not None and params.get("n_estimators") == INCUMBENT_MARKER_ESTIMATORS:
            return np.asarray(incumbent_maes, dtype=float)
        is_default = {k: params[k] for k in DEFAULT_SUPERVISED_LGBM_PARAMS} == dict(
            DEFAULT_SUPERVISED_LGBM_PARAMS
        )
        return np.asarray(default_maes if is_default else other_maes, dtype=float)

    monkeypatch.setattr(
        "retail_forecasting.forecasting.imputation_tuning._holdout_maes", fake_holdout_maes
    )


def test_split_temporal_windows_are_disjoint_and_cover_the_panel() -> None:
    """The two windows must partition the calendar, and neither may be empty.

    The axis is TIME, not series: both windows hold every series, and the teacher is fitted on
    the whole panel. Disjoint censoring seeds alone were never enough (every draw re-censors 30%
    of the same clean rows, so the two sets converged -- 99.7% overlap at 15/25 draws), and the
    earlier fix of partitioning by series shrank the teacher to a third of deployment size,
    which measurably changed the answer instead of merely adding noise.
    """
    panel = make_synthetic_panel(num_series=10, num_days=90)

    selection_mask, validation_mask = _split_temporal_windows(panel)

    assert not (selection_mask & validation_mask).any(), "a row cannot be in both windows"
    assert (selection_mask | validation_mask).all(), "every row belongs to one window"
    assert selection_mask.any() and validation_mask.any()
    assert panel.loc[selection_mask, "date"].max() < panel.loc[validation_mask, "date"].min()
    # Every series is present on BOTH sides -- that is what keeps the teacher at full size.
    assert set(panel.loc[selection_mask, "series_id"]) == set(
        panel.loc[validation_mask, "series_id"]
    )


def test_split_temporal_windows_holds_back_the_last_third_of_the_calendar() -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)

    _, validation_mask = _split_temporal_windows(panel)

    validation_days = panel.loc[validation_mask, "date"].nunique()
    assert validation_days == 30
    assert panel.loc[validation_mask, "date"].min() == sorted(panel["date"].unique())[60]


def test_split_temporal_windows_rejects_a_panel_it_cannot_partition() -> None:
    single_day = make_synthetic_panel(num_series=3, num_days=1)

    with pytest.raises(ValueError, match="at least 2"):
        _split_temporal_windows(single_day)


def test_tune_imputation_lgbm_records_the_temporal_split_in_its_metadata(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame
) -> None:
    panel = patched_train_only_loader
    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["n_series"] == panel["series_id"].nunique()
    assert metadata["selection_window_end"] < metadata["validation_window_start"]
    assert metadata["n_selection_eval_rows"] > 0
    assert metadata["n_validation_eval_rows"] > 0
    # The teacher must see far more clean rows than any single window can score: it is fitted
    # on the WHOLE panel, which is the entire point of masking instead of splitting.
    assert metadata["teacher_fit_rows"] > metadata["n_validation_eval_rows"]


def test_tune_imputation_lgbm_scores_on_disjoint_holdouts(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame
) -> None:
    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["strategy"] == "optuna_imputation_lgbm"
    assert metadata["n_trials_requested"] == 2
    assert metadata["n_selection_holdouts"] == 2
    assert metadata["n_validation_holdouts"] == 2

    # The validation draws must be disjoint from the ones the search minimized on: scoring the
    # winner on the draws that selected it is what makes a noise-picked gain look real.
    assert not set(metadata["selection_seeds"]) & set(metadata["validation_seeds"])
    assert metadata["best_mae_selection"] >= 0
    assert metadata["best_mae_validation"] >= 0
    assert metadata["default_mae_validation"] >= 0


def test_tune_imputation_lgbm_persists_params_when_winner_beats_defaults(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_holdout_maes(monkeypatch, default_maes=[1.0, 1.0], other_maes=[0.5, 0.5])

    params_path = tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    assert params_path == tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME
    params = json.loads(params_path.read_text(encoding="utf-8"))
    assert set(params) == set(DEFAULT_SUPERVISED_LGBM_PARAMS)

    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["persisted"] is True
    assert metadata["improvement_pct"] == pytest.approx(-50.0)
    assert metadata["improvement_ci95"][1] < 0


def test_tune_imputation_lgbm_skips_persisting_when_the_gain_could_be_a_coin_flip(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative mean is not enough: the interval has to clear zero.

    The gate used to compare point estimates, and passed a winner at -1.14% that measured
    -0.45% with a straddling interval on fresh draws.
    """
    _stub_holdout_maes(
        monkeypatch,
        default_maes=[1.0, 1.0, 1.0, 1.0, 1.0],
        other_maes=[0.5, 1.5, 0.9, 1.1, 0.95],
    )

    returned = tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=5,
    )

    metadata = json.loads(returned.read_text(encoding="utf-8"))
    assert metadata["improvement_pct"] < 0, "the mean improvement is negative..."
    assert metadata["improvement_ci95"][1] > 0, "...but the interval straddles zero"
    assert metadata["persisted"] is False
    assert not (tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME).exists()


def test_tune_imputation_lgbm_returns_the_defaults_when_nothing_beats_them(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search that finds nothing better must end on the defaults, not on its best guess.

    The defaults are enqueued as a trial, so they compete. When every other candidate is worse
    they win outright, the improvement is exactly zero, and the gate declines to persist. Before
    they were enqueued the search returned its best random candidate and only the gate stood
    between that and the pipeline.
    """
    _stub_holdout_maes(monkeypatch, default_maes=[0.5, 0.5], other_maes=[1.0, 1.0])

    returned = tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    # No params file: persisting a loser would silently switch the pipeline to hyperparameters
    # chosen by selection noise.
    assert not (tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME).exists()
    assert returned == tmp_path / METADATA_FILENAME

    metadata = json.loads(returned.read_text(encoding="utf-8"))
    assert metadata["persisted"] is False
    assert metadata["best_mae_validation"] == metadata["default_mae_validation"]
    assert metadata["improvement_pct"] == pytest.approx(0.0)


def test_tune_imputation_lgbm_removes_a_superseded_params_file_when_the_gate_fails(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected search must not leave an earlier winner in charge of the pipeline."""
    stale = tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME
    stale.write_text(json.dumps(dict(DEFAULT_SUPERVISED_LGBM_PARAMS)), encoding="utf-8")

    _stub_holdout_maes(monkeypatch, default_maes=[0.5, 0.5], other_maes=[1.0, 1.0])

    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    assert not stale.exists()


def test_tune_imputation_lgbm_keeps_an_incumbent_the_new_winner_cannot_beat(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beating the defaults is not enough to overwrite: the incumbent has to lose too.

    The defaults comparison is blind to whatever is on disk, so without this second gate a run
    measuring -4.65% replaced a -5.33% winner -- it scored better on the selection draws it was
    optimizing and worse on the held-out series, and nothing checked.
    """
    incumbent = _write_incumbent(tmp_path)
    before = incumbent.read_text(encoding="utf-8")

    # Challenger beats the defaults (0.8 < 1.0) but loses to the incumbent (0.8 > 0.5).
    _stub_holdout_maes(
        monkeypatch,
        default_maes=[1.0, 1.0, 1.0, 1.0],
        other_maes=[0.8, 0.8, 0.8, 0.8],
        incumbent_maes=[0.5, 0.5, 0.5, 0.5],
    )

    returned = tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=4,
    )

    # The incumbent survives byte-for-byte, and the run reports why it was not replaced.
    assert incumbent.read_text(encoding="utf-8") == before
    assert returned == tmp_path / METADATA_FILENAME

    metadata = json.loads(returned.read_text(encoding="utf-8"))
    assert metadata["improvement_ci95"][1] < 0, "it did beat the defaults..."
    assert metadata["beats_incumbent"] is False, "...but not the incumbent"
    assert metadata["persisted"] is False
    assert metadata["incumbent_mae_validation"] == pytest.approx(0.5)


def test_tune_imputation_lgbm_replaces_an_incumbent_the_new_winner_beats(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate must not be so sticky that a genuinely better search cannot land."""
    incumbent = _write_incumbent(tmp_path)

    # Challenger beats both the defaults (0.4 < 1.0) and the incumbent (0.4 < 0.8).
    _stub_holdout_maes(
        monkeypatch,
        default_maes=[1.0, 1.0, 1.0, 1.0],
        other_maes=[0.4, 0.4, 0.4, 0.4],
        incumbent_maes=[0.8, 0.8, 0.8, 0.8],
    )

    returned = tune_imputation_lgbm(
        _settings(tmp_path),
        # Four: the first two trials are the enqueued references, the defaults and the planted
        # incumbent. At two there is no budget left for the search to produce a challenger.
        n_trials=4,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=4,
    )

    assert returned == incumbent
    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["beats_incumbent"] is True
    assert metadata["persisted"] is True
    # Overwritten, so the sentinel marking the planted incumbent is gone.
    assert json.loads(incumbent.read_text(encoding="utf-8"))["n_estimators"] != (
        INCUMBENT_MARKER_ESTIMATORS
    )


def test_tune_imputation_lgbm_has_no_incumbent_to_beat_on_a_first_run(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing on disk the second gate must not block the first ever winner."""
    _stub_holdout_maes(monkeypatch, default_maes=[1.0, 1.0], other_maes=[0.5, 0.5])

    returned = tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    assert returned == tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME
    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["beats_incumbent"] is None
    assert metadata["incumbent_mae_validation"] is None
    assert metadata["persisted"] is True


def test_the_untuned_defaults_sit_inside_the_search_space() -> None:
    """The baseline is enqueued as a trial, so the space has to be able to express it.

    The regularizers are the ones that can break this: they are drawn log-uniformly from 1e-8,
    and no log scale reaches zero. Setting them back to LightGBM's own 0.0 would put the
    baseline outside the space, where Optuna enqueues it anyway behind a warning and leaves the
    Gaussian process holding a point it can never propose.
    """
    bounds: dict[str, tuple[float, float]] = {**_INT_BOUNDS, **_FLOAT_BOUNDS}

    for name, (low, high) in bounds.items():
        value = DEFAULT_SUPERVISED_LGBM_PARAMS[name]
        assert low <= value <= high, f"{name}={value} is outside the space's [{low}, {high}]"

    # Everything the space searches has a default, and `subsample_freq` is derived rather than
    # searched -- so the defaults cover the space exactly, plus that one.
    assert set(DEFAULT_SUPERVISED_LGBM_PARAMS) == set(bounds) | {"subsample_freq"}


def test_holdout_maes_are_unchanged_by_evaluating_the_draws_in_parallel() -> None:
    """Threading the draws must be an optimization, not a change of result.

    The draws are independent and each builds its own imputer, so the only way this could
    diverge is shared state. One value per draw, in the order given.
    """
    panel = make_synthetic_panel(num_series=4, num_days=80)
    selection_mask, _ = _split_temporal_windows(panel)
    holdouts = _build_holdout_set(panel, [42, 43, 44], selection_mask).draws
    params = dict(DEFAULT_SUPERVISED_LGBM_PARAMS)

    serial = np.asarray(
        [
            float(
                np.mean(
                    np.abs(
                        LatentDemandImputer(strategy="supervised", lgbm_params=params)
                        .impute(censored)
                        .loc[eval_idx, "latent_demand_est"]
                        .to_numpy(dtype=float)
                        - true_demand
                    )
                )
            )
            for censored, eval_idx, true_demand in holdouts
        ]
    )

    assert np.array_equal(_holdout_maes(holdouts, params), serial)


def _logged_run() -> mlflow.entities.Run:
    """The MLflow run the search under test just wrote.

    By identity rather than by searching the experiment, because the tests in this module do
    not get a store each: MLflow caches its store per tracking-URI string, the URI is the same
    relative path every time, and so every run in the module lands in whichever directory was
    current the first time MLflow was touched.
    """
    run = mlflow.last_active_run()
    assert run is not None, "the search logged no MLflow run"
    return mlflow.get_run(run.info.run_id)


def test_mlflow_records_why_a_run_lost_to_its_incumbent(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declined persist has to be readable in the UI, not only in the metadata file.

    Without the incumbent comparison logged, a run that beat the defaults and still did not
    persist shows good metrics, `persisted=False`, and no explanation anywhere.
    """
    _write_incumbent(tmp_path)
    _stub_holdout_maes(
        monkeypatch,
        default_maes=[1.0, 1.0, 1.0, 1.0],
        other_maes=[0.8, 0.8, 0.8, 0.8],
        incumbent_maes=[0.5, 0.5, 0.5, 0.5],
    )

    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=4,
    )

    run = _logged_run()
    assert run.data.metrics["mae_validation_incumbent"] == pytest.approx(0.5)
    assert "incumbent_ci95_low" in run.data.metrics
    assert "incumbent_ci95_high" in run.data.metrics
    assert run.data.tags["beats_incumbent"] == "False"
    assert run.data.tags["persisted"] == "False"
    # The verdict as one searchable value, so the UI does not need the two tags cross-read.
    assert run.data.tags["outcome"] == "lost_to_incumbent"


def test_mlflow_logs_no_incumbent_metrics_when_there_was_no_incumbent(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent, not zero. A sentinel is a number someone eventually averages."""
    _stub_holdout_maes(monkeypatch, default_maes=[1.0, 1.0], other_maes=[0.5, 0.5])

    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    run = _logged_run()
    assert "mae_validation_incumbent" not in run.data.metrics
    assert "incumbent_ci95_low" not in run.data.metrics
    assert "incumbent_ci95_high" not in run.data.metrics
    assert run.data.tags["beats_incumbent"] == "None"
    assert run.data.tags["outcome"] == "persisted"
    # The defaults comparison is logged either way -- that gate always runs.
    assert run.data.metrics["mae_validation_default"] == pytest.approx(1.0)


def test_mlflow_records_the_shape_of_the_experiment_that_produced_a_run(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs are only comparable if the UI says what each of them measured.

    The sizes are metrics rather than params because MLflow stores params as strings, and
    these are the numbers a search is sorted and filtered by -- `teacher_fit_rows` above all,
    which invariant 41 makes the variable deciding whether tuning helps at all.
    """
    _stub_holdout_maes(monkeypatch, default_maes=[1.0, 1.0, 1.0], other_maes=[0.5, 0.5, 0.5])

    returned = tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=7,
        n_selection_holdouts=2,
        n_validation_holdouts=3,
    )
    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert returned.exists()

    run = _logged_run()
    assert run.data.params["n_selection_holdouts"] == "2"
    assert run.data.params["n_validation_holdouts"] == "3"
    assert run.data.params["selection_window_end"] == metadata["selection_window_end"]
    assert run.data.params["validation_window_start"] == metadata["validation_window_start"]
    assert run.data.params["seed"] == "7"

    assert run.data.metrics["teacher_fit_rows"] == metadata["teacher_fit_rows"]
    assert run.data.metrics["n_selection_eval_rows"] == metadata["n_selection_eval_rows"]
    assert run.data.metrics["n_validation_eval_rows"] == metadata["n_validation_eval_rows"]

    # Reconstructible from `seed` and the two counts, and in the metadata artifact besides.
    assert "selection_seeds" not in run.data.params
    assert "validation_seeds" not in run.data.params


def test_mlflow_records_which_panel_a_run_actually_read(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`n_series` and `teacher_fit_rows` pin the panel's shape. Nothing pinned its contents.

    The panel cache is keyed on four dataset settings, not on the preprocessing config nor on
    the code version, so a panel built before a change to `prepare_daily_panel` is served
    unchanged afterwards -- under a run whose git_commit says otherwise. A digest is what tells
    two same-shaped runs apart, which matters because invariant 41 makes a params file valid
    only near the panel it was tuned on.
    """
    _stub_holdout_maes(monkeypatch, default_maes=[1.0, 1.0], other_maes=[0.5, 0.5])

    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    inputs = _logged_run().inputs.dataset_inputs
    assert len(inputs) == 1
    dataset = inputs[0].dataset
    assert dataset.name == "imputation_train_panel"
    assert dataset.digest
    assert dataset.source_type == "local"

    # The same panel must hash the same, and a single changed cell must not.
    expected = from_pandas(patched_train_only_loader, name="imputation_train_panel")
    assert dataset.digest == expected.digest

    touched = patched_train_only_loader.copy()
    touched.loc[touched.index[0], "observed_demand"] += 1.0
    assert from_pandas(touched, name="imputation_train_panel").digest != dataset.digest


def test_mlflow_records_which_configuration_a_run_came_from(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both answers to "which configuration", because they answer different questions.

    The hash covers the RESOLVED settings, so it moves when a value does -- by a hand-edited
    YAML or an environment override -- where the path would not. The path is what a human
    recognises in the UI, where a bare sha256 identifies nothing.
    """
    _stub_holdout_maes(monkeypatch, default_maes=[1.0, 1.0], other_maes=[0.5, 0.5])
    settings = _settings(tmp_path)

    tune_imputation_lgbm(
        settings,
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
        config_path=Path("configs/imputation_tuning.yaml"),
    )

    tags = _logged_run().data.tags
    assert tags["config_path"] == "configs/imputation_tuning.yaml"
    assert tags["config_hash"] == build_config_hash(settings)

    # A settings change the path cannot see has to move the hash. Copied rather than mutated:
    # the config models are frozen, which is why the hash is worth anything in the first place.
    changed = settings.model_copy(
        update={
            "dataset": settings.dataset.model_copy(
                update={"top_n_series": (settings.dataset.top_n_series or 0) + 1}
            )
        }
    )
    assert build_config_hash(changed) != tags["config_hash"]


def test_config_path_is_optional_and_says_so_when_missing(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Called as a library rather than through run.py, there is no --config to report."""
    _stub_holdout_maes(monkeypatch, default_maes=[1.0, 1.0], other_maes=[0.5, 0.5])

    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    tags = _logged_run().data.tags
    assert tags["config_path"] == "unknown"
    assert tags["config_hash"], "the hash never depends on how the run was launched"


def test_mlflow_gives_every_trial_its_own_nested_run(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The convergence curve says when the search improved. It cannot say with what.

    A metric series indexed by trial number has one number per step, so the thirteen
    hyperparameters that produced it have nowhere to live. Nested runs are also what lets the
    UI plot a hyperparameter against the objective across the whole search.
    """
    _stub_holdout_maes(monkeypatch, default_maes=[1.0, 1.0], other_maes=[0.5, 0.5])

    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=3,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    parent = _logged_run()
    children = mlflow.search_runs(
        experiment_names=[_MLFLOW_EXPERIMENT],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        output_format="list",
    )
    assert len(children) == 3, "one per trial that produced a value"

    # Every child carries the parameters of its own trial, not the winner's.
    for child in children:
        assert set(child.data.params) == set(_INT_BOUNDS) | set(_FLOAT_BOUNDS)
        assert "mae" in child.data.metrics

    # The winner is findable among them rather than left to be spotted by eye.
    assert parent.data.params["best_trial_number"] in {
        c.info.run_name.removeprefix("trial-").lstrip("0") or "0" for c in children
    }
    assert {c.info.run_name for c in children} == {"trial-0000", "trial-0001", "trial-0002"}


def test_censoring_draws_its_severity_from_the_same_window_it_censors() -> None:
    """A faked stockout must borrow its severity from a REAL stockout of its own window.

    The severity pool is what makes a synthetic stockout resemble a real one, so drawing it
    from outside the window imports that period's severity. Measured on the v1 panel: the early
    window's real stockouts hide 43.4% of a day against the late window's 30.5%, and 76% of all
    stockout rows sit in the early window -- so an unrestricted pool was effectively the early
    window's, and late-window rows got censored ~32% harder than they ever are. That erases, on
    the severity axis, the very regime shift a temporal holdout exists to measure.

    Pinned with disjoint severities per window so a leak is unambiguous rather than statistical.
    """
    early_hours, late_hours = 4.0, 12.0
    rows = []
    for series in (1, 2):
        for day in range(12):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
            # Alternate clean / stockout so both windows hold some of each.
            hours = 0.0 if day % 2 == 0 else (early_hours if day < 8 else late_hours)
            rows.append(
                {
                    "date": date,
                    "series_id": f"{series}_10{series}",
                    "observed_demand": 10.0 + day,
                    "stockout_hours": hours,
                }
            )
    panel = pd.DataFrame(rows)

    selection_mask, validation_mask = _split_temporal_windows(panel)
    assert set(panel.loc[selection_mask & (panel["stockout_hours"] > 0), "stockout_hours"]) == {
        early_hours
    }
    assert set(panel.loc[validation_mask & (panel["stockout_hours"] > 0), "stockout_hours"]) == {
        late_hours
    }

    for window_name, mask, own, foreign in (
        ("selection", selection_mask, early_hours, late_hours),
        ("validation", validation_mask, late_hours, early_hours),
    ):
        for seed in range(6):
            censored, eval_idx, _ = synthetic_censor_holdout(panel, seed=seed, censorable_mask=mask)
            stamped = set(censored.loc[eval_idx, "stockout_hours"])
            assert stamped == {own}, (
                f"{window_name} draw (seed {seed}) stamped {stamped}, expected only {own} -- "
                f"{foreign} would mean the severity pool leaked across the window boundary"
            )


def test_holdout_set_owns_its_window_dates_and_derived_counts() -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)
    selection_mask, validation_mask = _split_temporal_windows(panel)

    selection = _build_holdout_set(panel, [1, 2, 3], selection_mask)
    validation = _build_holdout_set(panel, [9001, 9002], validation_mask)

    # Dates come from the masks themselves, never from arithmetic around a cut date.
    assert selection.window_start == panel.loc[selection_mask, "date"].min()
    assert selection.window_end == panel.loc[selection_mask, "date"].max()
    assert selection.window_end < validation.window_start
    assert selection.n_eval_rows > 0 and validation.n_eval_rows > 0
    # The teacher sees the whole panel, so its training set dwarfs either window's scored rows.
    assert selection.teacher_fit_rows > selection.n_eval_rows
    assert selection.teacher_fit_rows == validation.teacher_fit_rows + (
        validation.n_eval_rows - selection.n_eval_rows
    )


def test_tune_imputation_lgbm_breaks_a_statistical_tie_on_the_mean(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the draws cannot separate challenger from incumbent, the better mean wins.

    The defaults gate insists on the interval because it guards a CLAIM: an improvement whose
    interval includes zero is a null result and must be reported as one. This gate only picks
    which of two files sits on disk, and there "keep whichever arrived first" is not a more
    defensible tiebreak than "keep the better mean" -- it only looks more cautious. It already
    cost a decision once: -5.33% and -4.65% measured CI95 [-0.0011, +0.0051] against each other,
    a tie, and the incumbent survived on seniority alone.

    The draws below straddle zero by construction -- the challenger wins two and loses two --
    while its mean lands a hair lower.
    """
    incumbent = _write_incumbent(tmp_path)
    _stub_holdout_maes(
        monkeypatch,
        default_maes=[1.0, 1.0, 1.0, 1.0],
        other_maes=[0.30, 0.70, 0.30, 0.68],
        incumbent_maes=[0.70, 0.30, 0.70, 0.30],
    )

    returned = tune_imputation_lgbm(
        _settings(tmp_path),
        # Four: the first two trials are the enqueued references, the defaults and the planted
        # incumbent. At two there is no budget left for the search to produce a challenger.
        n_trials=4,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=4,
    )

    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    lo, hi = metadata["incumbent_ci95"]
    assert lo < 0 < hi, "precondition: the draws cannot separate the two"
    assert metadata["best_mae_validation"] < metadata["incumbent_mae_validation"]
    assert metadata["beats_incumbent"] is True, "the tie is broken on the mean"
    assert metadata["persisted"] is True
    assert returned == incumbent


@pytest.mark.parametrize(
    ("stockout_hours", "missing"),
    [([1.0, 2.0, 3.0], "no clean rows to fake"), ([0.0, 0.0, 0.0], "no real stockouts to copy")],
)
def test_censoring_refuses_a_panel_it_cannot_build_a_holdout_from(
    stockout_hours: list[float], missing: str
) -> None:
    """Scoring zero rows is never a legitimate outcome, so this raises rather than returning empty.

    Both ingredients are required and for different reasons: a clean day is the only place the
    true demand is known, and a real stockout is the only realistic severity to fake it at.

    It used to return empty and let each caller decide, which is how the imputation study came
    to write an `imputation_quality.csv` of headers and no rows -- an empty artifact reading as
    "no result" rather than "this failed". Same failure mode as invariants 14 and 32.
    """
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "series_id": ["1_101"] * 3,
            "observed_demand": [10.0, 11.0, 12.0],
            "stockout_hours": stockout_hours,
        }
    )

    with pytest.raises(ValueError, match="Cannot build a synthetic-censoring holdout"):
        synthetic_censor_holdout(panel, seed=0)
    assert missing  # names which ingredient is absent, for the failure message


def test_imputer_refuses_both_hyperparameter_sources_at_once() -> None:
    """Two sources for the same 13 numbers is a caller mistake, not a precedence question.

    The old behaviour let the file on disk win over an explicit argument, which is the harmful
    direction: adding `model_path` to the tuning's call site -- easy to do while applying
    invariant 40, which mandates it for `pipeline.py` -- would have made every trial score the
    tuned file instead of its own params, producing a flat `best_value` with nothing to explain
    it. Failing at construction turns that silent wrong answer into an immediate error.
    """
    with pytest.raises(ValueError, match="not both"):
        LatentDemandImputer(
            strategy="supervised",
            lgbm_params={"n_estimators": 5, "learning_rate": 0.3, "max_depth": 2},
            model_path=Path("whatever.json"),
        )

    # Either one alone stays valid: the tuning passes params, the pipeline passes the file.
    assert LatentDemandImputer(strategy="supervised", lgbm_params={"n_estimators": 5}) is not None
    assert LatentDemandImputer(strategy="supervised", model_path=Path("whatever.json")) is not None


def test_imputer_returns_the_same_columns_whether_or_not_it_corrects_anything() -> None:
    """One public method must not have two output shapes.

    `impute()` took a passthrough branch for strategy ``none`` and for panels with no censored
    rows, and that branch omitted `original_observed_demand`. Two consumers assume otherwise:
    `run_imputation_comparison` reads it unguarded, and `_build_prediction_frame` tests for
    `latent_demand_est` before reaching for all three. Both were correct only because the v1
    panel is 71.6% stockout days -- one config change from a KeyError, not unreachable.
    """
    with_stockouts = make_synthetic_panel(num_series=2, num_days=20)
    clean_only = with_stockouts.copy()
    clean_only["stockout_hours"] = 0.0

    corrected = LatentDemandImputer(strategy="supervised").impute(with_stockouts)
    nothing_to_correct = LatentDemandImputer(strategy="supervised").impute(clean_only)
    strategy_none = LatentDemandImputer(strategy="none").impute(with_stockouts)

    expected = {"latent_demand_est", "original_observed_demand", "is_imputed"}
    for name, frame in (
        ("corrected", corrected),
        ("no censored rows", nothing_to_correct),
        ("strategy=none", strategy_none),
    ):
        added = set(frame.columns) - set(with_stockouts.columns)
        assert added == expected, f"{name} added {added}"

    # And the passthrough leaves demand untouched, which is what "not imputed" has to mean.
    assert not nothing_to_correct["is_imputed"].any()
    assert nothing_to_correct["observed_demand"].equals(clean_only["observed_demand"])
