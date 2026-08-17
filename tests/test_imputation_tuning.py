from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retail_forecasting.config import ModelConfig, Settings
from retail_forecasting.data.censorship import (
    DEFAULT_SUPERVISED_LGBM_PARAMS,
    IMPUTATION_LGBM_PARAMS_FILENAME,
)
from retail_forecasting.forecasting.imputation_tuning import tune_imputation_lgbm
from tests import make_synthetic_panel

METADATA_FILENAME = "imputation_lgbm_tuning_metadata.json"


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


def _stub_holdout_maes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    default_maes: list[float],
    other_maes: list[float],
) -> None:
    """Score the untuned defaults and everything else at fixed per-draw MAEs.

    One value per validation draw, so the tests drive the bootstrap interval the persist
    gate decides on -- not just its mean.
    """

    def fake_holdout_maes(holdouts, params) -> np.ndarray:
        is_default = {k: params[k] for k in DEFAULT_SUPERVISED_LGBM_PARAMS} == dict(
            DEFAULT_SUPERVISED_LGBM_PARAMS
        )
        return np.asarray(default_maes if is_default else other_maes, dtype=float)

    monkeypatch.setattr(
        "retail_forecasting.forecasting.imputation_tuning._holdout_maes", fake_holdout_maes
    )


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
    assert set(params) == {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "num_leaves",
        "min_child_samples",
    }

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


def test_tune_imputation_lgbm_skips_persisting_when_winner_loses_to_defaults(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert metadata["improvement_pct"] == pytest.approx(100.0)


def test_tune_imputation_lgbm_removes_a_superseded_params_file_when_the_gate_fails(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected search must not leave an earlier winner in charge of the pipeline."""
    stale = tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME
    stale.write_text(
        json.dumps(
            {
                "n_estimators": 999,
                "learning_rate": 0.01,
                "max_depth": 9,
                "num_leaves": 31,
                "min_child_samples": 20,
            }
        ),
        encoding="utf-8",
    )

    _stub_holdout_maes(monkeypatch, default_maes=[0.5, 0.5], other_maes=[1.0, 1.0])

    tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    assert not stale.exists()
