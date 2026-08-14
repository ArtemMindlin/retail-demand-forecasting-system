from __future__ import annotations

import json
from pathlib import Path

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


def _stub_mean_mae(
    monkeypatch: pytest.MonkeyPatch, *, default_mae: float, other_mae: float
) -> None:
    """Score the untuned defaults and everything else at fixed MAEs, to drive the persist gate."""

    def fake_mean_mae(holdouts, params) -> float:
        is_default = {k: params[k] for k in DEFAULT_SUPERVISED_LGBM_PARAMS} == dict(
            DEFAULT_SUPERVISED_LGBM_PARAMS
        )
        return default_mae if is_default else other_mae

    monkeypatch.setattr("retail_forecasting.forecasting.imputation_tuning._mean_mae", fake_mean_mae)


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
    _stub_mean_mae(monkeypatch, default_mae=1.0, other_mae=0.5)

    params_path = tune_imputation_lgbm(
        _settings(tmp_path),
        n_trials=2,
        seed=42,
        n_selection_holdouts=2,
        n_validation_holdouts=2,
    )

    assert params_path == tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME
    params = json.loads(params_path.read_text(encoding="utf-8"))
    assert set(params) == {"n_estimators", "learning_rate", "max_depth"}

    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["persisted"] is True
    assert metadata["improvement_pct"] == pytest.approx(-50.0)


def test_tune_imputation_lgbm_skips_persisting_when_winner_loses_to_defaults(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_mean_mae(monkeypatch, default_mae=0.5, other_mae=1.0)

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
