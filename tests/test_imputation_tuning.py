from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from retail_forecasting.config import ModelConfig, Settings
from retail_forecasting.data.censorship import IMPUTATION_LGBM_PARAMS_FILENAME
from retail_forecasting.forecasting.imputation_tuning import tune_imputation_lgbm
from tests import make_synthetic_panel


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


def test_tune_imputation_lgbm_writes_params_file(
    tmp_path: Path, patched_train_only_loader: pd.DataFrame
) -> None:
    settings = Settings(models=ModelConfig(models_dir=tmp_path, tuning_trials=2))

    params_path = tune_imputation_lgbm(settings, n_trials=2, seed=42)

    assert params_path == tmp_path / IMPUTATION_LGBM_PARAMS_FILENAME
    assert params_path.exists()
    params = json.loads(params_path.read_text(encoding="utf-8"))
    assert set(params) == {"n_estimators", "learning_rate", "max_depth"}

    metadata_path = tmp_path / "imputation_lgbm_tuning_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["n_trials_requested"] == 2
    assert metadata["strategy"] == "optuna_imputation_lgbm"
