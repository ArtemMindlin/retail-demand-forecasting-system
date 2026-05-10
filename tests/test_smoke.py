from __future__ import annotations

import json
from pathlib import Path

from retail_forecasting.config import (
    DatasetConfig,
    ModelConfig,
    ReportingConfig,
    Settings,
)
from retail_forecasting.forecasting.pipeline import run_experiment_from_frame
from tests import make_synthetic_panel


def test_smoke_run_generates_report(tmp_path: Path) -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)
    settings = Settings(
        dataset=DatasetConfig(
            top_n_series=3,
            min_history_days=70,
            horizon=7,
        ),
        reporting=ReportingConfig(
            output_dir=tmp_path,
            run_name="smoke_test",
            make_plots=False,
        ),
    )

    artifacts = run_experiment_from_frame(panel, settings)

    assert artifacts.run_directory is not None
    assert (artifacts.run_directory / "report.md").exists()
    metadata_path = artifacts.run_directory / "backtest_metadata.json"
    assert metadata_path.exists()
    assert not artifacts.metrics_summary.empty
    assert not artifacts.cost_summary.empty
    assert artifacts.backtest_metadata is not None
    assert artifacts.backtest_metadata.features.supervised_rows == len(
        artifacts.supervised_frame
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["features"]["supervised_rows"] == len(artifacts.supervised_frame)
    assert metadata["validation"]["folds_created"] == settings.validation.n_folds
    assert metadata["tuning"] is None
    assert metadata["drift"]["detector_name"] == "PageHinkleyDetector"
    assert metadata["drift"]["threshold"] == settings.drift.threshold


def test_smoke_run_serializes_tuning_metadata(tmp_path: Path) -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)
    settings = Settings(
        dataset=DatasetConfig(
            top_n_series=3,
            min_history_days=70,
            horizon=7,
        ),
        models=ModelConfig(
            use_tuning=True,
            tuning_trials=2,
        ),
        reporting=ReportingConfig(
            output_dir=tmp_path,
            run_name="smoke_test_tuning",
            make_plots=False,
        ),
    )

    artifacts = run_experiment_from_frame(panel, settings)

    assert artifacts.backtest_metadata is not None
    assert artifacts.backtest_metadata.tuning is not None
    assert artifacts.backtest_metadata.tuning.best_params.n_estimators > 0
    metadata = json.loads(
        (artifacts.run_directory / "backtest_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["tuning"]["strategy"] == "optuna_temporal_holdout"
    assert metadata["tuning"]["n_trials_requested"] == 2
    assert metadata["tuning"]["best_params"]["n_estimators"] > 0
    assert metadata["drift"]["min_instances"] == settings.drift.min_instances
