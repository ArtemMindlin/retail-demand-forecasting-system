from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retail_forecasting.config import (
    DatasetConfig,
    ModelConfig,
    ProjectConfig,
    ReportingConfig,
    Settings,
    SimulationConfig,
    ValidationConfig,
)
from retail_forecasting.simulation import run_operational_simulation
from tests import make_synthetic_panel


def _build_split_panels(
    train_days: int = 80, eval_days: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = make_synthetic_panel(num_series=3, num_days=train_days + eval_days)
    cutoff = full["date"].sort_values().unique()[train_days]
    train = full[full["date"] < cutoff].reset_index(drop=True)
    eval_panel = full[full["date"] >= cutoff].reset_index(drop=True)
    return train, eval_panel


@pytest.fixture
def patched_panel_loader(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_panel, eval_panel = _build_split_panels()

    def fake_load(*, dataset_config, preprocessing_config, split: str) -> pd.DataFrame:
        return train_panel.copy() if split == "train" else eval_panel.copy()

    monkeypatch.setattr("retail_forecasting.simulation.operations.load_prepared_panel", fake_load)
    return train_panel, eval_panel


def _fast_settings(tmp_path: Path, simulation_days: int = 8) -> Settings:
    return Settings(
        project=ProjectConfig(run_mode="simulate_ops"),
        dataset=DatasetConfig(
            top_n_series=3,
            min_history_days=10,
            horizon=3,
        ),
        validation=ValidationConfig(
            initial_train_days=20,
            n_folds=2,
            fold_size_days=4,
            calibration_days=7,
        ),
        models=ModelConfig(n_estimators=30, learning_rate=0.1, max_depth=4),
        reporting=ReportingConfig(
            output_dir=tmp_path,
            run_name="sim_test",
            make_plots=False,
        ),
        simulation=SimulationConfig(
            retrain_cadences=[None, 4],
            simulation_days=simulation_days,
            make_plots=False,
        ),
    )


def test_operational_simulation_produces_expected_outputs(
    tmp_path: Path, patched_panel_loader
) -> None:
    settings = _fast_settings(tmp_path, simulation_days=8)
    artifacts = run_operational_simulation(settings)

    assert artifacts.run_directory.exists()
    assert (artifacts.run_directory / "predictions_by_day.parquet").exists()
    assert (artifacts.run_directory / "cadence_summary.csv").exists()
    assert (artifacts.run_directory / "retrain_events.json").exists()
    assert (artifacts.run_directory / "report.md").exists()

    cadences_seen = set(artifacts.predictions_by_day["cadence"].unique())
    assert cadences_seen == {"never", "every_4d"}


def test_baseline_cadence_never_retrains(tmp_path: Path, patched_panel_loader) -> None:
    settings = _fast_settings(tmp_path, simulation_days=8)
    artifacts = run_operational_simulation(settings)

    never_events = [e for e in artifacts.retrain_events if e["cadence"] == "never"]
    assert never_events == []

    every4_events = [e for e in artifacts.retrain_events if e["cadence"] == "every_4d"]
    # With 8 simulation days and cadence=4, the counter hits 4 twice.
    assert len(every4_events) == 2


def test_predictions_include_realized_truth_within_window(
    tmp_path: Path, patched_panel_loader
) -> None:
    settings = _fast_settings(tmp_path, simulation_days=8)
    artifacts = run_operational_simulation(settings)

    complete = artifacts.predictions_by_day[artifacts.predictions_by_day["actuals_complete"]]
    assert not complete.empty
    assert complete["y_true"].notna().all()
    assert complete["total_cost"].ge(0).all()


def test_cadence_summary_reports_one_row_per_cadence(tmp_path: Path, patched_panel_loader) -> None:
    settings = _fast_settings(tmp_path, simulation_days=8)
    artifacts = run_operational_simulation(settings)

    summary = artifacts.cadence_summary
    assert set(summary["cadence"]) == {"never", "every_4d"}
    assert summary["n_observations"].gt(0).all()
