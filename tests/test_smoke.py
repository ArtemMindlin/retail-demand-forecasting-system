from __future__ import annotations

from pathlib import Path

from retail_forecasting.config import DatasetConfig, ReportingConfig, Settings
from retail_forecasting.forecasting.pipeline import run_experiment_from_frame
from tests import make_synthetic_panel


def test_smoke_run_generates_report(tmp_path: Path) -> None:
    panel = make_synthetic_panel(num_series=3, num_days=90)
    settings = Settings()
    settings.dataset = DatasetConfig(
        top_n_series=3,
        min_history_days=70,
        horizon=7,
    )
    settings.reporting = ReportingConfig(
        output_dir=tmp_path,
        run_name="smoke_test",
        make_plots=False,
    )

    artifacts = run_experiment_from_frame(panel, settings)

    assert artifacts.run_directory is not None
    assert (artifacts.run_directory / "report.md").exists()
    assert not artifacts.metrics_summary.empty
    assert not artifacts.cost_summary.empty
