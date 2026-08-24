"""What `write_run_artifacts` records in MLflow, alongside the files it writes to reports/."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from retail_forecasting import tracking
from retail_forecasting.config import ReportingConfig, Settings
from retail_forecasting.contracts.contracts_quality import EdaRunMetadata
from retail_forecasting.evaluation.reporting import RunArtifacts


def _artifacts() -> RunArtifacts:
    """Two models scored over the same origins, which is the shape a real run has."""
    panel = pd.DataFrame(
        {
            "series_id": ["1_101", "1_101", "2_202", "2_202"],
            "date": pd.to_datetime(["2024-06-01", "2024-06-02"] * 2),
            "observed_demand": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return RunArtifacts(
        prepared_panel=panel,
        supervised_frame=panel,
        predictions=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-06-02", "2024-06-02"]),
                "series_id": ["1_101", "2_202"],
                "y_pred": [12.0, 18.0],
                "order_quantity": [13.0, 19.0],
                "prediction_source": ["model", "model"],
                "fallback_level": [pd.NA, pd.NA],
                "model_name": ["catboost", "catboost"],
                "backend_name": ["conformal_catboost", "conformal_catboost"],
                "fold_id": [0, 0],
                "stockout_hours": [0.0, 0.0],
                "stockout_regime": ["low", "low"],
                "velocity_regime": ["fast_moving", "slow_moving"],
                "promo_regime": ["baseline_price", "baseline_price"],
                "seasonal_regime": ["standard_season", "standard_season"],
            }
        ),
        metrics_summary=pd.DataFrame(
            {
                "model_name": ["catboost", "seasonal_naive"],
                "backend_name": ["conformal_catboost", "heuristic"],
                "data_strategy": ["Latent_supervised", "Latent_supervised"],
                "observations": [1050, 1050],
                "mae": [4.33, 3.86],
                "winkler_score": [30.77, float("nan")],
            }
        ),
        fold_metrics=pd.DataFrame(),
        cost_summary=pd.DataFrame(
            {
                "model_name": ["catboost", "seasonal_naive"],
                "backend_name": ["conformal_catboost", "heuristic"],
                "data_strategy": ["Latent_supervised", "Latent_supervised"],
                "total_cost": [12154.58, 10402.94],
                "service_level": [0.675, 0.589],
            }
        ),
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings().model_copy(
        update={"reporting": ReportingConfig(output_dir=tmp_path, run_name="tracking_test")}
    )


def _logged_run(run_dir: Path) -> mlflow.entities.Run:
    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    found = mlflow.search_runs(
        experiment_names=[tracking.EXPERIMENT_RUNS],
        filter_string=f"attributes.run_name = '{run_dir.name}'",
        output_format="list",
    )
    assert found, "la corrida no quedó registrada"
    return found[0]


def test_metrics_carry_the_model_they_belong_to(tmp_path: Path) -> None:
    """One run scores several models, so a flat `mae` would keep overwriting itself."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    tracking.log_run_to_mlflow(
        artifacts=_artifacts(), settings=_settings(tmp_path), run_dir=run_dir
    )

    metrics = _logged_run(run_dir).data.metrics
    assert metrics["mae.catboost.Latent_supervised"] == pytest.approx(4.33)
    assert metrics["mae.seasonal_naive.Latent_supervised"] == pytest.approx(3.86)
    assert metrics["total_cost.catboost.Latent_supervised"] == pytest.approx(12154.58)
    # A metric the model has no value for is absent rather than logged as a NaN, which MLflow
    # would render as a real measurement.
    assert "winkler_score.seasonal_naive.Latent_supervised" not in metrics


def test_paths_stay_out_of_the_params(tmp_path: Path) -> None:
    """`search_runs` filters on params, so a machine-specific path would split equal runs."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    tracking.log_run_to_mlflow(
        artifacts=_artifacts(), settings=_settings(tmp_path), run_dir=run_dir
    )

    run = _logged_run(run_dir)
    assert "dataset.top_n_series" in run.data.params
    for excluded in ("dataset.local_cache_dir", "models.models_dir", "reporting.output_dir"):
        assert excluded not in run.data.params
    # The path is a tag instead: it is the bridge back to the row-level artifacts.
    assert run.data.tags["reports_run_dir"] == str(run_dir)


def test_artifacts_follow_the_tracking_store_rather_than_the_working_directory(
    tmp_path: Path,
) -> None:
    """MLflow resolves `mlruns/` against the cwd by default, independently of the store.

    That independence is what let a redirected test still write an `mlruns/` into the repo, so
    the artifact root is derived from the store instead.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "cost_by_model.png").write_bytes(b"not really a png")

    tracking.log_run_to_mlflow(
        artifacts=_artifacts(), settings=_settings(tmp_path), run_dir=run_dir
    )

    store_dir = Path(tracking.MLFLOW_TRACKING_URI.removeprefix("sqlite:///")).parent
    run = _logged_run(run_dir)
    assert run.info.artifact_uri.startswith(str(store_dir))
    assert not run.info.artifact_uri.startswith(str(Path.cwd()))

    client = mlflow.MlflowClient()
    assert [a.path for a in client.list_artifacts(run.info.run_id, "figures")] == [
        "figures/cost_by_model.png"
    ]


def test_a_broken_tracking_store_does_not_take_the_run_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The files are already written by then; losing the record beats losing the run."""
    from retail_forecasting.evaluation import reporting

    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "sqlite:////nonexistent/dir/mlflow.db")
    artifacts = _artifacts()
    settings = _settings(tmp_path)

    written = reporting.write_run_artifacts(artifacts, settings)

    assert written.run_directory is not None
    assert (written.run_directory / "metrics_summary.csv").exists()


def _eda_run_dir(tmp_path: Path) -> Path:
    """A run directory shaped like the EDA's: figures, summaries, and nothing else."""
    run_dir = tmp_path / "eda_run"
    run_dir.mkdir()
    (run_dir / "observed_demand_distribution.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "coverage_heatmap.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    pd.DataFrame({"series_id": ["1_101"], "observed_demand_sum": [4.0]}).to_csv(
        run_dir / "series_summary.csv", index=False
    )
    return run_dir


def _eda_metadata() -> EdaRunMetadata:
    return EdaRunMetadata(
        split="train",
        panel_source="data/processed/train_abc.parquet",
        n_series=50000,
        rows=4500000,
        date_min="2024-03-28",
        date_max="2024-06-25",
        configured_top_n_series=None,
        configured_min_history_days=70,
        configured_max_rows=None,
        imputation_strategy="supervised",
        drop_negative_sales=True,
        fill_missing_values=True,
        config_hash="f6417b",
        config_path="configs/eda/default.yaml",
        created_at="2026-08-23T21:44:14+00:00",
        git_commit="925a5d8",
    )


def _logged_eda_run(run_dir: Path) -> mlflow.entities.Run:
    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    found = mlflow.search_runs(
        experiment_names=[tracking.EXPERIMENT_EDA],
        filter_string=f"attributes.run_name = '{run_dir.name}'",
        output_format="list",
    )
    assert found, "la corrida de EDA no quedó registrada"
    return found[0]


def test_an_eda_run_uploads_its_figures_and_its_summaries(tmp_path: Path) -> None:
    """Both, so the run can be read in a browser without knowing the folder name."""
    run_dir = _eda_run_dir(tmp_path)
    tracking.log_eda_run_to_mlflow(
        metadata=_eda_metadata(),
        run_dir=run_dir,
        dataset_summary=pd.DataFrame({"rows": [4500000]}),
    )

    client = mlflow.tracking.MlflowClient()
    run_id = _logged_eda_run(run_dir).info.run_id
    figures = {item.path.split("/")[-1] for item in client.list_artifacts(run_id, "figures")}
    summaries = {item.path.split("/")[-1] for item in client.list_artifacts(run_id, "summaries")}

    assert figures == {"observed_demand_distribution.png", "coverage_heatmap.png"}
    assert summaries == {"series_summary.csv"}


def test_an_unset_dataset_limit_is_recorded_rather_than_dropped(tmp_path: Path) -> None:
    """A null `top_n_series` is what makes the EDA cover the whole panel, so its absence lies."""
    run_dir = _eda_run_dir(tmp_path)
    tracking.log_eda_run_to_mlflow(
        metadata=_eda_metadata(),
        run_dir=run_dir,
        dataset_summary=pd.DataFrame({"rows": [4500000]}),
    )

    params = _logged_eda_run(run_dir).data.params
    assert params["configured_top_n_series"] == "null"
    assert params["configured_max_rows"] == "null"
    assert params["configured_min_history_days"] == "70"


def test_the_panels_own_statistics_land_as_metrics(tmp_path: Path) -> None:
    """Params only tell runs apart; metrics let `search_runs` rank them."""
    run_dir = _eda_run_dir(tmp_path)
    tracking.log_eda_run_to_mlflow(
        metadata=_eda_metadata(),
        run_dir=run_dir,
        dataset_summary=pd.DataFrame(
            {
                "rows": [4500000],
                "date_min": pd.to_datetime(["2024-03-28"]),
                "zero_demand_rate": [0.0446],
            }
        ),
    )

    metrics = _logged_eda_run(run_dir).data.metrics
    assert metrics["zero_demand_rate"] == pytest.approx(0.0446)
    assert metrics["rows"] == pytest.approx(4500000)
    assert "date_min" not in metrics, "una fecha no es una métrica"
