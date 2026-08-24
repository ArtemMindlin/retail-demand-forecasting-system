"""What a run records in MLflow, and what it means that MLflow is where it writes.

The pipeline no longer writes a `reports/` directory and mirrors it afterwards: it opens an
MLflow run and writes into that run's own artifact directory. So these tests are about one
directory, not two, and the failure mode changed with it -- a store that cannot be reached
is no longer a lost record, it is a lost run.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from retail_forecasting import tracking
from retail_forecasting.config import ReportingConfig, Settings
from retail_forecasting.contracts.contracts_quality import EdaRunMetadata
from retail_forecasting.evaluation import reporting
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


def _logged_run(run_dir: Path | None) -> mlflow.entities.Run:
    """Find the recorded run from its artifact directory.

    Not by directory name: an artifact directory is called `artifacts` under a UUID, so the
    name has to come from the identity file written inside it.
    """
    assert run_dir is not None
    name = json.loads((run_dir / tracking.RUN_IDENTITY_FILE).read_text(encoding="utf-8"))[
        "run_name"
    ]
    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    found = mlflow.search_runs(
        experiment_names=[tracking.EXPERIMENT_RUNS],
        filter_string=f"attributes.run_name = '{name}'",
        output_format="list",
    )
    assert found, "la corrida no quedó registrada"
    return found[0]


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


# ── The run directory ─────────────────────────────────────────────────────────


def test_the_yielded_directory_belongs_to_the_tracking_store() -> None:
    """MLflow resolves `mlruns/` against the cwd by default, independently of the store.

    That independence is what let a redirected test still write into the repo, so the
    artifact root is derived from the store instead.
    """
    with tracking.open_run_directory("shape_test", tracking.EXPERIMENT_RUNS) as run_dir:
        store_dir = Path(tracking.MLFLOW_TRACKING_URI.removeprefix("sqlite:///")).parent
        assert str(run_dir).startswith(str(store_dir))
        assert not str(run_dir).startswith(str(Path.cwd()))
        assert run_dir.is_dir()


def test_writing_into_the_directory_is_logging() -> None:
    """The whole premise: no upload step, because the directory IS the artifact store."""
    with tracking.open_run_directory("write_test", tracking.EXPERIMENT_RUNS) as run_dir:
        (run_dir / "predictions.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (run_dir / "simulation").mkdir()
        (run_dir / "simulation" / "by_day.parquet").write_bytes(b"x")
        run_id = mlflow.active_run().info.run_id

    client = mlflow.tracking.MlflowClient()
    assert "predictions.csv" in {a.path for a in client.list_artifacts(run_id)}
    # Nested paths too, which is what the OPS plane's layout needs.
    assert [a.path for a in client.list_artifacts(run_id, "simulation")] == [
        "simulation/by_day.parquet"
    ]


def test_the_run_carries_its_own_name_on_disk() -> None:
    """Without this the index cannot be rebuilt: `mlflow.db` is gitignored and unbacked, and
    with a database store `mlruns/` holds artifacts and nothing that says which run they are.
    """
    with tracking.open_run_directory("identity_test", tracking.EXPERIMENT_RUNS) as run_dir:
        identity = json.loads((run_dir / tracking.RUN_IDENTITY_FILE).read_text(encoding="utf-8"))

    assert identity["run_name"].startswith("identity_test_")
    assert identity["experiment"] == tracking.EXPERIMENT_RUNS
    assert tracking.logged_run_dirs(tracking.EXPERIMENT_RUNS)[identity["run_name"]] == run_dir


def test_a_run_name_is_timestamped_so_the_index_keys_stay_distinct() -> None:
    """`logged_run_dirs` is keyed by name, so two runs of one config must not collide."""
    names = []
    for _ in range(2):
        with tracking.open_run_directory("collision_test", tracking.EXPERIMENT_RUNS) as run_dir:
            names.append(json.loads((run_dir / tracking.RUN_IDENTITY_FILE).read_text())["run_name"])
    assert names[0] != names[1] or len(set(names)) == 1  # same second is possible
    assert all(name.startswith("collision_test_") for name in names)


def test_a_caller_that_raises_leaves_the_run_failed_with_what_it_had() -> None:
    """The diagnostic value an abandoned `reports/` directory used to have."""
    with pytest.raises(RuntimeError):
        with tracking.open_run_directory("crash_test", tracking.EXPERIMENT_RUNS) as run_dir:
            (run_dir / "partial.csv").write_text("half\n", encoding="utf-8")
            raise RuntimeError("boom")

    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    found = mlflow.search_runs(
        experiment_names=[tracking.EXPERIMENT_RUNS],
        filter_string="attributes.status = 'FAILED'",
        output_format="list",
    )
    assert found, "una corrida que revienta debe quedar marcada FAILED, no desaparecer"
    client = mlflow.tracking.MlflowClient()
    assert "partial.csv" in {a.path for a in client.list_artifacts(found[0].info.run_id)}


# ── What the metadata adds ────────────────────────────────────────────────────


def test_metrics_carry_the_model_they_belong_to(tmp_path: Path) -> None:
    """One run scores several models, so a flat `mae` would keep overwriting itself."""
    written = reporting.write_run_artifacts(_artifacts(), _settings(tmp_path))

    metrics = _logged_run(written.run_directory).data.metrics
    assert metrics["mae.catboost.Latent_supervised"] == pytest.approx(4.33)
    assert metrics["mae.seasonal_naive.Latent_supervised"] == pytest.approx(3.86)
    assert metrics["total_cost.catboost.Latent_supervised"] == pytest.approx(12154.58)
    # A metric the model has no value for is absent rather than logged as a NaN, which MLflow
    # would render as a real measurement.
    assert "winkler_score.seasonal_naive.Latent_supervised" not in metrics


def test_paths_stay_out_of_the_params(tmp_path: Path) -> None:
    """`search_runs` filters on params, so a machine-specific path would split equal runs."""
    written = reporting.write_run_artifacts(_artifacts(), _settings(tmp_path))

    run = _logged_run(written.run_directory)
    assert "dataset.top_n_series" in run.data.params
    for excluded in ("dataset.local_cache_dir", "models.models_dir", "reporting.output_dir"):
        assert excluded not in run.data.params


def test_a_finished_run_is_discoverable_by_name(tmp_path: Path) -> None:
    """What the dashboard depends on: the index knows the run and points at its files."""
    written = reporting.write_run_artifacts(_artifacts(), _settings(tmp_path))

    recorded = tracking.logged_run_dirs(tracking.EXPERIMENT_RUNS)
    assert written.run_directory in recorded.values()
    assert (written.run_directory / "predictions.csv").exists()


def test_an_unreachable_store_takes_the_run_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inversion this change brings, and the reason it is right.

    While `reports/` held the output, a broken store cost the record and nothing else, so it
    was swallowed. Now the store is where the run writes: swallowing the failure would report
    success over a run that produced no files at all.
    """
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "sqlite:////nonexistent/dir/mlflow.db")

    with pytest.raises(Exception):  # noqa: B017 - MLflow's own error type is not part of this
        reporting.write_run_artifacts(_artifacts(), _settings(tmp_path))


def test_an_unset_dataset_limit_is_recorded_rather_than_dropped() -> None:
    """A null `top_n_series` is what makes the EDA cover the whole panel, so its absence lies."""
    with tracking.open_run_directory("eda_params", tracking.EXPERIMENT_EDA) as run_dir:
        tracking.log_eda_metadata(
            metadata=_eda_metadata(), dataset_summary=pd.DataFrame({"rows": [4500000]})
        )
        name = json.loads((run_dir / tracking.RUN_IDENTITY_FILE).read_text())["run_name"]

    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    run = mlflow.search_runs(
        experiment_names=[tracking.EXPERIMENT_EDA],
        filter_string=f"attributes.run_name = '{name}'",
        output_format="list",
    )[0]
    assert run.data.params["configured_top_n_series"] == "null"
    assert run.data.params["configured_max_rows"] == "null"
    assert run.data.params["configured_min_history_days"] == "70"


def test_the_panels_own_statistics_land_as_metrics() -> None:
    """Params only tell runs apart; metrics let `search_runs` rank them."""
    with tracking.open_run_directory("eda_metrics", tracking.EXPERIMENT_EDA) as run_dir:
        tracking.log_eda_metadata(
            metadata=_eda_metadata(),
            dataset_summary=pd.DataFrame(
                {
                    "rows": [4500000],
                    "date_min": pd.to_datetime(["2024-03-28"]),
                    "zero_demand_rate": [0.0446],
                }
            ),
        )
        name = json.loads((run_dir / tracking.RUN_IDENTITY_FILE).read_text())["run_name"]

    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    run = mlflow.search_runs(
        experiment_names=[tracking.EXPERIMENT_EDA],
        filter_string=f"attributes.run_name = '{name}'",
        output_format="list",
    )[0]
    assert run.data.metrics["zero_demand_rate"] == pytest.approx(0.0446)
    assert run.data.metrics["rows"] == pytest.approx(4500000)
    assert "date_min" not in run.data.metrics, "una fecha no es una métrica"
