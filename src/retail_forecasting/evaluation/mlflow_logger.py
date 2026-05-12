from __future__ import annotations

import mlflow

from retail_forecasting.config import Settings
from retail_forecasting.evaluation.reporting import RunArtifacts


def log_experiment_to_mlflow(artifacts: RunArtifacts, settings: Settings):
    """
    Logs the experiment parameters, metrics, and artifacts to MLflow.
    """
    mlflow.set_experiment(settings.reporting.run_name)

    with mlflow.start_run(
        run_name=f"run_{artifacts.run_directory.name}"
        if artifacts.run_directory
        else None
    ):
        # 1. Log Configuration Parameters
        # Flatten settings for MLflow
        params = {}
        for section_name, section_config in settings.model_dump().items():
            if isinstance(section_config, dict):
                for k, v in section_config.items():
                    # Handle nested lists/dicts by converting to string
                    params[f"{section_name}_{k}"] = str(v)
            else:
                params[section_name] = str(section_config)

        mlflow.log_params(params)

        # 2. Log Top-Level Metrics (from the Champion or overall)
        if not artifacts.cost_summary.empty:
            # We'll log the metrics of the best model (lowest sim_total_cost)
            best_model_idx = artifacts.cost_summary["sim_total_cost"].idxmin()
            best_model_row = artifacts.cost_summary.loc[best_model_idx]

            mlflow.log_metric(
                "champion_sim_total_cost", best_model_row["sim_total_cost"]
            )
            mlflow.log_metric(
                "champion_sim_service_level", best_model_row["sim_service_level"]
            )
            mlflow.log_metric("champion_total_cost", best_model_row["total_cost"])
            mlflow.log_metric("champion_service_level", best_model_row["service_level"])

            # Find the corresponding row in metrics_summary
            model_name = best_model_row["model_name"]
            backend_name = best_model_row["backend_name"]
            strat = best_model_row.get("data_strategy", None)

            if strat:
                met_row = artifacts.metrics_summary[
                    (artifacts.metrics_summary["model_name"] == model_name)
                    & (artifacts.metrics_summary["backend_name"] == backend_name)
                    & (artifacts.metrics_summary["data_strategy"] == strat)
                ]
            else:
                met_row = artifacts.metrics_summary[
                    (artifacts.metrics_summary["model_name"] == model_name)
                    & (artifacts.metrics_summary["backend_name"] == backend_name)
                ]

            if not met_row.empty:
                m = met_row.iloc[0]
                if "winkler_score" in m:
                    mlflow.log_metric("champion_winkler_score", m["winkler_score"])
                if "interval_coverage" in m:
                    mlflow.log_metric(
                        "champion_interval_coverage", m["interval_coverage"]
                    )
                if "mae" in m:
                    mlflow.log_metric("champion_mae", m["mae"])

        # 3. Log Artifacts
        if artifacts.run_directory and artifacts.run_directory.exists():
            mlflow.log_artifacts(
                str(artifacts.run_directory), artifact_path="run_outputs"
            )
