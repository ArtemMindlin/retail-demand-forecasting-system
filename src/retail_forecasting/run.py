from __future__ import annotations

import argparse
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from retail_forecasting.config import load_config
from retail_forecasting.contracts.contracts_config import ImputationStrategy, RunMode
from retail_forecasting.eda.pipeline import run_eda
from retail_forecasting.forecasting.pipeline import (
    run_experiment,
    run_fair_cost_backtest,
    run_retrain,
    run_scoring,
)
from retail_forecasting.simulation import run_operational_simulation
from retail_forecasting.utils.logging import configure as configure_logging
from retail_forecasting.utils.logging import fields, get_logger
from retail_forecasting.utils.provenance import freeze_git_commit

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for running the pipeline.

    Returns:
        The configured argument parser for experiment execution.
    """
    parser = argparse.ArgumentParser(
        description="Run the retail demand forecasting experiment pipeline.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/default.yaml",
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the experiment run name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for project.random_seed. A YAML that declares it beats the "
        "environment, so this is the only way to vary it per invocation.",
    )
    parser.add_argument(
        "--run-mode",
        default=None,
        choices=list(get_args(RunMode)),
        help="Optional override for the execution mode.",
    )
    parser.add_argument(
        "--imputation-strategy",
        default=None,
        choices=list(get_args(ImputationStrategy)),
        help="Optional override for preprocessing.imputation_strategy. It picks which arm "
        'an experiment runs: "none" scores Observed demand, anything else the '
        "reconstructed Latent demand.",
    )
    parser.add_argument(
        "--imputation-strategy",
        default=None,
        choices=list(get_args(ImputationStrategy)),
        help="Optional override for preprocessing.imputation_strategy. It picks which arm "
        'an experiment runs: "none" scores Observed demand, anything else the '
        "reconstructed Latent demand.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to analyze, validated against dataset.splits. Only eda reads it.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and execute the configured experiment run.

    Returns:
        None.

    Notes:
        The reporting output directory and run name can be overridden from the command line
        without modifying the YAML configuration.
    """
    args = build_parser().parse_args()

    configure_logging()
    freeze_git_commit()
    try:
        settings = load_config(args.config)
    except ValidationError as exc:
        raise SystemExit(_format_validation_error(exc)) from None
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    reporting_updates = {}
    project_updates = {}
    preprocessing_updates = {}
    if args.run_name is not None:
        reporting_updates["run_name"] = args.run_name
    if args.run_mode is not None:
        project_updates["run_mode"] = args.run_mode
    if args.seed is not None:
        project_updates["random_seed"] = args.seed
    if args.imputation_strategy is not None:
        preprocessing_updates["imputation_strategy"] = args.imputation_strategy

    if reporting_updates:
        new_reporting = settings.reporting.model_copy(update=reporting_updates)
        settings = settings.model_copy(update={"reporting": new_reporting})
    if project_updates:
        new_project = settings.project.model_copy(update=project_updates)
        settings = settings.model_copy(update={"project": new_project})
    if preprocessing_updates:
        new_preprocessing = settings.preprocessing.model_copy(update=preprocessing_updates)
        settings = settings.model_copy(update={"preprocessing": new_preprocessing})

    mode = settings.project.run_mode
    if mode == "retrain":
        run_retrain(settings)
        return
    if mode == "simulate_ops":
        sim_artifacts = run_operational_simulation(settings)
        fields(logger, {"escrito": sim_artifacts.run_directory})
        return
    if mode == "fair_cost_backtest":
        run_dir = run_fair_cost_backtest(settings)
        fields(logger, {"escrito": run_dir / "fair_cost_backtest.csv"})
        return
    if mode == "tune_imputation":
        from retail_forecasting.forecasting.imputation_tuning import tune_imputation_lgbm

        params_path = tune_imputation_lgbm(settings, config_path=Path(args.config))
        fields(logger, {"escrito": params_path})
        return
    if mode == "eda":
        fields(
            logger,
            {"escrito": run_eda(settings, split=args.split, config_path=Path(args.config))},
        )
        return
    if mode == "score_daily":
        artifacts = run_scoring(settings)
        assert artifacts.run_directory is not None
        fields(logger, {"escrito": artifacts.run_directory / "reorder_recommendations.csv"})
    else:
        artifacts = run_experiment(settings)
        assert artifacts.run_directory is not None
        fields(logger, {"escrito": artifacts.run_directory / "metrics_summary.csv"})


def _format_validation_error(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        if error["type"] == "greater_than":
            ctx = error.get("ctx") or {}
            message = f"{location} must be greater than {ctx.get('gt', '?')}."
        else:
            message = f"{location}: {error['msg']}"
        messages.append(f"- {message}")
    return "Invalid configuration:\n" + "\n".join(messages)


if __name__ == "__main__":
    main()
