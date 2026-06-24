from __future__ import annotations

import argparse
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from retail_forecasting.config import load_config
from retail_forecasting.contracts.contracts_config import RunMode
from retail_forecasting.forecasting.pipeline import (
    run_experiment,
    run_fair_cost_backtest,
    run_imputation_comparison,
    run_retrain,
    run_scoring,
)
from retail_forecasting.simulation import run_operational_simulation


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
        default="configs/experiment.yaml",
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the reporting output directory.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the experiment run name.",
    )
    parser.add_argument(
        "--run-mode",
        default=None,
        choices=list(get_args(RunMode)),
        help="Optional override for the execution mode.",
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
    try:
        settings = load_config(args.config)
    except ValidationError as exc:
        raise SystemExit(_format_validation_error(exc)) from None
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    # Handle Reporting overrides (output_dir and run_name)
    reporting_updates = {}
    project_updates = {}
    if args.output_dir is not None:
        reporting_updates["output_dir"] = Path(args.output_dir)
    if args.run_name is not None:
        reporting_updates["run_name"] = args.run_name
    if args.run_mode is not None:
        project_updates["run_mode"] = args.run_mode

    if reporting_updates:
        new_reporting = settings.reporting.model_copy(update=reporting_updates)
        settings = settings.model_copy(update={"reporting": new_reporting})
    if project_updates:
        new_project = settings.project.model_copy(update=project_updates)
        settings = settings.model_copy(update={"project": new_project})

    mode = settings.project.run_mode
    if mode == "experiment" and settings.preprocessing.compare_imputation:
        run_dir = run_imputation_comparison(settings)
        print(f"Imputation comparison written to: {run_dir / 'latent_strategies.csv'}")
        return
    if mode == "retrain":
        run_retrain(settings)
        return
    if mode == "simulate_ops":
        sim_artifacts = run_operational_simulation(settings)
        print(f"Simulation outputs written to: {sim_artifacts.run_directory}")
        return
    if mode == "fair_cost_backtest":
        run_dir = run_fair_cost_backtest(settings)
        print(f"Fair-cost backtest written to: {run_dir / 'fair_cost_backtest.csv'}")
        return
    if mode == "score_daily":
        artifacts = run_scoring(settings)
        assert artifacts.run_directory is not None
        print(
            "Operational outputs written to: "
            f"{artifacts.run_directory / 'reorder_recommendations.csv'}"
        )
    else:
        artifacts = run_experiment(settings)
        assert artifacts.run_directory is not None
        print(f"Report written to: {artifacts.run_directory / 'report.md'}")


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
