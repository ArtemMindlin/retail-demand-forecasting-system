from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from retail_forecasting.config import load_config
from retail_forecasting.forecasting.pipeline import run_experiment


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
        default="configs/default.yaml",
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
    if args.output_dir is not None:
        reporting_updates["output_dir"] = Path(args.output_dir)
    if args.run_name is not None:
        reporting_updates["run_name"] = args.run_name

    if reporting_updates:
        new_reporting = settings.reporting.model_copy(update=reporting_updates)
        settings = settings.model_copy(update={"reporting": new_reporting})

    artifacts = run_experiment(settings)
    if artifacts.run_directory is None:
        raise RuntimeError("Run finished without a report directory.")

    print(f"Report written to: {artifacts.run_directory / 'report.md'}")


def _format_validation_error(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        if error["type"] == "greater_than":
            message = f"{location} must be greater than {error['ctx']['gt']}."
        else:
            message = f"{location}: {error['msg']}"
        messages.append(f"- {message}")
    return "Invalid configuration:\n" + "\n".join(messages)


if __name__ == "__main__":
    main()
