from __future__ import annotations

import argparse
from pathlib import Path

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
    return parser


def main() -> None:
    """Parse CLI arguments and execute the configured experiment run.

    Returns:
        None.

    Notes:
        The reporting output directory can be overridden from the command line without modifying the YAML configuration.
    """
    args = build_parser().parse_args()
    settings = load_config(args.config)
    if args.output_dir is not None:
        settings.reporting.output_dir = Path(args.output_dir)

    artifacts = run_experiment(settings)
    if artifacts.run_directory is None:
        raise RuntimeError("Run finished without a report directory.")

    print(f"Report written to: {artifacts.run_directory / 'report.md'}")


if __name__ == "__main__":
    main()
