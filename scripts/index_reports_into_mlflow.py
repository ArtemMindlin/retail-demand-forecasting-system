"""Record run directories under ``reports/`` in MLflow, which is where the dashboard looks.

Two jobs. It makes runs written before the tracking instrumentation existed visible, and it
rebuilds the index from the files if `mlflow.db` is ever lost -- the store is a gitignored
sqlite database with no backup, while the run directories are the durable copy.

Runs already recorded under the same name are skipped, so this is safe to re-run.

Usage:
    uv run python scripts/index_reports_into_mlflow.py
    uv run python scripts/index_reports_into_mlflow.py --reports reports --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retail_forecasting.tracking import (
    EXPERIMENT_EDA,
    EXPERIMENT_OPS,
    EXPERIMENT_RUNS,
    index_run_directory,
    logged_run_dirs,
)

# `sampler_ab` is a closed side study kept only as evidence for a code comment, not a run.
_NOT_RUNS = ("sampler_ab",)


def _experiment_for(run_dir: Path) -> str:
    if run_dir.name.startswith("eda_"):
        return EXPERIMENT_EDA
    # The OPS plane has its own experiment: its costs are not walk-forward costs and must not
    # land in one `search_runs` beside them.
    if (run_dir / "simulation").is_dir():
        return EXPERIMENT_OPS
    return EXPERIMENT_RUNS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    already = {
        name
        for experiment in (EXPERIMENT_RUNS, EXPERIMENT_EDA)
        for name in logged_run_dirs(experiment)
    }

    for run_dir in sorted(d for d in args.reports.iterdir() if d.is_dir()):
        if run_dir.name in _NOT_RUNS:
            print(f"  omitida (no es una corrida)  {run_dir.name}")
            continue
        if run_dir.name in already:
            print(f"  ya registrada                {run_dir.name}")
            continue
        if args.dry_run:
            print(f"  se registraría               {run_dir.name}")
            continue
        run_id = index_run_directory(run_dir, _experiment_for(run_dir))
        print(f"  registrada {str(run_id)[:8]}          {run_dir.name}")


if __name__ == "__main__":
    main()
