"""Rebuild the run index from the artifacts on disk, when `mlflow.db` is gone or damaged.

The store is now the only copy of a run, and its two halves fail independently: `mlruns/` is
a directory tree that a backup catches, while `mlflow.db` is a gitignored sqlite database that
nothing backs up. Losing it would leave 70-odd MB of artifacts under UUID directories with no
way to tell which run each one was -- which is why `open_run_directory` writes an
`mlflow_run.json` into every artifact directory. This is the script that reads them back.

What returns is identity and files: the run's name, its experiment, and every artifact. What
does not is the params and metrics, which lived only in the lost database. A rebuilt run is
enough for the dashboard and for `resolve_run_dir`; it is not enough to compare runs by MAE.

Usage:
    uv run python scripts/rebuild_mlflow_index.py --dry-run
    uv run python scripts/rebuild_mlflow_index.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_forecasting.tracking import (
    EXPERIMENT_EDA,
    EXPERIMENT_OPS,
    EXPERIMENT_RUNS,
    RUN_IDENTITY_FILE,
    index_run_directory,
    logged_run_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlruns", type=Path, default=Path("mlruns"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    known = {
        name
        for experiment in (EXPERIMENT_RUNS, EXPERIMENT_EDA, EXPERIMENT_OPS)
        for name in logged_run_dirs(experiment)
    }

    for identity_file in sorted(args.mlruns.glob(f"*/artifacts/{RUN_IDENTITY_FILE}")):
        run_dir = identity_file.parent
        identity = json.loads(identity_file.read_text(encoding="utf-8"))
        name, experiment = identity["run_name"], identity["experiment"]

        if name in known:
            print(f"  ya en el índice   {name}")
            continue
        if args.dry_run:
            print(f"  se recuperaría    {name}  ({experiment})")
            continue
        run_id = index_run_directory(run_dir, experiment, run_name=name)
        print(f"  recuperada        {name}  -> {str(run_id)[:8]}")


if __name__ == "__main__":
    main()
