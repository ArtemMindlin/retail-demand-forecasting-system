"""Rebuild `tuning_pareto.csv` for searches that ran before the tuner wrote it.

The tuning mode used to dump its Pareto front to `reports/figures/pareto_<backend>.csv`, a
path relative to the working directory rather than the run's own artifact directory, so the
dashboard -- which reads artifacts out of the run store -- never found a front and rendered
an empty panel. The tuner now logs `tuning_pareto.csv` into its run; this repairs the runs
that finished before it did.

Nothing here recomputes a search. Every value comes from the nested trial runs MLflow already
recorded, so the rebuilt front is the one that search actually produced.

    uv run python scripts/backfill_tuning_pareto.py --list
    uv run python scripts/backfill_tuning_pareto.py --run forecasting_lightgbm_v4
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from retail_forecasting.forecasting.forecasting_tuning import PARETO_ARTIFACT

DB = Path("mlflow.db")
TUNING_EXPERIMENT = "forecasting_tuning"


def _searches(conn: sqlite3.Connection) -> list[tuple[str, str, str, int]]:
    """Parent runs of the tuning experiment, newest first, with their trial count."""
    rows = conn.execute(
        """
        SELECT r.run_uuid, r.name, r.artifact_uri,
               (SELECT COUNT(*) FROM tags t
                 WHERE t.key = 'mlflow.parentRunId' AND t.value = r.run_uuid) AS trials
          FROM runs r
          JOIN experiments e ON e.experiment_id = r.experiment_id
         WHERE e.name = ? AND r.name NOT LIKE 'trial-%'
         ORDER BY r.start_time DESC
        """,
        (TUNING_EXPERIMENT,),
    ).fetchall()
    return [(uuid, name, uri, n) for uuid, name, uri, n in rows if n]


def _front(conn: sqlite3.Connection, parent: str) -> pd.DataFrame:
    """One row per trial of ``parent``, in the schema the dashboard reads.

    `cost` is the search's first objective and the pinball loss it minimises; the column is
    named for what it measures rather than for the role it plays inside Optuna.
    """
    children = [
        r[0]
        for r in conn.execute(
            "SELECT run_uuid FROM tags WHERE key = 'mlflow.parentRunId' AND value = ?",
            (parent,),
        )
    ]
    if not children:
        return pd.DataFrame()

    placeholders = ",".join("?" * len(children))
    metrics = conn.execute(
        f"SELECT run_uuid, key, value FROM metrics WHERE run_uuid IN ({placeholders})",  # noqa: S608
        children,
    ).fetchall()
    params = conn.execute(
        f"SELECT run_uuid, key, value FROM params WHERE run_uuid IN ({placeholders})",  # noqa: S608
        children,
    ).fetchall()
    names = dict(
        conn.execute(
            f"SELECT run_uuid, name FROM runs WHERE run_uuid IN ({placeholders})",  # noqa: S608
            children,
        ).fetchall()
    )

    by_run: dict[str, dict[str, object]] = {run: {} for run in children}
    for run, key, value in metrics:
        by_run[run][key] = value
    for run, key, value in params:
        by_run[run][key] = value

    records = []
    for run, fields in by_run.items():
        if "cost" not in fields or "winkler" not in fields:
            continue
        record: dict[str, object] = {
            "trial_number": int(str(names[run]).removeprefix("trial-")),
            "pinball_loss": float(fields["cost"]),  # type: ignore[arg-type]
            "winkler_score": float(fields["winkler"]),  # type: ignore[arg-type]
            "is_on_front": bool(float(fields.get("is_on_front", 0.0))),  # type: ignore[arg-type]
        }
        record.update(
            {k: v for k, v in fields.items() if k not in {"cost", "winkler", "is_on_front"}}
        )
        records.append(record)

    frame = pd.DataFrame(records).sort_values("trial_number").reset_index(drop=True)
    # The winner is the cheapest trial ON the front, the same rule the search applies.
    on_front = frame[frame["is_on_front"]]
    best = on_front["pinball_loss"].idxmin() if not on_front.empty else None
    frame["is_selected"] = frame.index == best
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Search to repair, by run name.")
    parser.add_argument("--list", action="store_true", help="List searches and exit.")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    searches = _searches(conn)

    if args.list or not args.run:
        for uuid, name, uri, n in searches:
            has = (Path(uri) / PARETO_ARTIFACT).exists()
            print(f"  {'OK ' if has else '-- '} {name:30s} {n:4d} trials  {uuid}")
        if not args.run:
            print("\nPass --run <name> to rebuild one.")
        return

    match = [s for s in searches if s[1] == args.run]
    if not match:
        raise SystemExit(f"No search named {args.run!r}. Use --list.")
    uuid, name, uri, _ = match[0]

    frame = _front(conn, uuid)
    if frame.empty:
        raise SystemExit(f"{name} has no scored trials to rebuild from.")

    out = Path(uri) / PARETO_ARTIFACT
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"{name}: {len(frame)} trials, {int(frame['is_on_front'].sum())} on the front -> {out}")


if __name__ == "__main__":
    main()
