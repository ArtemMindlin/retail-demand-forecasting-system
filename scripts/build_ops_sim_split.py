"""Build the train/eval split consumed by the operational simulation (OPS plane).

The walk-forward OPS playback streams the ``eval`` split day by day. To maximise
the number of simulated weeks we carve the simulation window out of the existing
90-day prepared panel instead of using the tiny dataset-native eval split:

    train = first ``--train-days`` days   (warm-up + initial champion training)
    eval  = the remaining days            (the streamed "production" window)

Outputs are written to ``data/processed/ops_sim/`` under the hashed names that
``load_prepared_panel`` derives from the simulation config, so the
canonical ``data/processed`` splits are never touched. The simulation config
(``configs/simulate_ops/default.yaml``) points ``dataset.processed_panel_dir`` here.

The source panel decides the series universe and the calendar of the whole OPS
window, so it is validated and its provenance is recorded in the ``.built`` stamp
rather than left implicit. ``data/processed/train.parquet`` is the unhashed legacy
panel; hashed panels (``train_<key>.parquet``) belong to a specific config, so pass
``--source`` explicitly when the OPS window must match one of them.

Usage:
    python scripts/build_ops_sim_split.py --train-days 28 --n-series 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retail_forecasting.config import load_config
from retail_forecasting.data.dataset import panel_cache_filename

SOURCE_PANEL = Path("data/processed/train.parquet")
OUTPUT_DIR = Path("data/processed/ops_sim")
SIM_CONFIG = Path("configs/simulate_ops/default.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-days",
        type=int,
        default=28,
        help=(
            "Days of history reserved for warm-up + initial training. The rest "
            "becomes the streamed eval window. Minimum sensible value ~= "
            "warmup(7) + calibration(7) + a few trainable days."
        ),
    )
    parser.add_argument(
        "--n-series",
        type=int,
        default=100,
        help="Number of series to keep (subset for simulation speed).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_PANEL,
        help=(
            "Prepared panel to split. Defines the series universe and the calendar "
            "of the simulated window, so it is recorded in the .built stamp."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write the split.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SIM_CONFIG,
        help="Simulation config whose dataset settings name the output files.",
    )
    return parser


REQUIRED_COLUMNS = ("date", "series_id", "observed_demand", "stockout_hours")


def _load_source_panel(source: Path) -> pd.DataFrame:
    """Read the source panel, failing loudly instead of splitting something wrong."""
    if not source.exists():
        candidates = sorted(p.name for p in OUTPUT_DIR.parent.glob("train*.parquet"))
        raise SystemExit(
            f"Source panel not found: {source}\n"
            f"Prepared panels available in {OUTPUT_DIR.parent}/: "
            f"{', '.join(candidates) if candidates else '(none — run the pipeline first)'}"
        )
    panel = pd.read_parquet(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        raise SystemExit(f"{source} is not a canonical prepared panel; missing: {missing}")
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def main() -> None:
    args = build_parser().parse_args()

    panel = _load_source_panel(args.source)

    # Keep the highest-volume series so the demo has lively, non-degenerate SKUs.
    volume = panel.groupby("series_id")["observed_demand"].sum().sort_values(ascending=False)
    keep_series = volume.head(args.n_series).index
    panel = panel[panel["series_id"].isin(keep_series)].copy()

    dates = sorted(panel["date"].unique())
    if args.train_days >= len(dates):
        raise SystemExit(
            f"--train-days={args.train_days} but the panel only has {len(dates)} days."
        )
    cutoff = dates[args.train_days]  # first eval date (exclusive end of train)

    train = panel[panel["date"] < cutoff].copy()
    eval_ = panel[panel["date"] >= cutoff].copy()

    # Written under the names load_prepared_panel derives from the simulation
    # config, so the OPS plane finds this split instead of rebuilding one from raw.
    settings = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_name = panel_cache_filename(settings.dataset, "train")
    eval_name = panel_cache_filename(settings.dataset, "eval")
    train.to_parquet(args.output_dir / train_name, index=False)
    eval_.to_parquet(args.output_dir / eval_name, index=False)
    # Stamp file: the Makefile depends on this rather than on the parquet names,
    # which carry a hash of the dataset settings and change with the config. It also
    # records how the split was carved, so a run's window can be traced back.
    (args.output_dir / ".built").write_text(
        "\n".join(
            [
                train_name,
                eval_name,
                f"# source={args.source}",
                f"# train_days={args.train_days} n_series={panel['series_id'].nunique()}",
                f"# dates={pd.Timestamp(dates[0]).date()}..{pd.Timestamp(dates[-1]).date()}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"✅ Wrote OPS backtest split to {args.output_dir}")
    print(f"   source      : {args.source}")
    print(f"   files       : {train_name} | {eval_name}")
    print(f"   series kept : {panel['series_id'].nunique()}")
    print(
        f"   train       : {len(train):,} rows | "
        f"{pd.Timestamp(dates[0]).date()} → {pd.Timestamp(cutoff).date()} "
        f"({args.train_days} days)"
    )
    print(
        f"   eval        : {len(eval_):,} rows | "
        f"{pd.Timestamp(cutoff).date()} → {pd.Timestamp(dates[-1]).date()} "
        f"({len(dates) - args.train_days} days, ~{(len(dates) - args.train_days) // 7} weeks)"
    )


if __name__ == "__main__":
    main()
