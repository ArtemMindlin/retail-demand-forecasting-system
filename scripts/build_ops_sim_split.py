"""Build the train/eval split consumed by the operational simulation (OPS plane).

The walk-forward OPS playback streams the ``eval`` split day by day. To maximise
the number of simulated weeks we carve the simulation window out of the existing
90-day prepared panel instead of using the tiny dataset-native eval split:

    train = first ``--train-days`` days   (warm-up + initial champion training)
    eval  = the remaining days            (the streamed "production" window)

Outputs are written to ``data/processed/ops_sim/{train,eval}.parquet`` so the
canonical ``data/processed`` splits are never touched. The simulation config
(``configs/simulation.yaml``) points ``dataset.processed_panel_dir`` here.

Usage:
    python scripts/build_ops_sim_split.py --train-days 28 --n-series 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCE_PANEL = Path("data/processed/train.parquet")
OUTPUT_DIR = Path("data/processed/ops_sim")


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
        help="Prepared panel to split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write train.parquet / eval.parquet.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    panel = pd.read_parquet(args.source)
    panel["date"] = pd.to_datetime(panel["date"])

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output_dir / "train.parquet", index=False)
    eval_.to_parquet(args.output_dir / "eval.parquet", index=False)

    print(f"✅ Wrote OPS simulation split to {args.output_dir}")
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
