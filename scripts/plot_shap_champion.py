"""Render the SHAP beeswarm of the CHAMPION model for the results chapter.

Standalone, and deliberately not part of the experiment run. Computing SHAP for the
LightGBM champion inside the pipeline process segfaults (exit 139) after the fold loop has
trained both backends; the same call succeeds in isolation against the persisted model, on
the same frame and the same tuned hyperparameters. A SIGSEGV cannot be caught by the
`try/except` in `evaluation/xai.py`, so it takes the whole run down with it, artifacts
included. Until that crash is understood, the figure comes from here.

It reads the champion from the registry, loads its persisted `.pkl`, and rebuilds the
supervised frame from the configured panel, so the figure is reproducible from committed
inputs rather than being a hand-made PNG.

Usage:
    python scripts/plot_shap_champion.py --config configs/experiment/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from retail_forecasting.config import load_config
from retail_forecasting.data.censorship import LatentDemandImputer
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.drift.regime import label_all_regimes
from retail_forecasting.evaluation.xai import calculate_shap_values
from retail_forecasting.features.engineering import build_supervised_frame
from retail_forecasting.forecasting.pipeline import (
    champion_registry_path,
    load_champion_registry,
    resolve_champion_reference,
)
from retail_forecasting.models.conformal import ConformalForecaster
from retail_forecasting.utils.io import model_file_path
from retail_forecasting.visualization.plots import render_shap_summary

OUTPUT = Path("memoria/figures/shap_summary.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/default.yaml")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_config(args.config)

    champion = resolve_champion_reference(
        settings, load_champion_registry(champion_registry_path(settings))
    )
    model_path = model_file_path(settings.models.models_dir, champion.backend_name)
    if not model_path.exists():
        raise SystemExit(f"No saved champion at {model_path}. Run a backtest or retrain first.")
    model = ConformalForecaster.load(model_path)

    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    strategy = settings.preprocessing.imputation_strategy
    if strategy != "none":
        panel = LatentDemandImputer(
            strategy=strategy,
            model_path=settings.models.models_dir / settings.models.imputation_params_filename,
        ).impute(panel)

    frame, metadata = build_supervised_frame(
        panel=label_all_regimes(panel),
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )

    shap_values = calculate_shap_values(
        model=model,
        X=frame[metadata.feature_columns],
        sample_size=args.sample_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_shap_summary(shap_values=shap_values, output_path=args.output)
    print(f"✅ Wrote {args.output}")
    print(f"   champion: {champion.model_name} ({champion.backend_name}), source {champion.source}")
    print(f"   frame   : {len(frame)} rows, {len(metadata.feature_columns)} features")


if __name__ == "__main__":
    main()
