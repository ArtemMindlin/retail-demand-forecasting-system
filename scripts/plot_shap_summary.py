"""Render the SHAP beeswarm of the champion model used by the results chapter.

`TreeExplainer` on the LightGBM champion segfaults inside the full pipeline process,
and the crash does not reproduce outside it (loaded model, freshly fitted model, real
frame, CatBoost loaded first: all survive). `evaluation.xai` now runs the computation
in a child process so that crash costs the figure rather than the whole run, but the
figure still has to come from somewhere: this script produces it from the persisted
champion, without paying for a backtest to redraw a plot.

Reads the model the champion registry names, rebuilds the supervised frame the same
way the experiment does, and writes the beeswarm. Reproducible from artifacts, which
is what `docs/runs.md` requires of anything cited.

Usage:
    python scripts/plot_shap_summary.py [--sample-size 500] [--output <ruta>]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

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

matplotlib.use("Agg")

OUTPUT = Path("memoria/figures/shap_summary.png")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/default.yaml")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    settings = load_config(args.config)

    champion = resolve_champion_reference(
        settings, load_champion_registry(champion_registry_path(settings))
    )
    model_path = model_file_path(settings.models.models_dir, champion.backend_name)
    logger.info("campeón: %s (%s)", champion.model_name, model_path)

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
    features = frame[metadata.feature_columns]
    logger.info("frame supervisado: %d filas, %d features", len(features), features.shape[1])

    model = ConformalForecaster.load(model_path)
    shap_values = calculate_shap_values(model=model, X=features, sample_size=args.sample_size)
    if shap_values is None:
        raise SystemExit(
            "SHAP no pudo calcularse; el aviso anterior dice por qué. La figura no se escribe: "
            "sobrescribirla con una en blanco sería peor que dejar la anterior en su sitio."
        )

    # Imported here, like the pipeline does, because the module forces the Agg backend.
    from retail_forecasting.visualization.plots import render_shap_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_shap_summary(shap_values=shap_values, output_path=args.output)
    logger.info("escrito %s", args.output)


if __name__ == "__main__":
    main()
