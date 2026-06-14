"""Backfill ``drift_report.json`` for the latest API-visible run.

Loads the champion model and a sample of the prepared panel, computes SHAP
importance, and writes a real PSI drift report into the run directory the API
serves — without re-running the full experiment pipeline. Useful for runs that
predate the drift-report artifact, so the dashboard drift panel shows real
Population Stability Index values immediately.

Usage:
    python scripts/generate_drift_report.py [config_path] [n_series]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from retail_forecasting.api.main import _get_latest_run_path
from retail_forecasting.config import load_config
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.drift import label_all_regimes
from retail_forecasting.drift.psi import build_feature_drift_report
from retail_forecasting.evaluation.xai import calculate_shap_values
from retail_forecasting.features.engineering import build_supervised_frame
from retail_forecasting.models.conformal import ConformalForecaster


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml"
    n_series = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    settings = load_config(config_path)

    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    panel["date"] = pd.to_datetime(panel["date"])

    # Sample a subset of series (full history each) so lags stay valid while
    # keeping SHAP and PSI fast.
    sample_ids = panel["series_id"].drop_duplicates().head(n_series)
    panel = panel[panel["series_id"].isin(sample_ids)].copy()

    prepared = label_all_regimes(panel)
    frame, metadata = build_supervised_frame(
        prepared, settings.features, horizon=settings.dataset.horizon
    )

    model_path = settings.models.models_dir / f"{settings.business.champion_backend_name}.pkl"
    model = ConformalForecaster.load(model_path)
    print(f"Loaded champion: {model_path} | sample rows: {len(frame)}")

    shap_values = calculate_shap_values(model=model, X=frame.loc[:, metadata.feature_columns])
    report = build_feature_drift_report(supervised_frame=frame, shap_values=shap_values)

    run_dir: Path = _get_latest_run_path()
    out = run_dir / "drift_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {len(report)} feature(s) to {out}")
    for item in report:
        print(f"  {item['name']:<28} PSI={item['psi']:.3f}  {item['status']}")


if __name__ == "__main__":
    main()
