"""Generate the multi-objective Pareto front (Pinball Loss vs. Winkler Score) plot and CSV artifact."""

from pathlib import Path

import pandas as pd

from retail_forecasting.config import load_config
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.features.engineering import build_supervised_frame
from retail_forecasting.models.optimization import HyperparameterTuner
from retail_forecasting.visualization.plots import render_pareto_front


def main() -> None:
    settings = load_config("configs/experiment/default.yaml")
    panel = load_prepared_panel(settings.dataset, settings.preprocessing)
    supervised, _ = build_supervised_frame(panel, settings.features, settings.dataset.horizon)

    feature_cols = [
        col
        for col in supervised.columns
        if col
        not in (
            "series_id",
            "date",
            "target_lead_time_demand",
            "y_true",
            "stockout_hours",
            "stockout_regime",
            "velocity_regime",
            "promo_regime",
            "seasonal_regime",
        )
    ]

    tuner = HyperparameterTuner(settings, n_trials=20)
    result = tuner.tune_boosting(supervised, feature_cols)

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if result.pareto_front:
        pareto_df = pd.DataFrame([trial.model_dump() for trial in result.pareto_front])
        pareto_df.to_csv(output_dir / "tuning_pareto.csv", index=False)
        render_pareto_front(pareto_df, output_dir / "pareto_front.png")
        print(f"Generated {output_dir / 'pareto_front.png'} and {output_dir / 'tuning_pareto.csv'}")


if __name__ == "__main__":
    main()
