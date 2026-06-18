from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from retail_forecasting.config import Settings
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.data.quality import (
    raise_on_blocking_data_quality,
    validate_prepared_panel,
)
from retail_forecasting.drift import label_all_regimes
from retail_forecasting.features.engineering import build_inference_frame_with_fallback
from retail_forecasting.forecasting.pipeline import (
    _build_scoring_predictions,
    train_and_save_champion,
)
from retail_forecasting.inventory.cost_profiles import build_series_cost_profile
from retail_forecasting.inventory.newsvendor import attach_inventory_costs
from retail_forecasting.models.conformal import ConformalForecaster


@dataclass
class OperationalSimulationArtifacts:
    """Outputs produced by ``run_operational_simulation``."""

    predictions_by_day: pd.DataFrame
    cadence_summary: pd.DataFrame
    retrain_events: list[dict[str, Any]]
    run_directory: Path
    cumulative_cost_plot: Path | None = None
    report_path: Path | None = None
    cadence_models: dict[str, Path] = field(default_factory=dict)


def _cadence_label(cadence: int | None) -> str:
    return "never" if cadence is None else f"every_{cadence}d"


def _reveal_actuals(
    panel: pd.DataFrame,
    decision_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Sum observed demand per series across [decision_date, decision_date + H - 1]."""
    end_exclusive = decision_date + pd.Timedelta(days=horizon)
    window = panel[(panel["date"] >= decision_date) & (panel["date"] < end_exclusive)]
    if window.empty:
        return pd.DataFrame(columns=["series_id", "y_true", "actuals_days_observed"])
    grouped = window.groupby("series_id", as_index=False).agg(
        y_true=("observed_demand", "sum"),
        actuals_days_observed=("observed_demand", "size"),
    )
    return grouped


def _score_one_step(
    panel: pd.DataFrame,
    model_path: Path,
    settings: Settings,
    series_cost_profile: pd.DataFrame | None,
) -> pd.DataFrame:
    """Reproduce the run_scoring inference path on an in-memory panel slice."""
    prepared = label_all_regimes(panel)
    inference_frame, inference_metadata = build_inference_frame_with_fallback(
        prepared,
        settings.features,
        horizon=settings.dataset.horizon,
    )
    model = ConformalForecaster.load(model_path)
    return _build_scoring_predictions(
        inference_frame=inference_frame,
        feature_columns=inference_metadata.feature_columns,
        model=model,
        settings=settings,
        series_cost_profile=series_cost_profile,
    )


def _setup_cadence_models(
    settings: Settings,
    train_panel: pd.DataFrame,
    sim_models_root: Path,
) -> tuple[dict[str, Path], dict[str, int | None]]:
    """Train the initial champion and seed one model copy per retrain cadence.

    Returns ``(cadence_paths, cadence_int)`` mapping each cadence label to its
    model file and to its retrain period (``None`` = never retrain).
    """
    print(f"🔧 Training initial champion on train split ({len(train_panel)} rows)...")
    base_model_path = train_and_save_champion(
        settings, train_panel, models_dir=sim_models_root / "initial"
    )
    print(f"   ↳ saved to {base_model_path}")

    cadence_paths: dict[str, Path] = {}
    cadence_int: dict[str, int | None] = {}
    for cadence in settings.simulation.retrain_cadences:
        label = _cadence_label(cadence)
        cadence_dir = sim_models_root / label
        cadence_dir.mkdir(parents=True, exist_ok=True)
        cadence_model_path = cadence_dir / base_model_path.name
        shutil.copy2(base_model_path, cadence_model_path)
        cadence_paths[label] = cadence_model_path
        cadence_int[label] = cadence
    return cadence_paths, cadence_int


def _run_streaming_loop(
    eval_dates: list[Any],
    combined_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    cadence_paths: dict[str, Path],
    cadence_int: dict[str, int | None],
    horizon: int,
    settings: Settings,
    series_cost_profile: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Stream eval days: for each day×cadence score, retrain on schedule, cost the window.

    Returns ``(predictions_by_day, retrain_events)``.
    """
    retrain_events: list[dict[str, Any]] = []
    rows: list[pd.DataFrame] = []

    print(
        f"▶️  Streaming {len(eval_dates)} eval days across "
        f"{len(cadence_paths)} cadences (horizon={horizon})..."
    )

    for day_index, current_date in enumerate(eval_dates):
        available_history = combined_panel[combined_panel["date"] < current_date]
        actuals = _reveal_actuals(eval_panel, current_date, horizon)
        actuals_complete = not actuals.empty and bool(
            (actuals["actuals_days_observed"] >= horizon).all()
        )

        for label, model_path in cadence_paths.items():
            cadence_k = cadence_int[label]
            retrained_this_step = False
            if cadence_k is not None and (day_index + 1) % cadence_k == 0:
                t0 = time.perf_counter()
                train_and_save_champion(settings, available_history, models_dir=model_path.parent)
                retrain_events.append(
                    {
                        "cadence": label,
                        "day_index": day_index,
                        "decision_date": current_date.isoformat(),
                        "duration_seconds": round(time.perf_counter() - t0, 3),
                        "history_rows": int(len(available_history)),
                    }
                )
                retrained_this_step = True

            preds = _score_one_step(
                panel=available_history,
                model_path=model_path,
                settings=settings,
                series_cost_profile=series_cost_profile,
            )
            preds = preds.merge(
                actuals[["series_id", "y_true"]],
                on="series_id",
                how="left",
                suffixes=("", "_actual"),
            )
            if "y_true_actual" in preds.columns:
                preds["y_true"] = preds["y_true_actual"]
                preds = preds.drop(columns=["y_true_actual"])

            preds = attach_inventory_costs(
                preds, settings.inventory, series_cost_profile=series_cost_profile
            )
            preds["decision_date"] = current_date
            preds["day_index"] = day_index
            preds["cadence"] = label
            preds["retrained_this_step"] = retrained_this_step
            preds["actuals_complete"] = actuals_complete
            rows.append(preds)

        if (day_index + 1) % 5 == 0 or day_index == len(eval_dates) - 1:
            print(
                f"   day {day_index + 1}/{len(eval_dates)} ({current_date.date()}) "
                f"— retrains so far: {len(retrain_events)}"
            )

    return pd.concat(rows, ignore_index=True), retrain_events


def _persist_simulation_outputs(
    sim_root: Path,
    predictions_by_day: pd.DataFrame,
    cadence_summary: pd.DataFrame,
    retrain_events: list[dict[str, Any]],
    settings: Settings,
    eval_dates: list[Any],
) -> tuple[Path | None, Path]:
    """Write simulation artifacts to disk; return ``(plot_path, report_path)``."""
    predictions_by_day.to_parquet(sim_root / "predictions_by_day.parquet", index=False)
    cadence_summary.to_csv(sim_root / "cadence_summary.csv", index=False)
    (sim_root / "retrain_events.json").write_text(
        json.dumps(retrain_events, indent=2), encoding="utf-8"
    )

    plot_path: Path | None = None
    if settings.simulation.make_plots:
        plot_path = _plot_cumulative_cost(predictions_by_day, retrain_events, sim_root)

    report_path = _write_simulation_report(
        sim_root, cadence_summary, retrain_events, settings, eval_dates
    )
    return plot_path, report_path


def run_operational_simulation(settings: Settings) -> OperationalSimulationArtifacts:
    """Stream the eval split as if it were daily production data.

    Trains an initial champion on the train split, then iterates over eval
    dates. For each day every configured cadence scores its model, the realized
    cost is computed against the revealed window, and retraining is triggered
    according to the cadence period.
    """
    print("📥 Loading train and eval splits for operational simulation...")
    train_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    eval_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="eval",
    )
    raise_on_blocking_data_quality(validate_prepared_panel(train_panel, settings))
    raise_on_blocking_data_quality(validate_prepared_panel(eval_panel, settings))

    train_panel = train_panel.copy()
    eval_panel = eval_panel.copy()
    train_panel["date"] = pd.to_datetime(train_panel["date"])
    eval_panel["date"] = pd.to_datetime(eval_panel["date"])

    horizon = settings.dataset.horizon
    eval_dates = sorted(eval_panel["date"].unique())
    if settings.simulation.simulation_days is not None:
        eval_dates = eval_dates[: settings.simulation.simulation_days]
    if not eval_dates:
        raise ValueError("Eval split contained no usable dates for simulation.")

    sim_root = Path(settings.reporting.output_dir) / settings.reporting.run_name / "simulation"
    sim_root.mkdir(parents=True, exist_ok=True)
    sim_models_root = sim_root / "models"
    sim_models_root.mkdir(parents=True, exist_ok=True)

    cadence_paths, cadence_int = _setup_cadence_models(settings, train_panel, sim_models_root)

    # Combined panel keeps lag continuity across the train→eval boundary.
    combined_panel = pd.concat([train_panel, eval_panel], ignore_index=True)
    combined_panel = combined_panel.sort_values(["series_id", "date"]).reset_index(drop=True)

    series_cost_profile = None
    if settings.inventory.use_series_costs:
        series_cost_profile = build_series_cost_profile(
            label_all_regimes(train_panel), settings.inventory
        )

    predictions_by_day, retrain_events = _run_streaming_loop(
        eval_dates=eval_dates,
        combined_panel=combined_panel,
        eval_panel=eval_panel,
        cadence_paths=cadence_paths,
        cadence_int=cadence_int,
        horizon=horizon,
        settings=settings,
        series_cost_profile=series_cost_profile,
    )
    cadence_summary = _summarize_cadences(predictions_by_day, retrain_events)

    plot_path, report_path = _persist_simulation_outputs(
        sim_root, predictions_by_day, cadence_summary, retrain_events, settings, eval_dates
    )

    print(
        f"✅ Operational simulation complete. Outputs in {sim_root} "
        f"({len(retrain_events)} retrain events)"
    )

    return OperationalSimulationArtifacts(
        predictions_by_day=predictions_by_day,
        cadence_summary=cadence_summary,
        retrain_events=retrain_events,
        run_directory=sim_root,
        cumulative_cost_plot=plot_path,
        report_path=report_path,
        cadence_models=cadence_paths,
    )


def _summarize_cadences(
    predictions_by_day: pd.DataFrame,
    retrain_events: list[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate per-cadence performance over the fully-revealed window."""
    complete = predictions_by_day[predictions_by_day["actuals_complete"]].copy()
    rows = []
    retrain_counts: dict[str, int] = {}
    retrain_durations: dict[str, list[float]] = {}
    for event in retrain_events:
        retrain_counts[event["cadence"]] = retrain_counts.get(event["cadence"], 0) + 1
        retrain_durations.setdefault(event["cadence"], []).append(event["duration_seconds"])

    for cadence in predictions_by_day["cadence"].unique():
        group = complete[complete["cadence"] == cadence]
        total_cost = float(group["total_cost"].sum()) if not group.empty else 0.0
        stockout_units = float(group["stockout_units"].sum()) if not group.empty else 0.0
        overstock_units = float(group["overstock_units"].sum()) if not group.empty else 0.0
        observations = int(len(group))
        served = (
            float((group["y_true"] - group["stockout_units"]).sum()) if not group.empty else 0.0
        )
        demand = float(group["y_true"].sum()) if not group.empty else 0.0
        fill_rate = served / demand if demand > 0 else float("nan")
        service_level = (
            float((group["stockout_units"] == 0).mean()) if not group.empty else float("nan")
        )
        durations = retrain_durations.get(cadence, [])
        rows.append(
            {
                "cadence": cadence,
                "n_observations": observations,
                "total_cost": total_cost,
                "mean_cost_per_observation": (
                    total_cost / observations if observations else float("nan")
                ),
                "total_stockout_units": stockout_units,
                "total_overstock_units": overstock_units,
                "fill_rate": fill_rate,
                "service_level": service_level,
                "n_retrains": retrain_counts.get(cadence, 0),
                "mean_retrain_seconds": (sum(durations) / len(durations) if durations else 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)


def _plot_cumulative_cost(
    predictions_by_day: pd.DataFrame,
    retrain_events: list[dict[str, Any]],
    sim_root: Path,
) -> Path | None:
    try:
        import matplotlib
        import matplotlib.dates as mdates

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    complete = predictions_by_day[predictions_by_day["actuals_complete"]]
    if complete.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for cadence, group in complete.groupby("cadence"):
        daily = group.groupby("decision_date")["total_cost"].sum().sort_index()
        cumulative = daily.cumsum()
        ax.plot(
            cumulative.index.to_numpy(),
            cumulative.to_numpy(),
            label=f"cadence={cadence}",
            marker=".",
        )

    for event in retrain_events:
        ax.axvline(
            float(mdates.date2num(pd.Timestamp(event["decision_date"]))),
            color="grey",
            alpha=0.15,
            linestyle="--",
            linewidth=0.8,
        )

    ax.set_title("Cumulative inventory cost by retrain cadence")
    ax.set_xlabel("Decision date")
    ax.set_ylabel("Cumulative cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plot_path = sim_root / "cumulative_cost.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _write_simulation_report(
    sim_root: Path,
    cadence_summary: pd.DataFrame,
    retrain_events: list[dict[str, Any]],
    settings: Settings,
    eval_dates: list[pd.Timestamp],
) -> Path:
    horizon = settings.dataset.horizon
    table = _format_markdown_table(cadence_summary)
    lines = [
        "# Operational simulation report",
        "",
        f"- Streaming window: {len(eval_dates)} days "
        f"({eval_dates[0].date()} → {eval_dates[-1].date()})",
        f"- Horizon: {horizon} days",
        f"- Cadences evaluated: {list(cadence_summary['cadence'])}",
        f"- Total retrain events: {len(retrain_events)}",
        "",
        "## Cadence summary",
        "",
        table,
        "",
        "## Interpretation",
        "",
        "Lower `total_cost` over the same window indicates a better operational",
        "policy. Compare against the `never` baseline to quantify the value of",
        "retraining at the listed cadence.",
    ]
    report_path = sim_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    headers = list(df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        body_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_row, separator, *body_rows])
