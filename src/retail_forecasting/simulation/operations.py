"""Rolling-origin production backtest (the OPS plane).

What this module does: replays the eval split one decision date at a time, and at
every origin it scores the champion, decides a single-period Newsvendor order and
costs it against the demand that the window later revealed. Retrain cadences are
compared over the same origins.

What it deliberately does **not** do: it is not an inventory-state simulation.
There is no stock carried between periods, no order pipeline, no lead time and no
lost-sales propagation -- each origin is an independent single-period decision.
Costs are therefore comparable *across cadences* but are not the cost a real
replenishment policy would accumulate.

Two further scope limits worth remembering when reading the numbers:

- truth is ``observed_demand``, i.e. censored on stockout days. The champion
  forecasts censored demand and is scored against censored demand, which is
  self-consistent but understates the true shortage cost. This plane never
  applies ``LatentDemandImputer``.
- aggregation uses a non-overlapping origin grid (one origin every ``horizon``
  days). Scoring runs daily, so summing every daily origin would count each day
  of demand ``horizon`` times over.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
from retail_forecasting.inventory.newsvendor import attach_inventory_costs
from retail_forecasting.models.conformal import ConformalForecaster
from retail_forecasting.tracking import EXPERIMENT_OPS, log_ops_metadata, open_run_directory
from retail_forecasting.utils.logging import fields, get_logger, rule, thousands

logger = get_logger(__name__)


@dataclass
class OperationalSimulationArtifacts:
    """Outputs produced by ``run_operational_simulation``."""

    predictions_by_day: pd.DataFrame
    cadence_summary: pd.DataFrame
    retrain_events: list[dict[str, Any]]
    run_directory: Path
    cadence_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    cumulative_cost_plot: Path | None = None
    report_path: Path | None = None
    cadence_models: dict[str, Path] = field(default_factory=dict)


BASELINE_CADENCE_LABEL = "never"
# Origins below this count make the cadence comparison anecdotal: the clustered
# bootstrap has too few clusters for its interval to mean much.
MIN_ORIGINS_FOR_INFERENCE = 8
N_BOOTSTRAP_RESAMPLES = 2000


def _cadence_label(cadence: int | None) -> str:
    return "never" if cadence is None else f"every_{cadence}d"


def _independent_origins(predictions_by_day: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rows on the non-overlapping origin grid whose actuals fully landed.

    Scoring runs every day, so consecutive origins share ``horizon - 1`` days of
    demand. Keeping one origin every ``horizon`` days makes the summed cost an
    actual cost over the window instead of a ``horizon``-fold overcount, and it
    matches the weekly grid the dashboard plays back.
    """
    if predictions_by_day.empty:
        return predictions_by_day
    on_grid = predictions_by_day["day_index"] % horizon == 0
    return predictions_by_day[on_grid & predictions_by_day["actuals_complete"]].copy()


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


def _build_origin_frame(
    panel: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the one-row-per-series inference origin for a decision date.

    Feature construction does not depend on the model, so this runs once per
    decision date and is reused by every cadence.
    """
    prepared = label_all_regimes(panel)
    inference_frame, inference_metadata = build_inference_frame_with_fallback(
        prepared,
        settings.features,
        horizon=settings.dataset.horizon,
    )
    return inference_frame, inference_metadata.feature_columns


def _score_one_step(
    inference_frame: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
    settings: Settings,
) -> pd.DataFrame:
    """Reproduce the run_scoring inference path for one model at one origin.

    `data_strategy` is pinned to the observed signal: this plane never applies
    `LatentDemandImputer`, whatever the config says.
    """
    model = ConformalForecaster.load(model_path)
    return _build_scoring_predictions(
        inference_frame=inference_frame,
        feature_columns=feature_columns,
        model=model,
        settings=settings,
        data_strategy="Observed",
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
    fields(logger, {"campeón inicial": f"entrenando sobre {thousands(len(train_panel))} filas"})
    base_model_path = train_and_save_champion(
        settings, train_panel, models_dir=sim_models_root / "initial"
    )
    fields(logger, {"guardado": str(base_model_path)})

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
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Stream eval days: for each day×cadence score, retrain on schedule, cost the window.

    Returns ``(predictions_by_day, retrain_events)``.
    """
    retrain_events: list[dict[str, Any]] = []
    rows: list[pd.DataFrame] = []

    fields(
        logger,
        {
            "transmitiendo": f"{len(eval_dates)} días de eval",
            "cadencias": f"{len(cadence_paths)} · horizonte {horizon}d",
        },
    )

    for day_index, current_date in enumerate(eval_dates):
        # Origin convention (features/engineering.py): a row dated ``d`` carries the
        # demand of ``[d, d + horizon - 1]`` as its target and only lagged (>= 1 day)
        # features, so the inference origin for a decision taken on ``current_date``
        # is the row dated ``current_date`` itself -- exactly what ``run_scoring``
        # does with the newest day of history. Slicing at ``< current_date`` instead
        # produced a forecast for ``[t-1, t+horizon-2]`` and costed it against the
        # actuals of ``[t, t+horizon-1]``: a one-day-stale forecast scored against a
        # window shifted forward, which under a rising trend biases every cadence low
        # and deflates conformal coverage. Including the row of ``current_date`` leaks
        # nothing: its own demand only ever feeds later rows, which do not exist yet,
        # and every training target it could support needs demand up to
        # ``d + horizon - 1`` and is dropped as missing.
        available_history = combined_panel[combined_panel["date"] <= current_date]
        actuals = _reveal_actuals(eval_panel, current_date, horizon)
        actuals_complete = not actuals.empty and bool(
            (actuals["actuals_days_observed"] >= horizon).all()
        )
        inference_frame, feature_columns = _build_origin_frame(available_history, settings)

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
                inference_frame=inference_frame,
                feature_columns=feature_columns,
                model_path=model_path,
                settings=settings,
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

            preds = attach_inventory_costs(preds, settings.inventory)
            preds["decision_date"] = current_date
            preds["day_index"] = day_index
            preds["cadence"] = label
            preds["retrained_this_step"] = retrained_this_step
            preds["actuals_complete"] = actuals_complete
            rows.append(preds)

        if (day_index + 1) % 5 == 0 or day_index == len(eval_dates) - 1:
            logger.info(
                "  día %d/%d (%s) — reentrenos hasta ahora: %d",
                day_index + 1,
                len(eval_dates),
                current_date.date(),
                len(retrain_events),
            )

    return pd.concat(rows, ignore_index=True), retrain_events


def _persist_simulation_outputs(
    sim_root: Path,
    predictions_by_day: pd.DataFrame,
    cadence_summary: pd.DataFrame,
    cadence_comparison: pd.DataFrame,
    retrain_events: list[dict[str, Any]],
    settings: Settings,
    eval_dates: list[Any],
) -> tuple[Path | None, Path]:
    """Write simulation artifacts to disk; return ``(plot_path, report_path)``."""
    horizon = settings.dataset.horizon
    predictions_by_day.to_parquet(sim_root / "predictions_by_day.parquet", index=False)
    cadence_summary.to_csv(sim_root / "cadence_summary.csv", index=False)
    cadence_comparison.to_csv(sim_root / "cadence_comparison.csv", index=False)
    (sim_root / "retrain_events.json").write_text(
        json.dumps(retrain_events, indent=2), encoding="utf-8"
    )

    plot_path: Path | None = None
    if settings.simulation.make_plots:
        plot_path = _plot_cumulative_cost(predictions_by_day, retrain_events, sim_root, horizon)

    report_path = _write_simulation_report(
        sim_root, cadence_summary, cadence_comparison, retrain_events, settings, eval_dates
    )
    return plot_path, report_path


def run_operational_simulation(settings: Settings) -> OperationalSimulationArtifacts:
    """Replay the eval split origin by origin as if it were daily production data.

    Trains an initial champion on the train split, then iterates over eval
    dates. For each day every configured cadence scores its model, the realized
    single-period cost is computed against the revealed window, and retraining is
    triggered according to the cadence period. See the module docstring for what
    this backtest does and does not model.
    """
    rule(logger, "backtest de origen rodante")
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

    with open_run_directory(settings.reporting.run_name, EXPERIMENT_OPS) as run_dir:
        sim_root = run_dir / "simulation"
        sim_root.mkdir(parents=True, exist_ok=True)
        sim_models_root = sim_root / "models"
        sim_models_root.mkdir(parents=True, exist_ok=True)

        cadence_paths, cadence_int = _setup_cadence_models(settings, train_panel, sim_models_root)

        # Combined panel keeps lag continuity across the train→eval boundary.
        combined_panel = pd.concat([train_panel, eval_panel], ignore_index=True)
        combined_panel = combined_panel.sort_values(["series_id", "date"]).reset_index(drop=True)

        predictions_by_day, retrain_events = _run_streaming_loop(
            eval_dates=eval_dates,
            combined_panel=combined_panel,
            eval_panel=eval_panel,
            cadence_paths=cadence_paths,
            cadence_int=cadence_int,
            horizon=horizon,
            settings=settings,
        )
        cadence_summary = _summarize_cadences(predictions_by_day, retrain_events, horizon)
        cadence_comparison = _compare_cadences(
            predictions_by_day, horizon, random_seed=settings.project.random_seed
        )

        plot_path, report_path = _persist_simulation_outputs(
            sim_root,
            predictions_by_day,
            cadence_summary,
            cadence_comparison,
            retrain_events,
            settings,
            eval_dates,
        )

        fields(
            logger,
            {
                "escrito": str(sim_root),
                "reentrenos": str(len(retrain_events)),
            },
        )
        if not cadence_comparison.empty and bool(cadence_comparison["underpowered"].any()):
            logger.warning(
                "solo %d orígenes independientes: la comparación de cadencias es "
                "descriptiva, no concluyente",
                int(cadence_comparison["n_origins"].max()),
            )

        log_ops_metadata(
            settings=settings,
            cadence_summary=cadence_summary,
            cadence_comparison=cadence_comparison,
            n_origins=len(eval_dates),
            n_retrain_events=len(retrain_events),
        )

        return OperationalSimulationArtifacts(
            predictions_by_day=predictions_by_day,
            cadence_summary=cadence_summary,
            cadence_comparison=cadence_comparison,
            retrain_events=retrain_events,
            run_directory=sim_root,
            cumulative_cost_plot=plot_path,
            report_path=report_path,
            cadence_models=cadence_paths,
        )


def _summarize_cadences(
    predictions_by_day: pd.DataFrame,
    retrain_events: list[dict[str, Any]],
    horizon: int,
) -> pd.DataFrame:
    """Aggregate per-cadence performance over the non-overlapping origin grid."""
    complete = _independent_origins(predictions_by_day, horizon)
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
                "n_origins": int(group["decision_date"].nunique()) if not group.empty else 0,
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


def _compare_cadences(
    predictions_by_day: pd.DataFrame,
    horizon: int,
    random_seed: int,
) -> pd.DataFrame:
    """Compare every cadence against the no-retrain baseline, with uncertainty.

    A raw cost difference over a handful of weeks says nothing on its own, so each
    cadence is paired with the baseline on the same origins and the same series --
    the two policies see identical demand, which removes most of the variance --
    and the uncertainty comes from a bootstrap that resamples whole origins.
    Origins are the resampling unit because series within one week share the same
    demand shock and are not independent.

    ``n_origins`` below ``MIN_ORIGINS_FOR_INFERENCE`` is reported as
    ``underpowered``: the interval is then too wide to rank policies, and the
    honest reading is "this window cannot tell them apart".
    """
    complete = _independent_origins(predictions_by_day, horizon)
    if complete.empty or BASELINE_CADENCE_LABEL not in set(complete["cadence"]):
        return pd.DataFrame()

    paired = complete.pivot_table(
        index=["decision_date", "series_id"],
        columns="cadence",
        values="total_cost",
        aggfunc="sum",
    ).dropna()
    if paired.empty or BASELINE_CADENCE_LABEL not in paired.columns:
        return pd.DataFrame()

    # One row per origin: the decision unit whose costs are exchangeable.
    by_origin = paired.groupby(level="decision_date").sum()
    baseline = by_origin[BASELINE_CADENCE_LABEL].to_numpy(dtype=float)
    n_origins = len(by_origin)
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(0, n_origins, size=(N_BOOTSTRAP_RESAMPLES, n_origins))

    rows = []
    for cadence in by_origin.columns:
        if cadence == BASELINE_CADENCE_LABEL:
            continue
        candidate = by_origin[cadence].to_numpy(dtype=float)
        delta = candidate - baseline
        total_baseline = float(baseline.sum())
        cost_change_pct = (
            100.0 * float(delta.sum()) / total_baseline if total_baseline else float("nan")
        )
        resampled_baseline = baseline[draws].sum(axis=1)
        resampled_delta = delta[draws].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_pct = 100.0 * resampled_delta / resampled_baseline
        low, high = (float(x) for x in np.percentile(boot_pct[np.isfinite(boot_pct)], [2.5, 97.5]))
        rows.append(
            {
                "cadence": cadence,
                "baseline": BASELINE_CADENCE_LABEL,
                "n_origins": n_origins,
                "n_series": int(paired.index.get_level_values("series_id").nunique()),
                "cost_change_pct": cost_change_pct,
                "ci95_low_pct": low,
                "ci95_high_pct": high,
                "origins_cheaper_than_baseline": int((delta < 0).sum()),
                "conclusive": bool(
                    n_origins >= MIN_ORIGINS_FOR_INFERENCE and (low > 0.0 or high < 0.0)
                ),
                "underpowered": bool(n_origins < MIN_ORIGINS_FOR_INFERENCE),
            }
        )
    return pd.DataFrame(rows)


def _plot_cumulative_cost(
    predictions_by_day: pd.DataFrame,
    retrain_events: list[dict[str, Any]],
    sim_root: Path,
    horizon: int,
) -> Path | None:
    try:
        import matplotlib
        import matplotlib.dates as mdates

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    complete = _independent_origins(predictions_by_day, horizon)
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

    ax.set_title(
        f"Cumulative inventory cost by retrain cadence\n"
        f"(non-overlapping origins, one every {horizon} days)"
    )
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
    cadence_comparison: pd.DataFrame,
    retrain_events: list[dict[str, Any]],
    settings: Settings,
    eval_dates: list[pd.Timestamp],
) -> Path:
    horizon = settings.dataset.horizon
    n_origins = int(cadence_summary["n_origins"].max()) if not cadence_summary.empty else 0
    lines = [
        "# Rolling-origin production backtest (OPS plane)",
        "",
        f"- Streaming window: {len(eval_dates)} days "
        f"({eval_dates[0].date()} → {eval_dates[-1].date()})",
        f"- Horizon: {horizon} days",
        f"- Cadences evaluated: {list(cadence_summary['cadence'])}",
        f"- Total retrain events: {len(retrain_events)}",
        f"- Scored origins used for aggregation: {n_origins} "
        f"(one every {horizon} days, fully-revealed actuals only)",
        "",
        "## Cadence summary",
        "",
        _format_markdown_table(cadence_summary),
        "",
        "## Cadence vs no-retrain baseline",
        "",
        _format_markdown_table(cadence_comparison),
        "",
        _comparison_verdict(cadence_comparison),
        "",
        "## Scope and limitations",
        "",
        "- Not an inventory-state simulation: every origin is an independent",
        "  single-period Newsvendor decision. No stock is carried between periods,",
        "  there is no order pipeline and no lead time, so these costs compare",
        "  policies against each other rather than reproducing what a real",
        "  replenishment policy would accumulate.",
        f"- Aggregation uses one origin every {horizon} days. Scoring runs daily, so",
        "  consecutive origins overlap and summing all of them would count each day",
        f"  of demand up to {horizon} times.",
        "- Truth is `observed_demand`, censored on stockout days. Forecast and truth",
        "  are censored alike, which is self-consistent, but the shortage cost is a",
        "  lower bound on the real one. This plane never applies `LatentDemandImputer`.",
        "- Origins whose actuals have not fully landed are scored but excluded from",
        "  every aggregate: their `y_true` is a partial-window sum, which would make",
        "  shortage cost look artificially low.",
    ]
    report_path = sim_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _comparison_verdict(cadence_comparison: pd.DataFrame) -> str:
    """One-line honest reading of the paired comparison."""
    if cadence_comparison.empty:
        return "_(no baseline pairing available for this window)_"
    parts = []
    for row in cadence_comparison.itertuples(index=False):
        band = f"{row.cost_change_pct:+.1f}% [{row.ci95_low_pct:+.1f}%, {row.ci95_high_pct:+.1f}%]"
        if row.underpowered:
            reading = (
                f"only {row.n_origins} independent origins — underpowered, "
                "treat as descriptive, not as evidence one policy wins"
            )
        elif row.conclusive:
            reading = "interval excludes zero — the difference holds on this window"
        else:
            reading = "interval spans zero — indistinguishable from the baseline"
        parts.append(f"- `{row.cadence}` vs `{row.baseline}`: {band} — {reading}")
    return "\n".join(
        [
            "Cost change is paired per origin and series against the baseline; the",
            "interval is a 95% bootstrap CI resampling whole origins (series inside an",
            "origin share the same demand shock and are not independent).",
            "",
            *parts,
        ]
    )


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
