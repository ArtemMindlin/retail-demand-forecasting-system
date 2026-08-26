"""Fair-cost backtest: rank demand-signal strategies by inventory cost against a COMMON
ground truth.

Its own module rather than a corner of `pipeline.py` because it shares nothing with the
walk-forward experiment: no folds, no model fit, no conformal calibration, no forecast. What
it does share is the synthetic-censoring holdout of `data/censorship.py`, which the imputation
search builds on too.

It exists because nothing else ranks the reconstruction strategies. Invariant 42 rules out
scoring them by reconstruction MAE, and an experiment run scores ONE arm against its own
target, so two runs are not comparable. Here every strategy pays for the same days.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig, Settings
from retail_forecasting.contracts.contracts_backtesting import FairCostMetadata
from retail_forecasting.contracts.contracts_config import ImputationStrategy
from retail_forecasting.data.censorship import (
    SYNTHETIC_CENSORING_EVAL_FRACTION,
    LatentDemandImputer,
    synthetic_censor_holdout,
)
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.inventory.newsvendor import attach_inventory_costs
from retail_forecasting.tracking import (
    EXPERIMENT_RUNS,
    log_fair_cost_metadata,
    open_run_directory,
)
from retail_forecasting.utils.logging import fields, get_logger, rule, thousands
from retail_forecasting.utils.provenance import get_git_commit, utc_timestamp
from retail_forecasting.utils.stats import mean_ci95

logger = get_logger(__name__)

# Series sampled from the panel, small on purpose: the point is to isolate the imputation
# signal from the rest of the pipeline, and the artifact reports the sample it came from.
N_SERIES = 30

# Censoring draws per run. One was the whole design until it became clear that the ranking it
# produced had no interval around it, and that the ranking does invert with the panel.
N_DRAWS = 20

BASELINE_STRATEGY = "Observed"

# "none" leaves the censored sale untouched, so it IS the Observed (deflated) signal.
STRATEGIES: tuple[tuple[ImputationStrategy, str], ...] = (
    ("none", BASELINE_STRATEGY),
    ("supervised", "Latent_supervised"),
    ("historical_mean", "Latent_historical_mean"),
    ("clipped_scaling", "Latent_clipped_scaling"),
)


def evaluate_fair_inventory_cost(
    panel: pd.DataFrame,
    inventory_config: InventoryConfig,
    seed: int,
    imputer_params_path: Path | None = None,
    censorable_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Score every strategy on ONE censoring draw, against a common ground truth.

    ``censorable_mask`` restricts which rows may be censored and scored WITHOUT shrinking the
    panel, so the supervised imputer's teacher keeps its deployment-sized training set. Passing
    a smaller panel instead is the defect invariant 41 was written about: it shrank the teacher
    to a fraction of deployment size and changed the answer. Measured here on the same 293
    evaluation days, the same params file and the same ground truth, a 684-row teacher scored
    MAE 0.5640 against 0.4106 for the 16405-row one -- a 27% handicap, and one that fell on the
    supervised arm alone, since both heuristics are per-series or per-row.

    Returns one row per strategy: signal_mae, total_cost, fill_rate, mean_order, n_eval, the
    mean per-series safety-stock scale and the teacher's training size.
    """
    censored, eval_idx, true_demand = synthetic_censor_holdout(
        panel, seed, censorable_mask=censorable_mask
    )
    scored_rows = slice(None) if censorable_mask is None else censorable_mask

    # One cost pair for the whole catalogue, so every strategy is charged identically.
    critical_fractile = inventory_config.stockout_cost / (
        inventory_config.stockout_cost + inventory_config.overstock_cost
    )
    z = statistics.NormalDist().inv_cdf(critical_fractile)
    # PER SERIES, and from the CENSORED panel. Identical for all four strategies -- it is
    # computed before any imputation -- so the comparison stays isolated to the signal, while
    # the cushion stays proportional to the series it orders for. One catalogue-wide scalar
    # made z*sigma 120% of the smallest series' daily sales and 33% of the largest's, and at
    # ~6 units it swamped the one-to-two-unit differences between the signals: every strategy
    # ordered almost the same thing and the comparison measured the cushion. See invariant 44.
    scale_by_series = censored.loc[scored_rows].groupby("series_id")["observed_demand"].std()
    pooled = float(np.std(censored.loc[scored_rows, "observed_demand"].to_numpy(dtype=float)))
    sigma = censored.loc[eval_idx, "series_id"].map(scale_by_series).fillna(pooled).to_numpy(float)
    teacher_fit_rows = int((censored["stockout_hours"] == 0).sum())

    total_demand = float(true_demand.sum())
    records: list[dict[str, Any]] = []
    for strategy, label in STRATEGIES:
        imputed = LatentDemandImputer(strategy=strategy, model_path=imputer_params_path).impute(
            censored
        )
        signal = imputed.loc[eval_idx, "latent_demand_est"].astype(float).to_numpy()
        order_quantity = np.maximum(signal + z * sigma, 0.0)

        costed = attach_inventory_costs(
            pd.DataFrame({"y_true": true_demand, "order_quantity": order_quantity}),
            inventory_config,
        )
        stockout_units = float(costed["stockout_units"].sum())
        records.append(
            {
                "strategy": label,
                "signal_mae": float(np.mean(np.abs(signal - true_demand))),
                "total_cost": float(costed["total_cost"].sum()),
                "fill_rate": (
                    (1.0 - stockout_units / total_demand) * 100.0
                    if total_demand > 0
                    else float("nan")
                ),
                "mean_order": float(np.mean(order_quantity)),
                "n_eval": int(len(eval_idx)),
                "mean_order_policy_scale": float(np.mean(sigma)),
                "teacher_fit_rows": teacher_fit_rows,
            }
        )
    return pd.DataFrame(records)


def draw_costs(
    panel: pd.DataFrame,
    inventory_config: InventoryConfig,
    seeds: list[int],
    imputer_params_path: Path | None = None,
    censorable_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Every strategy scored on each draw, one row per (seed, strategy).

    Sequential on purpose: the LGBM teacher is pinned to one thread by the imputer itself, so
    the sweep costs a few minutes and threading it would only contend with whatever else the
    machine is running.
    """
    per_draw = [
        evaluate_fair_inventory_cost(
            panel, inventory_config, seed, imputer_params_path, censorable_mask
        ).assign(seed=seed)
        for seed in seeds
    ]
    return pd.concat(per_draw, ignore_index=True)


def summarize_draws(draws: pd.DataFrame) -> pd.DataFrame:
    """Average each strategy over the draws and price its cost gap against the baseline.

    The gap is PAIRED: both strategies saw the same censoring draw, so the draw's own
    difficulty cancels and what is left is the signal. The baseline's own gap columns are
    empty rather than zero -- a zero-width interval on the reference reads as a finding.
    """
    baseline = draws[draws["strategy"] == BASELINE_STRATEGY].set_index("seed")["total_cost"]
    baseline_mean = float(baseline.mean())

    records: list[dict[str, Any]] = []
    for label, group in draws.groupby("strategy", sort=False):
        costs = group.set_index("seed")["total_cost"]
        mean_cost = float(costs.mean())
        is_baseline = label == BASELINE_STRATEGY
        if is_baseline:
            delta = ci_low = ci_high = delta_pct = float("nan")
        else:
            deltas = (costs - baseline.reindex(costs.index)).to_numpy(dtype=float)
            delta = float(np.mean(deltas))
            ci_low, ci_high = mean_ci95(deltas)
            delta_pct = (mean_cost - baseline_mean) / baseline_mean * 100.0
        records.append(
            {
                "strategy": label,
                "signal_mae": float(group["signal_mae"].mean()),
                "total_cost": mean_cost,
                "fill_rate": float(group["fill_rate"].mean()),
                "mean_order": float(group["mean_order"].mean()),
                "cost_delta": delta,
                "cost_delta_pct": delta_pct,
                "cost_ci95_low": ci_low,
                "cost_ci95_high": ci_high,
                "n_eval": int(group["n_eval"].iloc[0]),
                "n_draws": int(len(group)),
            }
        )
    return pd.DataFrame(records)


def _build_metadata(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    panel: pd.DataFrame,
    evaluated_series: int,
    seeds: list[int],
    inventory_config: InventoryConfig,
    seed: int,
) -> FairCostMetadata:
    """The facts a reader needs to know whether this ranking applies to their question."""
    ranked = summary[summary["strategy"] != BASELINE_STRATEGY].sort_values("total_cost")
    best = ranked.iloc[0]
    return FairCostMetadata(
        baseline_strategy=BASELINE_STRATEGY,
        source_panel_series=int(panel["series_id"].nunique()),
        sampled_series=evaluated_series,
        panel_rows=len(panel),
        teacher_fit_rows=int(draws["teacher_fit_rows"].iloc[0]),
        panel_start=str(pd.Timestamp(panel["date"].min()).date()),
        panel_end=str(pd.Timestamp(panel["date"].max()).date()),
        n_draws=len(seeds),
        n_eval_rows=int(summary["n_eval"].iloc[0]),
        eval_fraction=SYNTHETIC_CENSORING_EVAL_FRACTION,
        seeds=seeds,
        critical_fractile=inventory_config.stockout_cost
        / (inventory_config.stockout_cost + inventory_config.overstock_cost),
        mean_order_policy_scale=float(draws["mean_order_policy_scale"].mean()),
        best_strategy=str(best["strategy"]),
        best_cost_delta_pct=float(best["cost_delta_pct"]),
        best_ci95=[float(best["cost_ci95_low"]), float(best["cost_ci95_high"])],
        best_beats_baseline=bool(best["cost_ci95_high"] < 0.0),
        seed=seed,
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
    )


def run_fair_cost_backtest(settings: Settings) -> Path:
    """Rank the demand-signal strategies by inventory cost. Trains no forecasting model.

    Returns:
        The created run directory path.
    """
    rule(logger, "backtest de coste justo · verdad común, sin forecasting")
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    source_series = int(panel["series_id"].nunique())
    seed = settings.project.random_seed
    # A MASK, not a subset: only the evaluated rows narrow, while the supervised imputer's
    # teacher keeps the whole panel it is given in deployment. See invariant 41.
    unique_ids = panel["series_id"].drop_duplicates().to_numpy()
    keep = unique_ids
    if len(unique_ids) > N_SERIES:
        keep = np.random.default_rng(seed).choice(unique_ids, size=N_SERIES, replace=False)
    censorable_mask = panel["series_id"].isin(keep)
    n_kept = int(panel.loc[censorable_mask, "series_id"].nunique())
    fields(
        logger,
        {
            "panel": f"{thousands(len(panel))} filas · {source_series} series (maestro)",
            "evaluadas": f"{n_kept} series muestreadas",
            "sorteos": f"{N_DRAWS} censuras sintéticas · {len(STRATEGIES)} estrategias",
        },
    )

    seeds = [seed + offset for offset in range(N_DRAWS)]
    draws = draw_costs(
        panel,
        settings.inventory,
        seeds,
        imputer_params_path=(
            settings.models.models_dir / settings.models.imputation_params_filename
        ),
        censorable_mask=censorable_mask,
    )
    summary = summarize_draws(draws)
    metadata = _build_metadata(summary, draws, panel, n_kept, seeds, settings.inventory, seed)
    # Provenance in the artifact itself: the ranking flips with the source panel, so the CSV
    # must say which one it came from rather than leaving it to the reader's memory.
    summary.insert(1, "source_panel_series", source_series)
    summary.insert(2, "sampled_series", n_kept)

    with open_run_directory(settings.reporting.run_name, EXPERIMENT_RUNS) as run_dir:
        out_path = run_dir / "fair_cost_backtest.csv"
        summary.to_csv(out_path, index=False)
        draws.to_csv(run_dir / "fair_cost_draws.csv", index=False)
        (run_dir / "fair_cost_metadata.json").write_text(
            metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        log_fair_cost_metadata(metadata=metadata, summary=summary, settings=settings)

        rule(logger, "coste de inventario contra una verdad COMÚN (menos es mejor)")
        for line in summary.to_string(index=False).splitlines():
            logger.info("  %s", line)
        fields(
            logger,
            {
                "mejor": f"{metadata.best_strategy} · {metadata.best_cost_delta_pct:+.2f}% "
                f"IC95 [{metadata.best_ci95[0]:+.2f}, {metadata.best_ci95[1]:+.2f}]",
                "bate al observado": "sí" if metadata.best_beats_baseline else "no concluyente",
                "escrito": str(out_path),
            },
        )
        return run_dir
