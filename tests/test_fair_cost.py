"""The fair-cost backtest: one common ground truth, one shared order policy, N draws.

The comparison this module guards is the only one that ranks reconstruction strategies
(invariant 42 rules out the reconstruction-MAE route), and it went into chapter 6 on a single
censoring draw with no interval around it. These tests pin the three properties that make the
ranking mean anything: the policy is shared, the scale is not an oracle, and the gap against
the baseline is paired.
"""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from retail_forecasting.contracts.contracts_backtesting import FairCostMetadata
from retail_forecasting.contracts.contracts_config import InventoryConfig
from retail_forecasting.data.censorship import synthetic_censor_holdout
from retail_forecasting.evaluation.latex_exporter import _cost_gap_column, _fair_cost_table
from retail_forecasting.forecasting.fair_cost import (
    BASELINE_STRATEGY,
    STRATEGIES,
    _build_metadata,
    draw_costs,
    evaluate_fair_inventory_cost,
    summarize_draws,
)
from tests.conftest import make_synthetic_panel

INVENTORY = InventoryConfig(overstock_cost=1.0, stockout_cost=4.0)
SEEDS = [42, 43, 44, 45, 46]


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_synthetic_panel(num_series=6, num_days=70)


@pytest.fixture
def draws(panel: pd.DataFrame) -> pd.DataFrame:
    return draw_costs(panel, INVENTORY, SEEDS)


def test_every_strategy_is_scored_once_per_draw(panel: pd.DataFrame) -> None:
    result = evaluate_fair_inventory_cost(panel, INVENTORY, seed=42)
    assert list(result["strategy"]) == [label for _, label in STRATEGIES]
    assert result["strategy"].iloc[0] == BASELINE_STRATEGY


def test_narrowing_the_evaluation_does_not_shrink_the_teacher(panel: pd.DataFrame) -> None:
    """The defect invariant 41 was written about, reproduced here and now fixed.

    Handing the imputer a smaller panel shrinks its teacher, which changes the answer: on the
    real 500-series panel the same 293 evaluation days scored MAE 0.5640 with a 684-row teacher
    against 0.4106 with the 16405-row one. The handicap fell on the supervised arm alone, since
    both heuristics are per-series or per-row -- so it biased the very ranking this backtest
    exists to produce. The mask narrows what is scored; the panel stays whole.
    """
    half = panel["series_id"].isin(sorted(panel["series_id"].unique())[:3])

    masked = evaluate_fair_inventory_cost(panel, INVENTORY, seed=42, censorable_mask=half)
    subset = evaluate_fair_inventory_cost(panel[half].reset_index(drop=True), INVENTORY, seed=42)

    assert masked["teacher_fit_rows"].iloc[0] > subset["teacher_fit_rows"].iloc[0]
    # Same evaluation width either way: only the teacher differs.
    assert masked["n_eval"].iloc[0] == subset["n_eval"].iloc[0]


def test_the_order_policy_is_shared_by_every_strategy(panel: pd.DataFrame) -> None:
    """The whole design: if the safety stock differed per strategy, the cost gap would mix
    the signal with the policy and stop being attributable to the reconstruction."""
    result = evaluate_fair_inventory_cost(panel, INVENTORY, seed=42)
    assert result["mean_order_policy_scale"].nunique() == 1
    assert result["n_eval"].nunique() == 1
    assert result["teacher_fit_rows"].nunique() == 1


def test_the_cushion_is_per_series_and_never_reads_the_answer_key(panel: pd.DataFrame) -> None:
    """Pins the order policy exactly, on the baseline arm where the signal is computable.

    Two properties at once. The scale is estimated on the CENSORED panel, so it is knowable
    at decision time -- the true demand of a censored day is not. And it is one scale PER
    SERIES: a single catalogue-wide scalar was ~6 units against series selling 5 a day, which
    swamped the one-to-two-unit differences between the signals and left the comparison
    measuring the cushion instead of the reconstruction.
    """
    censored, eval_idx, true_demand = synthetic_censor_holdout(panel, seed=42)
    scale_by_series = censored.groupby("series_id")["observed_demand"].std()
    sigma = censored.loc[eval_idx, "series_id"].map(scale_by_series).to_numpy(float)
    signal = censored.loc[eval_idx, "observed_demand"].to_numpy(float)
    z = statistics.NormalDist().inv_cdf(0.8)

    result = evaluate_fair_inventory_cost(panel, INVENTORY, seed=42)
    baseline = result[result["strategy"] == BASELINE_STRATEGY].iloc[0]

    assert baseline["mean_order"] == pytest.approx(
        float(np.mean(np.maximum(signal + z * sigma, 0.0)))
    )
    # Not one scalar wearing a per-series disguise.
    assert sigma.std() > 0
    # And nothing here came from the answer key.
    assert not np.allclose(sigma, np.std(true_demand))


def test_a_draw_is_reproducible_and_the_seed_actually_moves_it(panel: pd.DataFrame) -> None:
    same = evaluate_fair_inventory_cost(panel, INVENTORY, seed=42)
    again = evaluate_fair_inventory_cost(panel, INVENTORY, seed=42)
    other = evaluate_fair_inventory_cost(panel, INVENTORY, seed=99)

    pd.testing.assert_frame_equal(same, again)
    assert not np.allclose(same["total_cost"], other["total_cost"])


def test_draw_costs_scores_every_strategy_on_every_seed(draws: pd.DataFrame) -> None:
    assert len(draws) == len(SEEDS) * len(STRATEGIES)
    assert sorted(draws["seed"].unique()) == sorted(SEEDS)
    assert draws.groupby("seed")["strategy"].nunique().eq(len(STRATEGIES)).all()
    # Same panel every draw, so the scorable-row count is a constant and the paired
    # differences below are comparable across draws.
    assert draws["n_eval"].nunique() == 1


def test_the_baseline_has_no_gap_against_itself(draws: pd.DataFrame) -> None:
    """Empty, not zero. A zero-width interval on the reference row reads as a finding."""
    summary = summarize_draws(draws)
    baseline = summary[summary["strategy"] == BASELINE_STRATEGY].iloc[0]
    for column in ("cost_delta", "cost_delta_pct", "cost_ci95_low", "cost_ci95_high"):
        assert pd.isna(baseline[column])


def test_the_reported_cost_is_the_mean_over_the_draws(draws: pd.DataFrame) -> None:
    summary = summarize_draws(draws).set_index("strategy")
    for label, group in draws.groupby("strategy"):
        assert summary.loc[label, "total_cost"] == pytest.approx(group["total_cost"].mean())
        assert summary.loc[label, "signal_mae"] == pytest.approx(group["signal_mae"].mean())
    assert (summary["n_draws"] == len(SEEDS)).all()


def test_the_gap_is_paired_so_only_its_interval_differs_from_a_naive_one(
    draws: pd.DataFrame,
) -> None:
    """The mean of the paired differences equals the difference of the means -- pairing buys
    the INTERVAL, not the point estimate. Pinned so nobody 'simplifies' the interval away on
    the grounds that the mean is unchanged.
    """
    summary = summarize_draws(draws).set_index("strategy")
    baseline_cost = summary.loc[BASELINE_STRATEGY, "total_cost"]
    for label in summary.index.drop(BASELINE_STRATEGY):
        row = summary.loc[label]
        assert row["cost_delta"] == pytest.approx(row["total_cost"] - baseline_cost)
        assert row["cost_ci95_low"] <= row["cost_delta"] <= row["cost_ci95_high"]


def test_the_strategy_order_survives_the_summary(draws: pd.DataFrame) -> None:
    summary = summarize_draws(draws)
    assert list(summary["strategy"]) == [label for _, label in STRATEGIES]


def test_a_single_draw_cannot_produce_an_interval(panel: pd.DataFrame) -> None:
    """Better a refusal than a ranking with no uncertainty, which is what this replaced."""
    one = draw_costs(panel, INVENTORY, [42])
    with pytest.raises(ValueError, match="at least 2 draws"):
        summarize_draws(one)


def test_metadata_names_the_cheapest_reconstruction(panel: pd.DataFrame) -> None:
    draws = draw_costs(panel, INVENTORY, SEEDS)
    summary = summarize_draws(draws)
    metadata = _build_metadata(summary, draws, panel, 30, SEEDS, INVENTORY, seed=42)

    candidates = summary[summary["strategy"] != BASELINE_STRATEGY]
    assert metadata.best_strategy == candidates.loc[candidates["total_cost"].idxmin(), "strategy"]
    assert metadata.critical_fractile == pytest.approx(0.8)
    assert metadata.source_panel_series == panel["series_id"].nunique()
    assert metadata.sampled_series == 30
    assert metadata.teacher_fit_rows == draws["teacher_fit_rows"].iloc[0]
    assert metadata.n_draws == len(SEEDS)


def test_beating_the_baseline_is_decided_on_the_interval_not_the_sign(
    panel: pd.DataFrame, draws: pd.DataFrame
) -> None:
    """A negative mean with an interval straddling zero is a coin flip, not a win."""
    summary = summarize_draws(draws)
    metadata = _build_metadata(summary, draws, panel, 30, SEEDS, INVENTORY, seed=42)
    best = summary.set_index("strategy").loc[metadata.best_strategy]
    assert metadata.best_beats_baseline == bool(best["cost_ci95_high"] < 0.0)


def test_metadata_refuses_a_run_that_cannot_carry_an_interval() -> None:
    fields = {
        "baseline_strategy": "Observed",
        "source_panel_series": 500,
        "sampled_series": 30,
        "panel_rows": 45000,
        "teacher_fit_rows": 16405,
        "panel_start": "2024-03-01",
        "panel_end": "2024-06-25",
        "n_draws": 20,
        "n_eval_rows": 400,
        "eval_fraction": 0.3,
        "seeds": [1, 2],
        "critical_fractile": 0.8,
        "mean_order_policy_scale": 4.2,
        "best_strategy": "Latent_supervised",
        "best_cost_delta_pct": -3.1,
        "best_ci95": [-5.0, -1.0],
        "best_beats_baseline": True,
        "seed": 42,
        "created_at": "2026-08-26T00:00:00Z",
        "git_commit": None,
    }
    assert FairCostMetadata(**fields).n_draws == 20
    with pytest.raises(ValidationError):
        FairCostMetadata(**{**fields, "n_draws": 1})
    with pytest.raises(ValidationError):
        FairCostMetadata(**{**fields, "best_ci95": [-1.0]})


def _summary_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["Observed", "Latent_supervised"],
            "source_panel_series": [500, 500],
            "sampled_series": [30, 30],
            "signal_mae": [4.0, 2.5],
            "total_cost": [1234.5, 1200.25],
            "fill_rate": [90.0, 93.5],
            "mean_order": [12.0, 13.5],
            "cost_delta": [float("nan"), -34.25],
            "cost_delta_pct": [float("nan"), -2.77],
            "cost_ci95_low": [float("nan"), -50.5],
            "cost_ci95_high": [float("nan"), -18.0],
            "n_eval": [400, 400],
            "n_draws": [20, 20],
        }
    )


def test_the_latex_gap_column_leaves_the_baseline_blank() -> None:
    rendered = list(_cost_gap_column(_summary_fixture()))
    assert rendered[0] == "--"
    assert rendered[1] == "-34,25 [-50,50; -18,00]"


def test_the_latex_table_carries_the_gap_and_says_it_is_not_a_percentage() -> None:
    latex = _fair_cost_table(_summary_fixture())
    assert latex.count("&") // latex.count(r"\\") >= 5
    assert "IC 95" in latex
    assert "20 sorteos" in latex
    assert "no en puntos porcentuales" in latex
    assert "-34,25 [-50,50; -18,00]" in latex


def test_the_latex_table_still_renders_a_pre_interval_artifact() -> None:
    """A run predating the draws has no gap columns, and the exporter must not crash on the
    CSVs already in the store."""
    legacy = _summary_fixture().drop(
        columns=["cost_delta", "cost_delta_pct", "cost_ci95_low", "cost_ci95_high", "n_draws"]
    )
    latex = _fair_cost_table(legacy)
    assert "--" in latex
    assert "sorteos" not in latex
