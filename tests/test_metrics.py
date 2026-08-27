from __future__ import annotations

import pandas as pd
import pytest

from retail_forecasting.evaluation.metrics import _build_metric_record, winkler_score


def test_winkler_score_basic():
    actual = pd.Series([10.0, 10.0, 10.0])
    lower = pd.Series([8.0, 8.0, 8.0])
    upper = pd.Series([12.0, 12.0, 12.0])
    alpha = 0.2  # 80% interval

    # All actuals are inside the interval. Winkler = Width = 4.0
    score = winkler_score(actual, lower, upper, alpha)
    assert score == 4.0


def test_winkler_score_penalties():
    actual = pd.Series([15.0])  # Outside upper (12)
    lower = pd.Series([8.0])
    upper = pd.Series([12.0])
    alpha = 0.2

    # Width = 4
    # Over penalty = (2/0.2) * (15 - 12) = 10 * 3 = 30
    # Total = 34
    score = winkler_score(actual, lower, upper, alpha)
    assert score == 34.0

    actual_under = pd.Series([5.0])  # Outside lower (8)
    # Under penalty = (2/0.2) * (8 - 5) = 10 * 3 = 30
    # Total = 34
    score_under = winkler_score(actual_under, lower, upper, alpha)
    assert score_under == 34.0


def test_build_metric_record_includes_calibration():
    df = pd.DataFrame(
        {
            "y_true": [10.0, 20.0, 30.0],
            "y_pred": [11.0, 19.0, 31.0],
            "q_0_1": [8.0, 18.0, 28.0],
            "q_0_9": [12.0, 22.0, 32.0],
        }
    )

    record = _build_metric_record(df, "test_model", "test_backend")

    assert "interval_coverage" in record
    assert "interval_width" in record
    assert "winkler_score" in record
    assert record["interval_coverage"] == 1.0
    assert record["interval_width"] == 4.0
    assert record["winkler_score"] == 4.0


def test_build_metric_record_with_mismatched_coverage():
    df = pd.DataFrame(
        {
            "y_true": [15.0, 20.0, 30.0],  # 15 is outside [8, 12]
            "y_pred": [11.0, 19.0, 31.0],
            "q_0_1": [8.0, 18.0, 28.0],
            "q_0_9": [12.0, 22.0, 32.0],
        }
    )

    record = _build_metric_record(df, "test_model", "test_backend")

    # Coverage should be 2/3
    assert record["interval_coverage"] == pytest.approx(0.666, rel=1e-2)
    # Winkler for first row: 4 + (2/0.2)*(15-12) = 34
    # Winkler for others: 4
    # Mean: (34 + 4 + 4) / 3 = 42 / 3 = 14
    assert record["winkler_score"] == 14.0


def _prediction_rows(model: str, y_true: list[float], y_pred: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [0] * len(y_true),
            "model_name": [model] * len(y_true),
            "backend_name": [model] * len(y_true),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def test_wape_cannot_reorder_models_within_one_evaluation_set() -> None:
    """States what WAPE is NOT for, because it is easy to claim more than it gives.

    Every model scores the same rows, so `n` and `sum(y)` are constants and `wape` is `mae`
    rescaled by them: the ratio is identical across models and the ranking cannot move. Its
    value is reading an error ACROSS panels of different scale, where a raw MAE has no
    comparable unit, and doing so without MAPE's division by zero.
    """
    y_true = [100.0, 2.0, 40.0]
    good = _build_metric_record(_prediction_rows("a", y_true, [98.0, 2.5, 41.0]), "a", "a")
    bad = _build_metric_record(_prediction_rows("b", y_true, [80.0, 6.0, 30.0]), "b", "b")

    assert good["mae"] < bad["mae"]
    assert good["wape"] < bad["wape"]
    # The same constant for both, which is exactly why the order is preserved.
    assert good["wape"] / good["mae"] == pytest.approx(bad["wape"] / bad["mae"])
    assert good["wape"] == pytest.approx(3.5 / 142 * 100)


def test_the_signed_error_separates_over_from_under_prediction() -> None:
    """MAE and RMSE are blind to the sign, and the whole censoring argument is about a sign."""
    over = _build_metric_record(_prediction_rows("a", [10.0, 10.0], [12.0, 12.0]), "a", "a")
    under = _build_metric_record(_prediction_rows("b", [10.0, 10.0], [8.0, 8.0]), "b", "b")

    assert over["mae"] == under["mae"] == 2.0
    assert over["mean_error"] == pytest.approx(2.0)
    assert under["mean_error"] == pytest.approx(-2.0)
    assert over["bias_pct"] == pytest.approx(20.0)
    assert under["bias_pct"] == pytest.approx(-20.0)


def test_the_percentages_refuse_to_divide_by_no_demand() -> None:
    """Zero demand is 4.5% of this panel's days, which is what breaks MAPE."""
    record = _build_metric_record(_prediction_rows("a", [0.0, 0.0], [1.0, 2.0]), "a", "a")

    assert record["mae"] == pytest.approx(1.5)
    assert pd.isna(record["wape"])
    assert pd.isna(record["bias_pct"])


def test_the_naive_scaled_error_is_one_for_the_naive_itself() -> None:
    """`rel_mae_naive` below 1 beats the seasonal naive, above 1 loses to it.

    Chapter 7 argues "the MAE trap" in prose: the naive wins on point error. This turns that
    into a number, and the number the thesis has to defend is above 1.
    """
    from retail_forecasting.evaluation.metrics import summarize_predictions

    frame = pd.concat(
        [
            _prediction_rows("seasonal_naive", [10.0, 10.0], [12.0, 12.0]),
            _prediction_rows("catboost", [10.0, 10.0], [13.0, 13.0]),
        ],
        ignore_index=True,
    )

    summary, _ = summarize_predictions(frame)
    scaled = summary.set_index("model_name")["rel_mae_naive"]

    assert scaled["seasonal_naive"] == pytest.approx(1.0)
    assert scaled["catboost"] == pytest.approx(1.5)


def test_the_scaled_error_is_not_comparable_without_the_naive() -> None:
    """A scoring run carries the champion alone. Better an empty value than a false one."""
    from retail_forecasting.evaluation.metrics import summarize_predictions

    summary, _ = summarize_predictions(_prediction_rows("catboost", [10.0], [11.0]))

    assert summary["rel_mae_naive"].isna().all()


def test_each_fold_scales_against_its_own_naive() -> None:
    """Folds differ in difficulty, so one global denominator would mix the two effects."""
    from retail_forecasting.evaluation.metrics import summarize_predictions

    easy = _prediction_rows("seasonal_naive", [10.0], [11.0])
    hard = _prediction_rows("seasonal_naive", [10.0], [14.0])
    hard["fold_id"] = 1
    champion_easy = _prediction_rows("catboost", [10.0], [12.0])
    champion_hard = _prediction_rows("catboost", [10.0], [12.0])
    champion_hard["fold_id"] = 1

    _, folds = summarize_predictions(
        pd.concat([easy, hard, champion_easy, champion_hard], ignore_index=True)
    )
    scaled = folds.set_index(["fold_id", "model_name"])["rel_mae_naive"]

    assert scaled[(0, "catboost")] == pytest.approx(2.0)
    assert scaled[(1, "catboost")] == pytest.approx(0.5)


def _cost_rows(model: str, series_costs: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": list(series_costs),
            "model_name": [model] * len(series_costs),
            "backend_name": [model] * len(series_costs),
            "y_true": [10.0] * len(series_costs),
            "order_quantity": [10.0] * len(series_costs),
            "overstock_units": [0.0] * len(series_costs),
            "stockout_units": [0.0] * len(series_costs),
            "overstock_cost": [0.0] * len(series_costs),
            "stockout_cost": [0.0] * len(series_costs),
            "total_cost": list(series_costs.values()),
        }
    )


def _two_model_costs(n_series: int, factor: float, seed: int = 0) -> pd.DataFrame:
    """Two models over the same series, the champion cheaper by `factor` ON AVERAGE.

    The per-series ratio is deliberately noisy. A first version of this helper multiplied every
    series by exactly `factor`, which made the ratio identical in every cluster: the bootstrap
    then correctly returned a zero-width interval and called a 0.05% gap certain. That is the
    right answer to a degenerate question, and no model is uniformly better on every series.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    base = {f"s{i}": float(rng.uniform(5.0, 50.0)) for i in range(n_series)}
    champion = {k: v * factor * float(rng.uniform(0.7, 1.3)) for k, v in base.items()}
    return pd.concat(
        [_cost_rows("seasonal_naive", base), _cost_rows("catboost", champion)],
        ignore_index=True,
    )


def test_a_real_cost_advantage_comes_out_conclusive() -> None:
    from retail_forecasting.evaluation.metrics import summarize_costs

    summary = summarize_costs(_two_model_costs(200, factor=0.70)).set_index("model_name")

    assert summary.loc["catboost", "cost_change_pct"] < -20.0
    assert summary.loc["catboost", "ci95_high_pct"] < 0.0
    assert summary.loc["catboost", "conclusive"]


def test_the_difference_the_champion_kpi_turned_on_is_not_conclusive() -> None:
    """The reason this bootstrap exists.

    The champion KPI hinged on a 0.05% cost difference between CatBoost and the seasonal naive.
    That is plainly within noise, but "plainly" is not a measurement, and the methodology this
    project declares requires a difference to exceed the experiment's own variability.
    """
    from retail_forecasting.evaluation.metrics import summarize_costs

    summary = summarize_costs(_two_model_costs(200, factor=1.0005)).set_index("model_name")

    assert abs(summary.loc["catboost", "cost_change_pct"]) < 2.0
    assert not summary.loc["catboost", "conclusive"]
    assert summary.loc["catboost", "ci95_low_pct"] < 0.0 < summary.loc["catboost", "ci95_high_pct"]


def test_too_few_series_is_never_conclusive() -> None:
    """Same 30% advantage, five clusters. The honest reading is that the run cannot tell."""
    from retail_forecasting.evaluation.metrics import summarize_costs

    summary = summarize_costs(_two_model_costs(5, factor=0.70)).set_index("model_name")

    assert summary.loc["catboost", "cost_change_pct"] < 0.0
    assert not summary.loc["catboost", "conclusive"]


def test_the_naive_never_reports_a_gap_against_itself() -> None:
    from retail_forecasting.evaluation.metrics import summarize_costs

    summary = summarize_costs(_two_model_costs(200, factor=0.70)).set_index("model_name")

    assert summary.loc["seasonal_naive", "cost_change_pct"] == pytest.approx(0.0)
    assert not summary.loc["seasonal_naive", "conclusive"]


def test_without_the_naive_there_is_nothing_to_compare_against() -> None:
    """A scoring run carries the champion alone. Better an empty gap than a false one."""
    from retail_forecasting.evaluation.metrics import summarize_costs

    frame = _two_model_costs(200, factor=0.70)
    summary = summarize_costs(frame[frame["model_name"] != "seasonal_naive"])

    assert summary["cost_change_pct"].isna().all()
    assert not summary["conclusive"].any()


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    from retail_forecasting.evaluation.metrics import summarize_costs

    frame = _two_model_costs(100, factor=0.9)
    first = summarize_costs(frame, random_seed=7)["ci95_low_pct"].tolist()
    again = summarize_costs(frame, random_seed=7)["ci95_low_pct"].tolist()
    other = summarize_costs(frame, random_seed=8)["ci95_low_pct"].tolist()

    assert first == again
    assert first != other
