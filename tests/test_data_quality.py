from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from retail_forecasting.config import DataQualityConfig, ProjectConfig, Settings
from retail_forecasting.data.censorship import (
    DEFAULT_SUPERVISED_LGBM_PARAMS,
    LatentDemandImputer,
)
from retail_forecasting.data.quality import (
    DataQualityError,
    raise_on_blocking_data_quality,
    validate_prepared_panel,
)
from tests import make_synthetic_panel


def _single_series_panel_with_one_stockout() -> pd.DataFrame:
    """Minimal panel that reaches the supervised teacher: 99 clean days and one censored."""
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100),
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )
    panel.loc[99, "observed_demand"] = 0.0
    panel.loc[99, "stockout_hours"] = 24.0
    return panel


def test_validate_prepared_panel_blocks_duplicate_series_dates() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    duplicate_row = panel.iloc[[0]].copy()
    broken_panel = pd.concat([panel, duplicate_row], ignore_index=True)
    settings = Settings()

    report = validate_prepared_panel(broken_panel, settings)

    assert report.passed is False
    assert report.blocking_error_count > 0
    assert report.blocking_errors[0].code == "duplicate_series_date_rows"


def test_unparseable_dates_are_caught_by_their_own_check() -> None:
    """A string `date` column is only visible to `invalid_date_format`.

    This check looks redundant next to `null_key_columns` and is not: a null date trips
    both, but an object-dtype column holding "not-a-date" trips only this one, because a
    garbage string is not null. Nothing else in the gate inspects the dtype.

    It matters because `run_experiment_from_frame` takes an arbitrary panel. Without the
    check, strings reach `build_walk_forward_folds`, which compares them against a
    `Timestamp` and raises deep in the pipeline instead of at the gate.
    """
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date", "2024-01-03"],
            "series_id": ["a", "b", "c"],
            "observed_demand": [1.0, 2.0, 3.0],
            "stockout_hours": [0.0, 0.0, 0.0],
        }
    )

    report = validate_prepared_panel(panel, Settings())

    assert report.passed is False
    codes = {issue.code for issue in report.blocking_errors}
    assert codes == {"invalid_date_format"}, "null_key_columns does not see a garbage string"


def test_validate_prepared_panel_blocks_invalid_stockout_hours() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    panel.loc[0, "stockout_hours"] = 20.0
    settings = Settings()

    report = validate_prepared_panel(panel, settings)

    assert report.passed is False
    assert any(issue.code == "invalid_stockout_hours" for issue in report.blocking_errors)


def test_validate_prepared_panel_warns_on_high_missingness() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    panel["discount"] = pd.NA
    settings = Settings(data_quality=DataQualityConfig(max_missing_fraction_warning=0.01))

    report = validate_prepared_panel(panel, settings)

    assert report.warning_count > 0
    assert report.warnings[0].code == "high_missingness"


def test_validate_prepared_panel_blocks_stale_operational_data() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    panel["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        panel.groupby("series_id").cumcount(), unit="D"
    )
    settings = Settings(
        project=ProjectConfig(run_mode="score_daily"),
        data_quality=DataQualityConfig(max_data_age_days=1),
    )

    report = validate_prepared_panel(panel, settings)

    assert report.passed is False
    assert any(issue.code == "stale_data" for issue in report.blocking_errors)


def test_latent_demand_imputer_corrects_stockout_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )
    df.loc[99, "observed_demand"] = 0.0
    df.loc[99, "stockout_hours"] = 24.0

    imputed_df = LatentDemandImputer(strategy="supervised").impute(df)

    assert bool(imputed_df.loc[99, "is_imputed"]) is True
    assert imputed_df.loc[99, "observed_demand"] > 0.0
    assert imputed_df.loc[99, "original_observed_demand"] == 0.0


def test_latent_demand_imputer_uses_default_lgbm_params_when_unset() -> None:
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )
    df.loc[99, "observed_demand"] = 0.0
    df.loc[99, "stockout_hours"] = 24.0

    imputer = LatentDemandImputer(strategy="supervised")
    imputer.impute(df)

    assert imputer.model is not None
    assert imputer.model.n_estimators == 200
    assert imputer.model.learning_rate == 0.05
    assert imputer.model.max_depth == 6


def test_latent_demand_imputer_lgbm_params_override_is_used() -> None:
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )
    df.loc[99, "observed_demand"] = 0.0
    df.loc[99, "stockout_hours"] = 24.0

    imputer = LatentDemandImputer(
        strategy="supervised",
        lgbm_params={"n_estimators": 5, "learning_rate": 0.3, "max_depth": 2},
    )
    imputer.impute(df)

    assert imputer.model is not None
    assert imputer.model.n_estimators == 5
    assert imputer.model.learning_rate == 0.3
    assert imputer.model.max_depth == 2


def test_latent_demand_imputer_missing_model_path_falls_back_to_defaults(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )
    df.loc[99, "observed_demand"] = 0.0
    df.loc[99, "stockout_hours"] = 24.0

    imputer = LatentDemandImputer(
        strategy="supervised", model_path=tmp_path / "does_not_exist.json"
    )
    imputer.impute(df)

    assert imputer.model is not None
    assert imputer.model.n_estimators == 200


def test_latent_demand_imputer_loads_tuned_params_from_disk(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )
    df.loc[99, "observed_demand"] = 0.0
    df.loc[99, "stockout_hours"] = 24.0

    # A COMPLETE params file, because that is the only kind that exists: the file is always
    # written from an `ImputationBoostingParams` dump, and the imputer now reads it back
    # through that same contract. A three-key fixture tested a shape no run can produce.
    params_path = tmp_path / "imputation_lgbm_params.json"
    params_path.write_text(
        json.dumps(
            dict(DEFAULT_SUPERVISED_LGBM_PARAMS)
            | {"n_estimators": 7, "learning_rate": 0.2, "max_depth": 3}
        ),
        encoding="utf-8",
    )

    imputer = LatentDemandImputer(strategy="supervised", model_path=params_path)
    imputer.impute(df)

    assert imputer.model is not None
    assert imputer.model.n_estimators == 7
    assert imputer.model.learning_rate == 0.2
    assert imputer.model.max_depth == 3


def test_latent_demand_imputer_rejects_an_invalid_tuned_params_file(tmp_path) -> None:
    """An unusable params file must fail by name, not by LightGBM abort or silent repair.

    The seven integer hyperparameters used to be coerced with `int()` at read time, which
    repaired a float and let everything else through: a `num_leaves` under LightGBM's own
    floor of 1, a negative tree count, a mistyped key silently ignored. Reading through the
    contract the tuning writes with turns each of those into a message naming the field.
    """
    for patch, expected in (
        ({"num_leaves": 1}, "num_leaves"),
        ({"n_estimators": -5}, "n_estimators"),
        ({"n_estimators": 3254.5}, "n_estimators"),
        ({"boosting_typo": 3}, "boosting_typo"),
    ):
        params_path = tmp_path / "imputation_lgbm_params.json"
        params_path.write_text(
            json.dumps(dict(DEFAULT_SUPERVISED_LGBM_PARAMS) | patch), encoding="utf-8"
        )
        with pytest.raises(ValidationError, match=expected):
            LatentDemandImputer(strategy="supervised", model_path=params_path).impute(
                _single_series_panel_with_one_stockout()
            )


def test_raise_on_blocking_data_quality_raises_error() -> None:
    panel = make_synthetic_panel(num_series=2, num_days=80)
    broken_panel = panel.drop(columns=["observed_demand"])
    settings = Settings()

    report = validate_prepared_panel(broken_panel, settings)

    with pytest.raises(DataQualityError, match="Blocking data-quality checks failed"):
        raise_on_blocking_data_quality(report)


def test_supervised_imputer_keeps_observed_sales_and_fills_only_the_unstocked_slice() -> None:
    """A lightly-censored day must not have its real sales discarded.

    The old max(observed, predicted) rule returned the prediction whenever it exceeded the
    observed sale, throwing away the strongest evidence available on days that were mostly
    sellable. The reconciliation must instead add only the missing slice.

    Every clean day carries the same demand, so the teacher predicts that constant for a full
    day and the expected estimates are exact. The two censored days are chosen so the additive
    rule and the max rule disagree on BOTH -- max would return 10.0 for each, which is what
    this test exists to catch.
    """
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=102),
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )
    # 1.6h of a 16h window = 10% censored; 12.8h = 80% censored.
    panel.loc[100, ["observed_demand", "stockout_hours"]] = [9.5, 1.6]
    panel.loc[101, ["observed_demand", "stockout_hours"]] = [3.0, 12.8]

    imputed = LatentDemandImputer(strategy="supervised").impute(panel)

    estimate = imputed.loc[[100, 101], "latent_demand_est"].to_numpy()
    sold = imputed.loc[[100, 101], "original_observed_demand"].to_numpy()
    assert estimate == pytest.approx([9.5 + 0.1 * 10.0, 3.0 + 0.8 * 10.0])
    # Never below what actually sold, on any severity.
    assert (estimate >= sold).all()


def test_supervised_imputer_does_not_feed_stockout_severity_to_the_teacher() -> None:
    """Severity is a reconciliation input, not a feature.

    Training rows are clean days, where the ratio is identically 0, so the feature carried
    exactly 0 importance while implying the teacher knew about severity. It does not.
    """
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "series_id": "test_1",
            "observed_demand": 10.0,
            "stockout_hours": 0.0,
        }
    )
    df.loc[99, "observed_demand"] = 2.0
    df.loc[99, "stockout_hours"] = 8.0

    imputer = LatentDemandImputer(strategy="supervised")
    imputer.impute(df)

    assert imputer.model is not None
    assert "stockout_ratio" not in imputer.model.feature_name_
