from __future__ import annotations

import pandas as pd

from retail_forecasting.config import Settings
from retail_forecasting.data.fresh_retailnet import load_prepared_panel
from retail_forecasting.drift.regime_analysis import label_stockout_regime
from retail_forecasting.evaluation.metrics import summarize_costs, summarize_predictions
from retail_forecasting.evaluation.reporting import RunArtifacts, write_run_artifacts
from retail_forecasting.features.engineering import build_supervised_frame
from retail_forecasting.forecasting.backtesting import build_walk_forward_folds
from retail_forecasting.inventory.newsvendor import attach_inventory_costs, choose_order_quantity
from retail_forecasting.models.boosting import AutoBoostingModel
from retail_forecasting.models.naive import SeasonalNaiveModel
from retail_forecasting.utils.io import quantile_column_name


def run_experiment(settings: Settings) -> RunArtifacts:
    """Run the end-to-end experiment from configured data sources.

    Args:
        settings: Fully resolved experiment settings.

    Returns:
        The generated run artifacts, including predictions and summaries.

    Notes:
        The current implementation supports only the FreshRetailNet dataset and does not wire the official eval split into the pipeline.
    """
    if settings.dataset.source != "fresh_retailnet":
        raise ValueError(
            f"Unsupported dataset source '{settings.dataset.source}'. "
            "The current v1 implementation supports only 'fresh_retailnet'."
        )

    if settings.dataset.use_eval_as_holdout:
        raise NotImplementedError(
            "The official eval split is not wired into the v1 pipeline because its temporal "
            "semantics must be verified before using it as an external holdout."
        )

    prepared_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    return run_experiment_from_frame(prepared_panel, settings)


def run_experiment_from_frame(panel: pd.DataFrame, settings: Settings) -> RunArtifacts:
    """Run the full backtesting pipeline from an in-memory panel.

    Args:
        panel: Prepared daily panel to backtest.
        settings: Fully resolved experiment settings.

    Returns:
        The generated run artifacts, including metrics, costs, and reports.
    """
    prepared_panel = label_stockout_regime(panel)
    supervised_frame, feature_columns = build_supervised_frame(
        panel=prepared_panel,
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )
    folds = build_walk_forward_folds(
        panel=prepared_panel,
        validation_config=settings.validation,
        horizon=settings.dataset.horizon,
    )

    fold_predictions = []
    baseline_model = SeasonalNaiveModel(
        seasonal_period=settings.models.seasonal_period,
        horizon=settings.dataset.horizon,
    ).fit(prepared_panel)
    boosting_model: AutoBoostingModel | None = None

    for fold in folds:
        train_mask = supervised_frame["date"] <= fold.train_end_date
        validation_mask = (
            (supervised_frame["date"] >= fold.validation_start_date)
            & (supervised_frame["date"] <= fold.validation_end_date)
        )
        train_frame = supervised_frame.loc[train_mask].copy()
        validation_frame = supervised_frame.loc[validation_mask].copy()
        if train_frame.empty or validation_frame.empty:
            continue

        baseline_predictions = _build_baseline_predictions(
            validation_frame=validation_frame,
            baseline_model=baseline_model,
            fold_id=fold.fold_id,
            settings=settings,
        )
        fold_predictions.append(baseline_predictions)

        if boosting_model is None or settings.validation.retrain_each_fold:
            boosting_model = AutoBoostingModel(
                quantiles=settings.models.quantiles,
                random_seed=settings.project.random_seed,
                n_estimators=settings.models.n_estimators,
                learning_rate=settings.models.learning_rate,
                max_depth=settings.models.max_depth,
            )
            boosting_model.fit(
                train_frame.loc[:, feature_columns],
                train_frame["target_lead_time_demand"],
            )
        model_predictions = _build_boosting_predictions(
            validation_frame=validation_frame,
            feature_columns=feature_columns,
            model=boosting_model,
            fold_id=fold.fold_id,
            settings=settings,
        )
        fold_predictions.append(model_predictions)

    if not fold_predictions:
        raise ValueError("Backtest did not produce any validation predictions.")

    predictions = pd.concat(fold_predictions, ignore_index=True)
    metrics_summary, fold_metrics = summarize_predictions(predictions)
    cost_summary = summarize_costs(predictions)

    artifacts = RunArtifacts(
        prepared_panel=prepared_panel,
        supervised_frame=supervised_frame,
        predictions=predictions,
        metrics_summary=metrics_summary,
        fold_metrics=fold_metrics,
        cost_summary=cost_summary,
    )
    return write_run_artifacts(artifacts, settings)


def _build_baseline_predictions(
    validation_frame: pd.DataFrame,
    baseline_model: SeasonalNaiveModel,
    fold_id: int,
    settings: Settings,
) -> pd.DataFrame:
    """Build baseline forecasts and attach inventory costs for one fold.

    Args:
        validation_frame: Validation rows for the current fold.
        baseline_model: Fitted seasonal naive model.
        fold_id: Fold identifier used in reporting.
        settings: Fully resolved experiment settings.

    Returns:
        A prediction frame with baseline forecasts and cost columns.
    """
    prediction_frame = validation_frame.loc[
        :,
        ["date", "series_id", "target_lead_time_demand", "stockout_hours", "stockout_regime"],
    ].copy()
    prediction_frame["y_true"] = prediction_frame["target_lead_time_demand"]
    prediction_frame["y_pred"] = baseline_model.predict(validation_frame)
    prediction_frame["model_name"] = baseline_model.model_name
    prediction_frame["backend_name"] = "heuristic"
    prediction_frame["fold_id"] = fold_id
    prediction_frame["order_quantity"] = choose_order_quantity(
        predictions=prediction_frame,
        inventory_config=settings.inventory,
        quantile_columns=[],
        quantile_levels=[],
    )
    return attach_inventory_costs(prediction_frame, settings.inventory)


def _build_boosting_predictions(
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    model: AutoBoostingModel,
    fold_id: int,
    settings: Settings,
) -> pd.DataFrame:
    """Build boosting forecasts, quantiles, and inventory costs for one fold.

    Args:
        validation_frame: Validation rows for the current fold.
        feature_columns: Feature columns used by the model.
        model: Fitted boosting model.
        fold_id: Fold identifier used in reporting.
        settings: Fully resolved experiment settings.

    Returns:
        A prediction frame with point forecasts, quantiles, and cost columns.
    """
    prediction_frame = validation_frame.loc[
        :,
        ["date", "series_id", "target_lead_time_demand", "stockout_hours", "stockout_regime"],
    ].copy()
    prediction_frame["y_true"] = prediction_frame["target_lead_time_demand"]
    prediction_frame["y_pred"] = model.predict(validation_frame.loc[:, feature_columns])
    prediction_frame["model_name"] = settings.models.point_model
    prediction_frame["backend_name"] = model.backend_name
    prediction_frame["fold_id"] = fold_id

    quantile_predictions = model.predict_quantiles(validation_frame.loc[:, feature_columns])
    quantile_columns = []
    for quantile in settings.models.quantiles:
        column = quantile_column_name(quantile)
        prediction_frame[column] = quantile_predictions[column]
        quantile_columns.append(column)

    prediction_frame["order_quantity"] = choose_order_quantity(
        predictions=prediction_frame,
        inventory_config=settings.inventory,
        quantile_columns=quantile_columns,
        quantile_levels=settings.models.quantiles,
    )
    return attach_inventory_costs(prediction_frame, settings.inventory)
