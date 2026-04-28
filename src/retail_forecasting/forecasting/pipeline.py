from __future__ import annotations

import pandas as pd
import numpy as np

from retail_forecasting.config import Settings
from retail_forecasting.data.censorship import LatentDemandImputer
from retail_forecasting.data.fresh_retailnet import load_prepared_panel
from retail_forecasting.drift.regime_analysis import label_stockout_regime
from retail_forecasting.drift.detectors import PageHinkleyDetector
from retail_forecasting.evaluation.metrics import summarize_costs, summarize_predictions
from retail_forecasting.evaluation.reporting import RunArtifacts, write_run_artifacts
from retail_forecasting.forecasting.backtesting import build_walk_forward_folds
from retail_forecasting.features.engineering import (
    build_supervised_frame,
)
from retail_forecasting.inventory.newsvendor import (
    attach_inventory_costs,
    choose_order_quantity,
    run_sensitivity_analysis,
)
from retail_forecasting.models.boosting import AutoBoostingModel
from retail_forecasting.models.optimization import HyperparameterTuner
from retail_forecasting.models.catboosting import CatBoostingModel
from retail_forecasting.models.conformal import ConformalForecaster
from retail_forecasting.models.linear import RidgeBaselineModel
from retail_forecasting.models.naive import SeasonalNaiveModel
from retail_forecasting.models.statistical import AutoArimaModel
from retail_forecasting.utils.io import quantile_column_name


def run_experiment(settings: Settings) -> RunArtifacts:
    """Run the end-to-end experiment comparing Observed vs Latent demand."""
    if settings.dataset.source != "fresh_retailnet":
        raise ValueError(
            f"Unsupported dataset source '{settings.dataset.source}'. "
            "The current v1 implementation supports only 'fresh_retailnet'."
        )

    # 1. Load Original Panel
    raw_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )

    # 2. Run Strategy A: Observed Demand (Baseline)
    print("--- Running Strategy A: Observed Demand ---")
    artifacts_obs = run_experiment_from_frame(
        raw_panel, settings, data_strategy="Observed"
    )

    # 3. Run Strategy B: Latent Demand (Imputed)
    strategy_name = settings.preprocessing.imputation_strategy
    print(f"--- Running Strategy B: Latent Demand (Strategy: {strategy_name}) ---")
    imputer = LatentDemandImputer(strategy=strategy_name)
    imputed_panel = imputer.impute(raw_panel)
    artifacts_latent = run_experiment_from_frame(
        imputed_panel, settings, data_strategy=f"Latent_{strategy_name}"
    )

    # 4. Merge Artifacts
    merged_predictions = pd.concat(
        [artifacts_obs.predictions, artifacts_latent.predictions], ignore_index=True
    )
    merged_metrics, merged_folds = summarize_predictions(merged_predictions)
    merged_costs = summarize_costs(merged_predictions)
    merged_sens = run_sensitivity_analysis(merged_predictions, settings.inventory)

    final_artifacts = RunArtifacts(
        prepared_panel=raw_panel,
        supervised_frame=artifacts_obs.supervised_frame,
        predictions=merged_predictions,
        metrics_summary=merged_metrics,
        fold_metrics=merged_folds,
        cost_summary=merged_costs,
        sensitivity_summary=merged_sens,
        drifts=artifacts_obs.drifts,
    )

    return write_run_artifacts(final_artifacts, settings)


def run_experiment_from_frame(
    panel: pd.DataFrame, settings: Settings, data_strategy: str = "Observed"
) -> RunArtifacts:
    """Run the full backtesting pipeline from an in-memory panel."""
    prepared_panel = label_stockout_regime(panel)
    supervised_frame, feature_columns = build_supervised_frame(
        panel=prepared_panel,
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )

    # Build walk-forward folds
    folds = build_walk_forward_folds(
        panel=prepared_panel,
        validation_config=settings.validation,
        horizon=settings.dataset.horizon,
    )

    # State for cross-fold model reuse
    ridge_model: ConformalForecaster | None = None
    boosting_model: ConformalForecaster | None = None
    cat_model: ConformalForecaster | None = None
    arima_model: ConformalForecaster | None = None

    # Optional: Hyperparameter Tuning Phase
    best_boosting_params = {
        "n_estimators": settings.models.n_estimators,
        "learning_rate": settings.models.learning_rate,
        "max_depth": settings.models.max_depth
    }
    if settings.models.use_tuning:
        print(f"🔍 Starting Optuna Tuning for {data_strategy} strategy...")
        tuning_train_frame = supervised_frame[supervised_frame["date"] <= folds[0].train_end_date]
        tuner = HyperparameterTuner(settings, n_trials=settings.models.tuning_trials)
        best_boosting_params.update(tuner.tune_boosting(tuning_train_frame, feature_columns))

    # Drift detection state
    drift_detector = PageHinkleyDetector(threshold=15.0, min_instances=2)
    force_retrain = False
    detected_drifts = []

    fold_predictions = []

    baseline_model = SeasonalNaiveModel(
        seasonal_period=settings.models.seasonal_period, horizon=settings.dataset.horizon
    ).fit(prepared_panel)

    for fold in folds:
        # Prepare training and validation frames
        train_mask = supervised_frame["date"] <= fold.train_end_date
        validation_mask = (
            (supervised_frame["date"] >= fold.validation_start_date)
            & (supervised_frame["date"] <= fold.validation_end_date)
        )
        train_frame = supervised_frame.loc[train_mask].copy()
        validation_frame = supervised_frame.loc[validation_mask].copy()
        if train_frame.empty or validation_frame.empty:
            continue

        # Calibration split for conformal methods
        calib_days = 21
        max_train_date = train_frame["date"].max()
        calib_cutoff = max_train_date - pd.Timedelta(days=calib_days)
        sub_train_frame = train_frame[train_frame["date"] <= calib_cutoff].copy()
        calib_frame = train_frame[train_frame["date"] > calib_cutoff].copy()
        if sub_train_frame.empty:
            sub_train_frame = train_frame
            calib_frame = pd.DataFrame()

        # Reset force_retrain if it was active
        current_fold_retrained = force_retrain
        force_retrain = False

        # 1. Seasonal Naive Baseline
        fold_predictions.append(
            _build_baseline_predictions(
                validation_frame=validation_frame,
                baseline_model=baseline_model,
                fold_id=fold.fold_id,
                settings=settings,
                data_strategy=data_strategy,
            )
        )

        # 2. Ridge Regression (Linear Baseline)
        if ridge_model is None or settings.validation.retrain_each_fold or current_fold_retrained:
            base_ridge = RidgeBaselineModel(random_seed=settings.project.random_seed)
            base_ridge.fit(
                sub_train_frame.loc[:, feature_columns],
                sub_train_frame["target_lead_time_demand"],
            )
            ridge_model = ConformalForecaster(base_ridge)
            if not calib_frame.empty:
                ridge_model.calibrate(
                    calib_frame.loc[:, feature_columns],
                    calib_frame["target_lead_time_demand"],
                    alpha=settings.models.quantiles[0] * 2,
                )
        fold_predictions.append(
            _build_model_predictions(
                validation_frame=validation_frame,
                feature_columns=feature_columns,
                model=ridge_model,
                fold_id=fold.fold_id,
                settings=settings,
                data_strategy=data_strategy,
            )
        )

        # 3. LightGBM (Boosting)
        if boosting_model is None or settings.validation.retrain_each_fold or current_fold_retrained:
            # Prepare cost parameters if optimization is enabled
            cost_params = {}
            if settings.models.optimize_for_cost:
                cost_params = {
                    "overstock_cost": settings.inventory.overstock_cost,
                    "stockout_cost": settings.inventory.stockout_cost
                }
                
            base_lgb = AutoBoostingModel(
                quantiles=settings.models.quantiles,
                random_seed=settings.project.random_seed,
                **best_boosting_params,
                **cost_params
            )
            base_lgb.fit(
                sub_train_frame.loc[:, feature_columns],
                sub_train_frame["target_lead_time_demand"],
            )
            boosting_model = ConformalForecaster(base_lgb)
            if not calib_frame.empty:
                boosting_model.calibrate(
                    calib_frame.loc[:, feature_columns],
                    calib_frame["target_lead_time_demand"],
                    alpha=settings.models.quantiles[0] * 2,
                )
        boosting_preds = _build_model_predictions(
                validation_frame=validation_frame,
                feature_columns=feature_columns,
                model=boosting_model,
                fold_id=fold.fold_id,
                settings=settings,
                data_strategy=data_strategy,
            )
        fold_predictions.append(boosting_preds)

        # 4. CatBoost (Boosting)
        if cat_model is None or settings.validation.retrain_each_fold or current_fold_retrained:
            base_cat = CatBoostingModel(
                quantiles=settings.models.quantiles,
                random_seed=settings.project.random_seed,
                **best_boosting_params
            )
            base_cat.fit(
                sub_train_frame.loc[:, feature_columns],
                sub_train_frame["target_lead_time_demand"],
            )
            cat_model = ConformalForecaster(base_cat)
            if not calib_frame.empty:
                cat_model.calibrate(
                    calib_frame.loc[:, feature_columns],
                    calib_frame["target_lead_time_demand"],
                    alpha=settings.models.quantiles[0] * 2,
                )
        fold_predictions.append(
            _build_model_predictions(
                validation_frame=validation_frame,
                feature_columns=feature_columns,
                model=cat_model,
                fold_id=fold.fold_id,
                settings=settings,
                data_strategy=data_strategy,
            )
        )

        # 5. ARIMA (Statistical)
        if arima_model is None or settings.validation.retrain_each_fold or current_fold_retrained:
            base_arima = AutoArimaModel(
                seasonal_period=settings.models.seasonal_period,
                horizon=settings.dataset.horizon
            ).fit(prepared_panel)
            arima_model = ConformalForecaster(base_arima)
            if not calib_frame.empty:
                arima_model.calibrate(
                    calib_frame,
                    calib_frame["target_lead_time_demand"],
                    alpha=settings.models.quantiles[0] * 2,
                )
        fold_predictions.append(
            _build_model_predictions(
                validation_frame=validation_frame,
                feature_columns=feature_columns,
                model=arima_model,
                fold_id=fold.fold_id,
                settings=settings,
                data_strategy=data_strategy,
            )
        )

        # Update drift detector with current fold MAE
        boosting_fold_preds = fold_predictions[-3] 
        fold_mae = (boosting_fold_preds["y_true"] - boosting_fold_preds["y_pred"]).abs().mean()
        drift_status = drift_detector.update(fold_mae)
        
        if drift_status.is_drift:
            detected_drifts.append({
                "date": str(fold.validation_start_date.date()),
                "score": drift_status.score,
                "threshold": drift_status.threshold,
                "fold_id": fold.fold_id
            })
            if settings.validation.drift_triggered_retrain:
                print(f"⚠️ DRIFT DETECTED in Fold {fold.fold_id}. Forcing retrain for next fold.")
                force_retrain = True


    if not fold_predictions:
        raise ValueError("Backtest did not produce any validation predictions.")

    predictions = pd.concat(fold_predictions, ignore_index=True)
    metrics_summary, fold_metrics = summarize_predictions(predictions)
    cost_summary = summarize_costs(predictions)
    sensitivity_summary = run_sensitivity_analysis(
        predictions=predictions,
        base_inventory_config=settings.inventory,
    )
    
    report_extra = ""
    if detected_drifts:
        drift_str = ", ".join([f"Fold {d['fold_id']} (score={d['score']:.2f})" for d in detected_drifts])
        report_extra = f"**ALERT**: Concept drift detected and triggered adaptive retrains at: {drift_str}"

    artifacts = RunArtifacts(
        prepared_panel=prepared_panel,
        supervised_frame=supervised_frame,
        predictions=predictions,
        metrics_summary=metrics_summary,
        fold_metrics=fold_metrics,
        cost_summary=cost_summary,
        sensitivity_summary=sensitivity_summary,
        drifts=detected_drifts,
        report_extra=report_extra
    )
    return write_run_artifacts(artifacts, settings)


def _build_baseline_predictions(
    validation_frame: pd.DataFrame,
    baseline_model: SeasonalNaiveModel,
    fold_id: int,
    settings: Settings,
    data_strategy: str = "Observed",
) -> pd.DataFrame:
    """Build baseline forecasts for one fold."""
    cols_to_keep = ["date", "series_id", "target_lead_time_demand", "stockout_hours", "stockout_regime"]
    if "latent_demand_est" in validation_frame.columns:
        cols_to_keep.extend(["latent_demand_est", "is_imputed", "original_observed_demand"])
        
    prediction_frame = validation_frame.loc[:, cols_to_keep].copy()
    prediction_frame["y_true"] = prediction_frame["target_lead_time_demand"]
    prediction_frame["y_pred"] = baseline_model.predict(validation_frame)
    prediction_frame["model_name"] = baseline_model.model_name
    prediction_frame["backend_name"] = "heuristic"
    prediction_frame["fold_id"] = fold_id
    prediction_frame["data_strategy"] = data_strategy
    prediction_frame["order_quantity"] = choose_order_quantity(
        predictions=prediction_frame,
        inventory_config=settings.inventory,
        quantile_columns=[],
        quantile_levels=[],
    )
    return attach_inventory_costs(prediction_frame, settings.inventory)


def _build_model_predictions(
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    model: ConformalForecaster,
    fold_id: int,
    settings: Settings,
    data_strategy: str = "Observed",
) -> pd.DataFrame:
    """Build model forecasts and attach costs."""
    cols_to_keep = ["date", "series_id", "target_lead_time_demand", "stockout_hours", "stockout_regime"]
    if "latent_demand_est" in validation_frame.columns:
        cols_to_keep.extend(["latent_demand_est", "is_imputed", "original_observed_demand"])

    prediction_frame = validation_frame.loc[:, cols_to_keep].copy()
    prediction_frame["y_true"] = prediction_frame["target_lead_time_demand"]
    
    if hasattr(model.base_model, "model_name") and model.base_model.model_name == "auto_arima":
        prediction_frame["y_pred"] = model.predict(validation_frame)
    else:
        prediction_frame["y_pred"] = model.predict(validation_frame.loc[:, feature_columns])
        
    prediction_frame["model_name"] = model.model_name
    prediction_frame["backend_name"] = model.backend_name
    prediction_frame["fold_id"] = fold_id
    prediction_frame["data_strategy"] = data_strategy

    if hasattr(model.base_model, "model_name") and model.base_model.model_name == "auto_arima":
        quantile_predictions = model.predict_quantiles(validation_frame)
    else:
        quantile_predictions = model.predict_quantiles(validation_frame.loc[:, feature_columns])
    
    quantile_columns = []
    for quantile in settings.models.quantiles:
        column = quantile_column_name(quantile)
        if column in quantile_predictions:
            prediction_frame[column] = quantile_predictions[column]
            quantile_columns.append(column)

    prediction_frame["order_quantity"] = choose_order_quantity(
        predictions=prediction_frame,
        inventory_config=settings.inventory,
        quantile_columns=quantile_columns,
        quantile_levels=[float(c.replace("q_", "").replace("_", ".")) for c in quantile_columns],
    )
    return attach_inventory_costs(prediction_frame, settings.inventory)
