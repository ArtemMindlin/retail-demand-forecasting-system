from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_forecasting.config import Settings
from retail_forecasting.contracts.contracts_backtesting import FoldRunMetadata
from retail_forecasting.contracts.contracts_drift import DriftDetectorMetadata, DriftEvent
from retail_forecasting.contracts.contracts_tuning import BoostingParams
from retail_forecasting.data.censorship import LatentDemandImputer
from retail_forecasting.data.dataset import load_prepared_panel
from retail_forecasting.data.quality import (
    raise_on_blocking_data_quality,
    validate_prepared_panel,
)
from retail_forecasting.drift import label_all_regimes
from retail_forecasting.drift.detectors import PageHinkleyDetector
from retail_forecasting.evaluation.metrics import summarize_costs, summarize_predictions
from retail_forecasting.evaluation.reporting import (
    BacktestMetadata,
    DatasetMetadata,
    FeaturePipelineMetadata,
    ModelRunMetadata,
    RunArtifacts,
    ValidationMetadata,
    build_config_hash,
    get_git_commit,
    utc_timestamp,
    write_run_artifacts,
)
from retail_forecasting.evaluation.xai import calculate_shap_values
from retail_forecasting.features.engineering import (
    build_inference_frame_with_fallback,
    build_supervised_frame,
)
from retail_forecasting.forecasting.backtesting import build_walk_forward_folds
from retail_forecasting.inventory.cost_profiles import build_series_cost_profile
from retail_forecasting.inventory.newsvendor import (
    attach_inventory_costs,
    choose_order_quantity,
    run_sensitivity_analysis,
    summarize_pareto_frontier,
)
from retail_forecasting.inventory.simulation import simulate_inventory_policy
from retail_forecasting.models.boosting import AutoBoostingModel
from retail_forecasting.models.catboosting import CatBoostingModel
from retail_forecasting.models.conformal import ConformalForecaster
from retail_forecasting.models.naive import SeasonalNaiveModel
from retail_forecasting.models.optimization import HyperparameterTuner
from retail_forecasting.utils.io import quantile_column_name


def run_experiment(settings: Settings) -> RunArtifacts:
    """Run the end-to-end experiment comparing Observed vs Latent demand."""
    # 1. Load Original Panel
    raw_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    quality_report = validate_prepared_panel(raw_panel, settings)
    raise_on_blocking_data_quality(quality_report)

    # Load external holdout (eval) split
    print("📥 Loading external holdout (eval) split...")
    holdout_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="eval",
    )

    # 2. Run Strategy A: Observed Demand (Baseline)
    print("--- Running Strategy A: Observed Demand ---")
    artifacts_obs = run_experiment_from_frame(
        raw_panel, settings, data_strategy="Observed", holdout_panel=holdout_panel
    )

    # 3. Run Strategy B: Latent Demand (Imputed)
    strategy_name = settings.preprocessing.imputation_strategy
    print(f"--- Running Strategy B: Latent Demand (Strategy: {strategy_name}) ---")
    imputer = LatentDemandImputer(strategy=strategy_name)
    imputed_panel = imputer.impute(raw_panel)

    imputed_holdout = None
    if holdout_panel is not None:
        imputed_holdout = imputer.impute(holdout_panel)

    artifacts_latent = run_experiment_from_frame(
        imputed_panel,
        settings,
        data_strategy=f"Latent_{strategy_name}",
        holdout_panel=imputed_holdout,
    )

    # 4. Merge Artifacts
    merged_predictions = pd.concat(
        [artifacts_obs.predictions, artifacts_latent.predictions], ignore_index=True
    )

    # Extract cost profile from one of the strategies (they share the same series)
    # This is safe as the synthetic cost profile is built from the same initial panel
    sample_series_cost_profile = None
    if settings.inventory.use_series_costs:
        sample_series_cost_profile = build_series_cost_profile(raw_panel, settings.inventory)

    # Run dynamic inventory simulation on merged results
    merged_predictions = simulate_inventory_policy(
        merged_predictions,
        inventory_config=settings.inventory,
        series_cost_profile=sample_series_cost_profile,
    )

    merged_metrics, merged_folds = summarize_predictions(merged_predictions)
    merged_costs = summarize_costs(merged_predictions)
    merged_sens = run_sensitivity_analysis(merged_predictions, settings.inventory)
    merged_pareto = summarize_pareto_frontier(merged_predictions, settings.inventory)

    combined_metadata = None
    if artifacts_obs.backtest_metadata is not None:
        combined_metadata = artifacts_obs.backtest_metadata.model_copy(
            update={
                "data_strategy": f"Observed+Latent_{strategy_name}",
                "created_at": utc_timestamp(),
                "models": ModelRunMetadata(
                    models_run=sorted(merged_predictions["model_name"].dropna().unique().tolist()),
                    quantiles=settings.models.quantiles,
                    optimize_for_cost=settings.models.optimize_for_cost,
                    use_tuning=settings.models.use_tuning,
                    retrain_each_fold=settings.validation.retrain_each_fold,
                ),
            }
        )

    final_artifacts = RunArtifacts(
        prepared_panel=raw_panel,
        supervised_frame=artifacts_obs.supervised_frame,
        predictions=merged_predictions,
        metrics_summary=merged_metrics,
        fold_metrics=merged_folds,
        cost_summary=merged_costs,
        sensitivity_summary=merged_sens,
        pareto_frontier=merged_pareto,
        data_quality_report=quality_report,
        drifts=artifacts_obs.drifts,
        backtest_metadata=combined_metadata,
    )

    artifacts_with_files = write_run_artifacts(final_artifacts, settings)

    try:
        from retail_forecasting.evaluation.mlflow_logger import log_experiment_to_mlflow

        log_experiment_to_mlflow(artifacts_with_files, settings)
    except ImportError as e:
        print(f"MLflow logging skipped: {e}")

    return artifacts_with_files


def run_experiment_from_frame(
    panel: pd.DataFrame,
    settings: Settings,
    data_strategy: str = "Observed",
    holdout_panel: pd.DataFrame | None = None,
) -> RunArtifacts:
    """Run the full backtesting pipeline from an in-memory panel."""
    quality_report = validate_prepared_panel(panel, settings)
    raise_on_blocking_data_quality(quality_report)

    prepared_panel = label_all_regimes(panel)
    series_cost_profile = None
    if settings.inventory.use_series_costs:
        series_cost_profile = build_series_cost_profile(prepared_panel, settings.inventory)
    supervised_frame, feature_metadata = build_supervised_frame(
        panel=prepared_panel,
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )
    feature_columns = feature_metadata.feature_columns

    # Build holdout supervised frame separately. Concatenating panel+holdout gives the
    # holdout rows correct lag history, but we only keep holdout-date rows in the result.
    # This prevents holdout demand from entering training targets via shift(-horizon).
    holdout_supervised_frame: pd.DataFrame | None = None
    if holdout_panel is not None:
        combined_panel = pd.concat([panel, holdout_panel], ignore_index=True)
        combined_prepared = label_all_regimes(combined_panel)
        full_supervised, _ = build_supervised_frame(
            panel=combined_prepared,
            feature_config=settings.features,
            horizon=settings.dataset.horizon,
        )
        holdout_dates = set(holdout_panel["date"].unique())
        holdout_supervised_frame = full_supervised[
            full_supervised["date"].isin(holdout_dates)
        ].copy()

    # Build walk-forward folds (only on the original panel dates)
    folds = build_walk_forward_folds(
        panel=panel,
        validation_config=settings.validation,
        horizon=settings.dataset.horizon,
    )

    # State for cross-fold model reuse
    boosting_model: ConformalForecaster | None = None
    cat_model: ConformalForecaster | None = None

    # Optional: Hyperparameter Tuning Phase
    best_boosting_params = BoostingParams(
        n_estimators=settings.models.n_estimators,
        learning_rate=settings.models.learning_rate,
        max_depth=settings.models.max_depth,
    )
    tuning_metadata = None
    if settings.models.use_tuning:
        print(f"🔍 Starting Optuna Tuning for {data_strategy} strategy...")
        # Tuning only uses data available in the first fold's training set
        tuning_train_frame = supervised_frame[supervised_frame["date"] <= folds[0].train_end_date]
        tuner = HyperparameterTuner(settings, n_trials=settings.models.tuning_trials)
        tuning_result = tuner.tune_boosting(tuning_train_frame, feature_columns)
        best_boosting_params = tuning_result.best_params
        tuning_metadata = tuning_result.metadata

    # Drift detection state
    drift_detector = PageHinkleyDetector(
        threshold=settings.drift.threshold,
        delta=settings.drift.delta,
        min_instances=settings.drift.min_instances,
    )
    force_retrain = False
    detected_drifts: list[DriftEvent] = []
    max_drift_score = 0.0
    last_drift_score = 0.0

    fold_predictions = []
    fold_run_metadata: list[FoldRunMetadata] = []

    baseline_model = SeasonalNaiveModel(
        seasonal_period=settings.models.seasonal_period,
        horizon=settings.dataset.horizon,
    ).fit(panel)

    for fold in folds:
        # Prepare training and validation frames
        train_mask = supervised_frame["date"] <= fold.train_end_date
        validation_mask = (supervised_frame["date"] >= fold.validation_start_date) & (
            supervised_frame["date"] <= fold.validation_end_date
        )
        train_frame = supervised_frame.loc[train_mask].copy()
        validation_frame = supervised_frame.loc[validation_mask].copy()
        if train_frame.empty or validation_frame.empty:
            continue
        fold_run_metadata.append(
            FoldRunMetadata(
                fold_id=fold.fold_id,
                horizon=fold.horizon,
                train_end_date=str(fold.train_end_date.date()),
                validation_start_date=str(fold.validation_start_date.date()),
                validation_end_date=str(fold.validation_end_date.date()),
                train_rows=len(train_frame),
                validation_rows=len(validation_frame),
                train_series=train_frame["series_id"].nunique(),
                validation_series=validation_frame["series_id"].nunique(),
            )
        )

        # Calibration split for conformal methods
        max_train_date = train_frame["date"].max()
        calib_cutoff = max_train_date - pd.Timedelta(days=settings.validation.calibration_days)
        sub_train_frame = train_frame[train_frame["date"] <= calib_cutoff].copy()
        calib_frame = train_frame[train_frame["date"] > calib_cutoff].copy()
        if sub_train_frame.empty:
            sub_train_frame = train_frame
            calib_frame = pd.DataFrame()

        # Mondrian grouping variable for calibration
        calib_group_ids = None
        if not calib_frame.empty and "third_category_id" in calib_frame.columns:
            calib_group_ids = calib_frame["third_category_id"]

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
                series_cost_profile=series_cost_profile,
            )
        )

        # 2. LightGBM (Boosting)
        if (
            boosting_model is None
            or settings.validation.retrain_each_fold
            or current_fold_retrained
        ):
            base_lgb = AutoBoostingModel(
                quantiles=settings.models.quantiles,
                random_seed=settings.project.random_seed,
                n_estimators=best_boosting_params.n_estimators,
                learning_rate=best_boosting_params.learning_rate,
                max_depth=best_boosting_params.max_depth,
                overstock_cost=(
                    settings.inventory.overstock_cost if settings.models.optimize_for_cost else 1.0
                ),
                stockout_cost=(
                    settings.inventory.stockout_cost if settings.models.optimize_for_cost else 0.0
                ),
            )
            boosting_model = ConformalForecaster(base_lgb)
            boosting_model.fit(
                sub_train_frame.loc[:, feature_columns],
                sub_train_frame["target_lead_time_demand"],
            )
            if not calib_frame.empty:
                boosting_model.calibrate(
                    calib_frame.loc[:, feature_columns],
                    calib_frame["target_lead_time_demand"],
                    alpha=settings.models.quantiles[0] * 2,
                    group_ids=calib_group_ids,
                )

        boosting_preds = _build_model_predictions(
            validation_frame=validation_frame,
            feature_columns=feature_columns,
            model=boosting_model,
            fold_id=fold.fold_id,
            settings=settings,
            data_strategy=data_strategy,
            series_cost_profile=series_cost_profile,
        )
        fold_predictions.append(boosting_preds)

        # 3. CatBoost (Boosting)
        if cat_model is None or settings.validation.retrain_each_fold or current_fold_retrained:
            base_cat = CatBoostingModel(
                quantiles=settings.models.quantiles,
                random_seed=settings.project.random_seed,
                n_estimators=best_boosting_params.n_estimators,
                learning_rate=best_boosting_params.learning_rate,
                max_depth=best_boosting_params.max_depth,
            )
            cat_model = ConformalForecaster(base_cat)
            cat_model.fit(
                sub_train_frame.loc[:, feature_columns],
                sub_train_frame["target_lead_time_demand"],
            )
            if not calib_frame.empty:
                cat_model.calibrate(
                    calib_frame.loc[:, feature_columns],
                    calib_frame["target_lead_time_demand"],
                    alpha=settings.models.quantiles[0] * 2,
                    group_ids=calib_group_ids,
                )

        fold_predictions.append(
            _build_model_predictions(
                validation_frame=validation_frame,
                feature_columns=feature_columns,
                model=cat_model,
                fold_id=fold.fold_id,
                settings=settings,
                data_strategy=data_strategy,
                series_cost_profile=series_cost_profile,
            )
        )

        # Update drift detector with current fold MAE
        fold_mae = (boosting_preds["y_true"] - boosting_preds["y_pred"]).abs().mean()
        drift_status = drift_detector.update(fold_mae)
        last_drift_score = drift_status.score
        max_drift_score = max(max_drift_score, drift_status.score)

        if drift_status.is_drift:
            detected_drifts.append(
                DriftEvent(
                    date=str(fold.validation_start_date.date()),
                    score=drift_status.score,
                    threshold=drift_status.threshold,
                    fold_id=fold.fold_id,
                )
            )
            if settings.validation.drift_triggered_retrain:
                print(f"⚠️ DRIFT DETECTED in Fold {fold.fold_id}. Forcing retrain for next fold.")
                force_retrain = True

    # 4. Final Holdout Evaluation (Unseen data)
    holdout_boosting_model: ConformalForecaster | None = None
    holdout_cat_model: ConformalForecaster | None = None
    if holdout_supervised_frame is not None and not holdout_supervised_frame.empty:
        print(f"📊 Retraining on full train set before holdout evaluation ({data_strategy})...")
        holdout_frame = holdout_supervised_frame

        # Retrain both models on the entire supervised_frame so the holdout is evaluated
        # by a model that has seen all available training data, not just up to the last fold.
        calib_cutoff = supervised_frame["date"].max() - pd.Timedelta(
            days=settings.validation.calibration_days
        )
        full_sub_train = supervised_frame[supervised_frame["date"] <= calib_cutoff].copy()
        full_calib = supervised_frame[supervised_frame["date"] > calib_cutoff].copy()
        if full_sub_train.empty:
            full_sub_train = supervised_frame
            full_calib = pd.DataFrame()

        full_calib_group_ids = None
        if not full_calib.empty and "third_category_id" in full_calib.columns:
            full_calib_group_ids = full_calib["third_category_id"]

        base_lgb_final = AutoBoostingModel(
            quantiles=settings.models.quantiles,
            random_seed=settings.project.random_seed,
            n_estimators=best_boosting_params.n_estimators,
            learning_rate=best_boosting_params.learning_rate,
            max_depth=best_boosting_params.max_depth,
            overstock_cost=(
                settings.inventory.overstock_cost if settings.models.optimize_for_cost else 1.0
            ),
            stockout_cost=(
                settings.inventory.stockout_cost if settings.models.optimize_for_cost else 0.0
            ),
        )
        holdout_boosting_model = ConformalForecaster(base_lgb_final)
        holdout_boosting_model.fit(
            full_sub_train.loc[:, feature_columns],
            full_sub_train["target_lead_time_demand"],
        )
        if not full_calib.empty:
            holdout_boosting_model.calibrate(
                full_calib.loc[:, feature_columns],
                full_calib["target_lead_time_demand"],
                alpha=settings.models.quantiles[0] * 2,
                group_ids=full_calib_group_ids,
            )

        base_cat_final = CatBoostingModel(
            quantiles=settings.models.quantiles,
            random_seed=settings.project.random_seed,
            n_estimators=best_boosting_params.n_estimators,
            learning_rate=best_boosting_params.learning_rate,
            max_depth=best_boosting_params.max_depth,
        )
        holdout_cat_model = ConformalForecaster(base_cat_final)
        holdout_cat_model.fit(
            full_sub_train.loc[:, feature_columns],
            full_sub_train["target_lead_time_demand"],
        )
        if not full_calib.empty:
            holdout_cat_model.calibrate(
                full_calib.loc[:, feature_columns],
                full_calib["target_lead_time_demand"],
                alpha=settings.models.quantiles[0] * 2,
                group_ids=full_calib_group_ids,
            )

        if not holdout_frame.empty:
            # Baseline on holdout
            fold_predictions.append(
                _build_baseline_predictions(
                    validation_frame=holdout_frame,
                    baseline_model=baseline_model,
                    fold_id=999,  # Conventional ID for holdout
                    settings=settings,
                    data_strategy=data_strategy,
                    series_cost_profile=series_cost_profile,
                )
            )
            fold_predictions.append(
                _build_model_predictions(
                    validation_frame=holdout_frame,
                    feature_columns=feature_columns,
                    model=holdout_boosting_model,
                    fold_id=999,
                    settings=settings,
                    data_strategy=data_strategy,
                    series_cost_profile=series_cost_profile,
                )
            )
            fold_predictions.append(
                _build_model_predictions(
                    validation_frame=holdout_frame,
                    feature_columns=feature_columns,
                    model=holdout_cat_model,
                    fold_id=999,
                    settings=settings,
                    data_strategy=data_strategy,
                    series_cost_profile=series_cost_profile,
                )
            )

    # Persist final models to the stable models directory for operational serving
    _lgb_to_save = holdout_boosting_model if holdout_boosting_model is not None else boosting_model
    _cat_to_save = holdout_cat_model if holdout_cat_model is not None else cat_model
    _models_dir = settings.models.models_dir
    _models_dir.mkdir(parents=True, exist_ok=True)
    for _m in [_lgb_to_save, _cat_to_save]:
        if _m is not None:
            _m.save(_models_dir / f"{_m.backend_name}.pkl")

    if not fold_predictions:
        raise ValueError("Backtest did not produce any validation predictions.")

    predictions = pd.concat(fold_predictions, ignore_index=True)
    # Run dynamic inventory simulation
    predictions = simulate_inventory_policy(
        predictions,
        inventory_config=settings.inventory,
        series_cost_profile=series_cost_profile,
    )

    metrics_summary, fold_metrics = summarize_predictions(predictions)
    cost_summary = summarize_costs(predictions)
    sensitivity_summary = run_sensitivity_analysis(
        predictions=predictions,
        base_inventory_config=settings.inventory,
        series_cost_profile=series_cost_profile,
    )
    pareto_frontier = summarize_pareto_frontier(predictions, settings.inventory)

    report_extra = ""
    if detected_drifts:
        drift_str = ", ".join(
            [f"Fold {event.fold_id} (score={event.score:.2f})" for event in detected_drifts]
        )
        report_extra = (
            f"**ALERT**: Concept drift detected and triggered adaptive retrains at: {drift_str}"
        )

    backtest_metadata = BacktestMetadata(
        run_name=settings.reporting.run_name,
        data_strategy=data_strategy,
        created_at=utc_timestamp(),
        git_commit=get_git_commit(),
        config_hash=build_config_hash(settings),
        dataset=DatasetMetadata(
            rows=len(prepared_panel),
            series=prepared_panel["series_id"].nunique(),
            unique_dates=prepared_panel["date"].nunique(),
            date_min=str(prepared_panel["date"].min().date()),
            date_max=str(prepared_panel["date"].max().date()),
        ),
        features=FeaturePipelineMetadata(
            horizon=settings.dataset.horizon,
            lags=feature_metadata.lags,
            rolling_windows=feature_metadata.rolling_windows,
            feature_columns=len(feature_metadata.feature_columns),
            input_rows=feature_metadata.input_rows,
            supervised_rows=feature_metadata.output_rows,
            dropped_rows_missing_target=feature_metadata.dropped_rows_missing_target,
            dropped_rows_missing_features=feature_metadata.dropped_rows_missing_features,
        ),
        validation=ValidationMetadata(
            initial_train_days=settings.validation.initial_train_days,
            n_folds_requested=settings.validation.n_folds,
            fold_size_days=settings.validation.fold_size_days,
            folds_created=len(fold_run_metadata),
            folds=fold_run_metadata,
        ),
        models=ModelRunMetadata(
            models_run=sorted(predictions["model_name"].dropna().unique().tolist()),
            quantiles=settings.models.quantiles,
            optimize_for_cost=settings.models.optimize_for_cost,
            use_tuning=settings.models.use_tuning,
            retrain_each_fold=settings.validation.retrain_each_fold,
        ),
        tuning=tuning_metadata,
        drift=DriftDetectorMetadata(
            detector_name="PageHinkleyDetector",
            threshold=settings.drift.threshold,
            delta=settings.drift.delta,
            min_instances=settings.drift.min_instances,
            monitored_metric="boosting_fold_mae",
            observations_seen=drift_detector.observations_seen,
            alerts_detected=len(detected_drifts),
            max_score=max_drift_score,
            last_score=last_drift_score,
        ),
    )

    # 6. Optional: Explainability (SHAP)
    shap_values = None
    if settings.reporting.make_plots:
        # We explain the last trained model (trained on the most data)
        # using a sample from the supervised frame
        model_to_explain = cat_model if cat_model is not None else boosting_model
        if model_to_explain is not None:
            print(f"--- Calculating SHAP values for {model_to_explain.model_name} ---")
            shap_values = calculate_shap_values(
                model=model_to_explain,
                X=supervised_frame.loc[:, feature_columns],
            )

    artifacts = RunArtifacts(
        prepared_panel=prepared_panel,
        supervised_frame=supervised_frame,
        predictions=predictions,
        metrics_summary=metrics_summary,
        fold_metrics=fold_metrics,
        cost_summary=cost_summary,
        sensitivity_summary=sensitivity_summary,
        pareto_frontier=pareto_frontier,
        data_quality_report=quality_report,
        drifts=detected_drifts,
        report_extra=report_extra,
        backtest_metadata=backtest_metadata,
        shap_values=shap_values,
    )
    return write_run_artifacts(artifacts, settings)


def _build_baseline_predictions(
    validation_frame: pd.DataFrame,
    baseline_model: SeasonalNaiveModel,
    fold_id: int,
    settings: Settings,
    data_strategy: str = "Observed",
    series_cost_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build baseline forecasts for one fold."""
    cols_to_keep = [
        "date",
        "series_id",
        "target_lead_time_demand",
        "stockout_hours",
        "stockout_regime",
        "velocity_regime",
        "promo_regime",
        "seasonal_regime",
    ]
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
        series_cost_profile=series_cost_profile,
    )
    return attach_inventory_costs(
        prediction_frame,
        settings.inventory,
        series_cost_profile=series_cost_profile,
    )


def _build_model_predictions(
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    model: ConformalForecaster,
    fold_id: int,
    settings: Settings,
    data_strategy: str = "Observed",
    series_cost_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build model forecasts and attach costs."""
    cols_to_keep = [
        "date",
        "series_id",
        "target_lead_time_demand",
        "stockout_hours",
        "stockout_regime",
        "velocity_regime",
        "promo_regime",
        "seasonal_regime",
    ]
    if "latent_demand_est" in validation_frame.columns:
        cols_to_keep.extend(["latent_demand_est", "is_imputed", "original_observed_demand"])

    prediction_frame = validation_frame.loc[:, cols_to_keep].copy()
    prediction_frame["y_true"] = prediction_frame["target_lead_time_demand"]

    prediction_frame["y_pred"] = model.predict(validation_frame.loc[:, feature_columns])

    prediction_frame["model_name"] = model.model_name
    prediction_frame["backend_name"] = model.backend_name
    prediction_frame["fold_id"] = fold_id
    prediction_frame["data_strategy"] = data_strategy

    # Mondrian grouping variable: third_category_id is a strong candidate for retail
    group_ids = None
    if "third_category_id" in validation_frame.columns:
        group_ids = validation_frame["third_category_id"]

    quantile_predictions = model.predict_quantiles(
        validation_frame.loc[:, feature_columns],
        group_ids=group_ids,
    )

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
        series_cost_profile=series_cost_profile,
    )
    return attach_inventory_costs(
        prediction_frame,
        settings.inventory,
        series_cost_profile=series_cost_profile,
    )


def _instantiate_champion_base_model(settings: Settings) -> CatBoostingModel | AutoBoostingModel:
    backend = settings.business.champion_backend_name
    if backend == "conformal_catboost_official":
        return CatBoostingModel(
            quantiles=settings.models.quantiles,
            random_seed=settings.project.random_seed,
            n_estimators=settings.models.n_estimators,
            learning_rate=settings.models.learning_rate,
            max_depth=settings.models.max_depth,
        )
    return AutoBoostingModel(
        quantiles=settings.models.quantiles,
        random_seed=settings.project.random_seed,
        n_estimators=settings.models.n_estimators,
        learning_rate=settings.models.learning_rate,
        max_depth=settings.models.max_depth,
        overstock_cost=(
            settings.inventory.overstock_cost if settings.models.optimize_for_cost else 1.0
        ),
        stockout_cost=(
            settings.inventory.stockout_cost if settings.models.optimize_for_cost else 0.0
        ),
    )


def train_and_save_champion(
    settings: Settings,
    panel: pd.DataFrame,
    models_dir: Path | None = None,
) -> Path:
    """Fit the configured champion model on the full panel and persist it to disk."""

    prepared_panel = label_all_regimes(panel)
    supervised_frame, feature_metadata = build_supervised_frame(
        panel=prepared_panel,
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )
    feature_columns = feature_metadata.feature_columns

    calib_cutoff = supervised_frame["date"].max() - pd.Timedelta(
        days=settings.validation.calibration_days
    )
    train_frame = supervised_frame[supervised_frame["date"] <= calib_cutoff].copy()
    calib_frame = supervised_frame[supervised_frame["date"] > calib_cutoff].copy()
    if train_frame.empty:
        train_frame = supervised_frame
        calib_frame = pd.DataFrame()

    base_model = _instantiate_champion_base_model(settings)
    conformal = ConformalForecaster(base_model)
    conformal.fit(
        train_frame.loc[:, feature_columns],
        train_frame["target_lead_time_demand"],
    )
    if not calib_frame.empty:
        calib_group_ids = None
        if "third_category_id" in calib_frame.columns:
            calib_group_ids = calib_frame["third_category_id"]
        conformal.calibrate(
            calib_frame.loc[:, feature_columns],
            calib_frame["target_lead_time_demand"],
            alpha=settings.models.quantiles[0] * 2,
            group_ids=calib_group_ids,
        )

    resolved_dir = models_dir if models_dir is not None else settings.models.models_dir
    resolved_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolved_dir / f"{conformal.backend_name}.pkl"
    conformal.save(model_path)
    return model_path


def run_retrain(settings: Settings) -> Path:
    """Load data, train champion on all of it, and write the model to disk."""
    splits = []
    for split in settings.dataset.splits:
        splits.append(
            load_prepared_panel(
                dataset_config=settings.dataset,
                preprocessing_config=settings.preprocessing,
                split=split,
            )
        )
    raw_panel = pd.concat(splits, ignore_index=True)
    quality_report = validate_prepared_panel(raw_panel, settings)
    raise_on_blocking_data_quality(quality_report)
    model_path = train_and_save_champion(settings, raw_panel)
    print(f"✅ Champion retrained and saved to {model_path}")
    return model_path


def run_scoring(
    settings: Settings,
    panel: pd.DataFrame | None = None,
    model_path: Path | None = None,
) -> RunArtifacts:
    """Operational scoring using a pre-trained model — no retraining.

    When ``panel`` or ``model_path`` are provided they override the defaults
    (train split + champion model on disk), enabling reuse from the streaming
    simulation without duplicating the inference plumbing.
    """
    if panel is None:
        panel = load_prepared_panel(
            dataset_config=settings.dataset,
            preprocessing_config=settings.preprocessing,
            split="train",
        )
        quality_report = validate_prepared_panel(panel, settings)
        raise_on_blocking_data_quality(quality_report)
    else:
        quality_report = None

    if model_path is None:
        models_dir = settings.models.models_dir
        champion_backend = settings.business.champion_backend_name
        model_path = models_dir / f"{champion_backend}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}. Run a backtest or retrain first.")
    model = ConformalForecaster.load(model_path)
    print(f"✅ Loaded champion model: {model.backend_name} from {model_path}")

    prepared_panel = label_all_regimes(panel)
    series_cost_profile = None
    if settings.inventory.use_series_costs:
        series_cost_profile = build_series_cost_profile(prepared_panel, settings.inventory)

    inference_frame, inference_metadata = build_inference_frame_with_fallback(
        prepared_panel,
        settings.features,
        horizon=settings.dataset.horizon,
    )

    predictions = _build_scoring_predictions(
        inference_frame=inference_frame,
        feature_columns=inference_metadata.feature_columns,
        model=model,
        settings=settings,
        series_cost_profile=series_cost_profile,
    )

    artifacts = RunArtifacts(
        prepared_panel=prepared_panel,
        supervised_frame=pd.DataFrame(),
        predictions=predictions,
        metrics_summary=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        cost_summary=pd.DataFrame(),
        data_quality_report=quality_report,
    )
    return write_run_artifacts(artifacts, settings)


def _build_scoring_predictions(
    inference_frame: pd.DataFrame,
    feature_columns: list[str],
    model: ConformalForecaster,
    settings: Settings,
    series_cost_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate operational predictions from an inference frame without y_true."""
    frame = inference_frame.copy()
    model_mask = frame["prediction_source"] == "model"

    frame["y_pred"] = float("nan")
    frame["y_true"] = float("nan")
    frame["model_name"] = model.model_name
    frame["backend_name"] = model.backend_name
    frame["fold_id"] = 0
    frame["data_strategy"] = "Observed"

    if model_mask.any():
        model_features = frame.loc[model_mask, feature_columns]
        frame.loc[model_mask, "y_pred"] = model.predict(model_features)

        group_ids = None
        if "third_category_id" in frame.columns:
            group_ids = frame.loc[model_mask, "third_category_id"]

        quantile_preds = model.predict_quantiles(model_features, group_ids=group_ids)
        for quantile in settings.models.quantiles:
            col = quantile_column_name(quantile)
            if col not in frame.columns:
                frame[col] = float("nan")
            if col in quantile_preds:
                frame.loc[model_mask, col] = quantile_preds[col]

    cold_mask = ~model_mask
    if cold_mask.any() and "fallback_target_lead_time_demand" in frame.columns:
        frame.loc[cold_mask, "y_pred"] = frame.loc[cold_mask, "fallback_target_lead_time_demand"]

    quantile_columns = [
        quantile_column_name(q)
        for q in settings.models.quantiles
        if quantile_column_name(q) in frame.columns
    ]
    frame["order_quantity"] = choose_order_quantity(
        predictions=frame,
        inventory_config=settings.inventory,
        quantile_columns=quantile_columns,
        quantile_levels=[float(c.replace("q_", "").replace("_", ".")) for c in quantile_columns],
        series_cost_profile=series_cost_profile,
    )
    return attach_inventory_costs(
        frame, settings.inventory, series_cost_profile=series_cost_profile
    )
