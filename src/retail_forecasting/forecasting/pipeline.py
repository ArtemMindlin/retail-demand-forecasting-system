from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.config import Settings, build_config_hash
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
    champion_registry_path,
    load_champion_registry,
    resolve_champion_reference,
    write_run_artifacts,
)
from retail_forecasting.evaluation.xai import calculate_shap_values
from retail_forecasting.features.engineering import (
    build_inference_frame_with_fallback,
    build_supervised_frame,
)
from retail_forecasting.forecasting.backtesting import build_walk_forward_folds
from retail_forecasting.forecasting.tuned_params import (
    CORE_PARAMS,
    BoostingBackend,
    resolve_backend_params,
)
from retail_forecasting.inventory.newsvendor import (
    attach_inventory_costs,
    choose_order_quantity,
    run_sensitivity_analysis,
)
from retail_forecasting.models.boosting import LightGBMModel
from retail_forecasting.models.catboosting import CatBoostingModel
from retail_forecasting.models.conformal import ConformalForecaster
from retail_forecasting.models.naive import SeasonalNaiveModel
from retail_forecasting.models.optimization import HyperparameterTuner
from retail_forecasting.tracking import EXPERIMENT_RUNS, log_retrain_metadata, open_run_directory
from retail_forecasting.utils.io import (
    HOLDOUT_FOLD_ID,
    model_file_path,
    quantile_column_name,
    quantile_level_from_column,
)
from retail_forecasting.utils.logging import Table, fields, get_logger, rule, thousands
from retail_forecasting.utils.provenance import get_git_commit, utc_timestamp

logger = get_logger(__name__)


def _split_train_calibration(
    frame: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """Split a supervised frame into (sub_train, calibration, mondrian_group_ids).

    Falls back to training on the whole frame (empty calibration) when the split
    would leave no training rows.
    """
    horizon = settings.dataset.horizon
    calib_cutoff = frame["date"].max() - pd.Timedelta(days=settings.validation.calibration_days)
    train_cutoff = calib_cutoff - pd.Timedelta(days=horizon)
    sub_train = frame[frame["date"] <= train_cutoff].copy()
    calib = frame[frame["date"] > calib_cutoff].copy()
    if sub_train.empty:
        return frame, pd.DataFrame(), None

    group_ids = None
    if not calib.empty and "third_category_id" in calib.columns:
        group_ids = calib["third_category_id"]
    return sub_train, calib, group_ids


def _instantiate_boosting_base(
    model_cls: type[LightGBMModel] | type[CatBoostingModel],
    settings: Settings,
    params_by_backend: dict[str, dict[str, Any]],
) -> LightGBMModel | CatBoostingModel:
    """Build a boosting base model from its own resolved params and the inventory costs.

    Keyed per backend because a tuning run searches ONE of them: LightGBM's space and
    CatBoost's do not even share parameter names, and the previous single `BoostingParams`
    silently applied a LightGBM-tuned set to CatBoost as well.
    """
    params = params_by_backend[model_cls.model_name]
    return model_cls(
        quantiles=settings.models.quantiles,
        random_seed=settings.project.random_seed,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        overstock_cost=settings.inventory.overstock_cost,
        stockout_cost=settings.inventory.stockout_cost,
        extra_params={k: v for k, v in params.items() if k not in CORE_PARAMS},
    )


def _train_conformal_model(
    base_model: LightGBMModel | CatBoostingModel,
    sub_train: pd.DataFrame,
    calib: pd.DataFrame,
    group_ids: pd.Series | None,
    feature_columns: list[str],
    settings: Settings,
) -> ConformalForecaster:
    """Fit a ConformalForecaster around a base model and calibrate it if possible."""
    model = ConformalForecaster(base_model)
    model.fit(
        sub_train.loc[:, feature_columns],
        sub_train["target_lead_time_demand"],
    )
    if not calib.empty:
        model.calibrate(
            calib.loc[:, feature_columns],
            calib["target_lead_time_demand"],
            alpha=settings.models.quantiles[0] * 2,
            group_ids=group_ids,
        )
    return model


def run_experiment(settings: Settings) -> RunArtifacts:
    """Run the walk-forward experiment for ONE demand strategy."""
    rule(logger, "experimento walk-forward")
    started = time.monotonic()

    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    quality_report = validate_prepared_panel(panel, settings)
    raise_on_blocking_data_quality(quality_report)

    holdout_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="eval",
    )

    strategy = settings.preprocessing.imputation_strategy
    data_strategy = "Observed" if strategy == "none" else f"Latent_{strategy}"
    fields(
        logger,
        {
            "panel": f"{thousands(quality_report.checked_series)} series, "
            f"{thousands(len(panel))} filas",
            "ventana": f"{panel['date'].min().date()} → {panel['date'].max().date()}",
            "holdout": f"{thousands(len(holdout_panel))} filas "
            f"({holdout_panel['date'].min().date()} → {holdout_panel['date'].max().date()})",
            "calidad": f"{quality_report.warning_count} avisos",
            "estrategia": data_strategy,
            "folds": f"{settings.validation.n_folds} × {settings.validation.fold_size_days}d, "
            f"horizonte {settings.dataset.horizon}d",
        },
    )

    if strategy != "none":
        imputer = LatentDemandImputer(
            strategy=strategy,
            model_path=settings.models.models_dir / settings.models.imputation_params_filename,
        )
        panel = imputer.impute(panel)
        holdout_panel = imputer.impute(holdout_panel)

    artifacts = run_experiment_from_frame(
        panel=panel,
        settings=settings,
        data_strategy=data_strategy,
        holdout_panel=holdout_panel,
    )
    fields(
        logger,
        {
            "escrito": artifacts.run_directory,
            "tiempo": f"{time.monotonic() - started:.0f}s",
        },
    )
    return artifacts


@dataclass
class _FoldLoopResult:
    """Outputs of the walk-forward fold loop needed by later phases."""

    fold_predictions: list[pd.DataFrame] = field(default_factory=list)
    fold_run_metadata: list[FoldRunMetadata] = field(default_factory=list)
    boosting_model: ConformalForecaster | None = None
    cat_model: ConformalForecaster | None = None
    detected_drifts: list[DriftEvent] = field(default_factory=list)
    drift_observations: int = 0
    max_drift_score: float = 0.0
    last_drift_score: float = 0.0


def _build_supervised_frames(
    prepared_panel: pd.DataFrame,
    holdout_panel: pd.DataFrame | None,
    settings: Settings,
) -> tuple[pd.DataFrame, Any, pd.DataFrame | None]:
    """Build the supervised modeling frame and (optionally) the holdout frame.

    The holdout frame is built from the panel+holdout concatenation so its rows
    get correct lag history, but only holdout-date rows are kept — preventing
    holdout demand from leaking into training targets via shift(-horizon).
    """
    supervised_frame, feature_metadata = build_supervised_frame(
        panel=prepared_panel,
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )

    holdout_supervised_frame: pd.DataFrame | None = None
    if holdout_panel is not None:
        train_series_means = prepared_panel.groupby("series_id", sort=False)[
            "observed_demand"
        ].mean()
        prepared_holdout = label_all_regimes(
            holdout_panel, velocity_series_means=train_series_means
        )
        combined_prepared = pd.concat(
            [prepared_panel, prepared_holdout], ignore_index=True
        ).sort_values(["series_id", "date"], ignore_index=True)
        full_supervised, _ = build_supervised_frame(
            panel=combined_prepared,
            feature_config=settings.features,
            horizon=settings.dataset.horizon,
        )
        holdout_dates = set(holdout_panel["date"].unique())
        holdout_supervised_frame = full_supervised[
            full_supervised["date"].isin(holdout_dates)
        ].copy()

    return supervised_frame, feature_metadata, holdout_supervised_frame


def _run_tuning_phase(
    supervised_frame: pd.DataFrame,
    feature_columns: list[str],
    folds: list[Any],
    settings: Settings,
    data_strategy: str,
) -> tuple[BoostingParams, Any, pd.DataFrame | None]:
    """Run optional Optuna tuning; return (best_params, metadata, pareto_frame)."""
    best_params = BoostingParams(
        n_estimators=settings.models.n_estimators,
        learning_rate=settings.models.learning_rate,
        max_depth=settings.models.max_depth,
    )
    if not settings.models.use_tuning:
        return best_params, None, None

    # Tuning only uses data available in the first fold's training set.
    tuning_train_frame = supervised_frame[supervised_frame["date"] <= folds[0].train_end_date]
    fields(
        logger,
        {
            "tuning": f"Optuna, {settings.models.tuning_trials} pruebas",
            "sobre": f"{thousands(len(tuning_train_frame))} filas hasta {folds[0].train_end_date.date()}",
        },
    )
    tuner = HyperparameterTuner(settings, n_trials=settings.models.tuning_trials)
    tuning_result = tuner.tune_boosting(tuning_train_frame, feature_columns)

    tuning_pareto = None
    if tuning_result.pareto_front:
        tuning_pareto = pd.DataFrame([trial.model_dump() for trial in tuning_result.pareto_front])
        tuning_pareto.insert(0, "data_strategy", data_strategy)
    return tuning_result.best_params, tuning_result.metadata, tuning_pareto


def _run_fold_loop(
    folds: list[Any],
    supervised_frame: pd.DataFrame,
    feature_columns: list[str],
    baseline_model: SeasonalNaiveModel,
    best_boosting_params: dict[str, dict[str, Any]],
    settings: Settings,
    data_strategy: str,
) -> _FoldLoopResult:
    """Run the walk-forward loop: baseline + LightGBM + CatBoost per fold, with
    cross-fold model reuse and Page-Hinkley drift detection."""
    result = _FoldLoopResult()
    drift_detector = PageHinkleyDetector(
        threshold=settings.drift.threshold,
        delta=settings.drift.delta,
        min_instances=settings.drift.min_instances,
    )
    force_retrain = False

    # One row per fold: the line this loop repeats. Labels hoisted into the heading,
    # since eight labelled fields per fold wrap a terminal pane.
    progress = Table(
        logger,
        {"fold": 5, "train hasta": 12, "validación": 23, "filas": 8, "entrena": 9, "MAE": 8},
    )
    for fold in folds:
        train_mask = supervised_frame["date"] <= fold.train_end_date
        validation_mask = (supervised_frame["date"] >= fold.validation_start_date) & (
            supervised_frame["date"] <= fold.validation_end_date
        )
        train_frame = supervised_frame.loc[train_mask].copy()
        validation_frame = supervised_frame.loc[validation_mask].copy()
        if train_frame.empty or validation_frame.empty:
            continue
        result.fold_run_metadata.append(
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

        # Calibration split for conformal methods (Mondrian grouping included)
        sub_train_frame, calib_frame, calib_group_ids = _split_train_calibration(
            train_frame, settings
        )

        current_fold_retrained = force_retrain
        force_retrain = False
        trained: list[str] = []

        # 1. Seasonal Naive Baseline
        result.fold_predictions.append(
            _build_baseline_predictions(
                validation_frame=validation_frame,
                baseline_model=baseline_model,
                fold_id=fold.fold_id,
                settings=settings,
                data_strategy=data_strategy,
            )
        )

        # 2. LightGBM (Boosting)
        if (
            result.boosting_model is None
            or settings.validation.retrain_each_fold
            or current_fold_retrained
        ):
            trained.append("lgbm")
            result.boosting_model = _train_conformal_model(
                _instantiate_boosting_base(LightGBMModel, settings, best_boosting_params),
                sub_train_frame,
                calib_frame,
                calib_group_ids,
                feature_columns,
                settings,
            )

        boosting_preds = _build_model_predictions(
            validation_frame=validation_frame,
            feature_columns=feature_columns,
            model=result.boosting_model,
            fold_id=fold.fold_id,
            settings=settings,
            data_strategy=data_strategy,
        )
        result.fold_predictions.append(boosting_preds)

        # 3. CatBoost (Boosting)
        if (
            result.cat_model is None
            or settings.validation.retrain_each_fold
            or current_fold_retrained
        ):
            trained.append("cat")
            result.cat_model = _train_conformal_model(
                _instantiate_boosting_base(CatBoostingModel, settings, best_boosting_params),
                sub_train_frame,
                calib_frame,
                calib_group_ids,
                feature_columns,
                settings,
            )

        result.fold_predictions.append(
            _build_model_predictions(
                validation_frame=validation_frame,
                feature_columns=feature_columns,
                model=result.cat_model,
                fold_id=fold.fold_id,
                settings=settings,
                data_strategy=data_strategy,
            )
        )

        # Update drift detector with current fold MAE
        fold_mae = (boosting_preds["y_true"] - boosting_preds["y_pred"]).abs().mean()
        drift_status = drift_detector.update(fold_mae)
        result.last_drift_score = drift_status.score
        result.max_drift_score = max(result.max_drift_score, drift_status.score)

        progress.row(
            {
                "fold": f"{fold.fold_id}/{len(folds)}",
                "train hasta": fold.train_end_date.date(),
                "validación": f"{fold.validation_start_date.date()} → "
                f"{fold.validation_end_date.date()}",
                "filas": thousands(len(validation_frame)),
                "entrena": "+".join(trained) if trained else "reusa",
                "MAE": f"{fold_mae:.3f}",
            }
        )

        if drift_status.is_drift:
            result.detected_drifts.append(
                DriftEvent(
                    date=str(fold.validation_start_date.date()),
                    score=drift_status.score,
                    threshold=drift_status.threshold,
                    fold_id=fold.fold_id,
                )
            )
            logger.warning(
                "drift en el fold %s (score %.2f > umbral %.2f)%s",
                fold.fold_id,
                drift_status.score,
                drift_status.threshold,
                ", se fuerza reentreno" if settings.validation.drift_triggered_retrain else "",
            )
            if settings.validation.drift_triggered_retrain:
                force_retrain = True

    result.drift_observations = drift_detector.observations_seen
    return result


def _evaluate_on_holdout(
    holdout_supervised_frame: pd.DataFrame | None,
    supervised_frame: pd.DataFrame,
    feature_columns: list[str],
    baseline_model: SeasonalNaiveModel,
    best_boosting_params: dict[str, dict[str, Any]],
    settings: Settings,
    data_strategy: str,
) -> tuple[list[pd.DataFrame], ConformalForecaster | None, ConformalForecaster | None]:
    """Retrain both models on all training data and evaluate on the holdout split.

    ``None`` means no holdout was requested and skipping is correct (the synthetic-panel
    tests and the fair-cost backtest run this way). An EMPTY frame is a different thing: a
    holdout was requested and vanished during preparation or feature engineering, which is a
    defect, so it raises rather than returning a result that reads as a completed run without
    a holdout in it. Returning empty here is what hid the missing eval evaluation.

    Returns ``(holdout_predictions, holdout_boosting_model, holdout_cat_model)``.
    """
    if holdout_supervised_frame is None:
        return [], None, None
    if holdout_supervised_frame.empty:
        raise ValueError(
            f"[{data_strategy}] A holdout panel was supplied but its supervised frame is empty, "
            "so holdout evaluation would be skipped silently. Check that the holdout dates "
            "survive feature engineering and that its series overlap the training panel."
        )

    fields(logger, {"holdout": "reentrenando sobre el train completo antes de evaluar"})
    full_sub_train, full_calib, full_calib_group_ids = _split_train_calibration(
        supervised_frame, settings
    )
    holdout_boosting_model = _train_conformal_model(
        _instantiate_boosting_base(LightGBMModel, settings, best_boosting_params),
        full_sub_train,
        full_calib,
        full_calib_group_ids,
        feature_columns,
        settings,
    )
    holdout_cat_model = _train_conformal_model(
        _instantiate_boosting_base(CatBoostingModel, settings, best_boosting_params),
        full_sub_train,
        full_calib,
        full_calib_group_ids,
        feature_columns,
        settings,
    )

    predictions = [
        _build_baseline_predictions(
            validation_frame=holdout_supervised_frame,
            baseline_model=baseline_model,
            fold_id=HOLDOUT_FOLD_ID,
            settings=settings,
            data_strategy=data_strategy,
        ),
        _build_model_predictions(
            validation_frame=holdout_supervised_frame,
            feature_columns=feature_columns,
            model=holdout_boosting_model,
            fold_id=HOLDOUT_FOLD_ID,
            settings=settings,
            data_strategy=data_strategy,
        ),
        _build_model_predictions(
            validation_frame=holdout_supervised_frame,
            feature_columns=feature_columns,
            model=holdout_cat_model,
            fold_id=HOLDOUT_FOLD_ID,
            settings=settings,
            data_strategy=data_strategy,
        ),
    ]
    return predictions, holdout_boosting_model, holdout_cat_model


def _assemble_backtest_metadata(
    prepared_panel: pd.DataFrame,
    feature_metadata: Any,
    loop: _FoldLoopResult,
    predictions: pd.DataFrame,
    tuning_metadata: Any,
    settings: Settings,
    data_strategy: str,
) -> BacktestMetadata:
    """Assemble the structured backtest metadata from all run phases."""
    return BacktestMetadata(
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
            folds_created=len(loop.fold_run_metadata),
            folds=loop.fold_run_metadata,
        ),
        models=ModelRunMetadata(
            models_run=sorted(predictions["model_name"].dropna().unique().tolist()),
            quantiles=settings.models.quantiles,
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
            observations_seen=loop.drift_observations,
            alerts_detected=len(loop.detected_drifts),
            max_score=loop.max_drift_score,
            last_score=loop.last_drift_score,
        ),
    )


def run_experiment_from_frame(
    panel: pd.DataFrame,
    settings: Settings,
    data_strategy: str = "Observed",
    holdout_panel: pd.DataFrame | None = None,
    save_artifacts: bool = True,
) -> RunArtifacts:
    """Run the full backtesting pipeline from an in-memory panel."""
    quality_report = validate_prepared_panel(panel, settings)
    raise_on_blocking_data_quality(quality_report)

    prepared_panel = label_all_regimes(panel)

    supervised_frame, feature_metadata, holdout_supervised_frame = _build_supervised_frames(
        prepared_panel, holdout_panel, settings
    )
    feature_columns = feature_metadata.feature_columns

    # Walk-forward folds are built only on the original panel dates.
    folds = build_walk_forward_folds(
        panel=panel,
        validation_config=settings.validation,
        horizon=settings.dataset.horizon,
    )

    rule(logger, f"estrategia · {data_strategy}")
    fields(
        logger,
        {
            "features": f"{len(feature_columns)} columnas",
            "supervisado": f"{thousands(len(supervised_frame))} filas de "
            f"{thousands(feature_metadata.input_rows)}",
            "folds": thousands(len(folds)),
        },
    )

    in_run_params, tuning_metadata, tuning_pareto = _run_tuning_phase(
        supervised_frame, feature_columns, folds, settings, data_strategy
    )
    # `use_tuning` is an explicit request to search inside this run, so it wins over a
    # persisted winner. With it off -- which is every config in the repo -- each backend takes
    # its own tuned block when one matches this panel, and the YAML defaults otherwise.
    n_series = int(prepared_panel["series_id"].nunique())
    if settings.models.use_tuning:
        searched = in_run_params.model_dump()
        best_boosting_params = {"lightgbm": searched, "catboost": searched}
        fields(logger, {"hiperparámetros": "búsqueda dentro de la corrida (models.use_tuning)"})
    else:
        best_boosting_params = {}
        provenance = {}
        for backend in ("lightgbm", "catboost"):
            resolved, source = resolve_backend_params(settings, backend, n_series)
            best_boosting_params[backend] = resolved
            provenance[backend] = source
        fields(logger, provenance)

    baseline_model = SeasonalNaiveModel(
        seasonal_period=settings.models.seasonal_period,
        horizon=settings.dataset.horizon,
    ).fit(panel)

    loop = _run_fold_loop(
        folds,
        supervised_frame,
        feature_columns,
        baseline_model,
        best_boosting_params,
        settings,
        data_strategy,
    )

    holdout_preds, holdout_boosting_model, holdout_cat_model = _evaluate_on_holdout(
        holdout_supervised_frame,
        supervised_frame,
        feature_columns,
        baseline_model,
        best_boosting_params,
        settings,
        data_strategy,
    )
    fold_predictions = loop.fold_predictions + holdout_preds

    # Persist final models to the stable models directory for operational serving
    lgb_to_save = (
        holdout_boosting_model if holdout_boosting_model is not None else loop.boosting_model
    )
    cat_to_save = holdout_cat_model if holdout_cat_model is not None else loop.cat_model
    models_dir = settings.models.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_to_save in [lgb_to_save, cat_to_save]:
        if model_to_save is not None:
            model_to_save.save(model_file_path(models_dir, model_to_save.backend_name))

    if not fold_predictions:
        raise ValueError("Backtest did not produce any validation predictions.")

    # Every validation origin the models forecast. This is the evaluation set for
    # forecast quality: one row per series, date, model and strategy.
    validation_predictions = pd.concat(fold_predictions, ignore_index=True)

    # One decision per validation origin: the single-period Newsvendor quantity and
    # its cost, both attached fold by fold before the frames were concatenated.
    predictions = validation_predictions

    metrics_summary, fold_metrics = summarize_predictions(validation_predictions)
    cost_summary = summarize_costs(predictions, random_seed=settings.project.random_seed)
    sensitivity_summary = run_sensitivity_analysis(
        predictions=predictions,
        base_inventory_config=settings.inventory,
    )

    report_extra = ""
    if loop.detected_drifts:
        drift_str = ", ".join(
            [f"Fold {event.fold_id} (score={event.score:.2f})" for event in loop.detected_drifts]
        )
        report_extra = (
            f"**ALERT**: Concept drift detected and triggered adaptive retrains at: {drift_str}"
        )

    backtest_metadata = _assemble_backtest_metadata(
        prepared_panel,
        feature_metadata,
        loop,
        predictions,
        tuning_metadata,
        settings,
        data_strategy,
    )

    # Optional: Explainability (SHAP) on the last fold-trained model (seen the most data).
    shap_values = None
    if settings.reporting.make_plots:
        model_to_explain = loop.cat_model if loop.cat_model is not None else loop.boosting_model
        if model_to_explain is not None:
            fields(logger, {"SHAP": model_to_explain.model_name})
            shap_values = calculate_shap_values(
                model=model_to_explain,
                X=supervised_frame[feature_columns],
            )

    artifacts = RunArtifacts(
        prepared_panel=prepared_panel,
        supervised_frame=supervised_frame,
        predictions=predictions,
        validation_predictions=validation_predictions,
        metrics_summary=metrics_summary,
        fold_metrics=fold_metrics,
        cost_summary=cost_summary,
        sensitivity_summary=sensitivity_summary,
        tuning_pareto=tuning_pareto,
        data_quality_report=quality_report,
        drifts=loop.detected_drifts,
        report_extra=report_extra,
        backtest_metadata=backtest_metadata,
        shap_values=shap_values,
    )
    if not save_artifacts:
        return artifacts
    return write_run_artifacts(artifacts, settings)


def _init_prediction_frame(validation_frame: pd.DataFrame) -> pd.DataFrame:
    """Seed a prediction frame from a validation frame.

    Keeps the id/regime/target columns (plus latent-demand columns when present)
    and initializes ``y_true`` from the target. Callers then attach ``y_pred``,
    model metadata, quantiles, order quantity and costs.
    """
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
    return prediction_frame


def _build_baseline_predictions(
    validation_frame: pd.DataFrame,
    baseline_model: SeasonalNaiveModel,
    fold_id: int,
    settings: Settings,
    data_strategy: str = "Observed",
) -> pd.DataFrame:
    """Build baseline forecasts for one fold."""
    prediction_frame = _init_prediction_frame(validation_frame)
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
    return attach_inventory_costs(
        prediction_frame,
        settings.inventory,
    )


def _build_model_predictions(
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    model: ConformalForecaster,
    fold_id: int,
    settings: Settings,
    data_strategy: str = "Observed",
) -> pd.DataFrame:
    """Build model forecasts and attach costs."""
    prediction_frame = _init_prediction_frame(validation_frame)
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
        quantile_levels=[quantile_level_from_column(c) for c in quantile_columns],
    )
    return attach_inventory_costs(
        prediction_frame,
        settings.inventory,
    )


def _instantiate_champion_base_model(
    settings: Settings,
    n_series: int | None = None,
    backend_name: str | None = None,
    model_name: str | None = None,
) -> CatBoostingModel | LightGBMModel:
    """Build the champion's base model for the given backend/model identity.

    `backend_name`/`model_name` default to the static config when omitted, but callers that
    resolve a live champion (`resolve_champion_reference`) should pass those in explicitly --
    the config fields are only a fallback for the first run, before a registry exists.
    """
    resolved_backend_name = backend_name or settings.business.champion_backend_name
    resolved_model_name = model_name or settings.business.champion_model_name
    backend_key: BoostingBackend = (
        "catboost"
        if resolved_backend_name == "conformal_catboost_official"
        or resolved_model_name == "catboost"
        else "lightgbm"
    )
    series_count = n_series if n_series is not None else (settings.dataset.top_n_series or 500)
    resolved_params, source = resolve_backend_params(settings, backend_key, series_count)
    fields(logger, {f"campeón ({backend_key})": source})

    model_cls = CatBoostingModel if backend_key == "catboost" else LightGBMModel
    return model_cls(
        quantiles=settings.models.quantiles,
        random_seed=settings.project.random_seed,
        n_estimators=int(resolved_params["n_estimators"]),
        learning_rate=float(resolved_params["learning_rate"]),
        max_depth=int(resolved_params["max_depth"]),
        overstock_cost=settings.inventory.overstock_cost,
        stockout_cost=settings.inventory.stockout_cost,
        extra_params={k: v for k, v in resolved_params.items() if k not in CORE_PARAMS},
    )


def train_and_save_champion(
    settings: Settings,
    panel: pd.DataFrame,
    models_dir: Path | None = None,
    backend_name: str | None = None,
    model_name: str | None = None,
) -> Path:
    """Fit the champion model on the full panel and persist it to disk.

    `backend_name`/`model_name` identify which architecture to train; omit them to fall
    back to the static `BusinessConfig` fields (only correct before a champion registry
    exists -- callers with a registry should resolve it and pass the identity in).
    """
    supervised_frame, feature_metadata = build_supervised_frame(
        panel=panel,
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )
    feature_columns = feature_metadata.feature_columns

    train_frame, calib_frame, calib_group_ids = _split_train_calibration(supervised_frame, settings)

    n_series = int(panel["series_id"].nunique())
    conformal = _train_conformal_model(
        _instantiate_champion_base_model(settings, n_series, backend_name, model_name),
        train_frame,
        calib_frame,
        calib_group_ids,
        feature_columns,
        settings,
    )

    resolved_dir = models_dir if models_dir is not None else settings.models.models_dir
    resolved_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_file_path(resolved_dir, conformal.backend_name)
    conformal.save(model_path)
    return model_path


def run_retrain(settings: Settings) -> Path:
    """Load every split, train the champion on all of it, and write the model to disk + MLflow."""
    rule(logger, "reentreno del campeón")
    started = time.monotonic()

    loaded = Table(logger, {"split": 10, "series": 8, "filas": 10, "tiempo": 6})
    splits: list[pd.DataFrame] = []
    for split in settings.dataset.splits:
        mark = time.monotonic()
        panel = load_prepared_panel(
            dataset_config=settings.dataset,
            preprocessing_config=settings.preprocessing,
            split=split,
        )
        splits.append(panel)
        loaded.row(
            {
                "split": split,
                "series": thousands(panel["series_id"].nunique()),
                "filas": thousands(len(panel)),
                "tiempo": f"{time.monotonic() - mark:.0f}s",
            }
        )

    raw_panel = pd.concat(splits, ignore_index=True)
    quality_report = validate_prepared_panel(raw_panel, settings)
    raise_on_blocking_data_quality(quality_report)

    strategy = settings.preprocessing.imputation_strategy
    if strategy != "none":
        imputer = LatentDemandImputer(
            strategy=strategy,
            model_path=settings.models.models_dir / settings.models.imputation_params_filename,
        )
        raw_panel = imputer.impute(raw_panel)

    champion_registry = load_champion_registry(champion_registry_path(settings))
    champion_reference = resolve_champion_reference(settings, champion_registry)

    n_series = int(raw_panel["series_id"].nunique())
    fields(
        logger,
        {
            "panel": f"{thousands(n_series)} series, {thousands(len(raw_panel))} filas",
            "ventana": f"{raw_panel['date'].min().date()} → {raw_panel['date'].max().date()}",
            "campeón": f"{champion_reference.backend_name} (fuente: {champion_reference.source})",
            "calidad": f"{quality_report.warning_count} avisos",
        },
    )

    run_name = (
        settings.reporting.run_name
        if settings.reporting and settings.reporting.run_name
        else "retrain_champion"
    )
    with open_run_directory(run_name, EXPERIMENT_RUNS) as run_dir:
        model_path = train_and_save_champion(
            settings,
            raw_panel,
            backend_name=champion_reference.backend_name,
            model_name=champion_reference.model_name,
        )
        shutil.copy2(model_path, run_dir / model_path.name)

        try:
            sample_supervised, feat_meta = build_supervised_frame(
                panel=raw_panel.tail(200),
                feature_config=settings.features,
                horizon=settings.dataset.horizon,
            )
            sample_input = sample_supervised.loc[:, feat_meta.feature_columns].dropna().head(10)
        except Exception:
            sample_input = None

        log_retrain_metadata(
            settings=settings,
            model_path=model_path,
            n_series=n_series,
            panel_rows=len(raw_panel),
            supervised_rows=len(raw_panel),
            sample_input=sample_input,
        )

    fields(
        logger,
        {"escrito": model_path, "tiempo": f"{time.monotonic() - started:.0f}s"},
    )
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
    rule(logger, "scoring diario de reposición")
    started = time.monotonic()
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
        champion_reference = resolve_champion_reference(
            settings, load_champion_registry(champion_registry_path(settings))
        )
        model_path = model_file_path(models_dir, champion_reference.backend_name)
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}. Run a backtest or retrain first.")
    model = ConformalForecaster.load(model_path)

    prepared_panel = label_all_regimes(panel)

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
    result = write_run_artifacts(artifacts, settings)

    n_exceptions = len(result.exceptions) if result.exceptions is not None else 0
    decision_date = str(predictions["decision_date"].iloc[0]) if not predictions.empty else "-"
    fields(
        logger,
        {
            "campeón": f"{model.backend_name}",
            "series": f"{thousands(len(predictions))}",
            "fecha decisión": decision_date,
            "excepciones": f"{n_exceptions} avisos",
            "tiempo": f"{time.monotonic() - started:.1f}s",
        },
    )
    return result


def _build_scoring_predictions(
    inference_frame: pd.DataFrame,
    feature_columns: list[str],
    model: ConformalForecaster,
    settings: Settings,
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
        quantile_levels=[quantile_level_from_column(c) for c in quantile_columns],
    )
    # Cold-start rows have no calibrated quantiles, so the interpolation above resolves to
    # NaN for them. There is no distribution to apply a critical fractile to; the fallback
    # mean IS the order quantity for these rows.
    if cold_mask.any():
        fallback_orders = frame.loc[cold_mask, "y_pred"].to_numpy(dtype=float)
        if settings.inventory.clip_negative_orders:
            fallback_orders = np.maximum(fallback_orders, 0.0)
        frame.loc[cold_mask, "order_quantity"] = fallback_orders
    return attach_inventory_costs(frame, settings.inventory)
