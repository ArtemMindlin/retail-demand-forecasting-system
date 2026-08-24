from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig, Settings, build_config_hash
from retail_forecasting.contracts.contracts_backtesting import FoldRunMetadata
from retail_forecasting.contracts.contracts_config import ImputationStrategy
from retail_forecasting.contracts.contracts_drift import DriftDetectorMetadata, DriftEvent
from retail_forecasting.contracts.contracts_tuning import BoostingParams
from retail_forecasting.data.censorship import (
    SYNTHETIC_CENSORING_EVAL_FRACTION,
    LatentDemandImputer,
    synthetic_censor_holdout,
)
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
)
from retail_forecasting.models.boosting import LightGBMModel
from retail_forecasting.models.catboosting import CatBoostingModel
from retail_forecasting.models.conformal import ConformalForecaster
from retail_forecasting.models.naive import SeasonalNaiveModel
from retail_forecasting.models.optimization import HyperparameterTuner
from retail_forecasting.tracking import (
    EXPERIMENT_RUNS,
    open_run_directory,
)
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
    params: BoostingParams,
) -> LightGBMModel | CatBoostingModel:
    """Build a boosting base model from tuned params and inventory costs."""
    return model_cls(
        quantiles=settings.models.quantiles,
        random_seed=settings.project.random_seed,
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        overstock_cost=settings.inventory.overstock_cost,
        stockout_cost=settings.inventory.stockout_cost,
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


def evaluate_fair_inventory_cost(
    panel: pd.DataFrame,
    inventory_config: InventoryConfig,
    seed: int,
    eval_fraction: float = SYNTHETIC_CENSORING_EVAL_FRACTION,
    imputer_params_path: Path | None = None,
) -> pd.DataFrame:
    """Compare strategies on inventory cost against a COMMON ground truth.

    Returns one row per strategy: signal_mae, total_cost, fill_rate, mean_order, n_eval.
    """
    censored, eval_idx, true_demand = synthetic_censor_holdout(panel, seed, eval_fraction)
    n_eval = len(eval_idx)

    # Flat cost coefficients so every strategy is charged identically (isolates the signal).
    flat_config = inventory_config.model_copy(update={"use_series_costs": False})
    cr = flat_config.stockout_cost / (flat_config.stockout_cost + flat_config.overstock_cost)
    z = statistics.NormalDist().inv_cdf(cr)
    sigma = float(np.std(true_demand))  # shared safety-stock scale (same scalar for all)

    if "series_id" in panel.columns:
        series_ids = panel.loc[eval_idx, "series_id"].astype(str).to_numpy()
    else:
        series_ids = np.arange(n_eval)

    total_demand = float(true_demand.sum())
    records: list[dict[str, Any]] = []
    # "none" leaves the censored sale untouched → it IS the Observed (deflated) signal.
    strategies: tuple[tuple[ImputationStrategy, str], ...] = (
        ("none", "Observed"),
        ("supervised", "Latent_supervised"),
        ("historical_mean", "Latent_historical_mean"),
        ("clipped_scaling", "Latent_clipped_scaling"),
    )
    for strategy, label in strategies:
        imputed = LatentDemandImputer(strategy=strategy, model_path=imputer_params_path).impute(
            censored
        )
        signal = imputed.loc[eval_idx, "latent_demand_est"].astype(float).to_numpy()
        q_star = np.maximum(signal + z * sigma, 0.0)

        costed = attach_inventory_costs(
            pd.DataFrame(
                {"series_id": series_ids, "y_true": true_demand, "order_quantity": q_star}
            ),
            flat_config,
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
                "mean_order": float(np.mean(q_star)),
                "n_eval": int(n_eval),
            }
        )
    return pd.DataFrame(records)


def run_fair_cost_backtest(settings: Settings, n_series: int = 30) -> Path:
    """Lightweight backtest (no training) validating the fair inventory-cost comparison.

    Returns:
        The created run directory path.
    """
    print("\n" + "=" * 50)
    print("🧪 FAIR INVENTORY-COST BACKTEST (common ground truth · no forecasting)")
    print("=" * 50 + "\n")
    print("📂 Loading train panel...")
    panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    source_series = panel["series_id"].nunique() if "series_id" in panel.columns else 0
    if "series_id" in panel.columns and n_series:
        unique_ids = panel["series_id"].drop_duplicates().to_numpy()
        if len(unique_ids) > n_series:
            rng = np.random.default_rng(settings.project.random_seed)
            keep = rng.choice(unique_ids, size=n_series, replace=False)
            panel = panel[panel["series_id"].isin(keep)].reset_index(drop=True)
    n_kept = panel["series_id"].nunique() if "series_id" in panel.columns else 0
    print(f"✅ Panel: {len(panel):,} rows · {n_kept} of {source_series} series (sample)")

    result = evaluate_fair_inventory_cost(
        panel,
        settings.inventory,
        seed=settings.project.random_seed,
        imputer_params_path=(
            settings.models.models_dir / settings.models.imputation_params_filename
        ),
    )
    # Provenance: the ranking flips with the source panel, so the artifact must say
    # which one it came from rather than leaving it to the reader's memory.
    result.insert(1, "source_panel_series", source_series)
    result.insert(2, "sampled_series", n_kept)

    with open_run_directory(settings.reporting.run_name, EXPERIMENT_RUNS) as run_dir:
        out_path = run_dir / "fair_cost_backtest.csv"
        result.to_csv(out_path, index=False)

        print("\n── Inventory cost against a COMMON ground truth (lower = better) ──")
        print(result.to_string(index=False))
        print(f"\n✅ Fair-cost backtest written to: {out_path}\n")
        return run_dir


def run_experiment(settings: Settings) -> RunArtifacts:
    """Run the end-to-end experiment comparing Observed vs Latent demand."""
    rule(logger, "experimento walk-forward")
    started = time.monotonic()

    # 1. Load Original Panel
    raw_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="train",
    )
    quality_report = validate_prepared_panel(raw_panel, settings)
    raise_on_blocking_data_quality(quality_report)

    # Load external holdout (eval) split
    holdout_panel = load_prepared_panel(
        dataset_config=settings.dataset,
        preprocessing_config=settings.preprocessing,
        split="eval",
    )

    strategy_name = settings.preprocessing.imputation_strategy
    fields(
        logger,
        {
            "panel": f"{thousands(quality_report.checked_series)} series, "
            f"{thousands(len(raw_panel))} filas",
            "ventana": f"{raw_panel['date'].min().date()} → {raw_panel['date'].max().date()}",
            "holdout": f"{thousands(len(holdout_panel))} filas "
            f"({holdout_panel['date'].min().date()} → {holdout_panel['date'].max().date()})",
            "calidad": f"{quality_report.warning_count} avisos",
            "estrategias": f"Observed · Latent_{strategy_name}",
            "folds": f"{settings.validation.n_folds} × {settings.validation.fold_size_days}d, "
            f"horizonte {settings.dataset.horizon}d",
        },
    )

    # 2. Run Strategy A: Observed Demand (Baseline)
    artifacts_obs = run_experiment_from_frame(
        raw_panel,
        settings,
        data_strategy="Observed",
        holdout_panel=holdout_panel,
        save_artifacts=False,
    )

    # 3. Run Strategy B: Latent Demand (Imputed)
    imputer = LatentDemandImputer(
        strategy=strategy_name,
        model_path=settings.models.models_dir / settings.models.imputation_params_filename,
    )
    imputed_panel = imputer.impute(raw_panel)

    imputed_holdout = None
    if holdout_panel is not None:
        imputed_holdout = imputer.impute(holdout_panel)

    artifacts_latent = run_experiment_from_frame(
        imputed_panel,
        settings,
        data_strategy=f"Latent_{strategy_name}",
        holdout_panel=imputed_holdout,
        save_artifacts=False,
    )

    # 4. Merge Artifacts
    merged_validation = pd.concat(
        [
            frame
            for frame in (
                artifacts_obs.validation_predictions,
                artifacts_latent.validation_predictions,
            )
            if frame is not None
        ],
        ignore_index=True,
    )

    merged_predictions = merged_validation

    merged_metrics, merged_folds = summarize_predictions(merged_validation)
    merged_costs = summarize_costs(merged_predictions)
    merged_sens = run_sensitivity_analysis(merged_predictions, settings.inventory)
    tuning_fronts = [
        front
        for front in (artifacts_obs.tuning_pareto, artifacts_latent.tuning_pareto)
        if front is not None
    ]
    merged_tuning_pareto = pd.concat(tuning_fronts, ignore_index=True) if tuning_fronts else None

    combined_metadata = None
    if artifacts_obs.backtest_metadata is not None:
        combined_metadata = artifacts_obs.backtest_metadata.model_copy(
            update={
                "data_strategy": f"Observed+Latent_{strategy_name}",
                "created_at": utc_timestamp(),
                "models": ModelRunMetadata(
                    models_run=sorted(merged_predictions["model_name"].dropna().unique().tolist()),
                    quantiles=settings.models.quantiles,
                    use_tuning=settings.models.use_tuning,
                    retrain_each_fold=settings.validation.retrain_each_fold,
                ),
            }
        )

    final_artifacts = RunArtifacts(
        prepared_panel=raw_panel,
        supervised_frame=artifacts_obs.supervised_frame,
        predictions=merged_predictions,
        validation_predictions=merged_validation,
        metrics_summary=merged_metrics,
        fold_metrics=merged_folds,
        cost_summary=merged_costs,
        sensitivity_summary=merged_sens,
        tuning_pareto=merged_tuning_pareto,
        data_quality_report=quality_report,
        drifts=artifacts_obs.drifts,
        backtest_metadata=combined_metadata,
        shap_values=artifacts_obs.shap_values,
    )

    artifacts_with_files = write_run_artifacts(final_artifacts, settings)
    fields(
        logger,
        {
            "escrito": artifacts_with_files.run_directory,
            "tiempo": f"{time.monotonic() - started:.0f}s",
        },
    )
    return artifacts_with_files


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
    panel: pd.DataFrame,
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
        combined_prepared = label_all_regimes(pd.concat([panel, holdout_panel], ignore_index=True))
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
    best_boosting_params: BoostingParams,
    settings: Settings,
    data_strategy: str,
    series_cost_profile: pd.DataFrame | None,
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
                series_cost_profile=series_cost_profile,
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
            series_cost_profile=series_cost_profile,
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
                series_cost_profile=series_cost_profile,
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
    best_boosting_params: BoostingParams,
    settings: Settings,
    data_strategy: str,
    series_cost_profile: pd.DataFrame | None,
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
            series_cost_profile=series_cost_profile,
        ),
        _build_model_predictions(
            validation_frame=holdout_supervised_frame,
            feature_columns=feature_columns,
            model=holdout_boosting_model,
            fold_id=HOLDOUT_FOLD_ID,
            settings=settings,
            data_strategy=data_strategy,
            series_cost_profile=series_cost_profile,
        ),
        _build_model_predictions(
            validation_frame=holdout_supervised_frame,
            feature_columns=feature_columns,
            model=holdout_cat_model,
            fold_id=HOLDOUT_FOLD_ID,
            settings=settings,
            data_strategy=data_strategy,
            series_cost_profile=series_cost_profile,
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
    series_cost_profile = None
    if settings.inventory.use_series_costs:
        series_cost_profile = build_series_cost_profile(prepared_panel, settings.inventory)

    supervised_frame, feature_metadata, holdout_supervised_frame = _build_supervised_frames(
        panel, prepared_panel, holdout_panel, settings
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

    best_boosting_params, tuning_metadata, tuning_pareto = _run_tuning_phase(
        supervised_frame, feature_columns, folds, settings, data_strategy
    )

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
        series_cost_profile,
    )

    holdout_preds, holdout_boosting_model, holdout_cat_model = _evaluate_on_holdout(
        holdout_supervised_frame,
        supervised_frame,
        feature_columns,
        baseline_model,
        best_boosting_params,
        settings,
        data_strategy,
        series_cost_profile,
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
    cost_summary = summarize_costs(predictions)
    sensitivity_summary = run_sensitivity_analysis(
        predictions=predictions,
        base_inventory_config=settings.inventory,
        series_cost_profile=series_cost_profile,
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
    series_cost_profile: pd.DataFrame | None = None,
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
        series_cost_profile=series_cost_profile,
    )
    return attach_inventory_costs(
        prediction_frame,
        settings.inventory,
        series_cost_profile=series_cost_profile,
    )


def _instantiate_champion_base_model(settings: Settings) -> CatBoostingModel | LightGBMModel:
    backend = settings.business.champion_backend_name
    if backend == "conformal_catboost_official":
        return CatBoostingModel(
            quantiles=settings.models.quantiles,
            random_seed=settings.project.random_seed,
            n_estimators=settings.models.n_estimators,
            learning_rate=settings.models.learning_rate,
            max_depth=settings.models.max_depth,
            overstock_cost=settings.inventory.overstock_cost,
            stockout_cost=settings.inventory.stockout_cost,
        )
    return LightGBMModel(
        quantiles=settings.models.quantiles,
        random_seed=settings.project.random_seed,
        n_estimators=settings.models.n_estimators,
        learning_rate=settings.models.learning_rate,
        max_depth=settings.models.max_depth,
        overstock_cost=settings.inventory.overstock_cost,
        stockout_cost=settings.inventory.stockout_cost,
    )


def train_and_save_champion(
    settings: Settings,
    panel: pd.DataFrame,
    models_dir: Path | None = None,
) -> Path:
    """Fit the configured champion model on the full panel and persist it to disk."""
    supervised_frame, feature_metadata = build_supervised_frame(
        panel=panel,
        feature_config=settings.features,
        horizon=settings.dataset.horizon,
    )
    feature_columns = feature_metadata.feature_columns

    train_frame, calib_frame, calib_group_ids = _split_train_calibration(supervised_frame, settings)

    conformal = _train_conformal_model(
        _instantiate_champion_base_model(settings),
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
    """Load every split, train the champion on all of it, and write the model to disk."""
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

    fields(
        logger,
        {
            "panel": f"{thousands(raw_panel['series_id'].nunique())} series, "
            f"{thousands(len(raw_panel))} filas",
            "ventana": f"{raw_panel['date'].min().date()} → {raw_panel['date'].max().date()}",
            "campeón": settings.business.champion_backend_name,
            "calidad": f"{quality_report.warning_count} avisos",
        },
    )

    model_path = train_and_save_champion(settings, raw_panel)
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
        model_path = model_file_path(models_dir, champion_backend)
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
        quantile_levels=[quantile_level_from_column(c) for c in quantile_columns],
        series_cost_profile=series_cost_profile,
    )
    return attach_inventory_costs(
        frame, settings.inventory, series_cost_profile=series_cost_profile
    )
