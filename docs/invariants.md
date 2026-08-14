# Invariants

These rules protect the experimental validity and architecture of the project.

## Data Layer

1. Raw FreshRetailNet column names are isolated to `src/retail_forecasting/data/dataset.py`.

   Raw names include `dt`, `sale_amount`, and `stock_hour6_22_cnt`.

2. The prepared panel must use canonical project names:

   - `date`
   - `series_id`
   - `observed_demand`
   - `stockout_hours`

3. `series_id` means `store_id + "_" + product_id`.

   No downstream module should redefine it with a different key.

4. Data loading may cache raw and processed parquet files, but modeling logic must not depend on whether the panel came from cache or remote storage.

## Feature Engineering

5. `target_lead_time_demand` is created only in `src/retail_forecasting/features/engineering.py`.

6. The target is the sum of observed demand from the decision date through the configured horizon.

   For horizon `h`, row date `t` uses demand from `t` through `t + h - 1`.

7. Historical demand features must use positive lags.

   Examples allowed:

   - `grouped["observed_demand"].shift(1)`
   - `series.shift(1).rolling(...)`

   Examples not allowed:

   - `shift(0)` for observed demand
   - `shift(-1)` for features
   - rolling windows over unshifted observed demand

8. Variables not guaranteed to be known ex ante must enter the model only as lagged features.

   This includes realized weather, discount, and stockout information.

9. Calendar features may use the row date because they are known at the decision time.

## Validation

10. Validation must be temporal walk-forward.

    Random train/test splits are not valid for the main experiment.

11. Training rows for a fold must not have targets overlapping the validation period.

    The current invariant is:

    ```text
    train_end_date = validation_start_date - horizon
    ```

12. The conformal calibration split must embargo the horizon as well.

    Splitting a fold's training frame into sub-train and calibration follows the
    same rule as the fold split itself:

    ```text
    sub_train_end_date = calibration_start_date - horizon
    ```

    Without the gap the last `horizon - 1` sub-train rows carry targets covering
    days inside the calibration window, the conformity scores come out
    optimistically small, and the intervals undercover.

13. Forecast metrics are summarized over every validation origin.

    The dynamic Order-Up-To simulation narrows the frame to one decision per
    series and fold, because the ordering cadence equals the fold length. That
    subset is the correct domain for `summarize_costs()` and the sensitivity
    analysis, and the wrong one for `summarize_predictions()`: MAE, Winkler and
    coverage must not be computed on it.

14. `dataset.use_eval_as_holdout = true` must remain unsupported until the temporal meaning of the official eval split is documented and tested.

## Models

15. Models receive already-built feature matrices and targets.

    They must not load datasets, build raw features, choose order quantities, or write reports.

16. Forecast outputs must be non-negative demand forecasts.

17. Quantile forecasts must be monotonically non-decreasing by quantile level.

18. Quantile column names must be generated with `quantile_column_name()`.

## Inventory

19. Inventory decisions live in `src/retail_forecasting/inventory/`.

20. The current inventory policy is single-period newsvendor.

21. `order_quantity` is derived from point forecasts or quantile forecasts plus the critical fractile.

22. Cost columns must be derived from `y_true` and `order_quantity`.

   Required cost columns:

   - `overstock_units`
   - `stockout_units`
   - `overstock_cost`
   - `stockout_cost`
   - `total_cost`

## Evaluation and Reporting

23. Evaluation summarizes predictions and costs.

    It must not retrain models, mutate forecasts, or change order quantities.

24. The primary ranking metric for the TFG is economic cost, not MAE.

25. MAE and RMSE are diagnostic metrics.

26. Pinball loss and coverage are probabilistic diagnostics when quantile forecasts exist.

## Generated Outputs

27. Reports, plots, cached datasets, notebook outputs, and PDFs are generated artifacts.

    Do not treat them as source of truth for pipeline behavior.

28. Durable design decisions belong in the main methodology and design documents.

## Web Layer

See `docs/web_layer.md` for the full description.

29. The web layer renders and orchestrates. It must not duplicate forecasting,
    inventory, or evaluation logic.

    A new number on screen is computed in `api/services/` or in the pipeline,
    never in a view or a template.

30. `api/services/` must not import Django.

    Django only enters through `api/store.py`, `api/views/` and the settings
    module, which keeps the services unit-testable without a web client.

31. The dashboard has no database.

    System state lives in `reports/`; the only web-owned state is the operator
    session, carried in a signed cookie. Adding a relational store is a design
    change, not an implementation detail.

32. Missing artifacts render an explicit empty state that names the command
    which produces them.

    Never fabricate placeholder values. A run without `drift_report.json` says
    so; it does not render an empty panel that reads as "no drift".

33. User-supplied run names are validated against the artifact catalogue before
    touching the filesystem.

    `ArtifactStore.resolve_run` and the EDA figure lookup exist to make path
    traversal impossible by construction.

34. Charts are rendered as SVG on the server.

    No client-side charting library. The scenario parameters live in the query
    string so every dashboard state is a shareable, reloadable URL.

## Rolling-Origin Backtest (OPS plane)

35. The inference origin for a decision taken on date `d` is the row dated `d`.

    A row dated `d` targets demand over `[d, d + h - 1]` (invariant 6) and only
    carries lagged features, so slicing history at `< d` scores a forecast for
    `[d - 1, d + h - 2]` against the actuals of `[d, d + h - 1]`. This is not
    leakage, but it costs a one-day-stale forecast and, under a trend, deflates
    both accuracy and conformal coverage. `simulation/operations.py` slices at
    `<= d`.

36. Aggregates over the backtest use a non-overlapping origin grid.

    Scoring runs daily and the horizon is `h` days, so consecutive origins share
    `h - 1` days of demand. Summing every daily origin counts the same demand up
    to `h` times. Both `_summarize_cadences` and the dashboard keep one origin
    every `h` days.

37. Origins whose actuals have not fully landed are never aggregated.

    Their `y_true` is a partial-window sum, so any cost computed against it
    understates the shortage half. They are scored and stored, then excluded from
    every summary, chart and KPI.

38. Retrain cadences are compared paired, with an interval, and the comparison
    declares itself underpowered when there are too few origins.

    A raw cost difference over a handful of weeks ranks nothing. See
    `_compare_cadences`.

39. The OPS plane is not an inventory-state simulation.

    Each origin is an independent single-period Newsvendor decision: no carried
    stock, no order pipeline, no lead time, and truth is censored observed demand
    with no imputation. Its costs rank policies; they are not a replenishment
    ledger. Naming it a "simulation" in user-facing text is what this invariant
    exists to prevent.

40. `tune_imputation_lgbm()` (the `tune_imputation` run mode, `forecasting/imputation_tuning.py`)
    must load only `split="train"`, never `split="eval"` -- the same rule as invariant 14,
    applied to the supervised imputer's own hyperparameter search. It also persists only the
    winning hyperparameters, never fitted model weights: `LatentDemandImputer` always re-fits
    on the current panel's own clean days, so tuning cannot change the leakage or
    feature-space properties of imputation, only which 3 numbers the fit uses.

    Every `LatentDemandImputer` built in `forecasting/pipeline.py` must be given that
    params file via `model_path`. The constructor falls back to the untuned defaults when
    it is omitted, so a call site that forgets it runs a different `supervised` model from
    the rest of the pipeline -- and the imputation study would be comparing a strategy the
    champion run never uses. Enforced by `test_pipeline_imputers_all_read_the_tuned_params_file`.

41. The imputation search must select and validate on disjoint synthetic-censoring draws, and
    must not persist a winner that loses to the untuned defaults on the validation draws.

    On a single draw the MAE spread between trials (12-40% measured) is an order of magnitude
    larger than the differences between hyperparameter sets (~1%), so a single-draw search
    selects the lucky trial and its own objective value is not evidence of a gain -- a run that
    reported -1.4% in-sample measured -0.32% (CI95 crossing zero) on fresh draws. Hence the
    objective averages over `N_SELECTION_HOLDOUTS` draws, `improvement_pct` is computed only on
    the disjoint `N_VALIDATION_HOLDOUTS` draws, and `imputation_lgbm_params.json` is written
    only when the bootstrap `improvement_ci95` over those draws lies entirely below zero.
    `best_mae_selection` in the metadata is in-sample for the search and must never be quoted
    as the improvement.

    The decision is the interval, not the sign of the mean: a point comparison passed a winner
    at -1.14% that then measured -0.45% with a straddling interval on fresh draws. The same
    rule applies to anything read out of this run for the thesis -- an improvement whose
    interval includes zero is a null result and must be reported as one.

42. The synthetic-censoring holdout cannot rank imputation RECONCILIATION rules -- only
    hyperparameters and teacher models.

    `_synthetic_censor_holdout()` builds its ground truth as `observed = truth * (1 - r)`.
    Any rule that inverts that relation scores near-zero error by construction, not by
    merit: `observed / (1 - r)` measures MAE 0.02 against 0.74 for the rule it replaces.
    Hyperparameter comparisons stay valid because every candidate shares one reconciliation
    rule, so the assumption cancels. Reconciliation changes must be argued on modelling
    grounds and must never quote a reconstruction-MAE gain from this holdout -- including in
    `imputation_quality.csv`, which measures agreement with the generator rather than
    imputation quality in the field.
