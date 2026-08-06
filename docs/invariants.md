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

12. `dataset.use_eval_as_holdout = true` must remain unsupported until the temporal meaning of the official eval split is documented and tested.

## Models

13. Models receive already-built feature matrices and targets.

    They must not load datasets, build raw features, choose order quantities, or write reports.

14. Forecast outputs must be non-negative demand forecasts.

15. Quantile forecasts must be monotonically non-decreasing by quantile level.

16. Quantile column names must be generated with `quantile_column_name()`.

## Inventory

17. Inventory decisions live in `src/retail_forecasting/inventory/`.

18. The current inventory policy is single-period newsvendor.

19. `order_quantity` is derived from point forecasts or quantile forecasts plus the critical fractile.

20. Cost columns must be derived from `y_true` and `order_quantity`.

   Required cost columns:

   - `overstock_units`
   - `stockout_units`
   - `overstock_cost`
   - `stockout_cost`
   - `total_cost`

## Evaluation and Reporting

21. Evaluation summarizes predictions and costs.

    It must not retrain models, mutate forecasts, or change order quantities.

22. The primary ranking metric for the TFG is economic cost, not MAE.

23. MAE and RMSE are diagnostic metrics.

24. Pinball loss and coverage are probabilistic diagnostics when quantile forecasts exist.

## Generated Outputs

25. Reports, plots, cached datasets, notebook outputs, and PDFs are generated artifacts.

    Do not treat them as source of truth for pipeline behavior.

26. Durable design decisions belong in the main methodology and design documents.

## Web Layer

See `docs/web_layer.md` for the full description.

27. The web layer renders and orchestrates. It must not duplicate forecasting,
    inventory, or evaluation logic.

    A new number on screen is computed in `api/services/` or in the pipeline,
    never in a view or a template.

28. `api/services/` must not import Django.

    Django only enters through `api/store.py`, `api/views/` and the settings
    module, which keeps the services unit-testable without a web client.

29. The dashboard has no database.

    System state lives in `reports/`; the only web-owned state is the operator
    session, carried in a signed cookie. Adding a relational store is a design
    change, not an implementation detail.

30. Missing artifacts render an explicit empty state that names the command
    which produces them.

    Never fabricate placeholder values. A run without `drift_report.json` says
    so; it does not render an empty panel that reads as "no drift".

31. User-supplied run names are validated against the artifact catalogue before
    touching the filesystem.

    `ArtifactStore.resolve_run` and the EDA figure lookup exist to make path
    traversal impossible by construction.

32. Charts are rendered as SVG on the server.

    No client-side charting library. The scenario parameters live in the query
    string so every dashboard state is a shareable, reloadable URL.
