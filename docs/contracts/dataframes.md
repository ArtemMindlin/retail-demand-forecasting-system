# Dataframe Contracts

This document defines the dataframe schemas used by the v2 pipeline.

The goal is to make agent changes safer: code may evolve, but these contracts should remain stable unless the experiment design changes deliberately.

## Raw FreshRetailNet Split

Created by:

- remote parquet read in `load_raw_split()`
- optional local raw cache under `data/raw/fresh_retailnet/`

Known raw columns used by the current implementation:

| Column | Meaning |
| --- | --- |
| `dt` | raw date |
| `sale_amount` | observed demand signal |
| `stock_hour6_22_cnt` | stockout-hours signal |
| `discount` | discount/context signal |
| `holiday_flag` | holiday indicator |
| `activity_flag` | activity/promotion indicator |
| `precpt` | precipitation |
| `avg_temperature` | average temperature |
| `avg_humidity` | average humidity |
| `avg_wind_level` | average wind level |
| `city_id` | city id |
| `store_id` | store id |
| `management_group_id` | management group id |
| `first_category_id` | product hierarchy |
| `second_category_id` | product hierarchy |
| `third_category_id` | product hierarchy |
| `product_id` | product id |

Raw names should not be used outside the data layer.

## Prepared Panel

Created by:

- `prepare_daily_panel()`
- `load_prepared_panel()`

Required columns:

| Column | Type expectation | Meaning |
| --- | --- | --- |
| `date` | datetime-like | forecast origin / panel date |
| `series_id` | string-like | `store_id + "_" + product_id` |
| `observed_demand` | numeric | observed demand used by the current pipeline |
| `stockout_hours` | numeric | stockout intensity/context |
| `discount` | numeric | discount/context |
| `holiday_flag` | numeric/binary | known calendar/context flag |
| `activity_flag` | numeric/binary | known activity/context flag |
| `precpt` | numeric | realized weather, lag before model use |
| `avg_temperature` | numeric | realized weather, lag before model use |
| `avg_humidity` | numeric | realized weather, lag before model use |
| `avg_wind_level` | numeric | realized weather, lag before model use |
| `city_id` | id-like | static id |
| `store_id` | id-like | static id |
| `management_group_id` | id-like | static id |
| `first_category_id` | id-like | static id |
| `second_category_id` | id-like | static id |
| `third_category_id` | id-like | static id |
| `product_id` | id-like | static id |

Expected properties:

- one row per `series_id` and `date`
- sorted by `series_id`, then `date` before temporal feature construction
- no negative `observed_demand` when `drop_negative_sales = true`
- only series with at least `dataset.min_history_days` unique dates

## Supervised Frame

Created by:

- `build_feature_frame()`
- `build_supervised_frame()`
- `build_inference_frame()`
- `build_inference_frame_with_fallback()`

`build_feature_frame()` contains the shared feature transformation used by both
training/backtesting and inference. `build_supervised_frame()` adds
`target_lead_time_demand` and drops rows without full feature and target
availability. `build_inference_frame()` does not build a target and returns the
latest feature-complete row per `series_id`. `build_inference_frame_with_fallback()`
returns the latest row per `series_id` regardless of feature completeness and
adds routing information for model inference vs. cold-start fallback.

All three builders return a Pydantic `FeatureMetadata` object:

```python
frame, metadata = build_supervised_frame(...)
feature_columns = metadata.feature_columns
```

The metadata records the feature columns, target column, horizon, configured
lags/windows, input/output row counts, rows dropped due to missing targets or
features, and rows skipped in inference because they were not the latest valid
origin.

`build_inference_frame_with_fallback()` returns an `InferenceFallbackMetadata`
object instead. It records the feature columns, horizon, configured
lags/windows, total output rows, model-routed rows, cold-start rows, and the
count assigned to each fallback level.

Required carry-through columns:

| Column | Meaning |
| --- | --- |
| `date` | forecast origin |
| `series_id` | series key |
| `observed_demand` | observed demand at origin date |
| `stockout_hours` | stockout signal at origin date |
| `stockout_regime` | regime label if added before feature engineering |
| `velocity_regime` | velocity regime label |
| `promo_regime` | promotional regime label |
| `seasonal_regime` | seasonal regime label |

Required target columns:

| Column | Meaning |
| --- | --- |
| `target_lead_time_demand` | observed demand over the forecast horizon |
| `target_horizon_days` | configured horizon used to build target |

Feature column families:

| Pattern | Meaning |
| --- | --- |
| `demand_lag_<n>` | demand lagged by `n` days |
| `demand_roll_mean_<n>` | shifted rolling mean over past `n` days |
| `demand_roll_sum_<n>` | shifted rolling sum over past `n` days |
| `demand_roll_std_<n>` | shifted rolling std over past `n` days |
| `discount_lag_<n>` | lagged discount |
| `stockout_lag_<n>` | lagged stockout hours |
| `stockout_roll_mean_<n>` | shifted rolling stockout mean |
| `<weather_column>_lag_<n>` | lagged realized weather |
| calendar columns | known ex-ante calendar features |
| static id columns | store/product/category ids |

Target semantics:

```text
target_lead_time_demand(t, h) = sum(observed_demand[t : t + h - 1])
```

Feature semantics:

```text
features(t) may use information from dates <= t - 1,
except calendar/static features known at t.
```

## Inference Routing Frame

Created by:

- `build_inference_frame_with_fallback()`

Required routing columns:

| Column | Meaning |
| --- | --- |
| `prediction_source` | `model` when the row has a complete feature vector, otherwise `cold_start_fallback` |
| `fallback_level` | fallback hierarchy level used for cold-start rows: `series`, `product`, `third_category`, or `global` |
| `fallback_target_lead_time_demand` | fallback lead-time prediction; null for model-routed rows |

Fallback semantics:

```text
fallback_target_lead_time_demand = mean_daily_observed_demand * horizon
```

Hierarchy semantics:

```text
series_id -> product_id -> third_category_id -> global
```

## Fold Spec

Created by:

- `build_walk_forward_folds()`

Fields:

| Field | Meaning |
| --- | --- |
| `fold_id` | integer fold index |
| `horizon` | forecast horizon used to enforce the temporal gap |
| `train_end_date` | latest allowed training origin date |
| `validation_start_date` | first validation origin date |
| `validation_end_date` | last validation origin date |

`FoldSpec` is a frozen Pydantic contract. It validates non-negative fold ids,
positive horizons, chronological validation windows, and the required temporal
gap.

Required temporal relation:

```text
train_end_date = validation_start_date - horizon
```

## Prediction Frame

Created by:

- `_build_baseline_predictions()`
- `_build_boosting_predictions()`

Required columns:

| Column | Meaning |
| --- | --- |
| `date` | forecast origin |
| `series_id` | series key |
| `target_lead_time_demand` | original target |
| `y_true` | canonical actual value used for metrics/costs |
| `y_pred` | point forecast |
| `model_name` | model label |
| `backend_name` | implementation/backend label |
| `fold_id` | validation fold |
| `order_quantity` | inventory decision |
| `overstock_units` | excess units |
| `stockout_units` | unmet units |
| `overstock_cost` | overstock cost |
| `stockout_cost` | stockout cost |
| `total_cost` | total operating cost |

Optional columns:

| Pattern | Meaning |
| --- | --- |
| `q_<level>` | quantile forecast generated by `quantile_column_name()` |
| `stockout_hours` | stockout context |
| `stockout_regime` | stockout regime label |
| `velocity_regime` | velocity regime label |
| `promo_regime` | promotional regime label |
| `seasonal_regime` | seasonal regime label |

Expected properties:

- `y_true == target_lead_time_demand`
- `y_pred >= 0`
- `order_quantity >= 0` when `clip_negative_orders = true`
- quantile columns are monotonic by level within each row

## Business Output Frame

Intended for:

- daily batch export to the replenishment manager
- future `reorder_recommendations.csv` artifact

This is a business-facing contract derived from model predictions,
inventory decisions, and inference-routing metadata. It is not yet the
canonical persisted artifact of the current pipeline, but it defines the
target schema for business-oriented runs.

Required columns:

| Column | Meaning |
| --- | --- |
| `decision_date` | business decision date; usually the forecast origin date used to place tomorrow's order |
| `series_id` | SKU decision key |
| `store_id` | store identifier |
| `product_id` | product identifier |
| `predicted_lead_time_demand` | lead-time demand estimate used as the central recommendation signal |
| `order_quantity` | recommended reorder quantity |
| `prediction_source` | `model` when the recommendation came from a full model prediction, otherwise `cold_start_fallback` |
| `fallback_level` | fallback hierarchy used when `prediction_source = cold_start_fallback`; null otherwise |
| `risk_flag` | operational review flag such as `cold_start`, `high_uncertainty`, `drift_watch`, or `extreme_order_quantity` |
| `notes` | optional free-text or enumerated explanation for the replenishment manager |

Optional columns:

| Column | Meaning |
| --- | --- |
| `q_<level>` | optional quantile forecasts used to communicate uncertainty |
| `stockout_hours` | latest stockout context at the forecast origin |
| `stockout_regime` | stockout regime label when available |
| `velocity_regime` | velocity regime label when available |
| `promo_regime` | promotional regime label when available |
| `seasonal_regime` | seasonal regime label when available |
| `data_strategy` | `Observed` or latent-demand strategy label |
| `model_name` | champion model used to generate the recommendation |
| `backend_name` | implementation/backend label |

Expected properties:

- one row per `series_id` and `decision_date`
- `order_quantity >= 0`
- `predicted_lead_time_demand >= 0`
- `fallback_level` is null when `prediction_source = model`
- `fallback_level in {series, product, third_category, global}` when `prediction_source = cold_start_fallback`
- `risk_flag` may be null for standard recommendations

Semantics:

```text
predicted_lead_time_demand = y_pred when using model predictions
```

```text
predicted_lead_time_demand = fallback_target_lead_time_demand when using cold-start fallback
```

## Exceptions Frame

Intended for:

- future `exceptions.csv` artifact
- manual review queue for the replenishment manager

This frame contains only flagged recommendation rows that require additional
review before export to downstream procurement systems.

Required columns:

| Column | Meaning |
| --- | --- |
| `decision_date` | business decision date |
| `series_id` | SKU decision key |
| `store_id` | store identifier |
| `product_id` | product identifier |
| `risk_flag` | primary exception category |
| `order_quantity` | recommended reorder quantity under review |
| `prediction_source` | `model` or `cold_start_fallback` |
| `notes` | review context or explanation |

Optional columns:

| Column | Meaning |
| --- | --- |
| `fallback_level` | fallback hierarchy for cold-start rows |
| `predicted_lead_time_demand` | central demand estimate behind the recommendation |
| `q_<level>` | uncertainty context for manual review |
| `model_name` | model responsible for the recommendation |

Expected properties:

- every row must have a non-null `risk_flag`
- each row must correspond to a row in the business output frame
- exception rows should be a strict subset of the daily recommendation export

## Metrics Summary

Created by:

- `summarize_predictions()`

Expected columns:

| Column | Meaning |
| --- | --- |
| `model_name` | model label |
| `backend_name` | backend label |
| `observations` | number of prediction rows |
| `mae` | mean absolute error |
| `rmse` | root mean squared error |
| `pinball_q_*` | optional pinball losses |
| `coverage_*` | optional interval coverage |

MAE and RMSE are diagnostics, not the primary ranking criterion.

## Cost Summary

Created by:

- `summarize_costs()`

Expected columns:

| Column | Meaning |
| --- | --- |
| `model_name` | model label |
| `backend_name` | backend label |
| `observations` | number of prediction rows |
| `mean_order_quantity` | mean order quantity |
| `total_overstock_units` | aggregate overstock |
| `total_stockout_units` | aggregate stockout |
| `total_overstock_cost` | aggregate overstock cost |
| `total_stockout_cost` | aggregate stockout cost |
| `total_cost` | primary economic ranking metric |
| `mean_cost` | per-row mean cost |

Expected ordering:

- sorted ascending by `total_cost`

## Pareto Frontier

Created by:

- `summarize_pareto_frontier()`

Expected columns:

| Column | Meaning |
| --- | --- |
| `model_name` | model label |
| `backend_name` | backend label |
| `policy_name` | candidate inventory policy label |
| `order_scale` | multiplier applied to the selected `order_quantity` |
| `observations` | number of prediction rows evaluated |
| `mean_order_quantity` | mean candidate order quantity |
| `total_overstock_units` | aggregate candidate overstock |
| `total_stockout_units` | aggregate candidate stockout |
| `total_overstock_cost` | aggregate candidate overstock cost |
| `total_stockout_cost` | aggregate candidate stockout cost |
| `total_cost` | aggregate candidate economic cost |
| `mean_cost` | per-row mean candidate economic cost |
| `service_level` | share of rows with no stockout units |
| `fill_rate` | share of observed demand covered by candidate orders |
| `is_pareto_efficient` | true when no candidate dominates this row on cost, overstock, and stockout |

Optional grouping columns:

| Column | Meaning |
| --- | --- |
| `data_strategy` | observed vs imputed demand strategy |

Expected properties:

- candidate policies do not mutate `predictions.csv`
- Pareto efficiency is computed within each model/backend/strategy group
- objectives for Pareto dominance are `total_cost`, `total_overstock_units`, and `total_stockout_units`, all minimized
