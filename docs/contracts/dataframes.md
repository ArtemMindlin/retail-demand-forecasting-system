# Dataframe Contracts

This document defines the dataframe schemas used by the v1 pipeline.

The goal is to make agent changes safer: code may evolve, but these contracts should remain stable unless the experiment design changes deliberately.

## Raw FreshRetailNet Split

Created by:

- remote parquet read in `load_raw_split()`
- optional local raw cache under `data/raw/fresh_retailnet/`

Known raw columns used by v1:

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
| `observed_demand` | numeric | observed demand used by v1 |
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

- `build_supervised_frame()`

Required carry-through columns:

| Column | Meaning |
| --- | --- |
| `date` | forecast origin |
| `series_id` | series key |
| `observed_demand` | observed demand at origin date |
| `stockout_hours` | stockout signal at origin date |
| `stockout_regime` | regime label if added before feature engineering |

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

## Fold Spec

Created by:

- `build_walk_forward_folds()`

Fields:

| Field | Meaning |
| --- | --- |
| `fold_id` | integer fold index |
| `train_end_date` | latest allowed training origin date |
| `validation_start_date` | first validation origin date |
| `validation_end_date` | last validation origin date |

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

Expected properties:

- `y_true == target_lead_time_demand`
- `y_pred >= 0`
- `order_quantity >= 0` when `clip_negative_orders = true`
- quantile columns are monotonic by level within each row

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
