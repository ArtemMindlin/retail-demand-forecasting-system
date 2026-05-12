# Business Case

This document defines the business-facing operating context of the project.

The goal is to describe the system not as a generic forecasting benchmark, but
as a replenishment decision-support workflow for fresh retail.

## Core Decision

Primary business question:

```text
How many units of each SKU should be reordered tomorrow?
```

The system answers this question for each `series_id = store_id x product_id`
using demand forecasts, uncertainty estimates, and an inventory decision rule.

## Primary User

Primary user:

- replenishment manager

The replenishment manager is responsible for reviewing daily reorder
recommendations, focusing on flagged exceptions, and approving or overriding
orders before they are sent to the downstream procurement process.

## Business Objective

The project is optimized for decision quality, not forecasting quality alone.

The business objective is:

```text
minimize expected inventory cost while maintaining acceptable service
```

This objective is operationalized through a single-period newsvendor decision
rule that converts predicted lead-time demand into `order_quantity`.

## KPI Hierarchy

Primary KPI:

1. `total inventory cost`

Secondary KPIs:

1. `stockout cost`
2. `overstock / waste cost`
3. `service level` or `fill rate`
4. `forecast calibration`

Diagnostic KPIs:

1. `MAE`
2. `RMSE`
3. `pinball loss`
4. prediction interval coverage

Interpretation:

- a model with lower MAE is not automatically better if it produces worse
  reorder decisions;
- the champion policy is selected primarily on economic performance, with
  service-level degradation controlled as a constraint.

## Business Baseline

Current baseline policy:

```text
reorder using last week's same-day demand
```

In the repo, this is represented by the `SeasonalNaive` baseline.

Purpose of the baseline:

- provide a realistic heuristic comparator;
- represent a simple replenishment rule that a business could plausibly use
  without a learned model;
- test whether the ML system adds operational value beyond a repeat-last-week
  policy.

## Daily Operating Workflow

The intended batch workflow is:

1. Daily sales and context data arrive at the end of day `t`.
2. The data layer validates and prepares the canonical panel.
3. The forecasting layer predicts lead-time demand for each SKU.
4. The inventory layer converts predictions into `order_quantity`.
5. The system writes reorder recommendations and exception flags.
6. The replenishment manager reviews flagged SKUs.
7. Approved recommendations are exported to the downstream ERP or procurement
   process.
8. Realized future demand is logged for monitoring, backtesting, and
   retraining.

This is a batch decision system. The project does not require online serving to
be business-oriented.

### Daily Run Cadence

Recommended operating cadence:

1. end-of-day data for date `t` is closed and validated;
2. the batch scoring run executes before the next procurement cutoff;
3. reorder recommendations for date `t+1` are generated per SKU;
4. the replenishment manager reviews only flagged exceptions;
5. approved recommendations are exported to the downstream order system.

Operational assumption:

- the system is designed around one business cycle per day;
- the core output is a recommendation artifact, not an interactive prediction
  API.

## Required Business Output

The core business artifact should be a daily reorder recommendation table.

Recommended required columns:

- `decision_date`
- `series_id`
- `store_id`
- `product_id`
- `predicted_lead_time_demand`
- `q_0_1`
- `q_0_5`
- `q_0_9`
- `order_quantity`
- `prediction_source`
- `fallback_level`
- `risk_flag`
- `notes`

Semantics:

- `order_quantity` is the actual business recommendation;
- `prediction_source` distinguishes model output from fallback output;
- `risk_flag` identifies SKUs that need human review;
- `notes` can carry operational explanations such as `cold_start`,
  `high_uncertainty`, or `drift_watch`.

## Exception Handling

The system should not treat all SKUs as equally safe to automate.

Minimum exception types:

- `cold_start`
- `high_uncertainty`
- `drift_watch`
- `data_quality_warning`
- `extreme_order_quantity`

Operational expectation:

- non-flagged SKUs can be bulk-approved;
- flagged SKUs should be reviewed by the replenishment manager;
- severe issues should trigger fallback behavior or block export.

## Monitoring Policy

Monitoring is part of the business workflow only if it triggers actions.

### Performance Monitoring

Track:

- rolling `MAE`
- rolling `total_cost`
- `stockout_cost`
- `overstock_cost`

Purpose:

- detect whether recommendation quality is degrading over time.

### Uncertainty Monitoring

Track:

- interval coverage
- average interval width
- under-coverage frequency

Purpose:

- detect when uncertainty estimates stop being trustworthy.

### Operational Monitoring

Track:

- fallback usage rate
- cold-start rate
- missing data rate
- failed run count
- flagged SKU count

Purpose:

- ensure the daily workflow remains usable for the business.

### Business Monitoring

Track:

- service level
- fill rate
- manager override rate

Purpose:

- confirm the system improves the replenishment process rather than just model
  metrics.

## Monitoring Actions

Suggested default actions:

- cost degradation above threshold -> review challenger retraining
- drift detection -> trigger retraining candidate
- severe coverage degradation -> recalibrate or retrain
- fallback spike -> inspect data freshness and history sufficiency
- blocking data-quality failure -> stop recommendation export

## Retraining Policy

Default business retraining policy:

1. scheduled retraining on a fixed cadence, for example weekly;
2. additional retraining when drift is detected;
3. no automatic promotion unless the challenger outperforms the current
   champion on business KPIs.

The operational state of the current champion is persisted in
`champion_registry.json`. This registry stores the currently approved model
identity and allows future runs to evaluate challengers against the actually
deployed champion rather than only against static config defaults.

This keeps retraining tied to operational outcomes rather than arbitrary model
refreshes.

### Default Retraining Schedule

Recommended default:

- scheduled retraining once per week;
- additional retraining when monitoring or drift policy indicates degradation;
- no direct promotion of a retrained model without champion/challenger review.

This split allows the project to support both a stable weekly cadence and a
faster corrective path when the current model starts to degrade.

## Champion / Challenger Standard

Champion:

- current model used to generate reorder recommendations

Challenger:

- newly trained candidate evaluated against the champion before promotion
- persistent `champion_registry.json` tracking the active approved model

Promotion rule:

- challenger must improve `total inventory cost`;
- challenger must not degrade service level beyond an agreed threshold;
- challenger must pass validation and monitoring checks.

This project should treat model replacement as a governed business decision,
not simply as “newer model wins”.

### Acceptance Rule

Default promotion rule:

1. train a challenger on the latest approved data window;
2. evaluate it with walk-forward validation under the current KPI hierarchy;
3. promote it only if:
   - `total inventory cost` improves;
   - service level degradation stays within the agreed tolerance;
   - no blocking data-quality or monitoring issue appears.

If any of these conditions fail, the current champion remains in use.

## Data Quality Gates

Business-oriented runs should validate data before scoring.

Blocking checks:

- required columns missing
- duplicated `series_id + date`
- broken temporal ordering
- null key identifiers

Warning checks:

- unusual fallback rate
- suspicious demand spikes
- insufficient recent history for many SKUs

Informational checks:

- summary of missingness
- summary of filtered series

## Operational Ownership

Suggested ownership split:

- replenishment manager:
  consumes recommendations, reviews exceptions, overrides rare edge cases;
- ML/Ops owner:
  monitors model quality, drift, failed runs, and challenger promotion;
- data owner:
  investigates blocking input quality failures and broken upstream feeds.

This clarifies that model governance and order approval are related but
separate responsibilities.

## Audit Trail

Every run should be traceable.

The repo already supports this through:

- `backtest_metadata.json`
- `config_hash`
- `git_commit`
- fold metadata
- tuning metadata
- drift metadata

Business interpretation:

- every recommendation run should be explainable;
- every model version should be auditable;
- rollback should be feasible if a challenger underperforms.

## Minimal Target State For This Repo

To be considered a business-oriented MLOps project, this repo should expose:

1. a documented replenishment manager use case;
2. a daily reorder recommendation artifact;
3. a business baseline policy;
4. a KPI hierarchy centered on cost and service;
5. monitoring rules with explicit actions;
6. a champion/challenger promotion rule;
7. exception flags for manual review;
8. a reproducible metadata trail for every run.
