# ADR-0001: Use Observed Demand in v1

## Status

Accepted.

## Context

FreshRetailNet includes organic stockout signals, so observed sales can understate true latent demand. A full latent-demand reconstruction would require explicit censoring assumptions, validation strategy, and additional evaluation criteria.

The v1 project goal is to build a defensible end-to-end baseline that connects forecasting to inventory decisions without overloading the first implementation with a demand-recovery model.

## Decision

The v1 target is observed lead-time demand:

```text
target_lead_time_demand(t, h) = sum(observed_demand[t : t + h - 1])
```

Stockout information is retained as lagged context and for regime analysis. It is not used to reconstruct latent demand in v1.

## Consequences

- Metrics and inventory costs are interpretable as outcomes on observed demand, not true unmet demand.
- Results may understate demand during stockout-heavy periods.
- Any future latent-demand model must be introduced as a deliberate change to target semantics, docs, and tests.

## Harness Rules

- `target_lead_time_demand` is built only in `src/retail_forecasting/features/engineering.py`.
- Same-day `stockout_hours` must not enter `feature_columns`.
- Any change from observed demand to latent demand must update `docs/contracts/dataframes.md`, `docs/invariants.md`, and temporal leakage tests.
