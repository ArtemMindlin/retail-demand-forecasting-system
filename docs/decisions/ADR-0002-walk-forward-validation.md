# ADR-0002: Use Walk-Forward Temporal Validation

## Status

Accepted.

## Context

The project evaluates demand forecasts for inventory decisions. Random splits would mix future and past observations across train and validation, producing optimistic metrics and invalid operational conclusions.

The target aggregates future demand over a configured horizon, so the fold boundary must also avoid target overlap with the validation period.

## Decision

The main experiment uses walk-forward validation with an expanding training window.

For each fold:

```text
train_end_date = validation_start_date - horizon
```

This means the latest training target ends before validation begins.

## Consequences

- Evaluation better reflects how the system would be used over time.
- The number of usable folds depends on history length, fold size, and horizon.
- Model comparisons should be made over the same folds and forecast origins.

## Harness Rules

- Do not use random train/test splits for the main experiment.
- `build_walk_forward_folds()` owns fold boundary construction.
- `tests/test_temporal_leakage_contract.py` must fail if training targets overlap validation.
