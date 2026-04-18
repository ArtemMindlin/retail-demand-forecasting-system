# Harness Engineering Plan

Date: 2026-04-17

## Goal

Make the repo easier for agents and humans to change without breaking the experimental architecture.

The most important risk is not a runtime failure. The most important risk is producing valid-looking forecasting results with temporal leakage, inconsistent dataframe schemas, or blurred boundaries between forecasting, inventory, and evaluation.

## Completed

- Added `AGENTS.md` as the short repo map and agent entry point.
- Added `docs/invariants.md` for structural and methodological rules.
- Added `docs/contracts/dataframes.md` for dataframe schemas and expected properties.
- Restored `docs/experimental_design.md` as the methodological system of record.
- Renamed the function glossary file to `docs/function_glossary.md`.
- Added ADRs for observed demand, walk-forward validation, and newsvendor v1 policy.
- Added `tests/test_temporal_leakage_contract.py`.
- Added `tests/test_quantile_contract.py`.
- Added `tests/test_architecture_imports.py`.

## Current Checks

Run the full suite:

```bash
uv run pytest
```

High-signal harness checks:

```bash
uv run pytest tests/test_temporal_leakage_contract.py tests/test_quantile_contract.py tests/test_architecture_imports.py
```

## Next Recommended Checks

1. Add `tests/test_dataframe_contracts.py`.

   Validate prepared panel, supervised frame, prediction frame, metrics summary, and cost summary schemas against `docs/contracts/dataframes.md`.

2. Add `tests/test_raw_column_boundaries.py`.

   Ensure raw FreshRetailNet names such as `dt`, `sale_amount`, and `stock_hour6_22_cnt` do not appear outside the data layer.

3. Add `tests/test_config_contract.py`.

   Validate critical defaults and guardrails such as positive costs, quantiles in `(0, 1)`, and `use_eval_as_holdout = false`.

4. Document a single harness command in `README.md`.

   Keep it simple for now; do not add a heavier task runner unless the project grows.

## Open Decisions

- Whether to introduce latent-demand recovery after the v1 baseline is stable.
- Whether to replace the single-period newsvendor policy with base-stock or reorder-point policies.
- Whether the official `eval` split can be used as an external holdout after its temporal semantics are verified.
