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
- Added `tests/test_dataframe_contracts.py`.
- Added `tests/test_raw_column_boundaries.py`.
- Added `tests/test_config_contract.py`.
- Added `tests/test_generated_artifact_boundaries.py`.
- Documented the fast harness command in `AGENTS.md` and `README.md`.

## Current Checks

Run the full suite:

```bash
uv run pytest
```

High-signal harness checks:

```bash
uv run pytest tests/test_architecture_imports.py tests/test_temporal_leakage_contract.py tests/test_quantile_contract.py tests/test_dataframe_contracts.py tests/test_raw_column_boundaries.py tests/test_config_contract.py tests/test_generated_artifact_boundaries.py
```

## Next Recommended Checks

1. Keep `docs/contracts/dataframes.md` and the contract tests in sync when pipeline schemas change.

2. Keep the harness command documented in `AGENTS.md`, `README.md`, and this plan.

3. Add a docs reference check if broken links or stale file references become a recurring problem.

## Open Decisions

- Whether to introduce latent-demand recovery after the v1 baseline is stable.
- Whether to replace the single-period newsvendor policy with base-stock or reorder-point policies.
- Whether the official `eval` split can be used as an external holdout after its temporal semantics are verified.
