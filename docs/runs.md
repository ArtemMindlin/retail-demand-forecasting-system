# Run provenance

Which run backs which result in the thesis, and with what code.

The run store accumulates every execution and is gitignored, so without this file there
is no way to tell a citable run from a scratch one. That gap caused a real defect: the
conformal-calibration table in chapter 6 was published from a run predating the
target-leakage fix of 16 May 2026, and nothing in the repository recorded it.

**Rule: a number that reaches the thesis names its run here, with the commit that
produced it.** A run whose commit is not an ancestor of the current `HEAD` is not
citable — re-run it before citing.

## Current results (audit of 13 Aug 2026, commit `df15b1d`)

| Result in chapter 6 | Run | Panel | Notes |
| --- | --- | --- | --- |
| `tab:metrics_predictive` | `fresh_retailnet_large_20260811_125735` | 500 series, 45 000 rows | Generated; see "Generated tables" below. **Predates the one-arm-per-run split: that run holds both arms in one `metrics_summary.csv`. Re-running it now needs two runs, one per arm.** |
| `tab:conformal_metrics`, `tab:fold_coverage` | `fresh_retailnet_v2_20260811_123002` | 50 series, 4 500 rows | 1 050 evaluation origins |
| `tab:embargo_control` (embargo off) | `fresh_retailnet_v2_20260811_123106` | 50 series, 4 500 rows | Control: only the calibration embargo differs |
| `tab:scale_coverage`, simulated costs, `fig:mae_vs_servicio` | `fresh_retailnet_large_20260811_125735` | 500 series, 45 000 rows | 10 500 evaluation origins |
| `tab:metrics_cost` (fair cost) | `fresh_retailnet_large_20260811_184959` | 30 sampled from 500 | Generated. Ranking depends on the source panel — see below |
| `tab:imputation_reconstruction` | `imputation_compare_20260811_174500` | 500 series | **NOT REPRODUCIBLE and not citable.** The `compare_imputation` mode that wrote it is deleted, and invariant 42 forbids the ranking it shows anyway. Either restore the mode or drop the table from chapter 6 |

Every numbered table in chapter 6 is listed above. A table that is not in this list has
no declared provenance, which is the state that produced the August 13 findings: both
generated tables had drifted to June runs while this file declared newer ones, and
neither was listed here at all.

## Generated tables

`tab:metrics_predictive` and `tab:metrics_cost` are written by the exporter, never edited
by hand. Regenerate both together:

```bash
uv run python -m retail_forecasting.evaluation.latex_exporter \
    --metrics-run <corrida Observed> <corrida Latent> \
    --fair-cost-run fresh_retailnet_large_20260811_184959
```

Those are run names, resolved through MLflow — the same names this table cites. A directory
still works.

The runs are arguments, not defaults: the exporter used to pin them in `__main__`, which
is how it kept republishing June runs. It also could not execute at all between `14ad8b4`
(Jinja2 dropped) and `c274522`, so a stale table could not have been refreshed even
deliberately. If a table and this file disagree, run the command above before believing
either.

## Traps this file exists to prevent

**The processed-panel cache used to ignore `top_n_series`.** Before commit `94b1224`
the cache key was the split name alone, so all five configs shared one
`data/processed/train.parquet` and whichever panel was on disk won. A run of
`experiment.yaml` (50 series) silently reused a 500-series panel. The key now
fingerprints `top_n_series`, `min_history_days`, `max_rows` and `horizon`, so a config
change rebuilds the panel — the first run after switching configs is slower by design.

**The fair-cost backtest is sensitive to the panel it samples from.** It draws 30 series;
from the 500-series panel `Latent_supervised` is cheapest (1 774.82 vs 1 843.41 u.m.),
from the 50-series top-rotation subset the order inverts (927.80 vs 844.01). The
artifact records `source_panel_series`, so always read the ranking together with it.

**Subsets answer different questions.** The 50-series base subset isolates the economics
on high-rotation SKUs; the 500-series run tests conformal stability at scale; the 30-series
fair-cost backtest isolates the imputation signal under a common ground truth. Do not
average metrics across them or quote a figure from one beside a figure from another
without saying so.

**OPS-plane numbers are not walk-forward numbers.** The `simulate_ops` run answers a
different question (does the champion decay in production, does retraining pay off) with
a different costing model: one independent single-period Newsvendor decision per origin,
no carried stock, no lead time, and truth censored on stockout days. Read them under
three rules. Its costs come from one origin every `horizon` days — scoring is daily and
consecutive origins overlap, so any figure summed over all daily origins is a
`horizon`-fold overcount and must not be quoted. Origins whose actuals had not fully
landed are excluded from every aggregate. And the retrain-cadence comparison is only
citable when `cadence_comparison.csv` says `conclusive`; with `underpowered` true the
window has too few independent origins to rank the policies, and the honest sentence is
that it cannot tell them apart.

## Reproducing

An experiment scores ONE demand strategy, so each arm is its own run. `imputation_strategy`
in the YAML picks it, `--imputation-strategy` overrides it per invocation.

```bash
make run                 # base subset, Latent arm → configs/experiment/default.yaml
uv run python -m retail_forecasting.run --config configs/experiment/default.yaml \
    --imputation-strategy none   # the Observed arm of the same panel
make simulate            # OPS plane        → configs/simulate_ops/default.yaml (builds the split if missing)
uv run python -m retail_forecasting.run --config configs/experiment/large.yaml
uv run python -m retail_forecasting.run --config configs/experiment/default.yaml --run-mode fair_cost_backtest
```

The figures read finished runs:

```bash
uv run python scripts/plot_coverage_folds.py \
    --base fresh_retailnet_v2_20260811_123002 \
    --scale fresh_retailnet_large_20260811_125735
uv run python scripts/plot_mae_vs_service.py --run fresh_retailnet_large_20260811_125735
```

Those are run names, resolved through MLflow, not paths. A directory still works.

## Where a finished run lives

The pipeline opens an MLflow run and writes into `mlruns/<id>/artifacts/` directly, so
there is one copy and no `reports/` directory for a new run. Each artifact directory
carries an `mlflow_run.json` naming the run, because the tracking store is a gitignored
sqlite database with no backup and, with a database store, `mlruns/` holds artifacts and
nothing that says which run they belong to.

The runs this table cites were written before that and moved into the store, so they read
the same way as any other: by the name in the table, not by a path.

The store's two halves fail independently. `mlruns/` is a directory tree a backup catches;
`mlflow.db` is a gitignored sqlite index nothing backs up. Losing it leaves the artifacts
under UUID directories, which is why each carries its own `mlflow_run.json`. Rebuild with

```bash
uv run python scripts/rebuild_mlflow_index.py
```

What returns is identity and files, not the params and metrics — those lived only in the
index. Enough to cite a run and read it; not enough to compare runs by MAE.
