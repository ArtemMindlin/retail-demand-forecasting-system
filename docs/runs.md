# Run provenance

Which run backs which result in the thesis, and with what code.

The run store accumulates every execution and is gitignored, so without this file there
is no way to tell a citable run from a scratch one. That gap caused a real defect: the
conformal-calibration table in chapter 6 was published from a run predating the
target-leakage fix of 16 May 2026, and nothing in the repository recorded it.

**Rule: a number that reaches the thesis names its run here, with the commit that
produced it.** A run whose commit is not an ancestor of the current `HEAD` is not
citable — re-run it before citing.

## SUPERSEDED — every run below predates the changes of 26 Aug 2026

Nothing in the table that follows is citable. Eight changes landed on 26 Aug 2026, each of
which alters what the models see or how they are scored, so every number produced before them
describes a configuration that no longer exists:

| Change | Why it invalidates |
| --- | --- |
| `top_n_series` 50 → 500 in `experiment/default`, `retrain`, `score_daily`, `fair_cost_backtest` | ten times the panel; the champion is no longer selected on the 50 highest-rotation series |
| `rolling_windows` [7, 28] → [7, 14] | different features: level and dispersion over two weeks instead of four |
| `lags` [1, 7, 14, 28] → [1, 7, 14] | different features AND a shorter cold start: 70 origin dates per series instead of 56 |
| static ids int64 → pandas `category` | CatBoost's ordered target statistics engage for the first time; LightGBM partitions 352 store levels instead of carving numeric ranges |
| fair-cost order policy: safety-stock scale from `std(true_demand)` → from the censored panel | `tab:metrics_cost` only. Every absolute cost in it moves; the scale is no longer an oracle (invariant 44) |
| fair-cost order policy: `signal + z·σ` → the signal itself, and `signal_mae` dropped from the table | `tab:metrics_cost` only, and it decides the answer. The old cushion made knowing the TRUE demand 2.2% dearer than the uncorrected signal; without it the truth costs zero and the supervised arm lands 92.7% below the baseline. Every figure in that table changes by an order of magnitude (invariants 42 and 44) |
| fair-cost sample: panel subset → censoring mask | `tab:metrics_cost` only. The supervised imputer's teacher goes from 684 rows to the full panel's 16 405; its reconstruction MAE improves 27%, and neither heuristic is affected (invariant 44) |
| fair-cost ranking: 1 censoring draw → 20, with a paired 95% interval | `tab:metrics_cost` only. Its cost columns become means over draws and gain a gap column; the old single-draw ranking carried no uncertainty at all |

Two more changes are pending and will invalidate again, so do not re-cite until both land:
the LightGBM hyperparameter search (`run_mode = tune_forecasting`, in flight) and the
CatBoost one after it. A champion trained on untuned defaults is not the champion the
thesis will defend.

The fair-cost figures have been rewritten for the protocol they now describe, with every one
wrapped in `\pendiente{}` — it prints in red and in brackets so a provisional number cannot
ship silently. There are 25 of them, and they were NOT confined to chapter 6: the same three
numbers were asserted as fact in all three abstracts, in the conclusions, in the KPI table and
in the SDG annex. All of them wait on one run of `make backtest-fair-cost`, which is blocked
only by the forecasting search holding the machine. The estimates inside the markers come from a 6-draw check of the committed code,
not from a registered run, and must be replaced rather than merely confirmed.

The rewrite also retired a paragraph rather than updating it. It tells a story about an "unexpected ordering" in which
`historical_mean` is the most expensive of the four signals, dearer than the uncorrected
baseline. Under the current policy it is the second cheapest and the ordering follows what
each rule can recognise about stockout severity. The old ordering was an artefact of the flat
cushion, and that paragraph is gone.

`tab:imputation_reconstruction` is gone from chapter 6, not pending: the `compare_imputation`
mode that wrote it is deleted and invariant 42 forbids the ranking it showed. Restoring the
mode to republish a table the invariants reject would have been the wrong fix.

## Previous results (audit of 13 Aug 2026, commit `df15b1d`)

| Result in chapter 6 | Run | Panel | Notes |
| --- | --- | --- | --- |
| `tab:metrics_predictive` | `fresh_retailnet_large_20260811_125735` | 500 series, 45 000 rows | Generated; see "Generated tables" below. **Predates the one-arm-per-run split: that run holds both arms in one `metrics_summary.csv`. Re-running it now needs two runs, one per arm.** |
| `tab:conformal_metrics`, `tab:fold_coverage` | `fresh_retailnet_v2_20260811_123002` | 50 series, 4 500 rows | 1 050 evaluation origins |
| `tab:embargo_control` (embargo off) | `fresh_retailnet_v2_20260811_123106` | 50 series, 4 500 rows | Control: only the calibration embargo differs |
| `tab:scale_coverage`, simulated costs, `fig:mae_vs_servicio` | `fresh_retailnet_large_20260811_125735` | 500 series, 45 000 rows | 10 500 evaluation origins |
| `tab:metrics_cost` (fair cost) | `fresh_retailnet_large_20260811_184959` | 30 sampled from 500 | Generated. Ranking depends on the source panel — see below |

Every numbered table chapter 6 still has is listed above. A table that is not in this list
has no declared provenance, which is the state that produced the August 13 findings: both
generated tables had drifted to June runs while this file declared newer ones, and neither
was listed here at all.

## Generated tables

`tab:metrics_predictive` and `tab:metrics_cost` are written by the exporter, never edited
by hand. Regenerate both together:

```bash
uv run python -m retail_forecasting.evaluation.latex_exporter \
    --metrics-run <corrida Observed> <corrida Latent> \
    --fair-cost-run <corrida coste justo>
```

Those are run names, resolved through MLflow — the same names this table cites. A directory
still works.

Every run in the commands on this page is a placeholder, and deliberately not a real name:
take the name from the table above, which is the only place that says which run backs which
figure. Concrete names in a usage example are the same trap one layer out --- three of them
were still sitting in the `--help` text of the exporter and the two figure scripts, naming
runs this page had already marked superseded, where nobody reading the command would see the
banner.

The runs are arguments, not defaults: the exporter used to pin them in `__main__`, which
is how it kept republishing June runs. It also could not execute at all between `14ad8b4`
(Jinja2 dropped) and `c274522`, so a stale table could not have been refreshed even
deliberately. If a table and this file disagree, run the command above before believing
either.

## Diagnostic figures that no run mode produces

Three of the findings chapter 1 now lists as contributions are quantified by DIAGNOSTICS of the
committed code, not by any run mode's artifact: the reconstruction MAE of a ratio-only rule with
no model, the teacher-size comparison at 684 vs 16 405 clean rows, and the cost of a perfect
signal under each safety-stock variant. They are real measurements of the code as committed, and
they are reproducible, but nothing writes them to a run directory — so by this file's own rule
they cannot be cited. Every one of them is wrapped in `\pendiente{}` in the memoria.

Two ways to settle it, and the choice is not obvious. Either the fair-cost backtest grows a
diagnostic block that scores the control signals (the truth, and the ratio-only rule) alongside
the four strategies and writes them to `fair_cost_draws.csv` — which makes them citable and, as
a regression test, would have caught all three defects invariant 44 records. Or the numbers come
out of the memoria and the findings stay qualitative. Leaving them as unsourced figures is the
one option this file exists to forbid.

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
uv run python -m retail_forecasting.run --config configs/experiment/default.yaml
uv run python -m retail_forecasting.run --config configs/experiment/default.yaml --run-mode fair_cost_backtest
```

The figures read finished runs:

```bash
uv run python scripts/plot_coverage_folds.py \
    --base <corrida del subset base> --scale <corrida a escala>
uv run python scripts/plot_mae_vs_service.py --run <corrida a escala>
```

Those are run names, resolved through MLflow, not paths. A directory still works.

## The EDA figures of chapter 3

All 15 figures and every statistic in chapter 3 come from one run,
`eda_fresh_retailnet_v2_20260827_171901`, at commit `a151564`. Verified: every PNG under
`memoria/figures/eda/` is byte-identical to that run's artifacts — no post-hoc edits.

Three runs preceded it on the same day, and comparing them is what confirmed each change did
only what it claimed. `66d6206` rebuilt the demand distribution figure and left the other
fourteen byte-identical. `8b7c74c` translated the labels found inside labelling calls. `a151564`
carries the rest: the titles passed as arguments, the weekday names that arrive in English from
`dt.day_name()`, the tier names interpolated into a title, and two legend entries built with
interpolated figures. Eight of the fifteen changed between the last two, and they are exactly
the figures whose text was not a literal in a `set_*` call.

The `eda/` module has changed since, and none of it moves a figure: `make_grid` relocated to
`utils/plotting.py`, the pipeline moved from `make_run_directory` to `open_run_directory`, the
prints became a logger, and one figure was deleted — `top_series_total_demand.png`, which the
memoria never included. The EDA reads the prepared panel and applies no imputation, so the
`censorship.py` changes do not reach it either.

Chapter 3 describes the WHOLE 50 000-series panel (`configs/eda/default.yaml` sets
`top_n_series: null`), while every forecasting result describes 500. The chapter now says so in
its first subsection; do not carry a rate from one to the other.

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
