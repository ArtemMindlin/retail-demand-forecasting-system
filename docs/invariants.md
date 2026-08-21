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

12. The conformal calibration split must embargo the horizon as well.

    Splitting a fold's training frame into sub-train and calibration follows the
    same rule as the fold split itself:

    ```text
    sub_train_end_date = calibration_start_date - horizon
    ```

    Without the gap the last `horizon - 1` sub-train rows carry targets covering
    days inside the calibration window, the conformity scores come out
    optimistically small, and the intervals undercover.

13. Forecast metrics and inventory economics are summarized over the same frame:
    every validation origin.

    Each origin carries its own single-period Newsvendor decision, attached fold by
    fold by the per-fold builders before the frames are concatenated, so
    `summarize_predictions()`, `summarize_costs()` and the sensitivity analysis all
    read one row per series, date, model and strategy. A multi-period Order-Up-To
    simulation used to narrow the cost frame to one decision per series and fold;
    it was removed, and with it the `sim_*` columns. Anything that reintroduces a
    decision cadence coarser than the origin grid must re-split the two domains,
    because MAE, Winkler and coverage must not be computed on a decision subset.

14. The official `eval` split is wired as an external holdout, and its rows must never
    reach a training target.

    Its temporal semantics are verified: the raw split covers 2024-06-26 to 2024-07-02, the
    7 days immediately following the train split's last date (2024-06-25), 7 days per series.
    A clean forward holdout, so `run_experiment` loads it unconditionally
    (`forecasting/pipeline.py`) and the OPS plane streams it origin by origin.

    What this invariant protects is the frame construction, not the loading.
    `_build_supervised_frames` concatenates `panel + holdout_panel` before calling
    `build_supervised_frame` so the holdout rows get correct lag history, then keeps ONLY
    holdout-date rows. Both halves are load-bearing: without the concatenation the holdout's
    lags are null, and without the date filter the last `horizon - 1` train rows get targets
    covering holdout days via `shift(-horizon)` -- the same embargo as invariants 11 and 12,
    applied across the split boundary.

    A split other than `train` must INHERIT train's series universe, never recompute one.
    `prepare_daily_panel` takes `restrict_to_series` for this, and `load_prepared_panel` loads
    the train panel first to supply it. Both universe filters are wrong on a short forward
    split, in different ways:

    - `min_history_days` (70) counts days inside the split, so a 7-day split loses every
      series and the panel prepares EMPTY. This is what actually happened: holdout evaluation
      silently did not run in any experiment to date, and every number in the thesis comes
      from the walk-forward.
    - `top_n_series` ranks by demand summed over the split, so the top 50 of 7 days is a
      different set than the top 50 of train's 90 days -- measured on the real data: ZERO
      overlap, i.e. a populated holdout of 50 series the model never trained on. This is the
      more dangerous half, because it fails without looking empty.

    `max_rows` is excluded for the same reason: it truncates by raw row order, so on an
    inherited split it would cut days off some series and not others.

    A holdout that vanishes must fail the run, never be skipped. Two silent drops are now
    loud: `load_prepared_panel` raises when a non-train split prepares to zero rows, and
    `_evaluate_on_holdout` raises on an EMPTY frame while still accepting `None` (which
    legitimately means no holdout was requested -- the synthetic-panel tests and the
    fair-cost backtest run that way). This mirrors invariant 32's rule for the web layer: an
    empty state must never read as a result. An empty CACHED non-train panel is rebuilt rather
    than served, since it is an artifact of the old filters; the cache key is deliberately NOT
    bumped, as that would orphan the pre-built OPS split whose `.built` sentinel stops
    `make simulate` from regenerating it.

    Scope of what the holdout can support: with horizon 7 and a 7-day split, exactly ONE
    origin per series survives target construction -- 50 rows, all dated 2024-06-26. It is a
    single-date, cross-sectional check with no temporal replication, so it corroborates the
    walk-forward; it cannot rank models on its own and its metrics carry no time-series
    uncertainty. The OPS plane is unaffected by all of this: `scripts/build_ops_sim_split.py`
    writes its own pre-built split under the cache filename, returned from cache before any
    of this logic runs.

    Covered by `tests/test_holdout_split.py`. The `dataset.use_eval_as_holdout` flag this
    invariant once named no longer exists anywhere in `src/`, and `validate_settings()` does
    not mention `eval`.

## Models

15. Models receive already-built feature matrices and targets.

    They must not load datasets, build raw features, choose order quantities, or write reports.

16. Forecast outputs must be non-negative demand forecasts.

17. Quantile forecasts must be monotonically non-decreasing by quantile level.

18. Quantile column names must be generated with `quantile_column_name()`.

## Inventory

19. Inventory decisions live in `src/retail_forecasting/inventory/`.

20. The current inventory policy is single-period newsvendor.

21. `order_quantity` is derived from point forecasts or quantile forecasts plus the critical fractile.

22. Cost columns must be derived from `y_true` and `order_quantity`.

   Required cost columns:

   - `overstock_units`
   - `stockout_units`
   - `overstock_cost`
   - `stockout_cost`
   - `total_cost`

## Evaluation and Reporting

23. Evaluation summarizes predictions and costs.

    It must not retrain models, mutate forecasts, or change order quantities.

24. The primary ranking metric for the TFG is economic cost, not MAE.

25. MAE and RMSE are diagnostic metrics.

26. Pinball loss and coverage are probabilistic diagnostics when quantile forecasts exist.

## Generated Outputs

27. Reports, plots, cached datasets, notebook outputs, and PDFs are generated artifacts.

    Do not treat them as source of truth for pipeline behavior.

28. Durable design decisions belong in the main methodology and design documents.

## Web Layer

See `docs/web_layer.md` for the full description.

29. The web layer renders and orchestrates. It must not duplicate forecasting,
    inventory, or evaluation logic.

    A new number on screen is computed in `api/services/` or in the pipeline,
    never in a view or a template.

30. `api/services/` must not import Django.

    Django only enters through `api/store.py`, `api/views/` and the settings
    module, which keeps the services unit-testable without a web client.

31. The dashboard has no database.

    System state lives in `reports/`; the only web-owned state is the operator
    session, carried in a signed cookie. Adding a relational store is a design
    change, not an implementation detail.

32. Missing artifacts render an explicit empty state that names the command
    which produces them.

    Never fabricate placeholder values. A run without `drift_report.json` says
    so; it does not render an empty panel that reads as "no drift".

33. User-supplied run names are validated against the artifact catalogue before
    touching the filesystem.

    `ArtifactStore.resolve_run` and the EDA figure lookup exist to make path
    traversal impossible by construction.

34. Charts are rendered as SVG on the server.

    No client-side charting library. The scenario parameters live in the query
    string so every dashboard state is a shareable, reloadable URL.

## Rolling-Origin Backtest (OPS plane)

35. The inference origin for a decision taken on date `d` is the row dated `d`.

    A row dated `d` targets demand over `[d, d + h - 1]` (invariant 6) and only
    carries lagged features, so slicing history at `< d` scores a forecast for
    `[d - 1, d + h - 2]` against the actuals of `[d, d + h - 1]`. This is not
    leakage, but it costs a one-day-stale forecast and, under a trend, deflates
    both accuracy and conformal coverage. `simulation/operations.py` slices at
    `<= d`.

36. Aggregates over the backtest use a non-overlapping origin grid.

    Scoring runs daily and the horizon is `h` days, so consecutive origins share
    `h - 1` days of demand. Summing every daily origin counts the same demand up
    to `h` times. Both `_summarize_cadences` and the dashboard keep one origin
    every `h` days.

37. Origins whose actuals have not fully landed are never aggregated.

    Their `y_true` is a partial-window sum, so any cost computed against it
    understates the shortage half. They are scored and stored, then excluded from
    every summary, chart and KPI.

38. Retrain cadences are compared paired, with an interval, and the comparison
    declares itself underpowered when there are too few origins.

    A raw cost difference over a handful of weeks ranks nothing. See
    `_compare_cadences`.

39. The OPS plane is not an inventory-state simulation.

    Each origin is an independent single-period Newsvendor decision: no carried
    stock, no order pipeline, no lead time, and truth is censored observed demand
    with no imputation. Its costs rank policies; they are not a replenishment
    ledger. Naming it a "simulation" in user-facing text is what this invariant
    exists to prevent.

40. `tune_imputation_lgbm()` (the `tune_imputation` run mode, `forecasting/imputation_tuning.py`)
    must load only `split="train"`, never `split="eval"` -- the same rule as invariant 14,
    applied to the supervised imputer's own hyperparameter search. It also persists only the
    winning hyperparameters, never fitted model weights: `LatentDemandImputer` always re-fits
    on the current panel's own clean days, so tuning cannot change the leakage or
    feature-space properties of imputation, only which 13 hyperparameters the fit uses.

    Every `LatentDemandImputer` built in `forecasting/pipeline.py` must be given that
    params file via `model_path`. The constructor falls back to the untuned defaults when
    it is omitted, so a call site that forgets it runs a different `supervised` model from
    the rest of the pipeline -- and the imputation study would be comparing a strategy the
    champion run never uses. Enforced by `test_pipeline_imputers_all_read_the_tuned_params_file`.

41. The imputation search must select and validate on disjoint SERIES, and must not persist a
    winner that loses either to the untuned defaults or to the incumbent already on disk.

    On a single draw the MAE spread between trials (12-40% measured) is an order of magnitude
    larger than the differences between hyperparameter sets (~1%), so a single-draw search
    selects the lucky trial and its own objective value is not evidence of a gain -- a run that
    reported -1.4% in-sample measured -0.32% (CI95 crossing zero) on fresh draws. Hence the
    objective averages over `N_SELECTION_HOLDOUTS` draws, `improvement_pct` is computed only on
    the `N_VALIDATION_HOLDOUTS` draws, and `imputation_lgbm_params.json` is written only when
    the `improvement_ci95` over those draws lies entirely below zero. That interval is a
    Student-t CI for the mean of the paired per-draw differences, not a bootstrap: measured, the
    two agree to 0.0004 and give identical verdicts on every gate case, and where they diverge
    (a single outlying draw) the t interval is the WIDER one, so the gate errs strict. The OPS
    plane keeps a real bootstrap because it resamples whole origins as clusters.
    `best_mae_selection` in the metadata is in-sample for the search and must never be quoted
    as the improvement.

    Disjoint censoring SEEDS are not sufficient. Every draw censors
    `SYNTHETIC_CENSORING_EVAL_FRACTION` (0.30) of the same clean-row pool, so after n draws the
    pool is `1 - 0.7**n` covered and the two sets converge on the same rows: measured 82.4%
    overlap at 5/10 draws and 99.7% (7 fresh rows of 2185) at 15/25. That is a property of the
    fraction and the draw count, NOT of the panel size, so adding series cannot reduce it.

    The split is TEMPORAL, and it is applied as a MASK rather than a partition.
    `_split_temporal_windows()` holds back the last third of the calendar; `_build_holdouts`
    passes that mask to `synthetic_censor_holdout` as `censorable_mask`, so both windows keep
    the FULL panel and only the eligible evaluation rows differ. This matters more than the
    choice of axis, because the teacher's training size is not a detail of the measurement --
    it is the variable that decides the answer. Measured against the untuned defaults on the
    same imputer, the gain shrinks monotonically as the teacher grows and then REVERSES:

    | teacher clean rows | scored on | gain vs defaults |
    | --- | --- | --- |
    | 467 (15 held-out series) | series-disjoint | -5.33% |
    | 595 (last 30d of 50 series) | time-disjoint | -4.34% |
    | 1930 (all 50 series) | time-disjoint | -1.72% |
    | 14243 (500 series) | time-disjoint | **+12.44%** |
    | 18310 (500 series + eval) | eval week | **+13.20%** |

    The last two agree across different scoring periods, so this is scale, not calendar. An
    earlier design partitioned the panel by series, which shrank the teacher to a third of
    deployment size and produced the -5.33% headline; at the 500-series scale where
    `configs/experiment/imputation_compare.yaml` actually runs, those same params are 12% WORSE than not
    tuning. Hence `teacher_fit_rows` in the metadata: a tuned params file is only valid near the
    teacher size it was tuned at, and the file cannot express that on its own. Tune at the scale
    you deploy.

    The mask restricts BOTH populations: the rows that may be censored AND the real stockouts
    the severity ratios are drawn from. Restricting only the first was a live defect. Severity is
    what makes a synthetic stockout resemble a real one, and the two windows differ sharply on
    it -- the early window's real stockouts hide 43.4% of a day against the late window's 30.5%,
    with 76% of all stockout rows sitting early -- so an unrestricted pool was effectively the
    early window's, and late-window rows were censored ~32% harder than they ever are. That
    erased, on the severity axis, the regime shift a temporal holdout exists to measure, and it
    favoured whichever candidate handled harsh censoring in both windows alike. A window with no
    real stockouts of its own must fail loudly rather than fall back to the panel: the fallback
    reintroduces the defect silently, which is the failure mode invariants 14 and 32 both target.
    Pinned by `test_censoring_draws_its_severity_from_the_same_window_it_censors`, which uses
    disjoint per-window severities so a leak is unambiguous rather than statistical.

    `n_estimators` is capped below LightGBM's practical range for COMPUTE, and the cap must be
    justified by measurement rather than assumed harmless. At 500 series an 8000-tree fit costs
    ~34s, putting a 300x15 search near 43h. Measured on identical draws, holding every other
    hyperparameter fixed:

    | `n_estimators` | reconstruction MAE |
    | --- | --- |
    | 1500 | 0.46173 |
    | 3000 | 0.45354 |
    | 6000 | 0.44989 |
    | 8000 | 0.44985 |

    Returns die out entirely past 6000 (a further 0.00004), so the optimum sits there and not
    beyond. A 3000 cap costs 0.0037 of MAE -- smaller than the 0.0059 width of the validation
    interval that decides both gates, so it cannot change either verdict. That is the argument
    a cap needs: not "it seemed enough", but "what it forfeits is below what the measurement can
    resolve". Check the winner is not pinned AT the cap regardless, the same boundary check that
    caught the earlier `cat_smooth` and `num_leaves` floors.

    Two blind spots this design accepts, neither of them fixable by a different fraction. First,
    the two windows share the panel, so unlike the series partition it cannot detect
    hyperparameters overfitted to the PANEL, only to the window -- mitigated by the imputer
    refitting per panel, which makes panel-overfitting a weaker threat for hyperparameters than
    for weights. Second, the cut is deterministic (the last third, no seed), so exactly ONE
    window is ever tested: if the late calendar is peculiar, the verdict inherits that and
    nothing reveals it. Validating over several windows instead of one would fix the second and
    costs almost nothing, since validation runs three times per search rather than 300.

    `n_series`, `selection_window_end`, `validation_window_start` and the two `n_*_eval_rows`
    counts record the split; the two windows share every series by construction. The boundary
    dates are read off the masks, never derived by arithmetic around a cut date -- `cut_date - 1
    day` names a date absent from the panel wherever the calendar has a gap at the boundary, and
    `prepare_daily_panel` never reindexes to a complete calendar.

    Beating the defaults is necessary but NOT sufficient to write the file, because that
    comparison cannot see the incumbent: a run measuring -4.65% overwrote a -5.33% winner,
    having scored BETTER on the selection draws it was optimizing (0.2821 vs 0.2836) and worse
    on the held-out series (0.2872 vs 0.2851) -- textbook selection-set overfitting, and the
    defaults-only gate had no way to notice. So the incumbent is read
    BEFORE the search (by decision time it may be overwritten or deleted), both configs are
    scored on the same validation draws so the difference is paired, and `beats_incumbent`
    compares them.

    The two gates decide on DIFFERENT statistics, on purpose. The defaults gate is the
    interval, because it guards a CLAIM: an improvement whose interval includes zero is a null
    result and must be reported as one. A point comparison once passed a winner at -1.14% that
    then measured -0.45% with a straddling interval on fresh draws. The incumbent gate is the
    MEAN, because it guards no claim -- it only picks which of two files sits on disk. When the
    draws cannot separate the two, "keep whichever arrived first" is not a more defensible
    tiebreak than "keep the better mean"; it only looks more cautious, and it already decided a
    real case on seniority alone (-5.33% against -4.65% measured CI95 [-0.0011, +0.0051], a
    tie). This reduces to a single comparison: the t interval is symmetric about the mean, so a
    decisive verdict either way already agrees with it, and the mean does real work only in the
    straddling case. `incumbent_ci95` is still recorded, since without it the metadata cannot
    tell a decisive replacement from a coin flip won by a hair.

    Note the asymmetry in the failure branches -- failing the DEFAULTS gate deletes any params
    file (the pipeline should fall back to defaults rather than trust a winner this run
    rejected), while failing only the INCUMBENT gate leaves it untouched, since it is still a
    validated winner. `incumbent_mae_validation`/`incumbent_ci95`/`beats_incumbent` record the
    comparison, all null on a first run.

42. The synthetic-censoring holdout cannot rank imputation RECONCILIATION rules -- only
    hyperparameters and teacher models.

    `synthetic_censor_holdout()` builds its ground truth as `observed = truth * (1 - r)`.
    Any rule that inverts that relation scores near-zero error by construction, not by
    merit: `observed / (1 - r)` measures MAE 0.02 against 0.74 for the rule it replaces.
    Hyperparameter comparisons stay valid because every candidate shares one reconciliation
    rule, so the assumption cancels. Reconciliation changes must be argued on modelling
    grounds and must never quote a reconstruction-MAE gain from this holdout -- including in
    `imputation_quality.csv`, which measures agreement with the generator rather than
    imputation quality in the field.
