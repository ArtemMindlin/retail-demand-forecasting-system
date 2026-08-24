# GPSampler vs TPESampler for the imputation search

Closed decision, recorded so it is not reopened. Optuna's `GPSampler` is what
`tune_imputation` uses, and the reason is speed, not quality: the two samplers cannot be
told apart on this objective.

## What was measured

300 trials per search, 3 seeds (42, 1234, 7) per sampler, over the 500-series panel — six
searches, about 27 hours of compute. Every search scored its winner against the untuned
defaults on the same held-out draws, so `improvement %` below is comparable across rows.

| Sampler | Seed | MAE (validation) | Improvement | Minutes | `max_depth` | `colsample_bytree` |
| --- | --- | --- | --- | --- | --- | --- |
| gp  | 7    | 0.3379 | −6.32 % | 116 | 10 | 0.639 |
| gp  | 42   | 0.3311 | −7.62 % |  86 | 10 | 0.601 |
| gp  | 1234 | 0.3365 | −6.29 % | 127 | 11 | 0.555 |
| tpe | 7    | 0.3366 | −6.69 % | 198 | 10 | 0.587 |
| tpe | 42   | 0.3421 | −4.56 % | 121 | 11 | 0.555 |
| tpe | 1234 | 0.3322 | −7.48 % | 177 | 11 | 0.660 |

|  | Mean improvement | Std. dev. | Mean minutes |
| --- | --- | --- | --- |
| gp  | −6.74 % | 0.76 pp | 110 |
| tpe | −6.24 % | 1.51 pp | 165 |

## The verdict

**The samplers are indistinguishable and GP is 33 % faster.** The 0.50 pp gap between their
means sits inside the seed-to-seed spread of either one, so it measures the seed and not the
sampler. What does separate them is wall-clock: 110 minutes against 165.

Do not re-run this. Six searches at this scale is the evidence that would be gathered again,
and it already says the choice does not matter.

## What this evidence does NOT support

Every winner above picked `max_depth` 10–11, which is why `_INT_BOUNDS` in
`forecasting/imputation_tuning.py` raised that floor. But **these searches ran under the
starved-teacher regime that commit `b7900cb` replaced** — the series partition left the
teacher 467 clean rows where the pipeline gives it 14 243. Bounds narrowed on this evidence
were narrowed against a different objective, which is how `colsample_bytree` ended up with a
0.5 floor that a later 300-trial winner landed exactly on. The floors that remain are kept
because they are not binding, not because these numbers still justify them.

## Reproducing

The harness is `scripts/sampler_ab.py`. It resumes: every study lives in its own Optuna
storage and every finished search appends to a results file, so a relaunch picks up at the
last completed trial and an interruption costs the trial in flight.

```bash
uv run python scripts/sampler_ab.py configs/tune_imputation/default.yaml 300 42,1234,7
```
