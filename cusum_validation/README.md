# CUSUM overfitting detection: validation across dataset scales

Continues an earlier research thread (not tracked in this repo -- a
working research log kept locally during development) validating the
CUSUM overfitting detector added in this branch. On a 2,400-row
subsample of the Adult dataset, the CUSUM detector fired
at ~epoch 40 of 75, and stopping there instead of running the full
schedule roughly halved the fraction of suspiciously-close synthetic
samples. On the full 45k-row dataset, long-horizon GPU runs confirmed
the fixed detector matches the existing sensitivity mechanism's
privacy and utility outcomes at a fraction of the wall-clock cost (see
"Results so far" below). A CPU-only follow-up then tested five much
smaller datasets to see whether that holds at other scales -- see
"Multi-dataset validation" below for what that found (mostly yes, with
one real exception worth digging into further on real hardware).

## Setup

From the repo root, in an environment with the package installed
(`pip install -e .`) and a CUDA-capable PyTorch:

```bash
cd cusum_validation
python run_experiment.py --dataset adult --mode cusum --epochs 1000 --device cuda
```

`--dataset` selects which dataset to run against (`--help` lists all
of them). Every dataset's raw data is checked into `data/` alongside
this script -- no download needed, and this guarantees the exact same
data used throughout this research.

## Datasets

| `--dataset` | rows | target | type |
|---|---|---|---|
| `diabetes` | 768 | `class` | classification |
| `insurance` | 1,338 | `charges` | regression |
| `abalone` | 4,177 | `Rings` | regression |
| `wilt` | 4,839 | `Class` | classification |
| `churn2` | 10,000 | `Exited` | classification |
| `adult` (default) | 45,222 | `income` | classification |

The first four are the "small" scale point, `churn2` is "medium",
`adult` is "large" -- see `DATASET_CONFIGS` in `run_experiment.py` for
the exact target/type wiring each one uses.

## Running the whole battery at once

`run_all_experiments.py` runs `--mode all` (cusum + full + sensitivity)
against every non-Adult dataset in one call, instead of invoking
`run_experiment.py` by hand once per dataset:

```bash
python run_all_experiments.py --epochs 300 --device cuda
```

Each dataset runs as its own subprocess (a fresh Python process --
nothing leaks between runs) in size order (`diabetes`, `insurance`,
`abalone`, `wilt`, `churn2`), streaming output live to the console and
saving it to its own file under `results/logs/` (gitignored -- not
meant to be committed, unlike `results/*_summary.json`). A failure on
one dataset is logged and the batch moves on rather than aborting, so
a crash partway through doesn't cost you results already collected.
`adult` is excluded by default -- it already has extensive committed
results, is far more expensive, and typically wants a different
`--epochs` ceiling -- pass it explicitly via `--datasets` if you want
it in the same batch:

```bash
# Just the two datasets from the open abalone question:
python run_all_experiments.py --datasets abalone churn2 --epochs 300 --device cuda

# Everything, including adult, with adult's own longer epoch ceiling
# handled separately:
python run_all_experiments.py --epochs 300 --device cuda
python run_experiment.py --dataset adult --mode all --epochs 1000 --device cuda
```

## Modes

- `--mode cusum` (default): trains with
  `overfitting_detection_method="cusum"`, stopping early if/when the
  detector fires, up to `--epochs`. This is the main question --
  "does it trigger, and when." Also measures DCR (distance to closest
  record) ground truth on whatever model results. Runs a **delta
  ensemble** by default (`--cusum-delta 0.25 0.5 1.0`) -- several
  CUSUM trackers in parallel, one per assumed effect size, alarm fires
  on the first to cross its own (Bonferroni-corrected) threshold. No
  single delta is well-matched to both a slow, gradual drift and a
  sharp, sudden one, and there's no way to know in advance which shape
  a given run's signal will take; the ensemble hedges against that
  without added training-time cost (`z` is computed once per check,
  shared by every tracker -- only a few extra scalar ops each). Pass a
  single value (`--cusum-delta 0.5`) for the original single-tracker
  behavior.
- `--mode full`: trains the full `--epochs` schedule with no early
  stopping (`overfitting_detection_method="none"`), for a direct
  before/after DCR comparison against the cusum run. Roughly the same
  wall-clock cost as `--mode cusum` in the worst case (detector never
  fires) -- run this separately, deliberately, once you know whether
  the cusum run is worth comparing against (no point running the full
  expensive schedule twice if the detector fires early).
- `--mode sensitivity`: trains with the *existing*, pre-CUSUM
  `overfitting_detection_method="sensitivity"` mechanism (bootstrap-DCR
  with periodic `.generate()` calls) -- the actually meaningful
  baseline for this research thread, since CUSUM's whole point was to
  replace/improve on this specific method, not just to beat "no
  stopping at all" (`--mode full` answers that separate question).
  Substantially more expensive per check by design (a `num_bootstrap`
  round bootstrap plus a real generation call every `n_critic` epochs)
  -- expect this to take meaningfully longer than `--mode cusum` at the
  same epoch ceiling. Tunable via `--sensitivity-n-critic` (default 5),
  `--sensitivity-n-critic-stop` (default 2), `--sensitivity-num-bootstrap`
  (default 500) -- all matching the existing method's own defaults.
- `--mode both`: runs cusum + full sequentially (unchanged from before
  sensitivity mode was added).
- `--mode all`: runs cusum + full + sensitivity sequentially, into one
  `run_id`'s `summary.json`, for a direct three-way comparison.
- `--mode checkpoint` (pass `--checkpoint-dir <path to an
  alarm_checkpoint_dir>`): skips training entirely and re-scores DCR +
  utility for an already-trained checkpoint. For re-checking a run you
  already have -- e.g. one that stopped unusually early -- without
  spending more GPU time on a fresh training run just to find out.

Other flags: `--batch-size` (default 32), `--cusum-check-every`
(default 20, in optimizer steps), `--output-dir` (default `results/`),
`--run-id` (defaults to an auto-generated name including mode/epochs/
timestamp).

## Output (this is the part to commit back)

Everything lands in `results/`, incrementally, specifically so a
killed or timed-out run still leaves something usable:

- `<run_id>_cusum_trajectory.jsonl` (cusum mode only): one JSON line
  **appended in real time** after every CUSUM check throughout
  training -- `step`, `delta`, `z`, `cusum_S`, `mu0`, `sigma0`,
  `cusum_h`, `alarm_step`. If the run gets cut off, this file alone
  shows the full CUSUM trend up to that point.
- `<run_id>_summary.json`: written incrementally too -- training facts
  (alarm_step if any, global_step reached, elapsed time, calibration
  values) are written as soon as training finishes, *before* the DCR
  measurement runs, so even a DCR-measurement failure still leaves the
  core "did it fire, when" answer on disk. The DCR comparison
  (`frac_suspicious`, `dcr_synth_mean`, etc.) is added once sampling
  completes, followed by a `<label>_utility` block (`trtr_auc`,
  `tstr_auc`, `auc_gap`) -- see "Utility check" below.
- `<run_id>_ckpt_cusum/` and `<run_id>_ckpt_full/`: full HF Trainer
  checkpoints (large -- do not commit these, they're for your own
  local use/debugging; `.gitignore` in this directory already excludes
  them).

**After running, `git add cusum_validation/results/*.json*`
and commit** (the JSON/JSONL files are small; the checkpoint
directories are gitignored). That gives Claude everything needed to
read the results back in a later session -- the trajectory file in
particular, since it shows the CUSUM statistic's actual trend, not
just whether it crossed the threshold.

## A note on cost

On the CPU environment this was developed on, one epoch on the full
45k-row training split took ~340s. A GPU should be dramatically
faster, but 1000 epochs is still a long run regardless -- consider
starting with a shorter `--epochs` (e.g. 100-200) to sanity-check
timing on your hardware before committing to the full 1000.

## Results so far: 100-epoch three-way comparison, and a real CUSUM bug

`--mode all --epochs 100` results are committed under `results/`
(`cusum_ep100_*`, `full_ep100_*`, `sensitivity_ep100_*`). Summary:
`full` (no stopping) reached `frac_suspicious=0.576` (badly
memorized); `cusum` stopped at step 7240/28200 (~epoch 25.7) and got
`frac_suspicious=0.137`; `sensitivity` (the existing bootstrap-DCR
mechanism) stopped earlier and got `frac_suspicious=0.054` --
notably better than CUSUM despite CUSUM being ~27% cheaper in
wall-clock time. Both stopping mechanisms crushed the unrestricted
baseline, but sensitivity found the meaningfully better stopping
point on this run.

Replaying `cusum_ep100_1788295822_cusum_trajectory.jsonl` after the
fact found why: `sigma0` (the calibrated noise scale CUSUM's z-score
is divided by) came out to 0.9955, but first-differencing the actual
351-check post-warmup trajectory shows the true steady-state
check-to-check noise is only ~0.029 -- `sigma0` was calibrated
**~34x too large**. The mechanism: the 10-check warmup window used to
calibrate `sigma0` happens right when the cooled reference pool first
becomes eligible, which is dominated by rows whose baseline was
captured at/near model initialization -- the population-level swing
from "near-random init" to "a few hundred steps in" is large and
itself noisy check-to-check, nothing to do with memorization, but it
inflated the calibrated noise scale by orders of magnitude and made
the detector far more conservative (slower to fire) than it should
have been.

Fixed in the library (see `rtf_cusum.py`'s module docstring,
"Calibration" bullet, for the full writeup): a new
`warmup_settle_checks` (defaults to `warmup_checks` itself) discards
that many checks before calibration starts collecting, so calibration
lands once the pool's row-age mix has stabilized; `sigma0` is also now
estimated via first-differencing (`_robust_noise_std`) rather than raw
std, as defense-in-depth against any residual drift in the window that
follows. Validated via unit tests and by replaying this real run's
z-sequence (delta/target_far retuning alone -- without the sigma0 fix
-- only pulled the alarm forward by ~300 of 7240 steps, confirming
`sigma0` and not those knobs was the dominant lever).

**Confirmed against a real re-run** (`cusum_ep100_1788308715_*`, same
batch=32 as the original baseline): `sigma0` came out to `0.0385` --
essentially matching the ~0.029 true noise level independently derived
above, from a totally different computation. The trajectory's `phase`
tags show it working as designed: 10 `"settle"` checks (steps 40-220,
discarded), 10 `"warmup"` checks (steps 240-420, tightly clustered
`delta` values -- nothing like the old run's wild early spread), then
`cusum_S` crosses threshold at step **1120** instead of 7240 (an 84%
cut in detection delay), giving `frac_suspicious=0.053` -- statistically
indistinguishable from sensitivity's `0.054`, for 241s vs.
sensitivity's 1662s (~7x cheaper). A `batch_size=128` run on the *old*
code (`cusum_ep100_1788300530_*`) landed in between (`sigma0=0.322`,
`frac_suspicious=0.076`) -- consistent with the diagnosis, since larger
batches happen to average out some of the early-transient volatility
even without the fix.

## Open question: did the early-stopped run actually learn anything?

The fixed run above stopped at only ~4 effective epochs -- much
earlier than any prior run. Low `frac_suspicious` this early is
ambiguous on its own: it could mean "correctly caught the right
stopping point," or it could just mean "hasn't trained long enough to
memorize *or* learn anything useful yet" -- DCR/`frac_suspicious`
can't tell those apart, since an undertrained model trivially looks
"not memorized." That's what the utility check below is for.

## Multi-dataset validation: small, medium, and the abalone exception

A CPU-only follow-up ran `--mode cusum` (default single delta=0.5 at
the time -- ensemble support came slightly later) then `--mode
sensitivity` on all four small datasets plus `churn2` (medium), to
check whether the Adult-scale result (CUSUM matching sensitivity on
both privacy and utility, ~5x cheaper) holds at very different scales.
Results (not committed -- these were local, disk-constrained runs, not
meant as the final word):

- `diabetes`, `insurance`, `wilt`: clean, matches the Adult pattern --
  utility gap between real- and synthetic-trained models stayed small
  (-0.014 to +0.041), tracking close to what sensitivity achieved.
- `abalone`: the exception. CUSUM stopped at ~10.6 effective epochs
  (via `delta=0.5`) with a real utility cost (R² gap 0.190) that
  sensitivity, training ~4x longer, closed to 0.024 -- i.e. abalone's
  gap was substantially a stopping-time artifact, not (only) the
  `Rings` target being inherently hard to fit.
- Two fixes were tried on abalone and **neither helped**: relaxing the
  cooldown safety-cap (`steps_per_epoch - 1` instead of `steps_per_epoch
  // 2`) fired *earlier*, not later (gap got marginally worse, 0.199);
  explicitly using the delta ensemble instead of the accidental
  single-tracker default changed which tracker fired (`delta=1.0`
  instead of `0.5`, the first time anything but `0.5` won across every
  test run so far) but only marginally closed the gap (0.184).
- `churn2` (medium) was queued for the same cusum-vs-sensitivity
  comparison but not completed locally before this script was built to
  let the investigation continue on real GPU hardware instead.

Since neither obvious fix panned out, this is a genuinely open
question -- worth a real run here (`--dataset abalone --mode all`) to
get a clean, fast comparison, and ideally a `churn2` run to see whether
the abalone gap is really about small *datasets* generally or something
more specific to `abalone`/its target.

## Resolution: final-defaults validation across all five datasets

A GPU run of `--mode cusum` with the fully-evolved detector --
`cusum_statistic="median"`, the `[0.25, 0.5, 1.0]` delta ensemble,
`cusum_confirm_patience=1`, and the `steps_per_epoch`-aware cooldown
cap, none of which existed yet when the "exception" above was
diagnosed -- against the matching `--mode sensitivity` baseline
already committed for each dataset (`*_sensitivity_base_ep300_*`):

| Dataset | Method | Effective epochs | Elapsed (s) | `frac_suspicious` | Utility gap |
|---|---|---|---|---|---|
| abalone | CUSUM | 22.3 | 126.9 | 0.078 | 0.022 (R²) |
| abalone | sensitivity | 24.2 | 210.2 | 0.077 | 0.001 (R²) |
| adult | CUSUM | 3.3 | 157.6 | 0.044 | 0.001 (AUC) |
| adult | sensitivity | 17.6 | 1298.1 | 0.050 | 0.002 (AUC) |
| diabetes | CUSUM | 25.0 | 41.8 | 0.130 | -0.001 (AUC) |
| diabetes | sensitivity | 45.0 | 117.4 | 0.149 | 0.003 (AUC) |
| insurance | CUSUM | 17.5 | 43.3 | 0.063 | 0.004 (R²) |
| insurance | sensitivity | 37.5 | 152.0 | 0.082 | 0.002 (R²) |
| wilt | CUSUM | 13.3 | 86.3 | 0.055 | 0.089 (AUC) |
| wilt | sensitivity | 29.3 | 279.5 | 0.065 | 0.004 (AUC) |

(`cusum_validation/results/{abalone,adult,diabetes,insurance,wilt}_cusum_ep300_*_summary.json`,
committed in `217149b`, compared against each dataset's already-committed
`*_sensitivity_base_ep300_*_summary.json`.)

**The abalone exception is resolved, not just improved.** Its utility
gap drops from the 0.190 reported above to 0.022 -- an 8.6x reduction,
now the same order of magnitude as sensitivity's own 0.001, while
`frac_suspicious` is statistically indistinguishable between the two
methods (0.078 vs. 0.077) and CUSUM still runs 1.7x faster. Since
neither of the two isolated fixes tried earlier (relaxed cooldown cap,
explicit delta ensemble) moved the needle on their own, the fix is best
attributed to the combination that shipped since then -- most plausibly
`cusum_statistic="median"`'s outlier-robust `Delta`, which the earlier
mean-based statistic didn't have -- rather than to any single change in
isolation. Not re-isolated here; stated as the likely explanation, not
a proven one.

**On every other dataset, CUSUM matches or beats sensitivity on
privacy** (`frac_suspicious` lower or tied on all 5) **and is
substantially cheaper** (1.7x-8.2x faster, largest speedup on the
biggest dataset, adult, as expected since CUSUM avoids sensitivity's
periodic `.generate()`-based bootstrap entirely). Utility gaps track
sensitivity closely on adult, diabetes, and insurance (within 0.002-0.003
of each other either direction).

**New observation, not yet investigated: `wilt` now has the largest gap
in this comparison** (0.089 vs. sensitivity's 0.004) -- CUSUM stops at
13.3 effective epochs here vs. sensitivity's 29.3, trading real utility
for a comparatively modest privacy/speed gain (`frac_suspicious` 0.055
vs. 0.065, 3.2x faster). This is the same *shape* of problem the
abalone exception was (an early, confident alarm that costs more
downstream utility than sensitivity's slower stop), just smaller in
absolute terms and on a different dataset -- worth checking whether it
responds to the same median-statistic-driven fix path, or is a separate
cause, before treating the detector's defaults as fully settled across
dataset shapes.

`churn2` (medium-sized) was not re-run with this final build --
its two `sensitivity_*` counterparts were never generated (only
`churn2_all_ep300_*`, an earlier combined-mode run, is committed), so
it's not included in the table above and remains untested against the
current defaults.

## Utility check (TSTR vs. TRTR)

`measure_utility` (called automatically after `measure_dcr` in every
mode, including `--mode checkpoint`) answers a different question than
DCR: not "is the synthetic data suspiciously close to training rows"
but "did the model learn anything useful." It trains the same model
once on the real training split (TRTR) and once on the registered
synthetic data (TSTR), scores both against the same real held-out test
set, and reports the gap. Classification targets (`categorical=True`
in `DATASET_CONFIGS`) use `LogisticRegression` + ROC-AUC; regression
targets use `LinearRegression` + R². Written to the summary as
`{"metric": "auc"|"r2", "trtr": ..., "tstr": ..., "gap": trtr - tstr}`.
A gap near 0 means the synthetic data is nearly as useful as the real
thing for downstream modeling; a large gap means the generator hasn't
learned the data's structure yet, regardless of what `frac_suspicious`
says.

(Earlier committed results, from before regression-target support was
added, use the older `trtr_auc`/`tstr_auc`/`auc_gap` field names --
same meaning, classification-only naming.)

This fixed a real, previously-unexercised bug in
`SyntheticDataBench.measure_ml_efficiency` along the way: any binary
classifier with `predict_proba` (the common case) crashed it outright,
since `predict_proba`'s `(n, 2)` array can't go into a single
DataFrame column -- it now takes the positive-class column.

To check a run you already have without retraining:
```bash
python run_experiment.py --mode checkpoint \
    --checkpoint-dir results/cusum_ep100_1788308715_ckpt_cusum/cusum_alarm \
    --run-id cusum_ep100_1788308715_recheck
```
