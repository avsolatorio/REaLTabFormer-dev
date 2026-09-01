# CUSUM overfitting detection: long-horizon validation on full Adult

Continues an earlier research thread (not tracked in this repo -- a
working research log kept locally during development) validating the
CUSUM overfitting detector added in this branch. On a 2,400-row
subsample of the Adult dataset, the CUSUM detector fired
at ~epoch 40 of 75, and stopping there instead of running the full
schedule roughly halved the fraction of suspiciously-close synthetic
samples. On the full 45k-row dataset, a 25-epoch run (the most
practical on CPU) never fired, and the model already sat at the
natural non-memorized baseline rate. This experiment runs a much
longer horizon (default 1,000 epochs) on a GPU to see whether
memorization ever emerges on the full dataset, and whether the
detector still catches it with a useful lead time if it does.

## Setup

From the repo root, in an environment with the package installed
(`pip install -e .`) and a CUDA-capable PyTorch:

```bash
cd experiments/cusum_full_adult
python run_experiment.py --mode cusum --epochs 1000 --device cuda
```

The Adult dataset (`adult.data`, `adult.test`, standard UCI Census
Income files) is checked into `data/` alongside this script -- no
download needed, and this guarantees the exact same data used
throughout the rest of this research.

## Modes

- `--mode cusum` (default): trains with
  `overfitting_detection_method="cusum"`, stopping early if/when the
  detector fires, up to `--epochs`. This is the main question --
  "does it trigger, and when." Also measures DCR (distance to closest
  record) ground truth on whatever model results.
- `--mode full`: trains the full `--epochs` schedule with no early
  stopping (`overfitting_detection_method="none"`), for a direct
  before/after DCR comparison against the cusum run. Roughly the same
  wall-clock cost as `--mode cusum` in the worst case (detector never
  fires) -- run this separately, deliberately, once you know whether
  the cusum run is worth comparing against (no point running the full
  expensive schedule twice if the detector fires early).
- `--mode both`: runs both sequentially in one invocation.

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
  completes.
- `<run_id>_ckpt_cusum/` and `<run_id>_ckpt_full/`: full HF Trainer
  checkpoints (large -- do not commit these, they're for your own
  local use/debugging; `.gitignore` in this directory already excludes
  them).

**After running, `git add experiments/cusum_full_adult/results/*.json*`
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
