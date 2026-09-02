"""
CUSUM overfitting-detection validation, across datasets of very
different scale -- meant to be run on a GPU, not the CPU-only
environment this feature was developed and validated on.

Background (continues an earlier research thread validating the CUSUM
detector added in this branch, not tracked in this repo): on a
2400-row subsample of Adult, the CUSUM detector fired at ~epoch 40 of
a 75-epoch schedule, and stopping there instead of training the full
schedule roughly halved the fraction of suspiciously-close synthetic
samples (0.445 vs 0.890 frac_suspicious). On the FULL 45k-row Adult
dataset (the "large" scale point -- see `--dataset`), long-horizon runs
confirmed the fixed detector (settle-skip + robust sigma0 calibration +
delta ensemble) matches the existing sensitivity mechanism's privacy
and utility outcomes at a fraction of the wall-clock cost. A CPU-only
follow-up investigation across five much smaller datasets ("small":
diabetes/insurance/abalone/wilt, 768-4839 rows; "medium": churn2,
10,000 rows) found the same pattern holds on most of them, but not
uniformly -- abalone specifically showed a real utility gap versus
sensitivity (CUSUM stopping meaningfully earlier), and neither of the
two obvious fixes tried (relaxing the cooldown safety-cap, or
explicitly using the delta ensemble instead of the single-tracker
default) closed it. This script exists to let that investigation
continue on real GPU hardware instead of a disk- and CPU-constrained
local sandbox.

Usage:
    python run_experiment.py --dataset adult    --mode cusum --epochs 1000 --device cuda
    python run_experiment.py --dataset abalone  --mode all   --epochs 300  --device cuda
    python run_experiment.py --dataset diabetes --mode both  --epochs 300  --device cuda

`--dataset` selects which dataset to run against (see `DATASET_CONFIGS`
for the full list and each one's target column/type). All datasets
smaller than Adult are bundled in `data/` and load offline; no
network access or separate download needed for any of them.

Results are written incrementally to `results/` (see --output-dir) as
the run progresses, specifically so a killed/timed-out/interrupted run
still leaves usable, committable data:
  - `<run_id>_cusum_trajectory.jsonl`: one line per CUSUM check (mode=cusum
    only), appended in real time as training proceeds. Each line's
    `phase` is one of `"settle"` (discarded, pre-calibration -- just
    `step`), `"warmup"` (a raw calibration-window sample -- `step`,
    `delta`), or `"post_calibration"` (a real detection check -- step,
    Delta, Z, cusum_S/cusum_h (primary tracker, back-compat) plus
    cusum_S_by_delta/cusum_h_by_delta (every tracker in the ensemble),
    mu0, sigma0, alarm_step, alarm_delta -- which tracker actually
    fired, if any). This is the file to look at if the run gets cut
    off before finishing; the CUSUM trend up to that point is fully
    visible even without a final summary.
  - `<run_id>_summary.json`: written once, at the end of the run --
    full config, alarm_step (if any), timing, and the DCR ground-truth
    comparison (frac_suspicious, dcr_synth mean, etc.).

After running, commit the `results/` directory's new files back to the
repo (they're plain JSON/JSONL, small, and are exactly what the
research log needs to pick this validation back up).
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from realtabformer.realtabformer import REaLTabFormer
from realtabformer.rtf_analyze import SyntheticDataBench

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RANDOM_SEED = 1029
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_adult() -> pd.DataFrame:
    train_raw = pd.read_csv(
        DATA_DIR / "adult.data",
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    test_raw = pd.read_csv(
        DATA_DIR / "adult.test",
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )
    full = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
    full["income"] = full["income"].str.rstrip(".").str.strip()
    full = full.dropna().reset_index(drop=True)
    full = full.drop(columns=["education", "native-country"])
    return full


def _load_arff_data_section(path: Path, columns: list) -> pd.DataFrame:
    """Minimal ARFF data-section reader -- avoids scipy.io.arff's byte-string
    handling for nominal attributes, which is more friction than it's worth
    for these small, simple files."""
    lines = path.read_text().splitlines()
    start = (
        next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    )
    rows = [line.split(",") for line in lines[start:] if line.strip()]
    return pd.DataFrame(rows, columns=columns)


def load_diabetes() -> pd.DataFrame:
    # Pima Indians Diabetes -- 768 rows, 8 numeric features, binary target.
    cols = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "class"]
    df = _load_arff_data_section(DATA_DIR / "diabetes.arff", cols)
    for c in cols[:-1]:
        df[c] = pd.to_numeric(df[c])
    return df


def load_insurance() -> pd.DataFrame:
    # 1,338 rows, mixed types, continuous regression target ("charges").
    return pd.read_csv(DATA_DIR / "insurance.csv")


def load_abalone() -> pd.DataFrame:
    # 4,177 rows, one categorical feature (Sex), continuous regression
    # target ("Rings", conventionally treated as continuous -- age proxy).
    cols = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Rings",
    ]
    df = _load_arff_data_section(DATA_DIR / "abalone.arff", cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c])
    return df


def load_wilt() -> pd.DataFrame:
    # 4,839 rows, 5 numeric features, binary target ("Class", "2" = the
    # minority "diseased tree" class). Bundled as a CSV (fetched once via
    # sklearn's fetch_openml) rather than fetched at runtime, matching
    # every other dataset here being self-contained/offline.
    df = pd.read_csv(DATA_DIR / "wilt.csv")
    df["Class"] = df["Class"].astype(str)
    return df


def load_churn2() -> pd.DataFrame:
    # 10,000 rows -- the "medium" scale point between the ~1-5k-row small
    # datasets and Adult's 45k. Binary target ("Exited").
    df = pd.read_csv(DATA_DIR / "churn2.csv")
    return df.drop(columns=["RowNumber", "CustomerId", "Surname"])


# Each entry: (loader, target_col, categorical, target_pos_val). Small
# datasets are numbered by row count so `--dataset` choices sort roughly
# small-to-large in --help output.
DATASET_CONFIGS = {
    "diabetes": (load_diabetes, "class", True, "tested_positive"),
    "insurance": (load_insurance, "charges", False, None),
    "abalone": (load_abalone, "Rings", False, None),
    "wilt": (load_wilt, "Class", True, "2"),
    "churn2": (load_churn2, "Exited", True, 1),
    "adult": (load_adult, "income", True, ">50K"),
}


def attach_trajectory_logger(trajectory_path: Path):
    """Monkey-patches CUSUMOverfittingMonitor.maybe_check to append one
    JSON line per check to `trajectory_path`, in addition to its normal
    behavior. Done here, in the experiment script, rather than in the
    library itself -- the library has no business knowing about
    experiment-specific output paths; this is the same non-invasive
    wrap-and-append pattern used for debugging tracing earlier in this
    feature's own development.
    """
    from realtabformer import rtf_cusum

    orig_maybe_check = rtf_cusum.CUSUMOverfittingMonitor.maybe_check

    def logged_maybe_check(self, step, model, get_rows):
        settle_before = self._settle_checks_remaining
        warmup_before = len(self._warmup_deltas)
        fired = orig_maybe_check(self, step, model, get_rows)
        if self.history and self.history[-1][0] == step:
            # `stat_delta` is the paired-improvement statistic (called
            # `Delta` in the module docstring/code) -- unrelated to the
            # `delta` hyperparameter(s) tracked in `s_by_delta` below,
            # confusing as the shared name is.
            _, stat_delta, z, s_by_delta = self.history[-1]
            record = dict(
                phase="post_calibration",
                step=step,
                delta=stat_delta,
                z=z,
                cusum_S=s_by_delta.get(self.delta),  # primary tracker, back-compat
                cusum_S_by_delta=s_by_delta,
                mu0=self.mu0,
                sigma0=self.sigma0,
                cusum_h=self.cusum_h,  # primary tracker, back-compat
                cusum_h_by_delta=dict(self.cusum_h_by_delta),
                alarm_step=self.alarm_step,
                alarm_delta=self.alarm_delta,
            )
            with open(trajectory_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        elif self._settle_checks_remaining != settle_before:
            # A settle check was consumed (discarded, no Delta kept) --
            # logged mainly so it's visible how many checks the settle
            # window actually cost in real steps.
            record = dict(phase="settle", step=step)
            with open(trajectory_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        elif len(self._warmup_deltas) != warmup_before:
            # A calibration-window sample was collected -- logged raw
            # (unlike post-calibration checks, mu0/sigma0 aren't set
            # yet) specifically so a real run can directly confirm
            # whether the settle window actually avoided the volatile
            # early-training transient this fix targets.
            record = dict(phase="warmup", step=step, delta=self._warmup_deltas[-1])
            with open(trajectory_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return fired

    rtf_cusum.CUSUMOverfittingMonitor.maybe_check = logged_maybe_check


def measure_dcr(
    model, bench, train_df, test_df, device, label, summary_path, gen_batch=None
):
    n_needed = len(train_df) + len(test_df) + 300
    samples = model.sample(
        n_samples=n_needed,
        device=device,
        **({"gen_batch": gen_batch} if gen_batch else {}),
    )
    samples = samples.dropna().reset_index(drop=True)
    print(f"[{label}] generated {len(samples)} valid samples", flush=True)
    result = dict(label=label, n_samples_generated=len(samples))
    if len(samples) >= len(train_df) + len(test_df):
        bench.register_synthetic_data(samples)
        dcr_synth = bench.get_dcr(is_test=False)
        dcr_test = bench.get_dcr(is_test=True)
        threshold = dcr_test.quantile(0.05)
        frac_suspicious = float((dcr_synth < threshold).mean())
        result.update(
            dcr_synth_mean=float(dcr_synth.mean()),
            dcr_synth_min=float(dcr_synth.min()),
            dcr_test_mean=float(dcr_test.mean()),
            frac_suspicious=frac_suspicious,
        )
        print(
            f"[{label}] DCR: synth_mean={result['dcr_synth_mean']:.4f} "
            f"frac_suspicious={frac_suspicious:.4f}",
            flush=True,
        )
    else:
        result["error"] = "not enough valid samples for DCR bench"
        print(f"[{label}] not enough valid samples for DCR bench", flush=True)

    # Write/merge into the summary file incrementally -- if this is the
    # first result written, create the file; if a summary already
    # exists (e.g. mode=both, cusum result written first), merge in.
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing[label] = result
    summary_path.write_text(json.dumps(existing, indent=2))
    return result


def measure_utility(bench, label, summary_path, categorical: bool = True):
    """TSTR-vs-TRTR utility check -- a different question than
    measure_dcr's frac_suspicious. DCR alone can't distinguish "the
    model stopped too early to have learned anything useful yet" from
    "the model learned the distribution well and just isn't
    memorizing" -- both look identical on a pure privacy metric. This
    trains the SAME model once on the real training data (TRTR) and
    once on the synthetic data (TSTR), scores both on the same real
    held-out test set, and compares performance: a small gap means the
    synthetic data is nearly as useful for downstream modeling as the
    real thing; a large gap means the generator hasn't actually
    learned the data's structure yet, regardless of what DCR says.
    Classification targets use ROC-AUC (LogisticRegression);
    regression targets use R^2 (LinearRegression) -- set `categorical`
    to match the dataset's own target type.

    Requires bench.synth_train_df to already be populated -- i.e.
    measure_dcr's register_synthetic_data call must have run (and
    succeeded) first.
    """
    if bench.synth_train_df is None:
        print(
            f"[{label}] skipping utility check -- no synthetic data registered",
            flush=True,
        )
        return None

    if categorical:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        score_fn = roc_auc_score
        metric = "auc"
    else:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        model = LinearRegression()
        score_fn = r2_score
        metric = "r2"

    preds = bench.get_ml_efficiency(model)
    trtr = float(score_fn(preds["actual"], preds["original_predictions"]))
    tstr = float(score_fn(preds["actual"], preds["synthetic_predictions"]))
    gap = trtr - tstr

    print(
        f"[{label}] utility ({metric}): TRTR={trtr:.4f} TSTR={tstr:.4f} "
        f"gap={gap:+.4f} (closer to 0 = synthetic data is as useful as real)",
        flush=True,
    )

    result = dict(metric=metric, trtr=trtr, tstr=tstr, gap=gap)
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing[f"{label}_utility"] = result
    summary_path.write_text(json.dumps(existing, indent=2))
    return result


def run_cusum(args, run_id, full_df, target_col, categorical, target_pos_val):
    bench = SyntheticDataBench(
        data=full_df,
        target_col=target_col,
        categorical=categorical,
        target_pos_val=target_pos_val,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    train_df = bench.train_df.reset_index(drop=True)
    test_df = bench.test_df

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = results_dir / f"{run_id}_cusum_trajectory.jsonl"
    summary_path = results_dir / f"{run_id}_summary.json"
    trajectory_path.write_text("")  # truncate/create fresh

    attach_trajectory_logger(trajectory_path)

    print(
        f"\n=== RUN: overfitting_detection_method='cusum', "
        f"epochs={args.epochs}, device={args.device} ===",
        flush=True,
    )
    model = REaLTabFormer(
        model_type="tabular",
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=RANDOM_SEED,
        checkpoints_dir=str(results_dir / f"{run_id}_ckpt_cusum"),
    )
    t0 = time.time()
    trainer = model.fit(
        train_df,
        device=args.device,
        overfitting_detection_method="cusum",
        cusum_check_every=args.cusum_check_every,
        cusum_delta=args.cusum_delta,
    )
    elapsed = time.time() - t0
    mon = model.cusum_monitor
    steps_per_epoch = max(
        1,
        len(trainer.train_dataset)
        // (
            trainer.args.per_device_train_batch_size
            * trainer.args.gradient_accumulation_steps
        ),
    )
    print(
        f"\nRUN done in {elapsed:.1f}s. alarm_step={mon.alarm_step} "
        f"global_step={trainer.state.global_step} "
        f"effective_epochs={trainer.state.global_step / steps_per_epoch:.2f} "
        f"alarm_checkpoint_dir={mon.alarm_checkpoint_dir}",
        flush=True,
    )

    # Write the training-run facts into the summary immediately, before
    # the (much cheaper, but still non-zero) DCR measurement -- so even
    # if sampling/DCR fails, the core "did it fire, when, how long did
    # training take" facts are already on disk.
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing["cusum_training"] = dict(
        epochs_ceiling=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        cusum_check_every=args.cusum_check_every,
        cusum_deltas=mon.deltas,
        alarm_step=mon.alarm_step,
        alarm_delta=mon.alarm_delta,
        global_step=trainer.state.global_step,
        steps_per_epoch=steps_per_epoch,
        effective_epochs=trainer.state.global_step / steps_per_epoch,
        elapsed_s=elapsed,
        mu0=mon.mu0,
        sigma0=mon.sigma0,
        cusum_h=mon.cusum_h,  # primary tracker, back-compat
        cusum_h_by_delta=mon.cusum_h_by_delta,
        cooldown_steps_final=mon.cooldown_steps,
        alarm_checkpoint_dir=mon.alarm_checkpoint_dir,
    )
    summary_path.write_text(json.dumps(existing, indent=2))

    measure_dcr(
        model,
        bench,
        train_df,
        test_df,
        args.device,
        "cusum_stop_at_alarm",
        summary_path,
        gen_batch=args.gen_batch,
    )
    measure_utility(bench, "cusum_stop_at_alarm", summary_path, categorical=categorical)
    print(f"\nWrote {trajectory_path} and {summary_path}", flush=True)


def run_full(args, run_id, full_df, target_col, categorical, target_pos_val):
    bench = SyntheticDataBench(
        data=full_df,
        target_col=target_col,
        categorical=categorical,
        target_pos_val=target_pos_val,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    train_df = bench.train_df.reset_index(drop=True)
    test_df = bench.test_df

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{run_id}_summary.json"

    print(
        f"\n=== RUN: overfitting_detection_method='none' (full schedule), "
        f"epochs={args.epochs}, device={args.device} ===",
        flush=True,
    )
    model = REaLTabFormer(
        model_type="tabular",
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=RANDOM_SEED,
        checkpoints_dir=str(results_dir / f"{run_id}_ckpt_full"),
    )
    t0 = time.time()
    trainer = model.fit(
        train_df, device=args.device, overfitting_detection_method="none"
    )
    elapsed = time.time() - t0
    print(
        f"\nRUN done in {elapsed:.1f}s. global_step={trainer.state.global_step}",
        flush=True,
    )

    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing["full_training"] = dict(
        epochs_ceiling=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        global_step=trainer.state.global_step,
        elapsed_s=elapsed,
    )
    summary_path.write_text(json.dumps(existing, indent=2))

    measure_dcr(
        model,
        bench,
        train_df,
        test_df,
        args.device,
        "full_schedule",
        summary_path,
        gen_batch=args.gen_batch,
    )
    measure_utility(bench, "full_schedule", summary_path, categorical=categorical)
    print(f"\nWrote {summary_path}", flush=True)


def run_sensitivity(args, run_id, full_df, target_col, categorical, target_pos_val):
    """The EXISTING bootstrap-DCR overfitting_detection_method="sensitivity"
    (the default, pre-CUSUM mechanism) -- the actually meaningful
    baseline for this whole research thread, since CUSUM's point was to
    replace/improve on this, not just to beat "no stopping at all"
    (that's what --mode full is for). Substantially more expensive per
    check than CUSUM by design: a `--cusum-check-every`-style periodic
    check here means a `num_bootstrap`-round bootstrap plus a real
    `.generate()` call every `n_critic` epochs, which is exactly the
    cost CUSUM exists to avoid -- expect this mode to take meaningfully
    longer than --mode cusum at the same epoch ceiling.
    """
    bench = SyntheticDataBench(
        data=full_df,
        target_col=target_col,
        categorical=categorical,
        target_pos_val=target_pos_val,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    train_df = bench.train_df.reset_index(drop=True)
    test_df = bench.test_df

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{run_id}_summary.json"

    print(
        f"\n=== RUN: overfitting_detection_method='sensitivity' (existing "
        f"bootstrap-DCR mechanism), epochs={args.epochs}, "
        f"n_critic={args.sensitivity_n_critic}, "
        f"num_bootstrap={args.sensitivity_num_bootstrap}, "
        f"device={args.device} ===",
        flush=True,
    )
    model = REaLTabFormer(
        model_type="tabular",
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=RANDOM_SEED,
        checkpoints_dir=str(results_dir / f"{run_id}_ckpt_sensitivity"),
    )
    t0 = time.time()
    trainer = model.fit(
        train_df,
        device=args.device,
        target_col=target_col,
        n_critic=args.sensitivity_n_critic,
        n_critic_stop=args.sensitivity_n_critic_stop,
        num_bootstrap=args.sensitivity_num_bootstrap,
        gen_kwargs={"gen_batch": args.gen_batch} if args.gen_batch else None,
    )
    elapsed = time.time() - t0
    global_step = trainer.state.global_step
    # Same derivation run_cusum uses (real Trainer settings, not a
    # guessed batch/accumulation combo) -- so the two are comparable.
    steps_per_epoch = max(
        1,
        len(train_df)
        // (
            trainer.args.per_device_train_batch_size
            * trainer.args.gradient_accumulation_steps
        ),
    )
    stopped_epoch = global_step / steps_per_epoch
    print(
        f"\nRUN done in {elapsed:.1f}s. global_step={global_step} "
        f"(~epoch {stopped_epoch:.1f} of {args.epochs} ceiling)",
        flush=True,
    )

    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing["sensitivity_training"] = dict(
        epochs_ceiling=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        n_critic=args.sensitivity_n_critic,
        n_critic_stop=args.sensitivity_n_critic_stop,
        num_bootstrap=args.sensitivity_num_bootstrap,
        elapsed_s=elapsed,
        global_step=global_step,
        steps_per_epoch=steps_per_epoch,
        stopped_epoch=stopped_epoch,
    )
    summary_path.write_text(json.dumps(existing, indent=2))

    measure_dcr(
        model,
        bench,
        train_df,
        test_df,
        args.device,
        "sensitivity_stop",
        summary_path,
        gen_batch=args.gen_batch,
    )
    measure_utility(bench, "sensitivity_stop", summary_path, categorical=categorical)
    print(f"\nWrote {summary_path}", flush=True)


def run_from_checkpoint(args, run_id, full_df, target_col, categorical, target_pos_val):
    """Re-measure DCR + utility for an ALREADY-TRAINED checkpoint --
    no retraining, just reload and score. Rebuilds the same
    SyntheticDataBench split (same RANDOM_SEED, same full_df) any of
    the run_* functions would have used, so results are directly
    comparable to a fresh run's. Useful specifically for checking
    whether a run you already have (e.g. one that stopped very early)
    actually learned something, without spending more GPU time on a
    fresh training run just to find out.
    """
    bench = SyntheticDataBench(
        data=full_df,
        target_col=target_col,
        categorical=categorical,
        target_pos_val=target_pos_val,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    train_df = bench.train_df.reset_index(drop=True)
    test_df = bench.test_df

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{run_id}_summary.json"

    print(f"\n=== RUN: re-scoring checkpoint at {args.checkpoint_dir} ===", flush=True)
    model = REaLTabFormer.load_from_dir(args.checkpoint_dir)

    measure_dcr(
        model,
        bench,
        train_df,
        test_df,
        args.device,
        "checkpoint_recheck",
        summary_path,
        gen_batch=args.gen_batch,
    )
    measure_utility(bench, "checkpoint_recheck", summary_path, categorical=categorical)
    print(f"\nWrote {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        default="adult",
        help="Which dataset to run against. Small: diabetes (768 rows, "
        "classification), insurance (1338, regression), abalone (4177, "
        "regression), wilt (4839, classification). Medium: churn2 (10000, "
        "classification). Large: adult (45k, classification, the original "
        "validation target and default).",
    )
    parser.add_argument(
        "--mode",
        choices=["cusum", "full", "sensitivity", "both", "all", "checkpoint"],
        default="cusum",
        help="'both' runs cusum + full (unchanged from before sensitivity "
        "mode was added). 'all' runs cusum + full + sensitivity "
        "sequentially, all into the same run_id's summary.json for a "
        "direct 3-way comparison. 'checkpoint' skips training entirely "
        "and re-scores DCR + utility for an already-trained checkpoint "
        "(pass --checkpoint-dir) -- e.g. to check whether a run that "
        "stopped very early actually learned something, without "
        "spending more GPU time retraining just to find out.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Only used with --mode checkpoint: path to an alarm_checkpoint_dir "
        "from a previous run's summary.json (must contain rtf_config.json "
        "and rtf_model.pt -- REaLTabFormer.load_from_dir's format).",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--gen-batch",
        type=int,
        default=None,
        help="Batch size for .sample()'s generation loop -- controls wall-clock "
        "only, not correctness. Applies to every mode's final DCR/utility "
        "sampling, and (sensitivity mode only) each periodic critic-round "
        ".generate() call during training too. Defaults to REaLTabFormer's own "
        "default (128) if unset; raise it if the GPU still has headroom.",
    )
    parser.add_argument("--cusum-check-every", type=int, default=20)
    parser.add_argument(
        "--cusum-delta",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Target effect size(s) CUSUM is tuned to detect, in standard-error "
        "units. Pass one or more values -- more than one runs an ensemble of "
        "trackers in parallel (alarm fires on the first to cross its own "
        "Bonferroni-corrected threshold), hedging against not knowing in advance "
        "whether the drift will be slow/gradual or sharp/sudden. Defaults to a "
        "small ensemble; pass a single value (e.g. --cusum-delta 0.5) for the "
        "original single-tracker behavior.",
    )
    parser.add_argument(
        "--sensitivity-n-critic",
        type=int,
        default=5,
        help="Epoch interval between sensitivity checks (each one runs a "
        "bootstrap round plus a real .generate() call) -- the existing "
        "method's own default.",
    )
    parser.add_argument("--sensitivity-n-critic-stop", type=int, default=2)
    parser.add_argument("--sensitivity-num-bootstrap", type=int, default=500)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument(
        "--run-id",
        default=None,
        help="Prefix for output files; defaults to mode+epochs+timestamp.",
    )
    args = parser.parse_args()

    if args.mode == "checkpoint" and not args.checkpoint_dir:
        parser.error("--mode checkpoint requires --checkpoint-dir")

    if args.run_id is None:
        args.run_id = f"{args.dataset}_{args.mode}_ep{args.epochs}_{int(time.time())}"

    loader, target_col, categorical, target_pos_val = DATASET_CONFIGS[args.dataset]
    print(f"Loading {args.dataset} data from {DATA_DIR} ...", flush=True)
    full_df = loader()
    print(f"full cleaned dataset: {len(full_df)} rows", flush=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if args.mode in ("cusum", "both", "all"):
        run_cusum(args, args.run_id, full_df, target_col, categorical, target_pos_val)
    if args.mode in ("full", "both", "all"):
        run_full(args, args.run_id, full_df, target_col, categorical, target_pos_val)
    if args.mode in ("sensitivity", "all"):
        run_sensitivity(
            args, args.run_id, full_df, target_col, categorical, target_pos_val
        )
    if args.mode == "checkpoint":
        run_from_checkpoint(
            args, args.run_id, full_df, target_col, categorical, target_pos_val
        )

    print("\nDone. Commit the new files under results/ to share them.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
