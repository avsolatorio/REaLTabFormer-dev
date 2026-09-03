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

Pass `--paper-metrics` to additionally compute MLE and DM matching the
REaLTabFormer paper's own published benchmark table (Table 1): MLE
(machine-learning efficacy -- macro-F1 for classification, R^2 for
regression) via CatBoost trained repeatedly on TSTR/TRTR and scored
mean+/-std over many seeds, the same methodology as the tab-ddpm
benchmark framework (github.com/rotot0/tab-ddpm,
scripts/eval_catboost.py) that avsolatorio/REaLTabFormer-Experiments
reuses -- confirmed by reading that framework's actual eval code, not
guessed from file naming. CatBoost is fit with tab-ddpm's own
PER-DATASET-TUNED hyperparameters (see CATBOOST_TUNED_CONFIGS, copied
verbatim from rotot0/tab-ddpm/tuned_models/catboost/*.json) whenever
one exists for `--dataset`, not CatBoost's defaults -- confirmed via
`measure_mle`'s own real-vs-tuned comparison to be the actual source of
an earlier gap between this script's real-data (TRTR) F1 on diabetes
(0.716, CatBoost defaults) and the paper's published "Original" column
for the same dataset (0.776). DM (discriminator measure -- a
classifier's accuracy at telling real from synthetic rows apart, 50% =
best/indistinguishable) reuses this library's own
`SyntheticDataBench.get_discriminator_performance`, with
RandomForestClassifier(oob_score=True) as the default model -- the
only model type that method's own oob_score_ check would report
anything for, strongly suggesting it's what the paper's own DM column
was computed with. Requires `pip install catboost` (not otherwise a
dependency of this script or the library); off by default since it's
meaningfully more expensive than the existing frac_suspicious/AUC-or-R2
check (default 50 synthetic-seed + 10 real-seed CatBoost fits, plus 10
RandomForest fits, per label).

Sensitivity mode's own expensive part -- a pre-training bootstrap
threshold computation that only ever resamples the training data
against itself (no model involved) -- is cached to disk by default
(see --sensitivity-cache-dir/--no-sensitivity-cache), as a growable
pool of rounds rather than an exact-num_bootstrap snapshot: asking for
more than what's cached only computes the shortfall and extends the
same file. Its parallelism defaults to every CPU core
(--sensitivity-bootstrap-n-jobs -1) rather than the library's own
conservative built-in cap. And --sensitivity-num-bootstrap itself
defaults to a size-aware heuristic instead of a flat 500 -- small
datasets measurably need more rounds for a stable threshold (~29%
run-to-run instability at 500 rounds on a 768-row dataset), large ones
are already stable at 500 and each round gets more expensive as the
dataset grows, so the default scales down accordingly (see
default_sensitivity_num_bootstrap).
"""

import argparse
import json
import shutil
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

# Per-dataset CatBoost hyperparameters, copied verbatim from
# rotot0/tab-ddpm's tuned_models/catboost/<dataset>_cv.json -- these are
# the ACTUAL hyperparameters the tab-ddpm benchmark framework (and, via
# it, the REaLTabFormer paper's own Table 1) tunes and uses for MLE, not
# guessed or re-tuned here. `cat_features` from those files is NOT
# reused -- it's a column-index list specific to tab-ddpm's own internal
# feature ordering, meaningless against our own dataframe's column
# order, so `measure_mle` computes its own indices dynamically instead
# (unchanged from before). Only present for datasets tab-ddpm actually
# tuned a config for; `measure_mle` falls back to CatBoost defaults (no
# eval_set/early stopping) for any dataset missing here.
CATBOOST_TUNED_CONFIGS = {
    "diabetes": dict(
        learning_rate=0.026561564197335047,
        depth=3,
        l2_leaf_reg=0.8066571920706246,
        bagging_temperature=0.6363246451815178,
        leaf_estimation_iterations=9,
        iterations=2000,
        early_stopping_rounds=50,
        od_pval=0.001,
        task_type="CPU",
        thread_count=4,
    ),
    "insurance": dict(
        learning_rate=0.08663279761354557,
        depth=6,
        l2_leaf_reg=8.92855270774259,
        bagging_temperature=0.9636627605010293,
        leaf_estimation_iterations=4,
        iterations=2000,
        early_stopping_rounds=50,
        od_pval=0.001,
        task_type="CPU",
        thread_count=4,
    ),
    "abalone": dict(
        learning_rate=0.028050502468157906,
        depth=5,
        l2_leaf_reg=7.780211394737271,
        bagging_temperature=0.026696235942186064,
        leaf_estimation_iterations=9,
        iterations=2000,
        early_stopping_rounds=50,
        od_pval=0.001,
        task_type="CPU",
        thread_count=4,
    ),
    "wilt": dict(
        learning_rate=0.13877980376409904,
        depth=6,
        l2_leaf_reg=1.1040918394803323,
        bagging_temperature=0.9966446926502672,
        leaf_estimation_iterations=4,
        iterations=2000,
        early_stopping_rounds=50,
        od_pval=0.001,
        task_type="CPU",
        thread_count=4,
    ),
    "churn2": dict(
        learning_rate=0.4667069360390258,
        depth=3,
        l2_leaf_reg=8.856733942123162,
        bagging_temperature=0.2955334069354449,
        leaf_estimation_iterations=1,
        iterations=2000,
        early_stopping_rounds=50,
        od_pval=0.001,
        task_type="CPU",
        thread_count=4,
    ),
    "adult": dict(
        learning_rate=0.16886992997713726,
        depth=3,
        l2_leaf_reg=0.19334681185025449,
        bagging_temperature=0.11959130879575816,
        leaf_estimation_iterations=8,
        iterations=2000,
        early_stopping_rounds=50,
        od_pval=0.001,
        task_type="CPU",
        thread_count=4,
    ),
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


def sync_results_to_repo(output_dir: Path, run_id: str) -> None:
    """Copies this run's small, git-tracked result files (the
    `{run_id}_summary.json` and `{run_id}_cusum_trajectory.jsonl`, if
    present) from `output_dir` back into the script's own `results/`
    directory -- lets `--output-dir` point somewhere else entirely
    (e.g. a larger mount, to keep the big gitignored checkpoint
    directories off a small/disk-constrained git working copy) without
    losing the "just commit results/" workflow: no separate manual
    rsync step needed afterward. A no-op when `output_dir` already IS
    the repo's own `results/` directory (the common case, nothing to
    sync).

    Deliberately filters to *files* with a `.json`/`.jsonl` suffix,
    not a bare `{run_id}*` glob -- the checkpoint directories
    (`{run_id}_ckpt_cusum/` etc.) share that exact prefix and must
    never get pulled in here; they're gitignored on purpose (large,
    not meant to be committed). Called once per mode (at the end of
    each `run_*` function, same place each mode already prints "Wrote
    ..."), not just once at the very end of `main()` -- matching this
    script's existing incremental-write philosophy so a run killed
    partway through `--mode all` still leaves whatever finished so far
    synced back, not just whatever was still running when it died.
    """
    output_dir = Path(output_dir).resolve()
    repo_results_dir = (SCRIPT_DIR / "results").resolve()
    if output_dir == repo_results_dir:
        return

    repo_results_dir.mkdir(parents=True, exist_ok=True)
    copied = [
        p.name
        for p in output_dir.glob(f"{run_id}*")
        if p.is_file() and p.suffix in (".json", ".jsonl")
    ]
    for name in copied:
        shutil.copy2(output_dir / name, repo_results_dir / name)

    if copied:
        print(
            f"Synced {len(copied)} result file(s) from {output_dir} to "
            f"{repo_results_dir}: {', '.join(copied)}",
            flush=True,
        )


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


def measure_mle(
    bench,
    label,
    summary_path,
    categorical: bool,
    n_seeds_synthetic: int = 50,
    n_seeds_real: int = 10,
    dataset_name: str = None,
    n_jobs: int = 1,
):
    """Machine-learning efficacy (MLE) matching the metric definition
    used in the REaLTabFormer paper's benchmark table and the
    tab-ddpm evaluation framework it's built on (rotot0/tab-ddpm,
    scripts/eval_catboost.py -- confirmed by reading that script, not
    guessed): CatBoost trained repeatedly, on different seeds, on TSTR
    (synthetic) and separately on TRTR (real), each scored against the
    SAME real held-out test set -- macro-F1 for classification, R^2
    for regression -- reporting mean+/-std over the repeats. This is a
    materially more expensive but also more statistically grounded
    version of `measure_utility`'s single-fit AUC/R^2 check: a single
    fit's own training-seed variance is exactly what a single-seed
    check can't tell apart from a real effect (this is what would have
    settled whether wilt's utility-gap jump after the calibration-pace
    fix was signal or noise, instead of just flagging it as an open
    question).

    Defaults (50 synthetic-seed fits, 10 real-seed fits) mirror
    tab-ddpm's own eval_seeds.py defaults, for direct comparability
    with the numbers already published for these same datasets.

    When `dataset_name` has a matching entry in CATBOOST_TUNED_CONFIGS,
    uses those exact per-dataset-tuned hyperparameters (not CatBoost's
    defaults) -- this is the actual gap that made an earlier run's real
    (TRTR) MLE on diabetes (F1=0.716) come out well below the paper's
    published "Original" column (F1=0.776) for what's very likely the
    same underlying dataset: CatBoost defaults vs. tab-ddpm's tuned
    config, not a difference in methodology. Matching that also means
    matching its early-stopping setup (iterations=2000 capped by
    early_stopping_rounds=50 against a held-out eval_set) -- since our
    SyntheticDataBench only keeps a train/test split (no third val
    split like tab-ddpm's datasets ship with), a validation slice is
    carved out of `bench.train_df` here (20% held out, stratified for
    classification, random_state=777 -- tab-ddpm's own
    lib.data.read_changed_val defaults, reused for consistency) and
    used as the eval_set for EVERY fit, TSTR and TRTR alike, exactly
    mirroring how tab-ddpm always evaluates against the real val split
    regardless of which data trained the model. The TRTR fit excludes
    those held-out rows from its own training data (no leakage into its
    early-stopping signal); the TSTR fit's synthetic training rows are
    already disjoint from real data, so nothing needs excluding there.
    Falls back to CatBoost defaults (no eval_set/early stopping) when
    `dataset_name` is unset or has no tuned config on file.

    `n_jobs`: the 50+10 per-label seed fits are independent (different
    random seeds, no shared state) and were originally run one at a
    time -- on a many-core box that leaves most of the CPU idle while
    GPU-bound model training elsewhere finishes, since a single
    CatBoost fit here only uses `thread_count` threads (4, from the
    tuned config). Set `n_jobs=-1` (joblib's own "all cores"
    convention -- already a dependency via scikit-learn, nothing new
    to install) to fit seeds in parallel via joblib's process-based
    backend instead; each worker's own CatBoost `thread_count` is then
    forced to 1 to avoid oversubscription (n_jobs processes x
    thread_count threads each would otherwise exceed the core count).
    Leave at the default `n_jobs=1` (plain sequential, tuned_config's
    own thread_count honored) when running standalone on a
    CPU-constrained box, or to keep wall-clock/scheduling predictable
    for a single run.
    """
    if bench.synth_train_df is None:
        print(f"[{label}] skipping MLE -- no synthetic data registered", flush=True)
        return None

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        print(
            f"[{label}] skipping MLE -- catboost not installed "
            "(pip install catboost)",
            flush=True,
        )
        return None

    from joblib import Parallel, delayed
    from sklearn.metrics import f1_score, r2_score
    from sklearn.model_selection import train_test_split

    # Parallelizing across seeds (independent fits) means each worker
    # process must not ALSO ask CatBoost for `thread_count` threads of
    # its own -- n_jobs processes x thread_count threads each would
    # oversubscribe the box. Sequential (n_jobs=1) keeps the tuned
    # config's own thread_count untouched.
    per_fit_thread_count = 1 if n_jobs != 1 else None

    tuned_config = CATBOOST_TUNED_CONFIGS.get(dataset_name)
    if tuned_config is not None:
        print(
            f"[{label}] MLE using tab-ddpm's tuned CatBoost config for "
            f"'{dataset_name}'",
            flush=True,
        )
    else:
        print(
            f"[{label}] MLE: no tuned CatBoost config for "
            f"'{dataset_name}' -- using CatBoost defaults (not directly "
            "comparable to the paper's published magnitudes)",
            flush=True,
        )

    feature_cols = [c for c in bench.train_df.columns if c != bench.target_col]
    cat_feature_idx = [
        i for i, c in enumerate(feature_cols) if bench.train_df[c].dtype == object
    ]

    real_fit_df, val_df = bench.train_df, None
    if tuned_config is not None:
        strat = bench.train_df[bench.target_col] if categorical else None
        real_fit_df, val_df = train_test_split(
            bench.train_df, test_size=0.2, random_state=777, stratify=strat
        )

    def _fit_score(train_df, seed):
        X_train = train_df[feature_cols]
        X_test = bench.test_df[feature_cols]
        fit_kwargs = {}
        model_config = dict(tuned_config or {})
        if per_fit_thread_count is not None:
            model_config["thread_count"] = per_fit_thread_count

        if categorical:
            y_train = (train_df[bench.target_col] == bench.target_pos_val).astype(int)
            y_test = (bench.test_df[bench.target_col] == bench.target_pos_val).astype(
                int
            )
            model = CatBoostClassifier(
                **model_config,
                loss_function="Logloss",
                eval_metric="TotalF1",
                random_seed=seed,
                verbose=False,
                cat_features=cat_feature_idx,
            )
            if val_df is not None:
                y_val = (val_df[bench.target_col] == bench.target_pos_val).astype(int)
                fit_kwargs["eval_set"] = (val_df[feature_cols], y_val)
            model.fit(X_train, y_train, **fit_kwargs)
            pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
            return f1_score(y_test, pred, average="macro")
        else:
            y_train = train_df[bench.target_col]
            y_test = bench.test_df[bench.target_col]
            model = CatBoostRegressor(
                **model_config,
                eval_metric="RMSE",
                random_seed=seed,
                verbose=False,
                cat_features=cat_feature_idx,
            )
            if val_df is not None:
                fit_kwargs["eval_set"] = (
                    val_df[feature_cols],
                    val_df[bench.target_col],
                )
            model.fit(X_train, y_train, **fit_kwargs)
            pred = model.predict(X_test)
            return r2_score(y_test, pred)

    synth_scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(bench.synth_train_df, s) for s in range(n_seeds_synthetic)
    )
    real_scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(real_fit_df, s) for s in range(n_seeds_real)
    )

    metric = "f1_macro" if categorical else "r2"
    result = dict(
        metric=metric,
        synthetic_mean=float(np.mean(synth_scores)),
        synthetic_std=float(np.std(synth_scores)),
        real_mean=float(np.mean(real_scores)),
        real_std=float(np.std(real_scores)),
        n_seeds_synthetic=n_seeds_synthetic,
        n_seeds_real=n_seeds_real,
        used_tuned_config=tuned_config is not None,
    )
    print(
        f"[{label}] MLE ({metric}): synthetic={result['synthetic_mean']:.4f}"
        f"±{result['synthetic_std']:.4f}  real={result['real_mean']:.4f}"
        f"±{result['real_std']:.4f}",
        flush=True,
    )
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing[f"{label}_mle"] = result
    summary_path.write_text(json.dumps(existing, indent=2))
    return result


def measure_dm(bench, label, summary_path, n_seeds: int = 10, n_jobs: int = 1):
    """Discriminator measure (DM): a classifier's accuracy (as a
    percentage) at telling a real training row from a synthetic one on
    a held-out split -- 50% means real and synthetic are
    indistinguishable (best case), 100% means trivially separable.
    Reuses this library's own
    `SyntheticDataBench.get_discriminator_performance` rather than
    reimplementing it -- `compute_discriminator_predictions` already
    does exactly this, including an `oob_score_` check that only a
    bagging ensemble like RandomForestClassifier(oob_score=True)
    satisfies, which is a strong signal that's the model type the
    paper's own DM column was computed with (confirmed by reading that
    method's code, not assumed). Test classes are balanced by
    construction (equal real/synthetic counts in the held-out split),
    so plain accuracy is the right metric here, unlike an imbalanced
    classification problem where it wouldn't be. Repeated over
    `n_seeds` RandomForest fits (same train/test row split from the
    bench each time, only the forest's own random_state varies) for a
    mean+/-std instead of a single noisy draw.

    `n_jobs`: as in `measure_mle`, fits seeds in parallel via joblib
    when set to -1 (or any value != 1) -- each RandomForest's own
    `n_jobs` is then forced to 1 (its default -1 would otherwise
    compete with the outer per-seed parallelism for the same cores;
    many small independent forests beat one forest with excess inner
    parallelism here anyway, given how small n_estimators=200 on these
    dataset sizes actually is per fit).
    """
    if bench.synth_train_df is None or bench.synth_test_df is None:
        print(f"[{label}] skipping DM -- no synthetic data registered", flush=True)
        return None

    from joblib import Parallel, delayed
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rf_n_jobs = 1 if n_jobs != 1 else -1

    def _fit_one(seed):
        model = RandomForestClassifier(
            n_estimators=200, oob_score=True, random_state=seed, n_jobs=rf_n_jobs
        )
        preds = bench.get_discriminator_performance(model)
        return 100.0 * accuracy_score(preds["y_test"], preds["y_preds"])

    scores = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(seed) for seed in range(n_seeds))

    result = dict(
        dm_mean=float(np.mean(scores)), dm_std=float(np.std(scores)), n_seeds=n_seeds
    )
    print(
        f"[{label}] DM: {result['dm_mean']:.2f}±{result['dm_std']:.2f}% "
        "(50% = indistinguishable from real, best case)",
        flush=True,
    )
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing[f"{label}_dm"] = result
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
    if args.paper_metrics:
        measure_mle(
            bench,
            "cusum_stop_at_alarm",
            summary_path,
            categorical=categorical,
            n_seeds_synthetic=args.mle_n_seeds_synthetic,
            n_seeds_real=args.mle_n_seeds_real,
            dataset_name=args.dataset,
            n_jobs=args.eval_n_jobs,
        )
        measure_dm(
            bench,
            "cusum_stop_at_alarm",
            summary_path,
            n_seeds=args.dm_n_seeds,
            n_jobs=args.eval_n_jobs,
        )
    print(f"\nWrote {trajectory_path} and {summary_path}", flush=True)
    if not args.no_sync_results:
        sync_results_to_repo(Path(args.output_dir), run_id)


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
    if args.paper_metrics:
        measure_mle(
            bench,
            "full_schedule",
            summary_path,
            categorical=categorical,
            n_seeds_synthetic=args.mle_n_seeds_synthetic,
            n_seeds_real=args.mle_n_seeds_real,
            dataset_name=args.dataset,
            n_jobs=args.eval_n_jobs,
        )
        measure_dm(
            bench,
            "full_schedule",
            summary_path,
            n_seeds=args.dm_n_seeds,
            n_jobs=args.eval_n_jobs,
        )
    print(f"\nWrote {summary_path}", flush=True)
    if not args.no_sync_results:
        sync_results_to_repo(Path(args.output_dir), run_id)


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
        sensitivity_cache_dir=(
            None if args.no_sensitivity_cache else args.sensitivity_cache_dir
        ),
        sensitivity_bootstrap_n_jobs=args.sensitivity_bootstrap_n_jobs,
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
    if args.paper_metrics:
        measure_mle(
            bench,
            "sensitivity_stop",
            summary_path,
            categorical=categorical,
            n_seeds_synthetic=args.mle_n_seeds_synthetic,
            n_seeds_real=args.mle_n_seeds_real,
            dataset_name=args.dataset,
            n_jobs=args.eval_n_jobs,
        )
        measure_dm(
            bench,
            "sensitivity_stop",
            summary_path,
            n_seeds=args.dm_n_seeds,
            n_jobs=args.eval_n_jobs,
        )
    print(f"\nWrote {summary_path}", flush=True)
    if not args.no_sync_results:
        sync_results_to_repo(Path(args.output_dir), run_id)


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
    if args.paper_metrics:
        measure_mle(
            bench,
            "checkpoint_recheck",
            summary_path,
            categorical=categorical,
            n_seeds_synthetic=args.mle_n_seeds_synthetic,
            n_seeds_real=args.mle_n_seeds_real,
            dataset_name=args.dataset,
            n_jobs=args.eval_n_jobs,
        )
        measure_dm(
            bench,
            "checkpoint_recheck",
            summary_path,
            n_seeds=args.dm_n_seeds,
            n_jobs=args.eval_n_jobs,
        )
    print(f"\nWrote {summary_path}", flush=True)
    if not args.no_sync_results:
        sync_results_to_repo(Path(args.output_dir), run_id)


def default_sensitivity_num_bootstrap(n_rows: int) -> int:
    """Size-aware default for --sensitivity-num-bootstrap, replacing a
    single fixed 500 for every dataset. Empirically calibrated (not
    guessed): at the library's real defaults (frac=0.165/2, qt_max=
    0.05, qt_interval=100), an unseeded sensitivity threshold varied
    ~29% run-to-run on a 768-row dataset at num_bootstrap=500 --
    bumping to ~2000 rounds there cut that to ~7%. But at 1500 and
    5000 rows, the SAME num_bootstrap=500 was *already* down to ~10%
    spread -- evidently a fixed `frac` means a bigger dataset gives
    bigger, less noisy per-round subsets for free, so bigger datasets
    need FEWER extra rounds for the same stability, not more (the
    opposite of what you'd want if you just scaled --sensitivity-
    num-bootstrap up for every dataset uniformly, since each round
    also gets more expensive as the dataset grows).

    `1_500_000 // n_rows`, clipped to [500, 4000], reproduces ~2000 at
    n=768 (the verified-good value) and decays to the 500 floor by
    around n=3000 (matching 1500/5000 rows both already being fine at
    500). This is a starting point from three calibration points, not
    a precisely-derived formula -- the growable cache (see
    SyntheticDataBench.compute_sensitivity_threshold's cache_dir) means
    it's cheap to ask for more later if a specific dataset still looks
    unstable at this default.
    """
    return int(np.clip(round(1_500_000 / max(n_rows, 1)), 500, 4000))


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
    parser.add_argument(
        "--sensitivity-num-bootstrap",
        type=int,
        default=None,
        help="Bootstrap rounds for sensitivity's pre-training threshold "
        "computation. Defaults to a size-aware heuristic (see "
        "default_sensitivity_num_bootstrap) instead of a fixed 500 -- "
        "small datasets get meaningfully more rounds (measured ~29% "
        "run-to-run threshold instability at 500 rounds on a 768-row "
        "dataset), large ones stay at the 500 floor (already stable "
        "there, and each round gets more expensive as the dataset "
        "grows). Pass an explicit value to override.",
    )
    parser.add_argument(
        "--sensitivity-cache-dir",
        default=str(SCRIPT_DIR / "results" / ".sensitivity_cache"),
        help="Caches sensitivity's pre-training num_bootstrap-round threshold "
        "computation here, keyed on the training data's own content plus "
        "every parameter that affects it -- this step never touches the "
        "trained model (it only resamples the training data against "
        "itself), so a repeat run against the same dataset/settings can "
        "skip it entirely. On by default; pass --no-sensitivity-cache to "
        "disable.",
    )
    parser.add_argument(
        "--no-sensitivity-cache",
        action="store_true",
        default=False,
        help="Disable --sensitivity-cache-dir (always recompute the "
        "bootstrap threshold from scratch, matching the library's original "
        "behavior).",
    )
    parser.add_argument(
        "--sensitivity-bootstrap-n-jobs",
        type=int,
        default=-1,
        help="Worker count (joblib's own convention -- -1 means all cores) "
        "for sensitivity's pre-training bootstrap computation. Defaults to "
        "every core here, overriding the library's own conservative "
        "built-in default (min(max(2, cpu_count // 4), 16)) -- pass 1 for "
        "fully sequential, or a smaller positive number to cap how many "
        "cores this step claims.",
    )
    parser.add_argument(
        "--paper-metrics",
        action="store_true",
        default=False,
        help="Additionally compute MLE (CatBoost, macro-F1/R^2, TSTR+TRTR, "
        "mean+/-std over many seeds) and DM (RandomForest discriminator "
        "accuracy) matching the REaLTabFormer paper's own benchmark table "
        "methodology -- see the module docstring. Requires `pip install "
        "catboost`; off by default since it's meaningfully more expensive "
        "than the existing frac_suspicious/AUC-or-R2 check.",
    )
    parser.add_argument(
        "--mle-n-seeds-synthetic",
        type=int,
        default=50,
        help="CatBoost fits on the synthetic (TSTR) data, different seeds "
        "each -- only used with --paper-metrics. Matches tab-ddpm's own "
        "eval_seeds.py default.",
    )
    parser.add_argument(
        "--mle-n-seeds-real",
        type=int,
        default=10,
        help="CatBoost fits on the real (TRTR) data -- only used with "
        "--paper-metrics. Matches tab-ddpm's own eval_seeds.py default.",
    )
    parser.add_argument(
        "--dm-n-seeds",
        type=int,
        default=10,
        help="RandomForest discriminator fits, different seeds each -- only "
        "used with --paper-metrics.",
    )
    parser.add_argument(
        "--eval-n-jobs",
        type=int,
        default=-1,
        help="Only used with --paper-metrics: how many of the per-seed "
        "CatBoost/RandomForest fits to run in parallel (joblib's own "
        "convention -- -1 means all cores, 1 means the old fully-sequential "
        "behavior). These fits are CPU-only and independent regardless of "
        "--device, so this is what actually uses a machine's extra CPU "
        "cores during the MLE/DM step -- GPU is already used for the "
        "REaLTabFormer training itself via --device and isn't affected by "
        "this flag. Defaults to using every core; pass 1 to go back to "
        "sequential (e.g. for predictable wall-clock on a shared box), or a "
        "smaller positive number to cap how many cores this step claims.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results"),
        help="Where results (and, for --mode cusum/all, the gigabytes-heavy "
        "checkpoint directories) are written. Point this at a larger mount "
        "if the repo's own disk is tight -- the small, git-tracked summary/ "
        "trajectory files still get copied back into cusum_validation/"
        "results/ automatically afterward (see --no-sync-results), so the "
        "usual 'commit results/' workflow still works without a separate "
        "manual rsync step.",
    )
    parser.add_argument(
        "--no-sync-results",
        action="store_true",
        default=False,
        help="Skip copying this run's summary/trajectory files back into "
        "cusum_validation/results/ when --output-dir points elsewhere. On "
        "by default; pass this if you'd rather sync manually (e.g. you only "
        "want to commit a subset of what a big batch produced).",
    )
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

    if args.sensitivity_num_bootstrap is None:
        args.sensitivity_num_bootstrap = default_sensitivity_num_bootstrap(len(full_df))
        print(
            f"--sensitivity-num-bootstrap not set; using size-aware default "
            f"{args.sensitivity_num_bootstrap} for this {len(full_df)}-row "
            "dataset",
            flush=True,
        )

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
