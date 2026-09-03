"""
Runs the full CUSUM validation battery across multiple datasets in one
call -- a convenience wrapper around run_experiment.py's --mode (default
"all": cusum + full + sensitivity) for each dataset in turn, meant for a
GPU server where running each `--dataset` invocation by hand would be
tedious. Each dataset runs as its own subprocess (a fresh Python
process per dataset -- no state, random seeds, or monkeypatches leaking
across runs) and a failure in one dataset is logged and skipped rather
than aborting the whole batch, so a crash on run 3 of 5 doesn't cost you
the results already collected from runs 1 and 2.

Usage:
    python run_all_experiments.py --epochs 300 --device cuda
    python run_all_experiments.py --datasets abalone churn2 --epochs 300 --device cuda
    python run_all_experiments.py --mode cusum --epochs 300 --device cuda

Defaults to the five non-Adult datasets (small: diabetes/insurance/
abalone/wilt; medium: churn2), smallest first so you get fast feedback
before the longer runs. `adult` is excluded by default -- it already
has extensive committed results, is far more expensive than the others,
and typically wants a different --epochs ceiling; pass it explicitly
via --datasets if you want it in the batch.

Each dataset's full stdout/stderr streams live to the console AND is
saved to its own file under --log-dir, so a long unattended run still
leaves a complete record even if you're not watching it finish.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_EXPERIMENT = SCRIPT_DIR / "run_experiment.py"

# Smallest first -- fast feedback before committing to the longer runs.
# "adult" deliberately excluded by default; see module docstring.
DEFAULT_DATASETS = ["diabetes", "insurance", "abalone", "wilt", "churn2"]


def run_one(dataset, args, log_dir: Path) -> dict:
    run_id = f"{dataset}_{args.mode}_ep{args.epochs}_{int(time.time())}"
    log_path = log_dir / f"{run_id}.log"

    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "--dataset",
        dataset,
        "--mode",
        args.mode,
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        args.output_dir,
        "--run-id",
        run_id,
    ]
    if args.gen_batch:
        cmd += ["--gen-batch", str(args.gen_batch)]
    if args.paper_metrics:
        cmd += ["--paper-metrics"]

    print(
        f"\n{'=' * 70}\nSTARTING {dataset} (run_id={run_id})\n"
        f"Log: {log_path}\n{'=' * 70}",
        flush=True,
    )

    t0 = time.time()
    with open(log_path, "w") as logf:
        # Stream live to the console AND save to the log file, rather
        # than picking one -- a long unattended run still leaves a full
        # record, and a watched one still shows progress in real time.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        proc.wait()
    elapsed = time.time() - t0

    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"\n{dataset}: {status} in {elapsed:.1f}s\n", flush=True)
    return dict(
        run_id=run_id, ok=ok, status=status, elapsed_s=elapsed, log=str(log_path)
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to run, in order. Default: {DEFAULT_DATASETS}.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["cusum", "full", "sensitivity", "both", "all"],
        help="Passed through to run_experiment.py for every dataset. Default 'all' "
        "(cusum + full + sensitivity) -- the full three-way comparison.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gen-batch",
        type=int,
        default=None,
        help="Passed through to run_experiment.py -- see its own --help.",
    )
    parser.add_argument(
        "--paper-metrics",
        action="store_true",
        default=False,
        help="Passed through to run_experiment.py -- adds the paper-matching "
        "MLE (CatBoost)/DM (RandomForest discriminator) metrics to every "
        "dataset in the batch. Requires `pip install catboost`; meaningfully "
        "slower per run -- see run_experiment.py's own --help.",
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument(
        "--log-dir",
        default=str(SCRIPT_DIR / "results" / "logs"),
        help="Where each dataset's full stdout/stderr is saved.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    batch_t0 = time.time()
    for dataset in args.datasets:
        results[dataset] = run_one(dataset, args, log_dir)
    batch_elapsed = time.time() - batch_t0

    print(f"\n\n{'=' * 70}\nBATCH SUMMARY ({batch_elapsed:.1f}s total)\n{'=' * 70}")
    for dataset, info in results.items():
        print(
            f"  {dataset:12s} {info['status']:20s} {info['elapsed_s']:8.1f}s  {info['log']}"
        )

    failed = [d for d, info in results.items() if not info["ok"]]
    if failed:
        print(
            f"\n{len(failed)} of {len(results)} dataset(s) failed: {failed}. "
            "Check their logs above for the actual error.",
            flush=True,
        )
        sys.exit(1)

    print(
        f"\nAll {len(results)} datasets done. Commit results/ (the new "
        "summary.json/trajectory.jsonl files, not the gitignored "
        "checkpoint dirs or --log-dir) to share them.",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
